"""Kaggle image classification imports with a fresh, user-controlled split."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
from urllib.parse import urlparse

import torch

from .data import _image_sizes, _is_image, _load_images_parallel, _save
from .errors import BackendError

SPLITS = {"train", "training", "test", "testing", "val", "valid", "validation", "train_set", "test_set", "val_set", "valid_set", "validation_set", "train_data", "test_data", "validation_data"}


def dataset_handle(value: str) -> str:
    value = value.strip().rstrip("/")
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.hostname not in {"kaggle.com", "www.kaggle.com"} or not url.path.startswith("/datasets/"):
            raise BackendError("KAGGLE_HANDLE_INVALID")
        value = url.path.removeprefix("/datasets/").strip("/")
    if not re.fullmatch(r"[\w-]+/[\w-][\w.-]*(?:/versions/[1-9][0-9]*)?", value, flags=re.ASCII):
        raise BackendError("KAGGLE_HANDLE_INVALID")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative.replace("\\", "/")).resolve()
    if path != root and root not in path.parents:
        raise BackendError("KAGGLE_SUBDIRECTORY_INVALID")
    return path


def download_root(handle: str, subdirectory: str = "") -> Path:
    handle = dataset_handle(handle)
    try:
        import kagglehub
    except ImportError as exc:
        raise BackendError("KAGGLE_LIBRARY_MISSING") from exc
    try:
        root = Path(kagglehub.dataset_download(handle)).resolve()
    except Exception:
        # Keep credentials, URLs with tokens and SDK internals out of project/log UI.
        raise BackendError("KAGGLE_DOWNLOAD_FAILED") from None
    selected = _inside(root, subdirectory)
    if not selected.is_dir():
        raise BackendError("KAGGLE_SUBDIRECTORY_INVALID")
    return selected


def image_index(root: Path, label_file: str = "") -> tuple[list[Path], list[str], dict]:
    root = root.resolve()
    paths = sorted(p for p in root.rglob("*") if _is_image(p) and not any(part.startswith(".") or part == "__MACOSX" for part in p.relative_to(root).parts))
    if not paths:
        raise BackendError("KAGGLE_NO_IMAGES")
    if any(root not in p.resolve().parents for p in paths):
        raise BackendError("KAGGLE_SUBDIRECTORY_INVALID")
    # Image targets and detection annotations must not be mistaken for class folders.
    components = {part.lower() for p in paths for part in p.relative_to(root).parts[:-1]}
    annotation_files = any(root.rglob("*.txt")) or any(root.rglob("annotations.json"))
    if {"images", "masks"} <= components or {"inputs", "targets"} <= components or ("labels" in {p.name.lower() for p in root.rglob('*') if p.is_dir()} and annotation_files):
        raise BackendError("KAGGLE_IMAGE_TASK_MISMATCH")
    by_name: dict[str, list[Path]] = defaultdict(list)
    path_set = set(paths)
    for p in paths:
        for key in {p.name, p.stem}:
            by_name[key].append(p)
    labels: dict[Path, str] = {}
    tables = [_inside(root, label_file)] if label_file else sorted(p for p in root.rglob("*.csv") if "submission" not in p.name.lower())
    used_tables = []
    file_keys = {"filename", "file_name", "image", "image_id", "image_path", "filepath", "path", "id"}
    label_keys = {"label", "class", "class_name", "category", "target", "class_id"}
    for table in tables:
        if not table.is_file():
            raise BackendError("KAGGLE_LABEL_FILE_INVALID")
        with table.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = {column.strip().lower(): column for column in (reader.fieldnames or [])}
            fk = [columns[k] for k in sorted(file_keys) if k in columns]
            lk = [columns[k] for k in sorted(label_keys) if k in columns]
            if not fk or not lk:
                if label_file:
                    raise BackendError("KAGGLE_LABEL_FILE_INVALID")
                continue
            if len(fk) != 1 or len(lk) != 1:
                raise BackendError("KAGGLE_LABEL_FILE_INVALID")
            used_tables.append(table.relative_to(root).as_posix())
            for row in reader:
                filename, label = str(row.get(fk[0]) or "").strip(), str(row.get(lk[0]) or "").strip()
                if not filename or not label:
                    raise BackendError("KAGGLE_LABEL_FILE_INVALID")
                candidates = {_inside(root, filename), _inside(root, (table.parent.relative_to(root) / filename).as_posix())} & path_set
                if not candidates:
                    candidates = set(by_name.get(filename.replace("\\", "/").split("/")[-1], []))
                    scoped = {p for p in candidates if table.parent in p.parents}
                    if scoped:
                        candidates = scoped
                if len(candidates) != 1:
                    raise BackendError("KAGGLE_LABEL_IMAGE_AMBIGUOUS", filename)
                path = candidates.pop()
                if path in labels and labels[path] != label:
                    raise BackendError("KAGGLE_LABEL_CONFLICT", filename)
                labels[path] = label
    if used_tables:
        layout = "csv-labels"
    else:
        directories = [[part for part in p.relative_to(root).parts[:-1] if part.lower() not in SPLITS] for p in paths]
        # Remove shared wrapper folders, retaining the first actual class directory.
        while all(parts for parts in directories) and len({parts[0] for parts in directories}) == 1:
            directories = [parts[1:] for parts in directories]
        labels = {p: parts[0] for p, parts in zip(paths, directories) if parts}
        layout = "split-folders" if any(part.lower() in SPLITS for p in paths for part in p.relative_to(root).parts[:-1]) else "class-folders"
    missing = len(paths) - len(labels)
    if missing:
        raise BackendError("KAGGLE_UNLABELED_IMAGES", str(missing))
    names = [labels[p] for p in paths]
    if len(set(names)) < 2:
        raise BackendError("IMPORT_IMAGE_CLASSIFICATION_MIN_CLASSES")
    report = {"layout": layout, "samples": len(paths), "classes": dict(sorted(Counter(names).items())), "originalSplits": sorted({part.lower() for p in paths for part in p.relative_to(root).parts[:-1] if part.lower() in SPLITS}), "labelFiles": used_tables, "splitPolicy": "merge-and-resplit"}
    return paths, names, report


def inspect(handle: str, options: dict | None = None) -> dict:
    opts = options or {}
    root = download_root(handle, str(opts.get("subdirectory", "")))
    _, _, report = image_index(root, str(opts.get("label_file", "")))
    return {**report, "handle": dataset_handle(handle)}


def download(project: dict, task_id: str, handle: str, options: dict | None = None) -> dict:
    if task_id != "image.classification":
        raise BackendError("KAGGLE_IMAGE_TASK_MISMATCH")
    opts = options or {}
    try:
        resolution = int(opts.get("resolution", 64)); channels = int(opts.get("channels", 3))
        valid = float(opts.get("resolution", 64)) == resolution and float(opts.get("channels", 3)) == channels
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid or not 8 <= resolution <= 2048 or channels not in {1, 3}:
        raise BackendError("KAGGLE_IMAGE_OPTIONS_INVALID")
    root = download_root(handle, str(opts.get("subdirectory", "")))
    paths, labels, report = image_index(root, str(opts.get("label_file", "")))
    classes = sorted(set(labels)); lookup = {name: i for i, name in enumerate(classes)}
    x = _load_images_parallel(paths, resolution, channels)
    metadata = {"kaggleHandle": dataset_handle(handle), "importReport": report}
    # _save generates provisional indices; the Partitions panel owns the final split.
    return _save(project, task_id, x, torch.tensor([lookup[label] for label in labels]), f"Kaggle: {dataset_handle(handle)}", 42, classes, {"channels": channels, "resolution": resolution}, source="kaggle", source_path=f"https://www.kaggle.com/datasets/{dataset_handle(handle)}", metadata=metadata, original_sizes=_image_sizes(paths))
