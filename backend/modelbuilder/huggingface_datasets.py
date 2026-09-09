from __future__ import annotations

import csv
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .data import _apply_normalization, _save
from .errors import BackendError
from .storage import atomic_json, project_dir, workspace_root


# Revisions are pinned deliberately: public dataset schemas may change without
# notice, while a ModelBuilder project must remain reproducible.
DATASETS: dict[str, dict] = {
    "iris": {
        "name": "Iris",
        "repoId": "scikit-learn/iris",
        "revision": "0bda0ce801be0fa2f464ff845a9d5ceae99aad7d",
        "tasks": ["tabular.classification"],
        "sizeBytes": 5107,
        "license": "CC0-1.0",
        "description": "Clasifica tres especies de iris a partir de cuatro medidas florales.",
        "defaultOptions": {"normalization": "none", "batchSize": 16},
        "loader": "iris",
    },
    "smart-home-energy": {
        "name": "Smart Home Energy",
        "repoId": "hoangbang/smart-home-energy-prediction",
        "revision": "333972783790c56e81da66c95f8ace77af08034e",
        "tasks": ["tabular.regression", "tabular.reconstruction", "sequence.regression", "sequence.forecast"],
        "sizeBytes": 1132548,
        "license": "CC BY 4.0 (fuente UCI)",
        "description": "Temperatura, humedad y consumo de una vivienda; útil para regresión, anomalías y pronóstico.",
        "defaultOptions": {"normalization": "standard", "batchSize": 32, "window": 48, "horizon": 8, "stride": 6},
        "loader": "energy",
        "temporalSplits": True,
        "dataVersion": 3,
    },
    "ecg-arrhythmia": {
        "name": "MIT-BIH ECG Arrhythmia",
        "repoId": "dheerajthuvara/ecg-arrhythmia-data",
        "revision": "408fe0ac0c721e214c2ee7198120eedad8197a7d",
        "tasks": ["sequence.classification"],
        "sizeBytes": 82000000,
        "license": "MIT",
        "description": "Latidos ECG normalizados de 180 muestras en cinco clases de ritmo.",
        "defaultOptions": {"normalization": "none", "batchSize": 64, "max_samples": 10000},
        "loader": "ecg",
        "dataVersion": 2,
    },
    "mnist": {
        "name": "MNIST",
        "repoId": "ylecun/mnist",
        "revision": "77f3279092a1c1579b2250db8eafed0ad422088c",
        "tasks": ["image.classification", "image.reconstruction"],
        "sizeBytes": 18157506,
        "license": "MIT",
        "description": "70 000 dígitos manuscritos; base compacta para CNN, ViT y autoencoders.",
        "defaultOptions": {"channels": 1, "resolution": 28, "batchSize": 64, "max_samples": 12000},
        "loader": "mnist",
        "officialSplits": True,
    },
    "utkface": {
        "name": "UTKFace Cropped",
        "repoId": "py97/UTKFace-Cropped",
        "revision": "e80f21fb21fa3631380c3cb634b79be3ac2f4cba",
        "tasks": ["image.regression"],
        "sizeBytes": 106634631,
        "license": "MIT en el mirror; verificar términos de UTKFace",
        "description": "Rostros alineados con edad numérica para aprender regresión visual.",
        "defaultOptions": {"channels": 3, "resolution": 64, "batchSize": 32, "max_samples": 5000},
        "loader": "utkface",
        "dataVersion": 2,
    },
    "emotion": {
        "name": "DAIR.AI Emotion",
        "repoId": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "tasks": ["text.classification"],
        "sizeBytes": 1287193,
        "license": "Uso educativo y de investigación",
        "description": "20 000 mensajes en seis emociones: tristeza, alegría, amor, ira, miedo y sorpresa.",
        "defaultOptions": {"max_length": 64, "vocab_size": 4096, "batchSize": 32},
        "loader": "emotion",
        "officialSplits": True,
        "dataVersion": 2,
    },
    "wikitext-2": {
        "name": "WikiText-2 Raw",
        "repoId": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "tasks": ["text.language_model"],
        "sizeBytes": 7747362,
        "license": "CC BY-SA / GFDL",
        "description": "Artículos destacados de Wikipedia para predicción del siguiente token.",
        "defaultOptions": {"max_length": 64, "vocab_size": 4096, "batchSize": 64, "max_samples": 20000},
        "loader": "wikitext",
        "officialSplits": True,
    },
    "monuseg": {
        "name": "MoNuSeg",
        "repoId": "MedOtter/MoNuSeg",
        "revision": "630f3612e562e95732fefc9fc0e896fb4030bb7f",
        "tasks": ["image.segmentation.binary"],
        "sizeBytes": 92171048,
        "license": "CC BY-NC-SA 4.0",
        "description": "51 imágenes histopatológicas con máscaras binarias de núcleos celulares.",
        "defaultOptions": {"channels": 3, "resolution": 128, "batchSize": 8},
        "loader": "monuseg",
        "officialSplits": True,
    },
    "monusac": {
        "name": "MoNuSAC 2020",
        "repoId": "MedOtter/MoNuSAC2020",
        "revision": "097b60ee25ed0c0dd891399dc7cf7fae8748285f",
        "tasks": ["image.segmentation.multiclass"],
        "sizeBytes": 277460893,
        "license": "CC BY-NC-SA 4.0",
        "description": "Histopatología con segmentación de cuatro tipos celulares y fondo.",
        "defaultOptions": {"channels": 3, "resolution": 128, "batchSize": 8},
        "loader": "monusac",
        "officialSplits": True,
    },
}


def _cache_root() -> Path:
    root = workspace_root() / "cache" / "huggingface"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _marker(spec_id: str) -> Path:
    spec = DATASETS[spec_id]
    return _cache_root() / "markers" / spec_id / spec["revision"] / "complete.json"


def _options(spec: dict, options: dict | None) -> dict:
    result = {**spec.get("defaultOptions", {}), **(options or {})}
    return {key: value for key, value in result.items() if value is not None}


def _signature(options: dict) -> str:
    return json.dumps(options, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _existing_project_dataset(project: dict, task_id: str, spec_id: str, options: dict | None = None) -> dict | None:
    manifests = sorted((project_dir(project) / "datasets").glob("*/*/manifest.json"), reverse=True)
    for manifest_path in manifests:
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if value.get("task_id") != task_id or value.get("catalogId") != spec_id:
                continue
            if int(value.get("dataVersion",1))!=int(DATASETS.get(spec_id,{}).get("dataVersion",1)):
                continue
            expected_split_source="official" if DATASETS.get(spec_id,{}).get("officialSplits") else "temporal" if DATASETS.get(spec_id,{}).get("temporalSplits") and task_id.startswith("sequence") else None
            if expected_split_source and value.get("splitSource")!=expected_split_source:
                continue
            if options is not None and value.get("optionsSignature") != _signature(options):
                continue
            if Path(str(value.get("path", ""))).is_file():
                value["cacheStatus"] = "project"
                return {key: value[key] for key in value if key not in {"split_indices", "task_id", "schema_version"}}
        except (OSError, ValueError, TypeError):
            continue
    return None


def catalog(project: dict | None, task_id: str) -> list[dict]:
    result = []
    for spec_id, spec in DATASETS.items():
        if task_id not in spec["tasks"]:
            continue
        installed = bool(project and _existing_project_dataset(project, task_id, spec_id))
        result.append({
            "id": spec_id,
            **{key: spec[key] for key in ("name", "repoId", "revision", "tasks", "sizeBytes", "license", "description", "defaultOptions")},
            "cached": _marker(spec_id).is_file(),
            "installed": installed,
        })
    return result


def _hf_file(spec: dict, filename: str, offline: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            repo_id=spec["repoId"], filename=filename, repo_type="dataset",
            revision=spec["revision"], cache_dir=str(_cache_root() / "hub"),
            local_files_only=offline,
        ))
    except Exception as exc:
        raise BackendError("HF_DOWNLOAD_FAILED", f"{spec['repoId']}: {exc}") from exc


def _hf_dataset(spec: dict, *, name: str | None = None, split: str, offline: bool = False):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise BackendError("HF_DEPENDENCIES_MISSING") from exc
    try:
        config = DownloadConfig(cache_dir=str(_cache_root() / "hub"), local_files_only=offline)
        return load_dataset(
            spec["repoId"], name=name, split=split, revision=spec["revision"],
            cache_dir=str(_cache_root() / "datasets"), download_config=config,
        )
    except Exception as exc:
        raise BackendError("HF_DOWNLOAD_FAILED", f"{spec['repoId']}: {exc}") from exc


def _pil_tensor(image: Image.Image, resolution: int, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    image = image.convert(mode).resize((resolution, resolution), Image.Resampling.BILINEAR)
    array = np.asarray(image).copy()
    if channels == 1:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.uint8)


def _mask_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    array = np.asarray(image.convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)).copy()
    return torch.from_numpy(array).long()


def _load_iris(spec: dict, task_id: str, opts: dict, offline: bool):
    path = _hf_file(spec, "Iris.csv", offline)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    names = sorted({row["Species"] for row in rows}); labels = {name: index for index, name in enumerate(names)}
    x = torch.tensor([[float(row[key]) for key in features] for row in rows], dtype=torch.float32)
    x = _apply_normalization(x, str(opts.get("normalization", "none")))
    y = torch.tensor([labels[row["Species"]] for row in rows], dtype=torch.long)
    return x, y, names


def _load_energy(spec: dict, task_id: str, opts: dict, offline: bool):
    dataset = _hf_dataset(spec, split="train", offline=offline)
    numeric = [name for name in dataset.column_names if name not in {"date", "Appliances"}]
    values = torch.tensor([[float(row[name]) for name in numeric] for row in dataset], dtype=torch.float32)
    target = torch.tensor([[float(value)] for value in dataset["Appliances"]], dtype=torch.float32)
    if task_id == "tabular.regression":
        x, y = values, target
    elif task_id == "tabular.reconstruction":
        x, y = values, values.clone()
    else:
        window = max(8, int(opts.get("window", 48))); horizon = max(1, int(opts.get("horizon", 8))); stride = max(1, int(opts.get("stride", 6)))
        # Build windows independently inside each chronological partition.  If
        # windows were created first and split afterwards, adjacent partitions
        # would share most of their timestamps and leak validation/test data.
        boundaries = (0, round(len(values) * .7), round(len(values) * .85), len(values))
        input_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        lengths: list[int] = []
        for lower, upper in zip(boundaries, boundaries[1:]):
            starts = range(lower, upper - window - horizon + 1, stride)
            part_x = torch.stack([values[start:start + window] for start in starts])
            future = torch.stack([target[start + window:start + window + horizon] for start in starts])
            part_y = future.mean(1) if task_id == "sequence.regression" else future
            input_parts.append(part_x); target_parts.append(part_y); lengths.append(len(part_x))
        x = torch.cat(input_parts); y = torch.cat(target_parts)
        splits = _official_ranges(lengths)
    if task_id.startswith("sequence"):
        split_source="temporal"
    else:
        splits=_random_splits(len(x),42)
        split_source="generated"
    normalization = str(opts.get("normalization", "none"))
    if normalization != "none":
        train=x[splits["train"]].float();dims=(0,1) if x.ndim==3 else (0,)
        if normalization=="standard":
            center=train.mean(dim=dims,keepdim=True);scale=train.std(dim=dims,keepdim=True).clamp_min(1e-7);x=(x.float()-center)/scale
            opts["inputTransform"]={"mode":"standard","center":center.flatten().tolist(),"scale":scale.flatten().tolist()}
        else:
            low=torch.amin(train,dim=dims,keepdim=True);high=torch.amax(train,dim=dims,keepdim=True);scale=(high-low).clamp_min(1e-7);x=(x.float()-low)/scale
            if normalization=="minmax_sym": x=x*2-1
            opts["inputTransform"]={"mode":normalization,"low":low.flatten().tolist(),"high":high.flatten().tolist()}
        if task_id == "tabular.reconstruction": y=x.clone()
    if task_id in {"tabular.regression","sequence.regression","sequence.forecast"}:
        y=_standardize_targets(y,splits,opts)
    return x, y, None, splits, split_source


def _load_ecg(spec: dict, task_id: str, opts: dict, offline: bool):
    x=torch.from_numpy(np.load(_hf_file(spec,"data/processed/X.npy",offline),allow_pickle=False)).float()
    y=torch.from_numpy(np.load(_hf_file(spec,"data/processed/y.npy",offline),allow_pickle=False)).long()
    limit=max(5,int(opts.get("max_samples",10000)));per_class=max(1,limit//5);generator=torch.Generator().manual_seed(42);selected=[]
    for class_index in range(5):
        indices=(y==class_index).nonzero(as_tuple=False).flatten();indices=indices[torch.randperm(len(indices),generator=generator)[:per_class]];selected.extend(indices.tolist())
    selected=torch.tensor(selected)[torch.randperm(len(selected),generator=generator)];x=x[selected].unsqueeze(-1);y=y[selected]
    classes = ["Contracción auricular prematura", "Bloqueo de rama izquierda", "Normal", "Bloqueo de rama derecha", "Contracción ventricular prematura"]
    return x, y, classes


def _load_mnist(spec: dict, task_id: str, opts: dict, offline: bool):
    train_part=_hf_dataset(spec,split="train",offline=offline);test_part=_hf_dataset(spec,split="test",offline=offline)
    limit=min(len(train_part)+len(test_part),int(opts.get("max_samples",12000)))
    train_size=max(2,round(limit*len(train_part)/(len(train_part)+len(test_part))));test_size=limit-train_size
    parts=[train_part.select(range(train_size)),test_part.select(range(test_size))];resolution=int(opts["resolution"]);channels=int(opts["channels"])
    rows=[row for part in parts for row in part]
    x = torch.stack([_pil_tensor(row["image"], resolution, channels) for row in rows])
    splits=_train_validation_test_splits(train_size,test_size)
    if task_id == "image.reconstruction":
        x = x.float() / 255.0
        return x, x.clone(), None, splits
    return x, torch.tensor([int(row["label"]) for row in rows], dtype=torch.long), [str(index) for index in range(10)], splits


def _load_utkface(spec: dict, task_id: str, opts: dict, offline: bool):
    archive = _hf_file(spec, "UTKFace.tar.gz", offline)
    resolution = int(opts["resolution"]); channels = int(opts["channels"]); limit = int(opts.get("max_samples", 5000))
    images: list[torch.Tensor] = []; ages: list[float] = []
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if len(images) >= limit or not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filename = Path(member.name).name
            try:
                age = int(filename.split("_", 1)[0]); extracted = source.extractfile(member)
                if extracted is None: continue
                with Image.open(io.BytesIO(extracted.read())) as image:
                    images.append(_pil_tensor(image, resolution, channels))
                ages.append(float(age))
            except (ValueError, OSError):
                continue
    if len(images) < 10:
        raise BackendError("HF_DATASET_INVALID", "UTKFace no contiene suficientes imágenes válidas")
    x=torch.stack(images);y=torch.tensor(ages,dtype=torch.float32).unsqueeze(1);splits=_random_splits(len(x),42);y=_standardize_targets(y,splits,opts)
    return x,y,None,splits,"generated"


def _tokenize_texts(texts: list[str], opts: dict, labels: torch.Tensor | None = None, vocab: list[str] | None = None):
    import re
    words = [re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE) for text in texts]
    if vocab is None:
        counts: dict[str, int] = {}
        for row in words:
            for word in row: counts[word] = counts.get(word, 0) + 1
        max_vocab = int(opts.get("vocab_size", 4096)); special = ["<pad>", "<unk>"] + (["<cls>"] if opts.get("add_cls") else []); vocab = special + [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_vocab - len(special)]]
    lookup = {word: index for index, word in enumerate(vocab)}; length = int(opts.get("max_length", 64))
    x = torch.zeros(len(words), length, dtype=torch.long)
    for index, row in enumerate(words):
        ids = ([lookup["<cls>"]] if opts.get("add_cls") else []) + [lookup.get(word, 1) for word in row[:length - (1 if opts.get("add_cls") else 0)]]
        if ids: x[index, :len(ids)] = torch.tensor(ids)
    return x, labels, vocab


def _official_ranges(lengths: list[int]) -> dict[str, list[int]]:
    names=("train","validation","test");start=0;result={}
    for name,length in zip(names,lengths):
        result[name]=list(range(start,start+length));start+=length
    return result


def _random_splits(length: int, seed: int) -> dict[str,list[int]]:
    order=torch.randperm(length,generator=torch.Generator().manual_seed(seed)).tolist();a=round(length*.7);b=a+round(length*.15)
    return {"train":order[:a],"validation":order[a:b],"test":order[b:]}


def _contiguous_splits(length: int) -> dict[str,list[int]]:
    a=round(length*.7);b=a+round(length*.15)
    return {"train":list(range(a)),"validation":list(range(a,b)),"test":list(range(b,length))}


def _train_validation_test_splits(train_length: int,test_length: int,seed: int=42) -> dict[str,list[int]]:
    order=torch.randperm(train_length,generator=torch.Generator().manual_seed(seed)).tolist();validation_size=max(1,round(train_length*.1))
    return {"train":order[validation_size:],"validation":order[:validation_size],"test":list(range(train_length,train_length+test_length))}


def _standardize_targets(y: torch.Tensor,splits: dict[str,list[int]],opts: dict) -> torch.Tensor:
    train=y[splits["train"]].float();center=train.mean(0,keepdim=True);scale=train.std(0,keepdim=True).clamp_min(1e-7)
    opts["targetTransform"]={"center":center.flatten().tolist(),"scale":scale.flatten().tolist()}
    return (y.float()-center)/scale


def _proportional_limit(parts: list[list[str]], limit: int) -> list[list[str]]:
    total=sum(len(part) for part in parts)
    if limit<=0 or total<=limit:
        return parts
    exact=[len(part)*limit/total for part in parts]
    sizes=[max(1,int(value)) for value in exact]
    while sum(sizes)>limit:
        index=max((i for i,size in enumerate(sizes) if size>1),key=lambda i:sizes[i]-exact[i])
        sizes[index]-=1
    while sum(sizes)<limit:
        index=max(range(len(parts)),key=lambda i:exact[i]-sizes[i])
        sizes[index]+=1
    return [part[:size] for part,size in zip(parts,sizes)]


def _load_emotion(spec: dict, task_id: str, opts: dict, offline: bool):
    opts["add_cls"] = True
    parts = [_hf_dataset(spec, name="split", split=name, offline=offline) for name in ("train", "validation", "test")]
    text_parts=[[str(text) for text in part["text"]] for part in parts]
    texts=[text for part in text_parts for text in part]
    labels = torch.tensor([int(label) for part in parts for label in part["label"]], dtype=torch.long)
    _,_,vocab=_tokenize_texts(text_parts[0],opts)
    x, y, vocab = _tokenize_texts(texts, opts, labels, vocab)
    opts["vocab"] = vocab
    return x, y, ["Tristeza", "Alegría", "Amor", "Ira", "Miedo", "Sorpresa"], _official_ranges([len(part) for part in text_parts])


def _load_wikitext(spec: dict, task_id: str, opts: dict, offline: bool):
    parts = [_hf_dataset(spec, name="wikitext-2-raw-v1", split=name, offline=offline) for name in ("train", "validation", "test")]
    text_parts=[[str(text).strip() for text in part["text"] if str(text).strip()] for part in parts]
    text_parts=_proportional_limit(text_parts,int(opts.get("max_samples",20000)))
    _,_,vocab=_tokenize_texts(text_parts[0],opts)
    texts=[text for part in text_parts for text in part]
    x, _, vocab = _tokenize_texts(texts, opts, vocab=vocab)
    y = x.roll(-1, dims=1); y[:, -1] = 0; opts["vocab"] = vocab
    return x, y, vocab, _official_ranges([len(part) for part in text_parts])


def _load_segmentation(spec: dict, task_id: str, opts: dict, offline: bool):
    parts = [_hf_dataset(spec, split=name, offline=offline) for name in ("train", "test")]
    rows = [row for part in parts for row in part]
    splits=_train_validation_test_splits(len(parts[0]),len(parts[1]))
    resolution = int(opts["resolution"]); channels = int(opts["channels"]); mask_key = "mask" if task_id.endswith("binary") else "type_map"
    x = torch.stack([_pil_tensor(row["image"], resolution, channels) for row in rows])
    masks = torch.stack([_mask_tensor(row[mask_key], resolution) for row in rows])
    if task_id.endswith("binary"):
        return x, (masks > 0).long(), ["Objeto"], splits
    # Ambiguous class 5 is not a scored MoNuSAC class; fold it into background.
    masks[masks == 5] = 0
    return x, masks, ["Fondo", "Epitelial", "Linfocito", "Macrófago", "Neutrófilo"], splits


LOADERS: dict[str, Callable] = {
    "iris": _load_iris, "energy": _load_energy, "ecg": _load_ecg,
    "mnist": _load_mnist, "utkface": _load_utkface,
    "emotion": _load_emotion, "wikitext": _load_wikitext,
    "monuseg": _load_segmentation, "monusac": _load_segmentation,
}


def download(project: dict, task_id: str, dataset_id: str, options: dict | None = None) -> dict:
    spec = DATASETS.get(dataset_id)
    if not spec:
        raise BackendError("HF_DATASET_UNKNOWN", dataset_id)
    if task_id not in spec["tasks"]:
        raise BackendError("HF_DATASET_INCOMPATIBLE", f"{dataset_id} / {task_id}")
    opts = _options(spec, options)
    options_signature=_signature(opts)
    existing = _existing_project_dataset(project, task_id, dataset_id, opts)
    if existing:
        return existing

    marker = _marker(dataset_id); offline = marker.is_file(); loader = LOADERS[spec["loader"]]
    try:
        loaded = loader(spec, task_id, opts, offline)
    except BackendError:
        if not offline:
            raise
        # A stale marker must not make the feature permanently unusable.
        marker.unlink(missing_ok=True)
        loaded = loader(spec, task_id, opts, False)
    x,y,classes=loaded[:3]
    official_splits=loaded[3] if len(loaded)>3 else None
    if len(x) < 10 or len(x) != len(y) or not torch.isfinite(x.float()).all() or not torch.isfinite(y.float()).all():
        raise BackendError("HF_DATASET_INVALID", dataset_id)

    atomic_json(marker, {
        "datasetId": dataset_id, "repoId": spec["repoId"], "revision": spec["revision"],
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
    })
    summary = _save(
        project, task_id, x, y, spec["description"], 42, classes, opts,
        source="huggingface", source_path=f"https://huggingface.co/datasets/{spec['repoId']}",
        metadata={
            "catalogId": dataset_id, "repoId": spec["repoId"], "repoRevision": spec["revision"],
            "license": spec["license"], "downloadSizeBytes": spec["sizeBytes"],
            "optionsSignature": options_signature, "dataVersion":int(spec.get("dataVersion",1)), "cacheStatus": "disk" if offline else "downloaded",
        },
        split_indices_override=official_splits,
        split_source=(loaded[4] if len(loaded)>4 else "official" if official_splits else "generated"),
    )
    return summary
