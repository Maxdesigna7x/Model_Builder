from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import BackendError


def workspace_root() -> Path:
    configured = os.environ.get("MODELBUILDER_WORKSPACE")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve() / "workspace_data"


def slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return clean or "project"


def project_dir(project: dict) -> Path:
    project_id = str(project.get("id", "")).strip()
    if not project_id or not re.fullmatch(r"[a-zA-Z0-9_-]+", project_id):
        raise BackendError("PROJECT_INVALID_ID")
    configured_path = Path(str(project.get("path", ""))).expanduser()
    if configured_path.is_absolute() and (configured_path / "project.json").is_file():
        manifest = json.loads((configured_path / "project.json").read_text(encoding="utf-8"))
        if manifest.get("id") != project_id:
            raise BackendError("PROJECT_FOLDER_MISMATCH")
        return configured_path.resolve()
    return workspace_root() / "projects" / project_id


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def create_project(project: dict) -> dict:
    root = project_dir(project)
    if root.exists():
        raise BackendError("PROJECT_ALREADY_EXISTS")
    for relative in ("datasets", "experiments/active/draft", "runs", "predictions", "exports", "cache"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {**project, "schema_version": 1, "created_at": project.get("createdAt") or iso_now(), "updated_at": iso_now()}
    atomic_json(root / "project.json", manifest)
    return manifest


def open_project(path: str) -> dict:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "project.json" if root.is_dir() else root
    if manifest_path.name != "project.json" or not manifest_path.is_file():
        raise BackendError("PROJECT_MISSING_MANIFEST")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not value.get("id"):
        raise BackendError("PROJECT_INCOMPATIBLE_SCHEMA")
    value["path"] = str(manifest_path.parent)
    state_path = manifest_path.parent / "state.json"
    if state_path.is_file():
        value["savedState"] = json.loads(state_path.read_text(encoding="utf-8"))
    return value


def save_project_state(project: dict, state: dict) -> dict:
    root = project_dir(project)
    atomic_json(root / "state.json", state)
    manifest_path = root / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["updated_at"] = iso_now()
    manifest["architecture"] = state.get("architecture")
    manifest["task_id"] = state.get("task")
    atomic_json(manifest_path, manifest)
    return {"saved": True}


def delete_project(project: dict) -> dict:
    import shutil
    try:
        root = project_dir(project)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
    return {"deleted": True}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

