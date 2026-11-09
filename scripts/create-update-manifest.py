"""Create the signed Tauri update feed from both platform artifacts."""
from pathlib import Path
import datetime
import json
import os
import re
import sys
from urllib.parse import quote

root, tag = Path(sys.argv[1]), sys.argv[2]
repo = os.environ["GH_REPO"]
platforms = {}
versions = set()
for platform, pattern in (("linux-x86_64", "*.AppImage"), ("windows-x86_64-nsis", "*-setup.exe"), ("windows-x86_64-msi", "*.msi")):
    files = list(root.rglob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"Expected one {platform} artifact, got {len(files)}")
    artifact = files[0]
    match = re.search(r"_(\d+\.\d+\.\d+)_", artifact.name)
    if not match:
        raise RuntimeError(f"Missing installer version: {artifact.name}")
    versions.add(match[1])
    signature = artifact.with_name(artifact.name + ".sig").read_text().strip()
    if not signature:
        raise RuntimeError(f"Empty signature for {artifact.name}")
    platforms[platform] = {"url": f"https://github.com/{repo}/releases/download/{quote(tag, safe='')}/{quote(artifact.name, safe='')}", "signature": signature}
platforms["windows-x86_64"] = platforms["windows-x86_64-nsis"]
if len(versions) != 1:
    raise RuntimeError("Windows and Linux versions differ")
(root / "latest.json").write_text(json.dumps({"version": versions.pop(), "notes": "Actualización de ModelBuilder. Guarda tus cambios antes de instalar.", "pub_date": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"), "platforms": platforms}, indent=2) + "\n")
