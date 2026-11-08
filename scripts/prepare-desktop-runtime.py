"""Stage uv for the current build host. Install uv 0.9.30 before running."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

uv = shutil.which("uv")
if not uv:
    sys.exit("Install uv 0.9.30 and add it to PATH before building the desktop app.")
version = subprocess.check_output([uv, "--version"], text=True).strip()
if version.split()[:2] != ["uv", "0.9.30"]:
    sys.exit(f"Expected uv 0.9.30, got {version}.")
runtime = Path(__file__).resolve().parents[1] / "src-tauri" / "runtime"
runtime.mkdir(parents=True, exist_ok=True)
for existing in (runtime / "uv", runtime / "uv.exe"):
    if existing.exists():
        existing.unlink()
target = runtime / ("uv.exe" if os.name == "nt" else "uv")
shutil.copy2(uv, target)
print(f"Staged {version} at {target}")
