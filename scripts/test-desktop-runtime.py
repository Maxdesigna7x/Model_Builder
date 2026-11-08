"""Smoke-test a clean CPU runtime using the same commands as the desktop bootstrap."""
from pathlib import Path
import json
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="modelbuilder-smoke-") as tmp:
    tmp = Path(tmp)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(tmp / "python"), UV_NO_PROGRESS="1", UV_HTTP_TIMEOUT="300")
    venv = tmp / "venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(["uv", "venv", "--clear", "--managed-python", "--python", "3.12", str(venv)], env=env, check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--torch-backend", "cpu", "-r", str(root / "pyproject.toml")], env=env, check=True)
    subprocess.run([str(python), "-c", "import torch, torchvision, numpy, PIL, datasets, transformers, ultralytics, cv2"], env=env, check=True)
    env["MODELBUILDER_WORKSPACE"] = str(tmp / "workspace")
    result = subprocess.run([str(python), str(root / "backend/modelbuilder/engine.py")], input='{"action":"hello"}', text=True, capture_output=True, env=env, check=True)
    response = json.loads(result.stdout.splitlines()[-1])
    if not response.get("ok"):
        raise RuntimeError(response)
    print("Clean runtime and backend smoke test passed")
