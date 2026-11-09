"""Exercise the desktop install sequence in a clean environment.

Use --backend nvidia on a machine with a compatible NVIDIA driver to test CUDA.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cpu", "nvidia"), default="cpu")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="modelbuilder-smoke-") as tmp:
    tmp = Path(tmp)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(tmp / "python"), UV_NO_PROGRESS="1", UV_HTTP_TIMEOUT="300")
    venv = tmp / "venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    backend = "cu130" if args.backend == "nvidia" else "cpu"
    subprocess.run(["uv", "venv", "--clear", "--managed-python", "--python", "3.12", str(venv)], env=env, check=True)
    lock = tmp / "requirements.txt"
    subprocess.run(["uv", "pip", "compile", "--python", str(python), "--torch-backend", backend, "--no-header", "--no-annotate", "--output-file", str(lock), str(root / "pyproject.toml")], env=env, check=True)
    if args.backend == "cpu":
        cuda_lock = tmp / "nvidia-requirements.txt"
        subprocess.run(["uv", "pip", "compile", "--python", str(python), "--torch-backend", "cu130", "--no-header", "--no-annotate", "--output-file", str(cuda_lock), str(root / "pyproject.toml")], env=env, check=True)
        if "torch==2.12.0+cu130" not in cuda_lock.read_text():
            raise RuntimeError("NVIDIA resolution did not select CUDA PyTorch")
    packages = [line.strip() for line in lock.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not packages or any("==" not in package or package.startswith("-") for package in packages):
        raise RuntimeError("Invalid dependency list")
    for index, package in enumerate(packages, 1):
        print(f"[{index}/{len(packages)}] {package}", flush=True)
        subprocess.run(["uv", "pip", "install", "--no-deps", "--python", str(python), "--torch-backend", backend, package], env=env, check=True)
    subprocess.run(["uv", "pip", "check", "--python", str(python)], env=env, check=True)
    check = "import torch, torchvision, numpy, PIL, datasets, transformers, ultralytics, cv2"
    if args.backend == "nvidia":
        check += "; assert torch.version.cuda and torch.cuda.is_available(); torch.ones(1, device='cuda').cpu()"
    subprocess.run([str(python), "-c", check], env=env, check=True)
    env["MODELBUILDER_WORKSPACE"] = str(tmp / "workspace")
    result = subprocess.run([str(python), str(root / "backend/modelbuilder/engine.py")], input='{"action":"hello"}', text=True, capture_output=True, env=env, check=True)
    response = json.loads(result.stdout.splitlines()[-1])
    if not response.get("ok") or (args.backend == "nvidia" and not response["result"]["cuda"]):
        raise RuntimeError(response)
    print(f"Clean {args.backend} runtime and backend smoke test passed")
