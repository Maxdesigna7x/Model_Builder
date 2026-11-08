#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
fail() { echo "ERROR: $*" >&2; exit 1; }
version_at_least() { printf '%s\n' "$2" "$1" | sort -V -C; }
command -v apt-get >/dev/null || fail "Este instalador requiere Ubuntu/Debian."
command -v sudo >/dev/null || fail "sudo no está instalado."
command -v node >/dev/null || fail "Instala Node.js 20.19+ (rama 20) o 22.12+ antes de continuar."
NODE_VERSION="$(node -p 'process.versions.node')"
node -e 'const [m,n]=process.versions.node.split(".").map(Number);process.exit((m===20&&n>=19)||(m>=22&&(m>22||n>=12))?0:1)' || fail "Node.js 20.19+ (rama 20) o 22.12+ requerido; encontrado $NODE_VERSION."
command -v npm >/dev/null || fail "npm no está disponible."
PYTHON_BIN="$(command -v python3.12 || command -v python3.11 || true)"
[ -n "$PYTHON_BIN" ] || fail "Instala Python 3.11 o 3.12 y su paquete venv antes de continuar. No se usará Conda base."
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
version_at_least "$PYTHON_VERSION" "3.11.0" || fail "Python 3.11+ requerido; encontrado $PYTHON_VERSION."
PYTHON_MINOR="$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PYTHON_VENV_PACKAGE="python${PYTHON_MINOR}-venv"
echo "==> Dependencias nativas de Tauri"
sudo apt-get update
sudo apt-get install -y build-essential curl file pkg-config libayatana-appindicator3-dev librsvg2-dev libwebkit2gtk-4.1-dev libxdo-dev patchelf wget "$PYTHON_VENV_PACKAGE"
if ! command -v rustup >/dev/null; then
  echo "==> Instalando Rust estable"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
command -v cargo >/dev/null || fail "cargo no está disponible; abre una nueva terminal y reintenta."
echo "==> Entorno Python"
"$PYTHON_BIN" -m venv .venv
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu130"
fi
.venv/bin/python -m pip install --no-cache-dir --timeout 180 --retries 8 \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
  'torch==2.12.0' 'torchvision==0.27.0' 'numpy>=1.26,<2.0' 'Pillow>=10,<15'
if [ "$PYTHON_MINOR" = "3.12" ] && [ "$TORCH_INDEX" = "https://download.pytorch.org/whl/cu130" ]; then
  .venv/bin/python -m pip install --no-cache-dir -c constraints-linux-py312-cu130.txt -e '.[dev]'
else
  .venv/bin/python -m pip install --no-cache-dir -e '.[dev]'
fi
.venv/bin/python -m pip check
echo "==> Dependencias JavaScript"
npm ci
echo "Instalación completada. Ejecuta ./run.sh o ./run-web.sh."
