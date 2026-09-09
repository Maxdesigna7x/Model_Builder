#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
fail() { echo "ERROR: $*" >&2; exit 1; }
version_at_least() { printf '%s\n' "$1" "$2" | sort -V -C; }
command -v apt-get >/dev/null || fail "Este instalador requiere Ubuntu/Debian."
command -v sudo >/dev/null || fail "sudo no está instalado."
echo "==> Dependencias nativas de Tauri"
sudo apt-get update
sudo apt-get install -y build-essential curl file pkg-config libayatana-appindicator3-dev librsvg2-dev libwebkit2gtk-4.1-dev libxdo-dev patchelf wget python3.11-venv
command -v node >/dev/null || fail "Instala Node.js 20+ antes de continuar."
NODE_VERSION="$(node -p 'process.versions.node')"
version_at_least "$NODE_VERSION" "20.0.0" || fail "Node.js 20+ requerido; encontrado $NODE_VERSION."
command -v npm >/dev/null || fail "npm no está disponible."
PYTHON_BIN="$(command -v python3.11 || command -v python3 || true)"
[ -n "$PYTHON_BIN" ] || fail "Python 3.11+ requerido."
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
version_at_least "$PYTHON_VERSION" "3.11.0" || fail "Python 3.11+ requerido; encontrado $PYTHON_VERSION."
if ! command -v rustup >/dev/null; then
  echo "==> Instalando Rust estable"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
command -v cargo >/dev/null || fail "cargo no está disponible; abre una nueva terminal y reintenta."
echo "==> Entorno Python"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
echo "==> Dependencias JavaScript"
npm ci
echo "Instalación completada. Ejecuta ./run.sh o ./run-web.sh."
