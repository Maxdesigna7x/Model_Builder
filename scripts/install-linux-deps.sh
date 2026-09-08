#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  build-essential \
  curl \
  file \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  libwebkit2gtk-4.1-dev \
  libxdo-dev \
  patchelf \
  wget

echo "Dependencias nativas de Tauri instaladas. Ejecuta ./run.sh"
