#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  export MODELBUILDER_PYTHON="$(pwd)/.venv/bin/python"
fi

export PORT=${PORT:-1420}

npm run dev
