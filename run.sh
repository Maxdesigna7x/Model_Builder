#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-1420}

free_port() {
  local port=$1
  if command -v fuser >/dev/null 2>&1; then
    if fuser "$port/tcp" >/dev/null 2>&1; then
      echo "⚠️ El puerto $port está ocupado. Liberando puerto..."
      fuser -k -9 "$port/tcp" >/dev/null 2>&1 || true
      sleep 1
    fi
  elif command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -t -i:"$port" || true)
    if [ -n "$pids" ]; then
      echo "⚠️ El puerto $port está ocupado. Liberando proceso(s)..."
      kill -9 $pids >/dev/null 2>&1 || true
      sleep 1
    fi
  fi
}

free_port "$PORT"

npm run tauri dev
