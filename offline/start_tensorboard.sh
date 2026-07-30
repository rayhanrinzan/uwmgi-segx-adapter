#!/usr/bin/env bash
set -euo pipefail

ROOT="${UWMGI_ROOT:-$HOME/uwmgi-offline}"
UV="$ROOT/runtime/bin/uv"

export UV_CACHE_DIR="$ROOT/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/runtime/python"
export UV_OFFLINE=1

cd "$ROOT/segx"
"$UV" run --offline --frozen --no-dev --with tensorboard \
  tensorboard \
  --logdir "$ROOT/uwmgi-segx-adapter/workdir/experiments" \
  --host 127.0.0.1 \
  --port 6006
