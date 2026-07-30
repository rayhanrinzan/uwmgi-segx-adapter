#!/usr/bin/env bash
set -euo pipefail

SEGX_DIR="${SEGX_DIR:-/athena/imedslab/rar4037/segx}"
ADAPTER_DIR="${ADAPTER_DIR:-/athena/imedslab/rar4037/uwmgi-segx-adapter}"
DATA_ARCHIVE="${DATA_ARCHIVE:-/athena/imedslab/scratch/rar4037/segx_gi.tar.gz}"
BUILD_DIR="${BUILD_DIR:-/athena/imedslab/rar4037/uwmgi_offline_build}"
PAYLOAD_DIR="${PAYLOAD_DIR:-/athena/imedslab/rar4037/uwmgi_usb_payload}"

rm -rf "$BUILD_DIR" "$PAYLOAD_DIR"
mkdir -p "$BUILD_DIR/runtime/bin" "$BUILD_DIR/runtime/python" "$BUILD_DIR/runtime/uv-cache" "$PAYLOAD_DIR"

rsync -a --exclude=.git --exclude=.venv --exclude=__pycache__ \
  "$SEGX_DIR/" "$BUILD_DIR/segx/"

rsync -a --exclude=.git --exclude=__pycache__ \
  --exclude=workdir/experiments --exclude=workdir/logs \
  --exclude=workdir/wandb-offline --exclude=workdir/wandb-cache \
  "$ADAPTER_DIR/" "$BUILD_DIR/uwmgi-segx-adapter/"

cp "$DATA_ARCHIVE" "$PAYLOAD_DIR/segx_gi.tar.gz"
cp -L "$(command -v uv)" "$BUILD_DIR/runtime/bin/uv"
chmod +x "$BUILD_DIR/runtime/bin/uv"

UV="$BUILD_DIR/runtime/bin/uv"
export UV_CACHE_DIR="$BUILD_DIR/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$BUILD_DIR/runtime/python"
export UV_PROJECT_ENVIRONMENT="$BUILD_DIR/runtime/build-venv"

"$UV" python install 3.12 --install-dir "$UV_PYTHON_INSTALL_DIR" --no-bin
cd "$BUILD_DIR/segx"
"$UV" sync --frozen --no-dev --python 3.12
"$UV" run --frozen --no-dev --with wandb --with tensorboard --python 3.12 \
  python -c "import torch, wandb, tensorboard, segx; print('Offline dependencies cached')"

export UV_OFFLINE=1
"$UV" run --offline --frozen --no-dev --with wandb --with tensorboard --python 3.12 \
  python -c "import torch, wandb, tensorboard, segx; print('Offline bundle test passed')"
unset UV_OFFLINE UV_PROJECT_ENVIRONMENT
rm -rf "$BUILD_DIR/runtime/build-venv"

{
  echo "Created: $(date)"
  echo "Adapter commit: $(git -C "$ADAPTER_DIR" rev-parse HEAD)"
  echo -n "SegX commit: "
  git -C "$SEGX_DIR" rev-parse HEAD 2>/dev/null || echo unavailable
} > "$BUILD_DIR/VERSION.txt"

tar -czf "$PAYLOAD_DIR/uwmgi_code_runtime.tar.gz" \
  -C "$BUILD_DIR" segx uwmgi-segx-adapter runtime VERSION.txt

cp "$ADAPTER_DIR/offline/bootstrap_from_usb.sh" "$PAYLOAD_DIR/START_HERE.sh"
cp "$ADAPTER_DIR/offline/README.md" "$PAYLOAD_DIR/README.md"
chmod +x "$PAYLOAD_DIR/START_HERE.sh"

cd "$PAYLOAD_DIR"
sha256sum uwmgi_code_runtime.tar.gz segx_gi.tar.gz > SHA256SUMS

echo
echo "Payload created at: $PAYLOAD_DIR"
du -sh "$PAYLOAD_DIR"
ls -lh "$PAYLOAD_DIR"
