#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${UWMGI_ROOT:-$HOME/uwmgi-offline}"
PAYLOAD_COPY="$ROOT/payload"

for file in uwmgi_code_runtime.tar.gz segx_gi.tar.gz SHA256SUMS; do
  if [ ! -f "$SOURCE_DIR/$file" ]; then
    echo "ERROR: Missing $SOURCE_DIR/$file"
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not installed. The NVIDIA driver must be installed first."
  exit 1
fi

nvidia-smi
mkdir -p "$PAYLOAD_COPY"
cp -av "$SOURCE_DIR/uwmgi_code_runtime.tar.gz" "$SOURCE_DIR/segx_gi.tar.gz" "$SOURCE_DIR/SHA256SUMS" "$PAYLOAD_COPY/"

cd "$PAYLOAD_COPY"
sha256sum -c SHA256SUMS

tar -xzf uwmgi_code_runtime.tar.gz -C "$ROOT"
mkdir -p "$ROOT/data"
tar --warning=no-unknown-keyword -xzf segx_gi.tar.gz -C "$ROOT/data"

UV="$ROOT/runtime/bin/uv"
chmod +x "$UV"
export UV_CACHE_DIR="$ROOT/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/runtime/python"
export UV_OFFLINE=1

cd "$ROOT/segx"
rm -rf .venv
"$UV" sync --offline --frozen --no-dev --python 3.12

"$UV" run --offline --frozen --no-dev --with wandb --with tensorboard python - <<'PY'
import torch
import wandb
import tensorboard
import segx

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the NVIDIA GPU")
print("GPU:", torch.cuda.get_device_name(0))
print("Offline setup test passed")
PY

chmod +x \
  "$ROOT/uwmgi-segx-adapter/offline/train_local.sh" \
  "$ROOT/uwmgi-segx-adapter/offline/start_tensorboard.sh" \
  "$ROOT/uwmgi-segx-adapter/offline/export_results_to_usb.sh"

echo
echo "Installation complete."
echo "Project root: $ROOT"
echo
echo "Run a one-epoch test with:"
echo "  cd $ROOT/uwmgi-segx-adapter"
echo "  SEGX_EPOCHS=1 bash offline/train_local.sh"
