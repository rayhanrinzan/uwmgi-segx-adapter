#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash offline/export_results_to_usb.sh /media/$USER/SEGX_USB"
  exit 1
fi

USB_ROOT="$1"
ROOT="${UWMGI_ROOT:-$HOME/uwmgi-offline}"
ADAPTER_DIR="$ROOT/uwmgi-segx-adapter"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS="$USB_ROOT/results/$STAMP"

mkdir -p "$RESULTS"

rsync -avP "$ADAPTER_DIR/workdir/experiments/" "$RESULTS/experiments/"
rsync -avP "$ADAPTER_DIR/workdir/wandb-offline/" "$RESULTS/wandb-offline/"
rsync -avP "$ADAPTER_DIR/workdir/logs/" "$RESULTS/logs/"
cp "$ADAPTER_DIR/configs/gi_organ.yaml" "$RESULTS/"

cd "$RESULTS"
find . -type f ! -name RESULTS_SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > RESULTS_SHA256SUMS
sync

echo "Results copied to: $RESULTS"
