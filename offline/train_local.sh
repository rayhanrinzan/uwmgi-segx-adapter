#!/usr/bin/env bash
set -euo pipefail

ROOT="${UWMGI_ROOT:-$HOME/uwmgi-offline}"
SEGX_DIR="$ROOT/segx"
ADAPTER_DIR="$ROOT/uwmgi-segx-adapter"
DATA_DIR="$ROOT/data/segx_gi"
UV="$ROOT/runtime/bin/uv"

export UV_CACHE_DIR="$ROOT/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/runtime/python"
export UV_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export WANDB_MODE=offline
export WANDB_ENTITY="rayhanrinzan-cornell-university"
export WANDB_PROJECT="uwmgi-segx"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-resnet34-baseline-local}"
export WANDB_DIR="$ADAPTER_DIR/workdir/wandb-offline"
export WANDB_CACHE_DIR="$ADAPTER_DIR/workdir/wandb-cache"
export WANDB_DISABLE_GIT=true

export SEGX_BACKBONE="${SEGX_BACKBONE:-resnet34}"
export SEGX_FOLD="${SEGX_FOLD:-0}"
export SEGX_EPOCHS="${SEGX_EPOCHS:-1}"
export SEGX_BATCH_SIZE="${SEGX_BATCH_SIZE:-32}"
export SEGX_LR="${SEGX_LR:-0.0001}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
export WANDB_NAME="${WANDB_NAME:-resnet34-fold${SEGX_FOLD}-local-${TIMESTAMP}}"
LOG_DIR="$ADAPTER_DIR/workdir/logs"
LOG_FILE="$LOG_DIR/local_${TIMESTAMP}.log"

mkdir -p \
  "$LOG_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$ADAPTER_DIR/workdir/cache/splits_local" \
  "$ADAPTER_DIR/workdir/cache/data_local" \
  "$ADAPTER_DIR/workdir/experiments"

if [ ! -f "$DATA_DIR/metadata.csv" ]; then
  echo "ERROR: metadata.csv not found at $DATA_DIR"
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "Started: $(date)"
echo "Fold: $SEGX_FOLD | Epochs: $SEGX_EPOCHS | Batch size: $SEGX_BATCH_SIZE | LR: $SEGX_LR"
echo "Log: $LOG_FILE"

cd "$SEGX_DIR"
"$UV" run --offline --frozen --no-dev --with wandb python \
  "$ADAPTER_DIR/scripts/train_with_wandb.py" \
  config="$ADAPTER_DIR/configs/gi_organ.yaml" \
  data.data_dir="$DATA_DIR" \
  data.splits_dir="$ADAPTER_DIR/workdir/cache/splits_local" \
  data.data_cache_dir="$ADAPTER_DIR/workdir/cache/data_local" \
  snapshot_root="$ADAPTER_DIR/workdir/experiments" \
  model.params.encoder_weights=null \
  train.fold="$SEGX_FOLD" \
  train.num_epochs="$SEGX_EPOCHS" \
  train.bs="$SEGX_BATCH_SIZE" \
  train.lr="$SEGX_LR" \
  data.num_workers=4 \
  2>&1 | tee "$LOG_FILE"

echo "Finished successfully: $(date)"
