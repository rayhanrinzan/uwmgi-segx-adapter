#!/bin/bash
#SBATCH --exclude=ai-gpu15
#SBATCH --job-name=segx_gi
#SBATCH --partition=preempt_gpu
#SBATCH --account=imedslab
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/athena/imedslab/rar4037/uwmgi-segx-adapter/workdir/logs/train_%j.out
#SBATCH --error=/athena/imedslab/rar4037/uwmgi-segx-adapter/workdir/logs/train_%j.err

SEGX_DIR="/athena/imedslab/rar4037/segx"
ADAPTER_DIR="/athena/imedslab/rar4037/uwmgi-segx-adapter"
ARCHIVE="/athena/imedslab/scratch/rar4037/segx_gi.tar.gz"
DATA_DIR="/athena/imedslab/scratch/rar4037/segx_gi"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/athena/imedslab/scratch/rar4037/.cache/uv"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Confirm the W&B API key exists.
WANDB_KEY_FILE="$HOME/.config/wandb/api_key"

if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "ERROR: W&B API key not found at $WANDB_KEY_FILE"
    exit 1
fi

# W&B authentication and run settings.
export WANDB_API_KEY="$(cat "$WANDB_KEY_FILE")"
export WANDB_ENTITY="rayhanrinzan-cornell-university"
export WANDB_PROJECT="uwmgi-segx"
export WANDB_RUN_GROUP="resnet34-baseline"
export WANDB_NAME="resnet34-fold0-job-${SLURM_JOB_ID}"
export WANDB_MODE="online"

# W&B local files and TensorBoard synchronization root.
export WANDB_DIR="/athena/imedslab/rar4037/wandb"
export WANDB_CACHE_DIR="/athena/imedslab/rar4037/.cache/wandb"
export WANDB_TENSORBOARD_ROOT="$ADAPTER_DIR/workdir/experiments"

# Experiment settings recorded by the W&B wrapper.
export SEGX_BACKBONE="resnet34"
export SEGX_FOLD="0"
export SEGX_EPOCHS="30"
export SEGX_BATCH_SIZE="16"
export SEGX_LR="0.001"

mkdir -p \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_TENSORBOARD_ROOT" \
    "$ADAPTER_DIR/workdir/logs" \
    "$ADAPTER_DIR/workdir/cache/splits"

echo "Running on: $(hostname)"
echo "Started: $(date)"
echo "W&B run name: $WANDB_NAME"

nvidia-smi

# Unpack the dataset only when it is not already available.
if [ ! -f "$DATA_DIR/metadata.csv" ]; then
    echo "Unpacking dataset..."

    if [ ! -f "$ARCHIVE" ]; then
        echo "ERROR: Dataset archive not found at $ARCHIVE"
        exit 1
    fi

    tar --warning=no-unknown-keyword \
        -xzf "$ARCHIVE" \
        -C /athena/imedslab/scratch/rar4037
fi

if [ ! -f "$DATA_DIR/metadata.csv" ]; then
    echo "ERROR: metadata.csv not found at $DATA_DIR"
    exit 1
fi

# Install uv when it is unavailable.
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user uv
fi

# Set up the SegX environment.
cd "$SEGX_DIR"

uv python install 3.12
uv sync

# Train SegX while synchronizing its TensorBoard logs to W&B.
uv run --with wandb python \
    "$ADAPTER_DIR/scripts/train_with_wandb.py" \
    config="$ADAPTER_DIR/configs/gi_organ.yaml" \
    data.datasets.root_dir="$DATA_DIR" \
    data.splits_dir="$ADAPTER_DIR/workdir/cache/splits" \
    snapshot_root="$ADAPTER_DIR/workdir/experiments" \
    model.params.encoder_weights=null \
    train.fold="$SEGX_FOLD" \
    train.num_epochs="$SEGX_EPOCHS" \
    train.bs="$SEGX_BATCH_SIZE" \
    data.num_workers=4

echo "Finished: $(date)"
