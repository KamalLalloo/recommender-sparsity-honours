#!/bin/bash

#SBATCH --job-name=ml_tuning
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.out
#SBATCH --error=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.err

set -euo pipefail

PROJECT_DIR="/home-mscluster/klalloo/recommender-sparsity-honours"
PYTHON="/home-mscluster/klalloo/.conda/envs/recbole_env/bin/python"
SEQUENCE_SCRIPT="$PROJECT_DIR/slurm/run_movielens_tuning_sequence.slurm"
DATASET_DIR="data/recbole/movielens/baseline"

MODELS=(
    itemknn
    bpr
    ease
    neumf
    multivae
    gru4rec
    sasrec
    bert4rec
    lightgcn
)

MODEL_SEED=2025

TOTAL_EXPERIMENTS=${#MODELS[@]}
DEFAULT_LAST_INDEX=$((TOTAL_EXPERIMENTS - 1))

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch [Slurm options] $SEQUENCE_SCRIPT START_INDEX [LAST_INDEX]"
    echo "Tuning indices range from 0 to $DEFAULT_LAST_INDEX."
    exit 1
fi

CURRENT_INDEX="$1"
LAST_INDEX="${2:-$DEFAULT_LAST_INDEX}"

if ! [[ "$CURRENT_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: START_INDEX must be a non-negative integer."
    exit 1
fi

if ! [[ "$LAST_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LAST_INDEX must be a non-negative integer."
    exit 1
fi

if (( CURRENT_INDEX >= TOTAL_EXPERIMENTS )); then
    echo "ERROR: START_INDEX is outside the tuning list."
    echo "Maximum index: $DEFAULT_LAST_INDEX"
    exit 1
fi

if (( LAST_INDEX >= TOTAL_EXPERIMENTS )); then
    echo "ERROR: LAST_INDEX is outside the tuning list."
    echo "Maximum index: $DEFAULT_LAST_INDEX"
    exit 1
fi

if (( CURRENT_INDEX > LAST_INDEX )); then
    echo "ERROR: START_INDEX cannot exceed LAST_INDEX."
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable was not found:"
    echo "$PYTHON"
    exit 1
fi

if [[ ! -f "$SEQUENCE_SCRIPT" ]]; then
    echo "ERROR: Sequence script was not found:"
    echo "$SEQUENCE_SCRIPT"
    exit 1
fi

if [[ ! -d "$PROJECT_DIR/$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory was not found:"
    echo "$PROJECT_DIR/$DATASET_DIR"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/movielens.inter" ]]; then
    echo "ERROR: MovieLens interaction file was not found."
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/movielens.item" ]]; then
    echo "ERROR: MovieLens fixed item catalogue was not found."
    exit 1
fi

MODEL="${MODELS[$CURRENT_INDEX]}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p results/tuning/movielens

echo "============================================================"
echo "MOVIELENS BASELINE HYPERPARAMETER TUNING"
echo "============================================================"
echo "Job ID            : ${SLURM_JOB_ID:-unknown}"
echo "Job name          : ${SLURM_JOB_NAME:-unknown}"
echo "Node              : $(hostname)"
echo "Partition         : ${SLURM_JOB_PARTITION:-unknown}"
echo "Started at        : $(date)"
echo "Git commit        : $(git rev-parse HEAD)"
echo "Current index     : $CURRENT_INDEX"
echo "Last index        : $LAST_INDEX"
echo "Model             : $MODEL"
echo "Tuning seed       : $MODEL_SEED"
echo "Dataset directory : $DATASET_DIR"
echo "Selection metric  : validation NDCG@10"

echo "============================================================"
echo "GPU INFORMATION"
echo "============================================================"

nvidia-smi || true

echo "============================================================"
echo "PYTHON ENVIRONMENT"
echo "============================================================"

"$PYTHON" - <<'PY'
import sys
import recbole
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("RecBole:", recbole.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: GPU Slurm job cannot access CUDA."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "============================================================"
echo "STARTING TUNING: $MODEL"
echo "============================================================"

"$PYTHON" scripts/tune_model.py \
    --dataset movielens \
    --model "$MODEL" \
    --dataset-dir "$DATASET_DIR" \
    --seed "$MODEL_SEED" \
    --use-gpu

echo "============================================================"
echo "$MODEL TUNING COMPLETED"
echo "============================================================"
echo "Finished at: $(date)"

if (( CURRENT_INDEX < LAST_INDEX )); then
    NEXT_INDEX=$((CURRENT_INDEX + 1))
    NEXT_MODEL="${MODELS[$NEXT_INDEX]}"
    NEXT_JOB_NAME="${NEXT_MODEL}_ml_tuning"

    NEXT_JOB_ID=$(
        sbatch \
            --parsable \
            --job-name="$NEXT_JOB_NAME" \
            "$SEQUENCE_SCRIPT" \
            "$NEXT_INDEX" \
            "$LAST_INDEX"
    )

    echo "Next tuning job submitted successfully."
    echo "Next model : $NEXT_MODEL"
    echo "Next job ID: $NEXT_JOB_ID"
else
    echo "============================================================"
    echo "ALL MOVIELENS BASELINE TUNING COMPLETED"
    echo "============================================================"
fi