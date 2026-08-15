#!/bin/bash

#SBATCH --job-name=amazon_early_sequence
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.out
#SBATCH --error=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.err

set -euo pipefail

PROJECT_DIR="/home-mscluster/klalloo/recommender-sparsity-honours"
PYTHON="/home-mscluster/klalloo/.conda/envs/recbole_env/bin/python"
SEQUENCE_SCRIPT="$PROJECT_DIR/slurm/run_amazon_early_sequence.slurm"
DATASET_ROOT="data/recbole/amazon/early"

# Indices 0-23 = Early 50%; 24-47 = Early 25%; 48-71 = Early 10%.
MODELS=(
    pop
    itemknn
    bpr
    bpr
    bpr
    ease
    neumf
    neumf
    neumf
    multivae
    multivae
    multivae
    gru4rec
    gru4rec
    gru4rec
    sasrec
    sasrec
    sasrec
    bert4rec
    bert4rec
    bert4rec
    lightgcn
    lightgcn
    lightgcn
)

MODEL_SEEDS=(
    2025
    2025
    2025
    2026
    2027
    2025
    2025
    2026
    2027
    2025
    2026
    2027
    2025
    2026
    2027
    2025
    2026
    2027
    2025
    2026
    2027
    2025
    2026
    2027
)

MODEL_COUNT=${#MODELS[@]}

RETENTION_LEVELS=(
    50
    25
    10
)
RETENTION_COUNT=${#RETENTION_LEVELS[@]}
TOTAL_EXPERIMENTS=$((MODEL_COUNT * RETENTION_COUNT))
DEFAULT_LAST_INDEX=$((TOTAL_EXPERIMENTS - 1))

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch [Slurm options] $SEQUENCE_SCRIPT START_INDEX [LAST_INDEX]"
    echo "Experiments run with --run-type final."
    echo "Experiment indices range from 0 to $DEFAULT_LAST_INDEX."
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
    echo "ERROR: START_INDEX is outside the experiment list."
    echo "Maximum index: $DEFAULT_LAST_INDEX"
    exit 1
fi
if (( LAST_INDEX >= TOTAL_EXPERIMENTS )); then
    echo "ERROR: LAST_INDEX is outside the experiment list."
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

RETENTION_INDEX=$((CURRENT_INDEX / MODEL_COUNT))
MODEL_INDEX=$((CURRENT_INDEX % MODEL_COUNT))
RETENTION_LEVEL="${RETENTION_LEVELS[$RETENTION_INDEX]}"
DATASET_DIR="$DATASET_ROOT/$RETENTION_LEVEL"
MODEL="${MODELS[$MODEL_INDEX]}"
MODEL_SEED="${MODEL_SEEDS[$MODEL_INDEX]}"

if [[ ! -d "$PROJECT_DIR/$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory was not found:"
    echo "$PROJECT_DIR/$DATASET_DIR"
    exit 1
fi
if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/amazon.inter" ]]; then
    echo "ERROR: Dataset interaction file was not found:"
    echo "$PROJECT_DIR/$DATASET_DIR/amazon.inter"
    exit 1
fi
if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/amazon.item" ]]; then
    echo "ERROR: Fixed item catalogue was not found:"
    echo "$PROJECT_DIR/$DATASET_DIR/amazon.item"
    exit 1
fi

ITEM_CATALOGUE_COUNT=$(($(wc -l < "$PROJECT_DIR/$DATASET_DIR/amazon.item") - 1))
if (( ITEM_CATALOGUE_COUNT != 19022 )); then
    echo "ERROR: Amazon item catalogue should contain 19,022 items."
    echo "Found: $ITEM_CATALOGUE_COUNT"
    exit 1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$PROJECT_DIR"
mkdir -p logs
mkdir -p saved
mkdir -p results/raw

echo "============================================================"
echo "AMAZON EARLY SPARSITY SEQUENCE JOB STARTED"
echo "============================================================"
echo "Job ID            : ${SLURM_JOB_ID:-unknown}"
echo "Job name          : ${SLURM_JOB_NAME:-unknown}"
echo "Node              : $(hostname)"
echo "Partition         : ${SLURM_JOB_PARTITION:-unknown}"
echo "Started at        : $(date)"
echo "Project directory : $(pwd)"
echo "Git commit        : $(git rev-parse HEAD)"
echo "Python            : $PYTHON"
echo "Current index     : $CURRENT_INDEX"
echo "Last index        : $LAST_INDEX"
echo "Total experiments : $TOTAL_EXPERIMENTS"
echo "Model             : $MODEL"
echo "Model seed        : $MODEL_SEED"
echo "Retention level   : ${RETENTION_LEVEL}%"
echo "Dataset directory : $DATASET_DIR"
echo "Interaction file  : $PROJECT_DIR/$DATASET_DIR/amazon.inter"
echo "Item catalogue    : $PROJECT_DIR/$DATASET_DIR/amazon.item"
echo "Run type          : final"

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
import scipy
import torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("RecBole:", recbole.__version__)
print("SciPy:", scipy.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: This GPU Slurm job cannot access CUDA.")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "============================================================"
echo "STARTING $MODEL seed $MODEL_SEED"
echo "============================================================"
"$PYTHON" scripts/train_model.py \
    --dataset amazon \
    --model "$MODEL" \
    --dataset-dir "$DATASET_DIR" \
    --run-type final \
    --seed "$MODEL_SEED" \
    --use-gpu

echo "============================================================"
echo "$MODEL seed $MODEL_SEED COMPLETED"
echo "============================================================"
echo "Finished at: $(date)"

if (( CURRENT_INDEX < LAST_INDEX )); then
    NEXT_INDEX=$((CURRENT_INDEX + 1))
    NEXT_RETENTION_INDEX=$((NEXT_INDEX / MODEL_COUNT))
    NEXT_MODEL_INDEX=$((NEXT_INDEX % MODEL_COUNT))
    NEXT_RETENTION_LEVEL="${RETENTION_LEVELS[$NEXT_RETENTION_INDEX]}"
    NEXT_MODEL="${MODELS[$NEXT_MODEL_INDEX]}"
    NEXT_SEED="${MODEL_SEEDS[$NEXT_MODEL_INDEX]}"
    NEXT_JOB_NAME="amazon_${NEXT_MODEL}_early${NEXT_RETENTION_LEVEL}_seed${NEXT_SEED}"
    NEXT_JOB_ID=$(
        sbatch \
            --parsable \
            --job-name="$NEXT_JOB_NAME" \
            "$SEQUENCE_SCRIPT" \
            "$NEXT_INDEX" \
            "$LAST_INDEX"
    )
    echo "Next job submitted successfully."
    echo "Next job ID: $NEXT_JOB_ID"
else
    echo "============================================================"
    echo "AMAZON EARLY SPARSITY SEQUENCE COMPLETED"
    echo "============================================================"
fi
