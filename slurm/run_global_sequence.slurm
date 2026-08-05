#!/bin/bash

#SBATCH --job-name=global_sequence
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.out
#SBATCH --error=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.err

set -euo pipefail


# ==========================================================
# Fixed project settings
# ==========================================================

PROJECT_DIR="/home-mscluster/klalloo/recommender-sparsity-honours"

PYTHON="/home-mscluster/klalloo/.conda/envs/recbole_env/bin/python"

SEQUENCE_SCRIPT="$PROJECT_DIR/slurm/run_global_sequence.slurm"

DATASET_ROOT="data/recbole/movielens/global"


# ==========================================================
# Experiment sequence
# ==========================================================

MODELS=(
    pop
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

RETENTION_LEVELS=(
    50
    25
    10
)

MODEL_COUNT=${#MODELS[@]}
RETENTION_COUNT=${#RETENTION_LEVELS[@]}
TOTAL_EXPERIMENTS=$((MODEL_COUNT * RETENTION_COUNT))
DEFAULT_LAST_INDEX=$((TOTAL_EXPERIMENTS - 1))


# ==========================================================
# Arguments
# ==========================================================

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage:"
    echo "  sbatch [Slurm options] $SEQUENCE_SCRIPT \\"
    echo "      START_INDEX RUN_TYPE [LAST_INDEX]"
    echo
    echo "Examples:"
    echo
    echo "Run a two-experiment development test:"
    echo "  sbatch --job-name=pop_global50 \\"
    echo "      $SEQUENCE_SCRIPT 0 development 1"
    echo
    echo "Run every Global experiment:"
    echo "  sbatch --job-name=pop_global50 \\"
    echo "      $SEQUENCE_SCRIPT 0 final $DEFAULT_LAST_INDEX"
    echo
    echo "Experiment indices range from 0 to $DEFAULT_LAST_INDEX."
    exit 1
fi

CURRENT_INDEX="$1"
RUN_TYPE="$2"
LAST_INDEX="${3:-$DEFAULT_LAST_INDEX}"


# ==========================================================
# Argument validation
# ==========================================================

if ! [[ "$CURRENT_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: START_INDEX must be a non-negative integer."
    echo "Received: $CURRENT_INDEX"
    exit 1
fi

if ! [[ "$LAST_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LAST_INDEX must be a non-negative integer."
    echo "Received: $LAST_INDEX"
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

if [[ "$RUN_TYPE" != "development" &&
      "$RUN_TYPE" != "final" ]]; then
    echo "ERROR: RUN_TYPE must be development or final."
    echo "Received: $RUN_TYPE"
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


# ==========================================================
# Resolve the current experiment
# ==========================================================

# Models change first. After all 10 models have completed,
# the sequence moves to the next retention level.
#
# Indices:
#   0-9   = Global 50%
#   10-19 = Global 25%
#   20-29 = Global 10%

RETENTION_INDEX=$((CURRENT_INDEX / MODEL_COUNT))
MODEL_INDEX=$((CURRENT_INDEX % MODEL_COUNT))

RETENTION_LEVEL="${RETENTION_LEVELS[$RETENTION_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"

DATASET_DIR="$DATASET_ROOT/$RETENTION_LEVEL"

if [[ ! -d "$PROJECT_DIR/$DATASET_DIR" ]]; then
    echo "ERROR: Dataset directory was not found:"
    echo "$PROJECT_DIR/$DATASET_DIR"
    exit 1
fi

if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/movielens.inter" ]]; then
    echo "ERROR: Dataset interaction file was not found:"
    echo "$PROJECT_DIR/$DATASET_DIR/movielens.inter"
    exit 1
fi


# ==========================================================
# Environment setup
# ==========================================================

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p saved
mkdir -p results/raw


# ==========================================================
# Job information
# ==========================================================

echo "============================================================"
echo "GLOBAL SPARSITY SEQUENCE JOB STARTED"
echo "============================================================"
echo "Job ID            : ${SLURM_JOB_ID:-unknown}"
echo "Job name          : ${SLURM_JOB_NAME:-unknown}"
echo "Node              : $(hostname)"
echo "Partition         : ${SLURM_JOB_PARTITION:-unknown}"
echo "Started at        : $(date)"
echo "Project directory : $(pwd)"
echo "Git commit        : $(git rev-parse HEAD)"
echo "Python            : $PYTHON"
echo
echo "Current index     : $CURRENT_INDEX"
echo "Last index        : $LAST_INDEX"
echo "Total experiments : $TOTAL_EXPERIMENTS"
echo "Model index       : $MODEL_INDEX"
echo "Retention index   : $RETENTION_INDEX"
echo "Model             : $MODEL"
echo "Retention level   : ${RETENTION_LEVEL}%"
echo "Dataset directory : $DATASET_DIR"
echo "Run type          : $RUN_TYPE"
echo


# ==========================================================
# Environment verification
# ==========================================================

echo "============================================================"
echo "GPU INFORMATION"
echo "============================================================"

nvidia-smi || true

echo
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
    raise SystemExit(
        "ERROR: This GPU Slurm job cannot access CUDA."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY


# ==========================================================
# Run one experiment
# ==========================================================

echo
echo "============================================================"
echo "STARTING $MODEL - GLOBAL ${RETENTION_LEVEL}%"
echo "============================================================"

"$PYTHON" scripts/train_model.py \
    --model "$MODEL" \
    --dataset-dir "$DATASET_DIR" \
    --run-type "$RUN_TYPE" \
    --use-gpu

echo
echo "============================================================"
echo "$MODEL - GLOBAL ${RETENTION_LEVEL}% COMPLETED"
echo "============================================================"
echo "Finished at: $(date)"


# ==========================================================
# Submit the next experiment only after success
# ==========================================================

if (( CURRENT_INDEX < LAST_INDEX )); then
    NEXT_INDEX=$((CURRENT_INDEX + 1))

    NEXT_RETENTION_INDEX=$((NEXT_INDEX / MODEL_COUNT))
    NEXT_MODEL_INDEX=$((NEXT_INDEX % MODEL_COUNT))

    NEXT_RETENTION_LEVEL="${RETENTION_LEVELS[$NEXT_RETENTION_INDEX]}"
    NEXT_MODEL="${MODELS[$NEXT_MODEL_INDEX]}"

    NEXT_JOB_NAME="${NEXT_MODEL}_global${NEXT_RETENTION_LEVEL}"

    echo
    echo "Submitting next experiment..."
    echo "Next index     : $NEXT_INDEX"
    echo "Next model     : $NEXT_MODEL"
    echo "Next retention : ${NEXT_RETENTION_LEVEL}%"

    NEXT_JOB_ID=$(
        sbatch \
            --parsable \
            --job-name="$NEXT_JOB_NAME" \
            "$SEQUENCE_SCRIPT" \
            "$NEXT_INDEX" \
            "$RUN_TYPE" \
            "$LAST_INDEX"
    )

    echo "Next job submitted successfully."
    echo "Next job ID: $NEXT_JOB_ID"
else
    echo
    echo "============================================================"
    echo "GLOBAL SPARSITY SEQUENCE COMPLETED"
    echo "============================================================"
    echo "Final index     : $CURRENT_INDEX"
    echo "Final model     : $MODEL"
    echo "Final retention : ${RETENTION_LEVEL}%"
    echo "Completed at    : $(date)"
fi
