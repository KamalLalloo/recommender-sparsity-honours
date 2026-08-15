#!/bin/bash

#SBATCH --job-name=amazon_bert4rec_patch
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.out
#SBATCH --error=/home-mscluster/%u/recommender-sparsity-honours/logs/%x_%N_%j.err

set -euo pipefail

PROJECT_DIR="/home-mscluster/klalloo/recommender-sparsity-honours"
PYTHON="/home-mscluster/klalloo/.conda/envs/recbole_env/bin/python"
SEQUENCE_SCRIPT="$PROJECT_DIR/slurm/run_amazon_bert4rec_patch_sequence.slurm"
MODEL="bert4rec"
EXPECTED_ITEM_COUNT=19022

# Indices 0-9 = seed 2025, 10-19 = seed 2026, 20-29 = seed 2027.
EXPERIMENT_LABELS=(
    "baseline_seed2025"
    "global50_seed2025"
    "global25_seed2025"
    "global10_seed2025"
    "recent50_seed2025"
    "recent25_seed2025"
    "recent10_seed2025"
    "early50_seed2025"
    "early25_seed2025"
    "early10_seed2025"
    "baseline_seed2026"
    "global50_seed2026"
    "global25_seed2026"
    "global10_seed2026"
    "recent50_seed2026"
    "recent25_seed2026"
    "recent10_seed2026"
    "early50_seed2026"
    "early25_seed2026"
    "early10_seed2026"
    "baseline_seed2027"
    "global50_seed2027"
    "global25_seed2027"
    "global10_seed2027"
    "recent50_seed2027"
    "recent25_seed2027"
    "recent10_seed2027"
    "early50_seed2027"
    "early25_seed2027"
    "early10_seed2027"
)

SCENARIOS=(
    "Baseline"
    "Global"
    "Global"
    "Global"
    "Recent"
    "Recent"
    "Recent"
    "Early"
    "Early"
    "Early"
    "Baseline"
    "Global"
    "Global"
    "Global"
    "Recent"
    "Recent"
    "Recent"
    "Early"
    "Early"
    "Early"
    "Baseline"
    "Global"
    "Global"
    "Global"
    "Recent"
    "Recent"
    "Recent"
    "Early"
    "Early"
    "Early"
)

RETENTION_LEVELS=(
    100
    50
    25
    10
    50
    25
    10
    50
    25
    10
    100
    50
    25
    10
    50
    25
    10
    50
    25
    10
    100
    50
    25
    10
    50
    25
    10
    50
    25
    10
)

DATASET_DIRS=(
    "data/recbole/amazon/baseline"
    "data/recbole/amazon/global/50"
    "data/recbole/amazon/global/25"
    "data/recbole/amazon/global/10"
    "data/recbole/amazon/recent/50"
    "data/recbole/amazon/recent/25"
    "data/recbole/amazon/recent/10"
    "data/recbole/amazon/early/50"
    "data/recbole/amazon/early/25"
    "data/recbole/amazon/early/10"
    "data/recbole/amazon/baseline"
    "data/recbole/amazon/global/50"
    "data/recbole/amazon/global/25"
    "data/recbole/amazon/global/10"
    "data/recbole/amazon/recent/50"
    "data/recbole/amazon/recent/25"
    "data/recbole/amazon/recent/10"
    "data/recbole/amazon/early/50"
    "data/recbole/amazon/early/25"
    "data/recbole/amazon/early/10"
    "data/recbole/amazon/baseline"
    "data/recbole/amazon/global/50"
    "data/recbole/amazon/global/25"
    "data/recbole/amazon/global/10"
    "data/recbole/amazon/recent/50"
    "data/recbole/amazon/recent/25"
    "data/recbole/amazon/recent/10"
    "data/recbole/amazon/early/50"
    "data/recbole/amazon/early/25"
    "data/recbole/amazon/early/10"
)

MODEL_SEEDS=(
    2025
    2025
    2025
    2025
    2025
    2025
    2025
    2025
    2025
    2025
    2026
    2026
    2026
    2026
    2026
    2026
    2026
    2026
    2026
    2026
    2027
    2027
    2027
    2027
    2027
    2027
    2027
    2027
    2027
    2027
)

TOTAL_EXPERIMENTS=${#DATASET_DIRS[@]}
DEFAULT_LAST_INDEX=$((TOTAL_EXPERIMENTS - 1))

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch [Slurm options] $SEQUENCE_SCRIPT START_INDEX [LAST_INDEX]"
    echo "Experiments run with --run-type final."
    echo "Experiment indices range from 0 to $DEFAULT_LAST_INDEX."
    exit 1
fi
CURRENT_INDEX="$1"
LAST_INDEX="${2:-$DEFAULT_LAST_INDEX}"
if ! [[ "$CURRENT_INDEX" =~ ^[0-9]+$ ]]; then echo "ERROR: START_INDEX must be non-negative."; exit 1; fi
if ! [[ "$LAST_INDEX" =~ ^[0-9]+$ ]]; then echo "ERROR: LAST_INDEX must be non-negative."; exit 1; fi
if (( CURRENT_INDEX >= TOTAL_EXPERIMENTS || LAST_INDEX >= TOTAL_EXPERIMENTS || CURRENT_INDEX > LAST_INDEX )); then
    echo "ERROR: Invalid index range. Maximum index: $DEFAULT_LAST_INDEX"
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then echo "ERROR: Python executable was not found:"; echo "$PYTHON"; exit 1; fi
if [[ ! -f "$SEQUENCE_SCRIPT" ]]; then echo "ERROR: Sequence script was not found:"; echo "$SEQUENCE_SCRIPT"; exit 1; fi
if [[ ! -f "$PROJECT_DIR/scripts/bert4rec_patch.py" ]]; then echo "ERROR: BERT4Rec patch file was not found."; exit 1; fi

EXPERIMENT_LABEL="${EXPERIMENT_LABELS[$CURRENT_INDEX]}"
SCENARIO="${SCENARIOS[$CURRENT_INDEX]}"
RETENTION_LEVEL="${RETENTION_LEVELS[$CURRENT_INDEX]}"
DATASET_DIR="${DATASET_DIRS[$CURRENT_INDEX]}"
MODEL_SEED="${MODEL_SEEDS[$CURRENT_INDEX]}"

if [[ ! -d "$PROJECT_DIR/$DATASET_DIR" ]]; then echo "ERROR: Dataset directory was not found:"; echo "$PROJECT_DIR/$DATASET_DIR"; exit 1; fi
if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/amazon.inter" ]]; then echo "ERROR: Amazon interaction file was not found:"; echo "$PROJECT_DIR/$DATASET_DIR/amazon.inter"; exit 1; fi
if [[ ! -f "$PROJECT_DIR/$DATASET_DIR/amazon.item" ]]; then echo "ERROR: Fixed Amazon item catalogue was not found:"; echo "$PROJECT_DIR/$DATASET_DIR/amazon.item"; exit 1; fi
ITEM_CATALOGUE_COUNT=$(($(wc -l < "$PROJECT_DIR/$DATASET_DIR/amazon.item") - 1))
if (( ITEM_CATALOGUE_COUNT != EXPECTED_ITEM_COUNT )); then echo "ERROR: Amazon item catalogue should contain 19,022 items."; echo "Found: $ITEM_CATALOGUE_COUNT"; exit 1; fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$PROJECT_DIR"
mkdir -p logs saved results/raw

echo "============================================================"
echo "AMAZON PATCHED BERT4REC SEQUENCE JOB STARTED"
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
echo "Model             : BERT4Rec"
echo "Model seed        : $MODEL_SEED"
echo "Implementation    : Project-local patched BERT4Rec"
echo "Scenario          : $SCENARIO"
echo "Retention level   : ${RETENTION_LEVEL}%"
echo "Dataset directory : $DATASET_DIR"
echo "Interaction file  : $PROJECT_DIR/$DATASET_DIR/amazon.inter"
echo "Item catalogue    : $PROJECT_DIR/$DATASET_DIR/amazon.item"
echo "Item catalogue n  : $ITEM_CATALOGUE_COUNT"
echo "Run type          : final"

nvidia-smi || true
"$PYTHON" - <<'PY'
import sys
from pathlib import Path
import recbole
import torch
project_root = Path.cwd()
sys.path.insert(0, str(project_root / "scripts"))
from bert4rec_patch import PatchedBERT4Rec
import train_model
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("RecBole:", recbole.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: This GPU Slurm job cannot access CUDA.")
print("GPU:", torch.cuda.get_device_name(0))
print("BERT4Rec patch:", PatchedBERT4Rec.PATCH_DESCRIPTION)
if train_model.resolve_model_class("BERT4Rec") is not PatchedBERT4Rec:
    raise SystemExit("ERROR: BERT4Rec did not resolve to PatchedBERT4Rec.")
PY

"$PYTHON" scripts/train_model.py \
    --dataset amazon \
    --model bert4rec \
    --dataset-dir "$DATASET_DIR" \
    --run-type final \
    --seed "$MODEL_SEED" \
    --use-gpu

if (( CURRENT_INDEX < LAST_INDEX )); then
    NEXT_INDEX=$((CURRENT_INDEX + 1))
    NEXT_LABEL="${EXPERIMENT_LABELS[$NEXT_INDEX]}"
    NEXT_JOB_NAME="amazon_bert4rec_${NEXT_LABEL}_patch"
    NEXT_JOB_ID=$(sbatch --parsable --job-name="$NEXT_JOB_NAME" "$SEQUENCE_SCRIPT" "$NEXT_INDEX" "$LAST_INDEX")
    echo "Next job submitted successfully."
    echo "Next job ID: $NEXT_JOB_ID"
else
    echo "AMAZON PATCHED BERT4REC SEQUENCE COMPLETED"
fi
