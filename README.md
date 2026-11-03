# Recommender System Robustness Under Sparse User Feedback

Evaluating how recommender-system performance changes as user interaction histories become increasingly sparse.

**BSc Honours Research Project — University of the Witwatersrand**

- **Author:** Kamal Lalloo
- **Supervisor:** Dr. Hairong Bau
- **Project period:** June 2026 – October 2026
- **Framework:** RecBole

## Overview

This project evaluates the robustness of recommender systems under limited user feedback.

Ten recommendation models from classical, linear, neural, sequential, and graph-based paradigms are evaluated on two benchmark datasets under controlled reductions in training interaction data.

The experimental design keeps validation targets, test targets, and the item candidate catalogue fixed while progressively reducing only the training histories available to each model. This allows performance degradation to be compared consistently across sparsity conditions.

- the **amount** of training feedback retained; and
- the **temporal position** of the retained interactions.

The study investigates:

1. How does recommender-system performance change as user interaction data becomes progressively sparse?
2. How do different recommendation model families perform when users have limited interaction histories?
3. Do modern neural, sequential, and graph-based recommenders consistently outperform simpler classical approaches under sparse-feedback conditions?

## Datasets

| Dataset                     | Purpose                                                             |
| --------------------------- | ------------------------------------------------------------------- |
| **MovieLens-1M**            | Relatively dense benchmark used for controlled sparsity experiments |
| **Amazon Video Games 2023** | Naturally sparse real-world e-commerce recommendation dataset       |

Both datasets are converted to positive implicit-feedback data using ratings of **4 or higher**.

After preprocessing and iterative 5-core filtering:

| Dataset                 |  Users |  Items | Interactions |
| ----------------------- | -----: | -----: | -----------: |
| MovieLens-1M            |  6,034 |  3,125 |      574,376 |
| Amazon Video Games 2023 | 63,413 | 19,022 |      533,550 |

## Preprocessing

The preprocessing pipeline performs the following steps:

1. Load and validate the raw dataset.
2. Convert ratings of `>= 4` into positive implicit interactions.
3. Deduplicate repeated positive Amazon user-item interactions by retaining the most recent positive interaction.
4. Apply iterative **5-core filtering** so every remaining user and item has at least five interactions before sparsity simulation.
5. Sort interactions chronologically and create a deterministic `sequence_order`.
6. Create chronological leave-one-out targets:
   - final interaction → test
   - second-last interaction → validation
   - all earlier interactions → training

The 5-core constraint is applied to the full preprocessed dataset only. Sparse training conditions are not re-filtered to remain 5-core.

## Sparsity Scenarios

Sparsity is applied only to the training interactions. Validation and test interactions remain unchanged across all conditions.

Experiments use requested retention levels of:

- **100%**
- **50%**
- **25%**
- **10%**

For each user, the retained interaction count is calculated as:

```text
max(1, ceil(training_history_length × retention))
```

Because of integer rounding and the minimum-one rule, realised training retention may be slightly higher than the nominal retention level.

| Scenario                     | Description                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| **Baseline**                 | Retains the complete training history                                                           |
| **Random Per-User Sparsity** | Retains a deterministic random subset of each user's training interactions                      |
| **Recent-History Sparsity**  | Retains only the most recent training interactions for each user                                |
| **Early-Profile Sparsity**   | Retains only the earliest training interactions, representing a cold-start-like limited profile |

The repository historically uses the directory name `global` for the random per-user condition. It does **not** perform unrestricted global deletion across the complete interaction matrix.

Random per-user subsets are deterministic and nested across retention levels.

## Models

Ten recommender systems are evaluated:

| Category                              | Models                    |
| ------------------------------------- | ------------------------- |
| **Baseline**                          | Pop                       |
| **Classical Collaborative Filtering** | ItemKNN, BPR              |
| **Linear**                            | EASE                      |
| **Neural**                            | NeuMF, MultiVAE           |
| **Sequential**                        | GRU4Rec, SASRec, BERT4Rec |
| **Graph-Based**                       | LightGCN                  |

## Hyperparameter Tuning

Model hyperparameters are tuned separately for each dataset using a constrained grid search on the **100% baseline condition only**.

Candidate configurations are compared using validation **NDCG@10**.

The selected baseline configuration is then held fixed across all sparsity conditions for that dataset. This prevents sparsity comparisons from being confounded by separately retuning models at each retention level.

## Evaluation

Evaluation uses chronological leave-one-out splitting and **full-sort ranking** over a fixed item catalogue.

The same validation and test targets are preserved across baseline and sparse conditions.

Performance is measured using:

- Recall@5/10/20
- HitRate@5/10/20
- NDCG@5/10/20
- MRR@5/10/20

`NDCG@10` is used for validation-based checkpoint selection and hyperparameter tuning.

Under the single-target leave-one-out protocol, Recall@K and HitRate@K are equivalent success indicators, while NDCG@K and MRR@K additionally capture the rank position of the held-out item.

## Robustness Analysis

Model robustness is measured using performance degradation relative to the full-data baseline:

```text
Relative Drop =
    (Baseline Score - Sparse Score)
    / Baseline Score
```

A smaller relative drop indicates that a model retains more of its baseline performance under reduced feedback.

Relative degradation is analysed together with absolute metric values to avoid favouring models that are stable only because they begin from a low baseline score.

## Fixed Item Catalogue

Each dataset uses a fixed item catalogue across the baseline and all sparsity conditions.

This ensures that changes in model performance are caused by reduced training information rather than changes in the evaluation candidate universe.

Some validation or test items may become unseen in a sparse training condition while remaining valid evaluation candidates. These cases are tracked in the generated sparsity metadata.

## BERT4Rec Patch

The project includes a local compatibility patch for BERT4Rec under RecBole 1.2.1.

The patch corrects cross-entropy target selection so that a legitimate masked target at sequence position zero is not incorrectly discarded.

The patch changes only the masked-target validity logic. The BERT4Rec architecture, forward pass, masking procedure, prediction logic, and evaluation behaviour remain inherited from RecBole.

## Repository Structure

```text
.
├── configs/
│   ├── amazon/
│   ├── movielens/
│   └── tuning/
│
├── data/                         # generated locally; ignored by Git
│   ├── raw/
│   ├── processed/
│   └── recbole/
│
├── docs/
│   ├── proposal/
│   └── dissertation/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_checks.ipynb
│   ├── 03_results_verification.ipynb
│   ├── 04_baseline_analysis.ipynb
│   ├── 05_sparsity_analysis.ipynb
│   └── 06_cross_dataset_analysis.ipynb
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── tuning/
│
├── scripts/
│   ├── preprocess_movielens.py
│   ├── preprocess_amazon.py
│   ├── prepare_recbole_dataset.py
│   ├── prepare_amazon_recbole_dataset.py
│   ├── sparsity.py
│   ├── generate_sparsity_datasets.py
│   ├── generate_amazon_sparsity_datasets.py
│   ├── prepare_movielens_item_catalog.py
│   ├── prepare_amazon_item_catalog.py
│   ├── verify_datasets.py
│   ├── verify_amazon_datasets.py
│   ├── bert4rec_patch.py
│   ├── tune_model.py
│   ├── extract_best_tuning_configs.py
│   ├── train_model.py
│   └── process_results.py
│
├── slurm/                        # cluster experiment scripts
│
├── tests/
│   ├── test_sparsity.py
│   ├── test_catalogues.py
│   └── test_bert4rec_patch.py
│
├── requirements.txt
└── README.md
```

---

## Reproducing the Experiments

### 1. Clone the Repository

```bash
git clone https://github.com/KamalLalloo/recommender-sparsity-honours.git
cd recommender-sparsity-honours
```

### 2. Create a Python Environment

For example:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the recorded environment:

```bash
pip install -r requirements.txt
```

### 3. Obtain the Raw Datasets

Raw research datasets are not committed to Git.

Place the MovieLens-1M ratings file at:

```text
data/raw/movielens/ratings.dat
```

Place the Amazon Reviews 2023 Video Games review file at:

```text
data/raw/amazon/Video_Games.jsonl
```

The preprocessing scripts expect these paths.

### 4. Preprocess the Datasets

From the repository root:

```bash
python scripts/preprocess_movielens.py
python scripts/preprocess_amazon.py
```

These scripts perform implicit-feedback conversion, filtering, chronological ordering, and temporal leave-one-out preparation.

### 5. Prepare the Full-Data RecBole Datasets

```bash
python scripts/prepare_recbole_dataset.py
python scripts/prepare_amazon_recbole_dataset.py
```

### 6. Generate Sparse Conditions

```bash
python scripts/generate_sparsity_datasets.py
python scripts/generate_amazon_sparsity_datasets.py
```

This generates the random per-user, recent-history, and early-profile conditions at the configured retention levels.

### 7. Create the Fixed Item Catalogues

Run these after the baseline and sparse RecBole datasets have been generated:

```bash
python scripts/prepare_movielens_item_catalog.py
python scripts/prepare_amazon_item_catalog.py
```

The same `.item` catalogue is written to every condition for a dataset.

### 8. Verify Generated Datasets

```bash
python scripts/verify_datasets.py
python scripts/verify_amazon_datasets.py
```

These checks verify properties including:

- fixed validation/test targets;
- expected dataset structure;
- valid item catalogues;
- sparse-training subset relationships; and
- dataset consistency.

---

## Running the Tests

The repository includes tests for sparsification, catalogue handling, and the BERT4Rec compatibility patch.

Run:

```bash
pytest tests/
```

---

## Example Hyperparameter-Tuning Run

Tuning is restricted to the full-data baseline.

For example, to tune SASRec on MovieLens:

```bash
python scripts/tune_model.py \
  --dataset movielens \
  --model sasrec \
  --dataset-dir data/recbole/movielens/baseline \
  --seed 2025 \
  --use-gpu
```

For CPU execution, omit:

```text
--use-gpu
```

Tuning results are written under:

```text
results/tuning/
```

---

## Example Training Run

A model can be trained and evaluated using `scripts/train_model.py`.

For example, SASRec on the MovieLens full-data baseline:

```bash
python scripts/train_model.py \
  --dataset movielens \
  --model sasrec \
  --dataset-dir data/recbole/movielens/baseline \
  --run-type final \
  --seed 2025 \
  --use-gpu
```

An example sparse run using the historical `global` directory name for random per-user retention:

```bash
python scripts/train_model.py \
  --dataset movielens \
  --model sasrec \
  --dataset-dir data/recbole/movielens/global/50 \
  --run-type final \
  --seed 2025 \
  --use-gpu
```

Supported dataset names are:

```text
movielens
amazon
```

Supported models are:

```text
Pop
ItemKNN
BPR
EASE
NeuMF
MultiVAE
GRU4Rec
SASRec
BERT4Rec
LightGCN
```

Model names are accepted case-insensitively by the training script.

---

## Experimental Pipeline

```text
Raw datasets
    ↓
Positive-interaction conversion
    ↓
Amazon user-item deduplication
    ↓
Iterative 5-core filtering
    ↓
Deterministic chronological ordering
    ↓
Temporal validation/test split
    ↓
Training-history sparsification
    ├── Random per-user
    ├── Recent-history
    └── Early-profile
    ↓
Fixed item catalogue
    ↓
RecBole dataset preparation
    ↓
Baseline hyperparameter tuning
    ↓
Frozen dataset-specific configurations
    ↓
Final model training
    ↓
Full-sort evaluation
    ↓
Result processing
    ↓
Baseline, sparsity, and cross-dataset analysis
```

---

## Outputs

The experiment pipeline generates:

- raw experiment results in `results/raw/experiment_results.csv`
- tuning results for each dataset/model
- processed condition summaries
- relative robustness/degradation measures
- reproducibility metadata including configuration and dataset hashes
- baseline, sparsity, and cross-dataset analysis notebooks

## Proposal

The original research proposal is stored under:

```text
docs/proposal/
```

The final research-report materials are stored under:

```text
docs/dissertation/
```

---

## Data and Generated Artifacts

The repository intentionally does not version the complete research datasets or all generated experiment artifacts.

In particular, local/generated paths include:

```text
data/raw/
data/processed/
data/recbole/
results/raw/
results/processed/
models/checkpoints/
models/saved/
```

This keeps the repository focused on the code, configurations, tests, analysis workflow, and research documentation needed to reproduce the experiments.

---

The final research report is stored under:

```text
docs/dissertation/
```

## Scope

This project evaluates existing recommender systems rather than proposing a new recommendation algorithm.

Its main contribution is a controlled empirical comparison of how different recommendation paradigms degrade as user feedback becomes progressively limited.
