# Recommender System Robustness Under Sparse User Feedback

A controlled evaluation of how recommender-system performance changes when the quantity and temporal position of available user feedback are systematically reduced.

**BSc Honours Research Project — University of the Witwatersrand**

- **Author:** Kamal Lalloo
- **Supervisor:** Dr. Hairong Bau
- **Project period:** June–November 2026
- **Framework:** RecBole 1.2.1

Research report: _Not All Sparsity Is Equal: A Controlled Evaluation of Recommender Systems under Limited User Feedback_

---

## Overview

This project evaluates the robustness of recommender systems under limited user feedback.

Ten recommendation models spanning non-personalised, classical, linear, neural, sequential, and graph-based approaches are evaluated on MovieLens-1M and Amazon Video Games 2023.

The study varies both:

- the **amount of training feedback retained**; and
- the **temporal position of the retained interactions**.

Validation targets, test targets, and the item catalogue remain fixed across sparsity conditions so that changes in recommendation performance can be attributed more directly to the information available during training.

The study focuses on controlled robustness evaluation rather than proposing a new recommendation algorithm.

---

## Research Questions

The study examines:

1. How does recommendation performance change as the available user feedback decreases?
2. Does performance under sparse user histories depend on the temporal location of the retained feedback?
3. Are more complex recommender approaches consistently more robust than simpler alternatives?

---

## Datasets

Two benchmark datasets are used:

| Dataset                 |  Users |  Items | Interactions |
| ----------------------- | -----: | -----: | -----------: |
| MovieLens-1M            |  6,034 |  3,125 |      574,376 |
| Amazon Video Games 2023 | 63,413 | 19,022 |      533,550 |

MovieLens and Amazon were selected because they provide substantially different interaction-history regimes. MovieLens contains relatively long user histories, while Amazon remains much sparser after preprocessing.

Ratings of **4 or higher** are retained as positive interactions for implicit-feedback modelling.

---

## Experimental Design

### Preprocessing

The preprocessing pipeline:

1. loads and validates the raw dataset;
2. retains ratings of `>= 4` as positive interactions;
3. deduplicates repeated positive Amazon user-item interactions by retaining the most recent positive interaction;
4. applies iterative **5-core filtering**, ensuring that every remaining user and item has at least five interactions before sparsification;
5. orders interactions chronologically, using original source-row order to resolve timestamp ties deterministically; and
6. creates a per-user temporal leave-one-out split:
   - final interaction → test;
   - second-last interaction → validation;
   - all earlier interactions → training.

The 5-core constraint is applied before sparsification. Sparse training conditions are not re-filtered to remain 5-core.

### Sparse-Feedback Conditions

Sparsity is applied only to the training portion of each user's interaction history. Validation and test interactions remain fixed across all conditions.

The experiment uses nominal training-retention levels of:

- **100%**
- **50%**
- **25%**
- **10%**

For a user with `n` training interactions and retention rate `p`:

```text
retained = max(1, ceil(n × p))
```

The minimum-one rule prevents users from disappearing from the sparse training data. Consequently, realised retention can exceed the nominal percentage for users with short histories, particularly in the Amazon dataset.

Three sparse-feedback mechanisms are evaluated:

| Condition           | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| **Random per-user** | Retains a deterministic random subset from across each user's training history |
| **Recent-history**  | Retains the most recent training interactions                                  |
| **Early-profile**   | Retains the earliest training interactions                                     |
| **Baseline**        | Retains the complete training history                                          |

Random per-user subsets are deterministic and nested across retention levels: the 10% subset is contained within the 25% subset, which is contained within the 50% subset.

> **Repository naming note:** the directory name `global` is retained for historical reasons and refers to the **random per-user** condition. It does not perform unrestricted global deletion across the complete interaction matrix.

---

## Fixed Item Catalogue

Each dataset uses the same post-preprocessing item catalogue across the baseline and every sparse condition.

This ensures that performance differences are not caused by changes in the evaluation candidate universe.

Some validation or test items may no longer appear in a sparse training condition while remaining valid evaluation candidates. These cases are retained deliberately and are tracked by the generated sparsity metadata.

---

## Models

Ten recommender systems are evaluated:

| Category                              | Models                    |
| ------------------------------------- | ------------------------- |
| **Non-personalised**                  | Pop                       |
| **Classical collaborative filtering** | ItemKNN, BPR              |
| **Linear**                            | EASE                      |
| **Neural**                            | NeuMF, MultiVAE           |
| **Sequential**                        | GRU4Rec, SASRec, BERT4Rec |
| **Graph-based**                       | LightGCN                  |

The model set is intended to provide a cross-paradigm comparison rather than focus on a single recommender family.

---

## Hyperparameter Tuning

Hyperparameters are tuned separately for MovieLens and Amazon on the **100% full-data baseline only**.

Candidate configurations are selected using validation **NDCG@10**.

The selected dataset-specific configuration is then frozen and reused at 50%, 25%, and 10% retention under all three sparsity mechanisms.

The experiment therefore measures the robustness of a common baseline-selected configuration rather than the best performance obtainable by independently retuning each sparse condition.

Search spaces are defined in:

```text
configs/tuning/search_spaces.yaml
```

Tuning results are written under:

```text
results/tuning/
```

---

## Training

Stochastic models are evaluated using three seeds:

```text
2025
2026
2027
```

The stochastic models are:

- BPR
- NeuMF
- MultiVAE
- GRU4Rec
- SASRec
- BERT4Rec
- LightGCN

Deterministic models are run once per condition:

- Pop
- ItemKNN
- EASE

The final reported values for stochastic models are means across the three seeds.

---

## Evaluation

All models are evaluated using **full-sort ranking** over the fixed item catalogue rather than sampled-negative evaluation.

The same validation and test targets are maintained across the baseline and sparse conditions.

Reported metrics are:

- Recall@5, Recall@10, Recall@20
- HitRate@5, HitRate@10, HitRate@20
- NDCG@5, NDCG@10, NDCG@20
- MRR@5, MRR@10, MRR@20

**NDCG@10** is the primary metric and is also used for validation-based checkpoint selection and hyperparameter tuning.

Because temporal leave-one-out produces one relevant held-out test item per user, Recall@K and HitRate@K are numerically equivalent under this evaluation protocol.

---

## Robustness Analysis

Sparse-condition performance is compared with each model's full-data baseline using relative NDCG change:

```text
Relative change =
    (Sparse score - Baseline score)
    -------------------------------- × 100
              Baseline score
```

Therefore:

- a **negative** value indicates degradation;
- `0%` indicates no change; and
- a **positive** value indicates improvement relative to the full-data baseline.

Relative change is considered together with absolute recommendation quality. A model can appear relatively stable simply because it begins from a low baseline score, so robustness is not interpreted independently of absolute performance.

---

## Statistical Analysis

Statistical comparisons use aggregate model-condition results.

For stochastic models, the three training seeds are averaged before statistical testing so that repeated training runs are not treated as independent observations.

The analysis includes:

- Friedman omnibus tests;
- Kendall's `W` effect sizes;
- Nemenyi post-hoc model comparisons;
- Wilcoxon signed-rank comparisons with Holm correction; and
- Spearman rank correlations for cross-dataset agreement.

The intended level of inference is the aggregate model-condition level rather than individual-user significance.

---

## BERT4Rec Compatibility Patch

The project includes a small compatibility patch for BERT4Rec under RecBole 1.2.1.

The patch corrects masked-target validity handling so that a legitimate masked target at sequence position zero is not incorrectly discarded.

It changes only the target-validity handling. It does **not** modify:

- the BERT4Rec architecture;
- the forward pass;
- the masking procedure;
- prediction logic; or
- evaluation behaviour.

The implementation is located at:

```text
scripts/bert4rec_patch.py
```

---

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
Baseline, sparsity, statistical, and cross-dataset analysis
```

---

## Outputs

The experiment pipeline generates:

- raw experiment results;
- hyperparameter-tuning outputs;
- processed condition summaries;
- relative robustness measures;
- sparsity and reproducibility metadata;
- dataset/configuration hashes where recorded by the experiment scripts; and
- analysis notebooks for baseline, sparsity, and cross-dataset comparisons.

The primary raw experiment result file is generated at:

```text
results/raw/experiment_results.csv
```

Generated research data, raw/processed experiment outputs, model checkpoints, logs, and other large runtime artifacts are excluded from version control where specified by `.gitignore`.

They can be regenerated from the preprocessing, experiment, and analysis scripts provided in the repository.

---

## Analysis Notebooks

The notebooks provide the main exploratory, verification, and results-analysis workflow:

| Notebook                          | Purpose                                                |
| --------------------------------- | ------------------------------------------------------ |
| `01_data_exploration.ipynb`       | Dataset exploration                                    |
| `02_preprocessing_checks.ipynb`   | Verification of preprocessing and dataset construction |
| `03_results_verification.ipynb`   | Experiment-result validation                           |
| `04_baseline_analysis.ipynb`      | Full-data baseline comparisons                         |
| `05_sparsity_analysis.ipynb`      | Sparse-feedback analysis                               |
| `06_cross_dataset_analysis.ipynb` | Cross-dataset comparison                               |

---

## Research Documents

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

## Research Scope

This project evaluates existing recommender systems rather than proposing a new recommendation algorithm.

Its principal contribution is a controlled cross-paradigm evaluation showing that sparse-feedback robustness depends jointly on:

- interaction quantity;
- temporal position;
- model choice; and
- dataset characteristics.

The results demonstrate that sparsity should not be characterised only by how much interaction data remains. Which parts of a user's history remain available can materially change the behaviour and ranking of recommender systems.
