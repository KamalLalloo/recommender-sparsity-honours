# Recommender System Robustness Under Sparse User Feedback

A controlled evaluation of how recommender-system performance changes as user interaction histories become increasingly sparse.

**BSc Honours Research Project — University of the Witwatersrand**

- **Author:** Kamal Lalloo
- **Supervisor:** Dr. Hairong Bau
- **Project period:** June–October 2026
- **Framework:** RecBole 1.2.1

## Overview

This project evaluates the robustness of recommender systems under limited user feedback.

Ten recommendation models spanning non-personalised, classical, linear, neural, sequential, and graph-based approaches are evaluated on **MovieLens-1M** and **Amazon Video Games 2023**.

The study varies both:

- the **amount** of training feedback retained; and
- the **temporal position** of the retained interactions.

Validation targets, test targets, and the item catalogue remain fixed across sparsity conditions so that changes in performance can be attributed more directly to the information available during training.

The study examines how recommendation quality changes as feedback becomes sparse, whether recent interactions are more informative than equally sized random or early-history subsets, and whether more complex models are consistently more robust than simpler alternatives.

## Datasets

Both datasets are converted to positive implicit feedback by retaining ratings of **4 or higher**.

| Dataset | Users | Items | Interactions |
|---|---:|---:|---:|
| MovieLens-1M | 6,034 | 3,125 | 574,376 |
| Amazon Video Games 2023 | 63,413 | 19,022 | 533,550 |

The datasets provide contrasting interaction-history distributions: MovieLens contains substantially longer user histories, while Amazon remains much sparser after preprocessing.

## Experimental Design

### Preprocessing

The preprocessing pipeline:

1. converts ratings of `>= 4` to positive implicit interactions;
2. deduplicates repeated positive Amazon user-item interactions by retaining the most recent interaction;
3. applies iterative **5-core filtering**;
4. orders interactions chronologically; and
5. performs per-user temporal leave-one-out splitting:
   - final interaction → test
   - second-last interaction → validation
   - all earlier interactions → training

The 5-core constraint is applied only before sparsification. Sparse training conditions are not re-filtered.

### Sparse-feedback conditions

Sparsity is applied only to each user's training history at nominal retention levels of:

- **100%**
- **50%**
- **25%**
- **10%**

For user \(u\) with \(n_u\) training interactions and retention rate \(p\):

```text
retained = max(1, ceil(n_u × p))
```

The minimum-one rule prevents users from disappearing from the sparse training data, so realised retention may exceed the nominal percentage for short histories.

Three sparse-feedback mechanisms are evaluated:

| Condition | Description |
|---|---|
| **Random per-user** | Retains a deterministic random subset from across the user's training history |
| **Recent-history** | Retains the most recent training interactions |
| **Early-profile** | Retains the earliest training interactions |
| **Baseline** | Retains the complete training history |

Random subsets are deterministic and nested across retention levels.

> The repository historically uses the directory name `global` for the random per-user condition. It does **not** perform unrestricted global deletion across the full interaction matrix.

## Models

| Category | Models |
|---|---|
| Non-personalised | Pop |
| Classical | ItemKNN, BPR |
| Linear | EASE |
| Neural | NeuMF, MultiVAE |
| Sequential | GRU4Rec, SASRec, BERT4Rec |
| Graph-based | LightGCN |

## Training and Evaluation

Hyperparameters are tuned separately for each dataset on the **100% baseline condition** using validation **NDCG@10**.

The selected configuration is then frozen and reused across all sparse conditions. This evaluates robustness of the same baseline-selected model configuration rather than retuning each model for every sparsity level.

Evaluation uses **full-sort ranking** over a fixed item catalogue.

Reported metrics include:

- Recall@5/10/20
- HitRate@5/10/20
- NDCG@5/10/20
- MRR@5/10/20

`NDCG@10` is the primary metric and is also used for validation-based checkpoint selection.

Because the temporal leave-one-out protocol contains a single relevant test item per user, Recall@K and HitRate@K are numerically equivalent.

## Robustness Analysis

Sparse-condition performance is compared with each model's full-data baseline using relative change:

```text
Relative change =
    (Sparse score - Baseline score)
    / Baseline score × 100
```

Negative values indicate degradation.

Relative robustness is considered together with absolute recommendation quality, since a weak model can appear stable simply because it has less performance to lose.

## Fixed Item Catalogue

The item catalogue is held fixed across baseline and sparse conditions.

This prevents evaluation results from being confounded by changes in the candidate universe.

Some validation or test items may become unseen in a sparse training condition while remaining valid evaluation candidates. These cases are tracked in the generated sparsity metadata.

## BERT4Rec Compatibility Patch

The project includes a small compatibility patch for **BERT4Rec under RecBole 1.2.1**.

It corrects masked-target validity handling so that a legitimate target at sequence position zero is not incorrectly discarded.

The patch does **not** modify the model architecture, forward pass, masking procedure, prediction logic, or evaluation behaviour.

## Repository Structure

```text
.
├── configs/
│   ├── amazon/
│   ├── movielens/
│   └── tuning/
│
├── data/
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
└── tests/
    ├── test_sparsity.py
    ├── test_catalogues.py
    └── test_bert4rec_patch.py
```

## Experimental Pipeline

```text
Raw datasets
    ↓
Positive implicit-feedback conversion
    ↓
Amazon user-item deduplication
    ↓
Iterative 5-core filtering
    ↓
Chronological ordering
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
Final model training
    ↓
Full-sort evaluation
    ↓
Result processing and robustness analysis
```

## Outputs

The experiment pipeline produces:

- raw experiment results;
- hyperparameter-tuning outputs;
- processed condition summaries;
- relative robustness measures;
- configuration and dataset metadata; and
- baseline, sparsity, and cross-dataset analysis notebooks.

The primary raw result file is:

```text
results/raw/experiment_results.csv
```

## Research Documents

The original research proposal is available at:

```text
docs/proposal/proposal.pdf
```

The final research report is stored under:

```text
docs/dissertation/
```

## Scope

This project evaluates existing recommender systems rather than proposing a new recommendation algorithm.

Its main contribution is a controlled cross-paradigm evaluation of how recommender performance and robustness change when both the quantity and temporal position of available user feedback are systematically varied.
