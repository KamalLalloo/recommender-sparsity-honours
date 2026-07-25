# Recommender System Robustness Under Sparse User Feedback

Evaluating how recommender systems perform as user interaction data becomes increasingly sparse.

**BSc Honours Research Project – University of the Witwatersrand**

- **Author:** Kamal Lalloo
- **Supervisor:** Dr. Hairong Bau
- **Duration:** June 2026 – October 2026

## Overview

This project evaluates the robustness of recommender systems under sparse user feedback. Multiple recommendation paradigms are compared using controlled sparsity simulations to measure how recommendation quality degrades as available interaction data decreases.

## Research Question

> How robust are different recommender system models when user interaction data becomes increasingly sparse?

## Datasets

| Dataset                 | Purpose                                                     |
| ----------------------- | ----------------------------------------------------------- |
| MovieLens-1M            | Dense benchmark dataset for controlled sparsity experiments |
| Amazon Video Games 2023 | Naturally sparse real-world recommendation dataset          |

## Sparsity Scenarios

| Scenario                | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| Global Sparsity         | Randomly removes training interactions                         |
| Recent-History Sparsity | Retains only the most recent user interactions                 |
| Early-Profile Sparsity  | Retains only the earliest interactions (cold-start simulation) |

## Models

| Category   | Models                    |
| ---------- | ------------------------- |
| Baseline   | Pop                       |
| Classical  | ItemKNN, BPR              |
| Linear     | EASE                      |
| Neural     | NeuMF, MultiVAE           |
| Sequential | GRU4Rec, SASRec, BERT4Rec |
| Graph      | LightGCN                  |

## Evaluation

Performance is measured using standard top-K ranking metrics:

- Recall@K
- HitRate@K
- NDCG@K
- MRR@K

Model robustness is evaluated by measuring performance degradation under progressively sparse interaction conditions.

## Repository Structure

```text
configs/      Experiment configurations
data/         Datasets
docs/         Proposal and dissertation
notebooks/    Analysis notebooks
results/      Experimental results
scripts/      Preprocessing and training scripts
```

## Proposal

The research proposal is available at:

```text
docs/proposal/proposal.pdf
```
