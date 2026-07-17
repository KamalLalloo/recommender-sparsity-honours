# Recommender System Robustness Under Sparse User Feedback

Evaluating the impact of interaction sparsity on recommendation performance across multiple recommendation paradigms

**Honours Research Project, University of the Witwatersrand**
**Author:** Kamal Lalloo
**Supervisor:** Dr. Hairong Bau
**Timeline:** June 2026 – October 2026

## Overview

Modern recommender systems rely heavily on historical user–item interaction data. In real-world applications, however, user feedback is often sparse, incomplete, or unavailable for long periods of time. This sparsity presents a significant challenge for recommendation quality, particularly for models that depend on rich interaction histories to learn user preferences.

This project investigates how different recommender system architectures behave under progressively sparse user-feedback conditions. Rather than focusing solely on recommendation accuracy, the study evaluates **model robustness** by measuring how gracefully performance degrades as available interaction data is reduced.

The research compares a diverse set of recommendation paradigms, including classical collaborative filtering, linear recommendation models, neural recommenders, sequential recommenders, and graph-based approaches.

## Research Question

How robust are different recommender system models when user interaction data becomes increasingly sparse?

## Objectives

- Evaluate recommendation performance under varying levels of interaction sparsity.
- Compare robustness across different recommendation paradigms.
- Simulate controlled sparse-data and cold-start-like environments.
- Identify which model architectures degrade most gracefully when interaction data becomes limited.
- Provide reproducible experimental benchmarks for sparse recommendation research.

## Datasets

Two publicly available benchmark datasets are used:

### MovieLens-1M

A widely used recommendation benchmark dataset containing approximately one million movie ratings. MovieLens provides a relatively dense interaction environment and serves as a controlled benchmark for sparsity simulation experiments.

### Amazon Video Games 2023

A large-scale real-world e-commerce dataset containing naturally sparse user–item interactions. This dataset provides a realistic recommendation environment with incomplete and noisy user feedback.

## Sparsity Scenarios

The study evaluates recommendation models under three controlled sparsity conditions:

### Global Sparsity

Random removal of interactions from the training data to simulate decreasing interaction density.

### Recent-History Sparsity

Retention of only the most recent interactions within each user's interaction history.

Example:

[A, B, C, D, E, F] → [D, E, F]

### Early-Profile Sparsity

Retention of only the earliest interactions within each user's interaction history.

Example:

[A, B, C, D, E, F] → [A, B, C]

This scenario approximates cold-start-like recommendation environments where only limited user history is available.

## Models

### Baseline

- Pop

### Classical Collaborative Filtering

- ItemKNN
- BPR

### Linear Recommendation

- EASE

### Neural Recommendation

- NeuMF
- MultiVAE

### Sequential Recommendation

- GRU4Rec
- SASRec
- BERT4Rec

### Graph-Based Recommendation

- LightGCN

## Evaluation Metrics

Models are evaluated using standard top-K ranking metrics:

- Recall@K
- HitRate@K
- NDCG@K
- MRR@K

Performance robustness is measured using relative performance degradation between full-data and sparse-data conditions.

## Framework and Tools

- Python 3.11
- PyTorch
- RecBole
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Repository Structure

```text
configs/        Experiment configurations
data/           Raw and processed datasets
docs/           Proposal, literature review, dissertation
models/         Saved models and checkpoints
notebooks/      Data exploration and analysis notebooks
results/        Experimental outputs, tables, and figures
scripts/        Preprocessing and experiment scripts
```

## Proposal

The full research proposal can be found in:

```text
docs/proposal/proposal.pdf
```

## Current Status

Project setup and experimental infrastructure completed.

Upcoming phases include:

- Dataset acquisition and exploration
- Preprocessing pipeline implementation
- Sparsity simulation
- Baseline recommendation experiments
- Full model evaluation
- Robustness analysis
- Dissertation writing
