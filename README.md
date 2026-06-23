# Recommender System Robustness Under Sparse User Feedback

## Overview

This honours research project investigates how different recommender system architectures perform under sparse user-feedback conditions.

The study evaluates recommendation robustness by progressively reducing available interaction data and measuring performance degradation across multiple recommendation paradigms.

## Research Question

How robust are different recommender system models when user interaction data becomes increasingly sparse?

## Datasets

- MovieLens-1M
- Amazon Video Games 2023

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

- Recall@K
- HitRate@K
- NDCG@K
- MRR@K

## Framework

- Python 3.11
- PyTorch
- RecBole

## Author

Kamal Lalloo

Honours Research Project
