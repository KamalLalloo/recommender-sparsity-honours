# Methodology Changes Needed

Update the dissertation methodology so that it describes the **actual implemented pipeline**, not only the original proposal.

## 1. Preprocessing

- State that ratings **≥ 4** are converted to positive implicit-feedback interactions.
- State that iterative **5-core filtering** is applied to both users and items.
- For Amazon, explain that repeated positive user–item interactions are deduplicated by keeping the **most recent positive interaction**, with source-row order used to break timestamp ties.
- For MovieLens, state that duplicate positive user–item interactions were checked and none were retained.

## 2. Temporal Ordering and Splitting

- Explain that interactions are ordered using `sequence_order`.
- Timestamp is the primary ordering field, with original source-row order used to break timestamp ties.
- State that leave-one-out splitting is used:
  - final interaction → test;
  - second-last interaction → validation;
  - all earlier interactions → training.

- Mention that the same validation and test targets are preserved across every sparsity condition.

## 3. Sparsity Scenarios

Rename/descriptively explain the scenarios as:

- `global` → **random per-user sparsity**
- `recent` → **recent-history retention**
- `early` → **early-profile / cold-start-like sparsity**

Do not describe `global` as uniform removal across the full interaction matrix.

Do not describe `early` as genuine cold start, because users are still known and retain training interactions.

## 4. Retention Rule

State that for each user:

```text
retained interactions = max(1, ceil(training_history × retention))
```

Nominal retention levels are:

- 100%
- 50%
- 25%
- 10%

Also report the actual aggregate retention rates because the per-user ceiling causes them to differ slightly.

Especially for Amazon:

- 50% nominal → 54.71% actual
- 25% nominal → 29.77% actual
- 10% nominal → 18.38% actual

## 5. Random Sparsity Reproducibility

For random per-user sparsity:

- use global sparsity seed `2025`;
- derive deterministic user-specific selections;
- ensure nested subsets:
  - 10% ⊆ 25% ⊆ 50%.

## 6. Fixed Item Catalogue

State that the item catalogue is generated from the full post-preprocessing dataset and kept identical across baseline and all sparse conditions.

This ensures that full-sort evaluation uses the same candidate universe even when training interactions are removed.

## 7. Models

List the ten evaluated models:

- Pop
- ItemKNN
- BPR
- EASE
- NeuMF
- MultiVAE
- GRU4Rec
- SASRec
- BERT4Rec
- LightGCN

Group them by model family where useful.

## 8. Hyperparameter Tuning

State that:

- tuning is performed **only on the full baseline dataset**;
- validation **NDCG@10** selects the best candidate;
- the selected configuration is then frozen and reused across all sparsity conditions;
- tuning is not repeated separately for each sparse condition.

Document the Amazon LightGCN tuning exception if it remains unresolved.

## 9. Training and Seeds

State that:

- stochastic models use seeds **2025, 2026, and 2027**;
- deterministic models use one run;
- RecBole seed initialisation is performed explicitly before dataset/model construction;
- reproducibility is enabled.

## 10. Evaluation

State that RecBole uses:

- temporal leave-one-out evaluation;
- full-sort ranking over the fixed item catalogue;
- user grouping;
- `sequence_order` as the time field.

Metrics:

- Recall@5, @10, @20
- Hit@5, @10, @20
- NDCG@5, @10, @20
- MRR@5, @10, @20

Use **NDCG@10 as the primary metric**.

Because there is one held-out relevant item per user, Recall@K and Hit@K are equivalent in this setup.

## 11. Robustness Measurement

Define relative performance degradation as:

```text
(baseline score - sparse score) / baseline score
```

Report this mainly for:

- NDCG@10
- Recall@10
- MRR@10

Where the value is negative, describe it as an improvement rather than a degradation.

## 12. Sequence-Length Settings

Document the effective sequence limits:

- MovieLens: `MAX_ITEM_LIST_LENGTH = 200`
- Amazon: `MAX_ITEM_LIST_LENGTH = 50`

## 13. BERT4Rec Patch

Disclose that RecBole 1.2.1 BERT4Rec was locally patched so that a valid masked target at sequence position zero is not incorrectly discarded.

State that the patch changes the training-target validity logic only, while architecture and evaluation remain unchanged.

## 14. Final Dataset Statistics

Use the final implemented statistics.

### MovieLens-1M

- 6,034 users
- 3,125 items
- 574,376 interactions
- 562,308 training interactions
- 6,034 validation interactions
- 6,034 test interactions

### Amazon Reviews 2023 — Video Games

- 63,413 users
- 19,022 items
- 533,550 interactions
- 406,724 training interactions
- 63,413 validation interactions
- 63,413 test interactions

Describe Amazon as a **positive implicit-feedback 5-core variant of Amazon Reviews 2023 Video Games**, rather than implying it is the standard preprocessed Amazon 5-core benchmark.

## 15. Reproducibility

Mention that the experiment pipeline records:

- Git commit
- dataset hashes
- item-catalogue hashes
- configuration hashes
- seeds
- software versions
- runtime information
- checkpoint paths
- BERT4Rec patch status

Also state that the final experiment set was verified before analysis.
