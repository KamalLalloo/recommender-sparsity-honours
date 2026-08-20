"""
Extract the best baseline hyperparameter-tuning result for each
dataset/model combination.

Selection is based exclusively on validation NDCG@10.

Amazon LightGCN is intentionally excluded because its tuning run
has not completed yet.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_ROOT = PROJECT_ROOT / "results" / "tuning"
OUTPUT_FILE = TUNING_ROOT / "best_configs.csv"

DATASETS = [
    "movielens",
    "amazon",
]

MODELS = [
    "ItemKNN",
    "BPR",
    "EASE",
    "NeuMF",
    "MultiVAE",
    "GRU4Rec",
    "SASRec",
    "BERT4Rec",
    "LightGCN",
]

# Explicit temporary exclusion:
# Amazon LightGCN tuning has not completed yet.
EXCLUDED = {
    ("amazon", "LightGCN"),
}


def tuning_filename(model_name: str) -> str:
    """Return the filename produced by tune_model.py."""

    return f"{model_name.lower()}_tuning.csv"


def parse_overrides(value: str) -> dict:
    """Safely parse the stored Python-dict representation."""

    parsed = ast.literal_eval(value)

    if not isinstance(parsed, dict):
        raise ValueError(
            "parameter_overrides must contain a dictionary."
        )

    return parsed


def load_best_result(
    dataset_name: str,
    model_name: str,
) -> dict:
    """Load a tuning CSV and return its best validation row."""

    input_file = (
        TUNING_ROOT
        / dataset_name
        / tuning_filename(model_name)
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Missing tuning result: {input_file}"
        )

    results = pd.read_csv(input_file)

    required_columns = {
        "candidate_index",
        "dataset",
        "model",
        "parameter_overrides",
        "seed",
        "validation_ndcg@10",
        "training_time_seconds",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"{input_file} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError(
            f"Tuning file contains no candidates: {input_file}"
        )

    results["validation_ndcg@10"] = pd.to_numeric(
        results["validation_ndcg@10"],
        errors="raise",
    )

    if results["validation_ndcg@10"].isna().any():
        raise ValueError(
            f"Missing validation NDCG@10 values in {input_file}"
        )

    best_index = results["validation_ndcg@10"].idxmax()
    best_row = results.loc[best_index]

    overrides = parse_overrides(
        best_row["parameter_overrides"]
    )

    output = {
        "dataset": dataset_name,
        "model": model_name,
        "candidate_index": int(
            best_row["candidate_index"]
        ),
        "seed": int(best_row["seed"]),
        "validation_ndcg@10": float(
            best_row["validation_ndcg@10"]
        ),
        "training_time_seconds": float(
            best_row["training_time_seconds"]
        ),
        "parameter_overrides": repr(overrides),
        "parameter_overrides_json": json.dumps(
            overrides,
            sort_keys=True,
        ),
        "source_file": str(
            input_file.relative_to(PROJECT_ROOT)
        ),
    }

    # Also expose individual tuned parameters as columns.
    for parameter_name, parameter_value in overrides.items():
        output[f"param_{parameter_name}"] = parameter_value

    return output


def main() -> None:
    """Extract and save all completed tuning winners."""

    rows = []

    print("=" * 70)
    print("BEST BASELINE TUNING CONFIGURATIONS")
    print("Selection metric: validation NDCG@10")
    print("=" * 70)

    for dataset_name in DATASETS:
        for model_name in MODELS:

            if (dataset_name, model_name) in EXCLUDED:
                print(
                    f"SKIPPING: {dataset_name} / {model_name} "
                    "(tuning incomplete)"
                )
                continue

            best = load_best_result(
                dataset_name,
                model_name,
            )

            rows.append(best)

            print(
                f"{dataset_name:10s} "
                f"{model_name:10s} "
                f"NDCG@10={best['validation_ndcg@10']:.6f} "
                f"{best['parameter_overrides']}"
            )

    if not rows:
        raise RuntimeError(
            "No completed tuning results were found."
        )

    output = pd.DataFrame(rows)

    # Keep output deterministic and easy to inspect.
    dataset_order = {
        "movielens": 0,
        "amazon": 1,
    }

    model_order = {
        model: index
        for index, model in enumerate(MODELS)
    }

    output["_dataset_order"] = output["dataset"].map(
        dataset_order
    )
    output["_model_order"] = output["model"].map(
        model_order
    )

    output = (
        output
        .sort_values(
            ["_dataset_order", "_model_order"]
        )
        .drop(
            columns=[
                "_dataset_order",
                "_model_order",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    expected_count = (
        len(DATASETS) * len(MODELS)
        - len(EXCLUDED)
    )

    if len(output) != expected_count:
        raise RuntimeError(
            "Unexpected number of selected tuning results. "
            f"Expected {expected_count}, found {len(output)}."
        )

    print("\n" + "=" * 70)
    print(
        f"Selected configurations: {len(output)}"
    )
    print(
        f"Saved: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()