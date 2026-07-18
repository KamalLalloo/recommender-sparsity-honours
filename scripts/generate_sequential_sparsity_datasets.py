"""
Generate Sequential RecBole Sparsity Datasets

This script generates sparsity datasets for sequential
recommendation experiments.

Unlike the benchmark pipeline, sequential recommenders use a
single chronological interaction file (movielens.inter). The
sparsity scenarios are therefore applied directly to this file.

Generated directory structure
-----------------------------

baseline/
    movielens.inter

global/
    100/
        movielens.inter
    50/
        movielens.inter
    ...

recent/
early/

Example
-------
python scripts/generate_sequential_sparsity_datasets.py
"""

from pathlib import Path
import json

import pandas as pd

from sparsity import (
    apply_global_sparsity,
    apply_recent_history_sparsity,
    apply_early_profile_sparsity,
)

# ==========================================================
# Project Constants
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "recbole_sequential"
    / "movielens"
    / "baseline"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole_sequential"
    / "movielens"
)

RETENTION_LEVELS = {
    "100": 1.00,
    "50": 0.50,
    "25": 0.25,
    "10": 0.10,
}

SPARSITY_SCENARIOS = {
    "global": apply_global_sparsity,
    "recent": apply_recent_history_sparsity,
    "early": apply_early_profile_sparsity,
}

SEED = 2025


# ==========================================================
# File Loading
# ==========================================================

def load_interaction_file(filepath: Path) -> pd.DataFrame:
    """
    Load a RecBole interaction file.
    """

    print(f"Loading {filepath.name}...")

    dataframe = pd.read_csv(
        filepath,
        sep="\t",
    )

    print(
        f"Loaded {len(dataframe):,} interactions."
    )

    return dataframe


# ==========================================================
# File Saving
# ==========================================================

def save_interaction_file(
    dataframe: pd.DataFrame,
    filepath: Path,
) -> None:
    """
    Save a RecBole interaction file.
    """

    dataframe.to_csv(
        filepath,
        sep="\t",
        index=False,
    )


def save_dataset(
    output_dir: Path,
    interactions: pd.DataFrame,
) -> None:
    """
    Save one sequential dataset.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_interaction_file(
        interactions,
        output_dir / "movielens.inter",
    )


def save_metadata(
    output_dir: Path,
    interactions: pd.DataFrame,
    scenario: str,
    retention_level: str,
    seed: int,
) -> None:
    """
    Save dataset metadata.
    """

    user_column = next(
        column
        for column in interactions.columns
        if column.startswith("user_id")
    )

    item_column = next(
        column
        for column in interactions.columns
        if column.startswith("item_id")
    )

    metadata = {
        "dataset": "movielens",
        "scenario": scenario,
        "retention": int(retention_level),
        "seed": seed,
        "users": interactions[user_column].nunique(),
        "items": interactions[item_column].nunique(),
        "interactions": len(interactions),
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


# ==========================================================
# Summary
# ==========================================================

def print_summary(
    interactions: pd.DataFrame,
) -> None:
    """
    Print dataset summary.
    """

    print("\n" + "=" * 60)
    print("Baseline Dataset Summary")
    print("=" * 60)

    print(
        f"Interactions : {len(interactions):,}"
    )


# ==========================================================
# Dataset Generation
# ==========================================================

def generate_datasets(
    interactions: pd.DataFrame,
) -> None:
    """
    Generate all sequential sparsity datasets.
    """

    for scenario, sparsity_function in SPARSITY_SCENARIOS.items():

        print("\n" + "=" * 60)
        print(f"{scenario.upper()} SPARSITY")
        print("=" * 60)

        for level, retention in RETENTION_LEVELS.items():

            print("\n" + "-" * 60)
            print(f"Retention Level: {level}%")

            if scenario == "global":

                sparse_interactions = sparsity_function(
                    interactions=interactions,
                    retention=retention,
                    seed=SEED,
                )

            else:

                sparse_interactions = sparsity_function(
                    interactions=interactions,
                    retention=retention,
                )

            output_dir = (
                OUTPUT_ROOT
                / scenario
                / level
            )

            save_dataset(
                output_dir=output_dir,
                interactions=sparse_interactions,
            )

            save_metadata(
                output_dir=output_dir,
                interactions=sparse_interactions,
                scenario=scenario,
                retention_level=level,
                seed=SEED,
            )

            print(f"Saved to: {output_dir}")
            print(
                f"Interactions: {len(sparse_interactions):,}"
            )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Generate Sequential Sparsity Datasets")
    print("=" * 60)

    interactions = load_interaction_file(
        BASELINE_DIR / "movielens.inter"
    )

    print_summary(
        interactions,
    )

    generate_datasets(
        interactions,
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()