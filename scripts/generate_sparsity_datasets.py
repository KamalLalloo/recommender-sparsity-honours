"""
Generate RecBole Benchmark Datasets

This script creates benchmark datasets for the recommender
system experiments.

Current functionality
---------------------

1. Load the baseline MovieLens benchmark dataset.
2. Apply Global Sparsity.
3. Generate datasets for multiple retention levels.
4. Save the new RecBole datasets.

Example
-------
python scripts/generate_sparsity_datasets.py
"""

from pathlib import Path

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
    / "recbole"
    / "movielens"
    / "baseline"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole"
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
    Load a RecBole .inter file into a pandas DataFrame.
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


def load_baseline_dataset():
    """
    Load the baseline train, validation and test files.
    """

    train = load_interaction_file(
        BASELINE_DIR / "movielens.train.inter"
    )

    valid = load_interaction_file(
        BASELINE_DIR / "movielens.valid.inter"
    )

    test = load_interaction_file(
        BASELINE_DIR / "movielens.test.inter"
    )

    return train, valid, test


# ==========================================================
# File Saving
# ==========================================================

def save_interaction_file(
    dataframe: pd.DataFrame,
    filepath: Path,
) -> None:
    """
    Save a DataFrame as a RecBole .inter file.
    """

    dataframe.to_csv(
        filepath,
        sep="\t",
        index=False,
    )


def save_dataset(
    output_dir: Path,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Save a complete benchmark dataset.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_interaction_file(
        train,
        output_dir / "movielens.train.inter",
    )

    save_interaction_file(
        valid,
        output_dir / "movielens.valid.inter",
    )

    save_interaction_file(
        test,
        output_dir / "movielens.test.inter",
    )


# ==========================================================
# Summary
# ==========================================================

def print_summary(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Print a simple dataset summary.
    """

    print("\n" + "=" * 60)
    print("Baseline Dataset Summary")
    print("=" * 60)

    print(
        f"Training interactions   : {len(train):,}"
    )

    print(
        f"Validation interactions : {len(valid):,}"
    )

    print(
        f"Test interactions       : {len(test):,}"
    )


# ==========================================================
# Dataset Generation
# ==========================================================

def generate_datasets(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Generate all sparsity datasets.
    """

    for scenario, sparsity_function in SPARSITY_SCENARIOS.items():

        print("\n" + "=" * 60)
        print(f"{scenario.upper()} SPARSITY")
        print("=" * 60)

        for level, retention in RETENTION_LEVELS.items():

            print("\n" + "-" * 60)
            print(f"Retention Level: {level}%")

            if scenario == "global":

                sparse_train = sparsity_function(
                    interactions=train,
                    retention=retention,
                    seed=SEED,
                )

            else:

                sparse_train = sparsity_function(
                    interactions=train,
                    retention=retention,
                )

            output_dir = (
                OUTPUT_ROOT
                / scenario
                / level
            )

            save_dataset(
                output_dir=output_dir,
                train=sparse_train,
                valid=valid,
                test=test,
            )

            print(f"Saved to: {output_dir}")
            print(
                f"Training interactions: {len(sparse_train):,}"
            )


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print("=" * 60)
    print("Generate Sparsity Datasets")
    print("=" * 60)

    train, valid, test = load_baseline_dataset()

    #print("\nTraining columns:")
    #print(train.columns.tolist())

    print_summary(
        train,
        valid,
        test,
    )

    generate_datasets(
        train,
        valid,
        test,
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()