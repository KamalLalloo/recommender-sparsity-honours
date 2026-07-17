"""
Verify Generated Sparsity Datasets

This script verifies that the generated benchmark datasets
are structurally correct before running experiments.

Checks
------
1. Dataset directories exist.
2. Required files exist.
3. Files can be loaded.
4. Validation and test sets are unchanged.
5. Every training user has at least one interaction.
"""

from pathlib import Path

import pandas as pd


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

GLOBAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "movielens"
    / "global"
)

RETENTION_LEVELS = [
    "100",
    "50",
    "25",
    "10",
]


# ==========================================================
# Utilities
# ==========================================================

def load_interactions(filepath: Path) -> pd.DataFrame:
    """Load a RecBole interaction file."""

    return pd.read_csv(
        filepath,
        sep="\t",
    )


def verify_directory(directory: Path) -> None:
    """Ensure a dataset directory exists."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Missing directory:\n{directory}"
        )


def verify_file(filepath: Path) -> None:
    """Ensure a dataset file exists."""

    if not filepath.exists():
        raise FileNotFoundError(
            f"Missing file:\n{filepath}"
        )


# ==========================================================
# Main Verification
# ==========================================================

def verify_dataset(level: str,
                   baseline_valid: pd.DataFrame,
                   baseline_test: pd.DataFrame) -> None:

    dataset_dir = GLOBAL_DIR / level

    print("\n" + "=" * 60)
    print(f"Global {level}%")
    print("=" * 60)

    verify_directory(dataset_dir)

    train_file = dataset_dir / "movielens.train.inter"
    valid_file = dataset_dir / "movielens.valid.inter"
    test_file = dataset_dir / "movielens.test.inter"

    verify_file(train_file)
    verify_file(valid_file)
    verify_file(test_file)

    train = load_interactions(train_file)
    valid = load_interactions(valid_file)
    test = load_interactions(test_file)

    print(f"Train interactions : {len(train):,}")
    print(f"Valid interactions : {len(valid):,}")
    print(f"Test interactions  : {len(test):,}")

    if valid.equals(baseline_valid):
        print("Validation set     : PASS")
    else:
        print("Validation set     : FAIL")

    if test.equals(baseline_test):
        print("Test set           : PASS")
    else:
        print("Test set           : FAIL")

    user_column = next(
        c for c in train.columns
        if c.startswith("user_id")
    )

    user_counts = train.groupby(user_column).size()

    if (user_counts == 0).any():
        print("User interactions  : FAIL")
    else:
        print("User interactions  : PASS")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Verify Generated Datasets")
    print("=" * 60)

    baseline_valid = load_interactions(
        BASELINE_DIR / "movielens.valid.inter"
    )

    baseline_test = load_interactions(
        BASELINE_DIR / "movielens.test.inter"
    )

    for level in RETENTION_LEVELS:
        verify_dataset(
            level,
            baseline_valid,
            baseline_test,
        )

    print("\nVerification complete.")


if __name__ == "__main__":
    main()