"""
Prepare MovieLens-1M Dataset for RecBole

This script converts the processed MovieLens datasets into the
interaction file format required by RecBole.

Pipeline
--------
1. Load processed train/validation/test datasets
2. Validate datasets
3. Rename columns to RecBole format
4. Export .inter files
5. Print export summary
"""

from pathlib import Path
import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens"
)

RECBole_DATA = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "movielens"
)

RECBole_DATA.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load Datasets
# ==========================================================

def load_dataset(filename: str) -> pd.DataFrame:
    """Load a processed dataset."""

    return pd.read_csv(PROCESSED_DATA / filename)


# ==========================================================
# Validation
# ==========================================================

def validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Validate a processed dataset."""

    print(f"\nValidating {name} dataset...")

    assert not df.empty, f"{name} dataset is empty."

    assert df.isnull().sum().sum() == 0, (
        f"{name} contains missing values."
    )

    assert df.duplicated().sum() == 0, (
        f"{name} contains duplicate rows."
    )

    required_columns = [
        "user_id",
        "movie_id",
        "timestamp"
    ]

    assert list(df.columns) == required_columns, (
        f"{name} has unexpected columns."
    )

    print(f"{name} dataset validation passed.")


# ==========================================================
# Convert to RecBole Format
# ==========================================================

def convert_to_recbole(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to RecBole field names.
    """

    return df.rename(columns={

        "user_id": "user_id:token",

        "movie_id": "item_id:token",

        "timestamp": "timestamp:float"

    })


# ==========================================================
# Export Dataset
# ==========================================================

def export_inter_file(
    df: pd.DataFrame,
    filename: str
) -> None:
    """
    Export a RecBole interaction file.
    """

    output = RECBole_DATA / filename

    df.to_csv(
        output,
        sep="\t",
        index=False
    )

    print(f"Saved: {output}")


# ==========================================================
# Summary
# ==========================================================

def print_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame
) -> None:

    print("\nExport Summary")
    print("-" * 40)

    print(f"Train rows      : {len(train):,}")

    print(f"Validation rows : {len(validation):,}")

    print(f"Test rows       : {len(test):,}")

    print()

    total = (
        len(train)
        + len(validation)
        + len(test)
    )

    print(f"Total interactions: {total:,}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Preparing MovieLens Dataset for RecBole")
    print("=" * 60)

    train = load_dataset("train.csv")

    validation = load_dataset("validation.csv")

    test = load_dataset("test.csv")

    validate_dataset(train, "Train")

    validate_dataset(validation, "Validation")

    validate_dataset(test, "Test")

    train = convert_to_recbole(train)

    validation = convert_to_recbole(validation)

    test = convert_to_recbole(test)

    export_inter_file(
        train,
        "movielens.train.inter"
    )

    export_inter_file(
        validation,
        "movielens.valid.inter"
    )

    export_inter_file(
        test,
        "movielens.test.inter"
    )

    print_summary(
        train,
        validation,
        test
    )

    print("\nRecBole dataset preparation completed successfully.")


if __name__ == "__main__":
    main()