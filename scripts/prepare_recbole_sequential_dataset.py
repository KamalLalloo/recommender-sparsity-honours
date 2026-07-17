"""
Prepare MovieLens Sequential Dataset for RecBole

This script converts the chronologically ordered MovieLens dataset
into the interaction format required by RecBole's sequential
recommendation models.

Unlike the benchmark datasets used by the general recommenders,
this script exports a single interaction file. RecBole's
SequentialDataset performs the sequence generation internally.

Pipeline
--------
1. Load the chronological interaction dataset
2. Validate the dataset
3. Rename columns to RecBole format
4. Export movielens.inter
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "recbole_sequential"
    / "movielens"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the chronologically ordered interaction dataset.
    """

    return pd.read_csv(
        PROCESSED_DATA / "03_temporal_interactions.csv"
    )


# ==========================================================
# Validation
# ==========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the temporal interaction dataset.
    """

    print("\nValidating dataset...")

    assert not df.empty, (
        "Dataset is empty."
    )

    assert df.isnull().sum().sum() == 0, (
        "Dataset contains missing values."
    )

    assert df.duplicated().sum() == 0, (
        "Dataset contains duplicate rows."
    )

    required_columns = [
        "user_id",
        "movie_id",
        "timestamp",
    ]

    assert list(df.columns) == required_columns, (
        "Dataset contains unexpected columns."
    )

    print("Dataset validation passed.")


# ==========================================================
# Convert to RecBole Format
# ==========================================================

def convert_to_recbole(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to the field names expected by RecBole.
    """

    return df.rename(
        columns={
            "user_id": "user_id:token",
            "movie_id": "item_id:token",
            "timestamp": "timestamp:float",
        }
    )


# ==========================================================
# Export Dataset
# ==========================================================

def export_dataset(df: pd.DataFrame) -> None:
    """
    Export the sequential interaction dataset.
    """

    output_file = OUTPUT_DIR / "movielens.inter"

    df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print(f"\nSaved: {output_file}")


# ==========================================================
# Summary
# ==========================================================

def print_summary(df: pd.DataFrame) -> None:
    """
    Print dataset statistics.
    """

    print("\nExport Summary")
    print("-" * 40)

    print(
        f"Users        : {df['user_id:token'].nunique():,}"
    )

    print(
        f"Items        : {df['item_id:token'].nunique():,}"
    )

    print(
        f"Interactions : {len(df):,}"
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print("=" * 60)
    print("Preparing Sequential MovieLens Dataset for RecBole")
    print("=" * 60)

    dataset = load_dataset()

    validate_dataset(dataset)

    dataset = convert_to_recbole(dataset)

    export_dataset(dataset)

    print_summary(dataset)

    print(
        "\nSequential dataset preparation completed successfully."
    )


if __name__ == "__main__":
    main()