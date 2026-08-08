"""
Prepare Amazon Video Games Dataset for RecBole

This script converts the deterministically ordered Amazon
Video Games dataset into the unified interaction format
required by RecBole.

The original timestamp is retained for analysis, while
sequence_order is the deterministic ordering field used
for temporal splitting and sequential recommendation.

Pipeline
--------
1. Load the chronological interaction dataset
2. Validate the dataset and sequence_order
3. Rename columns to RecBole format
4. Export amazon.inter
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
    / "amazon"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "amazon"
    / "baseline"
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
    Load the chronologically ordered Amazon interaction dataset.
    """

    filepath = (
        PROCESSED_DATA
        / "03_temporal_interactions.csv"
    )

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Processed Amazon dataset not found:\n{filepath}"
        )

    return pd.read_csv(filepath)


# ==========================================================
# Validation
# ==========================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate the temporal interaction dataset.
    """

    print("\nValidating dataset...")

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if df.isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    if df.duplicated().any():
        raise ValueError(
            "Dataset contains duplicate rows."
        )

    required_columns = [
        "user_id",
        "item_id",
        "timestamp",
        "sequence_order",
    ]

    if list(df.columns) != required_columns:
        raise ValueError(
            "Dataset contains unexpected columns: "
            f"{list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(
        df["timestamp"]
    ):
        raise ValueError(
            "timestamp must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        df["sequence_order"]
    ):
        raise ValueError(
            "sequence_order must be numeric."
        )

    if df.duplicated(
        subset=[
            "user_id",
            "sequence_order",
        ]
    ).any():
        raise ValueError(
            "Dataset contains duplicate "
            "(user_id, sequence_order) pairs."
        )

    if df.duplicated(
        subset=[
            "user_id",
            "item_id",
        ]
    ).any():
        raise ValueError(
            "Dataset contains duplicate "
            "(user_id, item_id) interactions."
        )

    ordered = df.sort_values(
        [
            "user_id",
            "sequence_order",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    expected_sequence = (
        ordered
        .groupby(
            "user_id",
            sort=False,
        )
        .cumcount()
        .add(1)
        .astype("int64")
    )

    actual_sequence = (
        ordered["sequence_order"]
        .astype("int64")
        .reset_index(drop=True)
    )

    if not actual_sequence.equals(
        expected_sequence.reset_index(drop=True)
    ):
        raise ValueError(
            "sequence_order must equal 1..N "
            "for every user."
        )

    timestamp_diff = (
        ordered
        .groupby(
            "user_id",
            sort=False,
        )["timestamp"]
        .diff()
    )

    if (
        timestamp_diff.dropna() < 0
    ).any():
        raise ValueError(
            "At least one user's timestamps "
            "are not chronologically ordered."
        )

    print("Dataset validation passed.")


# ==========================================================
# Convert to RecBole Format
# ==========================================================

def convert_to_recbole(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rename columns to the field names expected by RecBole.
    """

    converted = df.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
            "sequence_order": "sequence_order:float",
        }
    )

    return converted[
        [
            "user_id:token",
            "item_id:token",
            "timestamp:float",
            "sequence_order:float",
        ]
    ]


# ==========================================================
# Export Dataset
# ==========================================================

def export_dataset(
    df: pd.DataFrame,
) -> Path:
    """
    Export the Amazon interaction dataset.
    """

    output_file = (
        OUTPUT_DIR
        / "amazon.inter"
    )

    df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved: {output_file}"
    )

    return output_file


# ==========================================================
# Summary
# ==========================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print dataset statistics.
    """

    print("\nExport Summary")
    print("-" * 40)

    print(
        f"Users        : "
        f"{df['user_id:token'].nunique():,}"
    )

    print(
        f"Items        : "
        f"{df['item_id:token'].nunique():,}"
    )

    print(
        f"Interactions : "
        f"{len(df):,}"
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print("=" * 60)
    print(
        "Preparing Amazon Video Games Dataset for RecBole"
    )
    print("=" * 60)

    dataset = load_dataset()

    validate_dataset(
        dataset
    )

    dataset = convert_to_recbole(
        dataset
    )

    export_dataset(
        dataset
    )

    print_summary(
        dataset
    )

    print(
        "\nAmazon RecBole dataset preparation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()