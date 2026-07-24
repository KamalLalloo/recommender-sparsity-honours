"""
Generate MovieLens Sparsity Datasets for RecBole

Sparsity is applied only to the training interactions. The original
validation and test interactions are appended unchanged.

Each generated dataset is saved as a single movielens.inter file.
RecBole then performs the chronological leave-one-out split internally.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sparsity import (
    apply_early_profile_sparsity,
    apply_global_sparsity,
    apply_recent_history_sparsity,
)


# ==========================================================
# Project Constants
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens"
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

EXPECTED_COLUMNS = [
    "user_id",
    "movie_id",
    "timestamp",
]

RECBOLE_COLUMN_MAPPING = {
    "user_id": "user_id:token",
    "movie_id": "item_id:token",
    "timestamp": "timestamp:float",
}


# ==========================================================
# File Loading
# ==========================================================

def load_dataset(filename: str) -> pd.DataFrame:
    """Load a processed MovieLens dataset."""

    filepath = PROCESSED_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    dataframe = pd.read_csv(filepath)

    print(
        f"Loaded {filename}: "
        f"{len(dataframe):,} interactions"
    )

    return dataframe


def load_processed_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the processed train, validation and test splits."""

    train = load_dataset("train.csv")
    validation = load_dataset("validation.csv")
    test = load_dataset("test.csv")

    return train, validation, test


# ==========================================================
# Validation
# ==========================================================

def validate_dataset(
    dataframe: pd.DataFrame,
    name: str,
) -> None:
    """Validate one processed dataset."""

    if dataframe.empty:
        raise ValueError(f"{name} dataset is empty.")

    if list(dataframe.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{name} has unexpected columns: "
            f"{list(dataframe.columns)}"
        )

    if dataframe.isnull().any().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    if dataframe.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate rows."
        )


def validate_processed_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate the processed leave-one-out splits."""

    validate_dataset(train, "Train")
    validate_dataset(validation, "Validation")
    validate_dataset(test, "Test")

    train_users = set(train["user_id"])
    validation_users = set(validation["user_id"])
    test_users = set(test["user_id"])

    if not (
        train_users
        == validation_users
        == test_users
    ):
        raise ValueError(
            "Train, validation and test must contain "
            "the same users."
        )

    if validation["user_id"].duplicated().any():
        raise ValueError(
            "Validation must contain one interaction per user."
        )

    if test["user_id"].duplicated().any():
        raise ValueError(
            "Test must contain one interaction per user."
        )

    max_train_times = train.groupby(
        "user_id"
    )["timestamp"].max()

    validation_times = validation.set_index(
        "user_id"
    )["timestamp"]

    test_times = test.set_index(
        "user_id"
    )["timestamp"]

    if not (max_train_times <= validation_times).all():
        raise ValueError(
            "Training interactions must occur before validation."
        )

    if not (validation_times <= test_times).all():
        raise ValueError(
            "Validation interactions must occur before test."
        )

    print("Processed split validation passed.")


def validate_generated_dataset(
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:
    """
    Confirm that validation and test remain the final two
    interactions for every user.
    """

    expected_users = set(validation["user_id"])

    if set(sparse_train["user_id"]) != expected_users:
        raise ValueError(
            "Sparse training data does not contain every user."
        )

    ordered = combined.sort_values(
        ["user_id", "timestamp"],
        kind="mergesort",
    )

    last_two = ordered.groupby(
        "user_id",
        sort=False,
        group_keys=False,
    ).tail(2)

    actual_validation = (
        last_two.groupby(
            "user_id",
            sort=False,
            group_keys=False,
        )
        .head(1)
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    actual_test = (
        last_two.groupby(
            "user_id",
            sort=False,
            group_keys=False,
        )
        .tail(1)
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    expected_validation = (
        validation
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    expected_test = (
        test
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    if not actual_validation.equals(expected_validation):
        raise ValueError(
            "Validation interactions were not preserved."
        )

    if not actual_test.equals(expected_test):
        raise ValueError(
            "Test interactions were not preserved."
        )


# ==========================================================
# Dataset Generation
# ==========================================================

def apply_sparsity(
    train: pd.DataFrame,
    scenario: str,
    retention: float,
) -> pd.DataFrame:
    """Apply sparsity only to the training interactions."""

    sparsity_function = SPARSITY_SCENARIOS[scenario]

    if scenario == "global":
        return sparsity_function(
            interactions=train,
            retention=retention,
            seed=SEED,
        )

    return sparsity_function(
        interactions=train,
        retention=retention,
    )


def combine_splits(
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Combine sparse training data with fixed evaluation data."""

    return (
        pd.concat(
            [sparse_train, validation, test],
            ignore_index=True,
        )
        .sort_values(
            ["user_id", "timestamp"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def convert_to_recbole(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Rename columns to RecBole field names."""

    return dataframe.rename(
        columns=RECBOLE_COLUMN_MAPPING
    )


# ==========================================================
# File Saving
# ==========================================================

def save_dataset(
    output_dir: Path,
    interactions: pd.DataFrame,
) -> None:
    """Save one RecBole dataset."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    interactions.to_csv(
        output_dir / "movielens.inter",
        sep="\t",
        index=False,
    )


def save_metadata(
    output_dir: Path,
    scenario: str,
    retention_level: str,
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:
    """Save information about one generated dataset."""

    metadata = {
        "dataset": "movielens",
        "scenario": scenario,
        "retention": int(retention_level),
        "seed": SEED if scenario == "global" else None,
        "users": int(combined["user_id"].nunique()),
        "items": int(combined["movie_id"].nunique()),
        "training_interactions": int(len(sparse_train)),
        "validation_interactions": int(len(validation)),
        "test_interactions": int(len(test)),
        "total_interactions": int(len(combined)),
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
# Main Generation
# ==========================================================

def generate_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Generate every scenario and retention level."""

    for scenario in SPARSITY_SCENARIOS:

        print("\n" + "=" * 60)
        print(f"{scenario.upper()} SPARSITY")
        print("=" * 60)

        for level, retention in RETENTION_LEVELS.items():

            sparse_train = apply_sparsity(
                train=train,
                scenario=scenario,
                retention=retention,
            )

            combined = combine_splits(
                sparse_train=sparse_train,
                validation=validation,
                test=test,
            )

            validate_generated_dataset(
                sparse_train=sparse_train,
                validation=validation,
                test=test,
                combined=combined,
            )

            output_dir = (
                OUTPUT_ROOT
                / scenario
                / level
            )

            save_dataset(
                output_dir=output_dir,
                interactions=convert_to_recbole(
                    combined
                ),
            )

            save_metadata(
                output_dir=output_dir,
                scenario=scenario,
                retention_level=level,
                sparse_train=sparse_train,
                validation=validation,
                test=test,
                combined=combined,
            )

            print(
                f"{scenario}/{level}%: "
                f"{len(sparse_train):,} training, "
                f"{len(combined):,} total"
            )


def main() -> None:
    """Run the dataset generation pipeline."""

    print("=" * 60)
    print("Generate MovieLens Sparsity Datasets")
    print("=" * 60)

    train, validation, test = load_processed_splits()

    validate_processed_splits(
        train=train,
        validation=validation,
        test=test,
    )

    generate_datasets(
        train=train,
        validation=validation,
        test=test,
    )

    print(
        "\nSparsity datasets generated successfully."
    )


if __name__ == "__main__":
    main()