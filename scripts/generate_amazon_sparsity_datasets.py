"""
Generate Amazon Video Games Sparsity Datasets for RecBole

Sparsity is applied only to the training interactions.
The original validation and test interactions are appended
unchanged.

Each generated dataset is saved as a single amazon.inter file.
RecBole then performs the chronological leave-one-out split
internally using sequence_order.
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
    / "amazon"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "amazon"
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

SPARSITY_SEED = 2025

EXPECTED_COLUMNS = [
    "user_id",
    "item_id",
    "timestamp",
    "sequence_order",
]

RECBOLE_COLUMN_MAPPING = {
    "user_id": "user_id:token",
    "item_id": "item_id:token",
    "timestamp": "timestamp:float",
    "sequence_order": "sequence_order:float",
}


# ==========================================================
# File Loading
# ==========================================================

def load_dataset(
    filename: str,
) -> pd.DataFrame:

    filepath = (
        PROCESSED_DATA_DIR
        / filename
    )

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    dataframe = pd.read_csv(
        filepath
    )

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

    train = load_dataset(
        "train.csv"
    )

    validation = load_dataset(
        "validation.csv"
    )

    test = load_dataset(
        "test.csv"
    )

    return (
        train,
        validation,
        test,
    )


# ==========================================================
# Dataset Validation
# ==========================================================

def validate_dataset(
    dataframe: pd.DataFrame,
    name: str,
) -> None:

    if dataframe.empty:
        raise ValueError(
            f"{name} dataset is empty."
        )

    if list(
        dataframe.columns
    ) != EXPECTED_COLUMNS:
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

    if not pd.api.types.is_numeric_dtype(
        dataframe["sequence_order"]
    ):
        raise ValueError(
            f"{name} sequence_order must be numeric."
        )

    if dataframe.duplicated(
        subset=[
            "user_id",
            "sequence_order",
        ]
    ).any():
        raise ValueError(
            f"{name} contains duplicate "
            "(user_id, sequence_order) pairs."
        )

    if dataframe.duplicated(
        subset=[
            "user_id",
            "item_id",
        ]
    ).any():
        raise ValueError(
            f"{name} contains duplicate "
            "(user_id, item_id) pairs."
        )


def validate_processed_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    validate_dataset(
        train,
        "Train",
    )

    validate_dataset(
        validation,
        "Validation",
    )

    validate_dataset(
        test,
        "Test",
    )

    train_users = set(
        train["user_id"]
    )

    validation_users = set(
        validation["user_id"]
    )

    test_users = set(
        test["user_id"]
    )

    if not (
        train_users
        == validation_users
        == test_users
    ):
        raise ValueError(
            "Train, validation and test "
            "must contain the same users."
        )

    if validation[
        "user_id"
    ].duplicated().any():
        raise ValueError(
            "Validation must contain "
            "one interaction per user."
        )

    if test[
        "user_id"
    ].duplicated().any():
        raise ValueError(
            "Test must contain "
            "one interaction per user."
        )

    max_train_sequence = (
        train
        .groupby("user_id")[
            "sequence_order"
        ]
        .max()
    )

    validation_sequence = (
        validation
        .set_index("user_id")[
            "sequence_order"
        ]
    )

    test_sequence = (
        test
        .set_index("user_id")[
            "sequence_order"
        ]
    )

    if not (
        max_train_sequence
        < validation_sequence
    ).all():
        raise ValueError(
            "Training sequence_order "
            "must be less than validation."
        )

    if not (
        validation_sequence
        < test_sequence
    ).all():
        raise ValueError(
            "Validation sequence_order "
            "must be less than test."
        )

    print(
        "Processed split validation passed."
    )


# ==========================================================
# Generated Dataset Validation
# ==========================================================

def validate_generated_dataset(
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:

    expected_users = set(
        validation["user_id"]
    )

    if set(
        sparse_train["user_id"]
    ) != expected_users:
        raise ValueError(
            "Sparse training data does "
            "not contain every user."
        )

    ordered = combined.sort_values(
        [
            "user_id",
            "sequence_order",
        ],
        kind="mergesort",
    )

    last_two = (
        ordered
        .groupby(
            "user_id",
            sort=False,
            group_keys=False,
        )
        .tail(2)
    )

    actual_validation = (
        last_two
        .groupby(
            "user_id",
            sort=False,
            group_keys=False,
        )
        .head(1)
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    actual_test = (
        last_two
        .groupby(
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

    if not actual_validation.equals(
        expected_validation
    ):
        raise ValueError(
            "Validation interactions "
            "were not preserved."
        )

    if not actual_test.equals(
        expected_test
    ):
        raise ValueError(
            "Test interactions "
            "were not preserved."
        )


# ==========================================================
# Sparsity
# ==========================================================

def apply_sparsity(
    train: pd.DataFrame,
    scenario: str,
    retention: float,
) -> pd.DataFrame:

    sparsity_function = (
        SPARSITY_SCENARIOS[
            scenario
        ]
    )

    if scenario == "global":
        return sparsity_function(
            interactions=train,
            retention=retention,
            seed=SPARSITY_SEED,
        )

    return sparsity_function(
        interactions=train,
        retention=retention,
    )


# ==========================================================
# Split Combination
# ==========================================================

def combine_splits(
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    return (
        pd.concat(
            [
                sparse_train,
                validation,
                test,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "user_id",
                "sequence_order",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ==========================================================
# RecBole Conversion
# ==========================================================

def convert_to_recbole(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    return (
        dataframe
        .rename(
            columns=RECBOLE_COLUMN_MAPPING
        )
        [
            [
                "user_id:token",
                "item_id:token",
                "timestamp:float",
                "sequence_order:float",
            ]
        ]
    )


# ==========================================================
# 100% Identity Validation
# ==========================================================

def validate_full_retention_identity(
    scenario: str,
    train: pd.DataFrame,
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:

    try:
        pd.testing.assert_frame_equal(
            sparse_train.reset_index(
                drop=True
            ),
            train.reset_index(
                drop=True
            ),
            check_dtype=True,
            check_like=False,
        )

    except AssertionError as error:

        raise ValueError(
            f"{scenario}/100% sparse "
            "training data is not identical "
            "to the original training split."
        ) from error

    expected_combined = (
        pd.concat(
            [
                train,
                validation,
                test,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "user_id",
                "sequence_order",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    try:
        pd.testing.assert_frame_equal(
            combined.reset_index(
                drop=True
            ),
            expected_combined,
            check_dtype=True,
            check_like=False,
        )

    except AssertionError as error:

        raise ValueError(
            f"{scenario}/100% combined "
            "dataset is not identical "
            "to the expected baseline."
        ) from error


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(
    output_dir: Path,
    interactions: pd.DataFrame,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    interactions.to_csv(
        output_dir
        / "amazon.inter",
        sep="\t",
        index=False,
    )


# ==========================================================
# Metadata
# ==========================================================

def save_metadata(
    output_dir: Path,
    scenario: str,
    retention_level: str,
    original_train: pd.DataFrame,
    sparse_train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:

    sparse_items = set(
        sparse_train["item_id"]
    )

    validation_unseen = (
        ~validation[
            "item_id"
        ].isin(sparse_items)
    )

    test_unseen = (
        ~test[
            "item_id"
        ].isin(sparse_items)
    )

    evaluation_user_count = (
        validation[
            "user_id"
        ].nunique()
    )

    actual_retention_fraction = (
        len(sparse_train)
        / len(original_train)
    )

    scenario_methods = {
        "global": "random_per_user",
        "recent": "recent_history",
        "early": "early_profile",
    }

    validation_unseen_count = int(
        validation_unseen.sum()
    )

    test_unseen_count = int(
        test_unseen.sum()
    )

    metadata = {

        "dataset": "amazon",

        "scenario": scenario,

        "scenario_method":
            scenario_methods[scenario],

        "retention":
            int(retention_level),

        "requested_retention_percent":
            int(retention_level),

        "seed":
            SPARSITY_SEED
            if scenario == "global"
            else None,

        "sparsity_seed":
            SPARSITY_SEED
            if scenario == "global"
            else None,

        "retention_rounding":
            "ceil",

        "minimum_training_interactions_per_user":
            1,

        "random_retention_levels_nested":
            scenario == "global",

        "users":
            int(
                combined[
                    "user_id"
                ].nunique()
            ),

        "items":
            int(
                combined[
                    "item_id"
                ].nunique()
            ),

        "original_training_interactions":
            int(len(original_train)),

        "training_interactions":
            int(len(sparse_train)),

        "actual_training_retention_fraction":
            round(
                actual_retention_fraction,
                12,
            ),

        "actual_training_retention_percent":
            round(
                actual_retention_fraction
                * 100,
                6,
            ),

        "original_training_items":
            int(
                original_train[
                    "item_id"
                ].nunique()
            ),

        "training_items":
            int(
                sparse_train[
                    "item_id"
                ].nunique()
            ),

        "validation_interactions":
            int(len(validation)),

        "test_interactions":
            int(len(test)),

        "total_interactions":
            int(len(combined)),

        "validation_target_interactions_unseen_in_training":
            validation_unseen_count,

        "test_target_interactions_unseen_in_training":
            test_unseen_count,

        "validation_unique_items_unseen_in_training":
            int(
                validation.loc[
                    validation_unseen,
                    "item_id",
                ].nunique()
            ),

        "test_unique_items_unseen_in_training":
            int(
                test.loc[
                    test_unseen,
                    "item_id",
                ].nunique()
            ),

        "users_with_unseen_validation_target":
            int(
                validation.loc[
                    validation_unseen,
                    "user_id",
                ].nunique()
            ),

        "users_with_unseen_test_target":
            int(
                test.loc[
                    test_unseen,
                    "user_id",
                ].nunique()
            ),

        "percentage_users_with_unseen_validation_target":
            round(
                (
                    validation_unseen_count
                    / evaluation_user_count
                    * 100
                ),
                6,
            ),

        "percentage_users_with_unseen_test_target":
            round(
                (
                    test_unseen_count
                    / evaluation_user_count
                    * 100
                ),
                6,
            ),
    }

    with open(
        output_dir
        / "metadata.json",
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

    for scenario in (
        SPARSITY_SCENARIOS
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"{scenario.upper()} SPARSITY"
        )

        print("=" * 60)

        for (
            level,
            retention,
        ) in RETENTION_LEVELS.items():

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

            if retention == 1.0:

                validate_full_retention_identity(
                    scenario=scenario,
                    train=train,
                    sparse_train=sparse_train,
                    validation=validation,
                    test=test,
                    combined=combined,
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
                interactions=(
                    convert_to_recbole(
                        combined
                    )
                ),
            )

            save_metadata(
                output_dir=output_dir,
                scenario=scenario,
                retention_level=level,
                original_train=train,
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


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print("=" * 60)

    print(
        "Generate Amazon Video Games "
        "Sparsity Datasets"
    )

    print("=" * 60)

    (
        train,
        validation,
        test,
    ) = load_processed_splits()

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
        "\nAmazon sparsity datasets "
        "generated successfully."
    )


if __name__ == "__main__":
    main()
