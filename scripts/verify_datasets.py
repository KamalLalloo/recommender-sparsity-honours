"""
Verify Generated RecBole Sparsity Datasets

The verifier checks the unified MovieLens RecBole datasets without
regenerating them. It validates file presence, RecBole columns,
interaction ordering, validation/test target preservation, 100%
control identity, and metadata consistency.

Example
-------
python scripts/verify_datasets.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "movielens"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens"
)

BASELINE_FILE = (
    DATA_ROOT
    / "baseline"
    / "movielens.inter"
)

SCENARIOS = [
    "global",
    "recent",
    "early",
]

LEVELS = [
    "100",
    "50",
    "25",
    "10",
]

EXPECTED_COLUMNS = [
    "user_id:token",
    "item_id:token",
    "timestamp:float",
    "sequence_order:float",
]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path: Path) -> None:
    """Fail when a required file is missing."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def load_interactions(path: Path) -> pd.DataFrame:
    """Load and validate the RecBole interaction columns."""

    require_file(path)

    dataframe = pd.read_csv(
        path,
        sep="\t",
    )

    if list(dataframe.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path} has unexpected columns: "
            f"{list(dataframe.columns)}"
        )

    return dataframe


def load_processed_target(filename: str) -> pd.DataFrame:
    """Load validation or test targets in RecBole column format."""

    path = PROCESSED_ROOT / filename
    require_file(path)

    dataframe = pd.read_csv(path)

    expected_columns = [
        "user_id",
        "movie_id",
        "timestamp",
        "sequence_order",
    ]

    if list(dataframe.columns) != expected_columns:
        raise ValueError(
            f"{path} has unexpected columns: "
            f"{list(dataframe.columns)}"
        )

    return dataframe.rename(
        columns={
            "user_id": "user_id:token",
            "movie_id": "item_id:token",
            "timestamp": "timestamp:float",
            "sequence_order": "sequence_order:float",
        }
    )[EXPECTED_COLUMNS]


def split_final_targets(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a unified dataset into train, validation, and test."""

    ordered = dataframe.sort_values(
        [
            "user_id:token",
            "sequence_order:float",
        ],
        kind="mergesort",
    )

    last_two = ordered.groupby(
        "user_id:token",
        sort=False,
        group_keys=False,
    ).tail(2)

    validation = (
        last_two.groupby(
            "user_id:token",
            sort=False,
            group_keys=False,
        )
        .head(1)
        .sort_values("user_id:token")
        .reset_index(drop=True)
    )

    test = (
        last_two.groupby(
            "user_id:token",
            sort=False,
            group_keys=False,
        )
        .tail(1)
        .sort_values("user_id:token")
        .reset_index(drop=True)
    )

    target_keys = set(
        zip(
            validation["user_id:token"],
            validation["sequence_order:float"],
        )
    ) | set(
        zip(
            test["user_id:token"],
            test["sequence_order:float"],
        )
    )

    train_mask = [
        key not in target_keys
        for key in zip(
            dataframe["user_id:token"],
            dataframe["sequence_order:float"],
        )
    ]

    train = dataframe.loc[
        train_mask
    ].reset_index(drop=True)

    return train, validation, test


def assert_frame_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    message: str,
) -> None:
    """Raise a clear error for DataFrame inequality."""

    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(message) from error


def verify_interaction_integrity(
    dataframe: pd.DataFrame,
    expected_users: set,
    expected_validation: pd.DataFrame,
    expected_test: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify rows, users, ordering, and preserved targets."""

    if dataframe.isnull().any().any():
        raise ValueError(
            f"{label} contains missing values."
        )

    if dataframe.duplicated().any():
        raise ValueError(
            f"{label} contains duplicate full rows."
        )

    if dataframe.duplicated(
        subset=[
            "user_id:token",
            "sequence_order:float",
        ]
    ).any():
        raise ValueError(
            f"{label} contains duplicate "
            "(user_id, sequence_order) pairs."
        )

    actual_users = set(dataframe["user_id:token"])

    if actual_users != expected_users:
        raise ValueError(
            f"{label} does not contain the expected user set."
        )

    user_counts = dataframe.groupby(
        "user_id:token"
    ).size()

    if (user_counts < 3).any():
        raise ValueError(
            f"{label} has a user with fewer than three "
            "total interactions."
        )

    for user_id, history in dataframe.groupby(
        "user_id:token"
    ):
        ordered_sequence = history.sort_values(
            "sequence_order:float",
            kind="mergesort",
        )["sequence_order:float"].values

        if not (
            ordered_sequence[:-1] < ordered_sequence[1:]
        ).all():
            raise ValueError(
                f"{label} sequence_order is not strictly "
                f"increasing for user {user_id}."
            )

    train, validation, test = split_final_targets(
        dataframe
    )

    if train.groupby("user_id:token").size().min() < 1:
        raise ValueError(
            f"{label} has a user with no training interaction."
        )

    assert_frame_equal(
        validation,
        expected_validation,
        f"{label} validation targets are not preserved.",
    )

    assert_frame_equal(
        test,
        expected_test,
        f"{label} test targets are not preserved.",
    )

    return train, validation, test


def verify_metadata(
    dataframe: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict,
    label: str,
) -> dict[str, object]:
    """Verify generated metadata against the saved file."""

    expected_total = metadata["total_interactions"]

    if len(dataframe) != expected_total:
        raise ValueError(
            f"{label} total_interactions metadata mismatch "
            f"({len(dataframe):,} != {expected_total:,})."
        )

    checks = {
        "users": dataframe["user_id:token"].nunique(),
        "items": dataframe["item_id:token"].nunique(),
        "training_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "training_items": train["item_id:token"].nunique(),
    }

    for key, actual_value in checks.items():
        if int(metadata[key]) != int(actual_value):
            raise ValueError(
                f"{label} metadata mismatch for {key}: "
                f"{metadata[key]} != {actual_value}."
            )

    original_training = int(
        metadata["original_training_interactions"]
    )
    actual_retention = len(train) / original_training
    metadata_retention = float(
        metadata["actual_training_retention_fraction"]
    )

    if abs(actual_retention - metadata_retention) > 1e-9:
        raise ValueError(
            f"{label} actual retention metadata mismatch."
        )

    train_items = set(train["item_id:token"])
    validation_unseen = ~validation[
        "item_id:token"
    ].isin(train_items)
    test_unseen = ~test["item_id:token"].isin(
        train_items
    )

    unseen_checks = {
        "validation_target_interactions_unseen_in_training":
            int(validation_unseen.sum()),
        "test_target_interactions_unseen_in_training":
            int(test_unseen.sum()),
        "validation_unique_items_unseen_in_training":
            int(
                validation.loc[
                    validation_unseen,
                    "item_id:token",
                ].nunique()
            ),
        "test_unique_items_unseen_in_training":
            int(
                test.loc[
                    test_unseen,
                    "item_id:token",
                ].nunique()
            ),
        "users_with_unseen_validation_target":
            int(
                validation.loc[
                    validation_unseen,
                    "user_id:token",
                ].nunique()
            ),
        "users_with_unseen_test_target":
            int(
                test.loc[
                    test_unseen,
                    "user_id:token",
                ].nunique()
            ),
    }

    for key, actual_value in unseen_checks.items():
        if int(metadata[key]) != actual_value:
            raise ValueError(
                f"{label} metadata mismatch for {key}: "
                f"{metadata[key]} != {actual_value}."
            )

    return {
        "scenario": metadata["scenario"],
        "retention": metadata["retention"],
        "training_interactions": len(train),
        "actual_retention_percent": float(
            metadata["actual_training_retention_percent"]
        ),
        "training_items": train["item_id:token"].nunique(),
        "unseen_validation_targets": int(
            validation_unseen.sum()
        ),
        "unseen_test_targets": int(test_unseen.sum()),
    }


def verify_full_retention_controls(
    baseline: pd.DataFrame,
) -> None:
    """Require byte-level and DataFrame equality for 100% controls."""

    baseline_hash = file_sha256(BASELINE_FILE)

    for scenario in SCENARIOS:
        control_file = (
            DATA_ROOT
            / scenario
            / "100"
            / "movielens.inter"
        )

        control_hash = file_sha256(control_file)

        if control_hash != baseline_hash:
            print("100% control hash mismatch:")
            print(f"  baseline: {baseline_hash}")
            print(f"  {scenario}: {control_hash}")
            raise ValueError(
                f"{scenario}/100 is not byte-identical "
                "to baseline."
            )

        control = load_interactions(control_file)

        try:
            pd.testing.assert_frame_equal(
                control,
                baseline,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"{scenario}/100 DataFrame is not identical "
                "to baseline."
            ) from error


def verify_all() -> None:
    """Run every verification check."""

    expected_validation = load_processed_target(
        "validation.csv"
    )
    expected_test = load_processed_target("test.csv")
    expected_users = set(expected_validation["user_id:token"])

    baseline = load_interactions(BASELINE_FILE)

    verify_interaction_integrity(
        baseline,
        expected_users,
        expected_validation,
        expected_test,
        "baseline",
    )

    summary_rows = []

    for scenario in SCENARIOS:
        for level in LEVELS:
            dataset_dir = DATA_ROOT / scenario / level
            inter_file = dataset_dir / "movielens.inter"
            metadata_file = dataset_dir / "metadata.json"

            require_file(inter_file)
            require_file(metadata_file)

            dataframe = load_interactions(inter_file)

            with open(
                metadata_file,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

            train, validation, test = (
                verify_interaction_integrity(
                    dataframe,
                    expected_users,
                    expected_validation,
                    expected_test,
                    f"{scenario}/{level}",
                )
            )

            summary_rows.append(
                verify_metadata(
                    dataframe,
                    train,
                    validation,
                    test,
                    metadata,
                    f"{scenario}/{level}",
                )
            )

    verify_full_retention_controls(baseline)

    print("\nDataset verification summary")
    print("-" * 80)
    print(
        pd.DataFrame(summary_rows)
        .sort_values(
            ["scenario", "retention"],
            ascending=[True, False],
        )
        .to_string(index=False)
    )

    print("\nAll MovieLens datasets passed verification.")
    print("Baseline and all 100% controls are identical.")


def main() -> None:
    """Entry point."""

    print("=" * 60)
    print("VERIFY GENERATED SPARSITY DATASETS")
    print("=" * 60)

    verify_all()


if __name__ == "__main__":
    main()
