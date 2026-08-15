"""
Verify generated MovieLens RecBole datasets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "movielens"
PROCESSED_ITEM_COLUMN = "movie_id"
INTER_FILE = "movielens.inter"
ITEM_FILE = "movielens.item"

DATA_ROOT = PROJECT_ROOT / "data" / "recbole" / DATASET_NAME
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / DATASET_NAME

SCENARIOS = ["global", "recent", "early"]
LEVELS = ["100", "50", "25", "10"]

INTER_COLUMNS = [
    "user_id:token",
    "item_id:token",
    "timestamp:float",
    "sequence_order:float",
]

ITEM_COLUMNS = ["item_id:token"]


def require_file(path: Path) -> None:
    """Require a file to exist."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def file_sha256(path: Path) -> str:
    """Return SHA-256 for a file."""

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def load_interactions(path: Path) -> pd.DataFrame:
    """Load one .inter file."""

    require_file(path)
    dataframe = pd.read_csv(path, sep="\t")

    if list(dataframe.columns) != INTER_COLUMNS:
        raise ValueError(
            f"{path} has unexpected columns: "
            f"{list(dataframe.columns)}"
        )

    return dataframe


def load_items(path: Path) -> pd.DataFrame:
    """Load one .item file."""

    require_file(path)
    dataframe = pd.read_csv(path, sep="\t")

    if list(dataframe.columns) != ITEM_COLUMNS:
        raise ValueError(
            f"{path} has unexpected columns: "
            f"{list(dataframe.columns)}"
        )

    if dataframe["item_id:token"].duplicated().any():
        raise ValueError(
            f"{path} contains duplicate catalogue items."
        )

    return dataframe


def load_processed_target(filename: str) -> pd.DataFrame:
    """Load processed validation/test targets in RecBole format."""

    path = PROCESSED_ROOT / filename
    require_file(path)

    dataframe = pd.read_csv(path)

    return dataframe.rename(
        columns={
            "user_id": "user_id:token",
            PROCESSED_ITEM_COLUMN: "item_id:token",
            "timestamp": "timestamp:float",
            "sequence_order": "sequence_order:float",
        }
    )[INTER_COLUMNS]


def split_unified(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split unified data into reconstructed train/valid/test."""

    ordered = dataframe.sort_values(
        ["user_id:token", "sequence_order:float"],
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

    train = dataframe.loc[
        [
            key not in target_keys
            for key in zip(
                dataframe["user_id:token"],
                dataframe["sequence_order:float"],
            )
        ]
    ].reset_index(drop=True)

    return train, validation, test


def row_identity(dataframe: pd.DataFrame) -> set[tuple]:
    """Return full-row identity tuples."""

    return set(
        map(
            tuple,
            dataframe[INTER_COLUMNS].itertuples(
                index=False,
                name=None,
            ),
        )
    )


def verify_interactions(
    dataframe: pd.DataFrame,
    catalogue: pd.DataFrame,
    expected_users: set,
    expected_validation: pd.DataFrame,
    expected_test: pd.DataFrame,
    baseline_train_rows: set[tuple] | None,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify one unified interaction dataset."""

    if dataframe.isnull().any().any():
        raise ValueError(f"{label} contains null fields.")

    if dataframe.duplicated().any():
        raise ValueError(f"{label} has duplicate rows.")

    if dataframe.duplicated(
        subset=["user_id:token", "sequence_order:float"]
    ).any():
        raise ValueError(
            f"{label} has duplicate (user_id, sequence_order)."
        )

    if set(dataframe["user_id:token"]) != expected_users:
        raise ValueError(
            f"{label} does not preserve expected users."
        )

    for user_id, history in dataframe.groupby("user_id:token"):
        sequence_values = history.sort_values(
            "sequence_order:float",
            kind="mergesort",
        )["sequence_order:float"].values
        if not (
            sequence_values[:-1] < sequence_values[1:]
        ).all():
            raise ValueError(
                f"{label} chronology is invalid for user {user_id}."
            )

    catalogue_items = set(catalogue["item_id:token"])
    missing_items = set(dataframe["item_id:token"]) - catalogue_items
    if missing_items:
        raise ValueError(
            f"{label} contains interaction items absent from {ITEM_FILE}."
        )

    train, validation, test = split_unified(dataframe)

    if train.groupby("user_id:token").size().min() < 1:
        raise ValueError(
            f"{label} has a user with no training interaction."
        )

    pd.testing.assert_frame_equal(
        validation.sort_values("user_id:token").reset_index(drop=True),
        expected_validation.sort_values("user_id:token").reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        test.sort_values("user_id:token").reset_index(drop=True),
        expected_test.sort_values("user_id:token").reset_index(drop=True),
        check_dtype=False,
    )

    if baseline_train_rows is not None:
        if not row_identity(train).issubset(baseline_train_rows):
            raise ValueError(
                f"{label} sparse train rows are not a baseline subset."
            )

    return train, validation, test


def verify_metadata(
    metadata: dict,
    dataframe: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
) -> None:
    """Verify metadata against actual data."""

    checks = {
        "total_interactions": len(dataframe),
        "training_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "users": dataframe["user_id:token"].nunique(),
        "items": dataframe["item_id:token"].nunique(),
    }

    for key, actual in checks.items():
        if int(metadata[key]) != int(actual):
            raise ValueError(
                f"{label} metadata mismatch for {key}."
            )

    actual_retention = (
        len(train)
        / int(metadata["original_training_interactions"])
    )

    if abs(
        actual_retention
        - float(metadata["actual_training_retention_fraction"])
    ) > 1e-9:
        raise ValueError(
            f"{label} actual retention metadata mismatch."
        )

    train_items = set(train["item_id:token"])
    validation_unseen = ~validation["item_id:token"].isin(train_items)
    test_unseen = ~test["item_id:token"].isin(train_items)

    unseen_checks = {
        "validation_target_interactions_unseen_in_training":
            int(validation_unseen.sum()),
        "test_target_interactions_unseen_in_training":
            int(test_unseen.sum()),
    }

    for key, actual in unseen_checks.items():
        if int(metadata[key]) != actual:
            raise ValueError(
                f"{label} metadata mismatch for {key}."
            )


def verify_nesting(
    train_sets: dict[str, set[tuple]],
    scenario: str,
) -> None:
    """Verify 10 <= 25 <= 50 <= baseline nesting."""

    checks = [
        ("10", "25"),
        ("25", "50"),
        ("50", "baseline"),
    ]

    for lower, higher in checks:
        if not train_sets[lower].issubset(train_sets[higher]):
            raise ValueError(
                f"{scenario} nesting failed: {lower} not subset of {higher}."
            )


def verify_all() -> None:
    """Run all MovieLens verification checks."""

    expected_validation = load_processed_target("validation.csv")
    expected_test = load_processed_target("test.csv")
    expected_users = set(expected_validation["user_id:token"])

    baseline_dir = DATA_ROOT / "baseline"
    baseline = load_interactions(baseline_dir / INTER_FILE)
    baseline_catalogue = load_items(baseline_dir / ITEM_FILE)

    baseline_train, _, _ = verify_interactions(
        baseline,
        baseline_catalogue,
        expected_users,
        expected_validation,
        expected_test,
        None,
        "baseline",
    )

    baseline_train_rows = row_identity(baseline_train)
    baseline_hash = file_sha256(baseline_dir / INTER_FILE)
    baseline_item_hash = file_sha256(baseline_dir / ITEM_FILE)

    rows = []

    for scenario in SCENARIOS:
        train_sets = {"baseline": baseline_train_rows}

        for level in LEVELS:
            dataset_dir = DATA_ROOT / scenario / level
            dataframe = load_interactions(dataset_dir / INTER_FILE)
            catalogue = load_items(dataset_dir / ITEM_FILE)

            if file_sha256(dataset_dir / ITEM_FILE) != baseline_item_hash:
                raise ValueError(
                    f"{scenario}/{level} catalogue differs from baseline."
                )

            train, validation, test = verify_interactions(
                dataframe,
                catalogue,
                expected_users,
                expected_validation,
                expected_test,
                baseline_train_rows,
                f"{scenario}/{level}",
            )

            train_sets[level] = row_identity(train)

            if level == "100":
                if file_sha256(dataset_dir / INTER_FILE) != baseline_hash:
                    raise ValueError(
                        f"{scenario}/100 is not byte-identical to baseline."
                    )

            metadata_file = dataset_dir / "metadata.json"
            require_file(metadata_file)

            with open(metadata_file, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            verify_metadata(
                metadata,
                dataframe,
                train,
                validation,
                test,
                f"{scenario}/{level}",
            )

            rows.append(
                {
                    "scenario": scenario,
                    "retention": level,
                    "training_interactions": len(train),
                    "total_interactions": len(dataframe),
                }
            )

        verify_nesting(train_sets, scenario)

    print("\nMovieLens verification summary")
    print("-" * 80)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nAll MovieLens datasets passed verification.")


def main() -> None:
    """Entry point."""

    print("=" * 60)
    print("VERIFY MOVIELENS DATASETS")
    print("=" * 60)
    verify_all()


if __name__ == "__main__":
    main()
