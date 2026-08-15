"""
Amazon Video Games 2023 Preprocessing Pipeline

This script performs the complete preprocessing workflow for the
Amazon Reviews 2023 Video Games dataset.

Pipeline
--------
1. Load the raw JSONL review dataset
2. Extract user_id, parent_asin, rating, and timestamp
3. Validate the raw data
4. Convert explicit ratings to implicit feedback (rating >= 4)
5. De-duplicate repeated positive user-item interactions
6. Apply iterative 5-core filtering
7. Sort interactions chronologically
8. Create deterministic sequence_order values
9. Create leave-one-out train/validation/test splits
10. Validate each preprocessing stage
11. Save processed datasets and preprocessing summary

Notes
-----
- parent_asin is used as the recommendation item identifier.
- Repeated positive interactions between the same user and item
  are de-duplicated by retaining the most recent positive review.
- The raw timestamp is retained for analysis.
- sequence_order is the deterministic ordering field intended
  for RecBole temporal splitting and sequential recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon"
    / "Video_Games.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Project Constants
# ==========================================================

IMPLICIT_RATING_THRESHOLD = 4.0
K_CORE = 5

# Number of JSONL records processed at a time.
#
# The raw Amazon file contains large text/image fields that are
# irrelevant to this project. Loading incrementally prevents
# pandas from loading the complete raw JSON objects into memory.
LOAD_CHUNK_SIZE = 200_000

RAW_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "_source_order",
]

INTERACTION_COLUMNS = [
    "user_id",
    "item_id",
    "timestamp",
]

TEMPORAL_COLUMNS = [
    "user_id",
    "item_id",
    "timestamp",
    "sequence_order",
]


# ==========================================================
# Load Dataset
# ==========================================================

def load_reviews(
    path: Path,
    chunk_size: int = LOAD_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Load required fields from the Amazon Video Games JSONL file.

    Only fields needed by the recommendation experiment are kept:

        user_id
        parent_asin -> item_id
        rating
        timestamp

    _source_order records the original JSONL row order and is
    retained temporarily so timestamp ties can be resolved
    deterministically.
    """

    print("\nLoading Amazon Video Games reviews...")

    if not path.is_file():
        raise FileNotFoundError(
            f"Raw Amazon dataset not found:\n{path}"
        )

    frames = []
    buffer = []

    source_order = 0
    loaded_records = 0

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON found in raw Amazon data "
                    f"at line {line_number}."
                ) from error

            required_fields = [
                "user_id",
                "parent_asin",
                "rating",
                "timestamp",
            ]

            missing_fields = [
                field
                for field in required_fields
                if field not in record
            ]

            if missing_fields:
                raise ValueError(
                    f"Raw Amazon record at line {line_number} "
                    "is missing required fields: "
                    f"{missing_fields}"
                )

            buffer.append(
                (
                    record["user_id"],
                    record["parent_asin"],
                    record["rating"],
                    record["timestamp"],
                    source_order,
                )
            )

            source_order += 1
            loaded_records += 1

            if len(buffer) >= chunk_size:

                frame = pd.DataFrame.from_records(
                    buffer,
                    columns=RAW_COLUMNS,
                )

                frames.append(frame)

                buffer = []

                print(
                    f"Loaded {loaded_records:,} reviews..."
                )

    if buffer:

        frame = pd.DataFrame.from_records(
            buffer,
            columns=RAW_COLUMNS,
        )

        frames.append(frame)

    if not frames:
        raise ValueError(
            "No reviews were loaded from the raw dataset."
        )

    reviews = pd.concat(
        frames,
        ignore_index=True,
    )

    del frames

    # Explicit numeric conversion catches malformed raw values.
    reviews["rating"] = pd.to_numeric(
        reviews["rating"],
        errors="raise",
    )

    reviews["timestamp"] = pd.to_numeric(
        reviews["timestamp"],
        errors="raise",
    ).astype("int64")

    reviews["_source_order"] = pd.to_numeric(
        reviews["_source_order"],
        errors="raise",
    ).astype("int64")

    print(
        f"Finished loading {len(reviews):,} reviews."
    )

    return reviews


# ==========================================================
# Raw Dataset Validation
# ==========================================================

def validate_raw_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate the extracted Amazon review dataset.
    """

    print("\nValidating raw Amazon dataset...")

    assert not df.empty, (
        "Dataset is empty."
    )

    assert list(df.columns) == RAW_COLUMNS, (
        "Raw dataset contains unexpected columns."
    )

    assert df["user_id"].isna().sum() == 0, (
        "Missing user_id values."
    )

    assert df["item_id"].isna().sum() == 0, (
        "Missing item_id values."
    )

    assert df["rating"].isna().sum() == 0, (
        "Missing rating values."
    )

    assert df["timestamp"].isna().sum() == 0, (
        "Missing timestamp values."
    )

    assert not df["user_id"].eq("").any(), (
        "Empty user_id values were found."
    )

    assert not df["item_id"].eq("").any(), (
        "Empty item_id values were found."
    )

    assert df["rating"].between(
        1.0,
        5.0,
        inclusive="both",
    ).all(), (
        "Ratings outside the expected 1-5 range were found."
    )

    assert (df["timestamp"] > 0).all(), (
        "Invalid non-positive timestamps were found."
    )

    assert not df["_source_order"].duplicated().any(), (
        "_source_order values must be unique."
    )

    duplicate_core_rows = int(
        df.duplicated(
            subset=[
                "user_id",
                "item_id",
                "rating",
                "timestamp",
            ]
        ).sum()
    )

    repeated_user_item_rows = int(
        df.duplicated(
            subset=[
                "user_id",
                "item_id",
            ],
            keep="last",
        ).sum()
    )

    print(
        f"Users                 : "
        f"{df['user_id'].nunique():,}"
    )

    print(
        f"Items                 : "
        f"{df['item_id'].nunique():,}"
    )

    print(
        f"Reviews               : "
        f"{len(df):,}"
    )

    print(
        f"Duplicate core rows   : "
        f"{duplicate_core_rows:,}"
    )

    print(
        f"Repeated user-items   : "
        f"{repeated_user_item_rows:,} removable rows"
    )

    print("Raw dataset validation passed.")


# ==========================================================
# Implicit Feedback Conversion
# ==========================================================

def filter_positive_reviews(
    df: pd.DataFrame,
    threshold: float = IMPLICIT_RATING_THRESHOLD,
) -> pd.DataFrame:
    """
    Retain reviews whose rating is greater than or equal to the
    implicit-feedback threshold.
    """

    print(
        "\nConverting ratings to positive implicit feedback..."
    )

    positive = df[
        df["rating"] >= threshold
    ].copy()

    print(
        f"Positive reviews retained: "
        f"{len(positive):,}"
    )

    return positive


# ==========================================================
# Positive Interaction De-duplication
# ==========================================================

def deduplicate_positive_interactions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove repeated positive interactions for the same user-item.

    If a user has multiple positive reviews for the same
    parent product, retain the most recent positive review.

    Timestamp ties are resolved using original source-row order.

    Returns
    -------
    implicit_df : pd.DataFrame
        One positive interaction per user-item pair.

    removed_count : int
        Number of repeated positive interaction rows removed.
    """

    print(
        "\nDe-duplicating positive user-item interactions..."
    )

    duplicate_mask = df.duplicated(
        subset=[
            "user_id",
            "item_id",
        ],
        keep="last",
    )

    # The mask above is informative only. Actual selection below
    # explicitly orders by timestamp to ensure that "most recent"
    # rather than merely "last in file" is retained.
    del duplicate_mask

    ordered = df.sort_values(
        by=[
            "user_id",
            "item_id",
            "timestamp",
            "_source_order",
        ],
        kind="mergesort",
    )

    before_count = len(ordered)

    deduplicated = (
        ordered
        .drop_duplicates(
            subset=[
                "user_id",
                "item_id",
            ],
            keep="last",
        )
        .copy()
    )

    removed_count = (
        before_count - len(deduplicated)
    )

    deduplicated = deduplicated.drop(
        columns=["rating"]
    )

    print(
        f"Repeated positive interactions removed: "
        f"{removed_count:,}"
    )

    print(
        f"Implicit interactions retained         : "
        f"{len(deduplicated):,}"
    )

    return deduplicated, removed_count


# ==========================================================
# Implicit Dataset Validation
# ==========================================================

def validate_implicit_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Validate the positive implicit-feedback dataset.
    """

    print("\nValidating implicit dataset...")

    assert not df.empty, (
        "Implicit dataset is empty."
    )

    assert "rating" not in df.columns, (
        "Rating column should not remain after conversion."
    )

    assert df["user_id"].isna().sum() == 0
    assert df["item_id"].isna().sum() == 0
    assert df["timestamp"].isna().sum() == 0

    assert not df.duplicated(
        subset=[
            "user_id",
            "item_id",
        ]
    ).any(), (
        "Duplicate user-item interactions remain after "
        "de-duplication."
    )

    print(
        f"Users        : {df['user_id'].nunique():,}"
    )

    print(
        f"Items        : {df['item_id'].nunique():,}"
    )

    print(
        f"Interactions : {len(df):,}"
    )

    print("Implicit dataset validation passed.")


# ==========================================================
# Iterative K-Core Filtering
# ==========================================================

def iterative_k_core(
    df: pd.DataFrame,
    k: int = K_CORE,
) -> tuple[pd.DataFrame, int]:
    """
    Apply iterative k-core filtering.

    Users and items with fewer than k positive interactions are
    repeatedly removed until the dataset no longer changes.
    """

    print(
        f"\nApplying iterative {k}-core filtering..."
    )

    filtered = df.copy()

    iterations = 0

    while True:

        iterations += 1

        previous_size = len(filtered)

        # -------------------------
        # Filter users
        # -------------------------

        user_counts = (
            filtered
            .groupby(
                "user_id",
                observed=True,
            )
            .size()
        )

        valid_users = user_counts[
            user_counts >= k
        ].index

        filtered = filtered[
            filtered["user_id"].isin(
                valid_users
            )
        ]

        # -------------------------
        # Filter items
        # -------------------------

        item_counts = (
            filtered
            .groupby(
                "item_id",
                observed=True,
            )
            .size()
        )

        valid_items = item_counts[
            item_counts >= k
        ].index

        filtered = filtered[
            filtered["item_id"].isin(
                valid_items
            )
        ]

        current_size = len(filtered)

        print(
            f"Iteration {iterations}: "
            f"{current_size:,} interactions, "
            f"{filtered['user_id'].nunique():,} users, "
            f"{filtered['item_id'].nunique():,} items"
        )

        if current_size == previous_size:
            break

    if filtered.empty:
        raise ValueError(
            "The dataset became empty during k-core filtering."
        )

    print(
        f"{k}-core filtering completed."
    )

    return (
        filtered.copy(),
        iterations,
    )


# ==========================================================
# K-Core Validation
# ==========================================================

def validate_k_core_dataset(
    df: pd.DataFrame,
    k: int = K_CORE,
) -> None:
    """
    Validate the final iterative k-core dataset.
    """

    print("\nValidating k-core dataset...")

    user_counts = (
        df.groupby(
            "user_id",
            observed=True,
        )
        .size()
    )

    item_counts = (
        df.groupby(
            "item_id",
            observed=True,
        )
        .size()
    )

    assert (user_counts >= k).all(), (
        f"Some users have fewer than {k} interactions."
    )

    assert (item_counts >= k).all(), (
        f"Some items have fewer than {k} interactions."
    )

    assert not df.duplicated(
        subset=[
            "user_id",
            "item_id",
        ]
    ).any(), (
        "Duplicate user-item interactions exist "
        "after k-core filtering."
    )

    print(
        f"Users        : {df['user_id'].nunique():,}"
    )

    print(
        f"Items        : {df['item_id'].nunique():,}"
    )

    print(
        f"Interactions : {len(df):,}"
    )

    print(
        f"Minimum user interactions : "
        f"{int(user_counts.min()):,}"
    )

    print(
        f"Minimum item interactions : "
        f"{int(item_counts.min()):,}"
    )

    print("K-core validation passed.")


# ==========================================================
# Chronological Ordering
# ==========================================================

def sort_chronologically(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort interactions deterministically for every user.

    Primary ordering:
        timestamp

    Tie breaker:
        original JSONL source-row order

    sequence_order is then assigned as 1..N independently
    for every user.
    """

    print(
        "\nSorting interactions chronologically..."
    )

    ordered = df.copy()

    assert "_source_order" in ordered.columns, (
        "Missing internal _source_order field."
    )

    ordered = ordered.sort_values(
        by=[
            "user_id",
            "timestamp",
            "_source_order",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ordered["sequence_order"] = (
        ordered
        .groupby(
            "user_id",
            sort=False,
            observed=True,
        )
        .cumcount()
        .add(1)
        .astype("int64")
    )

    ordered = ordered.drop(
        columns="_source_order"
    )

    ordered = ordered[
        TEMPORAL_COLUMNS
    ]

    print(
        "Chronological sorting completed."
    )

    return ordered


# ==========================================================
# Temporal Validation
# ==========================================================

def validate_temporal_order(
    df: pd.DataFrame,
) -> None:
    """
    Ensure every user's interactions are deterministically
    ordered and sequence_order equals 1..N.
    """

    print(
        "\nValidating chronological ordering..."
    )

    assert list(df.columns) == TEMPORAL_COLUMNS, (
        "Temporal dataset contains unexpected columns."
    )

    ordered = df.sort_values(
        [
            "user_id",
            "sequence_order",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Timestamp must never decrease within a user's sequence.
    timestamp_difference = (
        ordered
        .groupby(
            "user_id",
            sort=False,
            observed=True,
        )["timestamp"]
        .diff()
    )

    assert not (
        timestamp_difference.dropna() < 0
    ).any(), (
        "At least one user's timestamps are not "
        "non-decreasing."
    )

    # sequence_order must be exactly 1..N for every user.
    expected_sequence = (
        ordered
        .groupby(
            "user_id",
            sort=False,
            observed=True,
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

    assert actual_sequence.equals(
        expected_sequence.reset_index(drop=True)
    ), (
        "sequence_order does not equal 1..N "
        "for every user."
    )

    assert not ordered.duplicated(
        subset=[
            "user_id",
            "sequence_order",
        ]
    ).any(), (
        "Duplicate (user_id, sequence_order) values found."
    )

    print(
        "Chronological validation passed."
    )


# ==========================================================
# Leave-One-Out Split
# ==========================================================

def create_leave_one_out_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create chronological leave-one-out splits.

    For each user:

        final interaction       -> test
        second-last interaction -> validation
        all earlier interactions -> train

    Because iterative 5-core filtering has already been applied,
    every remaining user has at least five interactions.
    """

    print(
        "\nCreating leave-one-out split..."
    )

    max_sequence = (
        df.groupby(
            "user_id",
            observed=True,
        )["sequence_order"]
        .transform("max")
    )

    train = df[
        df["sequence_order"] <= (
            max_sequence - 2
        )
    ].copy()

    validation = df[
        df["sequence_order"] == (
            max_sequence - 1
        )
    ].copy()

    test = df[
        df["sequence_order"] == max_sequence
    ].copy()

    train = (
        train
        .sort_values(
            [
                "user_id",
                "sequence_order",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    validation = (
        validation
        .sort_values(
            [
                "user_id",
                "sequence_order",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values(
            [
                "user_id",
                "sequence_order",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print(
        "Leave-one-out split completed."
    )

    return (
        train,
        validation,
        test,
    )


# ==========================================================
# Leave-One-Out Validation
# ==========================================================

def validate_leave_one_out(
    complete: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate the chronological leave-one-out split.
    """

    print(
        "\nValidating leave-one-out split..."
    )

    train_users = set(
        train["user_id"].unique()
    )

    validation_users = set(
        validation["user_id"].unique()
    )

    test_users = set(
        test["user_id"].unique()
    )

    complete_users = set(
        complete["user_id"].unique()
    )

    assert (
        train_users
        == validation_users
        == test_users
        == complete_users
    ), (
        "User mismatch between complete/train/"
        "validation/test datasets."
    )

    user_count = len(complete_users)

    assert len(validation) == user_count, (
        "Validation must contain exactly "
        "one interaction per user."
    )

    assert len(test) == user_count, (
        "Test must contain exactly "
        "one interaction per user."
    )

    assert not validation[
        "user_id"
    ].duplicated().any(), (
        "Validation contains multiple interactions "
        "for a user."
    )

    assert not test[
        "user_id"
    ].duplicated().any(), (
        "Test contains multiple interactions "
        "for a user."
    )

    train_max_sequence = (
        train
        .groupby(
            "user_id",
            observed=True,
        )["sequence_order"]
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

    assert (
        train_max_sequence
        < validation_sequence
    ).all(), (
        "Each user's training sequence_order must "
        "be less than validation sequence_order."
    )

    assert (
        validation_sequence
        < test_sequence
    ).all(), (
        "Each user's validation sequence_order must "
        "be less than test sequence_order."
    )

    assert (
        len(train)
        + len(validation)
        + len(test)
        == len(complete)
    ), (
        "Train + validation + test interaction counts "
        "do not equal the complete k-core dataset."
    )

    print(
        f"Train interactions      : {len(train):,}"
    )

    print(
        f"Validation interactions : {len(validation):,}"
    )

    print(
        f"Test interactions       : {len(test):,}"
    )

    print(
        "Leave-one-out validation passed."
    )


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """
    Save a processed dataset without internal helper fields.
    """

    output = OUTPUT_DIR / filename

    if "sequence_order" in df.columns:

        output_df = df[
            TEMPORAL_COLUMNS
        ].copy()

    else:

        output_df = df[
            INTERACTION_COLUMNS
        ].copy()

    output_df.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )


# ==========================================================
# Save Summary
# ==========================================================

def save_summary(
    raw_df: pd.DataFrame,
    positive_df: pd.DataFrame,
    implicit_df: pd.DataFrame,
    k_core_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    duplicate_positive_rows_removed: int,
    iterations: int,
) -> None:
    """
    Save preprocessing statistics and methodological details.
    """

    repeated_timestamp_users = int(
        temporal_df.loc[
            temporal_df.duplicated(
                subset=[
                    "user_id",
                    "timestamp",
                ],
                keep=False,
            ),
            "user_id",
        ].nunique()
    )

    raw_repeated_user_item_rows = int(
        raw_df.duplicated(
            subset=[
                "user_id",
                "item_id",
            ],
            keep="last",
        ).sum()
    )

    positive_repeated_user_item_rows = int(
        positive_df.duplicated(
            subset=[
                "user_id",
                "item_id",
            ],
            keep="last",
        ).sum()
    )

    timestamp_min = int(
        temporal_df["timestamp"].min()
    )

    timestamp_max = int(
        temporal_df["timestamp"].max()
    )

    history_lengths = (
        temporal_df
        .groupby(
            "user_id",
            observed=True,
        )
        .size()
    )

    summary = {

        "dataset": (
            "Amazon Reviews 2023 - Video Games"
        ),

        "item_identifier": "parent_asin",

        "raw": {

            "users":
                int(
                    raw_df[
                        "user_id"
                    ].nunique()
                ),

            "items":
                int(
                    raw_df[
                        "item_id"
                    ].nunique()
                ),

            "reviews":
                int(len(raw_df)),

            "repeated_user_item_rows":
                raw_repeated_user_item_rows,

        },

        "implicit_conversion": {

            "rating_threshold":
                IMPLICIT_RATING_THRESHOLD,

            "positive_reviews_before_deduplication":
                int(len(positive_df)),

            "users_before_deduplication":
                int(
                    positive_df[
                        "user_id"
                    ].nunique()
                ),

            "items_before_deduplication":
                int(
                    positive_df[
                        "item_id"
                    ].nunique()
                ),

            "repeated_positive_user_item_rows":
                positive_repeated_user_item_rows,

        },

        "implicit": {

            "users":
                int(
                    implicit_df[
                        "user_id"
                    ].nunique()
                ),

            "items":
                int(
                    implicit_df[
                        "item_id"
                    ].nunique()
                ),

            "interactions":
                int(len(implicit_df)),

            "duplicate_positive_rows_removed":
                int(
                    duplicate_positive_rows_removed
                ),

            "deduplication_policy":
                (
                    "retain most recent positive "
                    "interaction per user-item pair"
                ),

        },

        "k_core": {

            "users":
                int(
                    k_core_df[
                        "user_id"
                    ].nunique()
                ),

            "items":
                int(
                    k_core_df[
                        "item_id"
                    ].nunique()
                ),

            "interactions":
                int(len(k_core_df)),

            "k":
                K_CORE,

            "iterations":
                int(iterations),

        },

        "splits": {

            "train_interactions":
                int(len(train_df)),

            "validation_interactions":
                int(len(validation_df)),

            "test_interactions":
                int(len(test_df)),

        },

        "post_processing": {

            "users":
                int(
                    temporal_df[
                        "user_id"
                    ].nunique()
                ),

            "items":
                int(
                    temporal_df[
                        "item_id"
                    ].nunique()
                ),

            "interactions":
                int(len(temporal_df)),

        },

        "history_lengths": {

            "mean_user_history_length":
                float(history_lengths.mean()),

            "median_user_history_length":
                float(history_lengths.median()),

            "minimum_user_history_length":
                int(history_lengths.min()),

            "maximum_user_history_length":
                int(history_lengths.max()),

        },

        "ordering": {

            "field":
                "sequence_order",

            "primary_order":
                "timestamp",

            "tie_breaker":
                "original_source_row_order",

            "users_with_repeated_timestamps":
                repeated_timestamp_users,

            "minimum_timestamp":
                timestamp_min,

            "maximum_timestamp":
                timestamp_max,

        },

    }

    output = (
        OUTPUT_DIR
        / "preprocessing_summary.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"Saved: {output}"
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Run the complete Amazon preprocessing pipeline.
    """

    print("=" * 60)
    print("Amazon Video Games 2023 Preprocessing")
    print("=" * 60)

    # ------------------------------------------------------
    # Stage 1: Load raw reviews
    # ------------------------------------------------------

    reviews = load_reviews(
        RAW_DATA
    )

    validate_raw_dataset(
        reviews
    )

    # ------------------------------------------------------
    # Stage 2: Positive implicit feedback
    # ------------------------------------------------------

    positive = filter_positive_reviews(
        reviews,
        threshold=IMPLICIT_RATING_THRESHOLD,
    )

    # ------------------------------------------------------
    # Stage 3: Positive user-item de-duplication
    # ------------------------------------------------------

    (
        implicit,
        duplicate_positive_rows_removed,
    ) = deduplicate_positive_interactions(
        positive
    )

    validate_implicit_dataset(
        implicit
    )

    save_dataset(
        implicit,
        "01_implicit_interactions.csv",
    )

    # ------------------------------------------------------
    # Stage 4: Iterative 5-core filtering
    # ------------------------------------------------------

    k_core, iterations = iterative_k_core(
        implicit,
        k=K_CORE,
    )

    validate_k_core_dataset(
        k_core,
        k=K_CORE,
    )

    save_dataset(
        k_core,
        "02_5core_interactions.csv",
    )

    # ------------------------------------------------------
    # Stage 5: Deterministic chronological ordering
    # ------------------------------------------------------

    temporal = sort_chronologically(
        k_core
    )

    validate_temporal_order(
        temporal
    )

    save_dataset(
        temporal,
        "03_temporal_interactions.csv",
    )

    # ------------------------------------------------------
    # Stage 6: Leave-one-out splitting
    # ------------------------------------------------------

    (
        train,
        validation,
        test,
    ) = create_leave_one_out_split(
        temporal
    )

    validate_leave_one_out(
        temporal,
        train,
        validation,
        test,
    )

    save_dataset(
        train,
        "train.csv",
    )

    save_dataset(
        validation,
        "validation.csv",
    )

    save_dataset(
        test,
        "test.csv",
    )

    # ------------------------------------------------------
    # Stage 7: Summary
    # ------------------------------------------------------

    save_summary(
        reviews,
        positive,
        implicit,
        k_core,
        temporal,
        train,
        validation,
        test,
        duplicate_positive_rows_removed,
        iterations,
    )

    print()
    print("=" * 60)
    print(
        "Amazon preprocessing completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
