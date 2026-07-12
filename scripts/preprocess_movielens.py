"""
MovieLens-1M Preprocessing Pipeline

This script performs the complete preprocessing workflow for the
MovieLens-1M dataset.

Pipeline
--------
1. Load the raw ratings dataset
2. Validate the raw data
3. Convert explicit ratings to implicit feedback (rating >= 4)
4. Apply iterative 5-core filtering
5. Sort interactions chronologically
6. Create leave-one-out train/validation/test splits
7. Validate each preprocessing stage
8. Save processed datasets and preprocessing summary
"""

from pathlib import Path
import json
import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "movielens"
    / "ratings.dat"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

def load_ratings(path: Path) -> pd.DataFrame:
    """Load the MovieLens ratings dataset."""

    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    )

    return ratings


# ==========================================================
# Validation
# ==========================================================

def validate_raw_dataset(df: pd.DataFrame) -> None:
    """Validate the raw MovieLens dataset."""

    print("\nValidating raw dataset...")

    assert not df.empty, "Dataset is empty."

    assert df["user_id"].isna().sum() == 0
    assert df["movie_id"].isna().sum() == 0
    assert df["rating"].isna().sum() == 0
    assert df["timestamp"].isna().sum() == 0

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows:,}")

    print("Raw dataset validation passed.")


# ==========================================================
# Implicit Conversion
# ==========================================================

def convert_to_implicit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ratings >= 4 and remove the rating column.
    """

    print("\nConverting to implicit feedback...")

    implicit = df[df["rating"] >= 4].copy()

    implicit.drop(columns=["rating"], inplace=True)

    return implicit


# ==========================================================
# Validation
# ==========================================================

def validate_implicit_dataset(df: pd.DataFrame) -> None:
    """Validate implicit dataset."""

    print("\nValidating implicit dataset...")

    assert "rating" not in df.columns

    assert df["user_id"].isna().sum() == 0
    assert df["movie_id"].isna().sum() == 0
    assert df["timestamp"].isna().sum() == 0

    print("Implicit dataset validation passed.")


# ==========================================================
# Iterative 5-Core Filtering
# ==========================================================

def iterative_k_core(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, int]:
    """
    Apply iterative k-core filtering.

    Users and items with fewer than k interactions are removed
    repeatedly until the dataset no longer changes.

    Returns
    -------
    filtered_df : pd.DataFrame
        The filtered interaction dataset.

    iterations : int
        Number of filtering iterations performed.
    """

    print(f"\nApplying iterative {k}-core filtering...")

    filtered = df.copy()
    iterations = 0

    while True:

        iterations += 1

        previous_size = len(filtered)

        # Remove users with fewer than k interactions
        user_counts = filtered.groupby("user_id").size()
        valid_users = user_counts[user_counts >= k].index

        filtered = filtered[
            filtered["user_id"].isin(valid_users)
        ]

        # Remove movies with fewer than k interactions
        movie_counts = filtered.groupby("movie_id").size()
        valid_movies = movie_counts[movie_counts >= k].index

        filtered = filtered[
            filtered["movie_id"].isin(valid_movies)
        ]

        current_size = len(filtered)

        print(
            f"Iteration {iterations}: "
            f"{current_size:,} interactions"
        )

        if current_size == previous_size:
            break

    print("5-core filtering completed.")

    return filtered, iterations


# ==========================================================
# Validation
# ==========================================================

def validate_k_core_dataset(
    df: pd.DataFrame,
    k: int = 5
) -> None:
    """
    Validate the k-core filtered dataset.
    """

    print("\nValidating k-core dataset...")

    user_counts = df.groupby("user_id").size()
    movie_counts = df.groupby("movie_id").size()

    assert (user_counts >= k).all(), (
        "Some users have fewer than "
        f"{k} interactions."
    )

    assert (movie_counts >= k).all(), (
        "Some movies have fewer than "
        f"{k} interactions."
    )

    print(
        f"Users: {df['user_id'].nunique():,}"
    )

    print(
        f"Movies: {df['movie_id'].nunique():,}"
    )

    print(
        f"Interactions: {len(df):,}"
    )

    print("K-core validation passed.")


# ==========================================================
# Chronological Ordering
# ==========================================================

def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort interactions chronologically for every user.
    """

    print("\nSorting interactions chronologically...")

    sorted_df = df.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)

    print("Chronological sorting completed.")

    return sorted_df

# ==========================================================
# Validation
# ==========================================================

def validate_temporal_order(df: pd.DataFrame) -> None:
    """
    Ensure every user's interactions
    are sorted chronologically.
    """

    print("\nValidating chronological ordering...")

    for _, user_df in df.groupby("user_id"):

        timestamps = user_df["timestamp"].values

        assert (
            timestamps[:-1] <= timestamps[1:]
        ).all(), (
            "Interactions are not sorted "
            "chronologically."
        )

    print("Chronological validation passed.")



# ==========================================================
# Leave-One-Out Split
# ==========================================================

def create_leave_one_out_split(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create chronological leave-one-out splits.

    For each user:
        - last interaction -> test
        - second-last interaction -> validation
        - all remaining interactions -> train
    """

    print("\nCreating leave-one-out split...")

    train = []
    validation = []
    test = []

    for _, user_df in df.groupby("user_id"):

        user_df = user_df.sort_values("timestamp")

        # Because we have already applied 5-core,
        # every user has at least 5 interactions.

        train.append(user_df.iloc[:-2])

        validation.append(user_df.iloc[-2:-1])

        test.append(user_df.iloc[-1:])

    train = pd.concat(train).reset_index(drop=True)
    validation = pd.concat(validation).reset_index(drop=True)
    test = pd.concat(test).reset_index(drop=True)

    print("Leave-one-out split completed.")

    return train, validation, test


# ==========================================================
# Validation
# ==========================================================

def validate_leave_one_out(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame
) -> None:
    """
    Validate the leave-one-out split.
    """

    print("\nValidating leave-one-out split...")

    train_users = set(train["user_id"].unique())
    validation_users = set(validation["user_id"].unique())
    test_users = set(test["user_id"].unique())

    assert train_users == validation_users == test_users, (
        "User mismatch between train/validation/test."
    )

    assert len(validation) == len(validation_users), (
        "Validation should contain one interaction per user."
    )

    assert len(test) == len(test_users), (
        "Test should contain one interaction per user."
    )

    print(f"Train interactions      : {len(train):,}")
    print(f"Validation interactions : {len(validation):,}")
    print(f"Test interactions       : {len(test):,}")

    print("Leave-one-out validation passed.")


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(df: pd.DataFrame, filename: str) -> None:

    output = OUTPUT_DIR / filename

    df.to_csv(output, index=False)

    print(f"\nSaved: {output}")


# ==========================================================
# Save Summary
# ==========================================================

def save_summary(
    raw_df: pd.DataFrame,
    implicit_df: pd.DataFrame,
    k_core_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    iterations: int
) -> None:

    summary = {

        "raw": {

            "users":
                int(raw_df["user_id"].nunique()),

            "movies":
                int(raw_df["movie_id"].nunique()),

            "interactions":
                int(len(raw_df))

        },

        "implicit": {

            "users":
                int(implicit_df["user_id"].nunique()),

            "movies":
                int(implicit_df["movie_id"].nunique()),

            "interactions":
                int(len(implicit_df))

        },

        "k_core": {

            "users":
                int(k_core_df["user_id"].nunique()),

            "movies":
                int(k_core_df["movie_id"].nunique()),

            "interactions":
                int(len(k_core_df)),

            "k": 5,

            "iterations":
                iterations

        }
        ,
        "splits": {

            "train_interactions":
                int(len(train_df)),

            "validation_interactions":
                int(len(validation_df)),

            "test_interactions":
                int(len(test_df))

        }

    }

    output = OUTPUT_DIR / "preprocessing_summary.json"

    with open(output, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved: {output}")

# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("MovieLens-1M Preprocessing")
    print("=" * 60)

    ratings = load_ratings(RAW_DATA)

    validate_raw_dataset(ratings)

    implicit = convert_to_implicit(ratings)

    validate_implicit_dataset(implicit)

    save_dataset(
        implicit,
        "01_implicit_interactions.csv"
    )

    k_core, iterations = iterative_k_core(
        implicit,
        k=5
    )

    validate_k_core_dataset(
        k_core,
        k=5
    )

    temporal = sort_chronologically(
        k_core
    )

    validate_temporal_order(
        temporal
    )

    train, validation, test = create_leave_one_out_split(
        temporal
    )

    validate_leave_one_out(
        train,
        validation,
        test
    )

    save_dataset(
        k_core,
        "02_5core_interactions.csv"
    )

    save_dataset(
        temporal,
        "03_temporal_interactions.csv"
    )

    save_dataset(
        train,
        "train.csv"
    )

    save_dataset(
        validation,
        "validation.csv"
    )

    save_dataset(
        test,
        "test.csv"
    )





    save_summary(
        ratings,
        implicit,
        k_core,
        train,
        validation,
        test,
        iterations
    )

    print(
        "\nStage 3 preprocessing completed successfully."
    )

if __name__ == "__main__":
    main()