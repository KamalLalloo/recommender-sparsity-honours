"""
Verify Generated RecBole Sparsity Datasets

Checks that every generated dataset:

- exists
- contains movielens.inter
- contains metadata.json
- loads successfully
- matches metadata
- every user has at least one interaction
- each user's interactions remain chronological

Example
-------
python scripts/verify_datasets.py
"""

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "movielens"
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


def verify_dataset(dataset_dir: Path) -> bool:
    """
    Verify one generated dataset.
    """

    print(f"\nChecking {dataset_dir.relative_to(DATA_ROOT)}")

    inter_file = dataset_dir / "movielens.inter"
    metadata_file = dataset_dir / "metadata.json"

    if not inter_file.exists():
        print("  ✗ Missing movielens.inter")
        return False

    if not metadata_file.exists():
        print("  ✗ Missing metadata.json")
        return False

    try:
        df = pd.read_csv(
            inter_file,
            sep="\t",
        )
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return False

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    expected = metadata["total_interactions"]

    if len(df) != expected:
        print(
            f"  ✗ Interaction count mismatch "
            f"({len(df):,} != {expected:,})"
        )
        return False

    user_col = next(
        c for c in df.columns
        if c.startswith("user_id")
    )

    item_col = next(
        c for c in df.columns
        if c.startswith("item_id")
    )

    time_col = next(
        c for c in df.columns
        if c.startswith("timestamp")
    )

    # Every user has interactions
    counts = df.groupby(user_col).size()

    if (counts < 1).any():
        print("  ✗ User with zero interactions")
        return False

    # Histories remain chronological
    for _, history in df.groupby(user_col):

        if not history[time_col].is_monotonic_increasing:
            print("  ✗ Non-chronological user history")
            return False

    print(f"  ✓ Users        : {df[user_col].nunique():,}")
    print(f"  ✓ Items        : {df[item_col].nunique():,}")
    print(f"  ✓ Interactions : {len(df):,}")

    return True


def main():

    print("=" * 60)
    print("VERIFY GENERATED SPARSITY DATASETS")
    print("=" * 60)

    total = 0
    passed = 0

    for scenario in SCENARIOS:
        for level in LEVELS:

            dataset_dir = (
                DATA_ROOT
                / scenario
                / level
            )

            total += 1

            if verify_dataset(dataset_dir):
                passed += 1

    print("\n" + "=" * 60)

    if passed == total:
        print(f"SUCCESS: {passed}/{total} datasets verified.")
    else:
        print(f"FAILED: {passed}/{total} datasets verified.")

    print("=" * 60)


if __name__ == "__main__":
    main()