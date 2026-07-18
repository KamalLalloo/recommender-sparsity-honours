"""
Verify Sequential Sparsity Datasets

Checks that every generated sequential dataset:

- exists
- contains movielens.inter
- contains metadata.json
- loads successfully
- has the expected number of interactions
- every user has at least one interaction
- each user's interactions remain chronologically ordered

Example
-------
python scripts/verify_generated_sequential_datasets.py
"""

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole_sequential"
    / "movielens"
)

SCENARIOS = ["global", "recent", "early"]

LEVELS = ["100", "50", "25", "10"]


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
        print(f"  ✗ Failed to load interaction file: {e}")
        return False

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(df) != metadata["interactions"]:
        print(
            f"  ✗ Interaction count mismatch "
            f"({len(df):,} != {metadata['interactions']:,})"
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

    interactions_per_user = df.groupby(user_col).size()

    if (interactions_per_user < 1).any():
        print("  ✗ User with zero interactions found")
        return False

    # chronological order check
    for _, group in df.groupby(user_col):
        if not group[time_col].is_monotonic_increasing:
            print("  ✗ User history is not chronological")
            return False

    print(
        f"  ✓ Users        : {df[user_col].nunique():,}"
    )
    print(
        f"  ✓ Items        : {df[item_col].nunique():,}"
    )
    print(
        f"  ✓ Interactions : {len(df):,}"
    )

    return True


def main():

    print("=" * 60)
    print("VERIFY SEQUENTIAL SPARSITY DATASETS")
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