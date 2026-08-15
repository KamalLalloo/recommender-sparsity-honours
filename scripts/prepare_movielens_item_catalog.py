"""
Prepare Fixed MovieLens Item Catalogue for RecBole.

The catalogue contains every MovieLens item in the complete
post-preprocessing temporal dataset and is written identically to
the baseline and every sparsity-condition directory.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPORAL_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens"
    / "03_temporal_interactions.csv"
)

RECBOLE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "movielens"
)

RETENTION_LEVELS = [
    "100",
    "50",
    "25",
    "10",
]

SCENARIOS = [
    "global",
    "recent",
    "early",
]


def load_item_catalogue() -> pd.DataFrame:
    """
    Load all unique MovieLens movie IDs.
    """

    print("\nLoading MovieLens item catalogue...")

    if not TEMPORAL_DATA.is_file():
        raise FileNotFoundError(
            "Temporal MovieLens dataset was not found:\n"
            f"{TEMPORAL_DATA}"
        )

    dataset = pd.read_csv(
        TEMPORAL_DATA,
        usecols=["movie_id"],
    )

    if dataset.empty:
        raise ValueError(
            "Temporal MovieLens dataset is empty."
        )

    if dataset["movie_id"].isna().any():
        raise ValueError(
            "Missing movie_id values were found."
        )

    items = (
        dataset[["movie_id"]]
        .drop_duplicates()
        .sort_values(
            "movie_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique catalogue items: {len(items):,}"
    )

    return items.rename(
        columns={
            "movie_id": "item_id:token",
        }
    )[["item_id:token"]]


def get_dataset_directories() -> list[Path]:
    """
    Return every MovieLens RecBole dataset directory.
    """

    directories = [
        RECBOLE_ROOT / "baseline",
    ]

    for scenario in SCENARIOS:
        for level in RETENTION_LEVELS:
            directories.append(
                RECBOLE_ROOT
                / scenario
                / level
            )

    return directories


def validate_dataset_directories(
    directories: list[Path],
) -> None:
    """
    Ensure every expected MovieLens RecBole dataset exists.
    """

    missing = []

    for directory in directories:
        if not directory.is_dir():
            missing.append(directory)
            continue

        if not (
            directory
            / "movielens.inter"
        ).is_file():
            missing.append(
                directory
                / "movielens.inter"
            )

    if missing:
        text = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Missing expected MovieLens dataset paths:\n"
            f"{text}"
        )


def save_catalogue(
    catalogue: pd.DataFrame,
    directories: list[Path],
) -> None:
    """
    Write the identical movielens.item file to every directory.
    """

    print("\nWriting fixed MovieLens item catalogue...")

    for directory in directories:
        output_file = directory / "movielens.item"

        catalogue.to_csv(
            output_file,
            sep="\t",
            index=False,
        )

        print(
            f"Saved: {output_file.relative_to(PROJECT_ROOT)}"
        )


def verify_catalogues(
    expected_catalogue: pd.DataFrame,
    directories: list[Path],
) -> None:
    """
    Confirm all catalogues are identical and cover all interactions.
    """

    expected_items = set(
        expected_catalogue["item_id:token"]
    )

    for directory in directories:
        item_file = directory / "movielens.item"
        inter_file = directory / "movielens.inter"

        saved = pd.read_csv(
            item_file,
            sep="\t",
        )

        pd.testing.assert_frame_equal(
            saved,
            expected_catalogue,
            check_dtype=True,
            check_like=False,
        )

        interactions = pd.read_csv(
            inter_file,
            sep="\t",
            usecols=["item_id:token"],
        )

        missing_items = (
            set(interactions["item_id:token"])
            - expected_items
        )

        if missing_items:
            raise ValueError(
                "Interaction items are missing from "
                f"movielens.item in {directory}."
            )

    print(
        "All MovieLens item catalogues are identical "
        "and cover every interaction item."
    )


def main() -> None:
    """
    Prepare and verify the fixed MovieLens item catalogue.
    """

    print("=" * 60)
    print("Prepare MovieLens Fixed Item Catalogue")
    print("=" * 60)

    catalogue = load_item_catalogue()
    directories = get_dataset_directories()

    validate_dataset_directories(
        directories
    )

    save_catalogue(
        catalogue,
        directories,
    )

    verify_catalogues(
        catalogue,
        directories,
    )

    print()
    print("=" * 60)
    print(
        "MovieLens fixed item catalogue "
        "prepared successfully."
    )
    print(f"Catalogue items: {len(catalogue):,}")
    print(f"Destinations   : {len(directories):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
