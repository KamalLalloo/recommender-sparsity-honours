"""
Prepare Fixed Amazon Item Catalogue for RecBole

This script creates a single Amazon .item catalogue containing
every item in the fully preprocessed Amazon dataset.

The identical catalogue is written to the baseline dataset and
every sparsity condition so that RecBole uses the same item
universe for full-sort evaluation across all experiments.
"""

from pathlib import Path

import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPORAL_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon"
    / "03_temporal_interactions.csv"
)

RECBOLE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "amazon"
)


# ==========================================================
# Expected Dataset Directories
# ==========================================================

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


# ==========================================================
# Load Catalogue
# ==========================================================

def load_item_catalogue() -> pd.DataFrame:
    """
    Load all unique items from the complete preprocessed
    Amazon dataset.
    """

    print("\nLoading Amazon item catalogue...")

    if not TEMPORAL_DATA.is_file():
        raise FileNotFoundError(
            "Temporal Amazon dataset was not found:\n"
            f"{TEMPORAL_DATA}"
        )

    dataset = pd.read_csv(
        TEMPORAL_DATA,
        usecols=["item_id"],
    )

    if dataset.empty:
        raise ValueError(
            "Temporal Amazon dataset is empty."
        )

    if dataset["item_id"].isna().any():
        raise ValueError(
            "Missing item_id values were found."
        )

    items = (
        dataset[["item_id"]]
        .drop_duplicates()
        .sort_values(
            "item_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if items["item_id"].duplicated().any():
        raise ValueError(
            "Duplicate item IDs remain in catalogue."
        )

    print(
        f"Unique catalogue items: "
        f"{len(items):,}"
    )

    return items


# ==========================================================
# RecBole Conversion
# ==========================================================

def convert_to_recbole(
    items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert item IDs to RecBole .item format.
    """

    converted = items.rename(
        columns={
            "item_id": "item_id:token",
        }
    )

    return converted[
        ["item_id:token"]
    ]


# ==========================================================
# Dataset Directories
# ==========================================================

def get_dataset_directories() -> list[Path]:
    """
    Return every Amazon RecBole dataset directory.
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


# ==========================================================
# Validation
# ==========================================================

def validate_dataset_directories(
    directories: list[Path],
) -> None:
    """
    Ensure every expected Amazon RecBole dataset exists.
    """

    print(
        "\nValidating Amazon RecBole directories..."
    )

    missing_directories = []
    missing_inter_files = []

    for directory in directories:

        if not directory.is_dir():
            missing_directories.append(
                directory
            )
            continue

        interaction_file = (
            directory
            / "amazon.inter"
        )

        if not interaction_file.is_file():
            missing_inter_files.append(
                interaction_file
            )

    if missing_directories:
        text = "\n".join(
            str(path)
            for path in missing_directories
        )

        raise FileNotFoundError(
            "Missing dataset directories:\n"
            f"{text}"
        )

    if missing_inter_files:
        text = "\n".join(
            str(path)
            for path in missing_inter_files
        )

        raise FileNotFoundError(
            "Missing amazon.inter files:\n"
            f"{text}"
        )

    print(
        f"Validated {len(directories)} "
        "Amazon dataset directories."
    )


# ==========================================================
# Save Catalogue
# ==========================================================

def save_catalogue(
    catalogue: pd.DataFrame,
    directories: list[Path],
) -> None:
    """
    Write the identical amazon.item file to every
    experiment directory.
    """

    print(
        "\nWriting fixed item catalogue..."
    )

    for directory in directories:

        output_file = (
            directory
            / "amazon.item"
        )

        catalogue.to_csv(
            output_file,
            sep="\t",
            index=False,
        )

        relative_path = (
            output_file.relative_to(
                PROJECT_ROOT
            )
        )

        print(
            f"Saved: {relative_path}"
        )


# ==========================================================
# Verify Saved Catalogues
# ==========================================================

def verify_catalogues(
    expected_catalogue: pd.DataFrame,
    directories: list[Path],
) -> None:
    """
    Confirm every amazon.item file is identical and covers all
    interaction items.
    """

    print(
        "\nVerifying saved catalogues..."
    )

    expected_items = set(
        expected_catalogue["item_id:token"]
    )

    for directory in directories:

        filepath = (
            directory
            / "amazon.item"
        )

        saved = pd.read_csv(
            filepath,
            sep="\t",
        )

        try:
            pd.testing.assert_frame_equal(
                saved,
                expected_catalogue,
                check_dtype=True,
                check_like=False,
            )

        except AssertionError as error:
            raise ValueError(
                "Item catalogue differs in:\n"
                f"{filepath}"
            ) from error

        interactions = pd.read_csv(
            directory / "amazon.inter",
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
                f"amazon.item in {directory}."
            )

    print(
        "All Amazon item catalogues are identical "
        "and cover every interaction item."
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print("=" * 60)
    print("Prepare Amazon Fixed Item Catalogue")
    print("=" * 60)

    items = load_item_catalogue()

    catalogue = convert_to_recbole(
        items
    )

    directories = (
        get_dataset_directories()
    )

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
        "Amazon fixed item catalogue "
        "prepared successfully."
    )
    print(f"Catalogue items: {len(catalogue):,}")
    print(f"Destinations   : {len(directories):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
