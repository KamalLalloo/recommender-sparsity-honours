"""
Regression tests for fixed RecBole candidate catalogues.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_amazon_item_catalog  # noqa: E402
import prepare_movielens_item_catalog  # noqa: E402


def create_recbole_directories(
    recbole_root: Path,
    dataset_name: str,
    item_values: list,
) -> list[Path]:
    """
    Create synthetic baseline and sparse RecBole directories.
    """

    directories = [
        recbole_root / "baseline",
    ]

    for scenario in ["global", "recent", "early"]:
        for level in ["100", "50", "25", "10"]:
            directories.append(
                recbole_root / scenario / level
            )

    for index, directory in enumerate(directories):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        retained_items = item_values[:]

        if index % 3 == 1:
            retained_items = item_values[:-1]
        elif index % 3 == 2:
            retained_items = item_values[1:]

        interactions = pd.DataFrame(
            {
                "user_id:token": [
                    f"user_{row_index}"
                    for row_index in range(
                        1,
                        len(retained_items) + 1,
                    )
                ],
                "item_id:token": retained_items,
                "timestamp:float": list(
                    range(
                        1,
                        len(retained_items) + 1,
                    )
                ),
                "sequence_order:float": list(
                    range(
                        1,
                        len(retained_items) + 1,
                    )
                ),
            }
        )

        interactions.to_csv(
            directory / f"{dataset_name}.inter",
            sep="\t",
            index=False,
        )

    return directories


@pytest.mark.parametrize(
    (
        "catalogue_module",
        "dataset_name",
        "temporal_column",
        "item_values",
    ),
    [
        (
            prepare_movielens_item_catalog,
            "movielens",
            "movie_id",
            [30, 10, 20],
        ),
        (
            prepare_amazon_item_catalog,
            "amazon",
            "item_id",
            ["B003", "B001", "B002"],
        ),
    ],
)
def test_catalogue_identity(
    monkeypatch,
    tmp_path,
    catalogue_module,
    dataset_name,
    temporal_column,
    item_values,
):
    """
    Fixed candidate catalogues must be identical and cover interactions.
    """

    temporal_data = tmp_path / f"{dataset_name}_temporal.csv"
    recbole_root = tmp_path / "recbole" / dataset_name

    pd.DataFrame(
        {
            temporal_column: item_values
            + [
                item_values[0]
            ]
        }
    ).to_csv(
        temporal_data,
        index=False,
    )

    directories = create_recbole_directories(
        recbole_root,
        dataset_name,
        sorted(item_values),
    )

    monkeypatch.setattr(
        catalogue_module,
        "TEMPORAL_DATA",
        temporal_data,
    )
    monkeypatch.setattr(
        catalogue_module,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        catalogue_module,
        "RECBOLE_ROOT",
        recbole_root,
    )

    loaded_catalogue = (
        catalogue_module.load_item_catalogue()
    )

    if list(loaded_catalogue.columns) == ["item_id"]:
        catalogue = catalogue_module.convert_to_recbole(
            loaded_catalogue
        )
    else:
        catalogue = loaded_catalogue

    discovered_directories = (
        catalogue_module.get_dataset_directories()
    )

    assert discovered_directories == directories

    catalogue_module.validate_dataset_directories(
        discovered_directories
    )
    catalogue_module.save_catalogue(
        catalogue,
        discovered_directories,
    )
    catalogue_module.verify_catalogues(
        catalogue,
        discovered_directories,
    )

    saved_catalogues = [
        pd.read_csv(
            directory / f"{dataset_name}.item",
            sep="\t",
        )
        for directory in discovered_directories
    ]

    for saved in saved_catalogues:
        assert list(saved.columns) == ["item_id:token"]
        pd.testing.assert_frame_equal(
            saved,
            saved_catalogues[0],
            check_dtype=True,
            check_like=False,
        )

        interaction_items = set()

        for directory in discovered_directories:
            interactions = pd.read_csv(
                directory / f"{dataset_name}.inter",
                sep="\t",
                usecols=["item_id:token"],
            )
            interaction_items.update(
                interactions["item_id:token"]
            )

        catalogue_items = set(
            saved["item_id:token"]
        )

        assert interaction_items <= catalogue_items
