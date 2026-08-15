"""
Verify generated Amazon RecBole datasets.
"""

from __future__ import annotations

import runpy
from pathlib import Path


module_globals = runpy.run_path(
    str(
        Path(__file__).resolve().parent
        / "verify_datasets.py"
    )
)

module_globals["DATASET_NAME"] = "amazon"
module_globals["PROCESSED_ITEM_COLUMN"] = "item_id"
module_globals["INTER_FILE"] = "amazon.inter"
module_globals["ITEM_FILE"] = "amazon.item"
module_globals["DATA_ROOT"] = (
    module_globals["PROJECT_ROOT"]
    / "data"
    / "recbole"
    / "amazon"
)
module_globals["PROCESSED_ROOT"] = (
    module_globals["PROJECT_ROOT"]
    / "data"
    / "processed"
    / "amazon"
)


def main() -> None:
    """Entry point."""

    print("=" * 60)
    print("VERIFY AMAZON DATASETS")
    print("=" * 60)
    module_globals["verify_all"]()


if __name__ == "__main__":
    main()
