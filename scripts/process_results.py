"""
Process Raw Experiment Results

This script converts the append-only raw experiment log into
clean, dataset-specific result files.

Processing steps
----------------
1. Load results/raw/experiment_runs.csv
2. Validate the required columns
3. Keep only rows where run_type == "final"
4. Normalize text values and dataset paths
5. Remove duplicate experiment runs
6. Keep the most recent run for each experiment configuration
7. Sort the results consistently
8. Write separate processed files for each dataset

Outputs
-------
results/processed/movielens_results.csv
results/processed/amazon_results.csv

Duplicate definition
--------------------
Two rows are considered duplicate experiment configurations when
they have the same:

- dataset
- experiment_type
- retention_level
- model
- seed

The most recent final run is retained.
"""

import argparse
from pathlib import Path

import pandas as pd


# ==========================================================
# Project Constants
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "experiment_results.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "processed"
)

EXPERIMENT_KEY_COLUMNS = [
    "dataset",
    "experiment_type",
    "retention_level",
    "model",
    "seed",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "run_type",
    "experiment_type",
    "retention_level",
    "dataset",
    "dataset_directory",
    "model",
    "seed",
]

DATASET_OUTPUT_FILES = {
    "movielens": "movielens_results.csv",
    "amazon": "amazon_results.csv",
}

EXPERIMENT_TYPE_ORDER = {
    "baseline": 0,
    "global": 1,
    "recent": 2,
    "early": 3,
}


# ==========================================================
# Command-Line Arguments
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Process the raw experiment log into clean, "
            "dataset-specific result files."
        )
    )

    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help=(
            "Path to the raw experiment CSV. "
            "Defaults to results/raw/experiment_results.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help=(
            "Directory where processed result files are written. "
            "Defaults to results/processed."
        ),
    )

    return parser.parse_args()


# ==========================================================
# Validation
# ==========================================================

def validate_input_file(input_file: Path) -> None:
    """
    Confirm that the raw results file exists and is not empty.
    """

    if not input_file.is_file():
        raise FileNotFoundError(
            "Raw experiment results file does not exist:\n"
            f"{input_file}"
        )

    if input_file.stat().st_size == 0:
        raise ValueError(
            "Raw experiment results file is empty:\n"
            f"{input_file}"
        )


def validate_required_columns(
    results: pd.DataFrame,
) -> None:
    """
    Confirm that the raw results contain all required columns.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in results.columns
    ]

    if missing_columns:
        missing_text = "\n".join(
            f"  - {column}"
            for column in missing_columns
        )

        raise ValueError(
            "The raw experiment results file is missing "
            "the following required columns:\n"
            f"{missing_text}"
        )


def validate_missing_values(
    results: pd.DataFrame,
) -> None:
    """
    Check required columns for missing values.
    """

    missing_counts = (
        results[REQUIRED_COLUMNS]
        .isna()
        .sum()
    )

    columns_with_missing_values = missing_counts[
        missing_counts > 0
    ]

    if not columns_with_missing_values.empty:
        missing_text = "\n".join(
            f"  - {column}: {count}"
            for column, count
            in columns_with_missing_values.items()
        )

        raise ValueError(
            "Missing values were found in required columns:\n"
            f"{missing_text}"
        )


# ==========================================================
# Data Cleaning
# ==========================================================

def normalize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize timestamps, labels, model names, paths, and seeds.
    """

    cleaned = results.copy()

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="raise",
    )

    cleaned["run_type"] = (
        cleaned["run_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    cleaned["dataset"] = (
        cleaned["dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    cleaned["experiment_type"] = (
        cleaned["experiment_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    cleaned["model"] = (
        cleaned["model"]
        .astype(str)
        .str.strip()
    )

    cleaned["dataset_directory"] = (
        cleaned["dataset_directory"]
        .astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
    )

    cleaned["retention_level"] = pd.to_numeric(
        cleaned["retention_level"],
        errors="raise",
    ).astype(int)

    cleaned["seed"] = pd.to_numeric(
        cleaned["seed"],
        errors="raise",
    ).astype(int)

    return cleaned


def keep_final_runs(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only rows explicitly marked as final.
    """

    return results[
        results["run_type"] == "final"
    ].copy()


def remove_duplicate_experiments(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate final experiments.

    Rows are sorted by timestamp and the newest row is retained
    for each experiment key.
    """

    sorted_results = results.sort_values(
        by="timestamp",
        ascending=True,
    )

    duplicate_mask = sorted_results.duplicated(
        subset=EXPERIMENT_KEY_COLUMNS,
        keep="last",
    )

    removed_duplicates = sorted_results[
        duplicate_mask
    ].copy()

    deduplicated_results = sorted_results[
        ~duplicate_mask
    ].copy()

    return deduplicated_results, removed_duplicates


def sort_processed_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort results into a stable, readable experiment order.
    """

    sorted_results = results.copy()

    sorted_results["_experiment_order"] = (
        sorted_results["experiment_type"]
        .map(EXPERIMENT_TYPE_ORDER)
        .fillna(999)
    )

    sorted_results = sorted_results.sort_values(
        by=[
            "dataset",
            "_experiment_order",
            "experiment_type",
            "retention_level",
            "model",
            "seed",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
            True,
        ],
    )

    sorted_results = sorted_results.drop(
        columns="_experiment_order"
    )

    return sorted_results.reset_index(drop=True)


def format_output_values(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert timestamps back to readable ISO strings.
    """

    formatted = results.copy()

    formatted["timestamp"] = (
        formatted["timestamp"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    return formatted


# ==========================================================
# Output
# ==========================================================

def write_dataset_results(
    results: pd.DataFrame,
    output_directory: Path,
) -> dict[str, int]:
    """
    Write one processed CSV for each supported dataset.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    written_counts = {}

    for dataset_name, filename in DATASET_OUTPUT_FILES.items():
        dataset_results = results[
            results["dataset"] == dataset_name
        ].copy()

        output_file = output_directory / filename

        dataset_results.to_csv(
            output_file,
            index=False,
        )

        written_counts[dataset_name] = len(
            dataset_results
        )

        print(
            f"Wrote {len(dataset_results):,} rows to "
            f"{output_file.relative_to(PROJECT_ROOT)}"
        )

    return written_counts


def write_removed_duplicates(
    duplicates: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Save removed duplicate runs for inspection.

    The file is only created when duplicates were found.
    """

    duplicate_file = (
        output_directory
        / "removed_duplicates.csv"
    )

    if duplicates.empty:
        if duplicate_file.exists():
            duplicate_file.unlink()

        print("No duplicate final experiments were found.")
        return

    formatted_duplicates = format_output_values(
        duplicates
    )

    formatted_duplicates.to_csv(
        duplicate_file,
        index=False,
    )

    print(
        f"Saved {len(duplicates):,} removed duplicate runs to "
        f"{duplicate_file.relative_to(PROJECT_ROOT)}"
    )


# ==========================================================
# Reporting
# ==========================================================

def print_duplicate_summary(
    duplicates: pd.DataFrame,
) -> None:
    """
    Display duplicate experiment keys that were removed.
    """

    if duplicates.empty:
        return

    print("\nRemoved duplicate experiment runs")
    print("-" * 60)

    display_columns = [
        "timestamp",
        "dataset",
        "experiment_type",
        "retention_level",
        "model",
        "seed",
    ]

    print(
        duplicates[display_columns]
        .to_string(index=False)
    )


def print_processing_summary(
    raw_count: int,
    final_count: int,
    processed_count: int,
    duplicate_count: int,
    written_counts: dict[str, int],
) -> None:
    """
    Print a summary of the processing operation.
    """

    print("\n" + "=" * 60)
    print("Result Processing Summary")
    print("=" * 60)

    print(f"Raw runs loaded       : {raw_count:,}")
    print(f"Final runs found      : {final_count:,}")
    print(f"Duplicates removed    : {duplicate_count:,}")
    print(f"Processed runs kept   : {processed_count:,}")

    for dataset_name, count in written_counts.items():
        print(
            f"{dataset_name.capitalize():<22}: "
            f"{count:,}"
        )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Process the raw experiment results.
    """

    arguments = parse_arguments()

    input_file = Path(arguments.input_file)
    output_directory = Path(arguments.output_dir)

    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_file

    if not output_directory.is_absolute():
        output_directory = (
            PROJECT_ROOT / output_directory
        )

    print("=" * 60)
    print("Processing Experiment Results")
    print("=" * 60)

    print(
        f"Input file : "
        f"{input_file.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Output dir : "
        f"{output_directory.relative_to(PROJECT_ROOT)}"
    )

    validate_input_file(input_file)

    raw_results = pd.read_csv(input_file)

    validate_required_columns(raw_results)
    validate_missing_values(raw_results)

    normalized_results = normalize_results(
        raw_results
    )

    final_results = keep_final_runs(
        normalized_results
    )

    if final_results.empty:
        raise ValueError(
            "No final experiment runs were found. "
            "Run experiments using '--run-type final' "
            "before processing the results."
        )

    (
        deduplicated_results,
        removed_duplicates,
    ) = remove_duplicate_experiments(
        final_results
    )

    processed_results = sort_processed_results(
        deduplicated_results
    )

    processed_results = format_output_values(
        processed_results
    )

    written_counts = write_dataset_results(
        processed_results,
        output_directory,
    )

    write_removed_duplicates(
        removed_duplicates,
        output_directory,
    )

    print_duplicate_summary(
        removed_duplicates
    )

    print_processing_summary(
        raw_count=len(raw_results),
        final_count=len(final_results),
        processed_count=len(processed_results),
        duplicate_count=len(removed_duplicates),
        written_counts=written_counts,
    )

    print("\nExperiment results processed successfully.")


if __name__ == "__main__":
    main()