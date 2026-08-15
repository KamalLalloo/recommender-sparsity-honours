"""
Process Raw Experiment Results.

This script validates the final experiment log, removes duplicate
configuration runs, computes robustness drops relative to matching
baselines, and writes dataset-specific processed outputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


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

SUPPORTED_DATASETS = [
    "movielens",
    "amazon",
]

DETERMINISTIC_MODELS = [
    "Pop",
    "ItemKNN",
    "EASE",
]

STOCHASTIC_MODELS = [
    "BPR",
    "NeuMF",
    "MultiVAE",
    "GRU4Rec",
    "SASRec",
    "BERT4Rec",
    "LightGCN",
]

FINAL_MODEL_SEEDS = [
    2025,
    2026,
    2027,
]

EXPECTED_CONDITIONS = [
    ("baseline", 100),
    ("global", 50),
    ("global", 25),
    ("global", 10),
    ("recent", 50),
    ("recent", 25),
    ("recent", 10),
    ("early", 50),
    ("early", 25),
    ("early", 10),
]

EXPERIMENT_TYPE_ORDER = {
    "baseline": 0,
    "global": 1,
    "recent": 2,
    "early": 3,
}

SCENARIO_OUTPUT_FILES = {
    "baseline": "baseline_results.csv",
    "global": "global_results.csv",
    "recent": "recent_results.csv",
    "early": "early_results.csv",
}

EXPERIMENT_KEY_COLUMNS = [
    "dataset",
    "experiment_type",
    "retention_level",
    "model",
    "model_seed",
    "config_sha256",
    "dataset_inter_sha256",
]

REQUIRED_COLUMNS = [
    "timestamp",
    "run_type",
    "experiment_type",
    "retention_level",
    "dataset",
    "dataset_directory",
    "model",
    "model_seed",
    "config_sha256",
    "dataset_inter_sha256",
]

REQUIRED_METRIC_COLUMNS = [
    "validation_ndcg@10",
    "test_recall@5",
    "test_recall@10",
    "test_recall@20",
    "test_hit@5",
    "test_hit@10",
    "test_hit@20",
    "test_ndcg@5",
    "test_ndcg@10",
    "test_ndcg@20",
    "test_mrr@5",
    "test_mrr@10",
    "test_mrr@20",
]

RELATIVE_DROP_METRICS = [
    "test_ndcg@10",
    "test_recall@10",
    "test_mrr@10",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Process the raw experiment log into clean, "
            "dataset-specific result files."
        )
    )

    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )

    return parser.parse_args()


def display_path(path: Path) -> str:
    """Return a project-relative path when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_input_file(input_file: Path) -> None:
    """Confirm that the raw results file exists and is not empty."""

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
    """Confirm that all required fields are present."""

    missing_columns = [
        column
        for column in (
            REQUIRED_COLUMNS
            + REQUIRED_METRIC_COLUMNS
        )
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Raw results are missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )


def normalize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize labels and numeric columns."""

    cleaned = results.copy()

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="raise",
    )

    for column in [
        "run_type",
        "dataset",
        "experiment_type",
    ]:
        cleaned[column] = (
            cleaned[column]
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

    cleaned["model_seed"] = pd.to_numeric(
        cleaned["model_seed"],
        errors="raise",
    ).astype(int)

    for column in REQUIRED_METRIC_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="raise",
        )

    return cleaned


def validate_metrics(
    results: pd.DataFrame,
) -> None:
    """Validate metrics and Recall/Hit equivalence."""

    for column in REQUIRED_METRIC_COLUMNS:
        invalid = results[
            (results[column] < 0)
            | (results[column] > 1)
        ]

        if not invalid.empty:
            raise ValueError(
                f"{column} contains values outside [0, 1]."
            )

    for k in [
        5,
        10,
        20,
    ]:
        difference = (
            results[f"test_recall@{k}"]
            - results[f"test_hit@{k}"]
        ).abs()

        if (difference > 1e-8).any():
            raise ValueError(
                f"Recall@{k} and Hit@{k} differ under "
                "single-target leave-one-out evaluation."
            )


def keep_final_runs(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only final runs."""

    final = results[
        results["run_type"] == "final"
    ].copy()

    if final.empty:
        raise ValueError(
            "No final experiment runs were found."
        )

    return final


def remove_duplicate_experiments(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the newest row per experiment identity."""

    sorted_results = results.sort_values(
        "timestamp",
        ascending=True,
    )

    duplicate_mask = sorted_results.duplicated(
        subset=EXPERIMENT_KEY_COLUMNS,
        keep="last",
    )

    return (
        sorted_results[~duplicate_mask].copy(),
        sorted_results[duplicate_mask].copy(),
    )


def validate_completeness(
    results: pd.DataFrame,
) -> None:
    """Warn about missing final experiment combinations."""

    expected_rows = []

    for dataset in SUPPORTED_DATASETS:
        for model in DETERMINISTIC_MODELS:
            for experiment_type, retention in EXPECTED_CONDITIONS:
                expected_rows.append(
                    (
                        dataset,
                        model,
                        2025,
                        experiment_type,
                        retention,
                    )
                )

        for model in STOCHASTIC_MODELS:
            for seed in FINAL_MODEL_SEEDS:
                for experiment_type, retention in EXPECTED_CONDITIONS:
                    expected_rows.append(
                        (
                            dataset,
                            model,
                            seed,
                            experiment_type,
                            retention,
                        )
                    )

    actual = set(
        zip(
            results["dataset"],
            results["model"],
            results["model_seed"],
            results["experiment_type"],
            results["retention_level"],
        )
    )

    missing = [
        row
        for row in expected_rows
        if row not in actual
    ]

    if missing:
        print(
            "WARNING: Missing expected final run combinations:"
        )
        print(
            pd.DataFrame(
                missing,
                columns=[
                    "dataset",
                    "model",
                    "model_seed",
                    "experiment_type",
                    "retention_level",
                ],
            ).to_string(index=False)
        )


def add_relative_drops(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute relative sparse-condition drops from baseline."""

    enriched = results.copy()

    for metric in RELATIVE_DROP_METRICS:
        enriched[
            f"relative_drop_{metric}"
        ] = math.nan

    baseline_lookup = {}

    for _, row in enriched[
        enriched["experiment_type"] == "baseline"
    ].iterrows():
        key = (
            row["dataset"],
            row["model"],
            row["model_seed"],
            row["config_sha256"],
        )
        baseline_lookup[key] = row

    for index, row in enriched.iterrows():
        key = (
            row["dataset"],
            row["model"],
            row["model_seed"],
            row["config_sha256"],
        )

        baseline = baseline_lookup.get(key)

        if baseline is None:
            print(
                "WARNING: Missing matching baseline for "
                f"{key}."
            )
            continue

        for metric in RELATIVE_DROP_METRICS:
            output_column = f"relative_drop_{metric}"

            if row["experiment_type"] == "baseline":
                enriched.at[index, output_column] = 0.0
                continue

            baseline_score = float(
                baseline[metric]
            )

            if baseline_score == 0:
                print(
                    "WARNING: Baseline score is zero for "
                    f"{key} {metric}; relative drop left blank."
                )
                continue

            enriched.at[index, output_column] = (
                (
                    baseline_score
                    - float(row[metric])
                )
                / baseline_score
            )

    return enriched


def sort_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Sort output rows consistently."""

    sorted_results = results.copy()
    sorted_results["_experiment_order"] = (
        sorted_results["experiment_type"]
        .map(EXPERIMENT_TYPE_ORDER)
        .fillna(999)
        .astype(int)
    )

    sorted_results = sorted_results.sort_values(
        [
            "dataset",
            "_experiment_order",
            "retention_level",
            "model",
            "model_seed",
            "timestamp",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
            True,
        ],
    )

    return sorted_results.drop(
        columns="_experiment_order"
    ).reset_index(drop=True)


def format_timestamps(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Convert timestamp objects to ISO strings."""

    formatted = results.copy()
    formatted["timestamp"] = (
        formatted["timestamp"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    return formatted


def write_outputs(
    results: pd.DataFrame,
    duplicates: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Write processed result files."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatted = format_timestamps(
        results
    )

    for dataset in SUPPORTED_DATASETS:
        dataset_dir = output_directory / dataset
        dataset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_results = formatted[
            formatted["dataset"] == dataset
        ].copy()

        dataset_results.to_csv(
            dataset_dir / "all_final_results.csv",
            index=False,
        )

        for experiment_type, filename in (
            SCENARIO_OUTPUT_FILES.items()
        ):
            dataset_results[
                dataset_results["experiment_type"]
                == experiment_type
            ].to_csv(
                dataset_dir / filename,
                index=False,
            )

        summary_metrics = [
            column
            for column in dataset_results.columns
            if (
                column.startswith("test_")
                or column.startswith("relative_drop_")
                or column
                in [
                    "training_time_seconds",
                    "evaluation_time_seconds",
                    "total_time_seconds",
                ]
            )
            and pd.api.types.is_numeric_dtype(
                results[column]
            )
        ]

        summary = (
            results[
                results["dataset"] == dataset
            ]
            .groupby(
                [
                    "dataset",
                    "experiment_type",
                    "retention_level",
                    "model",
                ],
                dropna=False,
            )[summary_metrics]
            .agg(["mean", "std", "count"])
        )

        summary.columns = [
            "_".join(column).strip("_")
            for column in summary.columns
        ]

        summary.reset_index().to_csv(
            dataset_dir / "summary_by_condition.csv",
            index=False,
        )

    if duplicates.empty:
        duplicate_file = (
            output_directory
            / "removed_duplicates.csv"
        )
        if duplicate_file.exists():
            duplicate_file.unlink()
    else:
        format_timestamps(
            duplicates
        ).to_csv(
            output_directory
            / "removed_duplicates.csv",
            index=False,
        )


def main() -> None:
    """Process the raw experiment results."""

    arguments = parse_arguments()

    input_file = Path(
        arguments.input_file
    )
    output_directory = Path(
        arguments.output_dir
    )

    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_file

    if not output_directory.is_absolute():
        output_directory = (
            PROJECT_ROOT / output_directory
        )

    input_file = input_file.resolve()
    output_directory = output_directory.resolve()

    print("=" * 60)
    print("Processing Experiment Results")
    print("=" * 60)
    print(f"Input file : {display_path(input_file)}")
    print(f"Output dir : {display_path(output_directory)}")

    validate_input_file(input_file)

    raw_results = pd.read_csv(input_file)
    validate_required_columns(raw_results)

    normalized = normalize_results(raw_results)
    validate_metrics(normalized)

    final_results = keep_final_runs(normalized)

    deduplicated, duplicates = (
        remove_duplicate_experiments(
            final_results
        )
    )

    validate_completeness(deduplicated)

    enriched = add_relative_drops(
        deduplicated
    )

    processed = sort_results(
        enriched
    )

    write_outputs(
        processed,
        duplicates,
        output_directory,
    )

    print("\nResult processing summary")
    print("-" * 60)
    print(f"Raw rows loaded    : {len(raw_results):,}")
    print(f"Final rows found   : {len(final_results):,}")
    print(f"Duplicates removed : {len(duplicates):,}")
    print(f"Rows kept          : {len(processed):,}")
    print(
        "\nExperiment results processed successfully."
    )


if __name__ == "__main__":
    main()
