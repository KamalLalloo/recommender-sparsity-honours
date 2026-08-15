"""
Constrained baseline-only RecBole model tuning.

This script evaluates small candidate grids on validation NDCG@10.
It never writes to results/raw/experiment_results.csv and does not
modify final model YAML files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from time import perf_counter

import yaml

import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEARCH_SPACE_FILE = (
    PROJECT_ROOT
    / "configs"
    / "tuning"
    / "search_spaces.yaml"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run constrained baseline-only validation tuning."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(train_model.SUPPORTED_DATASETS),
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
    )

    return parser.parse_args()


def load_search_space(
    model_name: str,
) -> list[dict]:
    """Load candidate parameter overrides for a model."""

    if not SEARCH_SPACE_FILE.is_file():
        raise FileNotFoundError(
            "Search-space file does not exist:\n"
            f"{SEARCH_SPACE_FILE}"
        )

    with open(
        SEARCH_SPACE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        spaces = yaml.safe_load(file) or {}

    model_space = spaces.get(
        model_name,
        {},
    )

    if not model_space:
        return [{}]

    keys = list(model_space)
    values = [
        model_space[key]
        for key in keys
    ]

    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def create_tuning_config(
    dataset_name: str,
    model_name: str,
    config_files: list[Path],
    dataset_directory: Path,
    use_gpu: bool,
    seed: int | None,
    overrides: dict,
) -> Config:
    """Create a RecBole config for one candidate."""

    from recbole.config import Config

    config_dict = {
        "use_gpu": use_gpu,
        **overrides,
    }

    if seed is not None:
        config_dict["seed"] = seed

    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=[
            str(path)
            for path in config_files
        ],
        config_dict=config_dict,
    )

    config["data_path"] = str(
        dataset_directory.resolve()
    )

    return config


def run_candidate(
    config: Config,
) -> tuple[float, dict, float]:
    """Train one candidate and return validation metrics."""

    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_logger, init_seed

    init_seed(
        config["seed"],
        config["reproducibility"],
    )

    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(
        config,
        dataset,
    )

    init_seed(
        config["seed"] + config["local_rank"],
        config["reproducibility"],
    )

    model_class = train_model.resolve_model_class(
        config["model"]
    )

    model = model_class(
        config,
        train_data._dataset,
    ).to(config["device"])

    trainer_class = get_trainer(
        config["MODEL_TYPE"],
        config["model"],
    )
    trainer = trainer_class(
        config,
        model,
    )

    start_time = perf_counter()

    _, validation_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )

    runtime = perf_counter() - start_time

    return (
        float(validation_result["ndcg@10"]),
        validation_result,
        runtime,
    )


def write_rows(
    output_file: Path,
    rows: list[dict],
) -> None:
    """Write tuning rows."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run constrained baseline tuning."""

    arguments = parse_arguments()

    dataset_name = arguments.dataset.strip().lower()
    model_name = train_model.resolve_model_name(
        arguments.model
    )

    dataset_directory = Path(
        arguments.dataset_dir
    )

    if not dataset_directory.is_absolute():
        dataset_directory = (
            PROJECT_ROOT / dataset_directory
        )

    dataset_directory = dataset_directory.resolve()

    if dataset_directory.name != "baseline":
        raise ValueError(
            "Tuning must run only on a baseline dataset "
            "directory."
        )

    train_model.validate_dataset_directory(
        dataset_directory,
        dataset_name,
    )

    config_files = train_model.build_config_files(
        dataset_name,
        model_name,
    )
    train_model.validate_config_files(
        config_files
    )

    candidates = load_search_space(
        model_name
    )

    rows = []

    print("=" * 60)
    print("Baseline Tuning")
    print("=" * 60)
    print(f"Dataset : {dataset_name}")
    print(f"Model   : {model_name}")
    print(f"Seed    : {arguments.seed}")
    print(f"Runs    : {len(candidates):,}")

    for index, overrides in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"\nCandidate {index}/{len(candidates)}: "
            f"{overrides}"
        )

        config = create_tuning_config(
            dataset_name,
            model_name,
            config_files,
            dataset_directory,
            arguments.use_gpu,
            arguments.seed,
            overrides,
        )

        score, metrics, runtime = run_candidate(
            config
        )

        row = {
            "candidate_index": index,
            "dataset": dataset_name,
            "model": model_name,
            "seed": int(config["seed"]),
            "validation_ndcg@10": score,
            "training_time_seconds": round(
                runtime,
                6,
            ),
            "parameter_overrides": repr(overrides),
        }

        for metric_name, metric_value in metrics.items():
            row[f"validation_{metric_name}"] = metric_value

        rows.append(row)

    output_file = (
        PROJECT_ROOT
        / "results"
        / "tuning"
        / dataset_name
        / f"{model_name.lower()}_tuning.csv"
    )

    write_rows(
        output_file,
        rows,
    )

    best_row = max(
        rows,
        key=lambda row: row["validation_ndcg@10"],
    )

    print("\nBest tuning result")
    print("-" * 60)
    print(
        f"Best validation NDCG@10: "
        f"{best_row['validation_ndcg@10']:.6f}"
    )
    print(
        f"Best parameter overrides: "
        f"{best_row['parameter_overrides']}"
    )
    print(
        f"Tuning results: "
        f"{output_file.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
