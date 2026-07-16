"""
Train and Evaluate a RecBole Recommender Model

This script runs a complete RecBole experiment using the
predefined MovieLens train, validation, and test files.

The model is selected through a command-line argument.

Examples
--------
python scripts/train_model.py --model Pop
python scripts/train_model.py --model ItemKNN
python scripts/train_model.py --model BPR
python scripts/train_model.py --model EASE

Pipeline
--------
1. Parse command-line arguments
2. Locate and validate configuration files
3. Load the RecBole configuration
4. Initialise reproducibility settings
5. Create the RecBole dataset
6. Create train, validation, and test DataLoaders
7. Initialise the selected recommendation model
8. Initialise the appropriate RecBole trainer
9. Fit the model
10. Evaluate the model on the test set
11. Display validation and test metrics
12. Save the experiment results to CSV
"""

import argparse
import csv
from datetime import datetime
from logging import getLogger
from pathlib import Path
from time import perf_counter

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    get_model,
    get_trainer,
    init_logger,
    init_seed,
)


# ==========================================================
# Project Constants
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "movielens"

CONFIG_DIR = (
    PROJECT_ROOT
    / "configs"
    / DATASET_NAME
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "baseline_results.csv"
)

SUPPORTED_MODELS = {
    "pop": "Pop",
    "itemknn": "ItemKNN",
    "bpr": "BPR",
    "ease": "EASE",
}


# ==========================================================
# Command-Line Arguments
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    The model name is accepted case-insensitively. For example,
    both '--model BPR' and '--model bpr' are accepted.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a RecBole recommender model "
            "on the processed MovieLens dataset."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model to run. Supported baseline models: "
            "Pop, ItemKNN, BPR, EASE."
        ),
    )

    return parser.parse_args()


def resolve_model_name(user_input: str) -> str:
    """
    Convert a case-insensitive model argument into the exact
    model name expected by RecBole.
    """

    normalised_name = user_input.strip().lower()

    if normalised_name not in SUPPORTED_MODELS:
        supported_text = ", ".join(
            SUPPORTED_MODELS.values()
        )

        raise ValueError(
            f"Unsupported model '{user_input}'. "
            f"Supported models are: {supported_text}."
        )

    return SUPPORTED_MODELS[normalised_name]


# ==========================================================
# Configuration
# ==========================================================

def build_config_files(model_name: str) -> list[Path]:
    """
    Build the list of configuration files for the selected model.

    Examples
    --------
    Pop      -> configs/movielens/pop.yaml
    ItemKNN  -> configs/movielens/itemknn.yaml
    BPR      -> configs/movielens/bpr.yaml
    EASE     -> configs/movielens/ease.yaml
    """

    model_config_filename = f"{model_name.lower()}.yaml"

    return [
        CONFIG_DIR / "dataset.yaml",
        CONFIG_DIR / "evaluation.yaml",
        CONFIG_DIR / model_config_filename,
    ]


def validate_config_files(
    config_files: list[Path]
) -> None:
    """
    Confirm that every required configuration file exists.
    """

    print("\nValidating configuration files...")

    missing_files = [
        path
        for path in config_files
        if not path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following configuration files are missing:\n"
            f"{missing_text}"
        )

    for path in config_files:
        print(
            f"Found: {path.relative_to(PROJECT_ROOT)}"
        )

    print("Configuration file validation passed.")


def create_config(
    model_name: str,
    config_files: list[Path],
) -> Config:
    """
    Create the RecBole configuration object.
    """

    print("\nLoading RecBole configuration...")

    config = Config(
        model=model_name,
        dataset=DATASET_NAME,
        config_file_list=[
            str(path)
            for path in config_files
        ],
    )

    print("RecBole configuration loaded successfully.")

    return config


# ==========================================================
# Dataset and DataLoaders
# ==========================================================

def create_recbole_dataset(config: Config):
    """
    Create the RecBole dataset from the benchmark files.
    """

    print("\nCreating RecBole dataset...")

    dataset = create_dataset(config)

    print("RecBole dataset created successfully.")

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Users        : {dataset.user_num - 1:,}")
    print(f"Items        : {dataset.item_num - 1:,}")
    print(f"Interactions : {dataset.inter_num:,}")

    return dataset


def prepare_dataloaders(config: Config, dataset):
    """
    Create train, validation, and test DataLoaders.

    Because benchmark_filename is configured, RecBole uses:
        movielens.train.inter
        movielens.valid.inter
        movielens.test.inter

    RecBole does not recreate the data split here. It loads the
    three split files that were generated during preprocessing.
    """

    print(
        "\nPreparing train, validation, "
        "and test DataLoaders..."
    )

    train_data, valid_data, test_data = data_preparation(
        config,
        dataset,
    )

    print("DataLoaders created successfully.")

    print("\nDataLoader Summary")
    print("-" * 40)
    print(f"Training batches   : {len(train_data):,}")
    print(f"Validation batches : {len(valid_data):,}")
    print(f"Test batches       : {len(test_data):,}")

    return train_data, valid_data, test_data


# ==========================================================
# Model and Trainer
# ==========================================================

def initialise_model(config: Config, train_data):
    """
    Load and initialise the configured recommendation model.
    """

    print(
        f"\nInitialising {config['model']} model..."
    )

    model_class = get_model(config["model"])

    # RecBole's quick-start implementation passes the dataset
    # stored by the training DataLoader to the model class.
    model = model_class(
        config,
        train_data._dataset,
    ).to(config["device"])

    print(
        f"{config['model']} model "
        "initialised successfully."
    )

    print("\nModel Summary")
    print("-" * 40)
    print(model)

    return model


def initialise_trainer(config: Config, model):
    """
    Select and initialise the correct RecBole trainer.
    """

    print("\nInitialising RecBole trainer...")

    trainer_class = get_trainer(
        config["MODEL_TYPE"],
        config["model"],
    )

    trainer = trainer_class(
        config,
        model,
    )

    print(
        "Trainer initialised successfully: "
        f"{trainer_class.__name__}"
    )

    return trainer


# ==========================================================
# Training and Evaluation
# ==========================================================

def train_recommender(
    config: Config,
    trainer,
    train_data,
    valid_data,
):
    """
    Fit the model and evaluate it on the validation set.

    saved=False is currently used because baseline pipeline
    validation does not require model checkpoints.
    """

    print("\nStarting model training...")

    start_time = perf_counter()

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=False,
        show_progress=config["show_progress"],
    )

    training_time = perf_counter() - start_time

    print("\nModel training completed successfully.")
    print(
        f"Training time: {training_time:.3f} seconds"
    )

    return (
        best_valid_score,
        best_valid_result,
        training_time,
    )


def evaluate_recommender(
    config: Config,
    trainer,
    test_data,
):
    """
    Evaluate the trained model on the test set.
    """

    print("\nStarting test evaluation...")

    start_time = perf_counter()

    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )

    evaluation_time = perf_counter() - start_time

    print("\nTest evaluation completed successfully.")
    print(
        f"Evaluation time: "
        f"{evaluation_time:.3f} seconds"
    )

    return test_result, evaluation_time


# ==========================================================
# Result Display
# ==========================================================

def print_metrics(
    title: str,
    metrics,
) -> None:
    """
    Print a RecBole metric dictionary in a readable format.
    """

    print(f"\n{title}")
    print("-" * 40)

    if not metrics:
        print("No metrics were returned.")
        return

    for metric_name, metric_value in metrics.items():
        try:
            formatted_value = (
                f"{float(metric_value):.6f}"
            )
        except (TypeError, ValueError):
            formatted_value = str(metric_value)

        print(
            f"{metric_name.upper():<15}"
            f"{formatted_value}"
        )


def print_experiment_summary(
    config: Config,
    best_valid_score,
    best_valid_result,
    test_result,
    training_time: float,
    evaluation_time: float,
) -> None:
    """
    Print the final experiment summary.
    """

    print("\n" + "=" * 60)
    print("Experiment Results")
    print("=" * 60)

    print(f"Dataset       : {config['dataset']}")
    print(f"Model         : {config['model']}")
    print(f"Device        : {config['device']}")
    print(f"Seed          : {config['seed']}")
    print(f"Valid metric  : {config['valid_metric']}")

    try:
        valid_score_text = (
            f"{float(best_valid_score):.6f}"
        )
    except (TypeError, ValueError):
        valid_score_text = str(best_valid_score)

    print(f"Best valid score: {valid_score_text}")

    print_metrics(
        "Best Validation Metrics",
        best_valid_result,
    )

    print_metrics(
        "Test Metrics",
        test_result,
    )

    print("\nRuntime")
    print("-" * 40)
    print(
        f"Training time   : "
        f"{training_time:.3f} seconds"
    )
    print(
        f"Evaluation time : "
        f"{evaluation_time:.3f} seconds"
    )
    print(
        "Total time      : "
        f"{training_time + evaluation_time:.3f} seconds"
    )


# ==========================================================
# Result Saving
# ==========================================================

def convert_metric_values(metrics) -> dict:
    """
    Convert RecBole metric values into normal Python values.

    RecBole may return NumPy scalar values. Converting them to
    floats ensures that the values can be written cleanly to CSV.
    """

    converted_metrics = {}

    if not metrics:
        return converted_metrics

    for metric_name, metric_value in metrics.items():
        try:
            converted_metrics[metric_name] = float(
                metric_value
            )
        except (TypeError, ValueError):
            converted_metrics[metric_name] = (
                str(metric_value)
            )

    return converted_metrics


def save_experiment_results(
    config: Config,
    best_valid_score,
    best_valid_result,
    test_result,
    training_time: float,
    evaluation_time: float,
) -> None:
    """
    Append one completed experiment to baseline_results.csv.

    Validation and test metrics are given prefixes so that their
    meanings remain clear in the results file.

    Examples
    --------
    validation_ndcg@10
    test_recall@10
    test_mrr@20
    """

    print("\nSaving experiment results...")

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_metrics = convert_metric_values(
        best_valid_result
    )

    test_metrics = convert_metric_values(
        test_result
    )

    result_row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "seed": int(config["seed"]),
        "device": str(config["device"]),
        "valid_metric": str(config["valid_metric"]),
        "best_valid_score": float(best_valid_score),
        "training_time_seconds": round(
            training_time,
            6,
        ),
        "evaluation_time_seconds": round(
            evaluation_time,
            6,
        ),
        "total_time_seconds": round(
            training_time + evaluation_time,
            6,
        ),
    }

    for metric_name, metric_value in validation_metrics.items():
        result_row[
            f"validation_{metric_name}"
        ] = metric_value

    for metric_name, metric_value in test_metrics.items():
        result_row[
            f"test_{metric_name}"
        ] = metric_value

    file_has_content = (
        RESULTS_FILE.exists()
        and RESULTS_FILE.stat().st_size > 0
    )

    if file_has_content:
        with open(
            RESULTS_FILE,
            "r",
            newline="",
            encoding="utf-8",
        ) as existing_file:
            reader = csv.DictReader(existing_file)
            existing_fieldnames = reader.fieldnames

        if existing_fieldnames != list(result_row.keys()):
            raise ValueError(
                "The existing baseline_results.csv header "
                "does not match the current result format. "
                "Because the file was manually created and "
                "should currently be empty, clear its contents "
                "and run the experiment again."
            )

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(result_row.keys()),
        )

        if not file_has_content:
            writer.writeheader()

        writer.writerow(result_row)

    print(
        "Experiment results saved successfully."
    )
    print(
        f"Results file: "
        f"{RESULTS_FILE.relative_to(PROJECT_ROOT)}"
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Run the complete RecBole experiment.
    """

    arguments = parse_arguments()

    model_name = resolve_model_name(
        arguments.model
    )

    config_files = build_config_files(
        model_name
    )

    print("=" * 60)
    print("RecBole Baseline Experiment")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_NAME}")
    print(f"Model        : {model_name}")

    # Validate and load configuration.
    validate_config_files(config_files)

    config = create_config(
        model_name,
        config_files,
    )

    # Initialise deterministic random seeds before dataset creation.
    init_seed(
        config["seed"],
        config["reproducibility"],
    )

    # Initialise RecBole logging.
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # Create the dataset and DataLoaders.
    dataset = create_recbole_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = prepare_dataloaders(
        config,
        dataset,
    )

    # Reset model-related randomness before model initialisation.
    init_seed(
        config["seed"] + config["local_rank"],
        config["reproducibility"],
    )

    # Initialise model and trainer.
    model = initialise_model(
        config,
        train_data,
    )

    logger.info(model)

    trainer = initialise_trainer(
        config,
        model,
    )

    # Train model.
    (
        best_valid_score,
        best_valid_result,
        training_time,
    ) = train_recommender(
        config,
        trainer,
        train_data,
        valid_data,
    )

    # Evaluate on test data.
    test_result, evaluation_time = evaluate_recommender(
        config,
        trainer,
        test_data,
    )

    # Display results.
    print_experiment_summary(
        config,
        best_valid_score,
        best_valid_result,
        test_result,
        training_time,
        evaluation_time,
    )

    # Save results.
    save_experiment_results(
        config,
        best_valid_score,
        best_valid_result,
        test_result,
        training_time,
        evaluation_time,
    )

    print("\n" + "=" * 60)
    print("Experiment completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()