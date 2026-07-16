"""
Train and Evaluate a RecBole Recommender Model

This script runs a complete RecBole experiment using the
predefined MovieLens train, validation, and test files.

Current experiment
------------------
Dataset: MovieLens-1M
Model: Pop

Pipeline
--------
1. Load configuration
2. Initialise reproducibility settings
3. Create the RecBole dataset
4. Create train, validation, and test DataLoaders
5. Initialise the recommendation model
6. Initialise the appropriate RecBole trainer
7. Fit the model
8. Evaluate the model on the test set
9. Display validation and test metrics
"""

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
# Experiment Constants
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Pop"
DATASET_NAME = "movielens"

CONFIG_FILES = [
    PROJECT_ROOT / "configs" / "movielens" / "dataset.yaml",
    PROJECT_ROOT / "configs" / "movielens" / "evaluation.yaml",
    PROJECT_ROOT / "configs" / "movielens" / "pop.yaml",
]


# ==========================================================
# Configuration
# ==========================================================

def validate_config_files() -> None:
    """
    Confirm that every required configuration file exists.
    """

    print("\nValidating configuration files...")

    missing_files = [
        path for path in CONFIG_FILES
        if not path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "The following configuration files are missing:\n"
            f"{missing_text}"
        )

    for path in CONFIG_FILES:
        print(f"Found: {path.relative_to(PROJECT_ROOT)}")

    print("Configuration file validation passed.")


def create_config() -> Config:
    """
    Create the RecBole configuration object.
    """

    print("\nLoading RecBole configuration...")

    config = Config(
        model=MODEL_NAME,
        dataset=DATASET_NAME,
        config_file_list=[
            str(path) for path in CONFIG_FILES
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
    """

    print("\nPreparing train, validation, and test DataLoaders...")

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

    print(f"\nInitialising {config['model']} model...")

    model_class = get_model(config["model"])

    # RecBole 1.2.1's own quick-start implementation passes
    # the dataset stored by the training DataLoader.
    model = model_class(
        config,
        train_data._dataset,
    ).to(config["device"])

    print(f"{config['model']} model initialised successfully.")

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

    saved=False is used during initial pipeline validation so
    that no checkpoint is required yet.
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
    print(f"Training time: {training_time:.3f} seconds")

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
    print(f"Evaluation time: {evaluation_time:.3f} seconds")

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
            formatted_value = f"{float(metric_value):.6f}"
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
        valid_score_text = f"{float(best_valid_score):.6f}"
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
    print(f"Training time   : {training_time:.3f} seconds")
    print(f"Evaluation time : {evaluation_time:.3f} seconds")
    print(
        "Total time      : "
        f"{training_time + evaluation_time:.3f} seconds"
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Run the complete RecBole experiment.
    """

    print("=" * 60)
    print("RecBole Baseline Experiment")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_NAME}")
    print(f"Model        : {MODEL_NAME}")

    # Validate and load configuration.
    validate_config_files()
    config = create_config()

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

    print("\n" + "=" * 60)
    print("Experiment completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()