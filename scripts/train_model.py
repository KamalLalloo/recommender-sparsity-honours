from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from logging import getLogger
from pathlib import Path
from time import perf_counter




PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "experiment_results.csv"
)

SUPPORTED_DATASETS = {
    "movielens",
    "amazon",
}

SUPPORTED_MODELS = {
    "pop": "Pop",
    "itemknn": "ItemKNN",
    "bpr": "BPR",
    "ease": "EASE",
    "neumf": "NeuMF",
    "multivae": "MultiVAE",
    "gru4rec": "GRU4Rec",
    "sasrec": "SASRec",
    "bert4rec": "BERT4Rec",
    "lightgcn": "LightGCN",
}


def parse_arguments() -> argparse.Namespace:
   

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a RecBole recommender model "
            "on a benchmark dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help=(
            "Dataset to use. Supported datasets: "
            + ", ".join(sorted(SUPPORTED_DATASETS))
            + "."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model to run. Supported models: "
            + ", ".join(SUPPORTED_MODELS.values())
            + "."
        ),
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Dataset directory containing the unified .inter file, "
            "such as data/recbole/amazon/baseline or "
            "data/recbole/movielens/global/50."
        ),
    )

    parser.add_argument(
        "--run-type",
        choices=["development", "final"],
        default="development",
        help=(
            "Whether this is a development/test run or a final "
            "experiment run. Defaults to development."
        ),
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help=(
            "Request CUDA execution. If omitted, the experiment "
            "must run on the CPU."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional model seed override. If omitted, "
            "the seed from the YAML/RecBole config is used."
        ),
    )

    return parser.parse_args()


def resolve_model_name(user_input: str) -> str:
   
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


def validate_dataset_directory(
    dataset_directory: Path,
    dataset_name: str,
) -> None:
   

    print("\nValidating dataset directory...")

    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            "Dataset directory does not exist:\n"
            f"{dataset_directory}"
        )

    interaction_filename = f"{dataset_name}.inter"
    item_filename = f"{dataset_name}.item"

    interaction_file = (
        dataset_directory
        / interaction_filename
    )

    item_file = (
        dataset_directory
        / item_filename
    )

    if not interaction_file.is_file():
        raise FileNotFoundError(
            "Dataset directory is missing the required "
            "interaction file:\n"
            f"{interaction_file}"
        )

    if not item_file.is_file():
        raise FileNotFoundError(
            "Dataset directory is missing the required "
            "item catalogue file:\n"
            f"{item_file}"
        )

    try:
        relative_path = dataset_directory.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative_path = dataset_directory

    print(f"Dataset directory: {relative_path}")
    print(f"Interaction file : {interaction_filename}")
    print(f"Item file        : {item_filename}")
    print("Dataset format   : Unified interaction file")
    print("Dataset directory validation passed.")


def build_config_files(
    dataset_name: str,
    model_name: str,
) -> list[Path]:
    

    config_dir = (
        PROJECT_ROOT
        / "configs"
        / dataset_name
    )

    model_config_filename = (
        f"{model_name.lower()}.yaml"
    )

    return [
        config_dir / "dataset.yaml",
        config_dir / "evaluation.yaml",
        config_dir / model_config_filename,
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
    dataset_name: str,
    model_name: str,
    config_files: list[Path],
    dataset_directory: Path,
    use_gpu: bool,
    seed_override: int | None,
) -> Config:
    """
    Create the RecBole configuration object.
    """

    print("\nLoading RecBole configuration...")

    from recbole.config import Config

    config_dict = {
        "use_gpu": use_gpu,
    }

    if seed_override is not None:
        config_dict["seed"] = seed_override

    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_file_list=[
            str(path)
            for path in config_files
        ],
        config_dict=config_dict,
    )

    # Override the placeholder data_path from dataset.yaml.
    config["data_path"] = str(
        dataset_directory.resolve()
    )

    print(
        "RecBole configuration loaded successfully."
    )

    return config


def read_sparsity_metadata(
    dataset_directory: Path,
) -> dict:
    

    metadata_file = dataset_directory / "metadata.json"

    if not metadata_file.is_file():
        return {}

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    """
    Return the SHA-256 digest for one file.
    """

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def config_sha256(
    config_files: list[Path],
) -> str:
    

    digest = hashlib.sha256()

    for path in sorted(
        config_files,
        key=lambda value: str(value),
    ):
        digest.update(
            str(path.relative_to(PROJECT_ROOT)).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
        digest.update(
            file_sha256(path).encode("utf-8")
        )
        digest.update(b"\0")

    return digest.hexdigest()


def get_git_commit() -> str | None:
   

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip() or None


def get_config_value(
    config,
    key: str,
):
   

    try:
        return config[key]
    except KeyError:
        return None



def create_recbole_dataset(config: Config):
   

    print("\nCreating RecBole dataset...")

    from recbole.data import create_dataset

    dataset = create_dataset(config)

    print("RecBole dataset created successfully.")

    print("\nDataset Summary")
    print("-" * 40)
    print(f"Users        : {dataset.user_num - 1:,}")
    print(f"Items        : {dataset.item_num - 1:,}")
    print(f"Interactions : {dataset.inter_num:,}")

    return dataset


def prepare_dataloaders(config: Config, dataset):
  

    print(
        "\nPreparing train, validation, "
        "and test DataLoaders..."
    )

    from recbole.data import data_preparation

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


def resolve_model_class(
    model_name: str,
):


    if model_name == "BERT4Rec":

        from bert4rec_patch import (
            PatchedBERT4Rec,
        )

        print(
            "Using project-local patched "
            "BERT4Rec implementation."
        )

        print(
            "BERT4Rec patch: "
            f"{PatchedBERT4Rec.PATCH_DESCRIPTION}"
        )

        return PatchedBERT4Rec

    from recbole.utils import get_model

    return get_model(
        model_name
    )


def initialise_model(config: Config, train_data):
    """
    Load and initialise the configured recommendation model.
    """

    print(
        f"\nInitialising {config['model']} model..."
    )

    model_class = resolve_model_class(
        config["model"]
    )

    # RecBole's quick-start implementation passes the dataset
    # stored by the training DataLoader to the model class.
    model = model_class(
        config,
        train_data._dataset,
    ).to(config["device"])

    if config["model"] == "BERT4Rec":
        if model.__class__.__name__ != "PatchedBERT4Rec":
            raise RuntimeError(
                "BERT4Rec was requested but the "
                "project-local patched implementation "
                "was not initialised."
            )

    print(
        f"{config['model']} model "
        "initialised successfully."
    )

    print("\nModel Summary")
    print("-" * 40)
    print(model)

    return model


def initialise_trainer(config: Config, model):
    
    print("\nInitialising RecBole trainer...")

    from recbole.utils import get_trainer

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


def train_recommender(
    config: Config,
    trainer,
    train_data,
    valid_data,
):


    print("\nStarting model training...")

    start_time = perf_counter()

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
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
   

    print("\nStarting test evaluation...")

    start_time = perf_counter()

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    evaluation_time = perf_counter() - start_time

    print("\nTest evaluation completed successfully.")
    print(
        f"Evaluation time: "
        f"{evaluation_time:.3f} seconds"
    )

    return test_result, evaluation_time


def print_metrics(
    title: str,
    metrics,
) -> None:
    

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
    dataset_directory: Path,
    best_valid_score,
    best_valid_result,
    test_result,
    training_time: float,
    evaluation_time: float,
    use_gpu_requested: bool,
) -> None:
  
    print("\n" + "=" * 60)
    print("Experiment Results")
    print("=" * 60)

    print(f"Dataset       : {config['dataset']}")
    print(
    f"Dataset path  : "
    f"{dataset_directory.relative_to(PROJECT_ROOT)}"
    )
    print(f"Model         : {config['model']}")
    print(f"GPU requested : {use_gpu_requested}")
    print(f"Device        : {config['device']}")
    print(f"Model seed    : {config['seed']}")
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



def convert_metric_values(metrics) -> dict:
   

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
    dataset_directory: Path,
    run_type,
    use_gpu_requested: bool,
    best_valid_score,
    best_valid_result,
    test_result,
    training_time: float,
    evaluation_time: float,
    trainer,
    model,
    config_files: list[Path],
    sparsity_metadata: dict,
) -> None:
    """
    Append one completed experiment to experiment_results.csv.

    Validation and test metrics are given prefixes so that their
    meanings remain clear in the results file.


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

    # Determine experiment type and retention level from the dataset directory.
    relative_dataset_path = dataset_directory.relative_to(
        PROJECT_ROOT
    )

    if relative_dataset_path.name == "baseline":
        experiment_type = "Baseline"
        retention_level = 100

    else:
        experiment_type = relative_dataset_path.parent.name.capitalize()

        try:
            retention_level = int(relative_dataset_path.name)
        except ValueError as error:
            raise ValueError(
                "Could not determine the retention level from "
                f"dataset directory: {relative_dataset_path}"
            ) from error

    dataset_name = str(config["dataset"])
    interaction_file = (
        dataset_directory
        / f"{dataset_name}.inter"
    )
    item_file = (
        dataset_directory
        / f"{dataset_name}.item"
    )

    checkpoint_path = getattr(
        trainer,
        "saved_model_file",
        None,
    )

    import recbole
    import torch

    result_row = {

        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "experiment_type": experiment_type,
        "retention_level": retention_level,

        # Experiment metadata
        "dataset": str(config["dataset"]),

        "dataset_directory": str(
            dataset_directory.relative_to(PROJECT_ROOT)
        ),

        "model": str(config["model"]),

        "model_seed": int(config["seed"]),

        "sparsity_seed": sparsity_metadata.get(
            "sparsity_seed"
        ),

        "scenario": sparsity_metadata.get(
            "scenario",
            (
                "baseline"
                if experiment_type == "Baseline"
                else experiment_type.lower()
            ),
        ),

        "scenario_method": sparsity_metadata.get(
            "scenario_method"
        ),

        "requested_retention_percent":
            sparsity_metadata.get(
                "requested_retention_percent",
                retention_level,
            ),

        "actual_training_retention_fraction":
            sparsity_metadata.get(
                "actual_training_retention_fraction",
                1.0 if retention_level == 100 else None,
            ),

        "actual_training_retention_percent":
            sparsity_metadata.get(
                "actual_training_retention_percent",
                100.0 if retention_level == 100 else None,
            ),

        "recbole_version": recbole.__version__,

        "torch_version": torch.__version__,

        "python_version": sys.version.split()[0],

        "git_commit": get_git_commit(),

        "config_sha256": config_sha256(
            config_files
        ),

        "dataset_inter_sha256": file_sha256(
            interaction_file
        ),

        "dataset_item_sha256": file_sha256(
            item_file
        ),

        "bert4rec_patch_active": (
            model.__class__.__name__
            == "PatchedBERT4Rec"
        ),

        "max_item_list_length": get_config_value(
            config,
            "MAX_ITEM_LIST_LENGTH",
        ),

        "time_field": str(config["TIME_FIELD"]),

        "checkpoint_path": (
            str(checkpoint_path)
            if checkpoint_path
            else None
        ),

        "device": str(config["device"]),

        "use_gpu_requested": bool(use_gpu_requested),

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
        "run_type": run_type,
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
                "The existing experiment_results.csv header "
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



def main() -> None:
    """
    Run the complete RecBole experiment.
    """

    arguments = parse_arguments()

    dataset_name = (
        arguments.dataset
        .strip()
        .lower()
    )

    dataset_directory = Path(
        arguments.dataset_dir
    )

    if not dataset_directory.is_absolute():
        dataset_directory = (
            PROJECT_ROOT / dataset_directory
        )

    dataset_directory = dataset_directory.resolve()

    model_name = resolve_model_name(
        arguments.model
    )

    validate_dataset_directory(
        dataset_directory,
        dataset_name,
    )

    config_files = build_config_files(
        dataset_name,
        model_name,
    )

    print("=" * 60)
    print("RecBole Experiment")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {dataset_name}")
    print(f"Model        : {model_name}")
    print(f"GPU requested: {arguments.use_gpu}")
    print(
        f"Dataset dir  : "
        f"{dataset_directory.relative_to(PROJECT_ROOT)}"
    )

    # Validate and load configuration.
    validate_config_files(config_files)

    config = create_config(
        dataset_name,
        model_name,
        config_files,
        dataset_directory,
        arguments.use_gpu,
        arguments.seed,
    )

    resolved_device = str(config["device"])

    if arguments.use_gpu:
        if not resolved_device.startswith("cuda"):
            raise RuntimeError(
                "GPU execution was requested, but RecBole "
                f"resolved device '{resolved_device}'."
            )
    else:
        if resolved_device.startswith("cuda"):
            raise RuntimeError(
                "CPU execution was requested, but RecBole "
                f"resolved device '{resolved_device}'."
            )

    print(f"Resolved device: {resolved_device}")

    from recbole.utils import init_logger, init_seed

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
        dataset_directory,
        best_valid_score,
        best_valid_result,
        test_result,
        training_time,
        evaluation_time,
        arguments.use_gpu,
    )

    # Save results.
    save_experiment_results(
        config,
        dataset_directory,
        arguments.run_type,
        arguments.use_gpu,
        best_valid_score,
        best_valid_result,
        test_result,
        training_time,
        evaluation_time,
        trainer,
        model,
        config_files,
        read_sparsity_metadata(
            dataset_directory
        ),
    )

    print("\n" + "=" * 60)
    print("Experiment completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
