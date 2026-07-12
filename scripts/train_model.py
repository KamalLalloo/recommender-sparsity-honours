from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILES = [

    str(PROJECT_ROOT / "configs" / "movielens" / "dataset.yaml"),

    str(PROJECT_ROOT / "configs" / "movielens" / "evaluation.yaml"),

    str(PROJECT_ROOT / "configs" / "movielens" / "pop.yaml")

]


def main():

    print("=" * 60)
    print("Testing RecBole Dataset Loading")
    print("=" * 60)

    config = Config(
        model="Pop",
        dataset="movielens",
        config_file_list=CONFIG_FILES
    )

    dataset = create_dataset(config)

    print()

    print("Dataset loaded successfully.")

    print(f"Users        : {dataset.user_num - 1}")

    print(f"Items        : {dataset.item_num - 1}")

    print(f"Interactions : {dataset.inter_num}")

    print()

    print("RecBole dataset verification completed successfully.")


if __name__ == "__main__":
    main()