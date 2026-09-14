from pathlib import Path

import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movielens"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "recbole"
    / "movielens"
    / "baseline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



def load_dataset() -> pd.DataFrame:
    

    return pd.read_csv(
        PROCESSED_DATA / "03_temporal_interactions.csv"
    )



def validate_dataset(df: pd.DataFrame) -> None:
   

    print("\nValidating dataset...")

    assert not df.empty, (
        "Dataset is empty."
    )

    assert df.isnull().sum().sum() == 0, (
        "Dataset contains missing values."
    )

    assert df.duplicated().sum() == 0, (
        "Dataset contains duplicate rows."
    )

    required_columns = [
        "user_id",
        "movie_id",
        "timestamp",
        "sequence_order",
    ]

    assert list(df.columns) == required_columns, (
        "Dataset contains unexpected columns."
    )

    assert pd.api.types.is_numeric_dtype(
        df["sequence_order"]
    ), (
        "sequence_order must be numeric."
    )

    assert not df.duplicated(
        subset=["user_id", "sequence_order"]
    ).any(), (
        "Dataset contains duplicate "
        "(user_id, sequence_order) pairs."
    )

    for user_id, user_df in df.groupby("user_id"):
        ordered_sequence = user_df.sort_values(
            "sequence_order",
            kind="mergesort",
        )["sequence_order"].values

        assert (
            ordered_sequence[:-1] < ordered_sequence[1:]
        ).all(), (
            f"sequence_order is not strictly increasing "
            f"for user {user_id}."
        )

    print("Dataset validation passed.")



def convert_to_recbole(df: pd.DataFrame) -> pd.DataFrame:
    
    converted = df.rename(
        columns={
            "user_id": "user_id:token",
            "movie_id": "item_id:token",
            "timestamp": "timestamp:float",
            "sequence_order": "sequence_order:float",
        }
    )

    return converted[
        [
            "user_id:token",
            "item_id:token",
            "timestamp:float",
            "sequence_order:float",
        ]
    ]



def export_dataset(df: pd.DataFrame) -> None:
    """
    Export the sequential interaction dataset.
    """

    output_file = OUTPUT_DIR / "movielens.inter"

    df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print(f"\nSaved: {output_file}")



def print_summary(df: pd.DataFrame) -> None:
   

    print("\nExport Summary")
    print("-" * 40)

    print(
        f"Users        : {df['user_id:token'].nunique():,}"
    )

    print(
        f"Items        : {df['item_id:token'].nunique():,}"
    )

    print(
        f"Interactions : {len(df):,}"
    )



def main() -> None:

    print("=" * 60)
    print("Preparing MovieLens Dataset for RecBole")
    print("=" * 60)

    dataset = load_dataset()

    validate_dataset(dataset)

    dataset = convert_to_recbole(dataset)

    export_dataset(dataset)

    print_summary(dataset)

    print(
        "\nRecBole dataset preparation completed successfully."
    )


if __name__ == "__main__":
    main()
