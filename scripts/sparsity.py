from __future__ import annotations

import random

import pandas as pd


def apply_global_sparsity(
    interactions: pd.DataFrame,
    retention: float,
    seed: int = 2025,
) -> pd.DataFrame:
    """
    Apply per-user global sparsity.

    Randomly retain the requested percentage of each user's
    interactions while preserving chronological order.
    """

    if not 0 < retention <= 1:
        raise ValueError(
            "Retention must be between 0 and 1."
        )

    # Automatically detect RecBole column names
    user_column = next(
        column
        for column in interactions.columns
        if column.startswith("user_id")
    )

    timestamp_column = next(
        column
        for column in interactions.columns
        if column.startswith("timestamp")
    )

    rng = random.Random(seed)

    retained_groups = []

    grouped = interactions.groupby(
        user_column,
        sort=False,
    )

    for _, user_history in grouped:

        interaction_count = len(user_history)

        retained_count = max(
            1,
            round(interaction_count * retention),
        )

        retained_indices = rng.sample(
            list(user_history.index),
            retained_count,
        )

        retained_user_history = (
            user_history.loc[retained_indices]
            .sort_values(timestamp_column)
        )

        retained_groups.append(
            retained_user_history
        )

    sparsified = pd.concat(
        retained_groups,
        ignore_index=True,
    )

    sparsified = (
        sparsified
        .sort_values(
            [user_column, timestamp_column]
        )
        .reset_index(drop=True)
    )

    return sparsified