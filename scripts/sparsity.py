from __future__ import annotations

import random

import pandas as pd


def _get_column_names(interactions: pd.DataFrame) -> tuple[str, str]:
    """
    Detect the user ID and timestamp column names used by RecBole.
    """

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

    return user_column, timestamp_column


def _validate_retention(retention: float) -> None:
    """
    Validate the requested retention level.
    """

    if not 0 < retention <= 1:
        raise ValueError(
            "Retention must be between 0 and 1."
        )


def apply_global_sparsity(
    interactions: pd.DataFrame,
    retention: float,
    seed: int = 2025,
) -> pd.DataFrame:
    """
    Apply Global Sparsity.

    Randomly retain the requested percentage of each user's
    interactions while preserving chronological order.
    """

    _validate_retention(retention)

    user_column, timestamp_column = _get_column_names(
        interactions
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

    return (
        sparsified
        .sort_values(
            [user_column, timestamp_column]
        )
        .reset_index(drop=True)
    )


def apply_recent_history_sparsity(
    interactions: pd.DataFrame,
    retention: float,
) -> pd.DataFrame:
    """
    Apply Recent History Sparsity.

    Retain the most recent interactions for each user.
    """

    _validate_retention(retention)

    user_column, timestamp_column = _get_column_names(
        interactions
    )

    retained_groups = []

    grouped = interactions.groupby(
        user_column,
        sort=False,
    )

    for _, user_history in grouped:

        user_history = user_history.sort_values(
            timestamp_column
        )

        interaction_count = len(user_history)

        retained_count = max(
            1,
            round(interaction_count * retention),
        )

        retained_user_history = user_history.tail(
            retained_count
        )

        retained_groups.append(
            retained_user_history
        )

    sparsified = pd.concat(
        retained_groups,
        ignore_index=True,
    )

    return (
        sparsified
        .sort_values(
            [user_column, timestamp_column]
        )
        .reset_index(drop=True)
    )


def apply_early_profile_sparsity(
    interactions: pd.DataFrame,
    retention: float,
) -> pd.DataFrame:
    """
    Apply Early Profile Sparsity.

    Retain the earliest interactions for each user.
    """

    _validate_retention(retention)

    user_column, timestamp_column = _get_column_names(
        interactions
    )

    retained_groups = []

    grouped = interactions.groupby(
        user_column,
        sort=False,
    )

    for _, user_history in grouped:

        user_history = user_history.sort_values(
            timestamp_column
        )

        interaction_count = len(user_history)

        retained_count = max(
            1,
            round(interaction_count * retention),
        )

        retained_user_history = user_history.head(
            retained_count
        )

        retained_groups.append(
            retained_user_history
        )

    sparsified = pd.concat(
        retained_groups,
        ignore_index=True,
    )

    return (
        sparsified
        .sort_values(
            [user_column, timestamp_column]
        )
        .reset_index(drop=True)
    )