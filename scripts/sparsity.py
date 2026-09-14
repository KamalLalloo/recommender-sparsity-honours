from __future__ import annotations

import hashlib
import math
import random

import pandas as pd


def _get_column_names(
    interactions: pd.DataFrame,
) -> tuple[str, str]:
  

    user_column = next(
        (
            column
            for column in interactions.columns
            if column.startswith("user_id")
        ),
        None,
    )

    order_column = next(
        (
            column
            for column in interactions.columns
            if column.startswith("sequence_order")
        ),
        None,
    )

    if order_column is None:
        order_column = next(
            (
                column
                for column in interactions.columns
                if column.startswith("timestamp")
            ),
            None,
        )

    if user_column is None:
        raise ValueError(
            "Could not find a user_id column."
        )

    if order_column is None:
        raise ValueError(
            "Could not find a sequence_order or timestamp column."
        )

    return user_column, order_column


def _validate_retention(retention: float) -> None:
  

    if not 0 < retention <= 1:
        raise ValueError(
            "Retention must be between 0 and 1."
        )


def _get_retained_count(
    interaction_count: int,
    retention: float,
) -> int:
   

    return max(
        1,
        math.ceil(
            interaction_count * retention
        ),
    )


def _get_user_seed(
    seed: int,
    user_id,
) -> int:
    """
    Derive a stable user-specific seed.
    """

    payload = f"{seed}:{user_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def apply_global_sparsity(
    interactions: pd.DataFrame,
    retention: float,
    seed: int = 2025,
) -> pd.DataFrame:
    """
    Apply random per-user interaction retention.
    """

    _validate_retention(retention)

    if retention == 1.0:
        return interactions.copy().reset_index(drop=True)

    user_column, order_column = _get_column_names(
        interactions
    )

    retained_groups = []

    for user_id, user_history in interactions.groupby(
        user_column,
        sort=False,
    ):
        user_history = user_history.sort_values(
            order_column,
            kind="mergesort",
        )

        retained_count = _get_retained_count(
            len(user_history),
            retention,
        )

        shuffled_indices = list(user_history.index)

        random.Random(
            _get_user_seed(
                seed,
                user_id,
            )
        ).shuffle(shuffled_indices)

        retained_groups.append(
            user_history
            .loc[
                shuffled_indices[:retained_count]
            ]
            .sort_values(
                order_column,
                kind="mergesort",
            )
        )

    return (
        pd.concat(
            retained_groups,
            ignore_index=True,
        )
        .sort_values(
            [user_column, order_column],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def apply_recent_history_sparsity(
    interactions: pd.DataFrame,
    retention: float,
) -> pd.DataFrame:
    

    _validate_retention(retention)

    if retention == 1.0:
        return interactions.copy().reset_index(drop=True)

    user_column, order_column = _get_column_names(
        interactions
    )

    retained_groups = []

    for _, user_history in interactions.groupby(
        user_column,
        sort=False,
    ):
        user_history = user_history.sort_values(
            order_column,
            kind="mergesort",
        )

        retained_groups.append(
            user_history.tail(
                _get_retained_count(
                    len(user_history),
                    retention,
                )
            )
        )

    return (
        pd.concat(
            retained_groups,
            ignore_index=True,
        )
        .sort_values(
            [user_column, order_column],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def apply_early_profile_sparsity(
    interactions: pd.DataFrame,
    retention: float,
) -> pd.DataFrame:
   
    _validate_retention(retention)

    if retention == 1.0:
        return interactions.copy().reset_index(drop=True)

    user_column, order_column = _get_column_names(
        interactions
    )

    retained_groups = []

    for _, user_history in interactions.groupby(
        user_column,
        sort=False,
    ):
        user_history = user_history.sort_values(
            order_column,
            kind="mergesort",
        )

        retained_groups.append(
            user_history.head(
                _get_retained_count(
                    len(user_history),
                    retention,
                )
            )
        )

    return (
        pd.concat(
            retained_groups,
            ignore_index=True,
        )
        .sort_values(
            [user_column, order_column],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
