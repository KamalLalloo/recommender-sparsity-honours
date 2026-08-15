from __future__ import annotations

import hashlib
import math
import random

import pandas as pd


def _get_column_names(
    interactions: pd.DataFrame,
) -> tuple[str, str]:
    """
    Detect the user ID and deterministic ordering fields.

    sequence_order is preferred. timestamp is retained only as a
    backward-compatible fallback.
    """

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
    """
    Validate the requested retention level.
    """

    if not 0 < retention <= 1:
        raise ValueError(
            "Retention must be between 0 and 1."
        )


def _get_retained_count(
    interaction_count: int,
    retention: float,
) -> int:
    """
    Return the retained interaction count for one user.

    The frozen project rule is ceiling retention with a minimum
    of one training interaction per user.
    """

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

    The repository keeps the historical folder name "global",
    but the academic meaning is random per-user sparsity. For
    each user, a stable seed reconstructs the same deterministic
    shuffle at every retention level, so 10% is a subset of 25%,
    25% is a subset of 50%, and 50% is a subset of baseline.
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
    """
    Apply recent-history sparsity.

    Retain the most recent training interactions for each user.
    """

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
    """
    Apply early-profile sparsity.

    Retain the earliest training interactions for each user.
    This is cold-start-like, not genuine cold start.
    """

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
