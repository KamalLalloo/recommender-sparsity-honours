"""
Regression tests for sparse-training-data construction.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sparsity import (  # noqa: E402
    apply_early_profile_sparsity,
    apply_global_sparsity,
    apply_recent_history_sparsity,
)


def make_interactions(lengths=(10, 12, 15)) -> pd.DataFrame:
    """
    Build deterministic per-user chronological interaction histories.
    """

    rows = []

    for user_number, history_length in enumerate(lengths, start=1):
        user_id = f"user_{user_number}"

        for sequence_order in range(1, history_length + 1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": (
                        f"item_{user_number}_{sequence_order}"
                    ),
                    "timestamp": (
                        1_700_000_000
                        + user_number * 1_000
                        + sequence_order
                    ),
                    "sequence_order": sequence_order,
                }
            )

    return pd.DataFrame(rows)


def interaction_keys(dataframe: pd.DataFrame) -> set[tuple[str, int]]:
    """
    Return stable interaction identities for subset checks.
    """

    return set(
        zip(
            dataframe["user_id"],
            dataframe["sequence_order"],
        )
    )


def user_keys(dataframe: pd.DataFrame) -> dict[str, set[int]]:
    """
    Return retained sequence_order values by user.
    """

    return {
        user_id: set(group["sequence_order"])
        for user_id, group in dataframe.groupby(
            "user_id",
            sort=False,
        )
    }


def assert_chronological(dataframe: pd.DataFrame) -> None:
    """
    Assert that output order is deterministic and chronological.
    """

    expected = dataframe.sort_values(
        ["user_id", "sequence_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        dataframe.reset_index(drop=True),
        expected,
    )


def test_random_sparsity_nested():
    """
    Random per-user sparsity must preserve nested evaluation populations.
    """

    baseline = make_interactions()

    sparse_50 = apply_global_sparsity(
        baseline,
        0.50,
        seed=2025,
    )
    sparse_25 = apply_global_sparsity(
        baseline,
        0.25,
        seed=2025,
    )
    sparse_10 = apply_global_sparsity(
        baseline,
        0.10,
        seed=2025,
    )

    keys_50 = user_keys(sparse_50)
    keys_25 = user_keys(sparse_25)
    keys_10 = user_keys(sparse_10)
    baseline_keys = user_keys(baseline)

    for user_id in baseline_keys:
        assert keys_10[user_id] <= keys_25[user_id]
        assert keys_25[user_id] <= keys_50[user_id]
        assert keys_50[user_id] <= baseline_keys[user_id]

    assert_chronological(sparse_50)
    assert_chronological(sparse_25)
    assert_chronological(sparse_10)


def test_random_sparsity_reproducible():
    """
    The same sparse-data seed must produce identical random retention.
    """

    baseline = make_interactions(
        lengths=(20, 20, 20, 20, 20)
    )

    first = apply_global_sparsity(
        baseline,
        0.50,
        seed=2025,
    )
    second = apply_global_sparsity(
        baseline,
        0.50,
        seed=2025,
    )
    different_seed = apply_global_sparsity(
        baseline,
        0.50,
        seed=2026,
    )

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(different_seed)


def test_recent_sparsity_nested():
    """
    Recent-history sparsity must retain nested most-recent histories.
    """

    baseline = make_interactions(
        lengths=(10, 12)
    )

    sparse_50 = apply_recent_history_sparsity(
        baseline,
        0.50,
    )
    sparse_25 = apply_recent_history_sparsity(
        baseline,
        0.25,
    )
    sparse_10 = apply_recent_history_sparsity(
        baseline,
        0.10,
    )

    keys_50 = user_keys(sparse_50)
    keys_25 = user_keys(sparse_25)
    keys_10 = user_keys(sparse_10)
    baseline_keys = user_keys(baseline)

    for user_id in baseline_keys:
        assert keys_10[user_id] <= keys_25[user_id]
        assert keys_25[user_id] <= keys_50[user_id]
        assert keys_50[user_id] <= baseline_keys[user_id]

    assert list(
        sparse_50.loc[
            sparse_50["user_id"] == "user_1",
            "sequence_order",
        ]
    ) == [6, 7, 8, 9, 10]
    assert list(
        sparse_25.loc[
            sparse_25["user_id"] == "user_1",
            "sequence_order",
        ]
    ) == [8, 9, 10]
    assert list(
        sparse_10.loc[
            sparse_10["user_id"] == "user_1",
            "sequence_order",
        ]
    ) == [10]


def test_early_sparsity_nested():
    """
    Early-profile sparsity must retain nested earliest histories.
    """

    baseline = make_interactions(
        lengths=(10, 12)
    )

    sparse_50 = apply_early_profile_sparsity(
        baseline,
        0.50,
    )
    sparse_25 = apply_early_profile_sparsity(
        baseline,
        0.25,
    )
    sparse_10 = apply_early_profile_sparsity(
        baseline,
        0.10,
    )

    keys_50 = user_keys(sparse_50)
    keys_25 = user_keys(sparse_25)
    keys_10 = user_keys(sparse_10)
    baseline_keys = user_keys(baseline)

    for user_id in baseline_keys:
        assert keys_10[user_id] <= keys_25[user_id]
        assert keys_25[user_id] <= keys_50[user_id]
        assert keys_50[user_id] <= baseline_keys[user_id]

    assert list(
        sparse_50.loc[
            sparse_50["user_id"] == "user_1",
            "sequence_order",
        ]
    ) == [1, 2, 3, 4, 5]
    assert list(
        sparse_25.loc[
            sparse_25["user_id"] == "user_1",
            "sequence_order",
        ]
    ) == [1, 2, 3]
    assert list(
        sparse_10.loc[
            sparse_10["user_id"] == "user_1",
            "sequence_order",
        ]
    ) == [1]


def test_every_user_retains_training_history():
    """
    Severe sparsity must never remove an evaluated user from training.
    """

    baseline = make_interactions(
        lengths=(1, 2, 3, 10)
    )
    expected_users = set(baseline["user_id"])

    sparse_outputs = [
        apply_global_sparsity(
            baseline,
            0.10,
            seed=2025,
        ),
        apply_recent_history_sparsity(
            baseline,
            0.10,
        ),
        apply_early_profile_sparsity(
            baseline,
            0.10,
        ),
    ]

    for sparse in sparse_outputs:
        assert set(sparse["user_id"]) == expected_users
        assert (
            sparse.groupby("user_id").size() >= 1
        ).all()


def test_100_percent_equals_baseline():
    """
    The 100% sparsity control must leave training data unchanged.
    """

    baseline = make_interactions()
    expected = baseline.reset_index(drop=True)

    sparse_outputs = [
        apply_global_sparsity(
            baseline,
            1.0,
            seed=2025,
        ),
        apply_recent_history_sparsity(
            baseline,
            1.0,
        ),
        apply_early_profile_sparsity(
            baseline,
            1.0,
        ),
    ]

    for sparse in sparse_outputs:
        pd.testing.assert_frame_equal(
            sparse,
            expected,
        )
