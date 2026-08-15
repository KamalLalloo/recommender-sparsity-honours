"""
Regression tests for the project-local BERT4Rec loss patch.
"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bert4rec_patch import PatchedBERT4Rec  # noqa: E402


def test_bert4rec_position_zero_target():
    """
    A real masked target at sequence position zero must remain valid.
    """

    masked_index = torch.tensor(
        [[0]]
    )
    pos_items = torch.tensor(
        [[42]]
    )

    valid_targets = (
        PatchedBERT4Rec._valid_target_mask(
            pos_items
        )
    )

    assert masked_index.reshape(-1)[0].item() == 0
    assert valid_targets.tolist() == [True]


def test_bert4rec_padding_target():
    """
    Item ID zero is padding and must be excluded from CE targets.
    """

    pos_items = torch.tensor(
        [[0, 7]]
    )

    valid_targets = (
        PatchedBERT4Rec._valid_target_mask(
            pos_items
        )
    )

    assert valid_targets.tolist() == [False, True]
