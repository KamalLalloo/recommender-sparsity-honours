"""
Project-local BERT4Rec loss patch for RecBole 1.2.1.

RecBole's BERT4Rec uses masked sequence positions to determine
whether a masked prediction target is valid.

Sequence position 0 is a legitimate position, so the condition

    masked_index > 0

incorrectly excludes real masked targets that occur at the first
sequence position.

RecBole reserves item ID 0 for padding. Therefore positive item
IDs provide an unambiguous validity mask:

    pos_items > 0

This subclass changes only the CE training-loss target selection.
The underlying BERT4Rec architecture, forward pass, prediction
logic, masking transform, and evaluation behaviour remain inherited
from RecBole.
"""

import torch
import torch.nn.functional as F

from recbole.model.sequential_recommender.bert4rec import (
    BERT4Rec as RecBoleBERT4Rec,
)


class PatchedBERT4Rec(RecBoleBERT4Rec):
    """
    RecBole BERT4Rec with corrected masked-target validity logic.
    """

    PATCH_DESCRIPTION = (
        "BERT4Rec CE target-mask fix: "
        "use positive item IDs instead of masked sequence positions"
    )

    def calculate_loss(self, interaction):
        """
        Calculate the masked-item CE loss.

        Real masked targets are identified by non-zero positive
        item IDs. This preserves legitimate targets at sequence
        position 0 and ignores zero-padded target slots.
        """

        if self.loss_type != "CE":
            raise NotImplementedError(
                "PatchedBERT4Rec currently supports only "
                "loss_type='CE'."
            )

        masked_item_seq = interaction[
            self.MASK_ITEM_SEQ
        ]

        pos_items = interaction[
            self.POS_ITEMS
        ]

        masked_index = interaction[
            self.MASK_INDEX
        ]

        sequence_output = self.forward(
            masked_item_seq
        )

        selection_matrix = self.multi_hot_embed(
            masked_index,
            masked_item_seq.size(-1),
        )

        selection_matrix = selection_matrix.view(
            masked_index.size(0),
            masked_index.size(1),
            -1,
        )

        masked_output = torch.bmm(
            selection_matrix,
            sequence_output,
        )

        item_embeddings = (
            self.item_embedding.weight[
                : self.n_items
            ]
        )

        logits = (
            torch.matmul(
                masked_output,
                item_embeddings.transpose(
                    0,
                    1,
                ),
            )
            + self.output_bias
        )

        flat_pos_items = pos_items.reshape(-1)

        valid_targets = (
            flat_pos_items > 0
        )

        # A transformed batch can theoretically contain no
        # usable masked targets. Returning a differentiable
        # zero loss avoids division-by-zero/NaN behaviour while
        # producing no parameter update for that batch.
        if not torch.any(valid_targets):
            return logits.sum() * 0.0

        flat_logits = logits.reshape(
            -1,
            self.n_items,
        )

        valid_logits = flat_logits[
            valid_targets
        ]

        valid_pos_items = flat_pos_items[
            valid_targets
        ]

        loss = F.cross_entropy(
            valid_logits,
            valid_pos_items,
            reduction="mean",
        )

        return loss
