import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        raise NotImplementedError


class MultiLabelBCELoss(BaseLoss):
    """Numerically stable BCE applied independently to each raw logit."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().clone())
        else:
            self.pos_weight = None

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            output_dict["clipwise_output"],
            target_dict["target"].to(torch.float32),
            pos_weight=self.pos_weight,
        )


class ClipCELoss(BaseLoss):
    """Legacy single-label loss retained only for old notebooks."""

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.cross_entropy(output_dict["clipwise_output"], target_dict["target"])


# Backwards-compatible name now points to the correct logits-based implementation.
ClipBCELoss = MultiLabelBCELoss
