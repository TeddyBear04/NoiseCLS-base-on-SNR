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


class SingleLabelCELoss(BaseLoss):
    """Softmax cross-entropy over the competing labels.

    Every clip carries exactly one label, so the 36 outputs compete instead of
    being 36 independent decisions. The loader stores the label as a float
    one-hot row, which ``F.cross_entropy`` accepts directly as a probability
    target, so nothing upstream has to change. A 1-D tensor of class indices
    also works, which is what the older notebooks pass.
    """

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        target = target_dict["target"]
        if target.ndim > 1:
            target = target.to(torch.float32)
        return F.cross_entropy(output_dict["clipwise_output"], target)


# Backwards-compatible names for notebooks written against the earlier API.
ClipCELoss = SingleLabelCELoss
ClipBCELoss = MultiLabelBCELoss
