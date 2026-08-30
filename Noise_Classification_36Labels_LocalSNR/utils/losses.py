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


class MultiTaskNoiseSNRLoss(BaseLoss):
    """Clip-level BCE plus masked, normalized local-SNR regression loss."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        snr_weight: float = 0.1,
        target_offset_db: float = 5.0,
        target_scale_db: float = 15.0,
        regression: str = "huber",
    ) -> None:
        super().__init__()
        if regression not in {"mse", "huber"}:
            raise ValueError(f"Unsupported SNR regression loss: {regression}")
        self.classification = MultiLabelBCELoss(pos_weight=pos_weight)
        self.snr_weight = float(snr_weight)
        self.target_offset_db = float(target_offset_db)
        self.target_scale_db = float(target_scale_db)
        self.regression = regression

    def components(self, output_dict: dict, target_dict: dict) -> dict[str, torch.Tensor]:
        classification_loss = self.classification(output_dict, target_dict)
        predicted = output_dict["local_snr_normalized"]
        target = (
            target_dict["local_snr_db"].to(torch.float32) - self.target_offset_db
        ) / self.target_scale_db
        mask = target_dict["local_snr_mask"].to(torch.bool)
        if predicted.shape != target.shape:
            raise ValueError(
                f"Local SNR prediction/target shapes differ: {predicted.shape} vs {target.shape}"
            )
        if self.regression == "mse":
            element_loss = F.mse_loss(predicted, target, reduction="none")
        else:
            element_loss = F.smooth_l1_loss(predicted, target, reduction="none")
        valid = mask.to(element_loss.dtype)
        snr_loss = (element_loss * valid).sum() / valid.sum().clamp_min(1.0)
        total = classification_loss + self.snr_weight * snr_loss
        return {"total": total, "classification": classification_loss, "snr": snr_loss}

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return self.components(output_dict, target_dict)["total"]


class ClipCELoss(BaseLoss):
    """Legacy single-label loss retained only for old notebooks."""

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        return F.cross_entropy(output_dict["clipwise_output"], target_dict["target"])


# Backwards-compatible name now points to the correct logits-based implementation.
ClipBCELoss = MultiLabelBCELoss
