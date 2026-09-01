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
    """Softmax cross-entropy for one-label-per-clip data.

    Targets arrive as the multi-hot rows the manifest defines. Every clip in this
    dataset carries exactly one label, so a row is a one-hot vector; mixup turns it
    into a convex combination of two of them. Both are handled by normalising the
    row into a distribution and taking the cross-entropy against it, which reduces
    to the usual index form when the row is one-hot.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing = float(label_smoothing)

    def forward(self, output_dict: dict, target_dict: dict) -> torch.Tensor:
        logits = output_dict["clipwise_output"]
        target = target_dict["target"].to(logits.dtype)
        distribution = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if self.label_smoothing > 0.0:
            classes = logits.size(1)
            distribution = (
                distribution * (1.0 - self.label_smoothing) + self.label_smoothing / classes
            )
        log_probability = F.log_softmax(logits, dim=1)
        return -(distribution * log_probability).sum(dim=1).mean()


def mixup_batch(
    waveform: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convex-combine a batch with a shuffled copy of itself.

    Mixup is the regulariser that stops the classifier memorising individual noise
    stems: the same waveform is never presented twice at the same weight, and the
    target it must predict moves with it.
    """
    if alpha <= 0.0 or waveform.size(0) < 2:
        return waveform, target
    weight = float(torch.distributions.Beta(alpha, alpha).sample())
    weight = max(weight, 1.0 - weight)
    permutation = torch.randperm(waveform.size(0), device=waveform.device)
    mixed_waveform = weight * waveform + (1.0 - weight) * waveform[permutation]
    mixed_target = weight * target + (1.0 - weight) * target[permutation]
    return mixed_waveform, mixed_target


class MultiTaskNoiseSNRLoss(BaseLoss):
    """Clip-level BCE plus masked, normalized local-SNR regression loss."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        snr_weight: float = 0.1,
        target_offset_db: float = 5.0,
        target_scale_db: float = 15.0,
        regression: str = "huber",
        classification: str = "ce",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if regression not in {"mse", "huber"}:
            raise ValueError(f"Unsupported SNR regression loss: {regression}")
        if classification not in {"ce", "bce"}:
            raise ValueError(f"Unsupported classification loss: {classification}")
        self.classification_kind = classification
        self.classification = (
            SingleLabelCELoss(label_smoothing=label_smoothing)
            if classification == "ce"
            else MultiLabelBCELoss(pos_weight=pos_weight)
        )
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
