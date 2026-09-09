import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MultiTaskLossConfig


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


def relative_l1_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative waveform error so weak high-SNR noise is not ignored."""
    numerator = (estimate - target).abs().mean(dim=-1)
    denominator = target.abs().mean(dim=-1).clamp_min(1e-5)
    return (numerator / denominator).mean()


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR in dB, returned independently for every sample."""
    eps = torch.finfo(estimate.dtype).eps
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True) / (
        target.square().sum(dim=-1, keepdim=True) + eps
    )
    projection = scale * target
    residual = estimate - projection
    ratio = projection.square().sum(dim=-1) / (residual.square().sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence plus log-magnitude error at several resolutions."""

    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = (
            (256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024),
        ),
    ) -> None:
        super().__init__()
        self.resolutions = resolutions

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = estimate.new_zeros(())
        for n_fft, hop, win_length in self.resolutions:
            window = torch.hann_window(win_length, device=estimate.device, dtype=estimate.dtype)
            estimate_magnitude = torch.stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                window=window,
                return_complex=True,
            ).abs()
            target_magnitude = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                window=window,
                return_complex=True,
            ).abs()
            difference = estimate_magnitude - target_magnitude
            spectral_convergence = torch.linalg.vector_norm(
                difference.flatten(1), dim=-1
            ) / torch.linalg.vector_norm(target_magnitude.flatten(1), dim=-1).clamp_min(1e-5)
            log_magnitude = F.l1_loss(
                torch.log(estimate_magnitude.clamp_min(1e-5)),
                torch.log(target_magnitude.clamp_min(1e-5)),
            )
            total = total + spectral_convergence.mean() + log_magnitude
        return total / len(self.resolutions)


def masked_huber_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    if estimate.shape != target.shape or estimate.shape != mask.shape:
        raise ValueError(
            "Local-SNR estimate, target and mask must have identical shapes; "
            f"got {tuple(estimate.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}"
        )
    error = (estimate - target).abs()
    delta_tensor = torch.as_tensor(delta, dtype=error.dtype, device=error.device)
    loss = torch.where(
        error <= delta_tensor,
        0.5 * error.square() / delta_tensor,
        error - 0.5 * delta_tensor,
    )
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


class BlackFeatherMultiTaskLoss(nn.Module):
    """CE + supervised separation + masked Local-SNR regression."""

    def __init__(self, config: MultiTaskLossConfig) -> None:
        super().__init__()
        self.config = config
        self.mrstft = MultiResolutionSTFTLoss()

    def forward(
        self,
        output: dict,
        target: dict,
        stage: str = "joint",
    ) -> dict[str, torch.Tensor]:
        estimate_noise = output["estimated_noise"].float()
        target_noise = target["noise_waveform"].float()
        zero = estimate_noise.new_zeros(())
        mrstft = zero
        relative_l1 = zero
        mean_si_sdr = zero
        separation = zero
        if stage != "heads":
            mrstft = self.mrstft(estimate_noise, target_noise)
            relative_l1 = relative_l1_loss(estimate_noise, target_noise)
            mean_si_sdr = si_sdr(estimate_noise, target_noise).mean()
            separation = (
                self.config.mrstft_weight * mrstft
                + self.config.relative_l1_weight * relative_l1
                - self.config.sisdr_weight * mean_si_sdr
            )
        classification = zero
        local_snr = zero
        if stage == "extractor":
            total = self.config.lambda_sep * separation
        elif stage in {"heads", "joint"}:
            classification = F.cross_entropy(output["clipwise_output"], target["target"])
            local_snr = masked_huber_loss(
                output["local_snr_output"],
                target["local_snr_db"],
                target["local_snr_mask"],
                self.config.snr_huber_delta_db,
            )
            if stage == "heads":
                total = (
                    self.config.lambda_cls * classification
                    + self.config.lambda_snr * local_snr
                )
            else:
                total = (
                    self.config.lambda_cls * classification
                    + self.config.lambda_sep * separation
                    + self.config.lambda_snr * local_snr
                )
        else:
            raise ValueError(f"Unknown loss stage: {stage}")
        return {
            "loss": total,
            "classification_loss": classification,
            "separation_loss": separation,
            "mrstft_loss": mrstft,
            "relative_l1_loss": relative_l1,
            "si_sdr_db": mean_si_sdr,
            "local_snr_loss": local_snr,
        }
