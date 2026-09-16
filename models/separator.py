"""STFT-mask noise separator, noise amplification, and SI-SDR loss."""

from __future__ import annotations

import torch
from torch import nn


class DilatedBlock(nn.Module):
    """Residual depthwise-separable dilated convolution over STFT frames."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size,
                padding=padding, dilation=dilation, groups=channels,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, channels),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.net(hidden)


class NoiseSeparator(nn.Module):
    """Estimate the noise waveform ``n_hat`` from a mixture ``[batch, samples]``.

    A small temporal convolution network predicts either a magnitude mask or a
    complex ratio mask over the mixture STFT. The complex mode learns phase as
    well as magnitude corrections and is the recommended setting for retraining.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        channels: int = 256,
        blocks: int = 8,
        kernel_size: int = 3,
        target_rms: float = 0.1,
        max_gain_db: float = 40.0,
        speech_threshold_db: float = -30.0,
        speech_ratio: float = 3.0,
        speech_target_rms: float = 0.08,
        speech_max_gain_db: float = 12.0,
        speech_attack_ms: float = 10.0,
        speech_release_ms: float = 200.0,
        mask_mode: str = "magnitude",
    ) -> None:
        super().__init__()
        self.config = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "channels": channels,
            "blocks": blocks,
            "kernel_size": kernel_size,
            "target_rms": target_rms,
            "max_gain_db": max_gain_db,
            "speech_threshold_db": speech_threshold_db,
            "speech_ratio": speech_ratio,
            "speech_target_rms": speech_target_rms,
            "speech_max_gain_db": speech_max_gain_db,
            "speech_attack_ms": speech_attack_ms,
            "speech_release_ms": speech_release_ms,
            "mask_mode": mask_mode,
        }
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_rms = target_rms
        self.max_gain = 10 ** (max_gain_db / 20)
        self.speech_threshold_db = speech_threshold_db
        self.speech_ratio = speech_ratio
        self.speech_target_rms = speech_target_rms
        self.speech_max_gain = 10 ** (speech_max_gain_db / 20)
        self.speech_attack_ms = speech_attack_ms
        self.speech_release_ms = speech_release_ms
        if mask_mode not in {"magnitude", "complex"}:
            raise ValueError("mask_mode must be 'magnitude' or 'complex'")
        self.mask_mode = mask_mode
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        frequency_bins = n_fft // 2 + 1
        self.input = nn.Sequential(
            nn.Conv1d(frequency_bins, channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, channels),
        )
        self.blocks = nn.Sequential(
            *(DilatedBlock(channels, kernel_size, 2**index) for index in range(blocks))
        )
        output_bins = frequency_bins * (2 if mask_mode == "complex" else 1)
        self.mask = nn.Conv1d(channels, output_bins, 1)
        if mask_mode == "complex":
            # Start from a stable 0.5 real mask and zero imaginary correction.
            nn.init.zeros_(self.mask.weight)
            nn.init.zeros_(self.mask.bias)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # STFT/ISTFT do not support BF16; keep the whole separator in FP32.
        with torch.autocast(device_type=mixture.device.type, enabled=False):
            mixture = mixture.float()
            spectrum = torch.stft(
                mixture, self.n_fft, self.hop_length, self.win_length,
                window=self.window, center=True, return_complex=True,
            )
            features = torch.log(spectrum.abs().clamp_min(1e-5))
            logits = self.mask(self.blocks(self.input(features)))
            if self.mask_mode == "complex":
                real_logits, imaginary_logits = logits.chunk(2, dim=1)
                # The real component may exceed [0, 1], as a cIRM needs to
                # correct destructive interference in the mixture spectrum.
                mask = torch.complex(
                    2 * torch.sigmoid(real_logits) - 0.5,
                    torch.tanh(imaginary_logits),
                )
            else:
                mask = torch.sigmoid(logits)
            return torch.istft(
                spectrum * mask, self.n_fft, self.hop_length, self.win_length,
                window=self.window, center=True, length=mixture.shape[-1],
            )

    def set_amplification(self, target_rms: float | None = None, max_gain_db: float | None = None) -> None:
        """Change the parameter-free loudness normalisation without retraining."""
        if target_rms is not None:
            self.target_rms = target_rms
            self.config["target_rms"] = target_rms
        if max_gain_db is not None:
            self.max_gain = 10 ** (max_gain_db / 20)
            self.config["max_gain_db"] = max_gain_db

    def amplify(self, noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Bring every estimated noise clip to ``target_rms`` before BEATs.

        Quiet high-SNR residuals are boosted by at most ``max_gain_db`` so that
        near-silent estimates are not blown up to pure artefacts. The gain is
        detached so it acts as a fixed loudness normalisation, and the result
        is peak-limited to [-1, 1] like a regular waveform.
        """
        rms = noise.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        gain = (self.target_rms / rms).clamp(max=self.max_gain).detach()
        amplified = noise * gain
        peak = amplified.abs().amax(dim=-1, keepdim=True).detach()
        return amplified / peak.clamp(min=1.0)

    def set_speech_conditioning(
        self,
        threshold_db: float | None = None,
        ratio: float | None = None,
        target_rms: float | None = None,
        max_gain_db: float | None = None,
        attack_ms: float | None = None,
        release_ms: float | None = None,
    ) -> None:
        """Update parameter-free speech WDRC settings without retraining."""
        if ratio is not None and ratio < 1:
            raise ValueError("speech_ratio must be at least 1")
        updates = {
            "speech_threshold_db": threshold_db,
            "speech_ratio": ratio,
            "speech_target_rms": target_rms,
            "speech_max_gain_db": max_gain_db,
            "speech_attack_ms": attack_ms,
            "speech_release_ms": release_ms,
        }
        for key, value in updates.items():
            if value is not None:
                setattr(self, key, value)
                self.config[key] = value
        if max_gain_db is not None:
            self.speech_max_gain = 10 ** (max_gain_db / 20)

    def compress_and_amplify_speech(
        self, speech: torch.Tensor, sample_rate: int = 16_000, eps: float = 1e-8
    ) -> torch.Tensor:
        """Apply envelope WDRC and bounded make-up gain to a speech estimate.

        The residual ``s_hat = mixture - n_hat`` is processed only after the
        separator, avoiding AGC amplification of background noise in speech
        pauses. Ten-millisecond RMS frames use asymmetric attack/release
        smoothing, a soft compression knee, and a peak guard.
        """
        if self.speech_ratio < 1:
            raise ValueError("speech_ratio must be at least 1")
        frame = max(1, round(sample_rate * 0.010))
        length = speech.shape[-1]
        padded = torch.nn.functional.pad(speech.float(), (0, (-length) % frame))
        frames = padded.unfold(-1, frame, frame)
        level_db = 20 * torch.log10(frames.pow(2).mean(dim=-1).add(eps).sqrt())
        attack = torch.exp(torch.tensor(-frame / (sample_rate * self.speech_attack_ms / 1000), device=speech.device))
        release = torch.exp(torch.tensor(-frame / (sample_rate * self.speech_release_ms / 1000), device=speech.device))
        smoothed = torch.empty_like(level_db)
        smoothed[:, 0] = level_db[:, 0]
        for index in range(1, level_db.shape[-1]):
            coefficient = torch.where(level_db[:, index] > smoothed[:, index - 1], attack, release)
            smoothed[:, index] = coefficient * smoothed[:, index - 1] + (1 - coefficient) * level_db[:, index]
        above_knee = (smoothed - self.speech_threshold_db).clamp_min(0)
        gain_db = above_knee * (1 / self.speech_ratio - 1)
        gain = (10 ** (gain_db / 20)).repeat_interleave(frame, dim=-1)[..., :length]
        compressed = speech * gain
        compressed_rms = compressed.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        makeup = (self.speech_target_rms / compressed_rms).clamp(max=self.speech_max_gain).detach()
        amplified = compressed * makeup
        peak = amplified.abs().amax(dim=-1, keepdim=True).detach()
        return amplified / (peak / 0.98).clamp(min=1.0)


class DemucsNoiseSeparator(NoiseSeparator):
    """Frozen pretrained Demucs vocal separator used to estimate environmental noise.

    Demucs exposes a ``vocals`` stem. For this dataset that is the speech
    estimate; the desired noise estimate is the mixture residual. It runs at
    Demucs's native stereo sample rate while this project remains mono 16 kHz.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        speech_source: str = "vocals",
        input_sample_rate: int = 16_000,
        shifts: int = 0,
        overlap: float = 0.25,
        target_rms: float = 0.1,
        max_gain_db: float = 20.0,
        speech_threshold_db: float = -30.0,
        speech_ratio: float = 3.0,
        speech_target_rms: float = 0.08,
        speech_max_gain_db: float = 12.0,
        speech_attack_ms: float = 10.0,
        speech_release_ms: float = 200.0,
    ) -> None:
        # Reuse the shared amplification/WDRC implementation; the small TCN is
        # unused because ``forward`` delegates separation to Demucs.
        super().__init__(
            channels=1, blocks=0, target_rms=target_rms, max_gain_db=max_gain_db,
            speech_threshold_db=speech_threshold_db, speech_ratio=speech_ratio,
            speech_target_rms=speech_target_rms, speech_max_gain_db=speech_max_gain_db,
            speech_attack_ms=speech_attack_ms, speech_release_ms=speech_release_ms,
        )
        try:
            from demucs import pretrained
        except ImportError as error:
            raise ImportError(
                "Demucs is required for separator_backend='demucs'. "
                "Run `pip install -r requirements.txt`."
            ) from error
        self.demucs = pretrained.get_model(model_name)
        self.demucs.eval()
        for parameter in self.demucs.parameters():
            parameter.requires_grad = False
        if speech_source not in self.demucs.sources:
            raise ValueError(
                f"Demucs model {model_name!r} has no {speech_source!r} source; "
                f"available={self.demucs.sources}"
            )
        self.speech_source = speech_source
        self.input_sample_rate = input_sample_rate
        self.shifts = shifts
        self.overlap = overlap
        self.config = {
            "model_name": model_name, "speech_source": speech_source,
            "input_sample_rate": input_sample_rate, "shifts": shifts, "overlap": overlap,
            "target_rms": target_rms, "max_gain_db": max_gain_db,
            "speech_threshold_db": speech_threshold_db, "speech_ratio": speech_ratio,
            "speech_target_rms": speech_target_rms, "speech_max_gain_db": speech_max_gain_db,
            "speech_attack_ms": speech_attack_ms, "speech_release_ms": speech_release_ms,
        }

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        from demucs.apply import apply_model
        from torchaudio.functional import resample

        original_length = mixture.shape[-1]
        audio = resample(mixture.float(), self.input_sample_rate, self.demucs.samplerate)
        audio = audio.unsqueeze(1).expand(-1, self.demucs.audio_channels, -1)
        stems = apply_model(
            self.demucs, audio, shifts=self.shifts, split=True,
            overlap=self.overlap, device=mixture.device,
        )
        source_index = self.demucs.sources.index(self.speech_source)
        speech = stems[:, source_index].mean(dim=1)
        speech = resample(speech, self.demucs.samplerate, self.input_sample_rate)
        speech = speech[..., :original_length]
        if speech.shape[-1] < original_length:
            speech = torch.nn.functional.pad(speech, (0, original_length - speech.shape[-1]))
        return mixture - speech


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB for every clip, shaped ``[batch]``."""
    estimate = estimate.float()
    reference = reference.float()
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (
        reference.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    target = scale * reference
    residual = estimate - target
    return 10 * torch.log10(
        (target.pow(2).sum(dim=-1) + eps) / (residual.pow(2).sum(dim=-1) + eps)
    )


def multi_resolution_stft_loss(
    estimate: torch.Tensor, reference: torch.Tensor, fft_sizes: tuple[int, ...] = (256, 512, 1024),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Spectral convergence plus log-magnitude loss at several resolutions."""
    losses = []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        estimate_spec = torch.stft(
            estimate.float(), n_fft, hop, n_fft, window=window.float(), return_complex=True
        ).abs()
        reference_spec = torch.stft(
            reference.float(), n_fft, hop, n_fft, window=window.float(), return_complex=True
        ).abs()
        convergence = (reference_spec - estimate_spec).norm(p="fro", dim=(-2, -1)) / (
            reference_spec.norm(p="fro", dim=(-2, -1)) + eps
        )
        log_magnitude = (
            torch.log(reference_spec + eps) - torch.log(estimate_spec + eps)
        ).abs().mean(dim=(-2, -1))
        losses.append((convergence + log_magnitude).mean())
    return torch.stack(losses).mean()


def separation_loss(
    estimate: torch.Tensor, reference: torch.Tensor, spectral_loss_weight: float = 0.0,
) -> torch.Tensor:
    """SI-SDR loss, optionally regularised by multi-resolution STFT fidelity."""
    loss = -si_sdr(estimate, reference).mean()
    if spectral_loss_weight:
        loss = loss + spectral_loss_weight * multi_resolution_stft_loss(estimate, reference)
    return loss


def separator_payload(separator: NoiseSeparator) -> dict:
    """Serializable architecture and weights for embedding in any checkpoint."""
    if isinstance(separator, DemucsNoiseSeparator):
        return {"kind": "demucs_pretrained", "config": dict(separator.config)}
    return {
        "kind": "stft_tcn",
        "config": dict(separator.config),
        "state": {
            key: value.detach().cpu().clone()
            for key, value in separator.state_dict().items()
        },
    }


def build_separator(payload: dict) -> NoiseSeparator:
    if payload.get("kind") == "demucs_pretrained":
        return DemucsNoiseSeparator(**payload["config"])
    separator = NoiseSeparator(**payload["config"])
    separator.load_state_dict(payload["state"])
    return separator
