import os
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_backbone import BaseBackbone
from config import LocalSNRConfig

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioModel(nn.Module):
    """
    Unified AudioModel Wrapper class.
    Connects AudioFrontend (GPU spectrogram extractor) with a CNN Backbone model complying with BaseBackbone contract.
    Accepts 1D raw waveform input and returns classification logits for the 4 feeding intensity classes.
    """
    def __init__(self, frontend: nn.Module, backbone: BaseBackbone) -> None:
        """
        Initialize AudioModel wrapper.

        Args:
            frontend (nn.Module): Audio preprocessing/spectrogram extractor (e.g. AudioFrontend).
            backbone (BaseBackbone): CNN backbone model inheriting from BaseBackbone.
        """
        super(AudioModel, self).__init__()
        
        # Type safety validation to enforce compliance with BaseBackbone interface contract
        assert isinstance(backbone, BaseBackbone), "Error: Provided backbone model must inherit from BaseBackbone!"
        
        self.frontend = frontend
        self.backbone = backbone

        logger.info("==================================================")
        logger.info("Initialized unified AudioModel wrapper:")
        logger.info(f"  - Frontend: {self.frontend.__class__.__name__}")
        logger.info(f"  - Backbone: {self.backbone.__class__.__name__}")
        logger.info("==================================================")

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass of the unified AudioModel.

        Args:
            input_tensor (torch.Tensor): Raw audio waveforms [Batch, Num_Samples].

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing classification logits 'clipwise_output' [Batch, Num_Classes].
        """
        # Phase 1: Transform raw waveforms into 2D Mel-spectrograms on GPU [Batch, 1, H, W]
        features = self.frontend(input_tensor)

        # Phase 2: Feature extraction and classification through CNN Backbone
        logits = self.backbone(features)

        # Format output dictionary to align with Trainer expectation
        return {
            "clipwise_output": logits
        }


class LocalSNRAudioModel(nn.Module):
    """Shared CNN with file-level noise and time-local SNR heads."""

    def __init__(
        self,
        frontend: nn.Module,
        backbone: BaseBackbone,
        local_snr_config: LocalSNRConfig,
        segment_count: int,
        noise_filter: nn.Module | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        channels = int(getattr(backbone, "feature_channels", 0))
        if channels <= 0:
            raise ValueError(
                f"{backbone.__class__.__name__} does not declare time-resolved feature_channels"
            )
        if segment_count <= 0:
            raise ValueError("segment_count must be positive")
        self.frontend = frontend
        self.backbone = backbone
        self.local_snr_config = local_snr_config
        self.segment_count = segment_count
        self.snr_head = nn.Sequential(
            nn.Linear(channels, local_snr_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(local_snr_config.dropout),
            nn.Linear(local_snr_config.hidden_dim, 1),
        )
        self.noise_filter = noise_filter
        self.sample_rate = int(sample_rate)
        # The frontend normalises log-mel per bin, which would erase the level shift the
        # adaptive gain introduces, so the classifier is told what gain was applied.
        self.condition_proj = None
        if noise_filter is not None:
            head = getattr(backbone, "fc_audioset", None)
            if head is None:
                raise ValueError(
                    f"{backbone.__class__.__name__} has no fc_audioset, so the gain "
                    "condition cannot be wired into its logits"
                )
            # Starts at zero so an untrained condition cannot disturb the logits.
            self.condition_proj = nn.Linear(2, head.out_features)
            nn.init.zeros_(self.condition_proj.weight)
            nn.init.zeros_(self.condition_proj.bias)

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        filtered = self.noise_filter(input_tensor) if self.noise_filter is not None else None
        classifier_input = (
            filtered["est_noise_amplified"] if filtered is not None else input_tensor
        )

        features = self.frontend(classifier_input)
        feature_map = self.backbone.forward_feature_map(features)
        temporal = self.backbone.temporal_features(feature_map)
        clip_logits = self.backbone.classify_temporal(temporal)

        aligned = F.adaptive_avg_pool1d(
            temporal.transpose(1, 2), self.segment_count
        ).transpose(1, 2)

        if filtered is None:
            snr_normalized = self.snr_head(aligned).squeeze(-1)
            snr_db = (
                snr_normalized * self.local_snr_config.target_scale_db
                + self.local_snr_config.target_offset_db
            )
            output = {
                "clipwise_output": clip_logits,
                "local_snr_normalized": snr_normalized,
                "local_snr_db": snr_db,
                "segment_features": aligned,
            }
        else:
            if self.condition_proj is not None:
                clip_logits = clip_logits + self.condition_proj(filtered["condition"])
            # Measured on the un-amplified pair: the gain would distort the ratio the
            # labels record.
            snr_db = self.noise_filter.segment_snr_db(
                filtered["est_speech"],
                filtered["est_noise"],
                self.sample_rate,
                self.local_snr_config,
            )
            snr_normalized = (
                snr_db - self.local_snr_config.target_offset_db
            ) / self.local_snr_config.target_scale_db
            output = {
                "clipwise_output": clip_logits,
                "local_snr_normalized": snr_normalized,
                "local_snr_db": snr_db,
                "segment_features": aligned,
                "est_noise": filtered["est_noise"],
                "gain": filtered["gain"],
                "clip_snr_db": filtered["snr_db"],
            }
        return output
