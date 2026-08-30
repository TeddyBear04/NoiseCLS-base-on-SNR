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
from models.base_backbone import BaseBackbone

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
