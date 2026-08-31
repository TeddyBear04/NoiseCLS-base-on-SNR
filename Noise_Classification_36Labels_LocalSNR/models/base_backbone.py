import torch
import torch.nn as nn

class BaseBackbone(nn.Module):
    """
    Abstract Base Class (ABC / Interface) standardizing CNN Backbone models.
    
    Interface Contract:
      - Input: 2D Mel-spectrogram tensor [Batch, 1, Time, Mel]
      - forward_feature_map: [Batch, Channels, Time', Frequency']
      - temporal_features: [Batch, Time', Channels]
      - classify_temporal: [Batch, Num_Classes]
    """
    def __init__(self) -> None:
        super(BaseBackbone, self).__init__()
        self.model_name = "base_backbone"
        self.feature_channels = 0

    def get_name(self) -> str:
        """
        Retrieve the name of the backbone model architecture.
        """
        return self.model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass. Subclasses must override this method.
        """
        raise NotImplementedError("Method 'forward' must be implemented in subclasses.")

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return features shaped [batch, channels, time, frequency]."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward_feature_map() for Local SNR"
        )

    @staticmethod
    def temporal_features(feature_map: torch.Tensor) -> torch.Tensor:
        """Collapse frequency while retaining time: [B,C,T,F] -> [B,T,C]."""
        if feature_map.ndim != 4:
            raise ValueError(f"Expected feature map [B,C,T,F], got {tuple(feature_map.shape)}")
        return torch.mean(feature_map, dim=3).transpose(1, 2)

    @staticmethod
    def pool_temporal(temporal: torch.Tensor) -> torch.Tensor:
        """PANNs-style global max+mean pooling over time."""
        if temporal.ndim != 3:
            raise ValueError(f"Expected temporal features [B,T,C], got {tuple(temporal.shape)}")
        channels_first = temporal.transpose(1, 2)
        return torch.max(channels_first, dim=2).values + torch.mean(channels_first, dim=2)

    def classify_temporal(self, temporal: torch.Tensor) -> torch.Tensor:
        """Map time-resolved features to file-level logits."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement classify_temporal() for Local SNR"
        )
