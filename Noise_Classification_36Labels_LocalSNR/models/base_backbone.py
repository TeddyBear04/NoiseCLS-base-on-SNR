import torch
import torch.nn as nn

class BaseBackbone(nn.Module):
    """
    Abstract Base Class (ABC / Interface) standardizing CNN Backbone models.
    
    Interface Contract:
      - Input:  2D Mel-spectrogram tensor [Batch, 1, H, W]
      - Output: Classification logits tensor [Batch, Num_Classes]
    """
    def __init__(self) -> None:
        super(BaseBackbone, self).__init__()
        self.model_name = "base_backbone"

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
