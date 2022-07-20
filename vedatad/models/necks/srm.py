import torch
import torch.nn as nn

from vedacore.misc import build_from_module, registry


@registry.register_module('neck')
class SRM(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self, srm_cfg):
        super(SRM, self).__init__()
        self.srm = build_from_module(srm_cfg, nn)

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.srm(x)
        x = x.squeeze(-1).squeeze(-1)

        return x


@registry.register_module('neck')
class SRMResizeFeature(nn.Module):
    """Resize Feature along temporal dimension"""

    def __init__(self, srm_cfg):
        super().__init__()
        self.pooling = nn.AvgPool1d(srm_cfg["kernel_size"])

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor):
        """forward function

        Args:
            x (torch.Tensor): The input feature. Shape: [B, C, T]

        Returns: torch.Tensor. the resized feature. shape: [B, C, T//kernel_size]

        """
        return self.pooling(x)
