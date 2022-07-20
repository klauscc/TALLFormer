from torch import nn
import torch
from typing import Sequence, Union
from vedacore.misc import registry
from vedatad.models.builder import build


@registry.register_module("neck")
class ReshapeFeatures(nn.Module):

    """reshape multi-scale features.
    Inputs are from features from S2,S3,S4 in swin.

    Stage   Orig_shape  Out-shape   OP
    S2      [B, 4C, T/2]   [B, 8C, T/8]   reshape->fc->norm
    S3      [B, 8C, T/2]   [B, 16C, T/4]   reshape->fc->norm
    S4      [B, 8C, T/2]   [B, 8C, T/2]   Identity
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int):
        """
        Args:
            out_channels (int): The feature channels after the reshape.

        """
        super().__init__()
        self.s2_mod = nn.Sequential(
            nn.Linear(in_channels[0], out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
        )
        self.s3_mod = nn.Sequential(
            nn.Linear(in_channels[1], out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
        )

    def init_weights(self):
        pass

    def forward(self, inps: Sequence[torch.Tensor]):
        """forward func

        Args:
            inps (Sequence[torch.Tensor]): The multi-scale features.

        Returns: Sequence[torch.Tensor]. [S4, S3, S2]

        """

        assert (
            isinstance(inps, Sequence) and len(inps) == 3
        ), "The multi-scale features must be [S2,S3,S4] in swin"
        s2, s3, s4 = inps

        ## S2: reshape->fc->norm
        b_s2, c_s2, t_s2 = s2.shape
        s2 = s2.reshape(b_s2, c_s2 * 4, t_s2 // 4)
        s2 = s2.permute(0, 2, 1)
        s2 = self.s2_mod(s2)
        s2 = s2.permute(0, 2, 1)

        ## S3: reshape->fc->norm
        b_s3, c_s3, t_s3 = s3.shape
        s3 = s3.reshape(b_s3, c_s3 * 2, t_s3 // 2)
        s3 = s3.permute(0, 2, 1)
        s3 = self.s3_mod(s3)
        s3 = s3.permute(0, 2, 1)

        return [s4, s3, s2]


@registry.register_module("neck")
class MultiScaleWrapper(nn.Module):

    """apply the same operation to each scale in multi-scale features."""

    def __init__(self, module_cfg):
        """TODO: to be defined."""
        super().__init__()
        self.mod = build(module_cfg, "neck")

    def init_weights(self):
        if isinstance(self.mod, nn.Sequential):
            for m in self.mod:
                if hasattr(m, "init_weights") and callable(m.init_weights):
                    m.init_weights()

    def forward(self, inps: Union[Sequence[torch.Tensor], torch.Tensor]):
        """
        Args:
            inps (Union[Sequence[torch.Tensor], torch.Tensor]): multi-scale features or single scale.
        """
        if isinstance(inps, Sequence):
            outs = [self.mod(x) for x in inps]
        elif isinstance(inps, torch.Tensor):
            outs = self.mod(inps)
        else:
            raise ValueError(f"input type not accepted: {type(inps)}")
        return outs
