import torch
import torch.nn as nn

from vedacore.misc import registry
from vedacore.modules.bricks.conv_module import ConvModule
from vedatad.models.modules.positional_encoding import PositionalEncoding
from vedatad.models.modules.swin_1d import Encoder, EncoderLayer1D
from vedatad.models.modules.transformer import (TransformerEncoder,
                                                TransformerEncoderLayer)


@registry.register_module("neck")
class SRMSwin(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self, srm_cfg):
        super(SRMSwin, self).__init__()
        # self.srm = build_from_module(srm_cfg, nn)

        in_channels = srm_cfg["in_channels"]
        out_channels = srm_cfg["out_channels"]
        self.out_channels = out_channels

        self.pooling = nn.AdaptiveAvgPool3d([None, 1, 1])
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

        self.with_transformer = False
        if hasattr(srm_cfg, "with_transformer") and srm_cfg["with_transformer"]:
            self.with_transformer = True
            trans_cfg = srm_cfg["transformer"]
            attn_layer = TransformerEncoderLayer(**trans_cfg["encoder_layer"])
            self.encoder = TransformerEncoder(
                attn_layer, num_layers=trans_cfg["num_layers"]
            )

            # positional encoding
            if hasattr(trans_cfg, "pos_enc") and trans_cfg["pos_enc"]:
                pos_enc = trans_cfg["pos_enc"]
                if isinstance(pos_enc, dict):
                    self.pe = build_from_cfg(pos_enc, registry, "pos_enc")
                else:
                    self.pe = PositionalEncoding(out_channels, scale_pe=True)
            else:
                self.pe = None

    def init_weights(self):
        pass

    def forward(self, x):
        """
        Args:
            x (torch.Tensor) : video input. Shape: (B,C1,D1,H1,W1) or (B,C1,D1).

        Returns:
            torch.Tensor. Features of shape (B, C2, D2).
        """
        if x.dim() == 5:  # [B, C1, D1, H1, W1]
            x = self.pooling(x)
            x = x.squeeze(-1).squeeze(-1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)  # [B, C2, D2]
        if self.with_transformer:
            x = x.permute(2, 0, 1)  # [D2, B, C2]
            if self.pe:
                # x *= math.sqrt(self.out_channels)
                x = self.pe(x)
            x = self.encoder(x)
            x = x.permute(1, 2, 0)  # [B, C2, D2]
        return x


@registry.register_module("neck")
class SRMSwinNorm(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self, srm_cfg):
        super(SRMSwinNorm, self).__init__()
        # self.srm = build_from_module(srm_cfg, nn)

        in_channels = srm_cfg["in_channels"]
        out_channels = srm_cfg["out_channels"]
        self.out_channels = out_channels

        self.pooling = nn.AdaptiveAvgPool3d([None, 1, 1])
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="LN"),
            act_cfg=dict(typename="ReLU"),
        )
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="LN"),
            act_cfg=dict(typename="ReLU"),
        )
        self.proj = ConvModule(
            out_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="LN"),
            act_cfg=None,
        )
        self.with_transformer = False
        if hasattr(srm_cfg, "with_transformer") and srm_cfg["with_transformer"]:
            self.with_transformer = True
            trans_cfg = srm_cfg["transformer"]
            attn_layer = TransformerEncoderLayer(**trans_cfg["encoder_layer"])
            self.encoder = TransformerEncoder(
                attn_layer, num_layers=trans_cfg["num_layers"]
            )

            # positional encoding
            if hasattr(trans_cfg, "pos_enc") and trans_cfg["pos_enc"]:
                pos_enc = trans_cfg["pos_enc"]
                if isinstance(pos_enc, dict):
                    self.pe = build_from_cfg(pos_enc, registry, "pos_enc")
                else:
                    self.pe = PositionalEncoding(out_channels, scale_pe=True)
            else:
                self.pe = None

    def init_weights(self):
        pass

    def forward(self, x):
        """
        Args:
            x (torch.Tensor) : video input. Shape: (B,C1,D1,H1,W1) or (B,C1,D1).

        Returns:
            torch.Tensor. Features of shape (B, C2, D2).
        """
        if x.dim() == 5:  # [B, C1, D1, H1, W1]
            x = self.pooling(x)
            x = x.squeeze(-1).squeeze(-1)
        x = self.conv1(x)
        x = self.conv2(x)  # [B, C2, D2]
        x = self.proj(x)  # [B, C2, D2]
        if self.with_transformer:
            x = x.permute(2, 0, 1)  # [D2, B, C2]
            if self.pe:
                # x *= math.sqrt(self.out_channels)
                x = self.pe(x)
            x = self.encoder(x)
            x = x.permute(1, 2, 0)  # [B, C2, D2]
        return x


@registry.register_module("neck")
class Transformer1DRelPos(nn.Module):

    """Docstring for Transformer1DRelPos."""

    def __init__(self, encoder_layer_cfg: dict, num_layers: int):

        super().__init__()
        encoder_layer = EncoderLayer1D(**encoder_layer_cfg)
        self.encoder = Encoder(encoder_layer, num_layers)

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor):
        """forward function

        Args:
            x (torch.Tensor): input with shape: [B,C,D].

        Returns: torch.Tensor. The same shape as input.

        """

        x = x.permute(0, 2, 1)  # [B,C,D] -> [B,D,C]

        x = self.encoder(x)  # [B,D,C]

        return x.permute(0, 2, 1)  # [B,C,D]
