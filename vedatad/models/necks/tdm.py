from functools import lru_cache
from typing import Sequence
from einops.einops import rearrange
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from vedacore.misc import registry
from vedacore.modules import ConvModule, constant_init, kaiming_init
from vedatad.models.modules.positional_encoding import PositionalEncoding
from vedatad.models.modules.transformer import TransformerEncoderLayer


@registry.register_module("neck")
class TDM(nn.Module):
    """Temporal Down-Sampling Module."""

    def __init__(
        self,
        in_channels,
        stage_layers=(1, 1, 1, 1),
        kernel_sizes=3,
        strides=2,
        paddings=1,
        dilations=1,
        out_channels=256,
        conv_cfg=dict(typename="Conv1d"),
        norm_cfg=dict(typename="BN1d"),
        act_cfg=dict(typename="ReLU"),
        out_indices=(0, 1, 2, 3, 4),
    ):
        super(TDM, self).__init__()

        self.in_channels = in_channels
        self.num_stages = len(stage_layers)
        self.stage_layers = stage_layers
        self.kernel_sizes = _ntuple(self.num_stages)(kernel_sizes)
        self.strides = _ntuple(self.num_stages)(strides)
        self.paddings = _ntuple(self.num_stages)(paddings)
        self.dilations = _ntuple(self.num_stages)(dilations)
        self.out_channels = _ntuple(self.num_stages)(out_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_indices = out_indices

        assert (
            len(self.stage_layers)
            == len(self.kernel_sizes)
            == len(self.strides)
            == len(self.paddings)
            == len(self.dilations)
            == len(self.out_channels)
        )

        self.td_layers = []
        for i in range(self.num_stages):
            td_layer = self.make_td_layer(
                self.stage_layers[i],
                in_channels,
                self.out_channels[i],
                self.kernel_sizes[i],
                self.strides[i],
                self.paddings[i],
                self.dilations[i],
                self.conv_cfg,
                self.norm_cfg,
                self.act_cfg,
            )
            in_channels = self.out_channels[i]
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, td_layer)
            self.td_layers.append(layer_name)

    @staticmethod
    def make_td_layer(
        num_layer,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        conv_cfg,
        norm_cfg,
        act_cfg,
    ):
        layers = []
        layers.append(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )
        for _ in range(1, num_layer):
            layers.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initiate the parameters."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)

        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)


@registry.register_module("neck")
class SelfAttnTDM(nn.Module):

    """self-attention TDM"""

    def __init__(
        self,
        in_channels,
        stage_layers=(1, 1, 1, 1),
        num_heads=8,
        kernel_sizes=None,
        strides=2,
        dropout=0.1,
        add_pe = True,
        out_channels=256,
        out_indices=(0, 1, 2, 3, 4),
        out_order="tbc",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_layers = stage_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.out_indices = out_indices
        self.out_order = out_order
        self.add_pe = add_pe

        if isinstance(self.kernel_sizes, int):
            self.kernel_sizes = [self.kernel_sizes] * len(self.stage_layers)

        assert out_order in ["tbc", "bct"], "output order must be `tbc` or `bct`"

        if self.add_pe:
            self.pe = PositionalEncoding(in_channels, dropout, scale_pe=True)

        self.reductions = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        for i in range(len(stage_layers)):
            layer_in_c = in_channels * 2 if i == 0 else out_channels * 2
            reduction = nn.Sequential(
                nn.Linear(layer_in_c, out_channels, bias=False),
                nn.LayerNorm(out_channels),
            )
            trans_layer = TransformerEncoderLayer(
                out_channels,
                num_heads,
                out_channels * 4,
                dropout=0.1,
                activation="relu",
            )
            self.reductions.append(reduction)
            self.trans_layers.append(trans_layer)

    def init_weights(self):
        """Initiate the parameters."""
        pass

    def downsample(self, x: torch.Tensor, s: int):
        """downsample using reshape with pad. If

        Args:
            x (torch.Tensor): the input to be downsampled. shape: [T,B,C]
            s (int): stride.

        Returns: downsample with pad.

        """
        t = x.shape[0]
        margin = t % s
        if margin != 0:
            pad_left = (s - margin) // 2
            pad_right = (s - margin) - pad_left
            x = torch.nn.functional.pad(x, pad=(0, 0, 0, 0, pad_left, pad_right))
        x = rearrange(x, "(t s) b c -> t b (s c)", s=s)
        return x

    @lru_cache
    def compute_mask(self, n: int, kernel_size: int, v=-1e9):
        """compute mask for local attention with kernel size.

        Args:
            n (int): TODO
            kernel_size (int): TODO

        Returns: torch.Tensor. shape: [n,n]. The masked locations are -1e9
            and unmasked locations are 0.

        """
        if kernel_size is None:
            return None
        half = kernel_size // 2
        mask1 = torch.ones(n, n).triu(diagonal=-half)
        mask2 = torch.ones(n, n).tril(diagonal=half)
        mask = mask1 * mask2
        mask = (1 - mask) * v
        return mask

    def forward(self, x: torch.Tensor):
        """forward fn

        Args:
            x (torch.Tensor): input feature with shape: [B,C,T]

        Returns: TODO

        """
        x = x.permute(2, 0, 1)  # [B,C,T] -> [T,B,C]
        if self.add_pe:
            x = self.pe(x)

        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, (reduction, trans_layer) in enumerate(
            zip(self.reductions, self.trans_layers)
        ):
            x = self.downsample(x, self.strides)
            x = reduction(x)
            if self.kernel_sizes is not None:
                mask = self.compute_mask(x.shape[0], kernel_size=self.kernel_sizes[i])
                mask = mask.to(x.device)
            else:
                mask = None
            x = trans_layer(x, mask)

            if (i + 1) in self.out_indices:
                outs.append(x)  # [T,B,C]

        if self.out_order == "bct":
            for i in range(len(outs)):
                outs[i] = outs[i].permute(1, 2, 0)

        # for i, out in enumerate(outs):
        #     print(i, out.shape)

        if len(outs) == 1:
            outs = outs[0]

        return tuple(outs)


@registry.register_module("neck")
class MultiScaleTDM(TDM):

    """multi-scale TDM"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inps: Sequence[torch.Tensor]):
        """TODO: Docstring for forward.

        Args:
            inps (Sequence[torch.Tensor]): multi-scale features.

        Returns: TODO

        """
        outs = []
        num_inp_stages = range(len(inps))
        x = inps[0]
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if (i + 1) in num_inp_stages:
                x = x + inps[i + 1]
            if (i + 1) in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
