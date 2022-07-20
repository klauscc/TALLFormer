#!/usr/bin/env python

import math

import torch
import torch.nn as nn
from einops.einops import rearrange

from vedacore.misc import registry
from vedacore.modules.bricks.conv_module import ConvModule
from vedatad.models.modules.masking import biband_mask, q_k_relation_mask
from vedatad.models.modules.positional_encoding import (AbsPosEmbedding,
                                                        PositionalEncoding)
from vedatad.models.modules.transformer import (TransformerDecoder,
                                                TransformerDecoderLayer)
from ..builder import build_neck


@registry.register_module("neck")
class DummyFPN(nn.Module):

    """convert non-FPN to 1-level FPN"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj_out = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=None,
            act_cfg=dict(typename="ReLU"),
        )

    def forward(self, x):
        return [self.proj_out(x)]

    def init_weights(self):
        pass


@registry.register_module("neck")
class AttnFPN(nn.Module):

    """FPN with skip connection"""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        neck_module_cfg,
        inp_add_pe=False,
        fpn_feat_kernel_size: int = None,
        only_attend_to_self=False,
        attention_mask_expand=0,
        norm_cfg: dict = None,
    ):
        """TODO: to be defined.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The out_channels of FPN.
            neck_module_cfg (list or dict): The neck module config.
                Default to None.
            fpn_feat_kernel_size (int): The kernel size for the FPN self-attention.
            only_attend_to_self (bool): Wehther the fpn features only attend to the corresponding high_res_feat.
            attention_mask_expand (float): The ratio to attend outside the corresponding areas.
            conv_norm (dict): The norm for the convolution that project backbone feature to the key/value of the cross-attention.

        """
        super(AttnFPN, self).__init__()

        self.out_channels = out_channels
        self.neck_module_cfg = neck_module_cfg
        self.fpn_feat_kernel_size = fpn_feat_kernel_size
        self.only_attend_to_self = only_attend_to_self
        self.attention_mask_expand = attention_mask_expand

        self.neck = build_neck(neck_module_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=norm_cfg,
            act_cfg=dict(typename="ReLU"),
        )

        self.inp_add_pe = inp_add_pe
        if inp_add_pe:
            self.inp_pe = PositionalEncoding(in_channels, scale_pe=True)

        self.pe = PositionalEncoding(out_channels, scale_pe=True)

        decoder_layer = TransformerDecoderLayer(
            out_channels, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
        )
        self.trans_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def init_weights(self):
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()

    def forward(self, x):
        """forward function

        Args:
            x (torch.Tensor): the features. shape: (B, C, T)

        Returns: tuple. The FPN features. Each element is a tensor of shape (B, C', T'). T' is different for different levels.

        """
        if self.inp_add_pe:
            x = rearrange(x, "b c t -> t b c")
            x = self.inp_pe(x)
            x = rearrange(x, "t b c -> b c t")
        pyramid_features = self.neck(x)
        high_res_feat = self.conv(x).permute(2, 0, 1)  # shape: [T1,B,C]
        outs = []
        for f in pyramid_features:  # shape: [B,C,T]
            f = f.permute(2, 0, 1)  # shape: [T,B,C]
            f = self.pe(f)

            f_mask = biband_mask(f.shape[0], self.fpn_feat_kernel_size, f.device)
            if self.only_attend_to_self:
                mem_mask = q_k_relation_mask(
                    f.shape[0],
                    high_res_feat.shape[0],
                    f.device,
                    expand=self.attention_mask_expand,
                )
            else:
                mem_mask = None
            f = self.trans_decoder(
                f, high_res_feat, tgt_mask=f_mask, memory_mask=mem_mask
            )
            f = f.permute(1, 2, 0)  # shape: [B,C,T]
            outs.append(f)
        return outs


@registry.register_module("neck")
class AttnFPNNorm(nn.Module):

    """FPN with skip connection"""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        neck_module_cfg,
    ):
        """TODO: to be defined.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The out_channels of FPN.
            neck_module_cfg (list or dict): The neck module config.
                Default to
            conv_norm (dict): The norm for the convolution that project backbone feature to the key/value of the cross-attention.

        """
        super(AttnFPNNorm, self).__init__()

        self.out_channels = out_channels
        self.neck_module_cfg = neck_module_cfg

        self.neck = build_neck(neck_module_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="LN"),
            act_cfg=dict(typename="ReLU"),
            order=("act", "conv", "norm"),
        )

        self.fpn_proj = ConvModule(
            out_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="LN"),
            act_cfg=None,
            order=("conv", "norm", "act"),
        )

        self.pe = PositionalEncoding(out_channels, scale_pe=True)

        decoder_layer = TransformerDecoderLayer(
            out_channels, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
        )
        self.trans_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def init_weights(self):
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()

    def forward(self, x):
        """forward function

        Args:
            x (torch.Tensor): the features. shape: (B, C, T)

        Returns: tuple. The FPN features. Each element is a tensor of shape (B, C', T'). T' is different for different levels.

        """
        pyramid_features = self.neck(x)
        high_res_feat = self.conv(x).permute(2, 0, 1)  # shape: [T1,B,C]
        outs = []
        for f in pyramid_features:  # shape: [B,C,T]
            f = self.fpn_proj(f)  # [B,C,T]
            f = f.permute(2, 0, 1)  # shape: [T,B,C]
            f = self.pe(f)
            f = self.trans_decoder(f, high_res_feat)
            f = f.permute(1, 2, 0)  # shape: [B,C,T]
            outs.append(f)
        return outs


@registry.register_module("neck")
class AttnFPNAbsPE(nn.Module):

    """FPN with skip connection"""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        num_fpn_levels,
        max_len,
        neck_module_cfg,
    ):
        """TODO: to be defined.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The out_channels of FPN.
            num_layers (int): The number of decoder layers.
            num_fpn_levels (int): The number of FPN levels.
            max_len (int): The maximum length of FPN features in temporal axis.
            neck_module_cfg (list or dict): The neck module config.

        """
        super(AttnFPNAbsPE, self).__init__()

        self.out_channels = out_channels
        self.neck_module_cfg = neck_module_cfg

        self.neck = build_neck(neck_module_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=None,
            act_cfg=dict(typename="ReLU"),
        )

        self.pes = nn.ModuleList()
        for _ in range(num_fpn_levels):
            self.pes.append(AbsPosEmbedding(max_len, out_channels, scale_pe=True))

        decoder_layer = TransformerDecoderLayer(
            out_channels, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
        )
        self.trans_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def init_weights(self):
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()

    def forward(self, x):
        """forward function

        Args:
            x (torch.Tensor): the features. shape: (B, C, T)

        Returns: tuple. The FPN features. Each element is a tensor of shape (B, C', T'). T' is different for different levels.

        """
        pyramid_features = self.neck(x)
        high_res_feat = self.conv(x).permute(2, 0, 1)  # shape: [T1,B,C]
        outs = []
        for i, f in enumerate(pyramid_features):  # shape: [B,C,T]
            f = f.permute(2, 0, 1)  # shape: [T,B,C]
            f = self.pes[i](f)
            f = self.trans_decoder(f, high_res_feat)
            f = f.permute(1, 2, 0)  # shape: [B,C,T]
            outs.append(f)
        return outs
