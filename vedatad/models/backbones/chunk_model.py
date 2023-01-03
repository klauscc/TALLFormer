# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================


import os

import torch
import torch.nn.functional as F

from vedacore.misc import registry
from vedatad.models.backbones.vswin import SwinTransformer3D


@registry.register_module("backbone")
class ChunkVideoSwin(SwinTransformer3D):
    """extract feature chunk wise"""

    def __init__(self, chunk_size, *args, do_pooling=False, forward_mode="batch", **kwargs):
        super(ChunkVideoSwin, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        self.forward_mode = forward_mode

        self.pool = torch.nn.AdaptiveAvgPool3d([None, 1, 1]) if do_pooling else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): inputs in two format:
                    1) chunk-first. shape: (num_chunks, B, C, chunk_size, H, W).
                    Returned feature is also chunk-first.
                    2) batch-first. shape: (B, C, D, H, W). Returned feature is
                    batch-first.
        """

        def forward_x(x):
            if x.dim() == 6:  # chunk first
                return self.forward_chunk_inp_output(x)
            elif x.dim() == 5:  # batch-first
                return self.forward_nochunk_inp_output(x)
            else:
                raise ValueError(f"dimension of x should be 5 or 6. Got: {x.shape}")

        if self.forward_mode == "batch":
            return forward_x(x)
        elif self.forward_mode == "split":
            # split inference to save GPU memory.
            l = x.shape[0]
            if l == 1: # split the temporal dimension.
                t = x.shape[2]
                x1 = x[:, :, : t // 2, :, :]
                x2 = x[:, :, t // 2 :, :, :]
                x1 = forward_x(x1)
                x2 = forward_x(x2)
                return torch.cat([x1, x2], dim=2)
            x1 = x[: l // 2]
            x2 = x[l // 2 :]
            x1 = forward_x(x1)
            x2 = forward_x(x2)
            return torch.cat([x1, x2], dim=0)

    def forward_chunk_inp_output(self, x):
        """
        Args:
            x (torch.Tensor): input with shape (num_chunks, B, C, chunk_size, H, W)

        Returns: torch.Tensor. The extract features.
            If `do_pooling` is True, the returned shape is (num_chunks, B, C, feat_chunk_size),
            else is (num_chunks, B, C, feat_chunk_size, H', W')
        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(num_chunks * B, C, chunk_size, H, W)
        x = super().forward(x)  # shape: [n, c, d, h, w]
        _, c, d, h, w = x.shape
        return_shape = (num_chunks, B, c, d, h, w)
        if self.pool:
            x = self.pool(x)
            x = x.squeeze(-1).squeeze(-1)
            return_shape = (num_chunks, B, c, d)
        return x.reshape(return_shape)

    def forward_nochunk_inp_output(self, x):
        """input is not chunk-first. During forward, first divide input into chunks
        and then first each chunks and reshape the output back to original format.

        Args:
            x (torch.Tensor): input with shape (B,C,D,H,W).

        Returns: torch.Tensor. The extract features.
            If `do_pooling` is True, the returned shape is (B, C1, D1),
            else is (B, C1, D1, H1, W1)
        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        pad_d = 0
        if D % chunk_size != 0:
            pad_d = chunk_size - (D % chunk_size)
            D = D + pad_d
            x = F.pad(x, (0, 0, 0, 0, 0, pad_d))
        num_chunks = D // chunk_size
        x = (
            x.reshape(B, C, num_chunks, chunk_size, H, W)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(B * num_chunks, C, chunk_size, H, W)
        )
        x = super().forward(x)  # shape: [n, c, d, h, w]
        _, c, d, h, w = x.shape
        x = (
            x.reshape(B, num_chunks, c, d, h, w)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(B, c, num_chunks * d, h, w)  # shape: [B, c, D//2, h, w]
        )
        x = x[:, :, : num_chunks * d - pad_d // 2, :, :].contiguous()

        if self.pool:
            x = self.pool(x)
            x = x.squeeze(-1).squeeze(-1)
        return x


@registry.register_module("backbone")
class ChunkVideoSwinWithChunkInput(SwinTransformer3D):
    """extract feature chunk wise. Receive input with chunk first."""

    def __init__(self, chunk_size, *args, do_pooling=False, **kwargs):
        super(ChunkVideoSwinWithChunkInput, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        self.pool = torch.nn.AdaptiveAvgPool3d([None, 1, 1]) if do_pooling else None

    def forward(self, x):
        """
        Args:
            x (Tensor[num_chunks, B, C, chunk_size, H, W]): input.

        Returns: torch.Tensor. The extract features.
            If `do_pooling` is True, the returned shape is (num_chunks, B, C, feat_chunk_size),
            else is (num_chunks, B, C, feat_chunk_size, H', W')

        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(num_chunks * B, C, chunk_size, H, W)
        x = super().forward(x)  # shape: [n, c, d, h, w]
        _, c, d, h, w = x.shape
        return_shape = (num_chunks, B, c, d, h, w)
        if self.pool:
            x = self.pool(x)
            x = x.squeeze(-1).squeeze(-1)
            return_shape = (num_chunks, B, c, d)
        return x.reshape(return_shape)
