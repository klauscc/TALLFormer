# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================


import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vedacore.misc import registry
from vedatad.models.backbones.resnet3d import ResNet3d
from vedatad.models.backbones.vswin import SwinTransformer3D
from vedatad.models.builder import build_backbone


class TemporalGradDrop(torch.autograd.Function):
    """gradient drop for backbone"""

    @staticmethod
    def forward(ctx, keep_indices, drop_indices, forward_fn, x, *params):
        """

        Args:
            keep_indices (list): The indices of chunks to keep gradients
            forward_fn (nn.Module.forward): forward function
            x (torch.Tensor): the input. Shape: [num_chunks, B, C, chunk_size, H, W]. D = num_chunks * chunk_size.
            *params (List of nn.Parameters): The parameters that need gradients.

        Returns: forward output.

        """
        x_w_grad = x[keep_indices]

        # save for backward
        ctx.keep_indices = keep_indices
        ctx.tensors = (x_w_grad.detach().clone(), params)
        ctx.forward_fn = forward_fn

        y = forward_fn(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        """backward

        Args:
            dy (torch.Tensor): Gradients to dy. shape: (num_chunks,B,C1,D1,H1,W1)

        Returns: gradients to inputs of forward.

        """
        keep_indices = ctx.keep_indices
        x_w_grad, params = ctx.tensors
        forward_fn = ctx.forward_fn

        # forward inputs need grads.
        with torch.enable_grad():
            y_w_grad = forward_fn(x_w_grad)  # [num_chunks_keep, B, C', D',H',W']
        d_y_w_grad = dy[keep_indices]
        params_grad = torch.autograd.grad(y_w_grad, params, d_y_w_grad)
        del x_w_grad, ctx.keep_indices
        return (
            None,
            None,
            None,
            None,
        ) + params_grad


def generate_indices(num_chunks, keep_ratio, mode="uniform"):
    """generate indices of inputs that need and don't need gradients.

    Args:
        num_chunks (int): number of chunks
        keep_ratio (float): keep ratio.

    Returns: TODO

    """
    if mode == "uniform":
        keep_indices = (
            np.floor(np.linspace(0, num_chunks - 1, int(num_chunks * keep_ratio)))
            .astype(np.int64)
            .tolist()
        )
    elif mode == "uniform_jitter":
        keep_indices = np.linspace(
            0, num_chunks - 1, int(num_chunks * keep_ratio), endpoint=False
        )
        jitters = np.random.uniform(
            0, (num_chunks - 1) / num_chunks / keep_ratio, size=len(keep_indices)
        )
        keep_indices = keep_indices + jitters
        keep_indices = keep_indices.astype(np.int64)
        keep_indices = np.clip(keep_indices, 0, num_chunks - 1).tolist()
    elif mode == "random":
        keep_indices = np.random.choice(
            np.arange(num_chunks), size=int(num_chunks * keep_ratio), replace=False
        ).tolist()
        keep_indices = sorted(keep_indices)
    else:
        raise ValueError(f"generate_indices: mode:{mode} not implemented")

    drop_indices = []
    for i in range(num_chunks):
        if i not in keep_indices:
            drop_indices.append(i)

    return keep_indices, drop_indices


@registry.register_module("backbone")
class GradDropChunkVideoSwin(SwinTransformer3D):
    """chunk-wise video swin with partial feedback."""

    def __init__(self, keep_ratio, chunk_size, *args, **kwargs):
        super(GradDropChunkVideoSwin, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio

        self.graddrop_op = TemporalGradDrop.apply

    def forward_fn(self, x):
        """forward function

        Args:
            x (torch.Tensor): input video. shape: (num_chunks,B,C,chunk_size,H,W)

        Returns: TODO

        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(
            num_chunks * B, C, chunk_size, H, W
        )  # shape: [num_chunks*B, C,D,H,W]
        y = super().forward(x)  # shape: [num_chunks*B, C', D', H', W']
        _, C1, D1, H1, W1 = y.shape
        y = y.reshape(num_chunks, B, C1, D1, H1, W1)
        return y

    def gather_trainable_parameters(self):
        """gather the trainable parameters
        Returns: List of trainable parameters.

        """
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def forward(self, x):
        """forward

        Args:
            x (torch.Tensor): input video. shape: (B,C,D,H,W)

        Returns: TODO

        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        assert D % chunk_size == 0, "D mod chunk_size must be 0."
        num_chunks = D // chunk_size

        # transpose input
        x = x.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # shape: (num_chunks, B, C, chunk_size, H, W)

        # generate indices
        keep_indices, drop_indices = generate_indices(num_chunks, self.keep_ratio)

        trainable_params = self.gather_trainable_parameters()
        y = self.graddrop_op(
            keep_indices, drop_indices, self.forward_fn, x, *trainable_params
        )  # shape: (num_chunks, B,C1, D1,H1,W1)

        num_chunks, B, C1, D1, H1, W1 = y.shape
        y = y.permute(1, 2, 0, 3, 4, 5).reshape(B, C1, num_chunks * D1, H1, W1)
        return y


@registry.register_module("backbone")
class GradDropChunkVideoSwinV2(SwinTransformer3D):
    """chunk-wise video swin with partial feedback."""

    def __init__(
        self,
        keep_ratio,
        chunk_size,
        *args,
        bp_idx_mode="uniform",
        forward_mode="batch",
        shift_inp=False,
        t_downsample=2,
        **kwargs,
    ):
        super(GradDropChunkVideoSwinV2, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio
        self.bp_idx_mode = bp_idx_mode
        self.forward_mode = forward_mode

        # shift input so that the chunks are different at each iteration.
        self.shift_inp = shift_inp
        self.t_downsample = t_downsample

    def forward_fn(self, x: torch.Tensor):
        """forward function

        Args:
            x (torch.Tensor): input video. shape: (num_chunks,B,C,chunk_size,H,W)

        Returns: the extracted features. shape: (num_chunks, B, C1, D1, H1, W1)

        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(
            num_chunks * B, C, chunk_size, H, W
        )  # shape: [num_chunks*B, C,D,H,W]
        outs = super().forward(x)  # shape: [num_chunks*B, C', D', H', W']

        def pooling_reshape_feat(x: torch.Tensor):
            _, C1, D1, H1, W1 = x.shape
            x = x.reshape(num_chunks * B, C1, D1, H1, W1)
            x = torch.nn.functional.adaptive_avg_pool3d(x, [None, 1, 1])
            x = x.squeeze(-1).squeeze(-1)
            x = x.reshape(num_chunks, B, C1, D1)
            return x

        if isinstance(outs, (list, tuple)):
            outs = [pooling_reshape_feat(x) for x in outs]
        elif isinstance(outs, torch.Tensor):
            outs = pooling_reshape_feat(outs)
        else:
            raise ValueError("not acceptted type.")
        return outs

    def forward(self, x: torch.Tensor):
        """forward

        Args:
            x (torch.Tensor): input video. shape: (B,C,D,H,W)

        Returns: The extracted features. shape: (B, C1, D1, H1, W1)

        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        keep_ratio = self.keep_ratio
        assert D % chunk_size == 0, "D mod chunk_size must be 0."
        num_chunks = D // chunk_size

        if self.shift_inp and self.training:
            pad_l = int(torch.randint(0, chunk_size, size=[]))
            pad_r = chunk_size - pad_l
            x = F.pad(x, pad=(0, 0, 0, 0, pad_l, pad_r, 0, 0, 0, 0))
            keep_ratio = (num_chunks * keep_ratio) / (num_chunks + 1)
            num_chunks = num_chunks + 1

        # transpose input
        x = x.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # shape: (num_chunks, B, C, chunk_size, H, W)

        # generate indices
        keep_indices, drop_indices = generate_indices(
            num_chunks, self.keep_ratio, self.bp_idx_mode
        )

        # ----- Batch forward ----------
        # orig v2.
        def batch_forward():
            with torch.no_grad():
                y_wo_grad = self.forward_fn(x[drop_indices].contiguous())
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        # --------------------------------

        # ----- Split forward ----------
        # new v2.
        def split_forward():
            with torch.no_grad():
                y_wo_grad = []
                drop_indices_splited = [
                    drop_indices[: len(drop_indices) // 2],
                    drop_indices[len(drop_indices) // 2 :],
                ]
                for idx in drop_indices_splited:
                    y_wo_grad.append(self.forward_fn(x[idx].contiguous()))
                y_wo_grad = torch.cat(y_wo_grad, dim=0)
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        forward_mode = self.forward_mode
        if forward_mode == "batch":
            y_wo_grad, y_w_grad = batch_forward()
        elif forward_mode == "split":
            y_wo_grad, y_w_grad = split_forward()
        else:
            raise ValueError(f"forward mode:{forward_mode} not supported")

        def merge_feats(y_w_grad: torch.Tensor, y_wo_grad: torch.Tensor):
            _, B, C1, D1 = y_w_grad.shape
            y = torch.zeros(
                num_chunks, B, C1, D1, device=y_w_grad.device, dtype=y_w_grad.dtype
            )
            y[keep_indices] = y_w_grad
            y[drop_indices] = y_wo_grad
            y = y.permute(1, 2, 0, 3).reshape(B, C1, num_chunks * D1)
            if self.shift_inp and self.training:
                start = pad_l // self.t_downsample
                y = y[:, :, start : start + (num_chunks - 1) * D1]
            return y

        if isinstance(y_w_grad, (list, tuple)):
            outs = [
                merge_feats(w_grad, wo_grad)
                for (w_grad, wo_grad) in zip(y_w_grad, y_wo_grad)
            ]
        else:
            outs = merge_feats(y_w_grad, y_wo_grad)
        return outs


@registry.register_module("backbone")
class GradDropModel(nn.Module):
    """chunk-wise video swin with partial feedback."""

    def __init__(
        self,
        backbone,
        keep_ratio,
        chunk_size,
        bp_idx_mode="uniform",
        forward_mode="batch",
        shift_inp=False,
        t_downsample=2,
        pooling=False,
    ):
        super(GradDropModel, self).__init__()
        self.backbone = backbone
        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio
        self.bp_idx_mode = bp_idx_mode
        self.forward_mode = forward_mode

        # shift input so that the chunks are different at each iteration.
        self.shift_inp = shift_inp
        self.t_downsample = t_downsample
        self.pooling = pooling

    def forward_fn(self, x: torch.Tensor):
        """forward function

        Args:
            x (torch.Tensor): input video. shape: (num_chunks,B,C,chunk_size,H,W)

        Returns: the extracted features. shape: (num_chunks, B, C1, D1, H1, W1)

        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(
            num_chunks * B, C, chunk_size, H, W
        )  # shape: [num_chunks*B, C,D,H,W]
        outs = self.backbone(x)  # shape: [num_chunks*B, C', D', H', W']

        def pooling_reshape_feat(x: torch.Tensor):
            if self.pooling:
                _, C1, D1, H1, W1 = x.shape
                x = x.reshape(num_chunks * B, C1, D1, H1, W1)
                x = torch.nn.functional.adaptive_avg_pool3d(x, [None, 1, 1])
                x = x.squeeze(-1).squeeze(-1)
            x = x.reshape(num_chunks, B, *x.shape[1:])
            return x

        if isinstance(outs, (list, tuple)):
            outs = [pooling_reshape_feat(x) for x in outs]
        elif isinstance(outs, torch.Tensor):
            outs = pooling_reshape_feat(outs)
        else:
            raise ValueError("not acceptted type.")
        return outs

    def forward(self, x: torch.Tensor):
        """forward

        Args:
            x (torch.Tensor): input video. shape: (B,C,D,H,W)

        Returns: The extracted features. shape: (B, C1, D1, H1, W1)

        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        keep_ratio = self.keep_ratio
        assert D % chunk_size == 0, "D mod chunk_size must be 0."
        num_chunks = D // chunk_size

        if self.shift_inp and self.training:
            pad_l = int(torch.randint(0, chunk_size, size=[]))
            pad_r = chunk_size - pad_l
            x = F.pad(x, pad=(0, 0, 0, 0, pad_l, pad_r, 0, 0, 0, 0))
            keep_ratio = (num_chunks * keep_ratio) / (num_chunks + 1)
            num_chunks = num_chunks + 1

        # transpose input
        x = x.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # shape: (num_chunks, B, C, chunk_size, H, W)

        # generate indices
        keep_indices, drop_indices = generate_indices(
            num_chunks, self.keep_ratio, self.bp_idx_mode
        )

        # ----- Batch forward ----------
        # orig v2.
        def batch_forward():
            with torch.no_grad():
                y_wo_grad = self.forward_fn(x[drop_indices].contiguous())
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        # --------------------------------

        # ----- Split forward ----------
        # new v2.
        def split_forward():
            with torch.no_grad():
                y_wo_grad = []
                drop_indices_splited = [
                    drop_indices[: len(drop_indices) // 2],
                    drop_indices[len(drop_indices) // 2 :],
                ]
                for idx in drop_indices_splited:
                    y_wo_grad.append(self.forward_fn(x[idx].contiguous()))
                y_wo_grad = torch.cat(y_wo_grad, dim=0)
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        forward_mode = self.forward_mode
        if forward_mode == "batch":
            y_wo_grad, y_w_grad = batch_forward()
        elif forward_mode == "split":
            y_wo_grad, y_w_grad = split_forward()
        else:
            raise ValueError(f"forward mode:{forward_mode} not supported")

        def merge_feats(y_w_grad: torch.Tensor, y_wo_grad: torch.Tensor):
            _, B, C1, D1 = y_w_grad.shape[:4]
            other_dims = list(y_w_grad.shape[4:])
            y = torch.zeros(
                [num_chunks, B, C1, D1] + other_dims,
                device=y_w_grad.device,
                dtype=y_w_grad.dtype,
            )
            y[keep_indices] = y_w_grad
            y[drop_indices] = y_wo_grad
            y = y.permute(1, 2, 0, 3, *list(range(4, 4 + len(other_dims)))).reshape(
                B, C1, num_chunks * D1, *other_dims
            )
            if self.shift_inp and self.training:
                start = pad_l // self.t_downsample
                y = y[:, :, start : start + (num_chunks - 1) * D1]
            return y

        if isinstance(y_w_grad, (list, tuple)):
            outs = [
                merge_feats(w_grad, wo_grad)
                for (w_grad, wo_grad) in zip(y_w_grad, y_wo_grad)
            ]
        else:
            outs = merge_feats(y_w_grad, y_wo_grad)
        return outs


@registry.register_module("backbone")
class GradDropI3D(GradDropModel):

    """Docstring for GradDropI3D."""

    def __init__(self, *args, backbone_pretrained="", **kwargs):
        """TODO: to be defined."""
        super().__init__(*args, **kwargs)
        backbone_cfg = self.backbone
        self.backbone = build_backbone(backbone_cfg)
        self.backbone_pretrained = backbone_pretrained

        pretrained = self.backbone_pretrained
        if os.path.isfile(pretrained):
            state = torch.load(pretrained)["state_dict"]
            new_state = {k.replace("backbone.", ""): v for k, v in state.items()}
            info = self.backbone.load_state_dict(new_state, strict=False)
            print(f"load pretrained: {pretrained}")
            print(info)
        else:
            if pretrained != "" and pretrained is not None:
                print(f"pretrained not exist: {pretrained}")

    def init_weights(self):
        pass


@registry.register_module("backbone")
class GradDropTimeSformer(GradDropModel):

    """GradDrop model for TimeSformer"""

    def __init__(self, *args, backbone_pretrained="", **kwargs):
        """TODO: to be defined."""
        super().__init__(*args, **kwargs)
        backbone_cfg = self.backbone
        from timesformer.models.vit import TimeSformer

        self.backbone = TimeSformer(**backbone_cfg)
        self.backbone_pretrained = backbone_pretrained

        pretrained = self.backbone_pretrained
        if os.path.isfile(pretrained):
            state = torch.load(pretrained)["model_state"]
            info = self.backbone.load_state_dict(state, strict=False)
            print(f"load pretrained: {pretrained}")
            print(info)
        else:
            if pretrained != "" and pretrained is not None:
                print(f"pretrained not exist: {pretrained}")

    def init_weights(self):
        pass
