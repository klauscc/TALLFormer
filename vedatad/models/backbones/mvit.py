import math
import os
from copy import copy
from functools import partial

import torch
from easydict import EasyDict
from einops import rearrange
from slowfast.models import stem_helper
from slowfast.models.attention import MultiScaleBlock
from slowfast.models.utils import round_width
from torch import nn
from torch.nn.init import trunc_normal_

from vedacore.misc import registry

checkpoint_wrapper = None


class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    DEFAULT_CFG = {
        "ACT_CHECKPOINT": False,
        "SPATIAL_SIZE": 224,
        "NUM_FRAMES": 32,
        "IN_CHANNELS": 3,
        "NUM_CLASSES": 600,
        "POOL_FIRST": False,
        "CLS_EMBED_ON": True,
        "ZERO_DECAY_POS_CLS": False,
        "SEP_POS_EMBED": True,
        "NUM_HEADS": 1,
        "EMBED_DIM": 96,
        "PATCH_KERNEL": (3, 7, 7),
        "PATCH_STRIDE": (2, 4, 4),
        "PATCH_PADDING": (1, 3, 3),
        "PATCH_2D": False,
        "MLP_RATIO": 4.0,
        "QKV_BIAS": True,
        "DROPPATH_RATE": 0.3,
        "DROPOUT_RATE": 0,
        "NORM": "layernorm",
        "NORM_STEM": False,
        "MODE": "conv",
        "DEPTH": 24,
        "POOL_Q_STRIDE": [[2, 1, 2, 2], [5, 1, 2, 2], [21, 1, 2, 2]],
        "POOL_KVQ_KERNEL": None,
        "DIM_MUL": [[2, 2.0], [5, 2.0], [21, 2.0]],
        "HEAD_MUL": [[2, 2.0], [5, 2.0], [21, 2.0]],
        "POOL_KV_STRIDE_ADAPTIVE": [1, 8, 8],
        "SEP_POS_EMBED": True,
    }

    def __init__(self, model_cfg):
        super().__init__()
        # Get parameters.
        cfg = EasyDict(copy(self.DEFAULT_CFG))
        for k, v in model_cfg.items():
            cfg[k] = v
        self.cfg = cfg
        pool_first = cfg.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.SPATIAL_SIZE
        temporal_size = cfg.NUM_FRAMES
        in_chans = cfg.IN_CHANNELS
        use_2d_patch = cfg.PATCH_2D
        self.patch_stride = cfg.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.NUM_CLASSES
        embed_dim = cfg.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.NUM_HEADS
        mlp_ratio = cfg.MLP_RATIO
        qkv_bias = cfg.QKV_BIAS
        self.drop_rate = cfg.DROPOUT_RATE
        depth = cfg.DEPTH
        drop_path_rate = cfg.DROPPATH_RATE
        mode = cfg.MODE
        self.cls_embed_on = cfg.CLS_EMBED_ON
        self.sep_pos_embed = cfg.SEP_POS_EMBED
        if cfg.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.PATCH_KERNEL,
            stride=cfg.PATCH_STRIDE,
            padding=cfg.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.DIM_MUL)):
            dim_mul[cfg.DIM_MUL[i][0]] = cfg.DIM_MUL[i][1]
        for i in range(len(cfg.HEAD_MUL)):
            head_mul[cfg.HEAD_MUL[i][0]] = cfg.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.DEPTH)]
        pool_kv = [[] for i in range(cfg.DEPTH)]
        stride_q = [[] for i in range(cfg.DEPTH)]
        stride_kv = [[] for i in range(cfg.DEPTH)]

        for i in range(len(cfg.POOL_Q_STRIDE)):
            stride_q[cfg.POOL_Q_STRIDE[i][0]] = cfg.POOL_Q_STRIDE[i][1:]
            if cfg.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.POOL_Q_STRIDE[i][0]] = cfg.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.POOL_KV_STRIDE_ADAPTIVE
            cfg.POOL_KV_STRIDE = []
            for i in range(cfg.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.POOL_KV_STRIDE)):
            stride_kv[cfg.POOL_KV_STRIDE[i][0]] = cfg.POOL_KV_STRIDE[i][1:]
            if cfg.POOL_KVQ_KERNEL is not None:
                pool_kv[cfg.POOL_KV_STRIDE[i][0]] = cfg.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.NORM_STEM else None

        self.blocks = nn.ModuleList()

        if cfg.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            if cfg.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x):
        x = self.patch_embed(x)

        T = self.cfg.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.SPATIAL_SIZE // self.patch_stride[1]
        W = self.cfg.SPATIAL_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 1:]
        x = rearrange(x, "b (t n) c -> b c t n", t=T)
        x = x.mean(-1)
        return x


@registry.register_module("backbone")
class ChunkMVit(MViT):
    """extract feature chunk wise"""

    def __init__(
        self, chunk_size, pretrained_model, *args, forward_mode="batch", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        self.forward_mode = forward_mode
        info = self.load_state_dict(
            torch.load(pretrained_model, map_location="cpu")["model_state"],
            strict=False,
        )
        print(f"Load checkpoint {os.path.basename(pretrained_model)}: {info}.")

    def init_weights(self):
        pass

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
            l = x.shape[0]
            if l == 1:
                return forward_x(x)
            x1 = x[: l // 2]
            x2 = x[l // 2 :]
            x1 = forward_x(x1)
            x2 = forward_x(x2)
            return torch.cat([x1, x2], dim=0)

    def forward_temporal_features(self, x: torch.Tensor):
        """forward and get temporal features.

        Args:
            x (torch.Tensor): The input frames. Shape: [B, C, T, H, W]

        Returns: torch.Tensor. shape: [B, C', T]. The temporal features.

        """
        return super().forward(x)

    def forward_chunk_inp_output(self, x):
        """
        Args:
            x (torch.Tensor): input with shape (num_chunks, B, C, chunk_size, H, W)

        Returns: torch.Tensor. The extract features.shape: (num_chunks, B, C, chunk_size)
        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(num_chunks * B, C, chunk_size, H, W)
        x = self.forward_temporal_features(x)
        _, c, d = x.shape
        return_shape = (num_chunks, B, c, d)
        return x.reshape(return_shape)

    def forward_nochunk_inp_output(self, x):
        """input is not chunk-first. During forward, first divide input into chunks
        and then first each chunks and reshape the output back to original format.

        Args:
            x (torch.Tensor): input with shape (B,C,D,H,W).

        Returns: torch.Tensor. The extract features. Shape: [B, C, D].
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
        x = self.forward_temporal_features(x)  # [B*num_chunks, C, T]
        _, c, d = x.shape
        x = (
            x.reshape(B, num_chunks, c, d)
            .permute(0, 2, 1, 3)
            .reshape(B, c, num_chunks * d)  # shape: [B, c, D//2]
        )
        x = x[:, :, : num_chunks * d - pad_d].contiguous()
        return x


if __name__ == "__main__":

    cfg = {}
    chunk_size = 32
    pretrained_model = "/home/fengchan/stor/pretrained_models/mvit/k400.pyth"
    model = ChunkMVit(
        chunk_size=chunk_size,
        pretrained_model=pretrained_model,
        forward_mode="batch",
        model_cfg={},
    )
    model.cuda()
    for i in range(10):
        x = torch.rand([1, 3, 32, 224, 224]).cuda()
        x = model(x)
        print(x.shape)
        loss = x.mean()
        loss.backward()
