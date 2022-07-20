from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from vedacore.misc import registry
from vedatad.partial_feedback import indice_selection
from ..builder import build_backbone, build_head, build_neck
from .base_detector import BaseDetector


@registry.register_module("detector")
class MemSingleStageDetector(BaseDetector):
    def __init__(self, chunk_size, backbone, head, neck=None):
        super().__init__()
        self.chunk_size = chunk_size
        self.backbone = build_backbone(backbone)
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.head = build_head(head)

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        self.head.init_weights()

    def forward_impl(self, x, frozen_features, keep_indices, drop_indices):
        """
        Args:
            x (torch.Tensor[num_chunks,B,C,chunk_size,H,W]): The imgs for bp chunks.
            frozen_features (torch.Tensor[num_nobp_chunks, B,C,feat_chunk_size]): The features for non-bp chunks.

        Returns: (feats, bp_features). `feat` is the final features by the entire model. `bp_features` is the features of bp-chunks from the backbone.
        """
        # extract the features for bp-chunks
        num_bp_chunks = x.shape[0]
        num_nobp_chunks = frozen_features.shape[0]
        num_chunks = num_bp_chunks + num_nobp_chunks

        if len(keep_indices) != 0:
            bp_features = self.backbone(x)  # [num_keep_chunks, B, C, feat_chunk_size]
            # compose features
            feats = torch.zeros(
                [num_chunks] + list(bp_features.shape[1:]),
                dtype=bp_features.dtype,
                device=bp_features.device,
            )
            feats[keep_indices] = bp_features
            feats[drop_indices] = frozen_features
        else:
            feats = frozen_features
            bp_features = None
        ## [num_chunks, B, C, feat_chunk_size] -> [B, C, num_chunks*feat_chunk_size]
        num_chunks, B, C, feat_chunk_size = feats.shape
        feats = feats.permute(1, 2, 0, 3).reshape(B, C, num_chunks * feat_chunk_size)

        if self.neck:
            feats = self.neck(feats)
        feats = self.head(feats)
        return feats, bp_features

    def forward_eval(self, x):
        feats = self.backbone(x)
        if self.neck:
            feats = self.neck(feats)
        feats = self.head(feats)
        return feats

    def forward(
        self, x, frozen_features=None, keep_indices=None, drop_indices=None, train=True
    ):
        if train:
            self.train()
            feats = self.forward_impl(x, frozen_features, keep_indices, drop_indices)
        else:
            self.eval()
            with torch.no_grad():
                feats = self.forward_eval(x)
        return feats


@registry.register_module("detector")
class MomentMemSingleStageDetector(BaseDetector):
    def __init__(
        self,
        chunk_size,
        momentum,
        backbone,
        head,
        neck=None,
        keep_ratio=0,
        eval_mode="momentum",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.momentum = momentum
        self.backbone = build_backbone(backbone)

        self.keep_ratio = keep_ratio
        self.eval_mode = eval_mode
        if self.eval_mode == "combine" and self.keep_ratio == 0:
            self.eval_mode = "momentum"

        print(f"eval_mode: {self.eval_mode}. keep_ratio: {self.keep_ratio}")

        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.head = build_head(head)

        self.m_backbone = deepcopy(self.backbone)
        self.m_backbone_initialized = False
        for param in self.m_backbone.parameters():
            param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        self.head.init_weights()

    @torch.no_grad()
    def update_momentum_backbone(self):
        """update the parameters of m_backbone."""
        if not self.m_backbone_initialized:
            for m_param, param in zip(
                self.m_backbone.parameters(), self.backbone.parameters()
            ):
                m_param.data[:] = param.data[:]
            self.m_backbone_initialized = True
        else:
            diff = []
            for m_param, param in zip(
                self.m_backbone.parameters(), self.backbone.parameters()
            ):
                m_param.data = (
                    self.momentum * m_param.data + (1 - self.momentum) * param.data
                )
                diff.append((m_param - param).abs().mean().detach().item())
            diff = np.array(diff).mean()
            print(f"update: diff:{diff}")

    def forward_impl(self, x, frozen_features, keep_indices, drop_indices):
        """
        Args:
            x (torch.Tensor[num_chunks,B,C,chunk_size,H,W]): The imgs for bp chunks.
            frozen_features (torch.Tensor[num_nobp_chunks, B,C,feat_chunk_size]): The features for non-bp chunks.

        Returns: (feats, bp_features). `feat` is the final features by the entire model. `bp_features` is the features of bp-chunks from the backbone.
        """
        # extract the features for bp-chunks
        num_bp_chunks = x.shape[0]
        num_nobp_chunks = frozen_features.shape[0]
        num_chunks = num_bp_chunks + num_nobp_chunks
        bp_features = self.backbone(x)  # [num_keep_chunks, B, C, feat_chunk_size]

        # compose features
        feats = torch.zeros(
            [num_chunks] + list(bp_features.shape[1:]),
            dtype=bp_features.dtype,
            device=bp_features.device,
        )
        feats[keep_indices] = bp_features
        feats[drop_indices] = frozen_features
        ## [num_chunks, B, C, feat_chunk_size] -> [B, C, num_chunks*feat_chunk_size]
        num_chunks, B, C, feat_chunk_size = feats.shape
        feats = feats.permute(1, 2, 0, 3).reshape(B, C, num_chunks * feat_chunk_size)

        if self.neck:
            feats = self.neck(feats)
        feats = self.head(feats)

        # calculate features to update memory bank
        with torch.no_grad():
            self.update_momentum_backbone()
            bp_features = self.m_backbone(x)

        return feats, bp_features

    def forward_eval_combined(self, x: torch.Tensor):
        """evaluation use combined backbone.

        Args:
            x (torch.Tensor): input with shape (B,C,D,H,W)

        Returns: TODO

        """
        # divide into chunks
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        num_chunks = D // chunk_size
        x = x.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # (num_chunks, B,C,chunk_size, H, W)
        keep_indices, drop_indices = indice_selection.generate_indices(
            D, chunk_size, keep_ratio=self.keep_ratio, mode="random"
        )
        feat1 = self.backbone(x[keep_indices].contiguous())
        feat2 = self.m_backbone(x[drop_indices].contiguous())
        # compose features
        feats = torch.zeros(
            [num_chunks] + list(feat1.shape[1:]),
            dtype=feat1.dtype,
            device=feat1.device,
        )
        feats[keep_indices] = feat1
        feats[drop_indices] = feat2
        ## [num_chunks, B, C, feat_chunk_size] -> [B, C, num_chunks*feat_chunk_size]
        num_chunks, B, C, feat_chunk_size = feats.shape
        feats = feats.permute(1, 2, 0, 3).reshape(B, C, num_chunks * feat_chunk_size)
        return feats

    def forward_eval(self, x):
        # print(f"eval_mode: {self.eval_mode}. keep_ratio: {self.keep_ratio}")
        if self.eval_mode == "momentum":
            # print("eval with momentum backbone")
            feats = self.m_backbone(x)
        elif self.eval_mode == "vanilla":
            # print("eval with backbone")
            feats = self.backbone(x)
        else:
            # print("eval with combined backbone")
            feats = self.forward_eval_combined(x)

        if self.neck:
            feats = self.neck(feats)
        feats = self.head(feats)
        return feats

    def forward(
        self, x, frozen_features=None, keep_indices=None, drop_indices=None, train=True
    ):
        if frozen_features is None and train:
            raise ValueError("frozen_features should not be none during training.")
        if train:
            self.train()
            feats = self.forward_impl(x, frozen_features, keep_indices, drop_indices)
        else:
            self.eval()
            with torch.no_grad():
                feats = self.forward_eval(x)
        return feats
