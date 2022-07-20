import json
import os
from time import time

import numpy as np
import torch
from easydict import EasyDict

from vedacore.misc import registry
from vedacore.optimizers import build_optimizer
from vedatad.criteria import build_criterion
from vedatad.partial_feedback.indice_selection import generate_indices
from vedatad.partial_feedback.memory_bank import load_features, write_features
from .base_engine import BaseEngine


@registry.register_module("engine")
class TrainEngine(BaseEngine):
    def __init__(self, model, criterion, optimizer):
        super().__init__(model)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

    def extract_feats(self, img):
        feats = self.model(img, train=True)
        return feats

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(
        self, imgs, video_metas, gt_segments, gt_labels, gt_segments_ignore=None
    ):
        feats = self.extract_feats(imgs)
        losses = self.criterion.loss(
            feats, video_metas, gt_segments, gt_labels, gt_segments_ignore
        )
        return losses


@registry.register_module("engine")
class MemBankTrainEngine(BaseEngine):
    def __init__(self, membank, model, criterion, optimizer):
        super().__init__(model)
        self.membank = EasyDict(membank)
        self.criterion = build_criterion(criterion)
        self.optimizer = build_optimizer(self.model, optimizer)

        if hasattr(self.membank, "frozen") and self.membank["frozen"]:
            self.frozen = True
        else:
            self.frozen = False

        with open(self.membank["mem_bank_meta_file"], "r") as f:
            self.membank_metas = json.load(f)

    def extract_feats(self, imgs, frozen_features, keep_indices, drop_indices):
        final_feats, update_feats = self.model(
            imgs, frozen_features, keep_indices, drop_indices, train=True
        )
        return final_feats, update_feats

    def forward(self, data):
        return self.forward_impl(**data)

    def forward_impl(
        self, imgs, video_metas, gt_segments, gt_labels, gt_segments_ignore=None
    ):
        B, C, D, H, W = imgs.shape
        chunk_size = self.membank["chunk_size"]

        keep_indices, drop_indices = generate_indices(
            D, chunk_size, self.membank.keep_ratio, self.membank.mode
        )

        # get bp-chunks
        num_chunks = D // chunk_size
        imgs = imgs.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # shape: (num_chunks, B, C , chunk_size, H,W)
        imgs = imgs[keep_indices].contiguous()

        # get features from membank
        t1 = time()
        frozen_features = []
        feat_chunk_size = chunk_size // self.membank["feat_downsample"]
        for video_meta in video_metas:
            video_name, tshift = video_meta["video_name"], video_meta["tshift"]
            video_name = os.path.basename(video_name)
            f_offset = tshift // self.membank["feat_downsample"]
            mem_bank_file = os.path.join(
                self.membank["mem_bank_dir"], video_name + ".mmap"
            )

            membank_shape = tuple(self.membank_metas[video_name]["feat_shape"])

            feature = load_features(
                mem_bank_file=mem_bank_file,
                shape=membank_shape,
                chunk_ids=drop_indices,
                chunk_size=feat_chunk_size,
                f_offset=f_offset,
            )
            frozen_features.append(feature)
        frozen_features = np.stack(
            frozen_features, axis=1
        )  # [num_nobp_chunks, B, C, feat_chunk_size]
        frozen_features = torch.from_numpy(frozen_features).to(imgs.device)
        t2 = time()
        # print(f"load frozen features cost {t2-t1}s")

        # extract features
        final_feats, update_feats = self.extract_feats(
            imgs, frozen_features, keep_indices, drop_indices
        )

        # update membank

        ## shape: [B, num_keep_chunks, C, feat_chunk_size]
        t3 = time()
        if (not self.frozen) and len(keep_indices) != 0:
            update_feats = update_feats.detach().cpu().numpy()
            for i, video_meta in enumerate(video_metas):
                video_name, tshift = video_meta["video_name"], video_meta["tshift"]
                video_name = os.path.basename(video_name)
                f_offset = tshift // self.membank["feat_downsample"]
                mem_bank_file = os.path.join(
                    self.membank["mem_bank_dir"], video_name + ".mmap"
                )
                membank_shape = tuple(self.membank_metas[video_name]["feat_shape"])

                write_features(
                    features=update_feats[:, i],
                    mem_bank_file=mem_bank_file,
                    shape=membank_shape,
                    chunk_ids=keep_indices,
                    chunk_size=feat_chunk_size,
                    f_offset=f_offset,
                )
            t4 = time()
            # print(f"update memory bank cost {t4-t3}s")

        losses = self.criterion.loss(
            final_feats, video_metas, gt_segments, gt_labels, gt_segments_ignore
        )
        return losses
