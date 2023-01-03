import json
import os

import einops
import torch
import torch.nn.functional as F
from vedatad.partial_feedback.memory_bank import write_features

from AFSD.anet_video_cls.BDNet import BDNet


class BDNetMemBank(BDNet):

    """Docstring for ."""

    def __init__(self, cfg, training=True, frame_num=768):
        """TODO: to be defined."""
        super().__init__(cfg, training, frame_num)
        self.chunk_size = cfg.chunk_size

        self.membank_cfg = cfg.membank_cfg
        self.shift_inp = cfg.membank_cfg.shift_inp
        with open(self.membank_cfg["mem_bank_meta_file"], "r") as f:
            self.membank_metas = json.load(f)

    def inference_backbone(self, x: torch.Tensor):
        return self.backbone(x)

    def forward_backbone(self, x: torch.Tensor, gd: dict = None):
        """TODO: Docstring for forward.

        Args:
            x (torch.Tensor): input frames. Shape: [B,C,T,H,W]
            gd (dict): The data contains the information for memory bank.
                {"keep_idx": keep_idx, # List[Int].
                "drop_idx": drop_idx,  # List[Int].
                "frozen_features": frozen_features, #Tensor. shape: [n_chunk, B, C, f_chunk_size]
                "metas": metas,}

        Returns: torch.Tensor. Bakcbone features with shape [B,C,T_f]

        """
        if gd is None:  # test mode
            return self.inference_backbone(x)

        if self.shift_inp:
            frame_pad_l = gd["frame_pad_l"]
            frame_pad_r = self.chunk_size - frame_pad_l
            x = F.pad(x, pad=(0, 0, 0, 0, frame_pad_l, frame_pad_r, 0, 0, 0, 0))

        ## Training
        # divide x to chunks
        keep_idx = gd["keep_idx"]
        drop_idx = gd["drop_idx"]
        video_metas = gd["metas"]

        x = einops.rearrange(
            x,
            "B C (n_chunks chunk_size) H W -> n_chunks B C chunk_size H W",
            chunk_size=self.chunk_size,
        )

        bp_x = x[keep_idx].contiguous()
        bp_features = self.backbone(bp_x)  # [n_bp, B, C, f_chunk_size]
        nobp_features = gd["frozen_features"]  # [n_nobp, B, C, f_chunk_size]

        num_chunks = bp_features.shape[0] + nobp_features.shape[0]

        # merge features
        backbone_feat = torch.zeros(
            (num_chunks,) + tuple(bp_features.shape[1:]),
            dtype=bp_features.dtype,
            device=bp_features.device,
        )  # [num_chunks, B, C, f_chunk_size]
        backbone_feat[keep_idx] = bp_features
        backbone_feat[drop_idx] = nobp_features
        backbone_feat = einops.rearrange(backbone_feat, "N B C L -> B C (N L)")

        if self.shift_inp:
            # remove padded features
            feat_pad_l = frame_pad_l // self.membank_cfg["t_downsample"]
            padded_feat_length = backbone_feat.shape[-1]
            non_padded_feat_length = (
                padded_feat_length - self.chunk_size // self.membank_cfg["t_downsample"]
            )
            backbone_feat = backbone_feat[:, :, feat_pad_l : feat_pad_l + non_padded_feat_length]

        # update memory bank.
        f_offset = -feat_pad_l if self.shift_inp else 0
        for i, video_meta in enumerate(video_metas):
            video_name = video_meta["video_name"]
            mem_bank_file = os.path.join(self.membank_cfg["mem_bank_dir"], video_name + ".mmap")
            membank_shape = tuple(self.membank_metas[video_name]["feat_shape"])
            feat_chunk_size = self.chunk_size // self.membank_cfg["t_downsample"]
            bp_features = bp_features.detach().cpu().numpy()
            write_features(
                features=bp_features[:, i],
                mem_bank_file=mem_bank_file,
                shape=membank_shape,
                chunk_ids=keep_idx,
                chunk_size=feat_chunk_size,
                f_offset=f_offset,
            )
        return backbone_feat

    def forward(self, x: torch.Tensor, gd: dict, ssl: bool = False):
        f = self.forward_backbone(x, gd)
        return self.forward_detector(f)
