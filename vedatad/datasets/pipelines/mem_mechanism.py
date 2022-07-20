# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import json
import os

import numpy as np

from vedacore.misc import registry
from vedatad.partial_feedback import indice_selection
from vedatad.partial_feedback.memory_bank import load_features
from vedatad.partial_feedback.indice_selection import indice_chunk_to_frame


@registry.register_module("pipeline")
class ChunkFreeze(object):

    """Select the index of chunks with bp and w/o. bp.

    Args:
        chunk_size (int): The chunk size to divide inputs
        keep_ratio (float): The ratio of the chunks not freezed.
        mode (str): The mode to select the bp-chunks. One of:
                        - "random": randomly selection.
                        - "uniform": select the indexs with the same intervals.
                        - "segments_2v1": Select the bp-chunks 2:1 ratio inside segments and background.

    """

    def __init__(self, num_frames, chunk_size, keep_ratio, mode):
        self._num_frames = num_frames
        self._chunk_size = chunk_size
        self._keep_ratio = keep_ratio
        self._mode = mode

    def __call__(self, results):
        """Call function to generate the keep_indices and drop_indices.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: The original results + 'drop_indices' and 'keep_indices' keys are added into result dict.

        """
        num_chunks = self._num_frames // self._chunk_size
        if self._mode == "random":
            keep_indices, drop_indices = indice_selection.random_selection(
                num_chunks, self._keep_ratio
            )
        else:
            raise ValueError(f"mode:{self._mode} not implemented")

        results["keep_indices"] = keep_indices
        results["drop_indices"] = drop_indices
        return results


@registry.register_module("pipeline")
class LoadFeatureFromMemBank(object):

    """load features from memory bank for the drop chunks"""

    def __init__(self, meta_file, mem_bank_dir, chunk_size, feat_downsample):
        """"""
        self.mem_bank_dir = mem_bank_dir
        self.chunk_size = chunk_size
        self.downsample = feat_downsample

        with open(meta_file, "r") as f:
            self.metas = json.load(f)

    def __call__(self, results):
        """call function to load the features from memory bank.

        Args:
            results (TODO): TODO

        Returns: TODO

        """
        video_name = results["video_info"]["video_name"]
        features = load_features(
            mem_bank_file=os.path.join(self.mem_bank_dir, video_name + ".mmap"),
            shape=self.metas[video_name]["feat_shape"],
            chunk_ids=results["drop_indices"],
            chunk_size=self.chunk_size // self.downsample,
            f_offset=results["tshift"] // self.downsample,
        )  # [num_chunks, C, chunk_size]
        results["frozen_features"] = features

        bp_frame_ids = indice_chunk_to_frame(results["keep_indices"], self.chunk_size)
        results["imgs"] = results["imgs"][bp_frame_ids]
        return results
