# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import numpy as np


def get_feat_ids(chunk_ids, chunk_size, f_offset):
    """get the ids of features.

    Args:
        chunk_ids (List): The index of chunks.
        chunk_size (int): The size of chunk.
        f_offset (int): The offset of the chunks.

    Returns: numpy.array[N,]. The ids of features.

    """
    feat_ids = []
    for i in chunk_ids:
        feat_ids.append(np.arange(i * chunk_size, (i + 1) * chunk_size))
    feat_ids = np.concatenate(feat_ids) + f_offset
    return feat_ids


def load_features(mem_bank_file, shape, chunk_ids, chunk_size, f_offset):
    """load features from memory bank.

    Args:
        mem_bank_file (str): The memory bank file.
        shape (tuple): The shape of the mmap file.
        chunk_ids (List): The index of chunks.
        chunk_size (int): The size of chunk.
        f_offset (int): The offset of the chunks.

    Returns: numpy.array[num_chunks, C, chunk_size]. The features.

    """
    if isinstance(mem_bank_file, str):
        fp = np.memmap(mem_bank_file, dtype="float32", mode="r", shape=tuple(shape))
    else:
        fp = mem_bank_file
    feat_ids = get_feat_ids(chunk_ids, chunk_size, f_offset)

    num_features = len(fp)
    if max(feat_ids) >= num_features or min(feat_ids) < 0:
        idx = feat_ids[feat_ids < num_features]
        pad_r = len(feat_ids) - len(idx)
        new_idx= idx[idx >= 0]
        pad_l = len(idx) - len(new_idx)

        features = fp[new_idx.tolist()].copy()
        features = np.pad(features, ((pad_l, pad_r), (0, 0)))  # [N,C]
    else:
        features = fp[feat_ids.tolist()].copy()
    features = np.reshape(features, (len(chunk_ids), chunk_size, -1))
    features = features.transpose((0, 2, 1))  # [num_chunks, C, chunk_size]
    return features


def write_features(features, mem_bank_file, shape, chunk_ids, chunk_size, f_offset):
    """write features to memory bank.

    Args:
        features (np.array[num_chunks, C, chunk_size]): The features to write.
        mem_bank_file (str): The memory bank file.
        shape (tuple): The shape of the mmap file.
        chunk_ids (List): The index of chunks.
        chunk_size (int): The size of chunk.
        f_offset (int): The offset of the chunks.

    """
    if isinstance(mem_bank_file, str):
        fp = np.memmap(mem_bank_file, dtype="float32", mode="r+", shape=tuple(shape))
    else:
        fp = mem_bank_file
    feat_ids = get_feat_ids(chunk_ids, chunk_size, f_offset)

    # reshape features [num_chunks, C, chunk_size] -> [num_chunks*chunk_size, C]
    features = features.transpose((0, 2, 1)).reshape((len(feat_ids), -1))

    ids = feat_ids[feat_ids < len(fp)]
    new_ids = ids[ids >= 0]
    pad_l = len(ids) - len(new_ids)
    fp[new_ids.tolist()] = features[pad_l:pad_l+len(new_ids)]
    if isinstance(mem_bank_file, str):
        fp.flush()
