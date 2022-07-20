# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import numpy as np


def indice_chunk_to_frame(chunk_ids, chunk_size):
    """convert chunk_ids to frame_ids

    Args:
        chunk_ids (List): chunk_ids.
        chunk_size (int): chunk size.

    Returns: numpy.array[N,]. The frame ids of these chunks.

    """
    frame_ids = []
    for i in chunk_ids:
        frame_ids.append(np.arange(i * chunk_size, (i + 1) * chunk_size))
    frame_ids = np.concatenate(frame_ids)
    return frame_ids


def generate_indices(num_frames, chunk_size, keep_ratio, mode):
    """generate the keep_indices and drop_indices (chunkwise).

    Args:
        num_frames (int): number of frames.
        chunk_size (int): The chunk size to divide inputs.
        keep_ratio (float): The ratio of the chunks not freezed.
        mode (str): The mode to select the bp-chunks. One of:
                        - "random": randomly selection.
                        - "uniform": select the indexs with the same intervals.
                        - "segments_2v1": Select the bp-chunks 2:1 ratio inside segments and background.

    Returns: (keep_indices, drop_indices). Each is a list.

    """
    num_chunks = num_frames // chunk_size
    if keep_ratio == 0:
        keep_indices = []
        drop_indices = list(range(num_chunks))
        return keep_indices, drop_indices

    if mode == "random":
        return random_selection(num_chunks, keep_ratio)
    if mode == "uniform":
        return uniform_selection(num_chunks, keep_ratio)
    else:
        raise ValueError(f"mode:{mode} not implemented")


def random_selection(num_chunks, keep_ratio):
    """randomly select the keep indices.

    Args:
        num_chunks (int): The total number of chunks
        keep_ratio (float): The ratio of bp-chunks.

    Returns: (keep_indices, drop_indices). Each is a list.

    """
    keep_indices = np.random.choice(
        np.arange(num_chunks),
        size=int(num_chunks * keep_ratio),
        replace=False,
    ).tolist()
    keep_indices = sorted(keep_indices)
    drop_indices = get_drop_indices(num_chunks, keep_indices)
    return keep_indices, drop_indices


def uniform_selection(num_chunks, keep_ratio):
    """uniformly select the keep indices.

    Args:
        num_chunks (int): The total number of chunks
        keep_ratio (float): The ratio of bp-chunks.

    Returns: (keep_indices, drop_indices). Each is a list.

    """
    keep_indices = (
        np.floor(np.linspace(0, num_chunks - 1, int(num_chunks * keep_ratio)))
        .astype(np.int64)
        .tolist()
    )
    keep_indices = sorted(keep_indices)
    drop_indices = get_drop_indices(num_chunks, keep_indices)
    return keep_indices, drop_indices


def get_drop_indices(num_chunks, keep_indices):
    """obtain the drop_indices given the keep_indices.

    Args:
        num_chunks (int): The total number of chunks
        keep_indices (List): The keep indices.

    Returns: List. The dropped indices (frozen chunks).

    """
    drop_indices = []
    for i in range(num_chunks):
        if i not in keep_indices:
            drop_indices.append(i)
    return drop_indices
