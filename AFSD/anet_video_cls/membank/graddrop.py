import numpy as np


def generate_indices(
    num_chunks: int, keep_ratio: float, max_keep_id: int = -1, mode="uniform_jitter"
):
    """generate indices of inputs that need and don't need gradients.

    Args:
        num_chunks (int): number of chunks
        keep_ratio (float): keep ratio.
        max_keep_id (int): The max id in keep_indices.

    Returns: Tuple: (keep_indices, drop_indices).

    """
    assert (
        max_keep_id < num_chunks
    ), f"max_keep_id: {max_keep_id} should be less than num_chunks: {num_chunks}"

    if max_keep_id == -1:
        max_keep_id = num_chunks - 1

    num_to_keep = int(num_chunks * keep_ratio)

    if num_to_keep > max_keep_id:
        keep_indices = list(range(num_to_keep))
    else:
        if mode == "uniform_jitter":
            keep_indices = np.linspace(0, max_keep_id, num_to_keep, endpoint=False)
            jitters = np.random.uniform(
                0, max_keep_id / (max_keep_id + 1) / keep_ratio, size=len(keep_indices)
            )
            keep_indices = keep_indices + jitters
            keep_indices = keep_indices.astype(np.int64)
            keep_indices = np.clip(keep_indices, 0, max_keep_id).tolist()
            keep_indices = list(set(keep_indices))  # remove duplication.
        else:
            raise ValueError(f"generate_indices: mode:{mode} not implemented")

    drop_indices = []
    for i in range(num_chunks):
        if i not in keep_indices:
            drop_indices.append(i)

    return keep_indices, drop_indices
