from functools import lru_cache
from math import ceil, floor

import torch


@lru_cache
def biband_mask(n: int, kernel_size: int, device: torch.device, v=-1e9):
    """compute mask for local attention with kernel size.

    Args:
        n (torch.Tensor): the input length.
        kernel_size (int): The local attention kernel size.
        device (torch.device): transformer mask to the device.

    Returns: torch.Tensor. shape: [n,n]. The masked locations are -1e9
        and unmasked locations are 0.

    """
    if kernel_size is None:
        return None
    half = kernel_size // 2
    mask1 = torch.ones(n, n).triu(diagonal=-half)
    mask2 = torch.ones(n, n).tril(diagonal=half)
    mask = mask1 * mask2
    mask = (1 - mask) * v
    return mask.to(device)


@lru_cache
def q_k_relation_mask(
    q_len: int, k_len: int, device: torch.device, expand: float = 0.5, v: float = -1e9
):

    """generate mask that query only attention to related key.

    Args:
        q_len (int): the query length.
        k_len (int): the key length.
        device (torch.device): device

    Kwargs:
        expand (float): expand the attention areas by `expand * ratio`. The `ratio` is `k_len / q_len`.
        v (float): negative value that small enough.

    Returns: torch.Tensor. The mask of shape (q_len, k_len).

    """
    mask = torch.ones(q_len, k_len, dtype=torch.float32)

    ratio = float(k_len) / q_len
    for i in range(q_len):
        mask[
            i,
            max(floor(i * ratio - expand * ratio), 0) : min(
                ceil((i + 1) * ratio + expand * ratio), k_len
            ),
        ] = 0
    mask *= v
    return mask.to(device)
