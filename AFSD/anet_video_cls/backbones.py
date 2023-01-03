import torch
from torch import nn


class ChunkWrapper(nn.Module):
    """extract feature chunk wise"""

    def __init__(
        self,
        backbone_model: nn.Module,
        chunk_size: int,
        shift_inp=False,
        t_downsample=2,
        do_pooling=False,
    ):
        super().__init__()
        self.backbone = backbone_model
        self.pool = torch.nn.AdaptiveAvgPool3d([None, 1, 1]) if do_pooling else None

        self.shift_inp = shift_inp
        self.t_downsample = t_downsample
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): inputs in two format:
                    1) chunk-first. shape: (num_chunks, B, C, chunk_size, H, W).
                    Returned feature is also chunk-first.
                    2) batch-first. shape: (B, C, D, H, W). Returned feature is
                    batch-first.
        """
        if x.dim() == 6:  # chunk first
            return self.forward_chunk_inp_output(x)
        elif x.dim() == 5:  # batch-first
            if self.shift_inp:
                pad_l = int(torch.randint(1, self.chunk_size, [], dtype=torch.int))
                pad_r = self.chunk_size - pad_l
                pad_l_data = torch.tile(x[:, :, 0:1, :, :], (1, 1, pad_l, 1, 1))
                pad_r_data = torch.tile(x[:, :, -1:, :, :], (1, 1, pad_r, 1, 1))
                x = torch.cat([pad_l_data, x, pad_r_data], 2)
            f = self.forward_nochunk_inp_output(x)

            if self.shift_inp:
                pad_f_l = pad_l // self.t_downsample
                unpadded_d = f.shape[2] - self.chunk_size // self.t_downsample
                f = f[:, :, pad_f_l : pad_f_l + unpadded_d]

            return f

    def forward_chunk_inp_output(self, x):
        """
        Args:
            x (torch.Tensor): input with shape (num_chunks, B, C, chunk_size, H, W)

        Returns: torch.Tensor. The extract features.
            If `do_pooling` is True, the returned shape is (num_chunks, B, C, feat_chunk_size),
            else is (num_chunks, B, C, feat_chunk_size, H', W')
        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(num_chunks * B, C, chunk_size, H, W)
        x = self.backbone(x)  # shape: [n, c, d, h, w]
        _, c, d, h, w = x.shape
        return_shape = (num_chunks, B, c, d, h, w)
        if self.pool:
            x = self.pool(x)
            x = x.squeeze(-1).squeeze(-1)
            return_shape = (num_chunks, B, c, d)
        return x.reshape(return_shape)

    def forward_nochunk_inp_output(self, x):
        """input is not chunk-first. During forward, first divide input into chunks
        and then first each chunks and reshape the output back to original format.

        Args:
            x (torch.Tensor): input with shape (B,C,D,H,W).

        Returns: torch.Tensor. The extract features.
            If `do_pooling` is True, the returned shape is (B, C1, D1),
            else is (B, C1, D1, H1, W1)
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
        x = self.backbone(x)  # shape: [n, c, d, h, w]
        _, c, d = x.shape[:3]
        other_dims = tuple(x.shape[3:])
        x = (
            x.reshape((B, num_chunks, c, d) + other_dims)
            .permute((0, 2, 1, 3) + tuple(range(4, x.ndim)))
            .reshape((B, c, num_chunks * d) + other_dims)  # shape: [B, c, D//2, h, w]
        )
        x = x[:, :, : num_chunks * d - pad_d // 2].contiguous()

        if self.pool:
            x = self.pool(x)
            x = x.squeeze(-1).squeeze(-1)
        return x
