import os

import torch
import torch.nn.functional as F
from einops.einops import rearrange
from timesformer.models.vit import TimeSformer

from vedacore.misc import registry


@registry.register_module("backbone")
class ChunkTimeSformer(TimeSformer):
    """extract feature chunk wise"""

    def __init__(self, chunk_size, *args, forward_mode="batch", **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        self.forward_mode = forward_mode
        pretrained_weights = kwargs["pretrained_model"]
        info = self.load_state_dict(
            torch.load(pretrained_weights, map_location="cpu")["model_state"],
            strict=False,
        )
        print(f"Load checkpoint {os.path.basename(pretrained_weights)}: {info}.")

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
        feat = self.model.forward_features(x, temporal_pooling=False)
        feat = feat[:, 1:]  # [b, (n t), m]
        feat = rearrange(feat, "b (n t) m -> b n t m", t=self.chunk_size)
        feat = feat.mean(1)  # [b, t, m]
        feat = rearrange(feat, "b t m -> b m t")
        return feat

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
