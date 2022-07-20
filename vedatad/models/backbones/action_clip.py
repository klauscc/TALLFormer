import os

import torch
from clip.model import VisualTransformer
from einops.einops import rearrange
from torch import nn

from vedacore.misc import registry


def build_vit(ckpt_path):
    """build vit from ckpt.

    Args:
        ckpt_path (str): the pretrained checkpoint path.

    Returns: nn.Module. The vit model.

    """
    state_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_heads = vision_width // 64
    vision_layers = len(
        [
            k
            for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]

    model = VisualTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
        joint=False,
        dropout=None,
        emb_dropout=0,
    )

    vis_state_dict = {}
    for k, v in state_dict.items():
        if "visual." in k:
            vis_state_dict[k.replace("visual.", "")] = v

    info = model.load_state_dict(vis_state_dict, strict=False)
    print(f"load ckpt {os.path.basename(ckpt_path)}: {info}")
    return model


@registry.register_module("backbone")
class ChunkActionClip(nn.Module):
    """extract feature chunk wise"""

    def __init__(
        self, chunk_size, pretrained_model, *args, forward_mode="batch", **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert chunk_size == 1, "chunk size must be 1 for action clip"
        self.chunk_size = chunk_size
        self.forward_mode = forward_mode

        self.vit = build_vit(pretrained_model)

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
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        feat = self.vit(x)  # bxt, c'
        feat = rearrange(feat, "(b t) c -> b c t", b=b)
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


if __name__ == "__main__":

    model = ChunkActionClip(
        chunk_size=1,
        pretrained_model="data/pretrained_models/action-clip/vit-b-16-32f.pt",
    )
    model.cuda()
    x = torch.rand([2, 3, 32, 224, 224]).cuda()
    print(model(x).shape)
