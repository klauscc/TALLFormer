import copy

import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input with shape: [B,D,C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention1D(nn.Module):

    """attention with relative positional encoding"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """init function

        Args:
            dim (int): feature dimension.
            num_heads (int): attention head.
            max_seq_len (int):  maximum sequence length.

        Kwargs:
            qkv_bias (bool): Default to False.
            qk_scale (float): If None, the value is `sqrt(dim)`.
            attn_drop (float): Default to 0.
            proj_drop (float): Defualt to 0.

        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.zeros(2 * max_seq_len - 1, num_heads)
        )  # 2*t-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(max_seq_len)

        relative_coords = coords_d[:, None] - coords_d[None, :]  # [seq_len, seq_len]
        relative_coords = relative_coords + (
            max_seq_len - 1
        )  # normalize  [0,2*seq_len-1)
        self.register_buffer("relative_position_index", relative_coords)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: torch.Tensor = None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(
            N, N, -1
        )  # seq_len,seq_len,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH,seq_len,seq_len
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            if mask.ndim == 2:  # [N,N]
                mask = mask.unsqueeze(0).unsqueeze(0)  # 1,1,N,N
                attn = attn + mask
                attn = self.softmax(attn)
            else:  # [nW,N,N]
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                    1
                ).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderLayer1D(nn.Module):

    """swin transformer 1D"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        """TODO: to be defined."""
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention1D(
            dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """forward function

        Args:
            x (torch.Tensor): input with shape [B,D,C]
            mask (torch.Tensor): attention mask. shape [D,D].

        Returns: torch.Tensor. The same shape as input [B,D,C]

        """
        shortcut = x
        # attn
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = shortcut + self.drop_path(x)
        # mlp
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):

    """Transformer_1D encoder"""

    def __init__(self, encoder_layer, num_layers):
        """

        Args:
            encoder_layer (TODO): TODO
            num_layers (TODO): TODO


        """
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """forward function

        Args:
            x (torch.Tensor): input with shape [B,T,C]

        Returns: torch.Tensor. The same shape as input: [B,T,C]

        """
        for mod in self.layers:
            x = mod(x, mask)
        return x


def dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_bias: torch.Tensor,
    attn_drop_layer: nn.Module,
    mask: torch.Tensor = None,
):
    """dot product attention

    Args:
        q (torch.Tensor): query with shape [B,nH,Nq,C]
        k (torch.Tensor): key with shape [B,nH,Nk,C]
        v (torch.Tensor): value with shape [B,nH,Nk,C]
        pos_bias (torch.Tensor): positional bias. shape: [nH,Nq,Nk]
        attn_drop_layer (nn.Module): The dropout layer for attention weight.

    Kwargs:
        mask (torch.Tensor): attention mask. shape: [nW,Nq,Nk]

    Returns: TODO

    """
    B_, nH, Nq, head_dim = q.shape
    Nk = k.shape[2]
    attn = q @ k.transpose(-2, -1)
    attn = attn + pos_bias.unsqueeze(0)  # B_, nH, Nq, Nk

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, nH, Nq, Nk) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, nH, Nq, Nk)
        attn = attn.softmax(dim=-1)
    else:
        attn = attn.softmax(dim=-1)

    attn = attn_drop_layer(attn)

    C = nH * head_dim
    x = (attn @ v).transpose(1, 2).reshape(B_, Nq, C)
    return x


class CrossAttention1D(nn.Module):

    """cross attention 1D."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """init function

        Args:
            dim (int): feature dimension.
            num_heads (int): attention head.
            max_seq_len (int):  maximum sequence length.

        Kwargs:
            qkv_bias (bool): Default to False.
            qk_scale (float): If None, the value is `sqrt(dim)`.
            attn_drop (float): Default to 0.
            proj_drop (float): Defualt to 0.

        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.zeros(2 * max_seq_len - 1, num_heads)
        )  # 2*t-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(max_seq_len)

        relative_coords = coords_d[:, None] - coords_d[None, :]  # [seq_len, seq_len]
        relative_coords = relative_coords + (
            max_seq_len - 1
        )  # normalize  [0,2*seq_len-1)
        self.register_buffer("relative_position_index", relative_coords)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(
            N, N, -1
        )  # seq_len,seq_len,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH,seq_len,seq_len
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
