# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from functools import lru_cache
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from vedacore.misc import registry
from vedacore.modules import ConvModule, xavier_init
from vedatad.models.modules.swin_1d import EncoderLayer1D
from vedatad.models.modules.transformer import TransformerEncoderLayer


@registry.register_module("neck")
class FPN(nn.Module):
    """Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        extra_convs_on_inputs=True,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = "on_input"
            else:
                self.add_extra_convs = "on_output"

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, _ConvNd):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                # This is a workaround when converting PyTorch model
                # to ONNX model
                prev_shape = tuple(map(lambda x: int(x), prev_shape))
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@registry.register_module("neck")
class SelfAttnFPN(nn.Module):

    """self-attention FPN"""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_heads: int = 8,
        kernel_size: int = None,
        max_seq_len: int = -1,
        trans_layer: str = "traditional",
        input_order="tbc",
        out_order="tbc",
    ):
        """TODO: to be defined."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_order = out_order
        self.input_order = input_order
        self.kernel_size = kernel_size
        self.trans_layer = trans_layer

        num_outs = len(in_channels)
        self.num_outs = num_outs

        assert out_order in ["tbc", "bct"], "output order must be `tbc` or `bct`"

        # build lateral proj layers
        self.lateral_projs = nn.ModuleList()
        for i in range(num_outs):
            proj = nn.Sequential(
                nn.Linear(in_channels[i], out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
            )
            self.lateral_projs.append(proj)

        # build transformer layers.
        self.trans_layers = nn.ModuleList()
        for i in range(num_outs):
            if trans_layer == "traditional":
                layer = TransformerEncoderLayer(
                    out_channels, num_heads, out_channels * 4
                )
            elif trans_layer == "swin_1d":
                layer = EncoderLayer1D(out_channels, num_heads, max_seq_len=max_seq_len)
            else:
                raise ValueError(f"trans_layer = {trans_layer} is not supported")
            self.trans_layers.append(layer)

    def init_weights(self):
        pass

    @lru_cache
    def compute_mask(self, n: int, kernel_size: int, device: torch.device, v=-1e9):
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

    def forward(self, inps: Sequence[torch.Tensor]):
        """forward function

        Args:
            inps (Sequence[torch.Tensor]): multi-level features. order: tbc.

        Returns: Sequence[torch.Tensor]. output FPN features. Each one with order `self.out_order`.

        """

        if self.input_order == "bct":
            inps = [x.permute(2, 0, 1) for x in inps]

        laterals = [proj(x) for x, proj in zip(inps, self.lateral_projs)]
        for i in range(len(laterals)):
            laterals[i] = laterals[i].permute(1, 2, 0)  # [B,C,T]

        for i in range(self.num_outs - 1, 0, -1):
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], size=int(laterals[i - 1].shape[2]), mode="nearest"
            # )

            prev_shape = laterals[i - 1].shape[2:]
            prev_shape = tuple(map(lambda x: int(x), prev_shape))
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="linear"
            )

        for i in range(len(laterals)):
            laterals[i] = laterals[i].permute(2, 0, 1)  # [T,B,C]

        outs = []
        for lateral, trans_layer in zip(laterals, self.trans_layers):
            with torch.no_grad():
                mask = self.compute_mask(
                    lateral.shape[0], self.kernel_size, lateral.device
                )
            if self.trans_layer == "swin_1d":
                lateral = lateral.permute(1,0,2) #[T,B,C]->[B,T,C]
                out = trans_layer(lateral, mask)  # [B,T,C]
                out = out.permute(1,0,2)  # [T,B,C]
            else:
                out = trans_layer(lateral, mask)  # [T,B,C]
            outs.append(out)

        if self.out_order == "bct":
            for i in range(len(outs)):
                outs[i] = outs[i].permute(1, 2, 0)
        return outs
