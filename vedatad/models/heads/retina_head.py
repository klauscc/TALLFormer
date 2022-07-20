# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from einops.einops import rearrange
import torch.nn as nn

from vedacore.misc import registry
from vedacore.modules import ConvModule, bias_init_with_prob, normal_init
from vedatad.models.modules.masking import biband_mask
from vedatad.models.modules.swin_1d import Encoder, EncoderLayer1D
from .anchor_head import AnchorHead


@registry.register_module("head")
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7, 7, conv_cfg=dict(typename='Conv1d'))
        >>> x = torch.rand(1, 7, 32)
        >>> cls_score, seg_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> seg_per_anchor = seg_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert seg_per_anchor == 2
    """

    def __init__(
        self,
        num_classes,
        num_anchors,
        in_channels,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=None,
        **kwargs
    ):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes, num_anchors, in_channels, **kwargs
        )

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.retina_cls = nn.Conv1d(
            self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1
        )
        self.retina_reg = nn.Conv1d(
            self.feat_channels, self.num_anchors * 2, 3, padding=1
        )

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                segment_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 2.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        segment_pred = self.retina_reg(reg_feat)
        return cls_score, segment_pred


@registry.register_module("head")
class SARetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7, 7, conv_cfg=dict(typename='Conv1d'))
        >>> x = torch.rand(1, 7, 32)
        >>> cls_score, seg_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> seg_per_anchor = seg_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert seg_per_anchor == 2
    """

    def __init__(
        self,
        num_classes,
        num_anchors,
        in_channels,
        kernel_size=5,
        num_layers=4,
        num_heads=8,
        max_seq_len=100,
        drop_path=0.1,
        in_order = "b c t",
        out_order = "b c t",
        **kwargs
    ):
        self.num_layers = num_layers
        self.kernel_size=kernel_size
        self.num_heads = num_heads
        self.drop_path = drop_path
        self.max_seq_len = max_seq_len
        self.in_order = in_order
        self.out_order = out_order
        super(SARetinaHead, self).__init__(
            num_classes, num_anchors, in_channels, **kwargs
        )

    def _init_layers(self):
        """Initialize layers of the head."""

        self.cls_layers = Encoder(
            EncoderLayer1D(
                self.in_channels,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
            ),
            num_layers=self.num_layers,
        )

        self.reg_layers = Encoder(
            EncoderLayer1D(
                self.in_channels,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
            ),
            num_layers=self.num_layers,
        )

        self.retina_cls = nn.Conv1d(
            self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1
        )
        self.retina_reg = nn.Conv1d(
            self.feat_channels, self.num_anchors * 2, 3, padding=1
        )

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                segment_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 2.
        """
        

        if self.in_order != "b t c":
            x = rearrange(x, f"{self.in_order} -> b t c")

        mask = biband_mask(x.shape[1], self.kernel_size, x.device)

        cls_feat = self.cls_layers(x, mask)
        reg_feat = self.reg_layers(x, mask)

        cls_feat = rearrange(cls_feat, "b t c -> b c t")
        reg_feat = rearrange(reg_feat, "b t c -> b c t")

        cls_score = self.retina_cls(cls_feat)
        segment_pred = self.retina_reg(reg_feat)
        return cls_score, segment_pred
