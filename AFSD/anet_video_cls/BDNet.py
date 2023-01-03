from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vedatad.models.builder import build_backbone, build_neck
from vedatad.models.backbones.temp_graddrop import GradDropModel

from AFSD.anet_video_cls.backbones import ChunkWrapper
from AFSD.common.config import config
from AFSD.common.i3d_backbone import InceptionI3d
from AFSD.common.layers import Unit1D, Unit3D
from AFSD.prop_pooling.boundary_pooling_op import BoundaryMaxPooling

num_classes = 2
freeze_bn = config["model"]["freeze_bn"]
freeze_bn_affine = config["model"]["freeze_bn_affine"]
crop_size = config["dataset"]["training"]["crop_size"]

layer_num = 6
conv_channels = 512
fpn_strides = [8, 16, 32, 64, 128, 256]
feat_t = 768 // 8


class I3D_BackBone(nn.Module):
    def __init__(
        self,
        final_endpoint="Mixed_5c",
        name="inception_i3d",
        in_channels=3,
        out_channels=conv_channels,
        freeze_bn=freeze_bn,
        freeze_bn_affine=freeze_bn_affine,
    ):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(
            final_endpoint=final_endpoint, name=name, in_channels=in_channels
        )
        self.final_endpoint = final_endpoint
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

        if crop_size == 96:
            kernel = [1, 3, 3]
        elif crop_size == 224:
            kernel = [1, 7, 7]
        else:
            raise ValueError(f"crop_size {crop_size} is not supported.")
        self.proj_out = nn.Sequential(
            Unit3D(
                in_channels=1024,
                output_channels=out_channels,
                kernel_shape=kernel,
                use_batch_norm=False,
                padding="spatial_valid",
                use_bias=True,
                activation_fn=None,
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def load_pretrained_weight(self, model_path=config["model"]["backbone_model"]):
        info = self._model.load_state_dict(torch.load(model_path), strict=False)
        print(f"load pretrained weight for backbone from {model_path}")
        print(info)

    def train(self, mode=True):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x = self._model.extract_features(x)[self.final_endpoint]
        x = self.proj_out(x)
        return x.squeeze(-1).squeeze(-1)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale).clamp(max=4)


class VideoActionClsHead(nn.Module):
    """action classification head"""

    def __init__(self, num_classes: int, in_channels: int, num_layers: int = 2):
        """TODO: to be defined.

        Args:
            num_classes (int): The number of classes (including background).
            in_channels (int): The number of input channels.

        Kwargs:
            num_layers (int): The number of classification layers.


        """
        super().__init__()

        self._num_classes = num_classes
        self._in_channels = in_channels
        self._num_layers = num_layers
        self.dropout = nn.Dropout(0.5)
        self.cls = nn.Linear(in_channels, num_classes)

    def forward(self, backbone_feat: torch.Tensor):
        """forward function

        Args:
            backbone_feat (torch.Tensor): the features of backbone. shape: [B,C,T]

        Returns: torch.Tensor. Logits of videos. shape: [B,num_classes]

        """
        x = backbone_feat.mean(dim=-1)  # [B,C]
        x = self.dropout(x)
        logits = self.cls(x)
        return logits


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(
                in_channels=in_channels,
                output_channels=proposal_channels,
                kernel_shape=1,
                activation_fn=None,
            ),
            # nn.InstanceNorm1d(proposal_channels),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True),
        )
        self.lr_conv = nn.Sequential(
            Unit1D(
                in_channels=in_channels,
                output_channels=proposal_channels * 2,
                kernel_shape=1,
                activation_fn=None,
            ),
            # nn.InstanceNorm1d(proposal_channels * 2),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.roi_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels,
                output_channels=proposal_channels,
                kernel_shape=1,
                activation_fn=None,
            ),
            # nn.InstanceNorm1d(proposal_channels * 2),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True),
        )

        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None,
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        # prop_feature = feature
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature


class CoarsePyramid(nn.Module):
    def __init__(self, model_cfg, feat_channels=(832, 1024), frame_num=768):
        super(CoarsePyramid, self).__init__()
        out_channels = conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num

        self.pyramids = build_neck(model_cfg.model["neck"])

        self.action_head = VideoActionClsHead(**model_cfg.model["action_head"])

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )

        self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        self.conf_proposal_branch = ProposalBranch(out_channels, 512)

        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None,
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None,
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(torch.Tensor([[(c + 0.5) / t, i] for c in range(t)]).view(-1, 2))
            t = t // 2

    def forward(self, backbone_feat, ssl=False):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        trip = []
        batch_num = backbone_feat.size(0)
        pyramid_feats = self.pyramids(backbone_feat)

        action_logits = self.action_head(backbone_feat)

        """print pyramid feats"""
        # for i, feat in enumerate(pyramid_feats):
        #     print(f"FPN level {i}: {feat.shape}")

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            # prior = torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1).to(feat.device)
            # priors.append(prior)
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)
            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                .view(batch_num, 2, -1)
                .permute(0, 2, 1)
                .contiguous()
                * fpn_strides[i]
            )
            confs.append(
                self.conf_head(conf_feat)
                .view(batch_num, num_classes, -1)
                .permute(0, 2, 1)
                .contiguous()
            )

            t = feat.size(2)
            # print(f"FPN level {i}, feat shape: {feat.shape}, locs shape: {locs[-1].shape}")
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                priors = self.priors[i][:, :1].expand(batch_num, t, 1).to(feat.device)
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat(
                    [
                        torch.round(l_segment - out_plen),
                        torch.round(l_segment + in_plen),
                        torch.round(r_segment - in_plen),
                        torch.round(r_segment + out_plen),
                    ],
                    dim=-1,
                )

                decoded_segments = torch.cat(
                    [
                        priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                        priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:],
                    ],
                    dim=-1,
                )
                plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat(
                    [
                        torch.round(decoded_segments[:, :, :1] - out_plen),
                        torch.round(decoded_segments[:, :, :1] + in_plen),
                        torch.round(decoded_segments[:, :, 1:] - in_plen),
                        torch.round(decoded_segments[:, :, 1:] + out_plen),
                    ],
                    dim=-1,
                )

            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(
                loc_feat, frame_level_feat, segments, frame_segments
            )
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(
                conf_feat, frame_level_feat, segments, frame_segments
            )
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = (
                    loc_prop_feat_[
                        :,
                        :ndim,
                    ]
                    .permute(0, 2, 1)
                    .contiguous()
                )

                end_loc_prop = (
                    loc_prop_feat_[
                        :,
                        ndim:,
                    ]
                    .permute(0, 2, 1)
                    .contiguous()
                )
                start_conf_prop = (
                    conf_prop_feat_[
                        :,
                        :ndim,
                    ]
                    .permute(0, 2, 1)
                    .contiguous()
                )
                end_conf_prop = (
                    conf_prop_feat_[
                        :,
                        ndim:,
                    ]
                    .permute(0, 2, 1)
                    .contiguous()
                )
                if ssl:
                    return trip
            prop_locs.append(
                self.prop_loc_head(loc_prop_feat)
                .view(batch_num, 2, -1)
                .permute(0, 2, 1)
                .contiguous()
            )
            prop_confs.append(
                self.prop_conf_head(conf_prop_feat)
                .view(batch_num, num_classes, -1)
                .permute(0, 2, 1)
                .contiguous()
            )
            centers.append(
                self.center_head(loc_prop_feat)
                .view(batch_num, 1, -1)
                .permute(0, 2, 1)
                .contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return (
            action_logits,
            loc,
            conf,
            prop_loc,
            prop_conf,
            center,
            priors,
            start,
            end,
            start_loc_prop,
            end_loc_prop,
            start_conf_prop,
            end_conf_prop,
        )


class BDNet(nn.Module):
    def __init__(
        self,
        model_cfg,
        training=True,
        frame_num=768,
    ):
        super(BDNet, self).__init__()

        self.coarse_pyramid_detection = CoarsePyramid(model_cfg, frame_num=frame_num)
        self.reset_params()
        self.boundary_max_pooling = BoundaryMaxPooling()
        self._training = training
        self.scales = [1, 4, 4]

        if model_cfg.get("backbone", None) == "i3d":
            self.backbone = I3D_BackBone()
            if self._training:
                self.backbone.load_pretrained_weight()
            backbone_cfg = model_cfg.get("backbone_cfg", None)
            if backbone_cfg is not None:
                if backbone_cfg["chunk_wrapper"]:
                    self.backbone = ChunkWrapper(
                        self.backbone,
                        chunk_size=backbone_cfg["chunk_size"],
                        shift_inp=backbone_cfg["shift_inp"],
                        t_downsample=backbone_cfg["t_downsample"],
                        do_pooling=backbone_cfg["do_pooling"],
                    )
        else:
            self.backbone = build_backbone(model_cfg.model["backbone"])

        if model_cfg.get("graddrop", False):
            print(model_cfg.graddrop_cfg)
            self.backbone = GradDropModel(self.backbone, **model_cfg.graddrop_cfg)

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if (
            isinstance(m, nn.Conv1d)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv3d)
            or isinstance(m, nn.ConvTranspose3d)
        ):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        # Initialization
        for modules in [
            self.coarse_pyramid_detection.loc_tower,
            self.coarse_pyramid_detection.conf_tower,
            self.coarse_pyramid_detection.loc_head,
            self.coarse_pyramid_detection.conf_head,
            self.coarse_pyramid_detection.loc_proposal_branch,
            self.coarse_pyramid_detection.conf_proposal_branch,
            self.coarse_pyramid_detection.prop_loc_head,
            self.coarse_pyramid_detection.prop_conf_head,
            self.coarse_pyramid_detection.center_head,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv1d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward_backbone(self, x: torch.Tensor):
        """forward backbone and get backbone_feat

        Args:
            x (torch.Tensor): the input features.

        Returns: torch.Tensor. backbone feature with shape: [B,C,T].

        """
        return self.backbone(x)

    def forward_detector(self, backbone_feat):
        """TODO: Docstring for forward_detector.

        Args:
            backbone_feat (TODO): TODO

        Returns: TODO

        """
        (
            action_logits,
            loc,
            conf,
            prop_loc,
            prop_conf,
            center,
            priors,
            start,
            end,
            start_loc_prop,
            end_loc_prop,
            start_conf_prop,
            end_conf_prop,
        ) = self.coarse_pyramid_detection(backbone_feat)
        return {
            "action_logits": action_logits,
            "loc": loc,
            "conf": conf,
            "priors": priors,
            "prop_loc": prop_loc,
            "prop_conf": prop_conf,
            "center": center,
            "start": start,
            "end": end,
            "start_loc_prop": start_loc_prop,
            "end_loc_prop": end_loc_prop,
            "start_conf_prop": start_conf_prop,
            "end_conf_prop": end_conf_prop,
        }

    def forward(self, x, proposals=None, ssl=False):
        backbone_feat = self.forward_backbone(x)
        return self.forward_detector(backbone_feat)
