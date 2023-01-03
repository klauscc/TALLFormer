# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import json
import os
from argparse import ArgumentParser
from glob import glob
from inspect import findsource
from math import ceil

import torch
import decord
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image
from vedatad.models.builder import build_backbone

swin_t_config = dict(
    typename="ChunkVideoSwin",
    chunk_size=32,
    patch_size=(2, 4, 4),
    in_chans=3,
    embed_dim=96,
    drop_path_rate=0.1,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(8, 7, 7),
    patch_norm=True,
    frozen_stages=2,
    use_checkpoint=False,
)

swin_b_config = dict(
    typename="ChunkVideoSwin",
    chunk_size=32,
    frozen_stages=2,
    use_checkpoint=True,
    patch_size=(2, 4, 4),
    in_chans=3,
    embed_dim=128,
    drop_path_rate=0.2,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=(8, 7, 7),
    patch_norm=True,
)

################CONFIGS##################

parser = ArgumentParser()
parser.add_argument("--video_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model", type=str, required=True, help="swin_b or swin_t")
parser.add_argument("--crop_size", type=int, default=224, help="The input size to the model.")

args = parser.parse_args()

### swin_base ActivityNet, 256x256
video_dir = args.video_dir
dst_dir = args.output_dir
meta_file = os.path.join(dst_dir, "meta.json")

if args.model == "swin_b":
    model_config = swin_b_config
    ckpt_path = "models/pretrained_models/vswin/swin_base_patch244_window877_kinetics600_22k_keysfrom_backbone.pth"
    FEAT_DIM = 1024
elif args.model == "swin_t":
    model_config = swin_t_config
    ckpt_path = "models/pretrained_models/vswin/swin_tiny_patch244_window877_kinetics400_1k_keysfrom_backbone.pth"
    FEAT_DIM = 768
else:
    raise ValueError(f"args.model: {args.model} not supported")

IMG_SHAPE = (args.crop_size, args.crop_size)
### end config

os.makedirs(dst_dir, exist_ok=True)
device = torch.device("cuda:0")

model = build_backbone(model_config).to(device)

## load pretrained weights on K400.
states = torch.load(ckpt_path)
new_state = {}
for k, v in states.items():
    new_state[k.replace("backbone.", "")] = v
info = model.load_state_dict(new_state, strict=False)
print(info)

model.train()  # simulate training.


BATCH_SIZE = 16
CHUNK_SIZE = 32
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], device=device)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], device=device)
########################################


def load_video(video_path):
    """load frames

    Args:
        video_path (string): path contains the frames.

    Returns: np.array. Each image is of shape (T,H,W,3).

    """
    base, ext = os.path.splitext(video_path)
    if ext == ".mp4":
        vr = decord.VideoReader(video_path)
        data = vr.get_batch(range(len(vr))).asnumpy()
        return data
    elif ext == ".npy":
        return np.load(video_path, "r")


def extract_one_video(video_path, dst_path):
    """extract features for one video

    Args:
        video_path (string): path contains the frames.
        dst_path (string): the memmap file to save the features.

    Returns: tuple[T,C]. the feature shape for video.

    """
    imgs = load_video(video_path)
    num_frames = len(imgs)
    input_length = BATCH_SIZE * CHUNK_SIZE
    features = []
    for frame_idx in range(0, num_frames, input_length):
        frames = imgs[frame_idx : frame_idx + input_length]  # [T,H,W,C]
        # to torch.Tensor
        frames = torch.from_numpy(frames.copy()).cuda()
        frames = frames.permute(3, 0, 1, 2)  # [C,T,H,W]
        frames = tf.center_crop(frames, IMG_SHAPE)
        frames = (frames - IMG_MEAN.view(3, 1, 1, 1)) / IMG_STD.view(3, 1, 1, 1)
        frames = frames.unsqueeze(0)  # [1,C,T,H,W]
        with torch.no_grad():
            feat = model(frames)
            feat = F.adaptive_avg_pool3d(feat, (None, 1, 1))  # [1,C,T,1,1]
            feat = feat.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)  # [T, C]
        features.append(feat.cpu().numpy())

    features = np.concatenate(features)
    fp = np.memmap(dst_path, dtype="float32", mode="w+", shape=features.shape)
    fp[:] = features[:]
    fp.flush()
    return num_frames, features.shape


video_paths = sorted(glob(os.path.join(video_dir, "*")))
metas = {}
np.random.shuffle(video_paths)
for i, p in enumerate(video_paths):
    video_name = os.path.basename(p)
    basename, ext = os.path.splitext(video_name)
    dst_path = os.path.join(dst_dir, basename + ".mmap")
    if os.path.isfile(dst_path):
        print(f"video {video_name} already extracted. skip")
        num_frames = len(decord.VideoReader(p))
        feat_shape = [ceil(num_frames / 2), FEAT_DIM]
    else:
        print(f"extract features for video: {video_name}")
        num_frames, feat_shape = extract_one_video(p, dst_path)
    print(f"{i}/{len(video_paths)}    num_frames:{num_frames}. feat_shape: {feat_shape}")
    metas[basename] = {"num_frames": num_frames, "feat_shape": feat_shape}

with open(meta_file, "w") as f:
    json.dump(metas, f, indent=4)
