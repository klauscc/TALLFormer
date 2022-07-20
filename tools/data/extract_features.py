# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import argparse
import json
import os
from glob import glob
from inspect import findsource

import numpy as np
import torch
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
split = "val"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="the supported values: swin_t_15fps_128x128, swin_b_15fps_256x256, swin_t_15fps_256x256",
)
args = parser.parse_args()

cfg = args.config

if cfg == "swin_t_15fps_128x128":
    ### swin_tiny, 15fps, 128x128
    video_dir = f"data/thumos14/frames_15fps/{split}"
    dst_dir = (
        f"data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112/{split}"
    )
    meta_file = f"data/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112/meta_{split}.json"
    model_config = swin_t_config
    IMG_SHAPE = (112, 112)
    FEAT_DIM = 768
    ckpt_path = "data/pretrained_models/vswin/swin_tiny_patch244_window877_kinetics400_1k_keysfrom_backbone.pth"
    ### end config
elif cfg == "swin_b_15fps_256x256":
    ### swin_base, 15fps, 256x256
    video_dir = f"data/thumos14/frames_15fps_256x256/{split}"
    dst_dir = (
        f"data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224/{split}"
    )
    meta_file = f"data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224/meta_{split}.json"
    model_config = swin_b_config
    IMG_SHAPE = (224, 224)
    FEAT_DIM = 1024
    ckpt_path = "data/pretrained_models/vswin/swin_base_patch244_window877_kinetics400_22k_keysfrom_backbone.pth"
    ### end config
elif cfg == "swin_t_15fps_256x256":
    ### swin_tiny, 15fps, 256x256
    video_dir = f"data/thumos14/frames_15fps_256x256/{split}"
    dst_dir = (
        f"data/thumos14/memory_mechanism/feat_swint_15fps_256x256_crop224x224/{split}"
    )
    meta_file = f"data/thumos14/memory_mechanism/feat_swint_15fps_256x256_crop224x224/meta_{split}.json"
    model_config = swin_t_config
    IMG_SHAPE = (224, 224)
    FEAT_DIM = 768
    ckpt_path = "data/pretrained_models/vswin/swin_tiny_patch244_window877_kinetics400_1k_keysfrom_backbone.pth"
    ### end config
else:
    raise ValueError("config: {cfg} not supported")

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


BATCH_SIZE = 8
CHUNK_SIZE = 32
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], device=device)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], device=device)
########################################


def load_video(video_path):
    """load frames

    Args:
        video_path (string): path contains the frames.

    Returns: list of image. Each image is of shape (H,W,3).

    """
    frame_paths = sorted(glob(os.path.join(video_path, "*.png")))
    imgs = []
    for img_path in frame_paths:
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img).astype(np.float32)
        imgs.append(img)
    return imgs


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
        frames = imgs[frame_idx : frame_idx + input_length]
        frames = np.stack(frames, axis=0)  # [T,H,W,C]
        # to torch.Tensor
        frames = torch.from_numpy(frames).cuda()
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
for i, p in enumerate(video_paths):
    video_name = os.path.basename(p)
    dst_path = os.path.join(dst_dir, video_name + ".mmap")
    print(f"extract features for video: {video_name}")
    num_frames, feat_shape = extract_one_video(p, dst_path)
    print(f"    num_frames:{num_frames}. feat_shape: {feat_shape}")
    metas[video_name] = {"num_frames": num_frames, "feat_shape": feat_shape}

with open(meta_file, "w") as f:
    json.dump(metas, f, indent=4)
