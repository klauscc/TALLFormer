# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import argparse
import os
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, help="the filepath of the checkpoint path")

args = parser.parse_args()

src_path = args.src
dst_path = src_path.replace(".pth", "_keysfrom_backbone.pth")

state = torch.load(src_path)
state = state["state_dict"]
print(f"src_path:{src_path}")
print(f"save converted ckpt to {dst_path}")
torch.save(state, dst_path)
