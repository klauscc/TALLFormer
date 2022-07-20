#!/usr/bin/env python

import argparse
import glob
import multiprocessing as mp
import os

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("thread_num", type=int)
parser.add_argument("--video_dir", type=str, default="data/anet/v1-3/train_val")
parser.add_argument(
    "--output_dir", type=str, default="data/anet/frames_128x128_480frames"
)
parser.add_argument("--resolution", type=str, default="128x128")
parser.add_argument("--max_frame", type=int, default=480)
args = parser.parse_args()

thread_num = args.thread_num
video_dir = args.video_dir
output_dir = args.output_dir
resolution = args.resolution
max_frame = args.max_frame

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = sorted(os.listdir(video_dir))
print("NUMBER of videos:", len(files))


def extract(file):
    file_name = os.path.splitext(file)[0]
    target_file = os.path.join(output_dir, file_name)
    os.makedirs(target_file, exist_ok=True)

    num_files = glob.glob(os.path.join(target_file, "*"))
    if len(num_files) >= args.max_frame // 3:
        return
    cap = cv2.VideoCapture(os.path.join(video_dir, file))
    max_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ratio = min(max_frame * 1.0 / frame_num, 1.0)
    target_fps = max_fps * ratio
    cmd = "ffmpeg -v quiet -i {} -qscale 0 -r {} -s {} {}".format(
        os.path.join(video_dir, file),
        target_fps,
        resolution,
        target_file + "/%05d.png",
    )
    print(cmd)
    os.system(cmd)


p = mp.Pool(thread_num)
p.map(extract, files)
