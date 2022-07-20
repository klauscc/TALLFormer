#!/usr/bin/env python

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, default="data/hacs/video_256")
parser.add_argument("--target_dir", type=str, default="data/hacs/videos_256x256")
parser.add_argument(
    "--class_name_filepath", type=str, default="data/annots/hacs/class_name.txt"
)


def move_videos(args):
    class_dirs = glob.glob(os.path.join(args.video_dir, "*"))
    print(class_dirs)

    os.makedirs(args.target_dir, exist_ok=True)

    # generate class_name file.
    class_name_f = open(args.class_name_filepath, "w")
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        class_name = class_name.replace("_", " ")
        class_name_f.write(class_name + "\n")

    # move videos
    for class_dir in class_dirs:
        cmd = f"mv {os.path.join(class_dir, '*')} {args.target_dir}"
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    args = parser.parse_args()
    move_videos(args)
