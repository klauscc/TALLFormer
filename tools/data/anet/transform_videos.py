import argparse
import multiprocessing as mp
import os

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("thread_num", type=int)
parser.add_argument("--video_dir", type=str, default="datasets/activitynet/v1-3/train_val")
parser.add_argument("--output_dir", type=str, default="datasets/activitynet/videos_112x112")
parser.add_argument("--resolution", type=str, default="112x112")
parser.add_argument("--max_frame", type=int, default=768)
args = parser.parse_args()

thread_num = args.thread_num
video_dir = args.video_dir
output_dir = args.output_dir
resolution = args.resolution
max_frame = args.max_frame

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = sorted(os.listdir(video_dir))


def sub_processor(pid, files):
    for file in files[:]:
        file_name = os.path.splitext(file)[0]
        target_file = os.path.join(output_dir, file_name + ".mp4")
        if os.path.exists(target_file) and os.stat(target_file).st_size != 0:
            # print('{} exists, skip.'.format(target_file))
            continue
        cap = cv2.VideoCapture(os.path.join(video_dir, file))
        max_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ratio = min(max_frame * 1.0 / frame_num, 1.0)
        target_fps = max_fps * ratio
        cmd = (
            "ffmpeg -v quiet -i {} -max_muxing_queue_size 9999 -qscale 0 -r {} -s {} -y {}".format(
                os.path.join(video_dir, file), target_fps, resolution, target_file
            )
        )
        print(cmd)
        os.system(cmd)


processes = []
video_num = len(files)
per_process_video_num = video_num // thread_num

for i in range(thread_num):
    if i == thread_num - 1:
        sub_files = files[i * per_process_video_num :]
    else:
        sub_files = files[i * per_process_video_num : (i + 1) * per_process_video_num]
    p = mp.Process(target=sub_processor, args=(i, sub_files))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
