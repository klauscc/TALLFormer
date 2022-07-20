import json

import matplotlib.pyplot as plt

annotation_file = "data/annots/anet/activity_net.v1-3.json"

data = json.load(open(annotation_file, "r"))

database = data["database"]

rates = []
durations = []
num_segments = []

for video_name, video_info in database.items():
    duration = video_info["duration"]
    annots = video_info["annotations"]

    if video_info["subset"] == "testing":
        continue

    num_segments.append(len(annots))
    durations.append(duration)
    for segment in annots:
        location = segment["segment"]
        seg_duration = location[1] - location[0]
        rate = seg_duration / duration
        rates.append(rate)


nbins = 40
plt.hist(rates, nbins)
plt.savefig("data/annots/anet/duration_dist.png")

# plt.hist(durations, nbins)
# plt.savefig("data/annots/anet/video_length.png")

# plt.hist(num_segments, nbins)
# plt.savefig("data/annots/anet/num_segments.png")
