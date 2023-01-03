import argparse
import os

import numpy as np
from vedatad.misc.logger import get_root_logger

from AFSD.evaluation.eval_detection import ANETdetection

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=str)
parser.add_argument("--epoch", type=str)
parser.add_argument("output_json", type=str)
parser.add_argument(
    "gt_json", type=str, default="data/annots/anet/activity_net_1_3_new.json", nargs="?"
)
args = parser.parse_args()

# setup logger
epoch = args.epoch
log_file = os.path.join(args.workspace, f"Test-epoch_{epoch}.log")
logger = get_root_logger(log_file=log_file, log_level="INFO")

logger.info(f"-----------Test Epoch {epoch}-------------")

tious = np.linspace(0.5, 0.95, 10)
anet_detection = ANETdetection(
    ground_truth_filename=args.gt_json,
    prediction_filename=args.output_json,
    subset="validation",
    tiou_thresholds=tious,
)
mAPs, average_mAP, ap = anet_detection.evaluate()
for (tiou, mAP) in zip(tious, mAPs):
    logger.info("mAP at tIoU {} is {}".format(tiou, mAP))

logger.info(f"Average mAP: {average_mAP}")
