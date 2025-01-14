import csv
import os
import glob
from argparse import ArgumentParser
from PIL import Image
import matplotlib


parser = ArgumentParser("Plot 10 random images.")
parser.add_argument("--dataset-root", required=True, type=str)
args = parser.parse_args()

with open(os.path.join(args.dataset_root, "annotations/train.csv"), "r") as csv_file:
    # jump first line
    next(csv_file)

    annotations = csv.DictReader(csv_file, delimiter=';')


# collect frame paths
frames = []

for frame in sorted(glob.glob(os.path.join(args.dataset_root, "annotated_frames/*"))):
    # get list of image paths
    frames += sorted(glob.glob(os.path.join(frame,"*.jpg")))

print("Hi")