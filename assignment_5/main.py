from venv import create

from assignment_4.reasoning.reason import DATABASE_NAME
from .dataset import plot_dataset_confirmation, load_dataset, SkijumpDataset
from .heatmaps import create_heatmaps
from argparse import ArgumentParser
import os


parser = ArgumentParser("Exercise 5")
parser.add_argument("--dataset-root", required=True, type=str)
parser.add_argument("--dataset-confirmation-images", type=int) # Exercise 5.1: plot n random images with their keypoints


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_root = args.dataset_root

    image_base_path = os.path.join(dataset_root, "annotated_frames")
    train_annotation_path = os.path.join(dataset_root, "annotations/train.csv")

    images, labels, boxes = load_dataset(train_annotation_path, image_base_path)
    dataset = SkijumpDataset(images, labels, boxes, validation_mode=False, heatmap_downscale=2)
    item = dataset[30][1]

    if args.dataset_confirmation_images is not None:
        plot_dataset_confirmation(train_annotation_path, image_base_path, args.dataset_confirmation_images)