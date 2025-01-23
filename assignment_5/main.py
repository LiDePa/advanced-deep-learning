from venv import create

from assignment_4.reasoning.reason import DATABASE_NAME
from .dataset import plot_dataset_confirmation, load_dataset, SkijumpDataset
from .heatmaps import create_heatmaps, plot_heatmap_confirmation
from argparse import ArgumentParser
import os


parser = ArgumentParser("Exercise 5")
# specify dataset root path
parser.add_argument("--dataset-root", required=True, type=str)
# Exercise 5.1: plot n random images with their keypoints
parser.add_argument("--dataset-confirmation-images", type=int)
# Exercise 5.3(b) plot 10 images with heatmap overlay
parser.add_argument("--heatmap-confirmation", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_root = args.dataset_root

    image_base_path = os.path.join(dataset_root, "annotated_frames")
    train_annotation_path = os.path.join(dataset_root, "annotations/train.csv")

    images, labels, boxes = load_dataset(train_annotation_path, image_base_path)
    dataset = SkijumpDataset(images, labels, boxes, validation_mode=False, heatmap_downscale=2)
    item = dataset[32][1]

    # TODO: make sure def augment isn't fucking things up here, maybe change normalize flag to plotting mode flag?

    if args.dataset_confirmation_images is not None:
        plot_dataset_confirmation(train_annotation_path, image_base_path, args.dataset_confirmation_images)

    if args.heatmap_confirmation:
            dataset_128 = SkijumpDataset(images, labels, boxes,
                                         validation_mode=False,
                                         heatmap_downscale=1,
                                         normalize=False)
            dataset_64 = SkijumpDataset(images, labels, boxes,
                                         validation_mode=False,
                                         heatmap_downscale=2,
                                         normalize=False)
            plot_heatmap_confirmation(dataset_128, dataset_64, len(images))