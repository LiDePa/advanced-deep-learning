from .dataset import load_dataset, plot_dataset_confirmation
from argparse import ArgumentParser
import os


parser = ArgumentParser("Plot 10 random images.")
parser.add_argument("--dataset-root", required=True, type=str)
parser.add_argument("--dataset-confirmation-images", type=int) # Exercise 5.1: plot n random images with their keypoints


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_root = args.dataset_root


    annotation_path = os.path.join(dataset_root, "annotations")
    image_base_path = os.path.join(dataset_root, "annotated_frames")
    dataset = load_dataset(annotation_path, image_base_path)

    if args.dataset_confirmation_images is not None:
        plot_dataset_confirmation(annotation_path, image_base_path, args.dataset_confirmation_images)