from .dataset import load_dataset
from argparse import ArgumentParser
import os


parser = ArgumentParser("Plot 10 random images.")
parser.add_argument("--dataset-root", required=True, type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_root = args.dataset_root


    # TODO: this is only the test annotations for testing
    test_annotation_path = os.path.join(dataset_root, "annotations")
    image_base_path = os.path.join(dataset_root, "annotated_frames")
    dataset = load_dataset(test_annotation_path, image_base_path)