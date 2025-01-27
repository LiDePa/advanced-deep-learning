from venv import create

from assignment_4.reasoning.reason import DATABASE_NAME
from .dataset import plot_dataset_confirmation, load_dataset, SkijumpDataset, create_skijump_subsets
from .heatmaps import plot_heatmap_confirmation
from .metric import pck
from argparse import ArgumentParser
import os
import numpy as np



parser = ArgumentParser("Exercise 5")
# specify dataset root path
parser.add_argument("--dataset-root", required=True, type=str)
# Exercise 5.1: plot n random images with their keypoints
parser.add_argument("--dataset-confirmation-images", type=int)
# Exercise 5.3(b) plot 10 images with heatmap overlay
parser.add_argument("--heatmap-confirmation", action="store_true")
# Exercise 5.3(b) plot 10 images with heatmap overlay
parser.add_argument("--augmentation-confirmation", action="store_true")
# Exercise 5.5 test pck score
parser.add_argument("--test-pck", action="store_true")



if __name__ == "__main__":
    args = parser.parse_args()
    dataset_root = args.dataset_root

    image_base_path = os.path.join(dataset_root, "annotated_frames")
    train_annotation_path = os.path.join(dataset_root, "annotations/train.csv")

    # example sample access for testing
    images, labels, boxes = load_dataset(train_annotation_path, image_base_path)
    dataset = SkijumpDataset(images, labels, boxes, validation_mode=False, heatmap_downscale=1, augment=False)
    sample = dataset[48]

    # example dataloader creation for testing
    train, val, test = create_skijump_subsets(dataset_root)



    if args.dataset_confirmation_images is not None:
        plot_dataset_confirmation(train_annotation_path, image_base_path, args.dataset_confirmation_images)

    if args.heatmap_confirmation:
        dataset_128 = SkijumpDataset(images, labels, boxes,
                                     validation_mode=False,
                                     heatmap_downscale=1,
                                     normalize=False,
                                     augment=False)
        dataset_64 = SkijumpDataset(images, labels, boxes,
                                     validation_mode=False,
                                     heatmap_downscale=2,
                                     normalize=False,
                                     augment=False)
        plot_heatmap_confirmation(dataset_128, dataset_64, len(images), task="heatmap")

    if args.augmentation_confirmation:
        dataset_128 = SkijumpDataset(images, labels, boxes,
                                     validation_mode=False,
                                     heatmap_downscale=1,
                                     normalize=False,
                                     augment=True)
        dataset_64 = SkijumpDataset(images, labels, boxes,
                                     validation_mode=False,
                                     heatmap_downscale=2,
                                     normalize=False,
                                     augment=True)
        plot_heatmap_confirmation(dataset_128, dataset_64, len(images), task="augment")

    if args.test_pck:
        # test pck score
        val_annotation_path = os.path.join(dataset_root, "annotations/val.csv")
        predictions_path = "assignment_5/predictions.csv"
        _, labels_val_test, _ = load_dataset(val_annotation_path, image_base_path)
        _, labels_predictions_test, _ = load_dataset(predictions_path, image_base_path)
        pck_overall_test_01, pck_keypoints_test_01 = pck(labels_val_test, labels_predictions_test, (10, 1), t=0.1)
        pck_overall_test_02, pck_keypoints_test_02 = pck(labels_val_test, labels_predictions_test, (10, 1), t=0.2)
        pck_keypoints_test_01 = np.round(pck_keypoints_test_01, 3)
        pck_overall_test_01 = np.round(pck_overall_test_01, 3)
        pck_keypoints_test_02 = np.round(pck_keypoints_test_02, 3)
        pck_overall_test_02 = np.round(pck_overall_test_02, 3)
        print(f"\n overall t=0.1: {pck_overall_test_01} \n keypoints t=0.1: {pck_keypoints_test_01}")
        print(f"\n overall t=0.2: {pck_overall_test_02} \n keypoints t=0.2: {pck_keypoints_test_02}")