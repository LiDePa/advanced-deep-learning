# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
import glob
import os


# currently set up to receive the annotation directory
def load_dataset(annotation_path: str, image_base_path: str, offset_columns: int = 4) -> Tuple[
    List[str], List[np.ndarray], List[np.ndarray]]:
    annotated_frame_paths = []
    keypoints_np = []

    # iterate through train, test and validation .csv files
    for csv_file_path in sorted(glob.glob(os.path.join(annotation_path, "*"))):
        with open(csv_file_path) as csv_file:
            # skip first two lines
            next(csv_file)
            next(csv_file)

            # iterate through each line=label in the .csv
            annotations = csv.reader(csv_file, delimiter=';')
            for label in annotations:
                # build name of corresponding .jpg frame for the label and add it to annotated_frame_paths
                frame_num = label[1]
                if len(frame_num) < 5: # frames are named with their index having at least 5 digits
                    frame_num = "0" + frame_num
                frame = f"{label[0]}_({frame_num}).jpg"
                annotated_frame_paths.append(
                    os.path.join(image_base_path, label[0], frame)
                )

                # create numpy array with keypoint coordinates
                keypoints_vector = np.array(label[-17*3:]) # take last 17 list entries, which contain keypoint values
                keypoints_matrix = keypoints_vector.reshape(17, 3)
                keypoints_np.append(keypoints_matrix)








class SkijumpDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, boxes, input_size=(128, 128), validation_mode=False, heatmap_downscale=1):
        """
        Initializes the dataset with the output of the load_dataset function. You can add more parameters to this function if
        necessary.
        :param images: List of image paths, output of load_dataset
        :param labels: List of labels, output of load_dataset
        :param boxes: List of bounding boxes, output of load_dataset
        :param input_size: Size of the returned images, this is the input size of the model
        :param validation_mode: If True, the dataset returns heatmaps instead of coordinates and does not use data augmentation
        :param heatmap_downscale: Factor by which the heatmaps are smaller compared to the input size
        """
        # TODO
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        TODO: Load image, crop, pad and resize to desired size
        :return: adjusted image and heatmaps in train mode | adjusted image, original ground truth coordinates,
        resizing factor, and used bounding box in validation/test mode
        """
        raise NotImplementedError

    @classmethod
    def augment(cls, img: np.ndarray, label: np.ndarray, rot: float, trans_x: float, trans_y: float, flip: bool):
        """
        Geometric augmentation of the image and adjustment of the labels.
        """
        # TODO
        raise NotImplementedError


def create_skijump_subsets(dataset_path: str, batch_size=16, image_size=(128, 128), heatmap_downscale=2) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    # TODO initialize train, validation and test data loader and return them in this order
    # dataset_path is the path to the base folder containing the annotations and the images in subdirectories (do not rename
    # them or the files)
    raise NotImplementedError
