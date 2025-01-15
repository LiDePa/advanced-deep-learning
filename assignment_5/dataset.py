# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
import glob
import os
from PIL import Image


# currently set up to receive the annotations directory
def load_dataset(annotation_path: str, image_base_path: str, offset_columns: int = 4) -> Tuple[
    List[str], List[np.ndarray], List[np.ndarray]]:

    annotated_frame_paths = []
    keypoints_np = []
    event_resolutions = {} # assume constant frame resolution across a given event; needed for bounding box padding

    # iterate through train, test and validation .csv files
    for csv_file_path in sorted(glob.glob(os.path.join(annotation_path, "*"))):
        with open(csv_file_path) as csv_file:
            # skip first two lines
            next(csv_file)
            next(csv_file)

            # iterate through each line(=label) in the .csv as a list of strings
            annotations = csv.reader(csv_file, delimiter=';')
            for label in annotations:
                # build name of corresponding .jpg frame for the label and add it to annotated_frame_paths
                frame_num = label[1]
                if len(frame_num) < 5: # .jpg frames are named with their index having at least 5 digits
                    frame_num = "0" + frame_num
                frame = f"{label[0]}_({frame_num}).jpg"
                frame_path = os.path.join(image_base_path, label[0], frame)
                annotated_frame_paths.append(frame_path)

                # save resolution of first frame that occurs for each event; is needed later for bounding box padding
                if label[0] not in event_resolutions:
                    with Image.open(frame_path) as img:
                        width, height = img.size
                    event_resolutions[label[0]] = [width, height]

                # create numpy array with keypoint coordinates and add it to keypoints_np
                keypoints_vector = np.array(label[-17*3:]) # take last 17 list entries, which contain keypoint values
                keypoints_matrix = keypoints_vector.reshape(17, 3)
                keypoints_np.append(keypoints_matrix)

                # get visible keypoints and determine width and height of tightest bounding box
                mask = keypoints_matrix[:,2] != "0"
                keypoints_visible = keypoints_matrix[mask]
                min_x = np.min(keypoints_visible[:, 0])
                max_x = np.max(keypoints_visible[:, 0])
                min_y = np.min(keypoints_visible[:, 1])
                max_y = np.max(keypoints_visible[:, 1])
                w = max_x - min_x
                h = max_y - min_y

                # pad the tightest bounding box by 20% in each direction
                x_padding = np.ceil(0.2 * w).astype(int)
                y_padding = np.ceil(0.2 * h).astype(int)
                min_x_padded = min_x - x_padding
                max_x_padded = max_x + x_padding
                if min_x_padded < 0:
                    min_x_padded = 0
                # if max_x_padded >
                # TODO: check if bounding box is too large on max values by comparing to valuesin event_resolutions










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
