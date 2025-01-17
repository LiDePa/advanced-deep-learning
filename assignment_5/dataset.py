# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
import glob
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random



def load_dataset(annotation_path: str, image_base_path: str, offset_columns: int = 4) -> Tuple[
    List[str], List[np.ndarray], List[np.ndarray]]:

    frame_paths = []
    keypoints = []
    bounding_boxes = []
    # assume constant frame resolution across a given event and read it only once on the first frame
    # needed for bounding box padding
    event_resolutions = {}

    with open(annotation_path) as csv_file:
        # skip first two lines
        next(csv_file)
        next(csv_file)

        # iterate through each line(=label) in the .csv as a list of immutable strings
        annotations = csv.reader(csv_file, delimiter=';')
        for label in annotations:
            # create shallow copy to not modify the structure of annotations
            label = label.copy()

            # build name of corresponding .jpg frame for the label and add it to frame_paths
            frame_num = label[1]
            if len(frame_num) < 5: # .jpg frames are named with their index having at least 5 digits
                frame_num = "0" + frame_num
            frame = f"{label[0]}_({frame_num}).jpg"
            frame_path = os.path.join(image_base_path, label[0], frame)
            frame_paths.append(frame_path)

            # save resolution of first frame that occurs for each event; is needed later for bounding box padding
            if label[0] not in event_resolutions:
                with Image.open(frame_path) as img:
                    width, height = img.size
                event_resolutions[label[0]] = [width, height]

            # create numpy array with keypoint coordinates and add it to keypoints
            keypoints_vector = np.array(label[-17*3:]).astype(np.int32) # take last 17 list entries, which contain keypoint values
            keypoints_matrix = keypoints_vector.reshape(17, 3)
            keypoints.append(keypoints_matrix.astype(np.ndarray))

            # get visible keypoints and determine width and height of tightest bounding box
            mask = keypoints_matrix[:,2] != "0"
            keypoints_visible = keypoints_matrix[mask]
            min_x_tight = np.min(keypoints_visible[:, 0])
            max_x_tight = np.max(keypoints_visible[:, 0])
            min_y_tight = np.min(keypoints_visible[:, 1])
            max_y_tight = np.max(keypoints_visible[:, 1])
            w = max_x_tight - min_x_tight
            h = max_y_tight - min_y_tight

            # pad the tightest bounding box by 20% in each direction
            x_padding = np.ceil(0.2 * w).astype(int)
            y_padding = np.ceil(0.2 * h).astype(int)
            min_x = min_x_tight - x_padding
            if min_x < 0:
                min_x = 0
            max_x = max_x_tight + x_padding
            if max_x > event_resolutions[label[0]][0]:
                max_x = event_resolutions[label[0]][0]
            min_y = min_y_tight - y_padding
            if min_y < 0:
                min_y = 0
            max_y = max_y_tight + y_padding
            if max_y > event_resolutions[label[0]][1]:
                max_y = event_resolutions[label[0]][1]
            bounding_boxes.append(np.array([min_x, max_x, min_y, max_y]))

    return frame_paths, keypoints, bounding_boxes



# most of the following function is written by the deepseek chatbot
# never had to learn how to use matplotlib, more of a matlab guy
# Exercise 5.1 is a perfect example for something, I will from now on use LLMs for
# you will find the prompt in my submission folder. Feel free to deduct points
def plot_dataset_confirmation(annotation_path: str, image_base_path: str, n_images: int):
    frame_paths, keypoints, _ = load_dataset(annotation_path, image_base_path)

    # randomly select n_images from the dataset
    selected_indices = random.sample(range(len(frame_paths)), n_images)

    # get keypoint names and define a fitting color map
    keypoint_names = [
        "Head", "Right Shoulder", "Right Elbow", "Right Hand",
        "Left Shoulder", "Left Elbow", "Left Hand",
        "Right Hip", "Right Knee", "Right Ankle",
        "Left Hip", "Left Knee", "Left Ankle",
        "Right Ski Tip", "Right Ski Tail", "Left Ski Tip", "Left Ski Tail"
    ]
    colors = plt.cm.get_cmap('tab20', len(keypoint_names)).colors

    project_folder = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    output_dir = os.path.join(project_folder, "output_plots")  # Output directory in the project folder
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Plot each selected image
    for idx in selected_indices:
        # Load the image
        image_path = frame_paths[idx]
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Get the keypoints for this image
        kps = keypoints[idx]  # Shape: (17, 3)

        # Get image height for scaling keypoint size
        image_height = image.size[1]
        keypoint_radius = int(image_height * 0.01)  # Proportional to image height

        # Plot each keypoint
        for i, (x, y, vis) in enumerate(kps):
            if vis == 0:  # Skip invisible keypoints
                continue
            # Draw the keypoint
            draw.ellipse(
                [(x - keypoint_radius, y - keypoint_radius), (x + keypoint_radius, y + keypoint_radius)],
                fill=tuple(int(c * 255) for c in colors[i]),  # Convert color to RGB
                outline="black"
            )

        # Display the image with keypoints
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")

        # Add a legend
        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markersize=10,
                label=keypoint_names[i]
            )
            for i in range(len(keypoint_names))
        ]
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.2, 1))

        # Save the plot
        output_path = os.path.join(output_dir, f"plot_{os.path.basename(image_path)}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {output_path}")



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
