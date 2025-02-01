# -*- coding: utf-8 -*-
from typing import List, Tuple

import numpy as np
import torch
from torch.onnx import select_model_mode_for_export
from torch.utils.data import DataLoader
import csv
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import cv2

from .heatmaps import create_heatmaps



def load_dataset(annotation_path: str, image_base_path: str, offset_columns: int = 4) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:

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
            # create shallow copy to not modify the structure of annotations, deeper strings are immutable
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
            keypoints_vector = np.array(label[-17*3:]).astype(np.int32)
            keypoints_matrix = keypoints_vector.reshape(17, 3)
            keypoints.append(keypoints_matrix.astype(np.ndarray))

            # get visible keypoints and determine width and height of tightest bounding box
            mask = keypoints_matrix[:,2] != 0
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
            min_x = max(0, min_x_tight - x_padding)
            max_x = min(max_x_tight + x_padding, event_resolutions[label[0]][0])
            min_y = max(0, min_y_tight - y_padding)
            max_y = min(max_y_tight + y_padding, event_resolutions[label[0]][1])
            bounding_boxes.append(np.array([min_x, max_x, min_y, max_y]))

    return frame_paths, keypoints, bounding_boxes



# Most of the following function is written by the deepseek chatbot.
# Exercise 5.1 is a perfect example for something, I will use LLMs for in the future.
# You will find the prompt in my submission folder. Feel free to deduct the points.
# All other code is written by me!
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

    # create output directory
    project_folder = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_folder, "output_plots")
    os.makedirs(output_dir, exist_ok=True)

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

        # save plot
        output_path = os.path.join(output_dir, f"plot_{os.path.basename(image_path)}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {output_path}")



class SkijumpDataset(torch.utils.data.Dataset):

    def __init__(self,
                 images,
                 labels,
                 boxes,
                 input_size=(128, 128),
                 validation_mode=False,
                 heatmap_downscale=1,
                 normalize=True, # can be set to False for visualization plots
                 augment=False,
                 aug_rotate=30.0,
                 aug_translate=0.1,
                 aug_flip=True):
        """
        Initializes the dataset with the output of the load_dataset function. You can add more parameters to this function if
        necessary.
        :param images: List of image paths, output of load_dataset
        :param labels: List of labels, output of load_dataset
        :param boxes: List of bounding boxes, output of load_dataset
        :param input_size: Size of the returned images, this is the input size of the model
        :param validation_mode: If True, the dataset returns heatmaps instead of coordinates and does not use data augmentation
        :param heatmap_downscale: Factor by which the heatmaps are smaller compared to the input size
        :param normalize: If True, normalize the image to ImageNet values
        :param augment: Augments samples if true
        :param aug_rotate: Augmentation: Maximum angle of random rotation
        :param aug_translate: Augmentation: Maximum proportion of random translation
        :param aug_flip: Augmentation: If true, image is horizontally flipped with a 50% chance
        """
        self._images = images
        self._labels = labels
        self._boxes = boxes
        self._input_size = input_size
        self._validation_mode = validation_mode
        self._heatmap_downscale = heatmap_downscale
        self._normalize = normalize
        self._augment = augment
        self._aug_rotate = aug_rotate
        self._aug_translate = aug_translate
        self._aug_flip = aug_flip

    def __getitem__(self, idx):
        """
        :return: adjusted image and heatmaps in train mode | adjusted image, original ground truth coordinates,
        resizing factor, and used bounding box in validation/test mode
        """
        # open image as PIL
        image = Image.open(self._images[idx]).convert("RGB")

        # crop image to bounding box
        image = image.crop((
            self._boxes[idx][0],
            self._boxes[idx][2],
            self._boxes[idx][1],
            self._boxes[idx][3]))

        # calculate scaling ratio considering input_size and heatmap_downscale
        # assume quadratic shape of input_size to avoid aspect ratio comparisons
        scaling_ratio = self._input_size[0] / max(image.size)

        # if cropped image is larger than input_size: bilinear scaling to input_size while keeping aspect ratio
        image.thumbnail(self._input_size)

        # convert to numpy array and normalize to ImageNet mean and standard deviation
        image = np.array(image) / 255.0
        if self._normalize: # can be set to False for visualization plots
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            image = (image - mean) / std

        # pad image either on the right or bottom
        padding = np.zeros((*self._input_size,image.shape[2]))
        padding[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        image = padding

        # validation mode:
        if self._validation_mode:
            image_name = os.path.basename(self._images[idx])
            image = torch.tensor(image).permute(2, 0, 1)
            return image, self._labels[idx], scaling_ratio, self._boxes[idx], image_name

        # training mode:
        elif not self._validation_mode:
            # adapt a copy of the label to the cropped image and scale the keypoints if cropped image has been scaled
            label = np.copy(self._labels[idx])
            scaling_ratio = min(scaling_ratio, 1.0)
            label[:, 0] = (self._labels[idx][:, 0] - self._boxes[idx][0]) * scaling_ratio
            label[:, 1] = (self._labels[idx][:, 1] - self._boxes[idx][2]) * scaling_ratio

            if self._augment:
                # introduce randomness to the augmentation parameters and call augment()
                aug_rotate = self._aug_rotate * random.random()
                aug_y_translate = self._aug_translate * (2 * random.random() - 1) * image.shape[1]
                aug_x_translate = self._aug_translate * (2 * random.random() - 1) * image.shape[0]
                aug_flip = random.choice([self._aug_flip, False])
                image, label = self.augment(img=image,
                                            label=label,
                                            rot=aug_rotate,
                                            trans_y=aug_y_translate,
                                            trans_x=aug_x_translate,
                                            flip=aug_flip)

            # create heatmaps depending on heatmap_downscale
            label[:,:2] = label[:,:2] / self._heatmap_downscale
            heatmap_size = self._input_size[0] / self._heatmap_downscale
            heatmap_size = int(round(heatmap_size))
            heatmap_size = (heatmap_size, heatmap_size)
            heatmaps = create_heatmaps(label, heatmap_size)

            image = torch.tensor(image).permute(2, 0, 1)

            return image, heatmaps

    def __len__(self):
        return len(self._images)


    @classmethod
    def augment(cls, img: np.ndarray, label: np.ndarray, rot: float, trans_x: float, trans_y: float, flip: bool):
        """
        Geometric augmentation of the image and adjustment of the labels.
        """

        height, width = img.shape[:2]
        label = np.copy(label)

        # unpad image back to the bounding-box crop
        # I would personally call augment() before padding the image, but I won't:
        # "augment() takes an image img as a numpy array that is already cropped and padded to quadratic size"
        img_greyscale = np.max(img, axis=2)
        row_indices, col_indices = np.where(img_greyscale != 0)
        max_row = np.max(row_indices)
        max_col = np.max(col_indices)
        img = img[:max_row+1, :max_col+1, :]
        height_unpadded, width_unpadded = img.shape[:2]

        ### ROTATE ###

        # put image into center of a new (likely larger) array to keep entire rotated image visible
        height_rotated = max(128,
                             np.ceil(np.abs(
                                 np.cos(np.deg2rad(rot))*height_unpadded
                                 + np.sin(np.deg2rad(rot))*width_unpadded)
                             ).astype(int)
                             )
        width_rotated = max(128,
                            np.ceil(np.abs(
                                np.cos(np.deg2rad(rot))*width_unpadded
                                + np.sin(np.deg2rad(rot))*height_unpadded)
                            ).astype(int)
                            )
        img_rotated = np.zeros((height_rotated, width_rotated, img.shape[2]))
        y_pad_rotate = np.round((height_rotated-height_unpadded)/2).astype(int)
        x_pad_rotate = np.round((width_rotated-width_unpadded)/2).astype(int)
        img_rotated[y_pad_rotate:y_pad_rotate+height_unpadded, x_pad_rotate:x_pad_rotate+width_unpadded,:] = img

        # adapt keypoints to the new array
        keypoints = label[:,:2]
        keypoints[:,0] = keypoints[:,0] + x_pad_rotate
        keypoints[:,1] = keypoints[:,1] + y_pad_rotate

        # create rotation matrix and rotate image inside the new array
        center = (width_rotated / 2, height_rotated / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -rot, scale=1.0)
        img_rotated = cv2.warpAffine(img_rotated, rotation_matrix, (width_rotated, height_rotated))
        img = img_rotated

        # make keypoints homogeneous, rotate them, put them back into label
        keypoints_rotated = np.ones_like(label)
        keypoints_rotated[:,:2] = keypoints
        keypoints_rotated = (rotation_matrix @ keypoints_rotated.T).T
        label[:,:2] = keypoints_rotated

        ### TRANSLATE ###

        # round trans_y and trans_x parameters as they are passed as float
        trans_y = np.round(trans_y).astype(int)
        trans_x = np.round(trans_x).astype(int)

        # translate image in y direction
        img_y_translated = np.zeros_like(img)
        if trans_y >= 0:
            img_y_translated[trans_y:,:,:] = img[:img.shape[0]-trans_y,:,:]
        elif trans_y < 0:
            img_y_translated[:img.shape[0]+trans_y,:,:] = img[-trans_y:,:,:]

        # translate y-translated image in x direction
        img_translated = np.zeros_like(img)
        if trans_x >= 0:
            img_translated[:,trans_x:,:] = img_y_translated[:,:img.shape[1]-trans_x,:]
        elif trans_x < 0:
            img_translated[:,:img.shape[1]+trans_x,:] = img_y_translated[:,-trans_x:,:]

        img = img_translated

        # translate keypoints inside label
        label[:,1] = label[:,1] + trans_y
        label[:,0] = label[:,0] + trans_x

        ### FLIP ###

        if flip:
            # flip image and keypoints inside label
            img = cv2.flip(img, 1)
            label[:,0] = width_rotated - label[:,0] - 1

            # permute label to account for mirroring of the depicted person
            permutation_order = [0, 4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9, 15, 16, 13, 14]
            label = label[permutation_order]

        ### CROP ###

        # crop image back to input size and adjust keypoints accordingly
        top_margin = np.floor((height_rotated - height) / 2).astype(int)
        left_margin = np.floor((width_rotated - width) / 2).astype(int)
        img = img[top_margin : top_margin + height, left_margin : left_margin + width]
        label[:,1] = label[:,1] - top_margin
        label[:,0] = label[:,0] - left_margin

        # identify keypoints that have been cropped off and set their visibility to zero
        invisible_keypoints = (
                (label[:,1] < 0) |
                (label[:,1] >= height) |
                (label[:,0] < 0) |
                (label[:,0] >= width)
        )
        label[invisible_keypoints,2] = 0

        return img, label



# the following function was added after submitting assignment_5
def create_skijump_subsets(dataset_path: str, batch_size=16, image_size=(128, 128), heatmap_downscale=2) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    # dataset_path is the path to the base folder containing the annotations and the images in subdirectories (do not rename
    # them or the files)

    annotations_dir = os.path.join(dataset_path, "annotations")
    images_dir = os.path.join(dataset_path, "annotated_frames")

    train_images, train_labels, train_boxes = load_dataset(
        os.path.join(annotations_dir, 'train.csv'),
        images_dir
    )
    val_images, val_labels, val_boxes = load_dataset(
        os.path.join(annotations_dir, 'val.csv'),
        images_dir
    )
    test_images, test_labels, test_boxes = load_dataset(    # test_labels are not real labels, not sure what to do here atm
        os.path.join(annotations_dir, 'test.csv'),
        images_dir,
    )

    train_dataset = SkijumpDataset(
        train_images, train_labels, train_boxes,
        input_size=image_size,
        validation_mode=False,
        heatmap_downscale=heatmap_downscale,
        augment=True
    )
    val_dataset = SkijumpDataset(
        val_images, val_labels, val_boxes,
        input_size=image_size,
        validation_mode=True,
        heatmap_downscale=heatmap_downscale,
        augment=False
    )
    test_dataset = SkijumpDataset(                          # test_labels are not real labels, not sure what to do here atm
        test_images, test_labels, test_boxes,
        input_size=image_size,
        validation_mode=True,
        heatmap_downscale=heatmap_downscale,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader