# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image


def create_heatmaps(labels: np.ndarray, heatmap_size=(128, 128), sigma=2):
    n_keypoints = labels.shape[0]

    # create 2D-meshgrids of heatmap_size for x and y coordinates
    # and broadcast both along a new axis of length n_keypoints
    x_coordinates = np.arange(heatmap_size[0], dtype=np.float32)
    y_coordinates = np.arange(heatmap_size[1], dtype=np.float32)
    x_meshgrid, y_meshgrid = np.meshgrid(x_coordinates, y_coordinates)
    x_meshgrids_3d = np.broadcast_to(
        x_meshgrid[np.newaxis, :, :], (n_keypoints, *heatmap_size)
    )
    y_meshgrids_3d = np.broadcast_to(
        y_meshgrid[np.newaxis, :, :], (n_keypoints, *heatmap_size)
    )

    # get x and y coordinates of the keypoints as individual vectors
    # and broadcast them along two new axes of shape heatmap_size
    x_labels_vector = np.array(labels[:,0], dtype=np.float32)
    y_labels_vector = np.array(labels[:,1], dtype=np.float32)
    x_labels_3d = np.broadcast_to(
        x_labels_vector[:, np.newaxis, np.newaxis], (n_keypoints, *heatmap_size)
    )
    y_labels_3d = np.broadcast_to(
        y_labels_vector[:, np.newaxis, np.newaxis], (n_keypoints, *heatmap_size)
    )

    # compute heatmaps of shape [n_keypoints, heatmap_size[0], heatmap_size[1]]
    exponent = -(((x_meshgrids_3d - x_labels_3d)**2 + (y_meshgrids_3d - y_labels_3d)**2) / (2*(sigma**2)))
    heatmaps = np.exp(exponent)

    return heatmaps



def plot_heatmap_confirmation(dataset_128: torch.utils.data.Dataset,
                              dataset_64: torch.utils.data.Dataset,
                              len_dataset: int):
    """
    creates .png images of 10 samples, 5 with 128px heatmaps and 5 with upscaled 64px heatmaps
    Args:
        dataset_128: a dataset created with 128px image and heatmap resolution
        dataset_64: a dataset created with 128px image and 64px heatmap resolution
        len_dataset: amount of samples in the dataset
    """
    # create output directory
    project_folder = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_folder, "output_plots")
    os.makedirs(output_dir, exist_ok=True)

    # get 5 random indices for each dataset
    indices_128 = random.sample(range(len_dataset), 5)
    indices_64 = random.sample(range(len_dataset), 5)

    for i in indices_128:
        image, heatmaps = dataset_128[i]
        heatmaps_2d = np.max(heatmaps, axis=0)

        image[:,:,0] = np.maximum(image[:,:,0], heatmaps_2d)

        plt.figure(figsize=(10,10))
        plt.imshow(image)

        output_path = os.path.join(output_dir, f"plot_{i+3}") # i+3 equals line index in .csv
        plt.savefig(output_path)
        plt.close()

    for i in indices_64:
        image, heatmaps = dataset_64[i]
        heatmaps_2d_64 = np.max(heatmaps, axis=0)
        heatmaps_2d = np.repeat(np.repeat(heatmaps_2d_64, 2, 0),2,1)

        image[:, :, 0] = np.maximum(image[:, :, 0], heatmaps_2d)

        plt.figure(figsize=(10,10))
        plt.imshow(image)

        output_path = os.path.join(output_dir, f"plot_{i+3}") # i+3 equals line index in .csv
        plt.savefig(output_path)
        plt.close()
