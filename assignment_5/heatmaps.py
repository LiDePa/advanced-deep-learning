# -*- coding: utf-8 -*-
import numpy as np


def create_heatmaps(labels: np.ndarray, heatmap_size=(128, 128), sigma=2):
    n_keypoints = labels.shape[0]
    sigma = 2

    # create 2D-meshgrids of heatmap_size
    # and broadcast them along a new axis of length n_keypoints
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