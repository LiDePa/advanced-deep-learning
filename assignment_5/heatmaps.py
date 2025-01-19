# -*- coding: utf-8 -*-
import numpy as np


def create_heatmaps(labels: np.ndarray, heatmap_size=(128, 128), sigma=2):
    sigma = 2
    x_coordinates = np.arange(heatmap_size[0], dtype=np.float32)
    y_coordinates = np.arange(heatmap_size[1], dtype=np.float32)
    x_meshgrid, y_meshgrid = np.meshgrid(x_coordinates, y_coordinates)
    x_meshgrids = np.zeros((labels.shape[0], *heatmap_size), dtype=np.float32)
    y_meshgrids = np.zeros((labels.shape[0], *heatmap_size), dtype=np.float32)
    #TODO: do broadcasting instead of the 2 lines above and beneath
    x_meshgrids[:] = x_meshgrid
    y_meshgrids[:] = y_meshgrid
    x_labels_vector = np.array(labels[:,0], dtype=np.float32)
    y_labels_vector = np.array(labels[:,1], dtype=np.float32)
    x_labels_matrix = np.broadcast_to(
        x_labels_vector[:, np.newaxis, np.newaxis], (len(x_labels_vector), *heatmap_size)
    )
    y_labels_matrix = np.broadcast_to(
        y_labels_vector[:, np.newaxis, np.newaxis], (len(y_labels_vector), *heatmap_size)
    )
    exponent = -(((x_meshgrids - x_labels_matrix)**2 + (y_meshgrids - y_labels_matrix)**2) / (2*(sigma**2)))
    heatmaps = np.exp(exponent)
    #TODO: try to use float YEEEES
    breakpoint()