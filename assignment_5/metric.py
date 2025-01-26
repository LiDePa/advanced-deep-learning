# -*- coding: utf-8 -*-
import numpy as np


# the following function was added after submitting assignment_5
# it is mostly written by the deepseek chatbot and only slightly corrected
def pck(annotations, predictions, torso_indices, t=0.1):
    annotations = np.array(annotations).astype(np.float32)  # Shape: (n_images, n_keypoints, 3)
    predictions = np.array(predictions).astype(np.float32)  # Shape: (n_images, n_keypoints, 3)

    # Extract torso keypoints (left hip and right shoulder)
    left_hip = annotations[:, torso_indices[0], :2].astype(np.float32)
    right_shoulder = annotations[:, torso_indices[1], :2].astype(np.float32)

    # Calculate torso size as the Euclidean distance between the two torso keypoints
    torso_size = np.linalg.norm(left_hip - right_shoulder, axis=1)

    # Calculate Euclidean distance between predicted and ground truth keypoints
    distances = np.linalg.norm(annotations[..., :2] - predictions[..., :2], axis=2)

    # Create a mask for visible keypoints
    visible_mask = annotations[..., 2] != 0

    # Calculate PCK for each keypoint
    pck_per_keypoint = np.zeros(annotations.shape[1])
    for k in range(annotations.shape[1]):
        # Only consider visible keypoints
        keypoint_distances = distances[:, k][visible_mask[:, k]]
        keypoint_torso_size = torso_size[visible_mask[:, k]]

        # Calculate PCK for this keypoint
        pck_per_keypoint[k] = np.mean(keypoint_distances <= t * keypoint_torso_size)

    # Calculate overall PCK
    overall_pck = np.mean(pck_per_keypoint)

    return overall_pck, pck_per_keypoint