# -*- coding: utf-8 -*-
import numpy as np



# the following function was added after submitting assignment_5
def pck(annotations, predictions, torso_indices, t=0.1):
    annotations = np.array(annotations).astype(np.float32)  # Shape: (n_images, n_keypoints, 3)
    predictions = np.array(predictions).astype(np.float32)  # Shape: (n_images, n_keypoints, 3)

    # Extract torso keypoints (left hip and right shoulder)
    left_hip = annotations[:, torso_indices[0], :2].astype(np.float32)
    right_shoulder = annotations[:, torso_indices[1], :2].astype(np.float32)

    # calculate torso sizes and threshold distances
    d_torso = np.linalg.norm(left_hip - right_shoulder, axis=1)
    d_max = d_torso * t

    # replace all true invisible keypoints with nan
    visible_mask = annotations[..., 2] != 0
    visible_mask_broadcast = visible_mask[:,:, np.newaxis]
    annotations = np.where(visible_mask_broadcast, annotations, np.nan)
    predictions = np.where(visible_mask_broadcast, predictions, np.nan)

    # identify where annotations and predictions disagree on keypoint visibility
    visible_mask_pred = predictions[...,2] != 0
    false_invisibles = ~visible_mask_pred & visible_mask

    # calculate distances between predicted and ground truth keypoints while keeping invisible keypoints as nan
    d_kp = np.linalg.norm(annotations[..., :2] - predictions[..., :2], axis=2)

    # identify correct keypoints and carry invisible ones as nan
    nan_mask = np.isnan(d_kp)
    correct_mask = d_kp < d_max[:, np.newaxis]
    correct_mask = correct_mask & ~false_invisibles
    correct_mask = np.where(nan_mask, np.nan, correct_mask)

    # calculate pck per keypoint and overall pck
    pck_per_keypoint = np.nanmean(correct_mask, axis=0)
    pck_overall= np.mean(pck_per_keypoint)

    return pck_overall, pck_per_keypoint