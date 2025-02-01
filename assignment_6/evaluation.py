# -*- coding: utf-8 -*-
import torch.utils.data


def get_coordinates_from_heatmaps(heatmaps: torch.Tensor, visibility_threshold, ratios, boxes):
    """
    Obtain keypoint coordinates from heatmaps
    :param heatmaps: heatmaps directly from the model with a batch dimension
    :param visibility_threshold: A threshold to handle keypoints that are not included in the image
    :param ratios: providing the ratio information to transform the coordinates back to the original image space (as calculated
    in Exercise 5.2b)
    :param boxes: providing the bounding box information to transform the coordinates back to the original image space (as
    calculated in Exercise 5.2b)
    :return: coordinates in origianl image size in shape dimensions batch_size x 17 x 3$
    """
    # TODO
    raise NotImplementedError


def evaluation(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device, visibility_threshold):
    """
    Evaluates the model on the given validation set. Calculates the PCK between the original annotations
    (given by the data loader) and the detected keypoints, with the ground truth coordinates as they are written in the csv file
    (NOT  on downsampled or cropped coordinates).
    :param val_loader: data loader for the validation set
    :param model: the model to evaluate
    :param device: the device to run the evaluation on
    :param visibility_threshold: A threshold to handle keypoints that are not included in the image
    :return: overall PCK at threshold 0.1
    """
    # TODO
    raise NotImplementedError


def load_and_evaluate_model(model_name: str, weights_path: str, dataset_path: str, result_file: str):
    """
    Loads a model and evaluates it on the test set, writes the file for the leaderboard
    :param model_name: resnet or hrnet
    :param weights_path: path to saved weights
    :param dataset_path: path to the dataset, general location, NOT specifically the test set
    :param result_file: path to the file that is written for the leaderboard, contains the results
    :return: nothing
    """
    assert model_name in ["resnet", "hrnet"]
    # TODO
    raise NotImplementedError
