# -*- coding: utf-8 -*-

from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

import glob
import os
import numpy as np


def get_simpsons_subsets(dataset_path):
    """
    Creates image lists and labels for the simpsons dataset
    :param dataset_path: path to the "imgs" folder of the simpsons dataset (this is important, do not use another path logic!)
    :return: list of training images, training labels, validation images, validation labels, and class class names
    """

    #define training and validation lists to be filled iteratively
    images_train = []
    images_val = []
    labels_train = []
    labels_val = []
    character_label = 0

    #iterate through each character folder and split the images into a training and a validation set
    for character_path in sorted(glob.glob(os.path.join(dataset_path, "*"))):
        images = glob.glob(os.path.join(character_path,"*.jpg")) #sorting shouldn't be necessary here
        n_images = len(images)
        n_train = int(np.ceil(n_images*0.6))

        images_train.extend(images[:n_train])
        images_val.extend(images[n_train:])

        character_label += 1
        labels_train.extend([character_label] * n_train)
        labels_val.extend([character_label] * (n_images - n_train))

    print(images_train, labels_train)


class SimpsonsDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, class_names, is_validation):
        """
        :param images: A List of image filenames
        :param labels: A List of corresponding labels
        :param class_names: A List of class names, corresponding to the labels
        :param is_validation: Flag indicating evaluation or training mode
        yields Image as numpy array and label
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def get_dataloader(dataset_path) -> Tuple[DataLoader, DataLoader, List[str]]:
    raise NotImplementedError
