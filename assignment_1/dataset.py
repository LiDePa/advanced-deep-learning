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
    #train liste definieren
    #validation liste definieren
    for character_path in sorted(glob.glob(os.path.join(dataset_path, "*"))):
        n_train = np.ceil(len(glob.glob(os.path.join(character_path,"*.jpg")))*0.6)
        print(n_train)
        #element 1-n zu train liste hinzufügen
        #rest zu validation liste hinzufügen


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
