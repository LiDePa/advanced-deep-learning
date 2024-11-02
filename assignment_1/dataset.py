# -*- coding: utf-8 -*-

from typing import Tuple, List
import torch
from torch.nn import Bilinear
from torch.utils.data import DataLoader
import glob
import os
import numpy as np
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.xpu import device
from torchvision.transforms.v2 import PILToTensor



def get_simpsons_subsets(dataset_path):
    """
    Creates image lists and labels for the simpsons dataset
    :param dataset_path: path to the "simpsons" folder of the simpsons dataset (this is important, do not use another path logic!)
    :return: list of training images, training labels, validation images, validation labels, and class class names
    """

    #define training and validation lists to be filled iteratively
    images_train = []
    labels_train = []
    images_val = []
    labels_val = []
    class_names = []
    character_label = 0

    #iterate through each character folder and split the images into a training and a validation set
    for character_path in sorted(glob.glob(os.path.join(dataset_path, "*"))):
        #get list of images in character folder and determine split size
        images = sorted(glob.glob(os.path.join(character_path,"*.jpg")))
        n_images = len(images)
        n_train = int(np.ceil(n_images*0.6))

        #add training and validation images to the respective lists
        images_train.extend(images[:n_train])
        images_val.extend(images[n_train:])

        #add the correct amount of labels to the respective lists
        labels_train.extend([character_label] * n_train)
        labels_val.extend([character_label] * (n_images - n_train))

        #add character name to the class names list
        class_names.append(os.path.basename(character_path))

        character_label += 1

        return images_train, labels_train, images_val, labels_val, class_names


class SimpsonsDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, class_names, is_validation):
        """
        :param images: A List of image filenames
        :param labels: A List of corresponding labels
        :param class_names: A List of class names, corresponding to the labels
        :param is_validation: Flag indicating evaluation or training mode
        yields Image as numpy array and label
        """
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.is_validation = is_validation
        self.pil_to_tensor = PILToTensor()
        self.transform = transforms.Compose([
            transforms.Resize(127, max_size=128),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image.thumbnail((128,128), Image.LANCZOS)

        width = image.width
        height = image.height
        x = self.pil_to_tensor(image)
        if width < 128:
            self.x = x.transforms.Pad(0,0, 128-width, 0)
        elif height < 128:
            self.x =x.transforms.Pad(0,0,0,128-height)


        self.y = self.labels[idx]
        return self.x, self.y

    def __len__(self):
        return len(self.x)


def get_dataloader(dataset_path) -> Tuple[DataLoader, DataLoader, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #create train dataset object and send it to device
    #create val dataset and send it to device
    #return both datasets
    raise NotImplementedError
