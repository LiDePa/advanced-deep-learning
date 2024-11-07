# -*- coding: utf-8 -*-
import sys
from typing import Tuple, List
import torch
from torch.nn import Bilinear
from torch.utils.data import DataLoader
import glob
import os
import numpy as np
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import PILToTensor
from PIL import Image



def get_simpsons_subsets(dataset_path):
    # define training and validation lists to be filled iteratively
    images_train = []
    labels_train = []
    images_val = []
    labels_val = []
    class_names = []
    character_label = 0

    # iterate through each character folder and split the images into a training and a validation set
    for character_path in sorted(glob.glob(os.path.join(dataset_path, "*"))):
        # get list of image paths
        images = sorted(glob.glob(os.path.join(character_path,"*.jpg")))

        # get number of images in folder and calculate number of images to use for training
        n_images = len(images)
        n_train = int(np.ceil(n_images*0.6))

        # slice list of image paths into training and validation set and add them to their respective lists
        images_train.extend(images[:n_train])
        images_val.extend(images[n_train:])

        # add class names to the respective list
        class_names.append(os.path.basename(character_path))

        # add the correct amount of labels to their respective lists and increase integer for next loop
        labels_train.extend([character_label] * n_train)
        labels_val.extend([character_label] * (n_images - n_train))
        character_label += 1

    return images_train, labels_train, images_val, labels_val, class_names



class SimpsonsDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, class_names, is_validation):
        self.images = images
        self.labels = labels
        self.class_names = class_names
        self.is_validation = is_validation
        self.pil_to_tensor = PILToTensor()
        self.normalize = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    # resize the image to 128x128 without distortion or cropping (padding with zeros);
    # normalize to ImageNet values and return both image and label as a tensor
    def __getitem__(self, idx):
        # open image as PIL
        image = Image.open(self.images[idx]).convert("RGB")

        # scale image such that the larger side is 128 pixels while keeping aspect ratio
        image.thumbnail((128,128))

        # pad image either on the right or bottom to arrive at a 128x128 image
        pad_image = transforms.Pad((0,0, 128-image.width, 128-image.height), fill=0, padding_mode='constant')
        image = pad_image(image)

        # turn PIL image into tensor and normalize to ImageNet values
        image_tensor = self.pil_to_tensor(image)
        x_tensor = self.normalize(image_tensor)

        y_tensor = torch.tensor(self.labels[idx])

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.images)



def get_dataloader(dataset_path) -> Tuple[DataLoader, DataLoader, List[str]]:
    if sys.gettrace() is None:
        num_workers = 16
    else:
        num_workers = 0

    simpsons_subsets = get_simpsons_subsets(dataset_path)
    train_dataset = SimpsonsDataset(images=simpsons_subsets[0],
                                    labels=simpsons_subsets[1],
                                    class_names=simpsons_subsets[4],
                                    is_validation=0)
    val_dataset = SimpsonsDataset(images=simpsons_subsets[2],
                                  labels=simpsons_subsets[3],
                                  class_names=simpsons_subsets[4],
                                  is_validation=1)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=64,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=64,
                                num_workers=num_workers,
                                shuffle=False,
                                drop_last=False)

    return train_dataloader, val_dataloader, simpsons_subsets[4]