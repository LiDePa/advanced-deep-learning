from __future__ import annotations
import glob
import os
from PIL import Image
from typing import Dict, List, Literal
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Compose, Transform

import torchvision.transforms.functional



class CarlaDataset(Dataset):

    def __init__(
            self: CarlaDataset,
            dataset_path: str,
            *,
            transforms: List[Transform] = list()
            ) -> None:
        """A dataset abstraction for the carla3.0_for_students dataset.

        Args:
            dataset_path (str): The root path of the dataset.
            transforms (List[Transform], optional): All the transformations that will be applied to each sample retrieved from
            this dataset. Defaults to list().
        """

        # TODO: Implement the initialization of a CarlaDataset.
        #       Use the "Compose" transform from transforms.py (not from torch.transforms!) to compose the transforms together.
        #       Do NOT load the whole dataset as images upon initialization, just store the paths.
        #       You should try to calculate all necessary information for fast loading here.

        self.transforms = Compose(*transforms)
        self.image_paths = sorted(glob.glob(os.path.join(dataset_path,"images/*.png")))
        self.label_paths = sorted(glob.glob(os.path.join(dataset_path,"segmentations/*.png")))
        self._dataset_path = dataset_path

    @property
    def dataset_path(self: CarlaDataset) -> str:
        return self._dataset_path

    def __len__(self: CarlaDataset) -> int:
        return len(self.image_paths)

    def __getitem__(self: CarlaDataset, idx: int) -> Dict[str, torch.Tensor]:
        """Loads a single sample from disk.

        Args:
            idx (int): The index of the sample given the __len__ of this object.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing an input image and the ground truth segmentation mask with all
            given transformations applied.
        """

        # load the image and its segmentation mask and convert them to tensors
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")
        image_tensor = torchvision.transforms.functional.to_tensor(image).float()
        label_tensor = torch.from_numpy(np.array(label)).to(torch.int8)

        # create sample dictionary
        sample = {
            "x": image_tensor,
            "y": label_tensor}

        # apply transformations if given
        if self.transforms:
            sample = self.transforms(sample)

        return sample



def get_carla_dataset(root: str, *, split: Literal["train", "val"], transforms: List[Transform] = list()) -> CarlaDataset:
    """Gives a split of the carla3.0_for_students dataset.

    Args:
        root (str): The dataset root path.
        split (Literal[&quot;train&quot;, &quot;val&quot;]): The choices of different datasets available.
        transforms (List[Transform], optional): The transformations that will be applied by the dataset to every sample.
        Defaults to list().

    Returns:
        SegmentationDataset: The correct split of the carla3.0_for_students dataset.
    """

    match split:
        case "train":
            return CarlaDataset(os.path.join(root, "train"), transforms=transforms)
        case "val":
            return CarlaDataset(os.path.join(root, "val"), transforms=transforms)
