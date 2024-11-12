from __future__ import annotations
import glob
import os
from PIL import Image
from typing import Dict, List, Literal
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Compose, Transform


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

        raise NotImplementedError(
                "CarlaDataset.__init__ has not been implemented yet.")

    @property
    def dataset_path(self: CarlaDataset) -> str:
        # TODO: Return the dataset root path

        raise NotImplementedError(
                "CarlaDataset.dataset_path has not been implemented yet.")

    def __len__(self: CarlaDataset) -> int:
        # TODO: Return the length of this dataset, i.e. how many samples it contains.

        raise NotImplementedError(
                "CarlaDataset.__len__ has not been implemented yet.")

    def __getitem__(self: CarlaDataset, idx: int) -> Dict[str, torch.Tensor]:
        """Loads a single sample from disk.

        Args:
            idx (int): The index of the sample given the __len__ of this object.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing an input image and the ground truth segmentation mask with all
            given transformations applied.
        """

        # TODO: Implement loading a data sample from disk.
        #       A sample is a dictionary of tensors.
        #       A sample contains an entry "x" with the input image and an entry "y" with the training target.
        #       See instructions in the assignment for more details.
        #       As a last step apply all transformations that were provided during initialization.

        raise NotImplementedError("No CarlaDataset has been implemented yet.")


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

    # TODO: Implement selecting a datasplit from the carla3.0_for_students base folder.

    raise NotImplementedError(
            "get_segmentation_dataset has not been implemented yet.")
