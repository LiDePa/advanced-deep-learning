from __future__ import annotations
import glob
import os
from PIL import Image
from typing import Dict, List, Literal
import numpy as np
import torch
import torch.utils

from .transforms import Compose, Transform


class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: List[str],
        segmentation_paths: List[str],
        *,
        transforms: List[Transform] = list(),
    ) -> None:
        super(SegmentationDataset, self).__init__()
        assert len(image_paths) == len(
            segmentation_paths), 'Expected the same number of segmentation masks as images'
        self.image_paths = image_paths
        self.segmentation_paths = segmentation_paths
        self.transforms = Compose(*transforms)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self: SegmentationDataset, idx) -> Dict[str, torch.Tensor]:
        image = load_image(self.image_paths[idx])
        segmentation_mask = load_label(self.segmentation_paths[idx])
        return self.transforms(
            dict(
                x=image,
                y=segmentation_mask
            )
        )


def load_image(image_path: str) -> torch.Tensor:
    """Loads an image from the given path in (3, H, W) format.

    Args:
        image_path (str): The path to load from.

    Returns:
        torch.Tensor: The image scaled to [0, 1] in torch.float type.
    """
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)
    image = image / 255.0
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image.copy()).to(dtype=torch.float)
    return image


def load_label(mask_path: str) -> torch.Tensor:
    """Loads a label from the given path in (H, W) format.

    Args:
        mask_path (str): The path to load from

    Returns:
        torch.Tensor: The label in torch.uint8 type.
    """
    label = Image.open(mask_path)
    label = np.asarray(label)
    label = np.transpose(label, (1, 0))
    label = torch.from_numpy(label.copy()).to(dtype=torch.uint8)
    return label


def get_segmentation_dataset(
    dataset_root: str,
    split: Literal["train", "val"],
    *,
    transforms: List[Transform] = list()
) -> SegmentationDataset:
    """Loads the carla_for_students3.0 dataset from the given root.

    Args:
        dataset_root (str): The location of the dataset.
        split (Literal[&quot;train&quot;, &quot;val&quot;]): The data split to be loaded.
        transforms (List[Transform], optional): A list of transforms that should be applied to every data sample. Defaults to list().

    Returns:
        SegmentationDataset: The dataset.
    """
    image_paths = sorted(glob.glob(os.path.join(
        dataset_root, split, 'images', '*.png')))
    segmentation_paths = sorted(glob.glob(os.path.join(
        dataset_root, split, 'segmentations', '*.png')))
    return SegmentationDataset(
        image_paths=image_paths,
        segmentation_paths=segmentation_paths,
        transforms=transforms
    )
