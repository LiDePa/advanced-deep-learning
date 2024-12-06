from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class Transform(ABC):

    @abstractmethod
    def __call__(self: Transform, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

#############################################################
#                      Exercise 3.1b                        #
#############################################################


class CutOut(Transform):

    def __init__(
        self: CutOut,
        scales: Tuple[float, float] = (0.3, 0.7),
        ignore_class: int = 255,
    ) -> None:
        self.scales = scales
        self.ignore_class = ignore_class

    def __call__(
        self: CutOut,
        sample: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Cuts a random region from the given data sample. Region size is determined by input size and the scales parameter given in the constructor. Applied to input images and pseudo-labels.

        Args:
            sample (Dict[str, torch.Tensor]): Input sample from which a random region will be cut.

        Returns:
            Dict[str, torch.Tensor]: The input sample with a random region cut out. Cut pseudo-labels are set to the ignore class given in the constructor.
        """
        x = sample['x']
        pseudo_labels = sample['pseudo_labels']

        # TODO: The cut_out function expects the "bbox" parameter.
        #       This should be a bounding box to be cut out from the data sample using the
        #       (x, y, x_size, y_size) format. This box needs to be uniformly drawn from
        #       a random distribution. The size of the bounding box should be determined
        #       by the size of the input and the minimum and maximum scale given in the
        #       constructor of the CutOut class.

        # get sample size as tensor
        sample_size = torch.tensor(list(x.shape[2:]))

        # create random cutout size and position within self.scale boundaries; +1 because tensor.to(torch.int64) floors
        cutout_size = ((torch.rand(2) * (self.scales[1] - self.scales[0]) + self.scales[0]) * sample_size + 1).to(torch.int64)
        cutout_position = (torch.rand(2) * (sample_size - cutout_size + 1)).to(torch.int64)

        # create bounding box from cutout_size and cutout_position
        bbox = (cutout_position[0].item(), cutout_position[1].item(), cutout_size[0].item(), cutout_size[1].item())

        x, pseudo_labels = cut_out(
            x, pseudo_labels, bbox, ignore_class=self.ignore_class)

        sample['x'] = x
        sample['pseudo_labels'] = pseudo_labels

        return sample


def cut_out(
    x: torch.Tensor,
    pseudo_labels: torch.Tensor,
    bbox: Tuple[int, int, int, int],
    ignore_class: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cuts the region defined by a bounding box from the given image and pseudo-labels.

    Args:
        x (torch.Tensor): The image a region will be cut from. Cut pixels are set to 0.
        pseudo_labels (torch.Tensor): The pseudo-labels a region will be cut from. Cut pseudo-labels are set to the ignore class.
        bbox (Tuple[int, int, int, int]): The bounding box defining the region to be cut.
        ignore_class (int, optional): The value cut out pseudo-labels are set to. Defaults to 255.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """

    x_pos, y_pos, x_size, y_size = bbox

    # create cutout mask
    cutout_mask = torch.ones_like(pseudo_labels, dtype=torch.uint8)
    cutout_mask[:, y_pos:y_pos + y_size, x_pos:x_pos + x_size] = 0

    # make mask broadcastable and apply it to samples and labels
    x *= cutout_mask.unsqueeze(1)
    pseudo_labels[~cutout_mask.bool()] = ignore_class

    return x, pseudo_labels



#############################################################
#                      DO NOT MODIFY                        #
#############################################################


class Compose(Transform):

    def __init__(self: Compose, *transforms: Transform):
        self._transforms = transforms

    def __call__(self: Compose, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for transform in self._transforms:
            sample = transform(sample)
        return sample


class Normalize(Transform):

    def __init__(self: Normalize,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 stdd: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
        self._mean = torch.tensor(list(mean), dtype=torch.float)
        self._stdd = torch.tensor(list(stdd), dtype=torch.float)

    @property
    def mean(self: Normalize) -> torch.Tensor:
        return self._mean

    @property
    def stdd(self: Normalize) -> torch.Tensor:
        return self._stdd

    def __call__(self: Normalize,
                 sample: Dict[str, torch.Tensor],
                 *,
                 targets: List[str] = list("x")) -> Dict[str, torch.Tensor]:
        """Normalizes the given images to the provided mean and standard deviation.

        Args:
            sample (Dict[str, torch.Tensor]): A single data sample.
            targets (List[str], optional): The targets within the sample that need to be normalized. Defaults to list("x").

        Returns:
            Dict[str, torch.Tensor]: The input sample with the targets normalized to the provided mean and standard deviation.
        """

        for target in targets:
            sample[target] = (sample[target] - self.mean[:,
                              None, None]) / self.stdd[:, None, None]
        return sample


class RandomCrop(Transform):

    def __init__(self: RandomCrop, crop_size: int = 192):
        super(RandomCrop, self).__init__()
        self._crop_size = crop_size

    @property
    def crop_size(self: RandomCrop) -> int:
        return self._crop_size

    def __call__(self: RandomCrop, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly crops the "x" and "y" entries of a data sample to the provided crop size.

        Args:
            sample (Dict[str, torch.Tensor]): A single data sample.

        Returns:
            Dict[str, torch.Tensor]: The input sample with the "x" and "y" entries cropped to the provided size.
        """

        y_pos = torch.randint(low=0, high=max(
            sample['x'].size(1) - self.crop_size, 1), size=(1,))
        x_pos = torch.randint(low=0, high=max(
            sample['x'].size(2) - self.crop_size, 1), size=(1,))
        sample['x'] = sample['x'][:, y_pos:y_pos +
                                  self.crop_size, x_pos:x_pos + self.crop_size]
        if 'y' in sample:
            sample['y'] = sample['y'][y_pos:y_pos +
                                      self.crop_size, x_pos:x_pos + self.crop_size]
        return sample


class RandomResizeCrop(RandomCrop):

    def __init__(
        self: RandomResizeCrop,
        crop_size: int = 192,
        scales: Tuple[float, float] = (0.75, 2.0)
    ) -> None:
        super(RandomResizeCrop, self).__init__(crop_size=crop_size)
        self._scales = scales

    @property
    def scales(self: RandomResizeCrop) -> Tuple[float, float]:
        return self._scales

    def __call__(self: RandomResizeCrop, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """First randomly resizes the "x" and "y" entries of the provided data sample then crops to the provided size.

        Args:
            sample (Dict[str, torch.Tensor]): A single data sample.

        Returns:
            Dict[str, torch.Tensor]: The input sample with the "x" and "y" entries resized randomly and then cropped to the correct size.
        """

        scale_factor = ((self.scales[1] - self.scales[0])
                        * torch.rand((1,)) + self.scales[0]).item()
        sample['x'] = F.interpolate(
            sample['x'][None, :], scale_factor=scale_factor, mode='bilinear').squeeze()
        if 'y' in sample:
            sample['y'] = F.interpolate(
                sample['y'][None, None, :], scale_factor=scale_factor, mode='nearest').squeeze()
        return super(RandomResizeCrop, self).__call__(sample=sample)
