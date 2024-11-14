from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class Transform(ABC):

    @abstractmethod
    def __call__(self: Transform, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class Compose(Transform):

    def __init__(self: Compose, *transforms: Transform):
        self._transforms = transforms

    def __call__(self: Compose, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for transform in self._transforms:
            sample = transform(sample)
        return sample


class Normalize(Transform):

    def __init__(
            self: Normalize,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            stdd: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:

        # TODO: Research Imagenet normalization. Put mean and standard deviation as default parameters of this function.
        #       Take note of the provided type hints, otherwise it might not work.

        if type(mean) is NotImplementedError:
            raise mean
        elif type(stdd) is NotImplementedError:
            raise stdd

        self._mean = torch.tensor(list(mean), dtype=torch.float)
        self._stdd = torch.tensor(list(stdd), dtype=torch.float)

    @property
    def mean(self: Normalize) -> torch.Tensor:
        return self._mean

    @property
    def stdd(self: Normalize) -> torch.Tensor:
        return self._stdd

    def __call__(self: Normalize, sample: Dict[str, torch.Tensor], *, targets: List[str] = list("x")) -> Dict[str, torch.Tensor]:
        """Normalizes the given images to the provided mean and standard deviation.

        Args:
            sample (Dict[str, torch.Tensor]): A single data sample.
            targets (List[str], optional): The targets (dic keys) within the sample that need to be normalized. Defaults to
            list("x").

        Returns:
            Dict[str, torch.Tensor]: The input sample with the targets normalized to the provided mean and standard deviation.
        """

        # add dimensions to mean and stdd to make them broadcast-able
        mean_broadcast = self._mean.view(3,1,1)
        stdd_broadcast = self._stdd.view(3,1,1)

        # update each target in the sample with its normalized tensor
        for target in targets:
            sample[target] = (sample[target] - mean_broadcast) / stdd_broadcast

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

        # TODO: Crop the the "x" and "y" entries of the sample to the provided size selecting a random region of the image

        raise NotImplementedError(
                "RandomCrop.__call__ has not been implemented yet.")


class RandomResizeCrop(RandomCrop):

    def __init__(self: RandomResizeCrop, crop_size: int = 192, scales: Tuple[float, float] = (0.75, 2.0)) -> None:
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
            Dict[str, torch.Tensor]: The input sample with the "x" and "y" entries resized randomly and then cropped to the
            correct size.
        """

        # TODO: Resize the "x" and "y" entries of the sample with a scale randomly selected from between the minimum and
        #  maximum value provided during initialization.
        #       Be careful which type of interpolation you use for the different inputs.
        #       Then use the provided line of code to then crop the images to the correct sizes.

        raise NotImplementedError(
                "RandomResizeCrop.__call__ has not been implemented yet.")

        return super(RandomResizeCrop, self).__call__(sample=sample)
