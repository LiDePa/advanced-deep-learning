from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import torch

import numpy as np


class Metric(ABC):

    def __init__(self: Metric, *, device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> None:
        self._device = device
        self._reset()

    def update(self: Metric, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        return self._update(predictions.to(self._device), labels.to(self._device))

    def reset(self: Metric) -> None:
        return self._reset()

    def compute(self: Metric) -> Any:
        return self._compute()

    @abstractmethod
    def _update(self: Metric, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _reset(self: Metric) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _compute(self: Metric) -> Any:
        raise NotImplementedError()


class MeanIntersectionOverUnion(Metric):

    def __init__(self, num_classes: int, *, ignore_class: int | None = None, **kwargs):
        self._num_classes = num_classes
        self._ignore_class = ignore_class
        self._reset()
        super(MeanIntersectionOverUnion, self).__init__(**kwargs)

    def _update(self: MeanIntersectionOverUnion, predictions: torch.Tensor, labels: torch.Tensor):
        """Updates the inner state of this metric such that the mean intersection over union can be calculated.

        Args:
            predictions (torch.Tensor): Predictions from which the mean intersection over union will be calculated.
            labels (torch.Tensor): Ground truth from which the mean intersection over union will be calculated.
        """

        # TODO: Update the inner state of this metric such that mean intersection over union can be calculated.
        #       Do not actually calculate the mean intersection over union value in this function.
        #       Do not retain all predictions and labels, this will cause your GPU to run out of memory.
        #       Make sure to move everything to the correct device.



        # TODO: move to device
        # TODO: ignore 255
        # TODO: ask chatgpt whether a tensor is retained here

        # create masks of true positive and of false pixel predictions while ignoring class 255
        tp_mask = ((predictions == labels) & (labels != 255))
        f_mask = (~tp_mask & (labels != 255))
        tp_mask.to(self._device)
        f_mask.to(self._device)

        # mask predictions tensor with only the tp-pixels to obtain an image of the correctly predicted classes
        tp_image = tp_mask * predictions
        fp_image = f_mask * predictions
        fn_image = f_mask * labels

        # flatten the tensor and add the number of occurrences for each class to their running variables
        self._tp_running += torch.bincount(tp_image.flatten())

        raise NotImplementedError(
                "Mean intersection over union _update has not been implemented yet.")

    def _compute(self: MeanIntersectionOverUnion) -> float:
        """Computes the mean intersection over union of the currently seen samples.

        Returns:
            float: The mean intersection over union.
        """

        # TODO: Use the inner state of this metric to calculate the current mean section over union.
        #       Do not use anything but the inner state.




        raise NotImplementedError(
                "Mean intersection over union _compute has not been implemented yet.")


    def _reset(self: MeanIntersectionOverUnion):
        """Resets the inner state of this metric.
        """

        # TODO: Reset the inner state of this metric.
        # This function is also called in the __init__ function of the class.

        # tensors with running variables for true positives, false positives and false negatives of each class
        self._tp_running = torch.empty(dtype=torch.uint8)
        self._fp_running = torch.empty(dtype=torch.uint8)
        self._fn_running = torch.empty(dtype=torch.uint8)
