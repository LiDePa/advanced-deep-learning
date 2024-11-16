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



        # TODO: ask chatgpt whether a tensor is retained here
        # TODO: testing, also with _ignore_class=None

        # send predictions and labels tensors to the device
        predictions = predictions.to(self._device)
        labels = labels.to(self._device)

        # create masks of true positive and of false pixel predictions while excluding _ignore_class
        tp_mask = ((predictions == labels) & (labels != self._ignore_class))
        f_mask = (~tp_mask & (labels != self._ignore_class))

        # mask predictions and labels tensors to obtain all true positive, false positive and false negative classes
        tp_classes = predictions[tp_mask]
        fp_classes = predictions[f_mask]
        fn_classes = labels[f_mask]

        # count the number of occurrences of each class and update running variables
        self._tp_running += torch.bincount(tp_classes, minlength=self._num_classes)
        self._fp_running += torch.bincount(fp_classes, minlength=self._num_classes)
        self._fn_running += torch.bincount(fn_classes, minlength=self._num_classes)

    def _compute(self: MeanIntersectionOverUnion) -> float:
        """Computes the mean intersection over union of the currently seen samples.

        Returns:
            float: The mean intersection over union.
        """

        # TODO: Use the inner state of this metric to calculate the current mean section over union.
        #       Do not use anything but the inner state.

        # calculate IoU for each class
        denominator = self._tp_running + self._fp_running + self._fn_running
        ious = torch.where(denominator > 0, self._tp_running / denominator, torch.tensor(0.0, device=self._device))

        return torch.mean(ious).item()


    def _reset(self: MeanIntersectionOverUnion):
        """Resets the inner state of this metric.
        """

        # TODO: Reset the inner state of this metric.
        # This function is also called in the __init__ function of the class.

        # create or reset tensors with running variables for true positives, false positives and false negatives of each class
        self._tp_running = torch.zeros(self._num_classes, dtype=torch.float32, device=self._device)
        self._fp_running = torch.zeros(self._num_classes, dtype=torch.float32, device=self._device)
        self._fn_running = torch.zeros(self._num_classes, dtype=torch.float32, device=self._device)