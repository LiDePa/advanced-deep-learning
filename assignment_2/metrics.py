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


# TODO: make sure init looks like in template
class MeanIntersectionOverUnion(Metric):

    def __init__(self, num_classes: int, *, ignore_class: int | None = None, **kwargs):
        self._num_classes = num_classes
        self._ignore_class = ignore_class
        super(MeanIntersectionOverUnion, self).__init__(**kwargs) # had to move this up to initialize running tensors in _reset()
        self._reset()
        # super(MeanIntersectionOverUnion, self).__init__(**kwargs)

    def _update(self: MeanIntersectionOverUnion, predictions: torch.Tensor, labels: torch.Tensor):
        """Updates the inner state of this metric such that the mean intersection over union can be calculated.

        Args:
            predictions (torch.Tensor): Predictions from which the mean intersection over union will be calculated.
            labels (torch.Tensor): Ground truth from which the mean intersection over union will be calculated.
        """

        # TODO: catch case _ignore_class=None but does that mean i evaluate class 255? Can I expect num_classes to be one higher then?

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

        # calculate intersection over union for each class
        denominator = self._tp_running + self._fp_running + self._fn_running
        ious = torch.where(denominator > 0, self._tp_running / denominator, torch.tensor(0.0, device=self._device))

        return torch.mean(ious) # returns a tensor because event handlers in training.py expect a tensor, not a float (they are calling .item() on the result)


    def _reset(self: MeanIntersectionOverUnion):
        """Resets the inner state of this metric.
        """

        # create or reset tensors with running variables for true positives, false positives and false negatives of each class
        self._tp_running = torch.zeros(self._num_classes, dtype=torch.float32, device=self._device)
        self._fp_running = torch.zeros(self._num_classes, dtype=torch.float32, device=self._device)
        self._fn_running = torch.zeros(self._num_classes, dtype=torch.float32, device=self._device)