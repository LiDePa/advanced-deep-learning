from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import torch


class Metric(ABC):

    def __init__(self: Metric, *, device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> None:
        self._device = device
        self._init_running_parameters()

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

    @abstractmethod
    def _init_running_parameters(self: Metric) -> None:
        raise NotImplementedError()


class MeanIntersectionOverUnion(Metric):

    def __init__(self, num_classes: int, *, ignore_class: int | None = None, **kwargs):
        self._num_classes = num_classes
        self._ignore_class = ignore_class
        super(MeanIntersectionOverUnion, self).__init__(**kwargs)

    def _update(self: MeanIntersectionOverUnion, predictions: torch.Tensor, labels: torch.Tensor):
        """Updates the inner state of this metric such that the mean intersection over union can be calculated.

        Args:
            predictions (torch.Tensor): Predicions against which the mean intersection over union will be calculated.
            labels (torch.Tensor): Ground truth against which the mean intersection over union will be calculated.
        """
        predictions = predictions.to(self._device)
        labels = labels.to(self._device)
        for cls in range(self._num_classes):
            TP = torch.sum(((predictions == cls) & (labels == cls)))
            if self._ignore_class is not None:
                FP = torch.sum(((predictions == cls) & (
                    labels != cls) & (labels != self._ignore_class)))
            else:
                FP = torch.sum(((predictions == cls) & (labels != cls)))
            FN = torch.sum(((predictions != cls) & (labels == cls)))
            union = (TP + FP + FN)
            self._intersections[cls] += TP
            self._unions[cls] += union

    def _compute(self: MeanIntersectionOverUnion) -> float:
        """Computes the mean intersection over union of the currently seen samples.

        Returns:
            float: The mean intersection over union.
        """
        ious = self._intersections / self._unions
        # When there was not a single pixel detected _unions does not contain anything, dividing by zero.
        ious = torch.nan_to_num(ious)
        ious[torch.logical_and(self._intersections == 0, self._unions == 0)] = 1
        mean_iou = ious.mean()
        return mean_iou

    def _reset(self: MeanIntersectionOverUnion):
        """Resets the inner state of this metric.
        """
        self._init_running_parameters()

    def _init_running_parameters(self: MeanIntersectionOverUnion) -> None:
        """Initializes the inner state of this metric.
        """
        self._intersections = torch.zeros(
            (self._num_classes,), dtype=torch.int, device=self._device)
        self._unions = torch.zeros(
            (self._num_classes,), dtype=torch.int, device=self._device)
