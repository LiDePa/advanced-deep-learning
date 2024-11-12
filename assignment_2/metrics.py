from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import torch


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

        raise NotImplementedError(
                "Mean intersection over union _reset has not been implemented yet.")
