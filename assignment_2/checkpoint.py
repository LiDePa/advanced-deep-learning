from __future__ import annotations
import os
from typing import Any, Dict

import torch

from .metrics import Metric


class BestCheckpointHandler:
    """A checkpointing strategy where only the model with the best performance is being saved.
    """

    def __init__(self: BestCheckpointHandler, output_dir: str, metric: Metric, *, filename: str = "") -> None:
        """Creates a checkpointing handler.

        Args:
            self (Checkpoint): The handler.
            output_dir (str): The directory where checkpoint will be saved.
            metric (Metric): The metric by which checkpoint is saved.
            filename (str, optional): Under which filename the model will be saved. Defaults to "".
        """

        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._metric = metric
        self._filename = filename
        self._best_value = float("-inf")
        self._last_output = None

    def checkpoint_if_necessary(self: BestCheckpointHandler, to_save: Dict[str, Any], *, filename_suffix: str = "") -> None:
        """Creates a checkpoint if the metric associated with this handler has improved since the last method call.

        Args:
            self (BestModelCheckpointHandler): The handler.
            to_save (Dict[str, Any]): A dictionary of objects that need to be saved according to the associated metric.
            filename_suffix (str, optional): Additional filename suffix that can be provided when no previous checkpoints should be overwritten. Defaults to "", i.e. an override.
        """
        curr_metric_value = self._metric.compute()
        if curr_metric_value > self._best_value:
            output_path = os.path.join(
                self._output_dir, f"{self._filename}{filename_suffix}.pth")
            torch.save(
                to_save,
                output_path
            )
            if self._last_output:
                os.remove(self._last_output)
            self._last_output = output_path
