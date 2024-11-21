from __future__ import annotations
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn


class EMAModel(nn.Module):

    def __init__(self: EMAModel,
                 model: nn.Module,
                 *,
                 decay_rate: float = 0.998,
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> None:
        """Creates an exponential moving average model of the given model.

        Args:
            model (nn.Module): The model for which to keep an exponential moving average
            decay_rate (float, optional): The decay rate of the exponential moving average. Defaults to 0.998.
            device (str, optional): The device on which to keep the inner model. Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".
        """
        super(EMAModel, self).__init__()

        self._model = deepcopy(model)
        self._model.eval()
        self._model.to(device=device)

        self._decay_rate = decay_rate
        self._device = device

    @property
    def model(self: EMAModel) -> nn.Module:
        return self._model

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Updates the inner model parameters using an exponential moving average.

        Args:
            model (nn.Module): The model from which to update the inner model parameters.
        """

        # iterate through all parameters  of the main model and update ema model according to the given formula
        for ema_param, model_param in zip(self._model.parameters(), model.parameters()):
            ema_param.copy_(self._decay_rate * ema_param + (1 - self._decay_rate) * model_param)

        # iterate through all buffers of the main model and update ema model according to the given formula
        for ema_buffer, model_buffer in zip(self._model.buffers(), model.buffers()):
            ema_buffer.copy_(self._decay_rate * ema_buffer + (1 - self._decay_rate) * model_buffer)

    def forward(self: EMAModel, *args, **kwargs) -> Any:
        """Forwards to the forward function of the inner model.

        Returns:
            Any: Returns the output of the inner model.
        """
        return self._model(*args, **kwargs)
