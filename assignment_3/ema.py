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
            decay_rate (float, optional): The decay rate of the exponential moving average. Defaults to 0.9999.
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
    def _update(self, model: nn.Module, update_fn) -> None:
        """Updates the inner model parameters using the given model and update function.

        Args:
            model (nn.Module): The model to use with the update function.
            update_fn (_type_): The update function.
        """
        for ema_v, model_v in zip(self._model.state_dict().values(), model.state_dict().values()):
            model_v = model_v.to(device=self._device)
            ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module):
        """Updates the inner model parameters using an exponential moving average.

        Args:
            model (nn.Module): The model from which to update the inner model parameters.
        """
        self._update(model, update_fn=lambda e,
                     m: self._decay_rate * e + (1. - self._decay_rate) * m)

    def set(self, model: nn.Module):
        """Sets the inner model parameters to the given model.

        Args:
            model (nn.Module): The model to set.
        """
        self._update(model, update_fn=lambda e, m: m)

    def forward(self: EMAModel, *args, **kwargs) -> Any:
        """Forwards to the forward function of the inner model.

        Returns:
            Any: Returns the output of the inner model.
        """
        return self._model(*args, **kwargs)