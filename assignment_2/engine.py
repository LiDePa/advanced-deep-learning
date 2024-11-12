from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict
import torch

from torch.utils.data import DataLoader
import tqdm


class Event(Enum):

    ENGINE_STARTED = -2
    EPOCH_STARTED = -1
    ITERATION_STARTED = 0
    ITERATION_COMPLETED = 1
    EPOCH_COMPLETED = 2
    ENGINE_FINISHED = 3


class Engine:

    def __init__(self: Event, process_fn: Callable[[Engine, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]) -> None:
        self._process_fn = process_fn
        self._iterations = -1
        self._epochs = -1
        self._last_output = None
        self._event_handlers = {event: [] for event in Event}

    @property
    def iteration(self: Engine) -> int:
        return self._iterations

    @property
    def epoch(self: Engine) -> int:
        return self._epochs

    @property
    def last_output(self: Engine) -> Dict[str, torch.Tensor]:
        return self._last_output

    def run(self: Engine, data_loader: DataLoader, *, epochs: int = 1, disable_tqdm: bool = False) -> None:
        self._emit(Event.ENGINE_STARTED, 0)
        with tqdm.tqdm(range(epochs), desc="epochs", disable=disable_tqdm) as pbar:
            for epoch in pbar:
                self._epochs = epoch
                self._emit(Event.EPOCH_STARTED, value=epoch)
                data_iter = iter(data_loader)
                for i, batch in enumerate(data_iter):
                    self._iterations += 1
                    pbar.set_postfix(iteration=i, refresh=True)
                    self._emit(Event.ITERATION_STARTED, value=self._iterations)
                    self._last_output = self._process_fn(self, batch)
                    self._emit(Event.ITERATION_COMPLETED,
                               value=self._iterations)
                self._emit(Event.EPOCH_COMPLETED, value=epoch)
        self._emit(Event.ENGINE_FINISHED, value=0)

    def add_event_handler(self: Engine, event: Event, handler: Callable[[Engine], Any], *, every: int = 1) -> None:
        self._event_handlers[event].append((every, handler))

    def on(self: Engine, event: Event, *, every: int = 1):

        def _on(handler: Callable[[Engine], Any]):
            self.add_event_handler(event=event, handler=handler, every=every)
            return handler

        return _on

    def _emit(self: Engine, event: Event, value: int) -> None:
        for every, event_handler in self._event_handlers[event]:
            if value % every == 0:
                event_handler(self)
