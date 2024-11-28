from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List
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
        self._event_handlers: Dict[Event, List[Callable[[Engine], Any]]] = {
            event: [] for event in Event}

    @property
    def iteration(self: Engine) -> int:
        """Current iteration that is being performed.
        """
        return self._iterations

    @property
    def epoch(self: Engine) -> int:
        """Current epoch that is being performed
        """
        return self._epochs

    @property
    def last_output(self: Engine) -> Dict[str, torch.Tensor]:
        """The last output given returned by the step function.
        """
        return self._last_output

    def run(self: Engine, data_loader: DataLoader, *, epochs: int = 1, disable_tqdm: bool = False) -> None:
        """Iterates over the given data loader for $n$ epochs. Calls the processing function for every data sample.

        Args:
            data_loader (DataLoader): The data loader to iterate over
            epochs (int, optional): How often to iterate over the data loader. Defaults to 1.
            disable_tqdm (bool, optional): Disables tqdm progress bar output. Useful for unit testing to not clutter the output. Defaults to False.
        """
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
        """Adds an event handler to this engine that will be called when ever the given event is emitted.
        The event handler is passed this engine instance.

        Args:
            event (Event): The event on which the handler will be triggered.
            handler (Callable[[Engine], Any]): The handler to be triggered. On being triggered receives this engine as input.
            every (int, optional): Controls the interval at which the event handler will be triggered. When set to 1, the handler will be called every time the event is emitted. Defaults to 1.
        """
        self._event_handlers[event].append((every, handler))

    def on(self: Engine, event: Event, *, every: int = 1):
        """Decorator to add the decorated function as an event handler to this engine.

        Args:
            event (Event): The event the handler should be triggered by.
            every (int, optional): Controls the interval at which the event handler will be triggered. When set to 1, the handler will be called every time the event is emitted. Defaults to 1.
        """

        def _on(handler: Callable[[Engine], Any]):
            self.add_event_handler(event=event, handler=handler, every=every)
            return handler

        return _on

    def _emit(self: Engine, event: Event, value: int) -> None:
        """Emits the given event internally. Triggers all event handlers that are registered on that event in the given interval.

        Args:
            event (Event): The event being emitted.
            value (int): The value of the event being emitted. Can be used to trigger events non-monotonically.
        """

        for _idx, (every, event_handler) in enumerate(self._event_handlers[event]):
            if value % every == 0:
                event_handler(self)
