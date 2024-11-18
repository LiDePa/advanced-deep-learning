from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.cuda.amp as amp

from .engine import Engine


def get_train_step(
        model: nn.Module,
        loss_fn,
        optimizer,
        *,
        scaler: amp.GradScaler = amp.GradScaler(),
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        autocast_device_type: str = 'cuda'
        ) -> Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Generates a training step function for supervised model training.

    Args:
        model (nn.Module): The model that will be trained in the training step.
        loss_fn (_type_): The loss the model will be trained with.
        optimizer (_type_): The optimizer the model will be used with
        scaler (amp.GradScaler, optional): The gradient scaler used for mixed precision training. Defaults to amp.GradScaler().
        device (_type_, optional): The device all parameters are moved to during the training step. Defaults to "cuda:0" if
        torch.cuda.is_available() else "cpu".
        autocast_device_type (str, optional): The device the torch.autocast() call be performed to. Defaults to 'cuda'.

    Returns:
        Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: A training step for the given model. Will perform
        a single gradient update.
    """

    def train_step(_engine: Engine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a single training step for the given batch.

        Args:
            _engine (Engine): The engine the model is being trained by.
            batch (Dict[str, torch.Tensor]): The batch for which a single gradient update should be performed.

        Returns:
            Dict[str, torch.Tensor]: Returns the batch + the outputs produced by the model and the current loss as a scalar.
        """

        # put model into train mode and throw away gradients from previous batch
        model.train()
        optimizer.zero_grad()

        # send image and label tensors to device
        batch["x"] = batch["x"].to(device)
        batch["y"] = batch["y"].to(device)

        # perform forward propagation and calculate loss using mixed precision
        with torch.autocast(autocast_device_type):
            output = model(batch["x"])
            loss = loss_fn(output, batch["y"])

        # backpropagate and update weights using mixed precision
        scaler.scale(loss.mean()).backward()
        scaler.step(optimizer)
        scaler.update()

        # detach gradients from loss and output and create return dict
        return_value = batch
        return_value["loss"] = loss.mean().item()
        return_value["outputs"] = output.detach()

        return return_value

    return train_step


def get_validation_step(
        model: nn.Module,
        *,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        ) -> Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Generates a validation step function.

    Args:
        model (nn.Module): The model that is being validated.
        device (_type_, optional): The device all the parameters are moved to during validation. Defaults to "cuda:0" if
        torch.cuda.is_available() else "cpu".

    Returns:
        Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: A validation step for the given model. Only
        produces model predictions.
    """

    @torch.no_grad()
    def validation_step(_engine: Engine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a single validation step for the given batch.

        Args:
            _engine (Engine): The engine the model is being validated by.
            batch (Dict[str, torch.Tensor]): The batch for which the validation is performed.

        Returns:
            Dict[str, torch.Tensor]: The validation batch + the outputs produced by the model as well as the actual class
            predictions.
        """

        model.eval()

        # send image and label tensors to device
        batch["x"] = batch["x"].to(device)
        batch["y"] = batch["y"].to(device)

        # perform forward propagation using mixed precision
        with torch.autocast(device):
            output = model(batch["x"])

        # create return dict
        return_value = batch
        return_value["outputs"] = output.detach()
        return_value["predictions"] = torch.argmax(output, dim=1)

        return return_value

    return validation_step
