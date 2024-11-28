from __future__ import annotations
from typing import Callable, Dict, List

import torch.utils
import torch.utils.data

from .transforms import Compose, Transform
from .engine import Engine

import torch
import torch.nn as nn
import torch.cuda.amp as amp

#############################################################
#                      Exercise 3.1                         #
#############################################################

def get_ss_train_step_hl(
    teacher: nn.Module,
    student: nn.Module,
    loss_fn,
    optimizer,
    *,
    use_dropout_perturbation: bool = False,
    scaler: amp.GradScaler = amp.GradScaler(),
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    autocast_device_type: str = "cuda",
    transforms: List[Transform] = list()
) -> Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Generates a training step function for self-supervised model training on hard pseudo-labels.

    Args:
        teacher (nn.Module): The teacher model used to generate hard pseudo-labels.
        student (nn.Module): The student model on which the gradient update will be performed.
        loss_fn (_type_): The loss function used.
        optimizer (_type_): The optimizer for performing the parameter update.
        use_dropout_perturbation (bool, optional): Wether or not to use dropout in the student model. Defaults to False.
        scaler (amp.GradScaler, optional): The gradient scaling function used for automatic mixed precision training. Defaults to amp.GradScaler().
        device (_type_, optional): The device the training step will be performed on. Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".
        autocast_device_type (str, optional): The device type for automatic mixed precision training. Defaults to "cuda".
        transforms (List[Transform], optional): A list of data transforms applied after generating hard pseudo-labels. Can be used for pseudo-label filtering and similar transforms. Defaults to list().

    Returns:
        Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: A training step for the given model. Will perform a single parameter update.
    """

    transforms = Compose(*transforms)

    def ss_train_step_hl(_engine: Engine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a single parameter update for the student model by generating hard pseudo-labels on the fly.

        Args:
            _engine (Engine): The engine this training step is called by. Not used.
            batch (Dict[str, torch.Tensor]): A single batch of unlabeled training data.

        Returns:
            Dict[str, torch.Tensor]: The original minibatch with the student output and loss added.
        """

        raise NotImplementedError(
            f'{ss_train_step_hl.__name__} has not been implemented yet.')

        # TODO: You need to write a training step function for self-training using hard pseudo labels.
        #
        #       To do this you need to first generate hard pseudo-labels using the teacher.
        #       Do not use automatic mixed precision during this step.
        #
        #       You then apply the given transforms to the data (important for later exercises).
        #       To do so, you need to pass them a dictionary containing "x" (the unlabeled images), 
        #       the "confidences" of the predicted pseudo-labels, and the "pseudo_labels".
        #       For more information refer to the assignment sheet.
        #
        #       You then perform a regular parameter update using the given optimizer on the student
        #       model. This is also where you should use automatic mixed precision.
        #
        #       The return statement given to you should not be changed and is required for logging.
        #       "prediction" should be the output of the student model and "loss" should be the loss.

        return batch | dict(prediction=prediction, loss=loss.item())

    return ss_train_step_hl

#############################################################
#                      Exercise 3.2                         #
#############################################################

def get_ss_train_step_sl(
    teacher: nn.Module,
    student: nn.Module,
    optimizer,
    *,
    scaler: amp.GradScaler = amp.GradScaler(),
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    autocast_device_type: str = "cuda",
) -> Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Generates a training step function for self-supervised training using nothing but teacher confidence values.

    Args:
        teacher (nn.Module): The teacher model used to generate the target confidence distribution.
        student (nn.Module): The student model a parameter update will be performed on.
        optimizer (_type_): The optimizer used to perform the parameter update.
        scaler (amp.GradScaler, optional): A gradient scaling function for automatic mixed precision training. Defaults to amp.GradScaler().
        device (_type_, optional): The device the training step will be performed on. Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".
        autocast_device_type (str, optional): The device type for automatic mixed precision training. Defaults to "cuda".

    Returns:
        Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: _description_
    """

    def ss_train_step_sl(_engine: Engine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a single parameter update on the student using output confidences directly. 

        Args:
            _engine (Engine): The engine the training step is called by.
            batch (Dict[str, torch.Tensor]): A minibatch of unlabeled training data.

        Returns:
            Dict[str, torch.Tensor]: The original minibatch with the student predictions and loss added.
        """

        raise NotImplementedError(
            f'{get_ss_train_step_sl.__name__} has not been implemented yet.')

        # TODO: You need to write a training step function for self-training using soft labels.
        #       To do this you first need to generate the per-class confidences of the teacher model
        #       on the given minibatch of unlabeled data.
        #       You then need to do the same for the student and perform a parameter update using the
        #       previously defined loss and given optimizer.
        #
        #       The given return statement is required for logging, and should be the same as in the 
        #       training step using hard pseudo-labels.

        return batch | dict(prediction=student_output, loss=loss.item())

    return ss_train_step_sl


def class_weighted_confidence_loss(student_confidences: torch.Tensor, teacher_confidences: torch.Tensor) -> torch.Tensor:
    """Calculates the per-class weighted confidence loss. Similar to MSE, but with errors being weighted per class.

    Args:
        student_confidences (torch.Tensor): The per-class confidences of the student model.
        teacher_confidences (torch.Tensor): The per-class confidences of the teacher model.

    Returns:
        torch.Tensor: The total loss.
    """

    raise NotImplementedError(
        f'{class_weighted_confidence_loss.__name__} has not been implemented yet.')

    # TODO: Calculate the class weighted confidence loss. This is essentially mean squared error
    #       with the error of each class weighted differently according to the mean confidence.
    #
    #       First calculate the mean confidence of the teacher model per class. This should
    #       give you a total of 15 values. Then calculate the squared error per-pixel-per-class
    #       and scale the error. Then take the mean.
    #       For more information refer to the assignment sheet.


#############################################################
#                      DO NOT MODIFY                        #
#############################################################

def get_supervised_train_step(
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
        device (_type_, optional): The device all parameters are moved to during the training step. Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".
        autocast_device_type (str, optional): The device the torch.autocast() call be performed to. Defaults to 'cuda'.

    Returns:
        Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: A training step for the given model. Will perform a single gradient update.
    """

    def train_step(_engine: Engine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a single training step for the given batch.

        Args:
            _engine (Engine): The engine the model is being trained by.
            batch (Dict[str, torch.Tensor]): The batch for which a single gradient update should be performed.

        Returns:
            Dict[str, torch.Tensor]: Returns the batch + the outputs produced by the model and the current loss as a scalar.
        """

        model.train()
        optimizer.zero_grad()
        x = batch["x"].to(device)
        y = batch["y"].to(device).long()
        with torch.autocast(device_type=autocast_device_type, dtype=torch.float16):
            prediction = model(x)
            loss = loss_fn(prediction, y).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return batch | dict(prediction=prediction, loss=loss.item())

    return train_step


def get_validation_step(
        model: nn.Module,
        *,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
) -> Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Generates a validation step function.

    Args:
        model (nn.Module): The model that is being validated.
        device (_type_, optional): The device all the parameters are moved to during validation. Defaults to "cuda:0" if torch.cuda.is_available() else "cpu".

    Returns:
        Callable[[Engine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: A validation step for the given model. Only produces model predictions.
    """

    @torch.no_grad()
    def validation_step(_engine: Engine, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs a single validation step for the given batch.

        Args:
            _engine (Engine): The engine the model is being validated by.
            batch (Dict[str, torch.Tensor]): The batch for which the validation is performed.

        Returns:
            Dict[str, torch.Tensor]: The validation batch + the outputs produced by the model as well as the actual class predictions.
        """
        model.eval()
        outputs = model(batch["x"].to(device=device),
                        use_dropout_perturbation=False)
        prediction = torch.argmax(outputs, dim=1)
        return batch | dict(outputs=outputs, prediction=prediction)

    return validation_step
