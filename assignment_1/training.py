# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
        model: Module, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
        class_names: List[str], epochs: int, log_dir, ema_model: Module = None
        ):
    """
    Basic training routine for the  classification task
    :param model: Neural network to train
    :param optimizer: optimizer to use
    :param val_loader: data loader for validation data
    :param train_loader: data loader for training data
    :param device: Device to use for training
    :param epochs: Number of epochs to train
    :param class_names: List of class names, order matching the label indices
    :param log_dir: Directory to log to (for tensorboard and checkpoints)
    :param ema_model: Used in Exercise 1.2(c)
    :return:
    """
    raise NotImplementedError


def evaluation(model: Module, val_loader: DataLoader, classes: List[str], device: torch.device):
    """
    Evaluation routine for the classification task
    :param model: the model to be evaluated
    :param val_loader: the data loader for the validation data
    :param classes: list of class names (order matching the label indices)
    :param device: the device to evaluate on
    :return: List of accuracies in percent for each class in same order as the classes list
    """
    raise NotImplementedError
