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
    model.to(device)
    best_accuracy = 0.0
    best_checkpoint = os.path.join(log_dir, "best_checkpoint.pth")

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_predictions = model(x)
            loss = torch.nn.functional.cross_entropy(y_predictions, y)
            loss.backward()
            optimizer.step()

        #get tensor containing accuracy for each class on the validation set and calculate the overall mean
        class_accuracies = evaluation(model, val_loader, class_names, device)
        mean_accuracy = np.mean(class_accuracies)

        #update best_accuracy and best_checkpoint if new accuracy is better
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            torch.save(model.state_dict(), best_checkpoint)

        print("Current mean: %.2f" % mean_accuracy, "Best: %.2f" % best_accuracy)


def evaluation(model: Module, val_loader: DataLoader, classes: List[str], device: torch.device):
    """
    Evaluation routine for the classification task
    :param model: the model to be evaluated
    :param val_loader: the data loader for the validation data
    :param classes: list of class names (order matching the label indices)
    :param device: the device to evaluate on
    :return: List of accuracies in percent for each class in same order as the classes list
    """
    model.eval()
    class_scores = np.full(len(classes), 0, float)
    class_totals = np.full(len(classes), 0, int)

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_predictions = model(x)
            classes_predicted = torch.max(y_predictions, 1)[1] #returns tensor of size batch-length with predicted class for each sample

            for i in range(len(classes_predicted)):
                class_totals[y[i]] += 1
                if classes_predicted[i] == y[i]:
                    class_scores[y[i]] += 1

    class_accuracies = 100 * class_scores / class_totals
    return class_accuracies
