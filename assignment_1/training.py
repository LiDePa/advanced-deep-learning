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

from torch import amp


def train(
        model: Module, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
        class_names: List[str], epochs: int, log_dir, ema_model: Module = None
        ):
    best_accuracy = 0.0
    best_checkpoint = os.path.join(log_dir, "best_checkpoint.pth")
    last_checkpoint = os.path.join(log_dir, "last_checkpoint.pth")
    os.makedirs(log_dir, exist_ok=True)

    # create writer for performance logging
    writer = SummaryWriter(log_dir=log_dir)

    # create grad scaler for mixed precision training
    scaler = amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()
        true_positives = np.full(len(class_names), 0, float) # to calculate mean training accuracy
        class_totals = np.full(len(class_names), 0, int) # to calculate mean training accuracy
        loss_running = 0.0

        for batch in train_loader:
            # get new training batch with labels and send it to the device
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # throw away gradients from previous loop
            optimizer.zero_grad()

            # execute forward propagation on the batch and calculate the loss using mixed precision
            with amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                y_predictions = model(x)
                loss = torch.nn.functional.cross_entropy(y_predictions, y)

            # track per-class training accuracies and loss
            classes_predicted = torch.max(y_predictions, 1)[1]
            for sample in range(len(classes_predicted)):
                class_totals[y[sample]] += 1
                if classes_predicted[sample] == y[sample]:
                    true_positives[y[sample]] += 1
            loss_running += loss

            # backpropagate the scaled loss and update weights with unscaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            # update scaler for next iteration
            scaler.update()

        # calculate mean training accuracy and mean loss
        mean_accuracy_train = np.mean(100 * np.divide(true_positives, class_totals, where=class_totals != 0, out=np.zeros_like(true_positives)))
        loss_train = loss_running / len(train_loader)

        # calculate mean validation accuracy
        class_accuracies = evaluation(model, val_loader, class_names, device)
        mean_accuracy_val = np.mean(class_accuracies)

        # overwrite best_checkpoint if new accuracy is best so far
        if mean_accuracy_val >= best_accuracy:
            best_accuracy = mean_accuracy_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict()}, best_checkpoint)

        # save current model and optimizer state to resume training
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, last_checkpoint)

        # console output and tensorboard logging
        print("Epoch: ", epoch, "| Validation Accuracy Mean: %.2f" % mean_accuracy_val, "| Best: %.2f" % best_accuracy)
        writer.add_scalars("Plot", {"Train Accuracy": mean_accuracy_train, "Validation Accuracy": mean_accuracy_val}, epoch)
        writer.add_scalars("Loss", {"Train": loss_train}, epoch)

    writer.close()



def evaluation(model: Module, val_loader: DataLoader, classes: List[str], device: torch.device):
    model.eval()
    true_positives = np.full(len(classes), 0, float) # to calculate mean validation accuracy
    class_totals = np.full(len(classes), 0, int) # to calculate mean validation accuracy

    # disable gradient updates from validation set
    with torch.no_grad():
        for batch in val_loader:
            # get new validation batch and labels and send them to the device
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # execute forward propagation on the batch
            y_predictions = model(x)

            # track per-class validation accuracies
            classes_predicted = torch.max(y_predictions, 1)[1]
            for sample in range(len(classes_predicted)):
                class_totals[y[sample]] += 1
                if classes_predicted[sample] == y[sample]:
                    true_positives[y[sample]] += 1

    # calculate per-class validation accuracies
    class_accuracies = 100 * np.divide(true_positives, class_totals, where=class_totals!=0, out=np.zeros_like(true_positives))
    return class_accuracies



