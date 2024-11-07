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

    # create grad scaler for mixed precision training
    scaler = amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()

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

            # backpropagate the scaled loss and update weights with unscaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            # update scaler for next iteration
            scaler.update()

        # get numpy array containing the accuracy [%] for each class on the validation set and calculate the overall mean
        class_accuracies = evaluation(model, val_loader, class_names, device)
        mean_accuracy = np.mean(class_accuracies)

        # update best_accuracy and best_checkpoint if new accuracy is better or the same
        if mean_accuracy >= best_accuracy:
            best_accuracy = mean_accuracy
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict()}, best_checkpoint)

        # save current model and optimizer state
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, last_checkpoint)

        print("Epoch: ", epoch, "Current mean: %.2f" % mean_accuracy, "Best: %.2f" % best_accuracy)



def evaluation(model: Module, val_loader: DataLoader, classes: List[str], device: torch.device):
    model.eval()
    class_scores = np.full(len(classes), 0, float)
    class_totals = np.full(len(classes), 0, int)

    with torch.no_grad():
        for batch in val_loader:
            # get new validation batch and labels and send them to the device
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # execute forward propagation on the batch and get a tensor of size batch-length with predicted class for each sample
            y_predictions = model(x)
            classes_predicted = torch.max(y_predictions, 1)[1]

            # count class appearances and correct predictions in class_totals and class_scores vectors
            for sample in range(len(classes_predicted)):
                class_totals[y[sample]] += 1
                if classes_predicted[sample] == y[sample]:
                    class_scores[y[sample]] += 1

    class_accuracies = 100 * class_scores / class_totals
    return class_accuracies
