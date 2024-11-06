# -*- coding: utf-8 -*-

import torch

import models
from dataset import get_dataloader
from training import train


def main(dataset_path, model_name, epochs, weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on " + device.type)

    # set up dataloaders and list of class names
    train_dataloader, val_dataloader, class_names = get_dataloader(dataset_path)

    # set up model and load weights if given
    model_class = getattr(models, model_name)
    model = model_class(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=0.0001)
    epochs_remaining = epochs
    if weights:
        checkpoint = torch.load(weights, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epochs_remaining = epochs - checkpoint["epoch"]

    # start training
    train(model, optimizer, train_dataloader, val_dataloader, device, class_names, epochs_remaining, "logs")


if __name__ == '__main__':
    dataset_path = "../datasets/simpsons"
    model_name = "ResNet18Model"
    epochs = 30
    weights = "logs/last_checkpoint.pth"
    main(dataset_path, model_name, epochs, weights)
