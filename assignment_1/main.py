# -*- coding: utf-8 -*-

import torch

import models
from dataset import get_dataloader
from training import train


def main(dataset_path, model_name, epochs, weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on " + device.type)

    

    train_dataloader, val_dataloader, class_names = get_dataloader(dataset_path)
    model_class = getattr(models, model_name)
    model = model_class(len(class_names))
    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=0.0001)
    train(model, optimizer, train_dataloader, val_dataloader, device, class_names, epochs, "logs")


if __name__ == '__main__':
    dataset_path = "../datasets/simpsons"
    model_name = "ResNet18Model"
    epochs = 30
    weights = None
    main(dataset_path, model_name, epochs, weights)
