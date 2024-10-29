# -*- coding: utf-8 -*-
import torch


class ResNet18Model(torch.nn.Module):
    def __init__(self, num_classes):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ConvNextTinyModel(torch.nn.Module):
    def __init__(self, num_classes):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
