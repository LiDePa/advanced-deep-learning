# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18Model(torch.nn.Module):
    def __init__(self, num_classes):
        #get pretrained ResNet18 model
        super(ResNet18Model, self).__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        #freeze the pretrained weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        #get input size of final layer and replace it with custom FC layer depending on num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ConvNextTinyModel(torch.nn.Module):
    def __init__(self, num_classes):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
