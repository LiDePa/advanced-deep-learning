# -*- coding: utf-8 -*-

import torch
from torchvision.models import resnet18, ResNet18_Weights


class HRNetModel(torch.nn.Module):
    def __init__(self, num_keypoints):
        super(HRNetModel, self).__init__()
        # TODO: Implement the model, usable with different number of keypoints




class ResNet18Model(torch.nn.Module):
    def __init__(self, num_keypoints):
        super(ResNet18Model, self).__init__()
        # TODO: Implement the model, usable with different number of keypoints

        # get ResNet18 backbone and modify it
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(
            *list(self.backbone.children())[:-2],
            torch.nn.Upsample(scale_factor=2, mode="bilinear")
        )

    def forward(self):
        raise NotImplementedError