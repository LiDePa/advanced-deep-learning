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

        # initialize resnet backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # initialize simple upsampling layer for readability
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        # initialize upsample blocks
        self.up1 = self._upsample_block(768, 256)
        self.up2 = self._upsample_block(384, 128)
        self.up3 = self._upsample_block(192, 64)
        self.up4 = self._upsample_block(128, 64)

        # initialize final heatmap output layer
        self.final_conv = torch.nn.Conv2d(64, num_keypoints, kernel_size=3, padding=1)

    # creates Conv2d and ReLU layer for each upsample block
    def _upsample_block(self, in_channels, out_channels):
        upsample_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        return upsample_block

    def forward(self, x):
        # encoder
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x_0 = self.backbone.relu(x)
        x = self.backbone.maxpool(x_0)
        x_1 = self.backbone.layer1(x)
        x_2 = self.backbone.layer2(x_1)
        x_3 = self.backbone.layer3(x_2)
        x = self.backbone.layer4(x_3)

        # upsample block 1
        x = self.upsample(x)
        x = torch.cat((x, x_3), dim=1)
        x = self.up1(x)

        # upsample block 2
        x = self.upsample(x)
        x = torch.cat((x, x_2), dim=1)
        x = self.up2(x)

        # upsample block 3
        x = self.upsample(x)
        x = torch.cat((x, x_1), dim=1)
        x = self.up3(x)

        # upsample block 4
        x = self.upsample(x)
        x = torch.cat((x, x_0), dim=1)
        x = self.up4(x)

        x = self.final_conv(x)

        return x