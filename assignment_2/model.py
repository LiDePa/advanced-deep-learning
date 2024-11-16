from torch.nn.functional import bilinear
from torchvision.models import resnet18, ResNet18_Weights
import torch


class ResNetSegmentationModel(torch.nn.Module):

    def __init__(self, num_classes, use_intermediate_features=False):

        super(ResNetSegmentationModel, self).__init__()

        # get a pretrained ResNet18 model
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # decrease the stride of the last two layers to retain a higher resolution
        self.backbone.layer3[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].conv2.stride = (1, 1)

        # construct a segmentation model using the convolutional layers of the pretrained resnet18 backbone
        self.model_segmentation = torch.nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            torch.nn.Conv2d(self.backbone.layer4[1].conv2.out_channels, num_classes, (1, 1)),
            torch.nn.Upsample((256, 512), mode="bilinear")) #scale_factor=8

    def forward(self, x):

        return self.model_segmentation(x)
