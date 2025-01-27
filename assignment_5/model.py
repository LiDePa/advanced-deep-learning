import torch
from torchvision.models import resnet18, ResNet18_Weights



class ResNetHPEModel(torch.nn.Module):
    def __init__(self):
        # initialize base class
        super(ResNetHPEModel, self).__init__()

        # get ResNet18 backbone and omit the final class-prediction layers avgpool and fc
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])


    def forward(self, x):
        raise NotImplementedError