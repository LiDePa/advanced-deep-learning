from torchvision.models import resnet18, ResNet18_Weights
import torch


class ResNetSegmentationModel(torch.nn.Module):

    def __init__(self, num_classes, use_intermediate_features=False):
        # TODO: Implement the Model initialization

        raise NotImplementedError("ResNetSegmentationModel.__init__ has not been implemented yet.")

    def forward(self, x):
        # TODO: Implement the forward pass of the model as described in the assignment

        raise NotImplementedError("ResNetSegmentationModel.forward has not been implemented yet.")
