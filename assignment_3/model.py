from torchvision.models import resnet18, ResNet18_Weights
import torch


class ResNetSegmentationModel(torch.nn.Module):

    def __init__(self, num_classes=15, use_intermediate_features=False):
        super(ResNetSegmentationModel, self).__init__()
        self.use_intermediate_features = use_intermediate_features
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Change stride parameters of last conv. blocks
        self.backbone.layer3[0].conv1.stride = (1, 1)
        self.backbone.layer3[0].downsample[0].stride = (1, 1)
        self.backbone.layer4[0].conv1.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

        # Change dilation rates accordingly (#ResearchCode)
        for i in range(1, len(self.backbone.layer3)):
            cur_block = self.backbone.layer3[i]
            cur_block.conv2.dilation = (2, 2)
            cur_block.conv2.padding = (2, 2)

        for i in range(1, len(self.backbone.layer4)):
            cur_block = self.backbone.layer4[i]
            cur_block.conv2.dilation = (4, 4)
            cur_block.conv2.padding = (4, 4)

        if self.use_intermediate_features:
            decoder_in_channels = 640
        else:
            decoder_in_channels = 512

        self.classifier = torch.nn.Conv2d(in_channels=decoder_in_channels,
                                          out_channels=num_classes,
                                          kernel_size=(1, 1))

    def forward(self, x, use_dropout_perturbation: bool = False):
        x_2 = x = self.backbone.conv1(x)
        x = self.backbone.relu(self.backbone.bn1(x))
        x = self.backbone.maxpool(x)

        if use_dropout_perturbation:
            raise NotImplementedError(
                f'The use of dropout in {self.forward.__name__} has not been implemented yet.')

            # TODO: Use dropout after applying layers 1-4.
            #       You are allowed to use torch.nn.functional to do this.
            #       Use a dropout rate of 20%.

        else:
            x_4 = x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

        if self.use_intermediate_features:
            x = torch.nn.functional.interpolate(
                x, scale_factor=4, mode='bilinear')
            x_4 = torch.nn.functional.interpolate(
                x_4, scale_factor=2, mode='bilinear')
            x = torch.cat([x_2, x_4, x], 1)
            x = self.classifier(x)
            x = torch.nn.functional.interpolate(
                x, scale_factor=2, mode='bilinear')
        else:
            x = self.classifier(x)
            x = torch.nn.functional.interpolate(
                x, scale_factor=8, mode='bilinear')

        return x
