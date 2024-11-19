from torchvision.models import resnet18, ResNet18_Weights
import torch


class ResNetSegmentationModel(torch.nn.Module):

    def __init__(self, num_classes, use_intermediate_features=False):
        super(ResNetSegmentationModel, self).__init__()
        self._use_intermediate_features = use_intermediate_features

        if use_intermediate_features:
            self._upsample_times2 = torch.nn.Upsample(scale_factor=2, mode="bilinear")
            self._upsample_times4 = torch.nn.Upsample(scale_factor=4, mode="bilinear")

        # get a pretrained ResNet18 model
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # decrease the stride of the last conv layers to retain a higher resolution
        self.backbone.layer3[0].conv1.stride = (1, 1)
        self.backbone.layer4[0].conv1.stride = (1, 1)

        # adapt the skip connections accordingly
        self.backbone.layer3[0].downsample[0].stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

        # construct a segmentation model using the resnet18 backbone and a simple upsampling layer
        self.model_segmentation = torch.nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            torch.nn.Conv2d(
                self.backbone.layer4[1].conv2.out_channels,
                num_classes,
                (1, 1)),
            torch.nn.Upsample(
                scale_factor=8,
                mode="bilinear"))
    # TODO: remove prints
    def forward(self, x):
        # use early-layer features for upsampling if flag is set
        if self._use_intermediate_features:
            print(x.shape)

            x = self.model_segmentation[0](x) # Conv2d - has stride=(2,2) -> first downsampling to 1/2 of original
            print("downsample:",x.shape)
            x = self.model_segmentation[1](x) # BatchNorm2d
            print(x.shape)
            x = self.model_segmentation[2](x) # ReLU
            print(x.shape)

            # store most expressive features that have 1/2 of original resolution and detach their gradients
            x_half = x
            x_half.detach()

            x = self.model_segmentation[3](x) # MaxPool2d - has stride=(2,2) -> second downsampling to 1/4 of original
            print("downsample:",x.shape)
            x = self.model_segmentation[4](x) # resnet layer1
            print(x.shape)

            # store most expressive features that have 1/4 of original resolution and detach their gradients
            x_quarter = x
            x_quarter.detach()

            x = self.model_segmentation[5](x) # resnet layer2 - contains Conv2d with stride=(2,2) -> third downsampling to 1/8 of original
            print("downsample:",x.shape)
            x = self.model_segmentation[6](x) # resnet layer3
            print(x.shape)
            x = self.model_segmentation[7](x) # resnet layer4
            print(x.shape)

            # upsample current features and x_quarter features such that they are half the original resolution
            x_times4 = self._upsample_times4(x)
            x_quarter_times2 = self._upsample_times2(x_quarter)

            # concatenate the current features with those from previous layers at half the original resolution
            x = torch.stack((x_times4, x_quarter_times2, x_half))


            """
            # Conv2d
            x = self.model_segmentation[8](x)
            print(x.shape)

            # Upsample
            x = self.model_segmentation[9](x)
            print(x.shape)
            breakpoint()
            """
            return x

        # use simple bilinear upscaling otherwise
        else:
            return self.model_segmentation(x)
