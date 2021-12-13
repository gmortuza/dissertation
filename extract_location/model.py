from read_config import Config
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models


class ExtractLocationModel(nn.Module):
    def __init__(self, config):
        super(ExtractLocationModel, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        conv1 = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride,
            padding=conv1.padding,
            dilation=conv1.dilation, groups=conv1.groups, bias=conv1.bias
        )
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        #
        self.output = nn.Linear(num_ftrs, 12)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.resnet18(x)
        return self.output(x)


def test():
    config_ = Config('../config.yaml')
    # sample input data
    inputs = torch.randn((32, 1, 20, 20))
    model = ExtractLocationModel(config_)
    outputs = model(inputs)
    print(outputs.shape)


if __name__ == '__main__':
    test()
