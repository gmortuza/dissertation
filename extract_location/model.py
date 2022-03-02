from read_config import Config
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from models.unet import UNet

EXPORTED_TRAIN_WIDTH = 40
EXPORTED_TRAIN_HEIGHT = 40


class ExtractLocationModel(nn.Module):
    def __init__(self, config):
        super(ExtractLocationModel, self).__init__()
        out_channel = 2
        self.unet = UNet(config, in_channel=1, out_channel=out_channel, features=[32, 64, 128, 256])
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * 40 * out_channel, 1024),
            nn.Dropout(.1),
            nn.LeakyReLU(.1),
            nn.Linear(1024, 512),
            nn.Dropout(.1),
            nn.LeakyReLU(.1),
            nn.Linear(512, 12),
        )

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        x = self.unet(x)
        x = self.output(x)
        return x


def test():
    config_ = Config('../config.yaml')
    # sample input data
    inputs = torch.randn((32, 1, 20, 20))
    model = ExtractLocationModel(config_)
    outputs = model(inputs)
    print(outputs.shape)


if __name__ == '__main__':
    test()
