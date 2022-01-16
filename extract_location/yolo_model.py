import torch
from torch import Tensor
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YoloModel(nn.Module):
    def __init__(self, config, in_channel=1, **kwargs):
        super(YoloModel, self).__init__()
        self.config = config


    def forward(self, x: Tensor) -> Tensor:

        return x


