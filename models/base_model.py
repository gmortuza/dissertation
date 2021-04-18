import torch
import torch.nn as nn


class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def single_block(self, input_channel, output_channel, kernel_size, stride):
        return nn.Sequential([
            nn.Conv2d(input_channel, output_channel, kernel_size, stride)
        ])

    def forward(self, x):
        pass


if __name__ == '__main__':
    pass