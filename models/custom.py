import torch
from torch import Tensor
import torch.nn as nn
from read_config import Config
from unet import UNet

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(

        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Custom(nn.Module):
    def __init__(self, config):
        super(Custom, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        # for i in range(1, 8, 2):

    def forward(self, x: Tensor) -> Tensor:
        x = UNet(self.config, in_channel=1, out_channel=16)(x)  # same size
        # increase size
        # TODO: increase the size by sub-pixel conv
        x = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)(x)
        # add input image here which is 64 px using bilinear conv


        # increase the resolution

        # x = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=7, padding=2, stride=1),
        #     # nn.Conv2d(16, 1, kernel_size=1)
        # )(x)
        return x


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    image = torch.rand((64, 1, 32, 32))
    output = model(image)
    expected_shape = (64, 16, 16, 16)
    print(output.shape)
    # assert output.shape == expected_shape, 'Shape did not match.\n\toutput shape is: ' + str(output.shape) + \
    #                                        '\n\texpected shape is: ' + str(expected_shape)


if __name__ == '__main__':
    test()
    import sys
    sys.exit(0)