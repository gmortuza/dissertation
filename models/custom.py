import torch
from torch import Tensor
import torch.nn as nn
from read_config import Config
from models.unet import UNet


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
        self.unet = UNet(self.config, in_channel=1, out_channel=16)
        self.intensity_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=2),
            nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1),
            nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, y) -> Tensor:
        # input_1, input_2, input_3, input_4, input_5 = x
        outputs = []
        output = torch.zeros_like(x[0])
        for inputs in x[:-1]:
            output = self.unet(inputs+output)
            output = self.intensity_conv(output)
            outputs.append(output)
        return outputs


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    # [32, 63, 125, 249]
    images = []
    for image_size in [32, 63, 125, 249]:
        image = torch.rand((64, 1, image_size, image_size))
        images.append(image)
    output = model(images)
    expected_shape = (64, 16, 16, 16)
    print(output.shape)
    # assert output.shape == expected_shape, 'Shape did not match.\n\toutput shape is: ' + str(output.shape) + \
    #                                        '\n\texpected shape is: ' + str(expected_shape)
    return


if __name__ == '__main__':
    test()
    import sys

    sys.exit(0)
