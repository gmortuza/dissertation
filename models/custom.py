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
        self.final_unet = UNet(self.config, in_channel=1, out_channel=1)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=2),
            nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1),
            nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, y) -> Tensor:
        input_1, input_2, input_3, input_4, input_5 = x
        output_0 = self.unet(input_1)  # same size
        output_1 = self.conv(output_0)
        #
        input_2 = input_2 + output_1
        output_2 = self.unet(input_2)
        output_2 = self.conv(output_2)
        #
        input_3 = input_3 + output_2
        output_3 = self.unet(input_3)
        output_3 = self.conv(output_3)
        #
        input_4 = input_4 + output_3
        output_4 = self.unet(input_4)
        output_4 = self.conv(output_4)
        # return output_1, output_2, output_3, output_4
        output_5 = self.final_unet(output_4)
        # final_output = self.conv_transposed(final_output)
        # final_output = self.conv(final_output)
        return output_1, output_2, output_3, output_4, output_5


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
