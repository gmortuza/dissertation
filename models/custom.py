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
        self.intensity_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=2),
            nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1),
            nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True)
        )
        self.position_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=2),
            nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1),
            nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, y) -> Tensor:
        input_1, input_2, input_3, input_4, input_5 = x
        output_1 = self.unet(input_1)  # same size
        output_1_intensity = self.intensity_conv(output_1)
        output_1_pos = self.position_conv(output_1)
        #
        input_2 = input_2 + output_1_intensity
        output_2 = self.unet(input_2)
        output_2_intensity = self.intensity_conv(output_2)
        output_2_pos = self.position_conv(output_2)
        #
        input_3 = input_3 + output_2_intensity
        output_3 = self.unet(input_3)
        output_3_intensity = self.intensity_conv(output_3)
        output_3_pos = self.position_conv(output_3)
        #
        input_4 = input_4 + output_3_intensity
        output_4 = self.unet(input_4)
        output_4_intensity = self.intensity_conv(output_4)
        output_4_pos = self.position_conv(output_4)

        # final = self.final_unet(y[-2])

        return [output_1_intensity, output_2_intensity, output_3_intensity, output_4_intensity], \
               [output_1_pos, output_2_pos, output_3_pos, output_4_pos]

        # return output_1, output_2, output_3, output_4


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
