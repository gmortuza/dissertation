import torch
from torch import Tensor
import torch.nn as nn
from read_config import Config
from models.unet import UNet
# from models.multi_res_unet import MultiResUNet as UNet


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
        self.model_1 = nn.Sequential(
            UNet(self.config, in_channel=1, out_channel=128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(8, 1, 9, 1, 4)
        )
        self.model_2 = nn.Sequential(
            UNet(self.config, in_channel=2, out_channel=128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(8, 1, 9, 1, 4)
        )
        # self.model_3 = nn.Sequential(
        #     UNet(self.config, in_channel=2, out_channel=64),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.PixelShuffle(2),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 1, 9, 1, 4)
        # )
        # self.model_4 = nn.Sequential(
        #     UNet(self.config, in_channel=2, out_channel=64),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.PixelShuffle(2),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 1, 9, 1, 4)
        # )

    # for validations we will use the previous output rather than the labels so we put default epochs value higher
    def forward(self, x: Tensor, y, epochs=100) -> Tensor:
        outputs = []
        # resolution 32 --> 128
        output = self.model_1(x[0])
        outputs.append(output)

        # resolution 128 --> 256
        inputs = torch.cat((x[1], output), dim=1)
        output = self.model_2(inputs)
        outputs.append(output)

        # # resolution 256 --> 512
        # inputs = torch.cat((x[2], output), dim=1)
        # output = self.model_3(inputs)
        # outputs.append(output)
        #
        # #
        # inputs = torch.cat((x[3], output), dim=1)
        # output = self.model_4(inputs)
        # outputs.append(output)

        return outputs


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    # [32, 63, 125, 249]
    images = []
    # a = [32, 63, 125, 249, 497]
    a = [32, 128, 512]
    for image_size in a:
        image = torch.rand((8, 1, image_size, image_size))
        images.append(image)
    outputs = model(images, images)
    expected_shape = (64, 16, 16, 16)
    for output in outputs:
        print(output.shape)
    # print(output.shape)
    # assert output.shape == expected_shape, 'Shape did not match.\n\toutput shape is: ' + str(output.shape) + \
    #                                        '\n\texpected shape is: ' + str(expected_shape)
    return


if __name__ == '__main__':
    test()
    import sys

    sys.exit(0)
