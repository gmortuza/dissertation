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
            UNet(self.config, in_channel=1, out_channel=16),
            nn.PixelShuffle(4)
            # nn.ConvTranspose2d(16, 8, kernel_size=8, stride=2),
            # nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1, bias=True),
            # nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1, bias=True),
            # nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            UNet(self.config, in_channel=2, out_channel=4),
            nn.PixelShuffle(2),
            # nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, bias=True),
            # nn.Conv2d(32, 16, kernel_size=5, padding=1, stride=1, bias=True),
            # nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=True),
            # nn.Conv2d(16, 1, kernel_size=5, padding=1, stride=1, bias=True),
            # nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            UNet(self.config, in_channel=2, out_channel=4),
            nn.PixelShuffle(2)
        )

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

        # resolution 256 --> 512
        inputs = torch.cat((x[2], output), dim=1)
        output = self.model_3(inputs)
        outputs.append(output)

        return outputs


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    # [32, 63, 125, 249]
    images = []
    # a = [32, 63, 125, 249, 497]
    a = [32, 64, 128, 512]
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
