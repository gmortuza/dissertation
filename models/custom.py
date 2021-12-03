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
        self.model_1 = nn.Sequential(
            UNet(self.config, in_channel=2, out_channel=16),
            nn.ConvTranspose2d(16, 8, kernel_size=8, stride=2),
            nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1, bias=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1, bias=True),
            # nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            UNet(self.config, in_channel=2, out_channel=16),
            nn.ConvTranspose2d(16, 8, kernel_size=8, stride=2),
            nn.Conv2d(8, 4, kernel_size=5, padding=0, stride=1, bias=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=0, stride=1, bias=True),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, y) -> Tensor:
        output = torch.zeros_like(x[0])
        outputs = []
        for idx in range(2):
            inputs = torch.cat([x[idx], output], dim=1)
            output = self.model_1(inputs)
            outputs.append(output)

        output = output * 100.

        for idx in range(2, 4):
            inputs = torch.cat([x[idx], output], dim=1)
            output = self.model_2(inputs)
            outputs.append(output)
        return outputs


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    # [32, 63, 125, 249]
    images = []
    # a = [32, 63, 125, 249, 497]
    a = [32, 64, 128, 256, 512]
    for image_size in a:
        image = torch.rand((8, 1, image_size, image_size))
        images.append(image)
    outputs = model(images, images)
    expected_shape = (64, 16, 16, 16)
    for output in outputs[0]:
        print(output.shape)
    # print(output.shape)
    # assert output.shape == expected_shape, 'Shape did not match.\n\toutput shape is: ' + str(output.shape) + \
    #                                        '\n\texpected shape is: ' + str(expected_shape)
    return


if __name__ == '__main__':
    test()
    import sys

    sys.exit(0)
