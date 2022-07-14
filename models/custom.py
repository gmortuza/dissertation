import torch
from torch import Tensor
import torch.nn as nn
from models.unet import UNet
# from models.multi_res_unet import MultiResUNet as UNet


class Custom(nn.Module):
    def __init__(self, config):
        super(Custom, self).__init__()
        self.config = config
        self.model_1 = nn.Sequential(
            UNet(self.config, in_channel=1, out_channel=128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 1, 9, 1, 4)
        )
        self.model_2 = nn.Sequential(
            UNet(self.config, in_channel=2, out_channel=128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 1, 9, 1, 4)
        )

    def upsample_overlapped_emitter(self, x):
        inputs = torch.cat((x, x), dim=1)
        x_2x = self.model_2(inputs)
        return x_2x

    # for validation, we will use the previous output rather than the labels so we put default epochs value higher
    def forward(self, x: Tensor, y, epochs=100) -> Tensor:
        outputs = []
        # resolution 32 --> 64
        # inputs = torch.cat((x[0], x[0]), dim=1)
        output = self.model_1(x[0])
        outputs.append(output)
        # resolution 64 --> 128
        # inputs = torch.cat((x[1], output), dim=1)
        output = self.model_1(output)
        outputs.append(output)
        # resolution 128 --> 256
        inputs = torch.cat((x[2], output), dim=1)
        output = self.model_2(inputs)
        outputs.append(output)
        # resolution 256 --> 512
        inputs = torch.cat((x[3], output), dim=1)
        output = self.model_2(inputs)
        outputs.append(output)

        return outputs


def test():
    from read_config import Config
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
