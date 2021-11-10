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
        # self.unets = UNet(self.config, in_channel=1, out_channel=16)
        self.prev_unets = UNet(self.config, in_channel=1, out_channel=8)
        self.first_unets = UNet(self.config, in_channel=1, out_channel=8)
        self.next_unets = UNet(self.config, in_channel=1, out_channel=8)
        self.outputs = nn.ModuleList()
        # Set unets for previous frames
        self.unets = nn.ModuleList()
        for in_channel in [24, 2, 2, 2]:
            self.unets.append(UNet(self.config, in_channel=in_channel, out_channel=16))
            # Final output
            self.outputs.append(nn.Sequential(
                nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=1, stride=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=5, padding=1, stride=1),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x: Tensor, y) -> Tensor:
        # input_1, input_2, input_3, input_4, input_5 = x
        outputs = []
        # output = torch.zeros_like(x[0])
        previous_output = self.prev_unets(x[0][:, [0], :, :])
        current_output = self.first_unets(x[0][:, [1], :, :])
        next_output = self.next_unets(x[0][:, [2], :, :])
        for idx, (unet_model, output_model) in enumerate(zip(self.unets, self.outputs)):
            if idx == 0:
                inputs = torch.cat([previous_output, current_output, next_output], dim=1)
            else:
                inputs = torch.cat([output, x[idx][:, [1], :, :]], dim=1)
            output = unet_model(inputs)
            output = output_model(output)
            outputs.append(output)
        return outputs


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    # get model parameters
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # [32, 63, 125, 249]
    # images = []
    # for image_size in [32, 64, 128, 256]:
    #     image = torch.rand((64, 3, image_size, image_size))
    #     images.append(image)
    # output = model(images, images)
    # expected_shape = (64, 16, 16, 16)
    # print(output.shape)
    # assert output.shape == expected_shape, 'Shape did not match.\n\toutput shape is: ' + str(output.shape) + \
    #                                        '\n\texpected shape is: ' + str(expected_shape)

    return


if __name__ == '__main__':
    test()
    import sys

    sys.exit(0)
