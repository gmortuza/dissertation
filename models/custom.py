import torch
from torch import Tensor
import torch.nn as nn

from models.edsr import EDSR
from read_config import Config
from models.unet import UNet


class Custom(nn.Module):
    def __init__(self, config):
        super(Custom, self).__init__()
        self.config = config
        self.models = nn.ModuleList()
        # self.remove_noise = nn.Sequential(
        #     UNet(config, in_channel=1, out_channel=16),
        #     nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=True)
        # )
        # self.model = nn.Sequential(
        #     UNet(self.config, in_channel=1, out_channel=16),
        #     nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2),
        #     nn.Conv2d(8, 1, kernel_size=5, padding=1, stride=1),
        #     )
        self.models.append(nn.Sequential(
            UNet(config, in_channel=1, out_channel=16),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
        ))
        for _ in range(3):
            self.models.append(nn.Sequential(
                UNet(config, in_channel=2, out_channel=16),
                nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            ))

    def forward(self, x: Tensor, y) -> Tensor:
        # output = torch.zeros_like(x[0])
        outputs = []
        for idx, model in enumerate(self.models):
            if idx == 0:
                inputs = x[0]
            else:
                inputs = torch.cat([output, x[idx]], dim=1)
            inputs = inputs - inputs.view(inputs.shape[0], inputs.shape[1], -1).mean(2).unsqueeze(-1).unsqueeze(-1)
            # Mean shift of the inputs
            output = model(inputs)
            # Add mean to the output
            output = output + output.view(output.shape[0], output.shape[1], -1).mean(2).unsqueeze(-1).unsqueeze(-1)
            outputs.append(output)
        return outputs


def test():
    config_ = Config('../config.yaml')
    model = Custom(config_)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # [32, 63, 125, 249]
    images = []
    for image_size in [32, 64, 128, 256]:
        image = torch.rand((16, 1, image_size, image_size))
        images.append(image)
    outputs = model(images, images)
    for output in outputs:
        print(output.shape)
    # assert output.shape == expected_shape, 'Shape did not match.\n\toutput shape is: ' + str(output.shape) + \
    #                                        '\n\texpected shape is: ' + str(expected_shape)
    return


if __name__ == '__main__':
    test()
    import sys

    sys.exit(0)
