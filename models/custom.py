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
        self.model = EDSR()

    def forward(self, x: Tensor, y) -> Tensor:
        output = torch.zeros_like(x[0])
        outputs = []
        for idx in range(4):
            # inputs = torch.cat([output, x[idx]], dim=1)
            inputs = output + x[idx]
            # Mean shift of the inputs
            inputs = inputs - inputs.view(inputs.shape[0], -1).mean(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            output = self.model(inputs)
            # Add mean to the output
            output = output + output.view(output.shape[0], -1).mean(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
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