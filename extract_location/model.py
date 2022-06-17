from read_config import Config
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from torchinfo import summary


class ExtractLocationModel(nn.Module):
    def __init__(self, config):
        super(ExtractLocationModel, self).__init__()
        self.model = models.resnet18(pretrained=False, progress=True)
        # make resnet 3 channel to one channel
        self.model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.model.fc = nn.Linear(512, 12, bias=True)

    def forward(self, x: Tensor, y: Tensor = None, epochs=None) -> Tensor:
        x = self.model(x)
        return x


def test():
    config_ = Config('../config.yaml')
    # sample input data
    input_shape = (32, 1 ,40 ,40)
    expected_output_shape = (32, 12)
    inputs = torch.randn(input_shape)
    model = ExtractLocationModel(config_)
    print(f"Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(summary(model, input_shape))
    # outputs = model(inputs)
    # assert expected_output_shape == outputs.shape


if __name__ == '__main__':
    test()
