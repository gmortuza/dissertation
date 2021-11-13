import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(self, in_channel=128, out_channel=128, residual_scaling=.1):
        super(ResBlock, self).__init__()
        self.residual_scaling = residual_scaling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x) * self.residual_scaling


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()
        # input conv
        self.input_conv = nn.Conv2d(1, 128, 3, 1, 1, bias=False)
        # Residual layer
        self.res_blocks = []
        for _ in range(8):
            self.res_blocks.append(ResBlock())
        self.res_blocks = nn.Sequential(*self.res_blocks)
        # Middle conv
        self.middle_conv = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        # Upsample layers
        self.up_sample = nn.Sequential(
            nn.Conv2d(128, 128 * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )
        # final conv
        self.final_conv = nn.Conv2d(128, 1, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        input_conv = self.input_conv(x)
        res_output = self.res_blocks(input_conv)
        middle_conv = self.middle_conv(res_output) + input_conv
        upsampled = self.up_sample(middle_conv)
        final_conv = self.final_conv(upsampled)
        return nn.ReLU(inplace=True)(final_conv)


def test():
    model = EDSR()
    inputs = torch.randn((32, 1, 16, 16))
    outputs = model(inputs)
    print(outputs.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


if __name__ == '__main__':
    test()
