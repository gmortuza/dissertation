import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn import functional as F

from read_config import Config


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, config, in_channel=1, out_channel=16, features=[16, 32, 64, 128]):
        super(UNet, self).__init__()
        self.config = config

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pools = nn.MaxPool2d(kernel_size=2, stride=2)

        #
        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature

        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=1, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.output = nn.Sequential(
            nn.Conv2d(features[0], out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pools(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        intensity = self.output(x)

        return intensity


def test():
    model = UNet(Config("../config.yaml"), out_channel=1)
    print(f"Model param {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    x = torch.randn(8, 1, 32, 32)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test()
