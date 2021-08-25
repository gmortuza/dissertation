import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn import functional as F


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
    def __init__(self, config, in_channel=1, features=[64, 128, 256, 512]):
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
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.x_pos_pred = nn.Linear(65536, config.max_number_of_emitter_per_frame)
        self.y_pos_pred = nn.Linear(65536, config.max_number_of_emitter_per_frame)
        self.x_std = nn.Linear(65536, config.max_number_of_emitter_per_frame)
        self.y_std = nn.Linear(65536, config.max_number_of_emitter_per_frame)
        self.noise = nn.Linear(65536, config.max_number_of_emitter_per_frame)
        self.photons = nn.Linear(65536, config.max_number_of_emitter_per_frame)

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

        x = torch.flatten(x, 1)
        # Pos will never be zero so using relu to remove convert negetive prediction to zero
        x_pos_pred = nn.ReLU()(self.x_pos_pred(x))
        y_pos_pred = nn.ReLU()(self.y_pos_pred(x))
        photons = nn.Sigmoid()(self.photons(x))
        x_std = nn.Sigmoid()(self.x_std(x))
        y_std = nn.Sigmoid()(self.y_std(x))
        # TODO: Add noise later
        # noise = nn.ReLU()(self.noise(x))
        return torch.stack((x_pos_pred, y_pos_pred, photons, x_std, y_std), dim=2)


if __name__ == '__main__':
    pass
