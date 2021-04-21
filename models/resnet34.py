import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F


class ResNet34(torch.nn.Module):
    def __init__(self, config):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=False)
        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding,
            dilation=conv1.dilation, groups=conv1.groups, bias=conv1.bias
        )
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.x = nn.Linear(num_ftrs, config.max_number_of_emitter_per_frame)
        self.y = nn.Linear(num_ftrs, config.max_number_of_emitter_per_frame)
        self.photons = nn.Linear(num_ftrs, config.max_number_of_emitter_per_frame)
        self.s_x = nn.Linear(num_ftrs, config.max_number_of_emitter_per_frame)
        self.s_y = nn.Linear(num_ftrs, config.max_number_of_emitter_per_frame)
        self.noise = nn.Linear(num_ftrs, config.max_number_of_emitter_per_frame)

    def forward(self, input_item):
        input_item = self.model(input_item)

        x = self.x(input_item)
        y = self.y(input_item)
        # photons = self.photons(input_item)
        # s_x = self.s_x(input_item)
        # s_y = self.s_y(input_item)
        # noise = self.noise(input_item)

        # x_mean, y_mean, photons, s_x, s_y, noise
        # output_target = torch.stack([x, y, photons, s_x, s_y, noise], dim=2)
        output_target = torch.stack([x, y], dim=2)

        return output_target




if __name__ == '__main__':
    model = ResNet34()
    print(model.model.feature)
