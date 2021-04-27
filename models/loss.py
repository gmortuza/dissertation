import torch
import torch.nn as nn

from utils import generate_image_from_points


class dNamNNLoss(nn.Module):
    def __init__(self, config):
        super(dNamNNLoss, self).__init__()
        self.config = config

    def forward(self, outputs, targets):
        # creare a image from the output
        output_image = generate_image_from_points(outputs, self.config)
        return nn.MSELoss(reduction='mean')(output_image, targets)

# def get_loss_fn():
#     def wrapper(outputs, targets):  # shape: bs, 30, 6
#         return torch.nn.MSELoss(reduction='mean')(outputs, targets)
#     return wrapper
