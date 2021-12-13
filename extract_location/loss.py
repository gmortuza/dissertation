import torch.nn as nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return nn.MSELoss()(outputs, targets)
