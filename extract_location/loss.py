import torch.nn as nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        # self.hubber = nn.HubberLoss()
        # weight for each loss
        self.single_emitter_loss = 0.

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # return self.mse(predictions, targets)
        # check if there is emitter in that positions
        # all the patch will have at least one emitter.
        # So we will only check if there are any second emitter in the patch
        emitters = targets[..., 6] == 1.
        no_emitters = targets[..., 6] == 0.
        multi_emitter_localization_loss = 0.
        multi_emitter_loss = 0.
        if emitters.any():
            # if the
            multi_emitter_localization_loss = self.mse(
                predictions[:, [1, 2, 7, 8]][emitters], targets[:, [1, 2, 7, 8]][emitters]
            )
            multi_emitter_loss = self.mse(
                predictions[:, [0, 6]][emitters], targets[:, [0, 6]][emitters]
            )

        single_emitter_localization_loss = self.mse(
            predictions[:, [1, 2]][no_emitters], targets[:, [1, 2]][no_emitters]
        )
        single_emitter_loss = self.mse(
            predictions[:, [0, 6]][no_emitters], targets[:, [0, 6]][no_emitters]
        )
        return (
            single_emitter_loss +
            single_emitter_localization_loss +
            multi_emitter_loss +
            multi_emitter_localization_loss
        )
