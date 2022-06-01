import torch.nn as nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.locations_loss = nn.HuberLoss()
        # self.locations_loss = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        # weight for each loss
        self.single_emitter_loss = 0.

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # TODO: Try hubber loss
        # return self.location_loss(predictions, targets)
        # check if there is emitter in that positions
        # all the patch will have at least one emitter.
        # So we will only check if there are any second emitter in the patch
        two_emitters = targets[..., 6] == 1.  # have multiple emitters in the patch
        one_emitters = targets[..., 6] == 0.  # Have single emitter in the patch
        single_emitter_loss = 0
        single_emitter_localization_loss = 0
        multi_emitter_localization_loss = 0.  # emitter location and photon count loss
        multi_emitter_loss = 0.  # if emitter is present loss
        if two_emitters.any():
            # If there are multiple emitters in the patch
            multi_emitter_localization_loss = self.locations_loss(
                predictions[:, [1, 2, 3, 7, 8, 9]][two_emitters], targets[:, [1, 2, 3, 7, 8, 9]][two_emitters]
            )
            # it's binary loss
            multi_emitter_loss = self.bce(
                predictions[:, [0, 6]][two_emitters], targets[:, [0, 6]][two_emitters]
            )
        if one_emitters.any():  # this is a rare case that all training example in a batch have two emitters
            # if there is a single emitter in the patch
            single_emitter_localization_loss = self.locations_loss(
                predictions[:, [1, 2, 3]][one_emitters], targets[:, [1, 2, 3]][one_emitters]
            )
            single_emitter_loss = self.bce(
                predictions[:, [0, 6]][one_emitters], targets[:, [0, 6]][one_emitters]
            )
        # TODO: put weight on different types of loss
        return (
            single_emitter_loss +
            single_emitter_localization_loss +
            10 * multi_emitter_loss +
            10 * multi_emitter_localization_loss
        )
