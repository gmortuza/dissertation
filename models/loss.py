import torch
from torch import Tensor
import torch.nn as nn
from torch import distributions as D
from geomloss import SamplesLoss
from torch.nn import KLDivLoss
from torchvision import models
import matplotlib.pyplot as plt
# from .metrics import normalized_cross_correlation

from utils import generate_image_from_points


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True, num_classes=1000).features[:36].eval().to('cuda:0')
        # Modify so that vgg takes single channel
        vgg = list(vgg.children())
        weight = vgg[0].weight
        vgg[0] = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        vgg[0].weight = nn.Parameter(torch.mean(weight, dim=1, keepdim=True))
        self.vgg = nn.Sequential(*vgg)
        for params in self.vgg.parameters():
            params.requires_grad = False

    def forward(self, output_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        output_features = self.vgg(output_tensor)
        target_features = self.vgg(target_tensor)
        return nn.MSELoss()(output_features, target_features)


class dNamNNLoss(nn.Module):
    def __init__(self, config):
        super(dNamNNLoss, self).__init__()
        self.config = config
        # self.criterion = SamplesLoss()

    def forward(self, outputs, targets):
        # output has two part,
        #   localized prediction
        #   counter
        # output_1, output_2, output_3, output_4, output_5 = outputs
        # target_1, target_2, target_3, target_4, target_5, target_6 = targets
        # return nn.L1Loss()(output_5, target_6)
        self.config.log_param("criterion", "L1")
        return sum(nn.L1Loss()(outputs[i], targets[i]) for i in range(len(outputs)))
        # return nn.L1Loss()(output_1, target_2) + nn.L1Loss()(output_2, target_3) + nn.L1Loss()(output_3, target_4) \
        #        + nn.L1Loss()(output_4, target_5) + nn.L1Loss()(output_5, target_6)
        # return self._mse_loss(output_1, target_5)
               # + self._mse_loss(output_2, target_3) + self._mse_loss(output_4, target_4)

        # predicted_intensity, predicted_location, threshold = outputs
        # targets_zero_to_one = targets.clone()
        # targets_zero_to_one[targets_zero_to_one == 0] = 1.
        # threshold_label, _ = targets_zero_to_one.view(targets_zero_to_one.shape[0], -1).min(dim=1)
        # threshold_label = threshold_label.unsqueeze(1)
        # targets_locations = targets.clone()
        # targets_locations[targets_locations > 0.] = 1.0
        #
        # return self._mse_loss(predicted_intensity, targets) + nn.BCEWithLogitsLoss()(predicted_location, targets_locations)
        # return nn.BCEWithLogitsLoss()(predicted_location, targets_locations)

    def __str__(self):
        return "L1"

    def _cross_entropy(self, outputs, targets):
        targets = targets * 500000.
        targets = targets.squeeze(1).long()
        return nn.CrossEntropyLoss()(outputs, targets)

    def _mse_loss(self, outputs, targets):
        return nn.MSELoss(reduction='mean')(outputs, targets) + nn.L1Loss()(outputs, targets)

    def _kl_dv_loss(self, outputs, targets):
        return KLDivLoss()(outputs.log_softmax(0), targets.softmax(0))

    def _gmm_loss(self, outputs, targets):
        output_gmm = self._create_gaussian_mixture(outputs)
        target_gmm = self._create_gaussian_mixture(targets)
        loss = output_gmm.log_prob(target_gmm.sample((10000, ))).mean()
        return loss

    def _create_gaussian_mixture(self, source):
        # Construct a batch of 3 Gaussian Mixture Models in 2D each
        # consisting of 5 random weighted bivariate normal distributions
        # >>> mix = D.Categorical(torch.rand(3, 5))
        # >>> comp = D.Independent(D.Normal(torch.randn(3, 5, 2), torch.rand(3, 5, 2)), 1)
        # >>> gmm = MixtureSameFamily(mix, comp)
        # TODO: Add noise
        # pos, photon, std, noise = source[:, :, [0, 1]], source[:, :, 2], source[:, :, [3, 4]], source[:, :, 5]
        pos, photon, std = source[:, :, [0, 1]], source[:, :, 2], source[:, :, [3, 4]]

        mix = D.Categorical(probs=photon)
        comp = D.Independent(D.Normal(pos, std), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm

# def get_loss_fn():
#     def wrapper(outputs, targets):  # shape: bs, 30, 6
#         return torch.nn.MSELoss(reduction='mean')(outputs, targets)
#     return wrapper
