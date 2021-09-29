import torch
from torch import Tensor
import torch.nn as nn
from torch import distributions as D
from geomloss import SamplesLoss
from torch.nn import KLDivLoss
from torchvision import models
import matplotlib.pyplot as plt
from .metrics import normalized_cross_correlation

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
        self.criterion = SamplesLoss()

    def forward(self, outputs, targets):
        # creare a image from the output
        # output_image = generate_image_from_points(outputs, self.config)
        # Create a gaussian mixture model from target
        # formatted_target = targets[:, :, [0, 1]].permute(1, 0, 2)
        # return self._kl_dv_loss(outputs, targets)
        # return self._cross_coreation_loss(outputs, targets)
        # return self._gmm_loss(outputs, targets)
        return self._mse_loss(outputs, targets)
        # return nn.MSELoss(reduction='mean')(outputs, targets)

    def _cross_entropy(self, outputs, targets):
        targets = targets * 500000.
        targets = targets.squeeze(1).long()
        return nn.CrossEntropyLoss()(outputs, targets)

    def _L1L2Loss(self, outputs, targets):
        # TODO: fix the shape during making the target
        # targets = targets.view(outputs.shape)
        return nn.MSELoss(reduction='mean')(outputs, targets)

    def _mse_loss(self, outputs, targets):
        return nn.MSELoss(reduction='mean')(outputs, targets)

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