import json
import os
import shutil

import torch
import numpy as np

from simulation.noise import get_noise


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_checkpoint(checkpoint_dir, model, config, optimizer=None) -> float:
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint_dir: (string) Directory where checkpoint file is located
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint_path = os.path.join(checkpoint_dir, "last.pth.tar")
    if not os.path.exists(checkpoint_path):
        config.logger.info(f"Checkpoint file doesn't exists {checkpoint_path}")
        return float('-inf')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    # Get accuracy
    with open(os.path.join(checkpoint_dir, "metrics_val_best_weights.json")) as f:
        best_accuracy = json.load(f)["accuracy"]
    return best_accuracy


def convert_device(tensors, device):
    if not tensors:
        return tensors
    elif isinstance(tensors, (int, float)):
        return torch.tensor(tensors, device=device)
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (tuple, list)):
        return [convert_device(tensor, device) for tensor in tensors]


def generate_image_from_points(frame_gts, config):
    # Remove all the gts that have negative value
    bs = frame_gts.size(0)
    movie = get_noise(config.noise_type, (bs, config.image_size, config.image_size), config.bg_model).to(config.device)
    for single_image_id in range(bs):
        # Remove all the non negative value from the frame
        idx = (frame_gts[single_image_id] >= 0).all(-1)
        single_frame_gt = frame_gts[single_image_id][idx]
        single_frame_gt[:, 2] *= 294776
        # The initial frame value will only have the noise
        n_photons = int(single_frame_gt.sum(0)[2])
        photon_pos_frame = torch.zeros((int(n_photons), 2), device=config.device)
        start = 0
        for i in range(single_frame_gt.shape[0]):
            photons = int(single_frame_gt[i, 2])
            mu = single_frame_gt[i, [0, 1]]
            cov = torch.tensor([[config.Imager_PSF * config.Imager_PSF, 0],
                            [0, config.Imager_PSF * config.Imager_PSF]])
            multivariate_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov).to(config.device)
            samples = multivariate_dist.sample((photons, ))
            photon_pos_frame[start: start+photons, :] = samples
            start += photons

        photon_pos_frame = photon_pos_frame.numpy()
        frame_without_noise, _, _ = np.histogram2d(photon_pos_frame[:, 1], photon_pos_frame[:, 0], bins=(range(config.image_size + 1), range(config.image_size + 1)))
        frame_without_noise = torch.from_numpy(frame_without_noise).to(config.device)
        movie[single_image_id] += frame_without_noise
        movie[single_image_id] = movie[single_image_id] / movie[single_image_id].norm()
    # Normalize
    movie = movie.view(bs, 1, config.image_size, config.image_size)
    movie.requires_grad = True
    return movie