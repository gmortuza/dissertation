import json
import os
import shutil

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2


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


def save_checkpoint(state, best_val_acc, config, val_metrics, name=''):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
    """
    if config.save_model_after_epoch_end == 0 or config.save_model_after_each_epoch % state['epoch'] == 0:
        return float('-inf')
    filepath = os.path.join(config.checkpoint_dir, name+'last.pth.tar')
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    torch.save(state, filepath)
    if val_metrics[config.save_model_based_on] >= best_val_acc:
        config.logger.info("New best accuracy found")
        shutil.copyfile(filepath, os.path.join(config.checkpoint_dir, name+'best.pth.tar'))
        best_json_path = os.path.join(
            config.checkpoint_dir, name + "metrics_val_best_weights.json")
        save_dict_to_json(val_metrics, best_json_path)
    last_json_path = os.path.join(
        config.checkpoint_dir, name + "metrics_val_last_weights.json")
    save_dict_to_json(val_metrics, last_json_path)
    return max(best_val_acc, val_metrics[config.save_model_based_on])


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


def load_checkpoint(model, config, optimizer=None, name='') -> float:
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint_dir: (string) Directory where checkpoint file is located
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if config.load_checkpoint:
        config.logger.info(f"Restoring parameters from {config.checkpoint_dir}")
        checkpoint_path = os.path.join(config.checkpoint_dir, name+"last.pth.tar")
        if not os.path.exists(checkpoint_path):
            config.logger.info(f"Checkpoint file doesn't exists {checkpoint_path}")
            return float('-inf')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optim_dict'])
        # Get accuracy
        with open(os.path.join(config.checkpoint_dir, "metrics_val_best_weights.json")) as f:
            best_accuracy = json.load(f)[config.save_model_based_on]
        return best_accuracy
    return float('-inf')


def create_random_labels_map(classes: int):
    labels_map = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3,))
    labels_map[0] = torch.zeros(3)
    return labels_map


def labels_to_image(img_labels: torch.Tensor, labels_map):
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = (img_labels == label_id)
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out


def show_components(img, labels):
    img = img.numpy()
    color_ids = torch.unique(labels)
    labels_map = create_random_labels_map(color_ids)
    labels_img = labels_to_image(labels, labels_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))

    # Showing Original Image
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("Orginal Image")

    # Showing Image after Component Labeling
    ax2.imshow(labels_img.permute(1, 2, 0).squeeze().numpy())
    ax2.axis('off')
    ax2.set_title("Component Labeling")
    plt.show()


def connected_components(image: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. warning::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       connected_components.html>`__.

    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _ = image_view.shape
    out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out.view_as(image)



def convert_device(tensors, device):
    if tensors is None:
        return None
    elif isinstance(tensors, (int, float)):
        return torch.tensor(tensors, device=device)
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (tuple, list)):
        return [convert_device(tensor, device) for tensor in tensors]