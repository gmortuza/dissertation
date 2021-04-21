import json
import os
import shutil

import torch


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
