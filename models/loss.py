import torch


def get_loss_fn():
    def wrapper(outputs, targets):  # shape: 64, 30, 6
        return torch.nn.MSELoss(reduction='mean')(outputs, targets)
    return wrapper
