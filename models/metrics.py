import torch


def get_r2_score(prediction, target):
    r2 = 1 - torch.sum((target - prediction) ** 2) / torch.sum((target - target.float().mean()) ** 2)
    return float(r2)


def get_mse(prediction, target):
    mse = torch.sum((target - prediction) ** 2) / torch.numel(target)
    return float(mse)  # MSE is a tensor object


# Source: https://github.com/yuta-hi/pytorch_similarity
def normalized_cross_correlation(prediction, target, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        prediction (~torch.Tensor): Input tensor.
        target (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = prediction.shape
    b = shape[0]

    # reshape
    x = prediction.view(b, -1)
    y = target.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    return float(ncc.detach().cpu())


metrics = {
    # 'accuracy': get_r2_score,
    'accuracy': normalized_cross_correlation
    # 'mse': get_mse
}