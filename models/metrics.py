import torch
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment


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
    # prediction = prediction[0]
    target = target[1]
    prediction = prediction[0]

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


def get_jaccard_index(prediction, target):
    SMOOTH = 1e-6
    p_intensity, p_location, _ = prediction
    p_location = p_location.detach().round().int()
    target = target.int()

    intersection = (p_location & target).sum()
    union = (p_location | target).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # smoothed to avoid 0/0
    return iou.cpu()


def get_ji_by_threshold(prediction, target):
    p_intensity, p_location, threshold = prediction
    p_intensity[p_intensity < threshold.unsqueeze(1).unsqueeze(1)] = 0.
    prediction_points = torch.nonzero(p_intensity)[:, [0, 2, 3]]
    target_points = torch.nonzero(target)[:, [0, 2, 3]]
    total_true_positive = 0
    for i in torch.unique(prediction_points[:, 0]):
        frame_prediction_point = prediction_points[prediction_points[:, 0] == i].tolist()
        frame_prediction_point = set(map(tuple, frame_prediction_point))
        frame_target_point = target_points[target_points[:, 0] == i].tolist()
        frame_target_point = set(map(tuple, frame_target_point))
        total_true_positive += len(frame_target_point.intersection(frame_prediction_point))

    return total_true_positive / (prediction_points.shape[0] + target_points.shape[0] - total_true_positive)


def get_psnr(prediction, target):
    prediction = prediction[0]
    target = target[1]
    mse = torch.mean((prediction - target) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).detach().cpu()

def get_SSIM(prediction, target):
    pass


metrics = {
    # 'accuracy': get_r2_score,
    # 'JI': get_jaccard_index,
    # 'JI_threshold': get_ji_by_threshold,
    'accuracy': normalized_cross_correlation,
    'psnr': get_psnr
    # 'mse': get_mse
}