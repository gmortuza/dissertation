import torch
from torch import Tensor
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np
from extract_location import point_extractor
import torch.nn as nn


def get_r2_score(prediction, target):
    r2 = 1 - torch.sum((target - prediction) ** 2) / torch.sum((target - target.float().mean()) ** 2)
    return float(r2)


def get_mse(prediction, target):
    mse = torch.sum((target - prediction) ** 2) / torch.numel(target)
    return float(mse)  # MSE is a tensor object


# Source: https://github.com/yuta-hi/pytorch_similarity
def cross_correlation(level):
    def normalized_cross_correlation(predictions, targets, reduction='mean', eps=1e-8):
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
        target = targets[level]
        prediction = predictions[level]

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

        dev_xy = torch.mul(x, y)
        dev_xx = torch.mul(x, x)
        dev_yy = torch.mul(y, y)

        dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
        dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

        ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                        torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
        ncc_map = ncc.view(b, *shape[1:])

        # reduce
        if reduction == 'mean':
            ncc = torch.mean(torch.sum(ncc, dim=1))
        elif reduction == 'sum':
            ncc = torch.sum(ncc)
        else:
            raise KeyError('unsupported reduction type: %s' % reduction)

        return float(ncc.detach().cpu()) * 100

    return normalized_cross_correlation


def get_jaccard_index(config):
    def jaccard_index(prediction, target):
        prediction = prediction[-1].detach().squeeze(1)
        target = target[-1].squeeze(1)
        SMOOTH = 1e-6
        pred_loc = torch.where(prediction > config.detection_threshold)
        pred_loc = set(zip(map(int, pred_loc[0]), map(int, pred_loc[1]), map(int, pred_loc[2])))
        target_loc = torch.where(target > 0.)
        target_loc = set(zip(map(int, target_loc[0]), map(int, target_loc[1]), map(int, target_loc[2])))
        intersection = len(pred_loc.intersection(target_loc))
        union = len(pred_loc.union(target_loc))
        iou = (intersection + SMOOTH) / (union + SMOOTH)
        # intersection = (p_location & target).sum()
        # union = (p_location | target).sum()
        # iou = (intersection + SMOOTH) / (union + SMOOTH)  # smoothed to avoid 0/0
        return iou

    return jaccard_index


def get_psnr(level):
    def psnr(predictions, targets):
        mse = torch.mean((predictions[level] - targets[level]) ** 2)
        max_number = torch.max(targets[level])
        return 20 * torch.log10(max_number / torch.sqrt(mse)).detach().cpu()

    return psnr


def get_SSIM(prediction, target):
    pass


def get_ji_rmse_efficiency_from_predictions(level, config):
    def ji_rmse_efficiency(predictions, targets):
        predictions = predictions[level]
        targets = targets[-1].cpu().numpy()
        frames = targets[:, 0, 0]
        predicted_points = []
        gt_points = []
        for frame_number, frame in zip(frames, predictions):
            frame_target = targets[targets[:, 0, 0] == frame_number][0]
            frame_target = frame_target[frame_target[:, 0] == frame_number]
            predicted_point = point_extractor.get_points(frame, config, frame_number, method='weighted_mean')
            gt_point = point_extractor.get_points_from_gt(frame_target, config)
            if len(predicted_point):
                predicted_points.extend(predicted_point)
            if len(gt_point):
                gt_points.extend(gt_point)
        jaccard_index, rmse, efficiency = get_ji_rmse_efficiency_from_formatted_points(torch.tensor(predicted_points), torch.tensor(gt_points))
        return jaccard_index, rmse, efficiency

    return ji_rmse_efficiency


def get_ji_rmse_nn(config, predictions, targets):
    predictions[:, [0, 6]] = nn.Sigmoid()(predictions[:, [0, 6]])
    formatted_predictions = point_extractor.get_formatted_points(predictions, config)
    formatted_targets = point_extractor.get_formatted_points(targets, config)

    ji, rmse, efficiency = get_ji_rmse_efficiency_from_formatted_points(formatted_predictions, formatted_targets)
    return ji, rmse, efficiency


def get_ji_rmse_efficiency_from_formatted_points(predicted_points: Tensor, gt_points: Tensor, radius=10):

    if not len(predicted_points) or not len(gt_points):
        return 0., 0., 0.
    # predicted_points = torch.Tensor(predicted_points)
    # gt_points = torch.Tensor(gt_points)
    true_positive = 0
    distances_from_points = []
    for frame_number in torch.unique(gt_points[:, 0]):
        f_predicted_points = predicted_points[predicted_points[:, 0] == frame_number][:, [1, 2]]
        f_gt_points = gt_points[gt_points[:, 0] == frame_number][:, [1, 2]]
        # Get pairwise distance
        if len(f_predicted_points) and len(f_gt_points):
            distances = pairwise_distances(f_predicted_points.cpu().detach(), f_gt_points.cpu())
            rec_ind, gt_ind = linear_sum_assignment(distances)
            assigned_distance = distances[rec_ind, gt_ind]
            true_positive += np.sum(assigned_distance <= radius)
            # Calculate the RMSE
            distances_from_points.extend(assigned_distance[assigned_distance <= radius].tolist())
            # distances_from_points = np.append(distances_from_points, assigned_distance)
    rmse = 0
    if len(distances_from_points) > 0:
        distances_from_points = np.asarray(distances_from_points)
        rmse = np.sqrt(np.sum(distances_from_points ** 2) / len(distances_from_points))
    ji = true_positive * 100 / (len(predicted_points) + len(gt_points) - true_positive)
    efficiency = get_efficiency(ji, rmse)
    return ji, rmse, efficiency


def get_efficiency(jaccard_index, rmse, alpha=1.0):
    # https://www.nature.com/articles/s41592-019-0364-4/
    return 100 - ((100 - jaccard_index) ** 2 + (alpha ** 2 * rmse ** 2)) ** .5
    # return (100 - ((100 * (100 - jaccard_index)) ** 2 + alpha ** 2 * rmse ** 2) ** 0.5) / 100


def metrics_for_points_extraction(config):
    def metrics(predictions, targets):
        ji, rmse, efficiency = get_ji_rmse_nn(config, predictions, targets)
        return {
            'JI_16': ji,
            'RMSE': rmse,
            'Efficiency': efficiency
        }
    return metrics


def metrics_for_image_superresolution(config, epoch):
    def metrics(predictions, targets):
        return_metrics =  {
            # 'cc_2': cross_correlation(0)(predictions, targets),
            'cc_4': cross_correlation(0)(predictions, targets),
            # 'cc_8': cross_correlation(2)(predictions, targets),
            'cc_16': cross_correlation(1)(predictions, targets),
        }
        if epoch >= config.JI_metrics_from_epoch:
            ji, rmse, efficiency = get_ji_rmse_efficiency_from_predictions(1, config)(predictions, targets)
            return_metrics['JI_16'] = ji
            return_metrics['rmse_16'] = rmse
            return_metrics['efficiency_16'] = efficiency

        return return_metrics
    return metrics


def get_metrics(config, epoch, points=False):
    if points:
        return metrics_for_points_extraction(config)
    return metrics_for_image_superresolution(config, epoch)
