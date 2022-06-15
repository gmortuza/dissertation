import torch
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from read_config import Config
import extract_points
import torch.nn as nn


# config = Config("config.yaml")


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


def get_psnr(level):
    def psnr(predictions, targets):
        mse = torch.mean((predictions[level] - targets[level]) ** 2)
        max_number = torch.max(targets[level])
        return 20 * torch.log10(max_number / torch.sqrt(mse)).detach().cpu()

    return psnr


def get_SSIM(prediction, target):
    pass


def get_ji_by_loc(level):
    def ji(predictions, targets):
        prediction = (predictions[1][level] > 0.5).to(torch.uint8)
        prediction = set(map(tuple, torch.nonzero(prediction)))
        target = set(map(tuple, torch.nonzero(targets[level])))
        return len(prediction.intersection(target)) * 100 / len(prediction.union(target))

    return ji


def get_ji_by_points(level, config):
    def ji(predictions, targets):
        predictions = predictions[level]
        targets = targets[-1].cpu().numpy()
        frames = targets[:, 0, 0]
        predicted_points = []
        gt_points = []
        for frame_number, frame in zip(frames, predictions):
            frame_target = targets[targets[:, 0, 0] == frame_number][0]
            frame_target = frame_target[frame_target[:, 0] == frame_number]
            predicted_point = extract_points.get_points(frame, frame_number, config)
            gt_point = extract_points.get_points_from_gt(frame_target, config)
            if len(predicted_point):
                predicted_points.extend(predicted_point)
            if len(gt_point):
                gt_points.extend(gt_point)
        jaccard_index, rmse = extract_points.get_ji_rmse(predicted_points, gt_points, 5)
        return jaccard_index

    return ji


def get_formatted_points(raw_points, config, start_pos=None):
    """
    The take the output of the neural network and format to measure the accuracy
    [p_c_1, x_1, y_1, s_x_1, s_y_1, photon_1, p_c_2, x_2, y_2, s_x_2, s_y_2, photon_2] -->
    [[frame_number, x_1, y_1, s_x_1, s_y_1, photon_1], [frame_number, x_2, y_2, s_x_2, s_y_2, photon_2]]
    Args:
        raw_points --> Outputs from the neural network
        config --> configuration file
        start_pos --> start position for each of the patch file. if not provided then start pos will be zeros
                -- [frame number, x_start, y_start]

    Returns:
    """
    if start_pos is None:  # it's during the training procedure
        start_position = torch.zeros((raw_points.shape[0], 3), device=raw_points.device)
        # convert the frame number using the batch size
        start_position[:, 0] = torch.as_tensor(torch.arange(0, raw_points.shape[0]), device=raw_points.device)
    else:  # it's during the inference
        start_position = torch.tensor(start_pos, device=raw_points.device)
    # attach frame number and start position of that patch to the output
    raw_points_frame_start = torch.cat((start_position, raw_points), dim=1)
    # convert the start position into nanometer
    # raw_points_frame_start[:, [1, 2]] *= config.Camera_Pixelsize
    raw_points_frame_start[:, [4, 5, 10, 11]] *= config.extracted_patch_size / config.location_multiplier
    raw_points_frame_start[:, [4, 5, 10, 11]] += raw_points_frame_start[:, [1, 2, 1, 2]]
    raw_points_frame_start[:, [4, 5, 10, 11]] *= config.Camera_Pixelsize * config.resolution_slap[0] / config.resolution_slap[-1]

    # extract points for first emitter
    first_emitter_loc = torch.where((raw_points_frame_start[:, 3] > .999))[0]
    first_emitter = raw_points_frame_start[first_emitter_loc, :]
    first_emitter = first_emitter[:, [0, 4, 5, 6, 7, 8]]

    # extract points for second emitters
    # second_emitter_loc = torch.where((raw_points_frame_start[:, 9] > .9) & (raw_points_frame_start[:, 12] > .001))[0]
    second_emitter_loc = torch.where((raw_points_frame_start[:, 9] > .95))[0]
    second_emitter = raw_points_frame_start[second_emitter_loc, :]
    second_emitter = second_emitter[:, [0, 10 ,11, 12, 13, 14]]

    # combine two emitters
    emitters = torch.cat((first_emitter, second_emitter), dim=0)
    return emitters



def get_ji_rmse_nn(config, predictions, targets):
    predictions[:, [0, 6]] = nn.Sigmoid()(predictions[:, [0, 6]])
    formatted_predictions = get_formatted_points(predictions, config)
    formatted_targets = get_formatted_points(targets, config)

    ji, rmse, efficiency = extract_points.get_ji_rmse_efficiency(formatted_predictions, formatted_targets)
    return ji, rmse, efficiency

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
            'cc_4': cross_correlation(0)(predictions, targets),
            'cc_8': cross_correlation(1)(predictions, targets),
            'cc_16': cross_correlation(2)(predictions, targets),
        }
        if epoch >= config.JI_metrics_from_epoch:
            return_metrics['JI_16'] = get_ji_by_points(2, config)(predictions, targets)
    return metrics


def get_metrics(config, epoch, points=False):
    if points:
        return metrics_for_points_extraction(config)
    return metrics_for_image_superresolution(config, epoch)
