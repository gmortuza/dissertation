import pickle
import torch

from read_config import Config
from utils import connected_components
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import cv2


def get_ji_rmse(predicted_points, gt_points):
    # predicted_points_set = set([(p[0], p[1], p[2]) for p in predicted_points])
    # gt_points_set = set([(p[0], p[1], p[2]) for p in gt_points])
    # intersection = len(predicted_points_set.intersection(gt_points_set))
    # union = len(predicted_points_set.union(gt_points_set))
    # return intersection / union, 0.
    if not len(predicted_points) or not len(gt_points):
        return 0., 0.
    radius = 10.
    predicted_points = np.asarray(predicted_points)
    gt_points = np.asarray(gt_points)
    true_positive = 0
    distances_from_points = np.asarray([])
    for frame_number in np.unique(gt_points[:, 0]):
        f_predicted_points = predicted_points[predicted_points[:, 0] == frame_number]
        f_gt_points = gt_points[gt_points[:, 0] == frame_number]
        # extract the points
        f_predicted_points = [(p[1], p[2]) for p in f_predicted_points]
        f_gt_points = [(p[1], p[2]) for p in f_gt_points]
        # Get pairwise distance
        if len(f_predicted_points) and len(f_gt_points):
            distances = pairwise_distances(f_predicted_points, f_gt_points)
            rec_ind, gt_ind = linear_sum_assignment(distances)
            assigned_distance = distances[rec_ind, gt_ind]
            true_positive += np.sum(assigned_distance <= radius)
            # Calculate the RMSE
            distances_from_points = np.append(distances_from_points, assigned_distance)
    rmse = np.sqrt(np.sum(distances_from_points ** 2) / len(distances_from_points))
    return true_positive * 100 / (len(predicted_points) + len(gt_points) - true_positive), rmse


def get_efficiency(jaccard_index, rmse, alpha=1.0):
    # https://www.nature.com/articles/s41592-019-0364-4/
    return 100 - ((100 - jaccard_index) ** 2 + (alpha ** 2 * rmse ** 2)) ** .5
    # return (100 - ((100 * (100 - jaccard_index)) ** 2 + alpha ** 2 * rmse ** 2) ** 0.5) / 100


def get_point(frame, labels, label_number, config):
    x, y = torch.where(labels == label_number)
    if len(x) < 10 or len(x) > 70:
        return None
    # if len(x) > 35:
    #     return None
    weights = frame[0][x, y]
    # TODO: do these things using pytorch
    x_mean = torch.sum(x * weights) / torch.sum(weights)
    y_mean = torch.sum(y * weights) / torch.sum(weights)
    # x_mean = np.average(x.float().cpu().numpy(), weights=weights.detach().cpu().numpy())
    # y_mean = np.average(y.float().cpu().numpy(), weights=weights.detach().cpu().numpy())
    image_patch = frame[:, int(x_mean.round()) - 5: int(x_mean.round()) + 5,
                  int(y_mean.round()) - 5: int(y_mean.round()) + 5]
    photon_count = torch.sum(image_patch)
    x_nm = x_mean * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
    y_nm = y_mean * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
    # x, y, s_x, s_y, photons
    return [float(x_nm), float(y_nm), 0, 0, float(photon_count)]


def get_points(frame, frame_number, config):
    ## Some test code
    # px_to_nm = config.Camera_Pixelsize * config.resolution_slap[0] / config.resolution_slap[-1]
    # dialate the frames
    # dialated_frame = cv2.erode(frame.detach().cpu().numpy(), np.ones((3, 3), np.uint8))
    binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    # numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, connectivity=4)
    # points = []
    # Calculate the details for each points
    # for stat, centroid in zip(stats[1:], centroids[1:]):
    #     photon_intensity = torch.sum(frame[0][stat[1]: stat[1] + stat[2], stat[0]: stat[0] + stat[3]])
    #     x_nm = centroid[1] * px_to_nm
    #     y_nm = centroid[0] * px_to_nm
    #     points.append([frame_number, float(x_nm), float(y_nm), 0, 0, float(photon_intensity)])
    # return points
    labels = torch.tensor(labels, device=frame.device)
    points = []
    # Get connected points
    # binary_frame = (frame > 0).float().unsqueeze(0)
    # labels = connected_components(binary_frame).squeeze(0).squeeze(0)
    unique_label = labels.unique()
    for label_number in unique_label[1:]:
        point = get_point(frame, labels, label_number, config)
        if point is not None:
            point = [frame_number] + point
            points.append(point)
    return points


def get_points_from_gt(gt, config):
    points = []
    for point in gt:
        # mu = torch.round(point[[5, 6]] * scale).int().tolist()
        points.append([int(point[0]), float(point[2] * config.Camera_Pixelsize), float(point[1] * config.Camera_Pixelsize), 0, 0, float(point[7])])
    return points


def main():
    predicted_points = []
    gt_points = []
    frames = []
    gts = []
    config_ = Config("config.yaml")
    config_.output_threshold = 0
    for frame_number in range(100):
        f_name = f"simulated_data/train/db_{frame_number}.pl"
        with open(f_name, 'rb') as handle:
            x, y = pickle.load(handle)
        y_gt, frame = y[-1], y[-3]
        gts.append(F.pad(y_gt, (0, 0, 0, 30 - y_gt.shape[0])))
        # inputs, labels = generate_labels(frame, frame_number, y_gt)
        predicted_point = get_points(frame, frame_number, config_)
        gt_point = get_points_from_gt(y_gt, config_)
        predicted_points.extend(predicted_point)
        gt_points.extend(gt_point)

    accuracy = get_ji_rmse(predicted_points, gt_points)
    print(f"JI: {accuracy}")


def get_label(frame, labels, label_number):
    x, y = torch.where(labels == label_number)
    weights = frame[0][x, y]
    x_mean = np.average(x.float().numpy(), weights=weights)
    y_mean = np.average(y.float().numpy(), weights=weights)
    x_px = int(x_mean.round())
    y_px = int(y_mean.round())
    image_patch = frame[:, x_px - 5: x_px + 5, y_px - 5: y_px + 5]

    return image_patch


def generate_labels(frame, frame_number, y_gt):
    gts = []
    patches = []
    scale = frame.shape[-1] / 32
    for point in y_gt:
        x_mean, y_mean = (point[[2, 1]] * scale).tolist()
        x_px, y_px = torch.round(point[[2, 1]] * scale).int().tolist()
        x_start, x_end = x_px - 5, x_px + 5
        y_start, y_end = y_px - 5, y_px + 5
        patch = frame[:, x_start: x_end, y_start: y_end]
        x = (x_mean - x_start) / (x_end - x_start)
        y = (y_mean - y_start) / (y_end - y_start)
        label = [y, x, float(point[7] / 20000.), float(point[8]), float(point[9])]
        gts.append(label)
        patches.append(patch)

    return patches, gts


if __name__ == '__main__':
    # Read data
    main()
