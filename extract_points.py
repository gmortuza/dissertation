"""
Contains basic functionality to extract points from a given up sampled image
the points are in nanometers and are in the format of a numpy array
the points can be extracted using the following methods:
    - weighted mean
    - MLE (maximum likelihood estimation) fitting using scipy.optimize.curve_fit
    - using picasso gaussian fitting
    - Neural network (TODO)
"""
import pickle
import torch

from read_config import Config
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment, curve_fit
import torch.nn.functional as F
import cv2
from extract_location import localize as picasso_localize
import time

SINGLE_EMITTER_WIDTH = 20
SINGLE_EMITTER_HEIGHT = 20

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                       + c * ((y - yo) ** 2)))
    return g.ravel()


def get_ji_rmse(predicted_points, gt_points, radius=10):

    if not len(predicted_points) or not len(gt_points):
        return 0., 0.
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


def fit_psf_using_mle(patch, x_initial, y_initial):
    x = np.linspace(0, patch.shape[1] - 1, patch.shape[1])
    y = np.linspace(0, patch.shape[0] - 1, patch.shape[0])
    x, y = np.meshgrid(x, y)

    initial_guess = (1, x_initial, y_initial, 1.32, 1.32, 0, 0)

    popt, pcov = curve_fit(twoD_Gaussian, (x, y), patch.ravel(), p0=initial_guess)

    return popt


def get_points(frame, frame_number, config, method='scipy'):
    if method == 'picasso':
        return get_point_picasso(frame, frame_number, config)
    elif method == 'weighted_mean':
        return get_point_weighted_mean(frame, frame_number, config)
    elif method == 'scipy':
        return get_point_scipy(frame, frame_number, config)
    else:
        raise NotImplementedError('Method', method, 'not supported')


def get_points_from_gt(gt, config):
    points = []
    for point in gt:
        points.append(
            [int(point[0]), float(point[2] * config.Camera_Pixelsize), float(point[1] * config.Camera_Pixelsize), 0, 0,
             float(point[7])])
    return points


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


def get_point_picasso(frame, frame_number, config) -> list:
    info = {
        'baseline': 100,
        'sensitive': 1.0,
        'gain': 1,
        'qe': .09,
        'sensitivity': 1,
    }
    parameters = {
        'Box Size': 7,
        'Min. Net Gradient': .005,
        'Pixelsize': config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1],
        'box': 7
    }
    formatted_labels = []
    labels = picasso_localize.localize(frame.cpu().numpy(), info, parameters)
    for label in labels:
        x = label[1] * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        y = label[2] * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        photon = label[3]
        formatted_labels.append([frame_number, y, x, 0, 0, photon])
    return formatted_labels


def get_point_weighted_mean(frame, frame_number, config) -> list:
    binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    labels = torch.tensor(labels, device=frame.device)
    points = []
    unique_label = labels.unique()
    for label_number in unique_label[1:]:
        x, y = torch.where(labels == label_number)
        if len(x) < 10:
           continue
        weights = frame[0][x, y]
        x_mean = torch.sum(x * weights) / torch.sum(weights)
        y_mean = torch.sum(y * weights) / torch.sum(weights)
        x_start, x_end = int(x_mean.round()) - SINGLE_EMITTER_WIDTH // 2, int(
            x_mean.round()) + SINGLE_EMITTER_WIDTH // 2
        y_start, y_end = int(y_mean.round()) - SINGLE_EMITTER_HEIGHT // 2, int(
            y_mean.round()) + SINGLE_EMITTER_HEIGHT // 2
        image_patch = frame[:, x_start: x_end, y_start: y_end]
        photon_count = torch.sum(image_patch)
        x_nm = x_mean * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        y_nm = y_mean * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        points.append([frame_number, float(x_nm), float(y_nm), 0, 0, float(photon_count)])
    return points


def get_point_scipy(frame, frame_number, config) -> list:
    binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    labels = torch.tensor(labels, device=frame.device)
    points = []
    unique_label = labels.unique()
    for label_number in unique_label[1:]:
        x, y = torch.where(labels == label_number)
        if len(x) < 10:
            continue
        weights = frame[0][x, y]
        x_mean = torch.sum(x * weights) / torch.sum(weights)
        y_mean = torch.sum(y * weights) / torch.sum(weights)
        x_start, x_end = int(x_mean.round()) - SINGLE_EMITTER_WIDTH // 2, int(
            x_mean.round()) + SINGLE_EMITTER_WIDTH // 2
        y_start, y_end = int(y_mean.round()) - SINGLE_EMITTER_HEIGHT // 2, int(
            y_mean.round()) + SINGLE_EMITTER_HEIGHT // 2
        image_patch = frame[:, x_start: x_end, y_start: y_end]
        photon_count = torch.sum(image_patch)
        try:
            popt = fit_psf_using_mle(image_patch[0], x_mean - x_start, y_mean - y_start)
        except RuntimeError:
            # we will have runtime error if two emitter are too close
            # we will ignore this case
            continue
        x = popt[2] + x_start
        y = popt[1] + y_start
        x_nm = x * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        y_nm = y * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        points.append([frame_number, float(x_nm), float(y_nm), 0, 0, float(photon_count)])
    return points


def main():
    predicted_points_picasso = []
    predicted_points_weighted_mean = []
    predicted_points_scipy = []
    # track time
    picasso_time = 0
    scipy_time = 0
    weighted_mean_time = 0
    gt_points = []
    gts = []
    config_ = Config("config.yaml")
    config_.output_threshold = 0
    for frame_number in range(100):
        f_name = f"simulated_data_multi/train/db_{frame_number}.pl"
        with open(f_name, 'rb') as handle:
            x, y = pickle.load(handle)
        y_gt, frame = y[-1], y[-3]
        gts.append(F.pad(y_gt, (0, 0, 0, 30 - y_gt.shape[0])))
        now = time.time()
        predicted_point_picasso = get_point_picasso(frame, frame_number, config_)
        picasso_time += time.time() - now
        now = time.time()
        predicted_point_weighted_mean = get_point_weighted_mean(frame, frame_number, config_)
        weighted_mean_time += time.time() - now
        now = time.time()
        predicted_point_scipy = get_point_scipy(frame, frame_number, config_)
        scipy_time += time.time() - now
        gt_point = get_points_from_gt(y_gt, config_)
        predicted_points_picasso.extend(predicted_point_picasso)
        predicted_points_weighted_mean.extend(predicted_point_weighted_mean)
        predicted_points_scipy.extend(predicted_point_scipy)
        gt_points.extend(gt_point)

    # print results for picasso
    print("=="*10, " Picasso (", round(picasso_time, 2), 'second)', "=="*10)
    picasso_ji, picasso_rmse = get_ji_rmse(predicted_points_picasso, gt_points)
    picasso_efficiency = get_efficiency(picasso_ji, picasso_rmse)
    print(f"JI: {picasso_ji}\t, RMSE: {picasso_rmse}\t, Efficiency: {picasso_efficiency}")

    # print results for weighted mean
    print("=="*10, " Weighted Mean (", round(weighted_mean_time, 2), 'second)', "=="*10)
    weighted_mean_ji, weighted_mean_rmse = get_ji_rmse(predicted_points_weighted_mean, gt_points)
    weighted_mean_efficiency = get_efficiency(weighted_mean_ji, weighted_mean_rmse)
    print(f"JI: {weighted_mean_ji}\t, RMSE: {weighted_mean_rmse}\t, Efficiency: {weighted_mean_efficiency}")

    # print results for scipy
    print("=="*10, " Scipy (", round(scipy_time, 2), 'second)', "=="*10)
    scipy_ji, scipy_rmse = get_ji_rmse(predicted_points_scipy, gt_points)
    scipy_efficiency = get_efficiency(scipy_ji, scipy_rmse)
    print(f"JI: {scipy_ji}\t, RMSE: {scipy_rmse}\t, Efficiency: {scipy_efficiency}")


if __name__ == '__main__':
    # Read data
    main()
