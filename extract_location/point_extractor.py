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
from extract_location import picasso_localize
import time
import extract_points_nn

SINGLE_EMITTER_WIDTH = 20
SINGLE_EMITTER_HEIGHT = 20
JACCARD_INDEX_RADIUS = 10

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





def fit_psf_using_mle(patch, x_initial, y_initial):
    x = np.linspace(0, patch.shape[1] - 1, patch.shape[1])
    y = np.linspace(0, patch.shape[0] - 1, patch.shape[0])
    x, y = np.meshgrid(x, y)

    initial_guess = (1, x_initial, y_initial, 1.32, 1.32, 0, 0)

    popt, pcov = curve_fit(twoD_Gaussian, (x, y), patch.ravel(), p0=initial_guess)

    return popt


def get_points(frame, frame_number, config, method='scipy'):
    if method == 'picasso':
        return get_point_picasso(frame, frame_number, config)[1]
    elif method == 'weighted_mean':
        return get_point_weighted_mean(frame, frame_number, config)[1]
    elif method == 'scipy':
        return get_point_scipy(frame, frame_number, config)[1]
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
    return None, formatted_labels


def get_point_weighted_mean(frame, frame_number, config) -> list:
    binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    labels = torch.tensor(labels, device=frame.device)
    patches = []
    points = []
    unique_label = labels.unique()
    for label_number in unique_label[1:]:
        x, y = torch.where(labels == label_number)
        if len(x) < 10 or len(x) > 100:
           continue
        weights = frame[0][x, y]
        x_mean = torch.sum(x * weights) / torch.sum(weights)
        y_mean = torch.sum(y * weights) / torch.sum(weights)
        x_start, x_end = int(x_mean.round()) - SINGLE_EMITTER_WIDTH // 2, int(
            x_mean.round()) + SINGLE_EMITTER_WIDTH // 2
        y_start, y_end = int(y_mean.round()) - SINGLE_EMITTER_HEIGHT // 2, int(
            y_mean.round()) + SINGLE_EMITTER_HEIGHT // 2
        image_patch = frame[:, x_start: x_end, y_start: y_end]
        patches.append(image_patch)
        photon_count = torch.sum(image_patch)
        x_nm = x_mean * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        y_nm = y_mean * config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]
        points.append([frame_number, float(x_nm), float(y_nm), 0, 0, float(photon_count)])
    return patches, points


def get_point_scipy(frame, frame_number, config) -> list:
    binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    labels = torch.tensor(labels, device=frame.device)
    points = []
    unique_label = labels.unique()
    patches = []
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
        patches.append(image_patch)
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
    return patches, points


def get_point_nn(frames, frame_numbers, config: Config) -> list:
    patches, start_position = extract_points_nn.format_point_from_batch(frames, config, frame_numbers)
    formatted_output = extract_points_nn.get_accuracy_from_inputs(patches, start_position, config)
    return formatted_output


def main():
    pass


if __name__ == '__main__':
    # Read data
    main()
