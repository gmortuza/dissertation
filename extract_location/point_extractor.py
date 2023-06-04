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
from extract_location import point_extractor_nn

SINGLE_EMITTER_WIDTH = 20
SINGLE_EMITTER_HEIGHT = 20
JACCARD_INDEX_RADIUS = 10


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
    raw_points_frame_start[:, [4, 5, 10, 11]] *= config.point_extraction_pixel_size

    # extract points for first emitter
    first_emitter_loc = torch.where((raw_points_frame_start[:, 3] > .999))[0]
    first_emitter = raw_points_frame_start[first_emitter_loc, :]
    first_emitter = first_emitter[:, [0, 4, 5, 6, 7, 8]]

    # extract points for second emitters
    # second_emitter_loc = torch.where((raw_points_frame_start[:, 9] > .9) & (raw_points_frame_start[:, 12] > .001))[0]
    second_emitter_loc = torch.where((raw_points_frame_start[:, 9] > .95))[0]
    second_emitter = raw_points_frame_start[second_emitter_loc, :]
    second_emitter = second_emitter[:, [0, 10, 11, 12, 13, 14]]

    # combine two emitters
    emitters = torch.cat((first_emitter, second_emitter), dim=0)
    return emitters


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


def get_points(frame, config, frame_number=None, method='scipy'):
    if method == 'picasso':
        return get_point_picasso(frame, config, frame_number)[1]
    elif method == 'weighted_mean':
        return get_point_weighted_mean(frame, config, frame_number)[1]
    elif method == 'scipy':
        return get_point_scipy(frame, config, frame_number)[1]
    elif method == 'nn':
        return get_point_nn(frame, config, frame_number)[1]
    else:
        raise NotImplementedError('Method', method, 'not supported')


def get_points_from_gt(gt, config):
    points = []
    for point in gt:
        if point[7] > 0:
            points.append(
                [int(point[0]), float(point[2] * config.Camera_Pixelsize), float(point[1] * config.Camera_Pixelsize),
                 0, 0, float(point[7])])
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


def get_point_picasso(frame, config, frame_number) -> list:
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


def get_point_weighted_mean(frame, config, frame_number) -> list:
    binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    labels = torch.tensor(labels, device=frame.device)
    patches = []
    points = []
    unique_label = labels.unique()
    for label_number in unique_label[1:]:
        x, y = torch.where(labels == label_number)
        if len(x) < 10 or len(x) > config.multi_emitter_threshold:
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
        pixel_to_nm = config.Camera_Pixelsize * config.resolution_slap[0] / frame.shape[-1]

        x_nm = x_mean * pixel_to_nm
        y_nm = y_mean * pixel_to_nm
        points.append([frame_number, float(x_nm), float(y_nm), 0, 0, float(photon_count)])
    return patches, points


def get_point_scipy(frame, config, frame_number) -> list:
    frame = frame.detach().cpu().numpy()
    binary_frame = (frame[0] > config.output_threshold).astype(np.int8)
    # binary_frame = (frame[0] > config.output_threshold).detach().cpu().numpy().astype(np.int8)
    *_, labels = cv2.connectedComponents(binary_frame, connectivity=4)
    # labels = torch.tensor(labels, device=frame.device)
    points = []
    patches = []
    for label_number in np.unique(labels)[1:]:
        x, y = np.where(labels == label_number)
        if len(x) < 10:
            continue
        weights = frame[0][x, y]
        x_mean = np.sum(x * weights) / np.sum(weights)
        y_mean = np.sum(y * weights) / np.sum(weights)
        x_start, x_end = int(x_mean.round()) - SINGLE_EMITTER_WIDTH // 2, int(
            x_mean.round()) + SINGLE_EMITTER_WIDTH // 2
        y_start, y_end = int(y_mean.round()) - SINGLE_EMITTER_HEIGHT // 2, int(
            y_mean.round()) + SINGLE_EMITTER_HEIGHT // 2
        image_patch = frame[:, x_start: x_end, y_start: y_end]
        patches.append(image_patch)
        photon_count = np.sum(image_patch)
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


def get_point_nn(frames, config, frame_numbers) -> list:
    # if frame and frame numbers are not list convert them to list
    frames = frames if isinstance(frames, list) else [frames]
    frame_numbers = frame_numbers if isinstance(frame_numbers, list) else [frame_numbers]
    patches, start_position = point_extractor_nn.get_inputs_from_frames(frames, config, frame_numbers)
    formatted_output = point_extractor_nn.extract_points_from_inputs(patches, start_position, config)
    return None, formatted_output


def combine_points_from_multiple_frames(points_with_res: list, config) -> list:
    threshold = 40
    # first add all the points into one list
    all_points = []
    combined_points = []
    for res, points in points_with_res.items():
        all_points += points
    # convert the list into numpy array
    all_points = np.asarray(all_points)
    # loop through the frame number
    for frame_number in np.unique(all_points[:, 0]):
        # get the points for this frame
        points = all_points[all_points[:, 0] == frame_number]
        # get their distance matrix
        distance_matrix = pairwise_distances(points[:, 1:3], points[:, 1:3])
        for col_idx in range(distance_matrix.shape[0]):
            if distance_matrix[0, col_idx] == -1:
                continue
            col = distance_matrix[:, col_idx]
            # get the index of the points that are close to this point
            close_points_idx = (np.where(col < threshold))[0]
            # take the lowest 4 points based on the distance
            close_points_idx = sorted(close_points_idx, key=lambda x: col[x])[:4]
            if len(close_points_idx) > 1:
                combined_points.append(np.average(points[close_points_idx], axis=0).tolist())
            # set all value of this close points to -1 in the distance matrix
            distance_matrix[close_points_idx, :] = float('inf')
            distance_matrix[:, close_points_idx] = float('inf')
    return combined_points


def main():
    pass


if __name__ == '__main__':
    # Read data
    main()
