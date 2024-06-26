import glob
import pickle
from typing import List

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from extract_location import point_extractor
import utils
from extract_location.generate_labels import pad_on_single_patch
from extract_location.model import ExtractLocationModel
from metrics import metrics
from read_config import Config


def add_padding_on_patches(patches: List[Tensor], start_positions: List[List], config: Config) \
        -> (List[Tensor], List[List]):
    """
    Take a list of patches of arbitrary shape and add padding to those patches to make them same shape
    Args:
        patches (): List of patches
        start_positions ():  List of start position of the patches. format: [frame_number, y_start, x_start]
        config ():  Configuration file

    Returns:
        start_pos: list of new start position of the patches after padding
        patches: list of padded patch
    """
    padded_patches = []
    for idx, patch in enumerate(patches):
        if patch.shape[0] > config.extracted_patch_size or patch.shape[1] > config.extracted_patch_size:
            continue
        patch, padding = pad_on_single_patch(patch[0], config)
        # Update the start position
        start_positions[idx][1] -= padding[2]
        start_positions[idx][2] -= padding[0]
        patch = patch.unsqueeze(0)
        padded_patches.append(patch)
    return padded_patches, start_positions


def extract_path_using_cc(frame: torch.Tensor, frame_number: int, config: Config) -> List[Tensor]:
    frame = frame[0]
    binary_frame = (frame > config.output_threshold).cpu().numpy().astype(np.int8)
    *_, stats, centroid = cv2.connectedComponentsWithStats(binary_frame, 4, cv2.CV_32S)
    # stats --> [x_start, y_start, width, height, num_element]
    patches = []
    start_position = []
    for stat in stats[1:]:
        if stat[-1] < 3:  # Ignore patch that have less than 3 connected component
            continue
        patch = frame[stat[1]: stat[1] + stat[3], stat[0]: stat[0] + stat[2]]
        patch = patch.unsqueeze(0)
        patches.append(patch)
        # Extract start position
        position = [frame_number, stat[1], stat[0]]
        start_position.append(position)
    return patches, start_position


def get_patches_from_frame(frame: torch.Tensor, frame_number: int, config: Config) -> List[Tensor]:
    patches, start_positions = extract_path_using_cc(frame, frame_number, config)
    # patches, points = get_point_weighted_mean(frame, frame_number, config)
    # Add padding around each patches
    patches, start_positions = add_padding_on_patches(patches, start_positions, config)
    return patches, start_positions


def get_inputs_from_dir(directory, config: Config):
    formatted_inputs = []
    start_positions_of_inputs = []
    # get the images that are 16x resolutions
    file_names = glob.glob(os.path.join(directory, 'db_*.pl'))
    gt_points = []
    for file_name in file_names[:100]:
        with open(file_name, 'rb') as handle:
            _, frames = pickle.load(handle)
            frame, gt = frames[4], frames[6]
            frame_number = int(file_name.split('/')[-1].split('.')[0].split('_')[-1])
            gt_point = point_extractor.get_points_from_gt(gt, config)
            patches, start_positions = get_patches_from_frame(frame, frame_number, config)
            gt_points.extend(gt_point)
            formatted_inputs.extend(patches)
            start_positions_of_inputs.extend(start_positions)
    formatted_inputs = torch.stack(formatted_inputs).to(config.device)
    return formatted_inputs, start_positions_of_inputs, gt_points


def get_inputs_from_frames(frames: torch.Tensor, config: Config, frame_numbers: List[int] = None):
    if frame_numbers is None:
        frame_numbers = torch.tensor(torch.arange(0, frames.shape[0]))
    patches, start_positions = [], []
    for frame_number, frame in zip(frame_numbers, frames):
        patch, start_position = get_patches_from_frame(frame, frame_number, config)
        patches.extend(patch)
        start_positions.extend(start_position)
    inputs = torch.stack(patches).to(config.device)
    return inputs, start_positions


def extract_points_from_inputs(inputs: torch.Tensor, start_positions: List[List], config: Config):
    if config.point_extractor is None:
        extractor_model = ExtractLocationModel(config).to(config.device)
        optimizer = Adam(extractor_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        _ = utils.load_checkpoint(extractor_model, config, optimizer, 'points')
    else:
        extractor_model = config.point_extractor

    outputs = extractor_model(inputs)
    outputs[:, [0, 6]] = nn.Sigmoid()(outputs[:, [0, 6]])
    formatted_output = point_extractor.get_formatted_points(outputs, config, start_positions)
    return formatted_output


def main(config):
    # get the data
    inputs, start_pos_of_inputs, gt_points = get_inputs_from_dir(config.val_dir, config)
    # get the model
    formatted_output = extract_points_from_inputs(inputs, start_pos_of_inputs, config)
    gt_points = torch.tensor(gt_points)
    nn_ji, nn_rmse, nn_efficiency, unrecognized_emitters = metrics.get_ji_rmse_efficiency_from_formatted_points(formatted_output, gt_points)
    print(f"NN ji: {nn_ji} \t nn rmse: {nn_rmse} \t efficiency: {nn_efficiency}")


if __name__ == '__main__':
    config_ = Config('../config.yaml')
    main(config_)
