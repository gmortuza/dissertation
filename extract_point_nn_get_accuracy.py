import glob

import pickle

import cv2
import numpy as np
import torch.nn as nn
from read_config import Config
from extract_location.model import ExtractLocationModel
import utils
from torch.optim import Adam
import extract_points
import torch


def get_formatted_data(config):
    # we will use validation dataset to get the accuracy
    dir_ = config.train_dir
    #
    formatted_inputs = []
    start_positions_of_inputs = []
    # get the images that are 16x resolutions
    file_names = glob.glob(dir_ + '/db_*.pl')
    gt_points = []
    cc_points = []  # the points that are extracted using connected components algorithm
    for file_name in file_names:
        with open(file_name, 'rb') as handle:
            _, frames = pickle.load(handle)
            frame, gt = frames[4], frames[6]
            # perform connected component analysis
            frame_number = int(file_name.split('/')[-1].split('.')[0].split('_')[-1])
            predicted_point = extract_points.get_points(frame, frame_number, config_)
            gt_point = extract_points.get_points_from_gt(gt, config_)
            gt_points.extend(gt_point)
            cc_points.extend(predicted_point)
            patches, start_positions = get_formatted_points(predicted_point, frame)
            formatted_inputs.extend(patches)
            start_positions_of_inputs.extend(start_positions)

    return formatted_inputs, start_positions_of_inputs, gt_points, cc_points


def get_formatted_points(points, frame):
    patches = []
    start_positions = []
    for point in points:
        x_px, y_px = int(round(point[1] * 512 / 107 / 32)), int(round(point[2] * 512 / 107 / 32))
        patch = frame[:, x_px - 10: x_px + 10, y_px - 10: y_px + 10]
        patches.append(patch)
        start_positions.append([point[0], x_px - 10, y_px - 10])
    return patches, start_positions


def get_formatted_outputs(outputs, start_positions_of_inputs):
    formatted_output = []
    for point, start in zip(outputs, start_positions_of_inputs):
        frame_number, x_start, y_start = start
        if nn.Sigmoid()(point[0]) > 0.5:
            x_px = x_start + point[1].item() * 20.
            y_px = y_start + point[2].item() * 20.
            # Convert into pixel coordinates
            x_nm = x_px * 107 * 32 / 512
            y_nm = y_px * 107 * 32 / 512
            formatted_output.append([frame_number, x_nm, y_nm, point[3].item(), point[4].item(), point[5].item()])
        if nn.Sigmoid()(point[6]) > 0.5:
            x_px = x_start + point[7].item() * 20.
            y_px = y_start + point[8].item() * 20.
            x_nm = x_px * 107 * 32 / 512
            y_nm = y_px * 107 * 32 / 512
            formatted_output.append([frame_number, x_nm, y_nm, point[9].item(), point[10].item(), point[11].item()])
    return formatted_output


def main(config):
    # get the data
    inputs, start_pos_of_inputs, gt_points, cc_points = get_formatted_data(config)
    inputs = torch.stack(inputs).to(config.device)
    # get the model
    model = ExtractLocationModel(config).to(config.device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_val_acc = utils.load_checkpoint(model, config, optimizer, 'points')
    outputs = model(inputs)
    formatted_output = get_formatted_outputs(outputs, start_pos_of_inputs)
    nn_ji, nn_rmse = extract_points.get_ji_rmse(formatted_output, gt_points)
    cc_ji, cc_rmse = extract_points.get_ji_rmse(cc_points, gt_points)
    nn_efficiency = extract_points.get_efficiency(nn_ji, nn_rmse)
    cc_efficiency = extract_points.get_efficiency(cc_ji, cc_rmse)
    print(f"NN ji: {nn_ji} \t nn rmse: {nn_rmse} \t efficiency: {nn_efficiency}")
    print(f"CC ji: {cc_ji} \t nn rmse: {cc_rmse} \t efficiency: {cc_efficiency}")


if __name__ == '__main__':
    config_ = Config('config.yaml')
    main(config_)
