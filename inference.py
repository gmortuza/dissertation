import argparse

import h5py
import numpy as np
import os
import torch
from torch import optim
import matplotlib.pyplot as plt

from extract_location import point_extractor
from metrics import metrics
import utils
from data_loader import fetch_data_loader
from read_config import Config
from models.get_model import get_model
from tqdm import tqdm
import pandas as pd

plt.grid(False)


def show(data, output, target, index, predicted_point):
    frame_number = int(target[-1][index][0][0].tolist())
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(data[0][index][1].cpu().detach().numpy(), cmap='gray')
    ax[0].grid(None)
    ax[0].set_title("Input")
    ax[1].imshow(output[0][index][0].cpu().detach().numpy(), cmap='gray')
    ax[1].grid(None)
    ax[1].set_title("Output")
    ax[2].imshow(target[0][index][0].cpu().detach().numpy(), cmap='gray')
    ax[2].grid(None)
    ax[2].set_title("Target")
    for gt_row in target[-1][index][target[-1][index][:, 0] != 0].numpy():
        ax[0].plot(gt_row[1], gt_row[2], marker='x', color="red")
        ax[2].plot(gt_row[1] * 2, gt_row[2] * 2, marker='x', color="red")
    # put marker on the predicted point
    predicted_point = np.asarray(predicted_point)
    predicted_point = predicted_point[predicted_point[:, 0] == frame_number]
    for single_point in predicted_point:
        ax[1].plot(single_point[2] / 50, single_point[1] / 50, marker='x', color="red")

    plt.suptitle("Frame number -- " + str(frame_number))
    # plt.grid(None)
    plt.tight_layout()
    plt.show()


def read_args():
    parser = argparse.ArgumentParser("Extract the result from a particular directory")
    parser.add_argument("-d", "--directory", help="Location of the directory", default="simulated_data/test")
    parser.add_argument("-c", "--config_file", help="Configuration file", default="config.yaml")
    parser.add_argument("-t", "--terminal", help="This is running from terminal", action="store_true")

    args = parser.parse_args()
    return args


def export_predictions_without_target(frames, frame_numbers, config):
    predicted_points = []
    for frame_number, frame in zip(frame_numbers, frames):
        predicted_point = point_extractor.get_points(frame, config, int(frame_number),
                                                     method=config.point_extraction_method)
        if len(predicted_point):
            predicted_points.extend(predicted_point)
    return predicted_points


def export_predictions(predictions, targets, config):
    targets = targets[-1].cpu().numpy()
    frames_numbers = targets[:, 0, 0]
    predicted_points = []
    gt_points = []
    if config.point_extraction_method == 'nn':
        predicted_points = point_extractor.get_points(predictions, config, frames_numbers, method='nn').tolist()
    for frame_number, frame in zip(frames_numbers, predictions):
        frame_target = targets[targets[:, 0, 0] == frame_number][0]
        frame_target = frame_target[frame_target[:, 0] == frame_number]
        if config.point_extraction_method != 'nn':
            predicted_point = point_extractor.get_points(frame, config, frame_number,
                                                         method=config.point_extraction_method)
            if len(predicted_point):
                predicted_points.extend(predicted_point)
        gt_point = point_extractor.get_points_from_gt(frame_target, config)
        if len(gt_point):
            gt_points.extend(gt_point)
    return predicted_points, gt_points


def save_points_for_picasso(points, config, image_resolution):
    points = np.asarray(points)
    points = points[np.argsort(points[:, 0])]
    predicted_points = np.rec.array(
        (
            points[:, 0],  # frames
            points[:, 2] / config.Camera_Pixelsize,  # x
            points[:, 1] / config.Camera_Pixelsize,  # y
            points[:, 5] * 1000,  # photons
            np.full(points[:, 0].shape, .85),  # s_x
            np.full(points[:, 0].shape, .85),  # s_y
            np.full(points[:, 0].shape, 1),  # background
            np.full(points[:, 0].shape, .009),  # lpx
            np.full(points[:, 0].shape, .009),  # lpy
        ), dtype=[
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ])
    content_for_yaml_file = f"Box Size: 7\nPixelsize: {config.Camera_Pixelsize}" \
                            f"\nFrames: {config.total_training_example}\n" \
                            f"Height: {config.image_size}\n" \
                            f"Width: {config.image_size}"
    with h5py.File(config_.output_dir + '/' + "output_picasso_" + str(image_resolution) + ".hdf5", "w") as locs_file:
        locs_file.create_dataset("locs", data=predicted_points)
        with open(config_.output_dir + '/' + "output_picasso_" + str(image_resolution) + ".yaml", "w") as yaml_file:
            yaml_file.write(content_for_yaml_file)


def inference_without_gt(config):
    model = config.upsample_model
    inference_data_loader = fetch_data_loader(config, type_="inference")
    outputs = []
    predicted_points = {}
    for scale in config.resolution_slap[1:]:
        outputs.append(torch.zeros(scale, scale, device=config.device))
        predicted_points[scale] = []
    for data_batch, frame_numbers in tqdm(inference_data_loader):
        data_batch = utils.convert_device(data_batch, config.device)
        output_batch = model(data_batch, data_batch)
        for index, output in enumerate(output_batch):
            outputs[index] += output.squeeze(1).sum(dim=0).detach()
            predicted_points[output.shape[-1]].extend(export_predictions_without_target(output, frame_numbers, config))
    return predicted_points, outputs


def inference_with_gt(config):
    # Get the model
    model = config.upsample_model
    # test_data_loader = fetch_data_loader(config, type_='test')
    train_data_loader, valid_data_loader = fetch_data_loader(config, type_='train')
    output = torch.zeros((config.resolution_slap[config.extract_point_from_resolution],
                          config.resolution_slap[config.extract_point_from_resolution]), device=config.device)
    gt = torch.zeros((config.resolution_slap[config.extract_point_from_resolution],
                      config.resolution_slap[config.extract_point_from_resolution]), device=config.device)
    #
    predicted_points = []
    target_points = []
    for data_batch, gt_batch, frame_numbers in tqdm(valid_data_loader, total=len(valid_data_loader)):
        data_batch = utils.convert_device(data_batch, config.device)
        output_batch = model(data_batch, data_batch)
        predicted_point, target_point = export_predictions(output_batch[config.extract_point_from_resolution],
                                                           gt_batch, config)
        # for i in range(data_batch[0].shape[0]):
        #     show(data_batch, output_batch, gt_batch, i, predicted_point)
        predicted_points.extend(predicted_point)
        target_points.extend(target_point)
        # output += torch.squeeze(output_batch.detach(), axis=1).sum(axis=0)
        # gt += torch.squeeze(gt_batch[3].detach().to(config.device), axis=1).sum(axis=0)
    # TODO: extract points from each of these frames
    # save the final output image
    # save_points_for_picasso(predicted_points, config)
    # Get JI
    predicted_points = torch.tensor(predicted_points)
    target_points = torch.tensor(target_points)
    jaccard_index, rmse, efficiency = metrics.get_ji_rmse_efficiency_from_formatted_points(predicted_points,
                                                                                           target_points, radius=250)
    print(config.extract_point_from_resolution, "Jaccard Index: ", jaccard_index)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    # plt.imsave(config.output_dir+"/gt_output_1.tiff", output.cpu().numpy(), cmap='gray')
    # TODO: remove this section later
    # save the ground truth data as well for comparison
    plt.imsave(config.output_dir + '/' + str(config.resolution_slap[-1]) + '_gt_output.tiff', gt.cpu().numpy(),
               cmap='gray')


def compare_method_resolution(config):
    methods = ['weighted_mean', 'nn']
    resolutions = [-1, -2, -3, -4]
    for method in methods:
        print("Method: ", method)
        print("==" * 20)
        for resolution in resolutions:
            config_.extract_point_from_resolution = resolution
            config_.point_extraction_method = method
            config_.point_extraction_pixel_size = config_.Camera_Pixelsize * config_.resolution_slap[0] / \
                                                  config_.resolution_slap[config_.extract_point_from_resolution]
            inference_with_gt(config_)


if __name__ == '__main__':
    args = read_args()
    config_ = Config(args.config_file, from_terminal=args.terminal, purpose='inference')
    # compare_method_resolution(config_)

    inference_without_gt(config_)
