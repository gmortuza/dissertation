import argparse
import collections

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


def show_separability(input_batch, output_batch, gt_batch, predicted_points, idx):
    # save the input figure
    frame_number = int(gt_batch[-1][idx][0][0].tolist())
    fig, ax = plt.subplots()
    ax.imshow(input_batch[0][idx][0].detach().cpu(), cmap='gray')
    ax.grid(False)
    for gt_row in gt_batch[-1][idx][gt_batch[-1][idx][:, 0] != 0].numpy():
        ax.plot(gt_row[1], gt_row[2], marker='x', color="red")
    fig.show()
    fig.savefig(f"/data/golam/dnam_nn/separability/input_{frame_number}.png", dpi=300, bbox_inches='tight')
    # Save the output from different resolution
    for i in range(len(output_batch)):
        shape = output_batch[i].shape[-1]
        frame_gt_points = np.asarray(predicted_points[shape])
        frame_gt_points = frame_gt_points[frame_gt_points[:, 0] == frame_number]


        fig, ax = plt.subplots()
        ax.imshow(output_batch[i][idx][0].detach().cpu(), cmap='gray')
        ax.grid(False)
        for gt_row in frame_gt_points:
            x, y = gt_row[1] * (shape // 32) / 107, gt_row[2] * (shape // 32) / 107
            # if x == 0 or y == 0: break
            ax.plot(y, x, marker='x', color="red")
        fig.show()
        fig.savefig(f"/data/golam/dnam_nn/separability/output_{frame_number}_{shape}.png", dpi=300, bbox_inches='tight')


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
    nm_to_pixel = config.Camera_Pixelsize
    points = np.asarray(points)
    points = points[np.argsort(points[:, 0])]
    predicted_points = np.rec.array(
        (
            points[:, 0],  # frames
            points[:, 2] / nm_to_pixel,  # x
            points[:, 1] / nm_to_pixel,  # y
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


def save_points_for_picasso_multi_resolution(points_with_res: dict, config: Config):
    combine_points = point_extractor.combine_points_from_multiple_frames(points_with_res, config)
    save_points_for_picasso(combine_points, config, config.resolution_slap[0])
    for resolution, points in points_with_res.items():
        save_points_for_picasso(points, config, resolution)

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
    save_points_for_picasso_multi_resolution(predicted_points, config)
    config.logger.info("Inference is done")
    return predicted_points, outputs


def inference_with_gt(config):
    # Get the model
    model = config.upsample_model
    # test_data_loader = fetch_data_loader(config, type_='test')
    valid_data_loader = fetch_data_loader(config, type_='test')
    output = torch.zeros((config.resolution_slap[config.extract_point_from_resolution],
                          config.resolution_slap[config.extract_point_from_resolution]), device=config.device)
    gt = torch.zeros((config.resolution_slap[config.extract_point_from_resolution],
                      config.resolution_slap[config.extract_point_from_resolution]), device=config.device)
    #
    predicted_points = []
    target_points = []
    predicted_points = collections.defaultdict(list)
    for data_batch, gt_batch, frame_numbers in tqdm(valid_data_loader, total=len(valid_data_loader)):
        data_batch = utils.convert_device(data_batch, config.device)
        with torch.no_grad():
            output_batch = model(data_batch, data_batch)
        # export predictions for each resolution slap
        output_batch[0] /= 255
        output_batch[1] /= 255
        for output in output_batch:
            config.point_extraction_pixel_size = config.Camera_Pixelsize * config.resolution_slap[0] / \
                                                 output.shape[-1]
            predicted_point, target_point = export_predictions(output, gt_batch, config)
            predicted_points[output.shape[-1]].extend(predicted_point)
        target_points.extend(target_point)
        # output += torch.squeeze(output_batch.detach(), axis=1).sum(axis=0)
        # gt += torch.squeeze(gt_batch[3].detach().to(config.device), axis=1).sum(axis=0)
    # TODO: extract points from each of these frames
    # save the final output image
    # save_points_for_picasso(predicted_points, config)
    # Get JI
    target_points = torch.tensor(target_points)
    for resolution, predictions in predicted_points.items():
        predictions = torch.tensor(predictions)
        radius = 10
        jaccard_index, rmse, efficiency, unrecognized_emitters = metrics.get_ji_rmse_efficiency_from_formatted_points(predictions,
                                                                                               target_points,
                                                                                               radius=radius)
        unrecognized_emitters = unrecognized_emitters.astype('int')
        np.savetxt(config.output_dir + '/' + str(resolution) + '_unrecognized_emitters.csv', unrecognized_emitters.astype('int'),
                   delimiter=',', header='Frame,distance,photons,explainable')
        print(config.point_extraction_method, resolution, radius, "Jaccard Index: ", jaccard_index)
    # predicted_points = torch.tensor(predicted_points)

    save_points_for_picasso_multi_resolution(predicted_points, config)
    # plt.rcParams['figure.dpi'] = 600
    # plt.rcParams['savefig.dpi'] = 600
    # plt.imsave(config.output_dir+"/gt_output_1.tiff", output.cpu().numpy(), cmap='gray')
    # TODO: remove this section later
    # save the ground truth data as well for comparison
    # plt.imsave(config.output_dir + '/' + str(config.resolution_slap[-1]) + '_gt_output.tiff', gt.cpu().numpy(),
    #            cmap='gray')


def compare_method_resolution(config):
    # methods = ['picasso', 'scipy', 'weighted_mean']
    methods = ['weighted_mean', 'picasso']
    for method in methods:
        print("Method: ", method)
        print("==" * 20)
        # for resolution in resolutions:
        # config_.extract_point_from_resolution = resolution
        config_.point_extraction_method = method
        # config_.point_extraction_pixel_size = config_.Camera_Pixelsize * config_.resolution_slap[0] / \
        #                                       config_.resolution_slap[config_.extract_point_from_resolution]
        inference_with_gt(config_)
    print("Inference is done")


if __name__ == '__main__':
    args = read_args()
    config_ = Config(args.config_file, from_terminal=args.terminal, purpose='inference')
    compare_method_resolution(config_)

    # inference_without_gt(config_)
