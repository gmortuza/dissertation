# Convert pickle to target dataset
import sys

import numpy as np
import torch
from read_config import Config
from utils import generate_image_from_points
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_target_gt(input_tensor: torch.Tensor, config: Config, start_frame: int, end_frame: int, types: str = "image") -> torch.Tensor:
    gt_as_points = get_gt_as_points(input_tensor, config, start_frame, end_frame)
    if types == "points":
        return gt_as_points
    else:
        return get_gt_as_image(input_tensor, config, start_frame, end_frame)

def get_gt_as_points(input_tensor: torch.Tensor, config: Config, start_frame: int, end_frame: int) -> torch.Tensor:
    """
    This will convert raw ground truth into neural network's target
    input_tensor is of shape: (None, 11)
    Each of them will contain
    frame_num, x, y, x_mean, y_mean, x_drifted, y_drifted, photons, s_x, s_y, noise
    :param input_tensor:
    :return:
    """
    number_of_frame = end_frame - start_frame
    # TODO: During adding noise change 5 to 6
    target_tensor = torch.zeros(size=(number_of_frame, config.max_number_of_emitter_per_frame, 5), device=config.device)
    for frame_id in input_tensor[:, 0].unique():
        frame_gts = input_tensor[input_tensor[:, 0] == frame_id]
        # image = generate_image_from_points(frame_gts.view(1, frame_gts.size(0), -1), config)
        # Generate image from gts
        # x_mean, y_mean, photons, s_x, s_y, noise
        # TODO: Add noise later
        # target_tensor[int(frame_id) - start_frame, :len(frame_gts), :] = frame_gts[:, [3, 4, 7, 8, 9, 10]]
        target_tensor[int(frame_id) - start_frame, :len(frame_gts), :] = frame_gts[:, [3, 4, 7, 8, 9]]
        # target_tensor[int(frame_id) - start_frame, :len(frame_gts), :] = frame_gts[:, [3, 4]]
    # Normalize the photons
    total_photon = target_tensor[:, :, 2].sum(1)
    have_photons = total_photon > 0
    target_tensor[have_photons, :, 2] /= total_photon[have_photons].view(-1, 1)
    # If std is zero then torch through an error during creating GMM. So clipping that with a very small number
    target_tensor[:, :, [3, 4]] = torch.clip(target_tensor[:, :, [3, 4]], min=.00001, max=10)
    return target_tensor


def get_gt_as_image(input_tensor: torch.Tensor, config: Config, start_frame: int, end_frame: int) -> torch.Tensor:
    total_frame = end_frame - start_frame
    high_res_image_size = config.image_size * config.output_resolution
    high_res_movie = torch.zeros((total_frame, high_res_image_size, high_res_image_size))
    for idx, frame_id in tqdm(enumerate(range(start_frame, end_frame)), total=total_frame, desc="Generating target",
                              disable=config.progress_bar_disable):
        # Take the points for this frame only
        frame_blinkers = input_tensor[input_tensor[:, 0] == frame_id]
        for blinker in frame_blinkers:
            mu = torch.round(blinker[[1, 2]].to(config.device) * config.output_resolution).int()
            high_res_movie[idx][mu[1]][mu[0]] += blinker[7]
    # Save higher resolution image for test
    # high_res_movie_single = torch.sum(high_res_movie, axis=0)
    # high_res_movie_single[high_res_movie_single > 0] = 255.
    # plt.imsave("test1.tiff", high_res_movie_single, cmap='gray')
    return high_res_movie


def generate_target_from_path(path: str, config: Config):
    if isinstance(config, str):
        config = Config(config)
    input_tensor = torch.load(path)
    path_arr = path.replace("_gt.pl", "").split("_")
    start_frame, end_frame = int(path_arr[-2]) - 1, int(path_arr[-1])
    return generate_target_gt(input_tensor, config, start_frame, end_frame)


if __name__ == '__main__':
    path = "/Users/golammortuza/workspace/nam/dnam_nn/simulated_data/train/data_1_1050_gt.pl"
    config_path = "config.yaml"
    dataset = generate_target_from_path(path, config_path)
    print(dataset)

