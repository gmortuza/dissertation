from functools import partial

import h5py
import time
from torch import multiprocessing
# from multiprocessing.pool import ThreadPool as Pool
# import multiprocessing.Pool as Pool
from torch.multiprocessing import Pool
import torch
import numpy as np
from read_config import Config
from simulation import generate_binding_site_position, distribute_photons_single_binding_site, convert_frame, get_binding_site_position_distribution
from noise import get_noise
from drift import get_drift
import concurrent.futures
import copy
import torch.autograd.profiler as profiler

from tqdm import tqdm

# torch.manual_seed(1234)
# np.random.seed(1234)
torch.multiprocessing.set_sharing_strategy('file_system')
# multiprocessing.set_start_method("fork")
torch.multiprocessing.set_start_method('spawn', force=True)
# print(torch.multiprocessing.get_start_method())

# torch.use_deterministic_algorithms(True)


def show_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(func.__name__, ": Execution time is: ", round(time.time() - start, 2), "seconds")
        return res
    return wrapper


class GenerateData():
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.binding_site_position = None  # tensor in shape (2, num_of_event)
        self.num_of_binding_site = None
        self.distributed_photon = None  # tensor in shape(num_binding_event, total frames)
        # Initially the movie is just a random noise
        self.movie = get_noise(self.config)  # Tensor of shape (num_of_frames, height, width)
        self.frame_wise_noise = self.movie.mean((1, 2)).share_memory_()  # Tensor of shape (num_of_frames)
        # self.movie = torch.zeros((self.config.frames, self.config.image_size, self.config.image_size),
        #                          device=self.config.device, dtype=torch.float64)
        self.drifts = get_drift(self.config).share_memory_()  # tensor of shape (num_of_frames, 2)
        # This will be used to sample from multivariate distribution
        # This ground truth is only for visualization
        # Normally we don't need this ground truth for training purpose
        # For training purpose we will export something tensor shape
        # TODO: export training example supporting neural network training
        self.available_cpu_core = int(self.config.num_of_process)

    def get_frame_range(self):
        num_of_frame_in_single_file = self.config.frames // self.config.split_into
        frame_range = []
        for i in range(self.config.split_into):
            single_range = [i * num_of_frame_in_single_file, (i+1) * num_of_frame_in_single_file]
            frame_range.append(single_range)
        frame_range[-1][-1] = self.config.frames
        return frame_range

    def generate(self):
        self.binding_site_position = generate_binding_site_position(self.config).share_memory_()
        self.binding_site_position_distribution = get_binding_site_position_distribution(self.binding_site_position, self.config)
        self.num_of_binding_site = self.binding_site_position.shape[1]

        self.distribute_photons()
        frame_range = self.get_frame_range()
        for frame_start, frame_end in frame_range:
            self.convert_into_image(frame_start, frame_end)
        # self.save_image()

    @show_execution_time
    def distribute_photons(self):
        self.config.logger.info("Distributing photons")
        self.distributed_photon = torch.zeros((self.num_of_binding_site, self.config.frames),
                                              device=self.config.device).share_memory_()
        # Setting up method for multiprocessing
        photon_distributor = partial(distribute_photons_single_binding_site, config=self.config,
                                     num_of_binding_site=self.num_of_binding_site)
        # Number of core available
        if self.config.num_of_process > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.num_of_process) as executor:
                single_frame_distributed_photon = executor.map(photon_distributor, range(self.num_of_binding_site))
        else:
            single_frame_distributed_photon = map(photon_distributor, range(self.num_of_binding_site))

        for site_id, photon in tqdm(single_frame_distributed_photon, desc="Distributing photons", total=self.num_of_binding_site, disable=self.config.progress_bar_disable):
            self.distributed_photon[site_id] = photon

    @show_execution_time
    def convert_into_image(self, frame_start, frame_end):
        self.config.logger.info("Converting into Images")
        p_convert_frame = partial(convert_frame, config=self.config, drifts=self.drifts,
                                  distributed_photon=self.distributed_photon, frame_wise_noise=self.frame_wise_noise,
                                  binding_site_position_distribution=self.binding_site_position_distribution)
        if self.config.num_of_process > 1:
            with Pool(self.available_cpu_core) as pool:
                frame_details = pool.map(p_convert_frame, torch.arange(frame_start, frame_end))
        else:
            frame_details = map(p_convert_frame, torch.arange(frame_start, frame_end))

        # we save images in tensor for training purpose
        ground_truth = self.save_frames_in_tensor(frame_details, frame_start, frame_end)
        if self.config.save_for_picasso:
            self.save_frames_in_hdf5(ground_truth, frame_start, frame_end)

    def save_frames_in_tensor(self, frame_details, frame_start, frame_end):
        # concatnenating tensor in a loop might be slower
        # So we are creating a large tensor
        # frame_num, x, y, x_mean, y_mean, x_drifted, y_drifted, photons, s_x, s_y, noise

        combined_ground_truth = torch.zeros(((frame_end-frame_start) * self.config.max_number_of_emitter_per_frame, 11), device=self.config.device).share_memory_()
        current_num_of_emitter = 0
        for frame_id, frame, gt_infos in tqdm(frame_details, desc="Distributing photons", total=frame_end - frame_start, disable=self.config.progress_bar_disable):
            self.movie[frame_id, :, :] += frame
            emitter_to_keep = min(len(gt_infos), self.config.max_number_of_emitter_per_frame)
            combined_ground_truth[current_num_of_emitter: current_num_of_emitter+emitter_to_keep, :] = gt_infos[:emitter_to_keep, :]
            current_num_of_emitter += emitter_to_keep

        combined_ground_truth = combined_ground_truth[: current_num_of_emitter, :]

        torch.save(combined_ground_truth, self.config.output_file + f"_ground_truth_{frame_start + 1}_{frame_end}.pl")
        torch.save(self.movie[frame_start: frame_end], self.config.output_file + f"_{frame_start+1}_{frame_end}.pl")
        return combined_ground_truth

    @show_execution_time
    def save_frames_in_hdf5(self, ground_truth, frame_start, frame_end):
        # First save the movie
        self.movie[frame_start: frame_end].cpu().numpy().tofile(self.config.output_file + f"_{frame_start+1}_{frame_end}.raw")
        ground_truth = ground_truth.cpu().numpy()
        # frame_num, x, y, x_mean, y_mean, x_drifted, y_drifted, photons, s_x, s_y, noise
        gt_without_drift = np.rec.array(
            (
                ground_truth[:, 0],  # frames
                ground_truth[:, 1],  # x
                ground_truth[:, 2],  # y
                ground_truth[:, 7],  # photons
                ground_truth[:, 8],  # s_x
                ground_truth[:, 9],  # s_y
                ground_truth[:, 10],  # background
                np.full(ground_truth[:, 0].shape, .009),  # lpx
                np.full(ground_truth[:, 0].shape, .009),  # lpy
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

        gt_with_drift = np.rec.array(
            (
                ground_truth[:, 0],  # frames
                ground_truth[:, 5],  # x
                ground_truth[:, 6],  # y
                ground_truth[:, 7],  # photons
                ground_truth[:, 8],  # s_x
                ground_truth[:, 9],  # s_y
                ground_truth[:, 10],  # background
                np.full(ground_truth[:, 0].shape, .009),  # lpx
                np.full(ground_truth[:, 0].shape, .009),  # lpy
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

        content_for_yaml_file = f"Box Size: 7\nPixelsize: {self.config.Camera_Pixelsize}" \
                                f"\nFrames: {self.config.frames}\n" \
                                f"Height: {self.config.image_size}\n" \
                                f"Width: {self.config.image_size}"
        with h5py.File(self.config.output_file + f"_{frame_start+1}_{frame_end}_gt_without_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_without_drift)
            with open(self.config.output_file + f"_{frame_start + 1}_{frame_end}_gt_without_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)

        with h5py.File(self.config.output_file + f"_{frame_start + 1}_{frame_end}_gt_with_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_with_drift)
            with open(self.config.output_file + f"_{frame_start + 1}_{frame_end}_gt_with_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # print(torch.multiprocessing.get_start_method())
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()
