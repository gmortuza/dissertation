import os
from functools import partial

import h5py
import time
from torch.multiprocessing import Pool
import torch
import numpy as np
from read_config import Config
from simulation.simulate import generate_binding_site_position, distribute_photons_single_binding_site, convert_frame,\
    get_binding_site_position_distribution
from simulation.noise import get_noise
from simulation.drift import get_drift
import pandas as pd


from tqdm import tqdm, trange

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


class GenerateData:
    def __init__(self, config):
        self.config = config
        self.binding_site_position = None  # tensor in shape (2, num_of_event)
        self.num_of_binding_site = None
        self.distributed_photon = None  # tensor in shape(num_binding_event, total frames)
        self.binding_site_position_distribution = None
        self.drifts = get_drift(self.config).share_memory_()  # tensor of shape (num_of_frames, 2)
        self.available_cpu_core = int(self.config.num_of_process)

    def get_frame_range(self):
        num_of_frame_in_single_file = self.config.frame_to_generate // self.config.split_into
        frame_range = []

        for i in range(self.config.split_into):
            f_start = i * num_of_frame_in_single_file
            f_end = (i+1) * num_of_frame_in_single_file
            single_range = [f_start, f_end]
            frame_range.append(single_range)
        frame_range[-1][1] = self.config.frame_to_generate
        return frame_range

    # @show_execution_time
    def generate(self):
        self.binding_site_position = generate_binding_site_position(self.config).share_memory_()
        # Assign a multivariate normal distribution for each position where the mean will be binding_site_position
        self.binding_site_position_distribution = get_binding_site_position_distribution(self.binding_site_position,
                                                                                         self.config)
        self.num_of_binding_site = self.binding_site_position.shape[1]

        self.distribute_photons()
        frame_range = self.get_frame_range()
        self.config.logger.info("Converting into frames.....")

        if self.config.num_of_process > 1:
            with Pool(processes=self.config.num_of_process) as pool:
                pool.map(self.convert_into_image, frame_range)
        else:
            for range_ in tqdm(frame_range, desc="Converting image", disable=self.config.progress_bar_disable):
                self.convert_into_image(range_)
        self.config.logger.info("Finish converting the data")

    # @show_execution_time
    def distribute_photons(self):
        self.config.logger.info("Distributing photons")
        self.distributed_photon = torch.zeros((self.num_of_binding_site, self.config.frame_to_generate),
                                              device=self.config.device).share_memory_()
        # Setting up method for multiprocessing
        photon_distributor = partial(distribute_photons_single_binding_site, config=self.config,
                                     num_of_binding_site=self.num_of_binding_site)

        single_frame_distributed_photon = map(photon_distributor, range(self.num_of_binding_site))

        for site_id, photon in tqdm(single_frame_distributed_photon, desc="Distributing photons", total=self.num_of_binding_site, disable=self.config.progress_bar_disable):
            self.distributed_photon[site_id] = photon

    # @show_execution_time
    def convert_into_image(self, frame_range):
        frame_start, frame_end = frame_range
        combined_ground_truth = torch.zeros(((frame_end-frame_start) * self.config.max_number_of_emitter_per_frame, 11),
                                            device=self.config.device)
        # self.config.logger.info("Generating ", str(frame_range), "frames")
        current_num_of_emitter = 0
        noise_shape = (frame_end - frame_start, self.config.resolution_slap[0], self.config.resolution_slap[0])
        movie = get_noise(self.config.noise_type, noise_shape, self.config.bg_model)
        # 32 px with noise, 32 px without noise, 63px, 125px, 249px
        movies = [movie]
        frames_to_generate = self.config.resolution_slap[:-1] if self.config.data_gen_type == 'single_distribute' else \
            self.config.resolution_slap
        for image_size in frames_to_generate:
            movie = torch.zeros((frame_end - frame_start, image_size, image_size))
            movies.append(movie)
        frame_wise_noise = movie.mean((1, 2))  # Tensor of shape (num_of_frames)
        p_convert_frame = partial(convert_frame, frame_started=frame_start, config=self.config, drifts=self.drifts,
                                  distributed_photon=self.distributed_photon, frame_wise_noise=frame_wise_noise,
                                  binding_site_position_distribution=self.binding_site_position_distribution)

        for frame_id in range(frame_start, frame_end):
            frame_id, frames, gt_infos = p_convert_frame(frame_id=frame_id)
            # this is the input images with the noise
            movies[0][frame_id - frame_start, :, :] += frames[0]
            # these are the label images. There will be multiple label images based on the user configuration
            for frame, movie in zip(frames, movies[1:]):
                movie[frame_id - frame_start, :, :] = frame
            # sort ground truth based on the most bright spot
            gt_infos = gt_infos[gt_infos[:, 7].sort(descending=True)[1]]
            emitter_to_keep = min(len(gt_infos), self.config.max_number_of_emitter_per_frame)
            combined_ground_truth[current_num_of_emitter: current_num_of_emitter + emitter_to_keep, :] \
                = gt_infos[:emitter_to_keep, :]
            current_num_of_emitter += emitter_to_keep
            # frame_num, x, y, x_mean, y_mean, x_drifted, y_drifted, photons, s_x, s_y, noise
            # target[frame_id - frame_start][:][:emitter_to_keep] = gt_infos[:emitter_to_keep, [3, 4, 7, 8, 9, 10]]
        combined_ground_truth = combined_ground_truth[: current_num_of_emitter, :]

        torch.save(combined_ground_truth, self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}_gt.pl")
        # extract the frame average info into csv
        # TODO: Move this during creating the frame
        df = pd.DataFrame(combined_ground_truth[:, [0, 7]].cpu().numpy(), columns=['frame_number', 'photons_count'])
        df.groupby('frame_number').agg({'frame_number': 'count', 'photons_count': 'sum'}).rename(
            columns={'frame_number': 'emitter_count'}).reset_index().to_csv(self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}.csv", index=False)

        torch.save(movies[0], self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}_{movies[0].shape[-1]}_with_noise.pl")
        for movie in movies[1:]:
            torch.save(movie, self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}_{movie.shape[-1]}.pl")

        # torch.save(target, self.config.simulated_file_name + f"_{frame_start+1}_{frame_end}_target.pl")
        if self.config.save_for_picasso:
            self.save_frames_in_hdf5(movie[0], combined_ground_truth, frame_start, frame_end)

    # @show_execution_time
    def save_frames_in_hdf5(self, movie, ground_truth, frame_start, frame_end):
        file_name = self.config.file_name_to_save + f"_{frame_start  + 1}_{frame_end}"
        movie = movie.cpu().numpy()
        movie.tofile(file_name + ".raw")
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
                                f"\nFrames: {self.config.frame_to_generate}\n" \
                                f"Height: {self.config.image_size}\n" \
                                f"Width: {self.config.image_size}"
        with h5py.File(self.config.file_name_to_save + f"_{frame_start+1}_{frame_end}_gt_without_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_without_drift)
            with open(self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}_gt_without_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)

        with h5py.File(self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}_gt_with_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_with_drift)
            with open(self.config.file_name_to_save + f"_{frame_start + 1}_{frame_end}_gt_with_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)


if __name__ == "__main__":
    configuration = Config(config_file_path="../config.yaml")
    generate_data = GenerateData(configuration)
    generate_data.generate()
