import h5py
import time

import multiprocessing
import torch
import numpy as np
from read_config import Config
from simulation import Simulation
from noise import get_noise
from drift import get_drift
from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)


def show_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(func.__name__, ": Execution time is: ", round(time.time() - start, 2), "seconds")
        return res
    return wrapper


class GenerateData(Simulation):
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.binding_site_position = None  # tensor in shape (2, num_of_event)
        self.num_of_binding_site = None
        self.distributed_photon = None  # tensor in shape(num_binding_event, total frames)
        # Initially the movie is just a random noise
        self.movie = get_noise(self.config)  # Tensor of shape (num_of_frames, height, width)
        self.frame_wise_noise = self.movie.mean((1, 2))  # Tensor of shape (num_of_frames)
        # self.movie = torch.zeros((self.config.frames, self.config.image_size, self.config.image_size),
        #                          device=self.config.device, dtype=torch.float64)
        self.drifts = get_drift(self.config)  # tensor of shape (num_of_frames, 2)
        # This ground truth is only for visualization
        # Normally we don't need this ground truth for training purpose
        # For training purpose we will export something tensor shape
        # TODO: export training example supporting neural network training
        self.available_cpu_core = int(np.ceil(multiprocessing.cpu_count()))

    def generate(self):
        self.binding_site_position = self.generate_binding_site_position()
        self.num_of_binding_site = self.binding_site_position.shape[1]

        self.distribute_photons()
        self.convert_into_image()
        self.save_image()
        self.save_ground_truth()

    @show_execution_time
    def distribute_photons(self):
        self.config.logger.info("Distributing photons")
        self.distributed_photon = torch.zeros((self.num_of_binding_site, self.config.frames), device=self.config.device)
        # Number of core available
        # =========  For debugging purpose
        for site in tqdm(range(self.num_of_binding_site), desc="Distributing photons"):
            self.distribute_photons_single_binding_site(site)
        # =========
        # pool = multiprocessing.Pool(processes=self.available_cpu_core)
        # pool.map(self.distribute_photons_single_binding_site, np.arange(self.num_of_binding_site))
        # pool.close()
        # pool.join()

    @show_execution_time
    def convert_into_image(self):
        self.config.logger.info("Converting into Images")

        # =========  For debugging purpose
        # for frame_id in tqdm(range(self.config.frames), desc="Converting into image"):
        #     self.convert_frame(frame_id)
        # =========
        # pool = multiprocessing.Pool(processes=self.available_cpu_core)
        # frame_details = pool.map(self.convert_frame, np.arange(self.config.frames))
        # pool.close()
        # pool.join()
        frame_details = map(self.convert_frame, np.arange(self.config.frames))
        # We save images in hdf5 for visualization purpose
        self.save_frames_in_hdf5(frame_details)
        # we save images in tensor for training purpose
        self.save_frames_in_tensor(frame_details)

    def save_frames_in_tensor(self, frame_details):
        pass

    @show_execution_time
    def save_frames_in_hdf5(self, frame_details):
        gt_frames = []
        gt_x_without_drift = []
        gt_y_without_drift = []
        gt_x_with_drift = []
        gt_y_with_drift = []
        gt_photons = []
        gt_noise = []

        for (frame_id, frame, gt_pos) in frame_details:
            # putting all the frame together
            self.movie[frame_id, :, :] += frame
            if gt_pos.any():
                gt_pos = gt_pos if len(gt_pos) > 1 else [gt_pos]
                gt_x_without_drift.extend(self.binding_site_position[0, gt_pos].tolist())
                gt_y_without_drift.extend(self.binding_site_position[1, gt_pos].tolist())

                gt_x_with_drift.extend((self.binding_site_position[0, gt_pos] + self.drifts[frame_id, 0].numpy()).tolist())
                gt_y_with_drift.extend((self.binding_site_position[1, gt_pos] + self.drifts[frame_id, 1].numpy()).tolist())
                gt_photons.extend(self.distributed_photon[gt_pos, frame_id].tolist())
                gt_noise.extend([self.frame_wise_noise[frame_id].item()] * len(gt_pos))
                gt_frames.extend([frame_id] * len(gt_pos))

        gt_without_drift = np.rec.array(
            (
                np.asarray(gt_frames),
                np.asarray(gt_x_without_drift),
                np.asarray(gt_y_without_drift),
                np.asarray(gt_photons),
                np.asarray(gt_noise),  # background
                np.full_like(gt_y_without_drift, .009),  # lpx
                np.full_like(gt_y_without_drift, .009),  # lpy
            ), dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])

        gt_with_drift = np.rec.array(
            (
                np.asarray(gt_frames),
                np.asarray(gt_x_with_drift),
                np.asarray(gt_y_with_drift),
                np.asarray(gt_photons),
                np.asarray(gt_noise),  # background
                np.full_like(gt_y_with_drift, .009),  # lpx
                np.full_like(gt_y_with_drift, .009),  # lpy
            ), dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])

        content_for_yaml_file = f"Box Size: 7\nPixelsize: {self.config.Camera_Pixelsize}" \
                                f"\nFrames: {self.config.frames}\n" \
                                f"Height: {self.config.image_size}\n" \
                                f"Width: {self.config.image_size}"
        with h5py.File(self.config.output_file + "_gt_without_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_without_drift)
            with open(self.config.output_file + "_gt_without_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)

        with h5py.File(self.config.output_file + "_gt_with_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_with_drift)
            with open(self.config.output_file + "_gt_with_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)


    def save_image(self):
        self.movie.numpy().tofile(self.config.output_file+".raw")
        # Save the info file
        # TODO: save here

    @show_execution_time
    def save_ground_truth(self):
        self.config.logger.info("Saving the ground truth")



if __name__ == "__main__":
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()
