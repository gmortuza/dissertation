import h5py
import time

import multiprocessing
import torch
import numpy as np
from read_config import Config
from simulation import Simulation
from drift import get_drift
from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)

def show_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print("Execution time is: ", round(time.time() - start, 2), "seconds")
        return res
    return wrapper


class GenerateData(Simulation):
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.binding_site_position = None  # tensor in shape (2, num_of_event)
        self.num_of_binding_site = None
        self.distributed_photon = None  # tensor in shape(num_binding_event, total frames)
        self.movie = None  # Tensor of shape (num_of_frames, height, width)
        self.drift_x, self.drift_y = get_drift(self.config)
        # This ground truth is only for visualization
        # Normally we don't need this ground truth for training purpose
        # For training purpose we will export something tensor shape
        self.gt_frame = []
        self.gt_x_without_drift = []
        self.gt_y_without_drift = []
        self.gt_x_with_drift = []
        self.gt_y_with_drift = []
        self.gt_noise = []
        self.gt_photon = []
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
        self.distributed_photon = torch.zeros((self.num_of_binding_site, self.config.Frames), device=self.config.device)
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
        self.movie = torch.zeros((self.config.Frames, self.config.image_size, self.config.image_size),
                                 device=self.config.device, dtype=torch.float64)
        # =========  For debugging purpose
        for frame_id in tqdm(range(self.config.Frames), desc="Converting into image"):
            self.convert_frame(frame_id)
        # =========
        # pool = multiprocessing.Pool(processes=self.available_cpu_core)
        # pool.map(self.convert_frame, np.arange(self.config.Frames))
        # pool.close()
        # pool.join()

    def save_image(self):
        self.movie.numpy().tofile(self.config.output_file+".raw")
        # Save the info file
        # TODO: save here

    @show_execution_time
    def save_ground_truth(self):
        self.config.logger.info("Saving the ground truth")
        gt_without_drift = np.rec.array(
            (
                np.asarray(self.gt_frame),
                np.asarray(self.gt_x_without_drift),
                np.asarray(self.gt_y_without_drift),
                np.asarray(self.gt_photon),
                np.asarray(self.gt_noise),  # background
                np.full_like(self.gt_y_without_drift, .009),  # lpx
                np.full_like(self.gt_y_without_drift, .009),  # lpy
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
                np.asarray(self.gt_frame),
                np.asarray(self.gt_x_with_drift),
                np.asarray(self.gt_y_with_drift),
                np.asarray(self.gt_photon),
                np.asarray(self.gt_noise),  # background
                np.full_like(self.gt_y_with_drift, .009),  # lpx
                np.full_like(self.gt_y_with_drift, .009),  # lpy
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
                                f"\nFrames: {self.config.Frames}\n" \
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


if __name__ == "__main__":
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()
