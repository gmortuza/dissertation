import os

import multiprocessing
import torch
import numpy as np
from read_config import Config
from simulation import Simulation
from drift import get_drift
from tqdm import tqdm


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
        self.gt_x = []
        self.gt_y = []
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

    def save_ground_truth(self):
        self.config.logger.info("Saving the ground truth")


if __name__ == "__main__":
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()
