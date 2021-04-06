import multiprocessing
import torch
import numpy as np
from read_config import Config
from simulation import Simulation


class GenerateData(Simulation):
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.binding_site_position = None  # tensor in shape (2, num_of_event)
        self.num_of_binding_site = None
        self.distributed_photon = None  # tensor in shape(num_binding_event, total frames)

    def generate(self):
        self.binding_site_position = self.generate_binding_site_position()
        self.num_of_binding_site = self.binding_site_position.shape[1]

        self.distribute_photons()
        self.convert_into_image()
        self.save_ground_truth()

    def distribute_photons(self):
        self.config.logger.info("Distributing photons")
        self.distributed_photon = torch.zeros((self.num_of_binding_site, self.config.Frames), device=self.config.device)
        # Number of core available
        available_cpu_core = int(np.ceil(multiprocessing.cpu_count()))
        pool = multiprocessing.Pool(processes=available_cpu_core)
        pool.map(self.distribute_photons_single_binding_site, np.arange(self.num_of_binding_site))
        pool.close()
        pool.join()

    def convert_into_image(self):
        self.config.logger.info("Converting into Images")

    def save_ground_truth(self):
        self.config.logger.info("Saving the ground truth")


if __name__ == "__main__":
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()
