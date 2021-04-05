import torch
import numpy
from read_config import Config
from simulation import Simulation


class GenerateData(Simulation):
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.handle_x = None  # Blinking events x coordinate
        self.handle_y = None  # Blinking events y coordinate

    def generate(self):
        self.distribute_photons()
        self.convert_into_image()
        self.save_ground_truth()

    def distribute_photons(self):
        self.handle_x, self.handle_y = self.generate_blinking_event()

    def convert_into_image(self):
        pass

    def save_ground_truth(self):
        pass


if __name__ == "__main__":
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()
