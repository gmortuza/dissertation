import os
import sys

from simulation.generate_data import GenerateData
from read_config import Config


def simulate(config: Config):
    """

    :param config:
    :type config:
    :return:
    :rtype:
    """
    if isinstance(config, str):
        config = Config(config)
    for sim_type in ['train', 'validation', 'test']:
        if sim_type == 'train':
            config.frame_to_generate = int(config.total_frames * (1 - config.validation_split - config.test_split))
            config.simulated_data_dir = os.path.join(config.input_dir, "train")
        elif sim_type == 'test':
            config.frame_to_generate = int(config.total_frames * config.test_split)
            config.simulated_data_dir = os.path.join(config.input_dir, "test")
        else:  # validation split
            config.frame_to_generate = int(config.total_frames * config.validation_split)
            config.simulated_data_dir = os.path.join(config.input_dir, "validation")

        if config.frame_to_generate < 1:
            continue

        if not os.path.exists(config.simulated_data_dir):
            os.makedirs(config.simulated_data_dir)

        config.file_name_to_save = os.path.join(config.simulated_data_dir, config.simulated_file_name)
        config.logger.info(f"Simulation: Generating {sim_type} data")
        config.simulation_type = sim_type
        # We want to save that for picasso to compare our result with picasso module
        config.save_for_picasso = True if sim_type == 'test' else False
        generate_data = GenerateData(config)
        generate_data.generate()
    config.logger.info("Finish generating the data")


if __name__ == '__main__':
    from_borah = True if len(sys.argv) > 1 else False
    config = Config("config.yaml", from_borah)

    simulate(config)
