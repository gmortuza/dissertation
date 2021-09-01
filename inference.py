import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt

import utils
from data_loader import fetch_data_loader
from read_config import Config
from models.get_model import get_model


def read_args():
    parser = argparse.ArgumentParser("Extract the result from a particular directory")
    parser.add_argument("-d", "--directory", help="Location of the directory", default="simulated_data/test")
    parser.add_argument("-c", "--config_file", help="Configuration file", default="config.yaml")

    args = parser.parse_args()
    return args


def main():
    args = read_args()
    config = Config(args.config_file)
    # Get the model
    model = get_model(config)
    # model will only be used for evaluation so no need for backpropagation
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Restore model weights
    config.logger.info(f"Restoring parameters from {config.checkpoint_dir}")
    _ = utils.load_checkpoint(config.checkpoint_dir, model, config, optimizer)
    test_data_loader = fetch_data_loader(config, type_='test')
    output = None
    for data_batch in test_data_loader:
        output_batch = model(data_batch)
        if output is None:
            output = output_batch
        else:
            output = torch.cat((output, output_batch), axis=0)
    # TODO: extract points from each of these frames
    # Single output
    single_frame = torch.sum(output, axis=0).view(800, 800).detach().cpu().numpy()
    # save the final output image
    plt.imsave("final_output.tiff", single_frame, cmap='gray')


if __name__ == '__main__':
    main()
