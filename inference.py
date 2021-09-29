import argparse
import torch
from torch import optim
import matplotlib.pyplot as plt

import utils
from data_loader import fetch_data_loader
from read_config import Config
from models.get_model import get_model
from generate_target import generate_target_from_path


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
    output = torch.zeros((config.image_size * config.output_resolution, config.image_size * config.output_resolution),
                         device=config.device)
    for data_batch in test_data_loader:
        output_batch = model(data_batch)
        output += torch.squeeze(output_batch.detach(), axis=1).sum(axis=0)
    # TODO: extract points from each of these frames
    # save the final output image
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.imsave("final_output.tiff", output.cpu().numpy(), cmap='gray')


if __name__ == '__main__':
    main()
    # config = Config('config.yaml')
    # images_1 = generate_target_from_path('simulated_data/train/data_1_8000_gt.pl', config, target='images').squeeze(1).sum(axis=0)
    # images_2 = generate_target_from_path('simulated_data/train/data_8001_16000_gt.pl', config, target='images').squeeze(1).sum(axis=0)
    # image = images_1 + images_2
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['savefig.dpi'] = 300
    # image = image.detach().cpu()
    # plt.imshow(image.detach().cpu(), cmap='gray')
    # plt.imsave("inference.tiff", image, cmap='gray')
    # plt.show()