import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
from read_config import Config
from generate_target import generate_target_from_path
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import transforms


class SMLMDataset(Dataset):
    def __init__(self, data_set_dir, config):
        gt_file_names = glob.glob(f"{data_set_dir}/*_gt.pl")
        self.data, self.target = self.get_data_and_target(gt_file_names, config)

    def get_data_and_target(self, gt_file_names: list, config):
        data = None
        target = None
        for single_gt_file in gt_file_names:
            data_file_name = single_gt_file.replace("_gt", "")
            single_data = torch.load(data_file_name)
            single_data_shape = single_data.shape
            single_data = single_data.reshape(single_data_shape[0], 1, single_data_shape[1], single_data_shape[2])
            single_target = generate_target_from_path(single_gt_file, config)
            if data is None:
                data = single_data
                target = single_target
            else:
                data = torch.cat((data, single_data), dim=0)
                target = torch.cat((target, single_target), dim=0)
        return data, target

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x = x / x.norm()
        return x, y


def fetch_data_loader(config: Config, shuffle: bool = True):
    """

    :param types:
    :param config:
    :param shuffle
    :return:
    """
    train_dir = os.path.join(config.input_dir, "train")
    val_dir = os.path.join(config.input_dir, "validation")
    test_dir = os.path.join(config.input_dir, "test")
    train_dataset = SMLMDataset(train_dir, config)
    val_dataset = SMLMDataset(val_dir, config)
    test_dataset = SMLMDataset(test_dir, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=shuffle, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=shuffle, pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    config = Config('config.yaml')
    dl = fetch_data_loader(config)
    x, y = next(iter(dl[0]))
