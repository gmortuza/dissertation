import glob

import torch
from torch.utils.data import Dataset, DataLoader
from read_config import Config
from generate_target import generate_target_from_path
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torchvision import transforms


class SMLMDataset(Dataset):
    def __init__(self, config: Config):
        gt_file_names = glob.glob(f"{config.simulated_data_dir}/*_gt.pl")
        self.data, self.target = self.get_data_and_target(gt_file_names, config)

    def get_data_and_target(self, gt_file_names: list, config: Config):
        data = None
        target = None
        for single_gt_file in gt_file_names:
            data_file_name = single_gt_file.replace("_gt", "")
            single_data = torch.load(data_file_name, config.device)
            single_data_shape = single_data.shape
            single_data = single_data.reshape(single_data_shape[0], 1, single_data_shape[1], single_data_shape[2])
            single_target = generate_target_from_path(single_gt_file, config).to(device=config.device)
            if data is None:
                data = single_data
                target = single_target
            else:
                data = torch.cat((data, single_data), dim=0)
                target = torch.cat((target, single_target), dim=0)
            # TODO: remove this later
            break
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
    dataset = SMLMDataset(config)
    indices = np.arange(len(dataset))
    split = int(np.floor(config.validation_split * len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=valid_sampler, pin_memory=True)

    return train_loader, valid_loader


if __name__ == '__main__':
    config = Config('../config.yaml')
    dl = fetch_data_loader(config)
    x, y = next(iter(dl[0]))
