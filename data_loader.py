import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
from read_config import Config
from generate_target import generate_target_from_path


class SMLMDataset(Dataset):
    def __init__(self, data_set_dir, config):
        gt_file_names = glob.glob(f"{data_set_dir}/*_gt.pl")
        self.data, self.target = self.get_data_and_target(gt_file_names, config)

    @staticmethod
    def get_data_and_target(gt_file_names: list, config: Config):
        data = None
        target = None
        for single_gt_file in gt_file_names:
            data_file_name = single_gt_file.replace("_gt", "")
            single_data = torch.load(data_file_name)
            single_data_shape = single_data.shape
            single_data = single_data.reshape(single_data_shape[0], 1, single_data_shape[1], single_data_shape[2])
            single_target = generate_target_from_path(single_gt_file, config)
            # Normalize target here so that we can change the datatype to avoid memory crash
            # single_target = single_target / single_target.norm()
            # single_target = single_target
            if data is None:
                data = single_data
                target = single_target
            else:
                data = torch.cat((data, single_data), dim=0)
                target = torch.cat((target, single_target), dim=0)
        return data, target

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index: int):
        x: torch.Tensor = self.data[index]
        y: torch.Tensor = self.target[index]
        # Normalize the data
        # TODO: change 50000 to maximum assigned photons
        x = x / 50000.
        y = y / 50000.
        return x, y


def fetch_data_loader(config: Config, shuffle: bool = True):
    """

    Args:
        config ():
        shuffle (object):
    """
    train_dir = os.path.join(config.input_dir, "train")
    val_dir = os.path.join(config.input_dir, "validation")
    test_dir = os.path.join(config.input_dir, "test")
    train_dataset = SMLMDataset(train_dir, config)
    val_dataset = SMLMDataset(val_dir, config)
    test_dataset = SMLMDataset(test_dir, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle)
    valid_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    config_ = Config('config.yaml')
    dl = fetch_data_loader(config_)
    x_, y_ = next(iter(dl[0]))
