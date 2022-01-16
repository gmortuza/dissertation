import glob
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from read_config import Config


# Pytorch dataloader for the dataset
class ExtractPoints(Dataset):
    def __init__(self, data_dir, config):
        super(ExtractPoints, self).__init__()
        self.file_names = glob.glob(f"{data_dir}/*.pl")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        with open(file_name, "rb") as f:
            data, label = pickle.load(f)
            label = label.nan_to_num(0)
        return data.unsqueeze(0), label


def fetch_data_loader(config: Config, shuffle: bool = True, type_: str = 'train'):
    """

    Args:
        type_ ():
        config ():
        shuffle (object):
    """
    if type_ == 'train':
        train_dir = os.path.join(config.train_dir, "points")
        val_dir = os.path.join(config.val_dir, "points")
        train_dataset = ExtractPoints(train_dir, config)
        val_dataset = ExtractPoints(val_dir, config)
        config.log_param("num_training", len(train_dataset))
        config.log_param("num_validation", len(val_dataset))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                                  pin_memory=True)
        valid_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                                  pin_memory=True)

        return train_loader, valid_loader
    else:
        test_dir = os.path.join(config.input_dir, "test")
        test_dataset = ExtractPoints(test_dir, config, "test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=shuffle)

        return test_loader


if __name__ == '__main__':
    config_ = Config('../config.yaml')
    dl = fetch_data_loader(config_)
    x_, y_ = next(iter(dl[0]))
    print(x_.shape)
    print(y_.shape)
