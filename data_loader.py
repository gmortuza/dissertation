import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader

from read_config import Config
from generate_target import generate_target_from_path


class SMLMDataset(Dataset):
    def __init__(self, data_set_dir, config, type_='train'):
        gt_file_names = glob.glob(f"{data_set_dir}/*_gt.pl")
        self.config = config
        self.type_ = type_
        if self.type_ == 'train':
            self.data, self.target = self.get_data_and_target(gt_file_names)
        else:
            self.data = self.get_data_and_target(gt_file_names)

    def get_data_and_target(self, gt_file_names: list):
        data = None
        target = None
        for single_gt_file in gt_file_names:
            # TODO: Find a way of getting the input other than this
            data_file_name = single_gt_file.replace("_gt", "")
            single_data = torch.load(data_file_name)
            single_data_shape = single_data.shape
            single_data = single_data.reshape(single_data_shape[0], 1, single_data_shape[1], single_data_shape[2])
            # there will be no target during test
            if self.type_ != 'test':
                single_target = generate_target_from_path(single_gt_file, self.config, target="points")
            # Normalize target here so that we can change the datatype to avoid memory crash
            # single_target = single_target / single_target.norm()
            # single_target = single_target
            if data is None:
                data = single_data
                if self.type_ != 'test':
                    target = single_target
            else:
                data = torch.cat((data, single_data), dim=0)
                if self.type_ != 'test':
                    target = torch.cat((target, single_target), dim=0)
        if self.type_ != 'test':
            return data, target
        return data

    def _get_image_from_point(self, point: torch.Tensor) -> torch.Tensor:
        # points --> [x, y, photons]
        high_res_image_size = self.config.image_size * self.config.output_resolution
        high_res_movie = torch.zeros((high_res_image_size, high_res_image_size), device=self.config.device)
        # TODO: remove this for loop and vectorize this
        for blinker in point[point[:, 2] > 0.]:
            mu = torch.round(blinker[[0, 1]] * self.config.output_resolution).int()
            high_res_movie[mu[1]][mu[0]] += blinker[2]
        return high_res_movie.unsqueeze(0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        x: torch.Tensor = self.data[index]
        # Normalize the data
        # TODO: change 50000 to maximum assigned photons
        x = x / 50000.
        # Upscale the input
        x = torch.nn.Upsample(scale_factor=self.config.output_resolution, mode="bilinear")(x.unsqueeze(0))[0]
        if self.type_ == 'train':
            y: torch.Tensor = self._get_image_from_point(self.target[index])
            # convert points into images
            y = y / 50000.
            return x, y
        return x


def fetch_data_loader(config: Config, shuffle: bool = True, type_: str = 'train'):
    """

    Args:
        type_ ():
        config ():
        shuffle (object):
    """
    if type_ == 'train':
        train_dir = os.path.join(config.input_dir, "train")
        val_dir = os.path.join(config.input_dir, "validation")
        train_dataset = SMLMDataset(train_dir, config)
        val_dataset = SMLMDataset(val_dir, config)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle)
        valid_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=shuffle)

        return train_loader, valid_loader
    else:
        test_dir = os.path.join(config.input_dir, "test")
        test_dataset = SMLMDataset(test_dir, config, "test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=shuffle)

        return test_loader


if __name__ == '__main__':
    config_ = Config('config.yaml')
    dl = fetch_data_loader(config_)
    x_, y_ = next(iter(dl[0]))
