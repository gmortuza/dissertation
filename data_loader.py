import glob
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from read_config import Config


class SMLMDataset(Dataset):
    def __init__(self, dataset_dir, config, type_='train'):

        self.config = config
        self.type_ = type_
        self.dataset_dir = dataset_dir
        self.total_data = self._upsample_images()

    def _upsample_images(self):
        file_names = glob.glob(f"{self.dataset_dir}/data_*_gt.pl")
        # If there are already upsampled images we will return the number of images
        upsampled_file_names = glob.glob(f"{self.dataset_dir}/up_{self.config.output_resolution}_*")
        if len(upsampled_file_names) > 0:
            return len(upsampled_file_names)
        else:
            total = 0
            normalize_factor = 20000.
            upsample_fn = torch.nn.Upsample(scale_factor=self.config.output_resolution, mode='bilinear',
                                            align_corners=True)
            for file_name in sorted(file_names):
                start = int(file_name.split('_')[-3]) - 1
                input_ = torch.load(file_name.replace('_gt', ''), ).unsqueeze(1) / normalize_factor
                label_ = torch.load(file_name)
                label_[:, 7] /= normalize_factor
                for idx, single_input in tqdm(enumerate(input_, start), total=input_.shape[0],
                                              desc="Upsampling the data individual",
                                              disable=self.config.progress_bar_disable, leave=False):
                    single_input_upsampled = upsample_fn(single_input.unsqueeze(0)).squeeze(0)
                    if self.type_ == 'test':
                        single_label_upsampled = None
                    else:
                        single_label = label_[label_[:, 0] == idx]
                        single_label_upsampled = self._get_image_from_point(single_label)
                        # single_label_upsampled = self._convert_into_sparse_tensor(single_label)
                        # combine_training = torch.cat((single_input_upsampled, single_label_upsampled), dim=0)
                    f_name = f"{self.dataset_dir}/up_{self.config.output_resolution}_{idx}.pl"
                    # save the input and label as pickle
                    with open(f_name, 'wb') as handle:
                        pickle.dump([single_input_upsampled, single_label_upsampled], handle)
                    # torch.save(combine_training, f_name)
                    total += 1
            return total

    # def _convert_into_sparse_tensor(self, points):
    #     high_res_image_size = self.config.image_size * self.config.output_resolution
    #     x_cor = []
    #     y_cor = []
    #     v = []
    #     for blinker in points:
    #         mu = torch.round(blinker[[1, 2]] * self.config.output_resolution).int()
    #         x_cor.append(int(mu[0]))
    #         y_cor.append(int(mu[1]))
    #         v.append(blinker[7].float())
    #     sparse_tensor_i = [[0] * len(y_cor), y_cor, x_cor]
    #     sparse_tensor = torch.sparse_coo_tensor(sparse_tensor_i, v, (1, high_res_image_size, high_res_image_size),
    #                                             device=self.config.device)
    #     return sparse_tensor

    def _get_image_from_point(self, point: torch.Tensor) -> torch.Tensor:
        # points --> [x, y, photons]
        high_res_image_size = self.config.image_size * self.config.output_resolution
        high_res_movie = torch.zeros((high_res_image_size, high_res_image_size), device=self.config.device)
        # TODO: remove this for loop and vectorize this
        for blinker in point[point[:, 7] > 0.]:
            mu = torch.round(blinker[[5, 6]] * self.config.output_resolution).int()
            high_res_movie[mu[1]][mu[0]] += blinker[7]
        return high_res_movie.unsqueeze(0)

    def __len__(self):
        return self.total_data

    def __getitem__(self, index: int):
        f_name = f"{self.dataset_dir}/up_{self.config.output_resolution}_{index}.pl"
        with open(f_name, 'rb') as handle:
            x, y = pickle.load(handle)
        # if y is none then it's test. so we don't require the label
        return x, y if y is not None else x


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
