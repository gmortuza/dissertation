import glob
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from read_config import Config


class SMLMDataset(Dataset):
    def __init__(self, dataset_dir, config, type_='train'):

        self.config = config
        self.type_ = type_
        self.dataset_dir = dataset_dir
        self.image_sizes = [32, 63, 125, 249]
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

            for file_name in sorted(file_names):
                start = int(file_name.split('_')[-3]) - 1
                input_ = torch.load(file_name.replace('_gt', ''), ).unsqueeze(1) / normalize_factor
                label_ = torch.load(file_name)
                label_[:, 7] /= normalize_factor
                for idx, single_input in tqdm(enumerate(input_, start), total=input_.shape[0],
                                              desc="Upsampling the data individual",
                                              disable=self.config.progress_bar_disable, leave=False):
                    single_input_upsampled = self._get_upsample_input(single_input)
                    # if self.type_ == 'test':
                    #     single_label_upsampled = None
                    # else:
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

    def _get_upsample_input(self, single_input):
        single_input = single_input.unsqueeze(0)
        up_scaled_images = []
        for image_size in self.image_sizes:
            up_scaled_image = torch.nn.Upsample(size=image_size, mode='bilinear', align_corners=True)(single_input)
            up_scaled_images.append(up_scaled_image.squeeze(0))
        return up_scaled_images


    def _get_image_from_point(self, point: torch.Tensor) -> torch.Tensor:
        # points --> [x, y, photons]
        high_res_images = []
        for image_size in self.image_sizes:
            high_res_movie = torch.zeros((image_size, image_size), device=self.config.device)
            res_scale = image_size / 32
            # TODO: remove this for loop and vectorize this
            for blinker in point[point[:, 7] > 0.]:
                mu = torch.round(blinker[[5, 6]] * res_scale).int()
                high_res_movie[mu[1]][mu[0]] += blinker[7]
            high_res_images.append(high_res_movie.unsqueeze(0))
        return high_res_images

    def __len__(self):
        return self.total_data

    def _transform(self):
        return transforms.Compose([
            transforms.Normalize((0.), (1.))
        ])

    def __getitem__(self, index: int):
        f_name = f"{self.dataset_dir}/up_{self.config.output_resolution}_{index}.pl"
        with open(f_name, 'rb') as handle:
            x, y = pickle.load(handle)
        # if y is none then it's test. so we don't require the label
        if y is None:
            return x
        # Normalize x
        x[0] -= x[0].mean()
        return x, y


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
