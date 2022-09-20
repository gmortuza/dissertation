import glob
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import imageio

from read_config import Config
import utils
import numpy as np


class SMLMDataset(Dataset):
    def __init__(self, dataset_dir, config, type_='train'):

        self.config = config
        self.type_ = type_
        self.dataset_dir = dataset_dir
        if type_ != 'inference':
            self.total_data = self._upsample_images()

    def _upsample_images(self):
        file_names = glob.glob(f"{self.dataset_dir}/data_*_gt.pl")
        # If there are already upsampled images we will return the number of images
        upsampled_file_names = glob.glob(f"{self.dataset_dir}/db_*")
        if len(upsampled_file_names) > 0:
            return len(upsampled_file_names)
        else:
            total = 0
            for file_name in sorted(file_names):
                start = int(file_name.split('_')[-3]) - 1
                input_ = torch.load(file_name.replace('_gt', '_32_with_noise'), )
                input_ = utils.normalize(input_)
                all_label = []
                total_frames = self.config.resolution_slap[
                               :-1] if self.config.data_gen_type == 'single_distribute' else \
                    self.config.resolution_slap
                for image_sizes in total_frames:
                    all_label.append(utils.normalize(
                        torch.load(file_name.replace('_gt', '_' + str(image_sizes))).to(self.config.device)).unsqueeze(
                        1))
                # TODO: set the device according to config
                label_ = torch.load(file_name).to(torch.device('cpu'))
                for idx, single_input in tqdm(enumerate(input_, start), total=input_.shape[0],
                                              desc="Upsampling the data individual",
                                              disable=self.config.progress_bar_disable, leave=False):
                    single_input_upsampled = self._get_upsample_input(single_input)
                    labels = [
                        # TODO: set device according to config
                        torch.tensor(label.data[idx - start].cpu().numpy(), device=torch.device('cpu')) for label in
                        all_label
                    ]
                    single_label = label_[label_[:, 0] == idx]
                    single_label_upsampled = self._get_image_from_point(single_label, [512])
                    labels.append(single_label_upsampled[0])
                    labels.append(single_label)
                    f_name = f"{self.dataset_dir}/db_{idx}.pl"
                    # save the input and label as pickle
                    with open(f_name, 'wb') as handle:
                        pickle.dump([single_input_upsampled, labels], handle)
                    # torch.save(combine_training, f_name)
                    total += 1
            return total

    def _get_upsample_input(self, single_input):
        single_input = single_input.unsqueeze(0).unsqueeze(0)
        up_scaled_images = []
        for image_size in self.config.resolution_slap:
            up_scaled_image = torch.nn.Upsample(size=image_size, mode='bilinear', align_corners=True)(single_input)
            # TODO: set device according to config
            up_scaled_images.append(up_scaled_image.squeeze(0).to(torch.device('cpu')))
        return up_scaled_images

    def _get_image_from_point(self, point: torch.Tensor, image_sizes: list = None) -> torch.Tensor:
        # points --> [x, y, photons]
        image_sizes = self.config.resolution_slap if image_sizes is None else image_sizes
        high_res_images = []
        for image_size in image_sizes:
            # TODO: set device according to config
            high_res_movie = torch.zeros((image_size, image_size), device=torch.device('cpu'))
            res_scale = image_size / self.config.resolution_slap[0]
            # TODO: remove this for loop and vectorize this
            for blinker in point[point[:, 7] > 0.]:
                mu = torch.round(blinker[[5, 6]]).int()
                high_res_movie[mu[1]][mu[0]] += blinker[7]
            high_res_images.append(high_res_movie.unsqueeze(0))
        return high_res_images

    def __len__(self):
        if self.type_ == 'inference':
            return len(glob.glob(self.dataset_dir + "/*.tif"))
        return self.total_data if self.config.total_training_example == -1 else self.config.total_training_example
        # return self.total_data

    def _get_inference_data(self, idx):
        idx += 1
        # get previous frame
        frame_id = format(idx, '05d')
        current_frame_name = f"{self.dataset_dir}/" + frame_id + ".tif"
        current_frame = imageio.imread(current_frame_name)
        current_frame = torch.tensor(current_frame.astype('float32'))
        current_frame = current_frame / current_frame.max()
        current_frames = self._get_upsample_input(current_frame)
        if idx - 1 == 0:
            previous_frame = torch.zeros_like(current_frame)
        else:
            previous_frame_id = format(idx - 1, '05d')
            previous_frame_name = f"{self.dataset_dir}/" + previous_frame_id + ".tif"
            previous_frame = imageio.imread(previous_frame_name)
            previous_frame = torch.tensor(previous_frame.astype('float32'))
            previous_frame = previous_frame / previous_frame.max()
        previous_frames = self._get_upsample_input(previous_frame)
        # get the next frame
        if idx > len(self):
            next_frame = torch.zeros_like(current_frame)
        else:
            next_frame_id = format(idx + 1, '05d')
            next_frame_name = f"{self.dataset_dir}/" + next_frame_id + ".tif"
            next_frame = imageio.imread(next_frame_name)
            next_frame = torch.tensor(next_frame.astype('float32'))
            next_frame = next_frame / next_frame.max()
        next_frames = self._get_upsample_input(next_frame)
        x_data = []
        for x_p, x_, x_n in zip(previous_frames, current_frames, next_frames):
            x_data.append(torch.cat((x_p, x_, x_n), dim=0) * 255.)
        return x_data, idx

    def __getitem__(self, index: int):
        if self.type_ == 'inference':
            return self._get_inference_data(index)
        f_name = f"{self.dataset_dir}/db_{index}.pl"
        with open(f_name, 'rb') as handle:
            x, y = pickle.load(handle)

        pre_f_name = f"{self.dataset_dir}/db_{index - 1}.pl"
        next_f_name = f"{self.dataset_dir}/db_{index + 1}.pl"
        try:
            with open(pre_f_name, 'rb') as handle:
                x_prev, y_prev = pickle.load(handle)
        except FileNotFoundError:
            x_prev = [torch.zeros_like(x_) for x_ in x]

        try:
            with open(next_f_name, 'rb') as handle:
                x_next, y_next = pickle.load(handle)
        except FileNotFoundError:
            x_next = [torch.zeros_like(x_) for x_ in x]

        x_data = []
        for x_p, x_, x_n in zip(x_prev, x, x_next):
            x_data.append(torch.cat((x_p, x_, x_n), dim=0))

        # Reshape last dimension to be (30, 11)
        y[6] = F.pad(y[6], (0, 0, 0, 30 - y[6].shape[0]))
        # del x[4]
        # del x[3]
        # del x[1]
        # del y[5]
        # del y[3]
        # del y[1]
        del y[0]
        for i in range(5):
            y[i] *= 255.0
            x_data[i] *= 255.0
        return x_data, y, index


def fetch_data_loader(config: Config, shuffle: bool = True, type_: str = 'train'):
    """

    Args:
        type_ ():
        config ():
        shuffle (object):
    """
    if type_ == 'train':
        train_dataset = SMLMDataset(config.train_dir, config)
        val_dataset = SMLMDataset(config.val_dir, config)
        train_dataset = Subset(train_dataset, [3, 6, 13, 23, 24, 30, 37, 39, 40, 43, 55, 56, 61, 65, 66, 72, 91])
        config.log_param("num_training", len(train_dataset))
        config.log_param("num_validation", len(val_dataset))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                                  pin_memory=True)
        valid_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                                  pin_memory=True)

        return train_loader, valid_loader
    elif type_ == 'test':
        test_dir = os.path.join(config.input_dir, "test")
        test_dataset = SMLMDataset(test_dir, config, "test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=shuffle)

        return test_loader
    elif type_ == 'inference':
        inference_dir = os.path.join(config.input_dir, "inference")
        inference_dir = '/data/golam/dnam_nn/microtuble'
        inference_dataset = SMLMDataset(inference_dir, config, "inference")
        inference_loader = DataLoader(inference_dataset, batch_size=config.batch_size, shuffle=shuffle)

        return inference_loader


if __name__ == '__main__':
    config_ = Config('config.yaml')
    dl = fetch_data_loader(config_, type_='inference')
    x_, y_ = next(iter(dl))
