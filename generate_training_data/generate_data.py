import h5py
import time
from torch import multiprocessing
# from multiprocessing.pool import ThreadPool as Pool
# import multiprocessing.Pool as Pool
from torch.multiprocessing import Pool
import torch
import numpy as np
from read_config import Config
from simulation import Simulation
from noise import get_noise
from drift import get_drift
import concurrent.futures
import torch.autograd.profiler as profiler

from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

# torch.use_deterministic_algorithms(True)


def show_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(func.__name__, ": Execution time is: ", round(time.time() - start, 2), "seconds")
        return res
    return wrapper


class GenerateData(Simulation):
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.binding_site_position = None  # tensor in shape (2, num_of_event)
        self.num_of_binding_site = None
        self.distributed_photon = None  # tensor in shape(num_binding_event, total frames)
        # Initially the movie is just a random noise
        self.movie = get_noise(self.config)  # Tensor of shape (num_of_frames, height, width)
        self.frame_wise_noise = self.movie.mean((1, 2))  # Tensor of shape (num_of_frames)
        # self.movie = torch.zeros((self.config.frames, self.config.image_size, self.config.image_size),
        #                          device=self.config.device, dtype=torch.float64)
        self.drifts = get_drift(self.config)  # tensor of shape (num_of_frames, 2)
        # This will be used to sample from multivariate distribution
        self.scale_tril = self.get_scale_tril()
        # This ground truth is only for visualization
        # Normally we don't need this ground truth for training purpose
        # For training purpose we will export something tensor shape
        # TODO: export training example supporting neural network training
        self.available_cpu_core = int(np.ceil(multiprocessing.cpu_count()))


    def generate(self):
        self.binding_site_position = self.generate_binding_site_position()
        self.num_of_binding_site = self.binding_site_position.shape[1]

        self.distribute_photons()
        self.convert_into_image()
        # self.save_image()

    @show_execution_time
    def distribute_photons(self):
        self.config.logger.info("Distributing photons")
        self.distributed_photon = torch.zeros((self.num_of_binding_site, self.config.frames), device=self.config.device)
        # Number of core available
        # =========  For debugging purpose
        for site in tqdm(range(self.num_of_binding_site), desc="Distributing photons"):
            self.distribute_photons_single_binding_site(site)
        # =========
        # pool = Pool(processes=self.available_cpu_core)
        # pool.map(self.distribute_photons_single_binding_site, np.arange(self.num_of_binding_site))
        # pool.close()
        # pool.join()

    @show_execution_time
    def convert_into_image(self):
        self.config.logger.info("Converting into Images")

        # =========  For debugging purpose
        # frame_details = map(self.convert_frame, np.arange(self.config.frames))
        # =========
        # print(self.available_cpu_core)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.available_cpu_core) as executor:
            frame_details = executor.map(self.convert_frame, range(self.config.frames))
        # pool = Pool(processes=self.available_cpu_core)
        # frame_details = pool.map(self.convert_frame, np.arange(self.config.frames))
        # pool.close()
        # pool.join()

        # We save images in hdf5 for visualization purpose
        # self.save_frames_in_hdf5(frame_details)
        # we save images in tensor for training purpose
        self.save_frames_in_tensor(frame_details)

    def save_frames_in_tensor(self, frame_details):
        # concatnenating tensor in a loop might be slower
        # So we are creating a large tensor
        combined_ground_truth = torch.zeros((self.config.frames * self.config.max_number_of_emitter_per_frame, 9))
        current_num_of_emitter = 0
        for frame_id, frame, gt_infos in frame_details:
            self.movie[frame_id, :, :] += frame
            emitter_to_keep = min(len(gt_infos), self.config.max_number_of_emitter_per_frame)
            combined_ground_truth[current_num_of_emitter: current_num_of_emitter+emitter_to_keep, 1:] \
                = gt_infos[:emitter_to_keep, 1:]
            combined_ground_truth[current_num_of_emitter: current_num_of_emitter+emitter_to_keep, 0] = \
                torch.tensor([frame_id] * emitter_to_keep)
            current_num_of_emitter += emitter_to_keep
        combined_ground_truth = combined_ground_truth[: current_num_of_emitter, :]
        torch.save(combined_ground_truth, self.config.output_file + "_ground_truth.pl")
        torch.save(self.movie, self.config.output_file + ".pl")


    @show_execution_time
    def save_frames_in_hdf5(self, frame_details):
        gt_frames = []
        gt_x_without_drift = []
        gt_y_without_drift = []
        gt_x_with_drift = []
        gt_y_with_drift = []
        gt_photons = []
        gt_noise = []
        gt_sx = []
        gt_sy = []

        for (frame_id, frame, gt_infos) in frame_details:
            # putting all the frame together
            # self.movie[frame_id, :, :] += frame
            if gt_infos.any():  # gt_id, x_mean, y_mean, photons, sx, sy, noise, x, y
                # gt_pos = gt_pos if len(gt_pos) > 1 else [gt_pos]
                gt_x_without_drift.extend(gt_infos[:, 1].tolist())
                gt_y_without_drift.extend(gt_infos[:, 2].tolist())
                gt_x_with_drift.extend((gt_infos[:, 1] + self.drifts[frame_id, 0].numpy()).tolist())
                gt_y_with_drift.extend((gt_infos[:, 2] + self.drifts[frame_id, 1].numpy()).tolist())
                gt_photons.extend(gt_infos[:, 3].tolist())
                gt_noise.extend(gt_infos[:, 6].tolist())
                gt_sx.extend(gt_infos[:, 4].tolist())
                gt_sy.extend(gt_infos[:, 5].tolist())
                gt_frames.extend([frame_id] * len(gt_infos))


        gt_without_drift = np.rec.array(
            (
                np.asarray(gt_frames),
                np.asarray(gt_x_without_drift),
                np.asarray(gt_y_without_drift),
                np.asarray(gt_photons),
                np.asarray(gt_sx),
                np.asarray(gt_sy),
                np.asarray(gt_noise),  # background
                np.full_like(gt_y_without_drift, .009),  # lpx
                np.full_like(gt_y_without_drift, .009),  # lpy
            ), dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("sx", "f4"),
                ("sy", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])

        gt_with_drift = np.rec.array(
            (
                np.asarray(gt_frames),
                np.asarray(gt_x_with_drift),
                np.asarray(gt_y_with_drift),
                np.asarray(gt_photons),
                np.asarray(gt_sx),
                np.asarray(gt_sy),
                np.asarray(gt_noise),  # background
                np.full_like(gt_y_with_drift, .009),  # lpx
                np.full_like(gt_y_with_drift, .009),  # lpy
            ), dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("sx", "f4"),
                ("sy", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])

        content_for_yaml_file = f"Box Size: 7\nPixelsize: {self.config.Camera_Pixelsize}" \
                                f"\nFrames: {self.config.frames}\n" \
                                f"Height: {self.config.image_size}\n" \
                                f"Width: {self.config.image_size}"
        with h5py.File(self.config.output_file + "_gt_without_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_without_drift)
            with open(self.config.output_file + "_gt_without_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)

        with h5py.File(self.config.output_file + "_gt_with_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=gt_with_drift)
            with open(self.config.output_file + "_gt_with_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)


    def save_image(self):
        self.movie.numpy().tofile(self.config.output_file+".raw")
        # Save the info file
        # TODO: save here


if __name__ == "__main__":
    # with profiler.profile(record_shapes=True) as prof:
    #     with profiler.record_function("model_inference"):
    generate_data = GenerateData(config_file="config.yaml")
    generate_data.generate()

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
