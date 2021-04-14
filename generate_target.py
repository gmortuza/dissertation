# Convert pickle to target dataset
import torch
from read_config import Config


def generate_target_gt(input_tensor: torch.Tensor, config: Config, start_frame: int, end_frame: int) -> torch.Tensor:
    """
    This will convert raw ground truth into neural network's target
    input_tensor is of shape: (None, 11)
    Each of them will contain
    frame_num, x, y, x_mean, y_mean, x_drifted, y_drifted, photons, s_x, s_y, noise
    :param input_tensor:
    :return:
    """
    number_of_frame = end_frame - start_frame
    target_tensor = torch.zeros(size=(number_of_frame, config.max_number_of_emitter_per_frame, 6), device=config.device)
    for frame_id in input_tensor[:, 0].unique():
        frame_gts = input_tensor[input_tensor[:, 0] == frame_id]
        target_tensor[int(frame_id) - start_frame, :, :len(frame_gts)] = frame_gts[:, [3, 4, 7, 8, 9, 10]]
    return target_tensor


def generate_target_from_path(path: str, config_path: Config):
    config = Config(config_path)
    input_tensor = torch.load(path)
    path_arr = path.replace(".pl", "").split("_")
    start_frame, end_frame = int(path_arr[-2]) - 1, int(path_arr[-1])
    generate_target_gt(input_tensor, config, start_frame, end_frame)


if __name__ == '__main__':
    path = "simulated_data/test_ground_truth_1_5000.pl"
    config_path = "config.yaml"
    generate_target_from_path(path, config_path)
