import sys
sys.path.append("/data/golam/dnam_nn")
from read_config import Config
import pickle
import time
from extract_location import point_extractor
import torch.nn.functional as F
from metrics import metrics
import os
import torch

def print_results(frames, frame_numbers, gt_points, method_name: str, method, config: Config):
    start_time = time.time()
    predicted_points = []
    if method_name == 'nn':
        predicted_points = point_extractor.get_point_nn(frames, config, frame_numbers)
    else:
        for frame_number, frame in zip(frame_numbers, frames):
            _, predicted_point = method(frame, config, frame_number)
            predicted_points.extend(predicted_point)
    predicted_points = torch.tensor(predicted_points)
    gt_points = torch.tensor(gt_points)
    ji, rmse, efficiency = metrics.get_ji_rmse_efficiency_from_formatted_points(predicted_points, gt_points)
    total_time = time.time() - start_time
    # print("==" * 10, f" {method_name} (", round(total_time, 2), 'second)', "==" * 10)
    print(f"{ji}\t{rmse}\t{efficiency}")
    # print(f"JI: {ji}\t, RMSE: {rmse}\t, Efficiency: {efficiency}")


def comparison(methods: list, config):
    gt_points = []

    frames = []
    frame_numbers = []
    # Read data
    for frame_number in range(1000):
        f_name = os.path.join(config.train_dir, f"db_{frame_number}.pl")
        # f_name = f"simulated_data_multi/validation/db_{frame_number}.pl"
        with open(f_name, 'rb') as handle:
            x, y = pickle.load(handle)
        y_gt, frame = y[-1], y[-3]
        y_gt = point_extractor.get_points_from_gt(y_gt, config)
        gt_points.extend(y_gt)
        frames.append(frame)
        frame_numbers.append(frame_number)
    for method_name, method in methods.items():
        print_results(frames, frame_numbers, gt_points, method_name, method, config)

if __name__ == '__main__':
    methods_ = {
        # 'nn': point_extractor.get_point_nn,
        # 'picasso': point_extractor.get_point_picasso,
        'weighted_mean': point_extractor.get_point_weighted_mean,
        # 'scipy': point_extractor.get_point_scipy
    }
    config = Config("../config.yaml")
    config.output_threshold = 0
    file_names = ['08', '09', '1', '11', '12', '13', '14', '15']
    # for file in file_names:
    config.input_dir = 'simulated_data_multi_' + file_names[0]
    import sys
    sys.stdout = open(config.input_dir, 'w')
    print(config.input_dir)
    for i in range(30, 250):
        config.multi_emitter_threshold = i
        comparison(methods_, config)
    sys.stdout.close()