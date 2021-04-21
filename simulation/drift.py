import numpy as np
import torch


def get_drift(config):
    # linear/random_walk
    drift_method = config.drift_method
    # Drift for x direction for each 1000 frame
    drift_x = config.drift_x
    # Drift on y direction for each 1000 frame
    drift_y = config.drift_y
    # single frame drift
    single_frame_drift = torch.tensor([drift_x, drift_y], device=config.device) / 1000

    # If the frame size is huge then it's efficient in GPU. Other wise it's better to take the sample from CPU

    if drift_method == "linear":
        frame_drift = single_frame_drift.expand(config.total_frames, 2)
    else:
        # Default value is random walk
        mean = torch.tensor([0., 0.], device=config.device)

        frame_drift = torch.distributions.normal.Normal(mean, single_frame_drift).sample((config.total_frames, ))

    drifts = torch.cumsum(frame_drift, dim=0)
    # Save drift
    with open(config.simulated_file_name + "_drift.csv", "w") as drift_file:
        drift_file.write("dx, dy\n")
        np.savetxt(drift_file, drifts.cpu().numpy(), '%s', ',')
    return drifts


if __name__ == '__main__':
    from read_config import Config
    config = Config("config.yaml")

    drift = get_drift(config)
