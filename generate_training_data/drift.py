import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)


def get_drift(config):
    # linear/random_walk
    drift_method = config.drift_method
    # Drift for x direction for each 1000 frame
    drift_x = config.drift_x
    # Drift on y direction for each 1000 frame
    drift_y = config.drift_y
    # single frame drift
    single_drift_x = drift_x / 1000
    single_drift_y = drift_y / 1000
    # If the frame size is huge then it's efficient in GPU. Other wise it's better to take the sample from CPU

    if drift_method == "linear":
        frame_drift_x, frame_drift_y = torch.full((config.Frames,), single_drift_x, device=config.device), \
                                       torch.full((config.Frames,), single_drift_y, device=config.device)
    else:
        # Default value is random walk
        mean = torch.tensor(0., device=config.device)
        single_drift_x = torch.tensor(single_drift_x, device=config.device)
        single_drift_y = torch.tensor(single_drift_y, device=config.device)

        frame_drift_x = torch.distributions.normal.Normal(mean, single_drift_x).sample((config.Frames, ))
        frame_drift_y = torch.distributions.normal.Normal(mean, single_drift_y).sample((config.Frames, ))

    frame_drift_x = torch.cumsum(frame_drift_x, dim=0)
    frame_drift_y = torch.cumsum(frame_drift_y, dim=0)
    # Save drift
    with open(config.output_file + "_drift.csv", "w") as drift_file:
        drift_file.write("dx, dy\n")
        np.savetxt(drift_file, np.transpose(np.vstack((frame_drift_x.numpy(), frame_drift_y.numpy()))), '%s', ',')

    return frame_drift_x, frame_drift_y


if __name__ == '__main__':
    from read_config import Config
    config = Config("config.yaml")

    drift_x, drift_y = get_drift(config)
