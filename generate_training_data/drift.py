import numpy as np
import torch

torch.manual_seed(1234)
np.random.seed(1234)
torch.use_deterministic_algorithms(True)


def get_drift(config):
    # linear/random_walk
    drift_method = config.drift_method
    # Drift for x direction for each 1000 frame
    drift_x = config.drift_x
    # Drift on y direction for each 1000 frame
    drift_y = config.drift_y
    # single frame drift
    single_drift_x = torch.tensor(drift_x / 1000, device=config.device)
    single_drift_y = torch.tensor(drift_y / 1000, device=config.device)
    # If the frame size is huge then it's efficient in GPU. Other wise it's better to take the sample from CPU

    if drift_method == "linear":
        frame_drift = torch.stack((torch.full((config.frames,), single_drift_x),
                                   torch.full((config.frames,), single_drift_y)), dim=1)
    else:
        # Default value is random walk
        mean = torch.tensor(0., device=config.device)
        sigma = torch.tensor([single_drift_x, single_drift_y]).to(config.device)

        frame_drift = torch.distributions.normal.Normal(mean, sigma).sample((config.frames, ))

    drifts = torch.cumsum(frame_drift, dim=0)
    # Save drift
    with open(config.output_file + "_drift.csv", "w") as drift_file:
        drift_file.write("dx, dy\n")
        np.savetxt(drift_file, drifts.numpy(), '%s', ',')

    return drifts


if __name__ == '__main__':
    from read_config import Config
    config = Config("config.yaml")

    drift = get_drift(config)
