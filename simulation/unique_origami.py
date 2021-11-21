import numpy as np
import torch


def get_unique_origami(config):
    """
    Generate the random origami
    :return:
    """
    x_distance, y_distance = config.distance_x, config.distance_y
    row, column = config.origami_row, config.origami_column
    num_total_origami, num_unique_origami = config.total_origami, config.unique_origami

    x = torch.arange(0, x_distance * column, x_distance, device=config.device)
    y = torch.arange(0, y_distance * row, y_distance, device=config.device)
    mesh_x, mesh_y = torch.meshgrid(x, y)
    mesh_x, mesh_y = mesh_x.ravel(), mesh_y.ravel()
    sample_origami = torch.randint(2, size=(num_unique_origami, row * column), device=config.device)
    # np.savetxt(config.output_file + "_gt-origami.txt", sample_origami.cpu().numpy(), fmt='%i')
    sample_origami = sample_origami.to(torch.bool)
    # sample_origami = np.ones_like(sample_origami, dtype=np.int).astype(np.bool)
    unique_origamies = []
    # TODO: remove loop do everything using tensor
    # create the unique origami
    for origami_id, single_origami in enumerate(sample_origami):
        single_origami_x, single_origami_y = mesh_x[single_origami], mesh_y[single_origami]
        # Move center of the mass to the origin
        if config.origami_mean:
            single_origami_x = single_origami_x - torch.mean(single_origami_x)
            single_origami_y = single_origami_y - torch.mean(single_origami_y)
        # Convert pixel to nanometer
        # We will create the higher resolution image first then downsample that image to
        # get our main training image
        scale = config.resolution_slap[-1] / config.resolution_slap[0]
        nm_per_pixel = config.Camera_Pixelsize / scale
        single_origami_x = single_origami_x / nm_per_pixel
        single_origami_y = single_origami_y / nm_per_pixel
        #
        unique_origamies.append(torch.stack((single_origami_x, single_origami_y)))
    return unique_origamies


if __name__ == '__main__':
    from read_config import Config
    configuration = Config("config.yaml")

    unique_origami = get_unique_origami(configuration)
    print(unique_origami)
