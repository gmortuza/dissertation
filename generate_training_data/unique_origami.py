import numpy as np
import torch


def get_unique_origami(config):
    """
    Generate the random origami
    :return: origami_counter --> How many times each of the origami appeared
    """
    x_distance, y_distance = config.distance_x, config.distance_y
    row, column = config.origami_row, config.origami_column
    num_total_origami, num_unique_origami = config.total_origami, config.unique_origami

    x = torch.arange(0, x_distance * column, x_distance)
    y = torch.arange(0, y_distance * row, y_distance)
    mesh_x, mesh_y = torch.meshgrid(x, y)
    mesh_x, mesh_y = mesh_x.ravel(), mesh_y.ravel()
    sample_origami = torch.randint(2, size=(num_unique_origami, row * column))
    np.savetxt(config.output_file + "_gt-origami.txt", sample_origami.numpy(), fmt='%i')
    sample_origami = sample_origami.to(torch.bool)
    # sample_origami = np.ones_like(sample_origami, dtype=np.int).astype(np.bool)
    unique_origamies = {}
    # create the unique origami
    for origami_id, single_origami in enumerate(sample_origami):
        single_origami_x, single_origami_y = mesh_x[single_origami], mesh_y[single_origami]
        # Move center of the mass to the origin
        if config.origami_mean:
            single_origami_x = single_origami_x - torch.mean(single_origami_x)
            single_origami_y = single_origami_y - torch.mean(single_origami_y)
        # Convert pixel to nanometer
        single_origami_x = single_origami_x / config.Camera_Pixelsize
        single_origami_y = single_origami_y / config.Camera_Pixelsize

        unique_origamies[origami_id] = {}
        unique_origamies[origami_id]["x_cor"] = single_origami_x
        unique_origamies[origami_id]["y_cor"] = single_origami_y
    return unique_origamies


if __name__ == '__main__':
    from read_config import Config
    configuration = Config("config.yaml")

    unique_origami = get_unique_origami(configuration)
    print(unique_origami)
