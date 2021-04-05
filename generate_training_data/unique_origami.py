import numpy as np


def get_unique_origami(config):
    """
    Generate the random origami
    :return: origami_counter --> How many times each of the origami appeared
    """
    x_distance, y_distance = config.distance_x, config.distance_y
    row, column = config.origami_row, config.origami_column
    num_total_origami, num_unique_origami = config.total_origami, config.unique_origami

    x = np.arange(0, x_distance * column, x_distance)
    y = np.arange(0, y_distance * row, y_distance)
    mesh_x, mesh_y = np.meshgrid(x, y)
    mesh_x, mesh_y = mesh_x.ravel(), mesh_y.ravel()
    sample_origami = np.random.randint(2, size=(num_unique_origami, row * column), dtype=np.int)
    np.savetxt(config.output_file + "_gt-origami.txt", sample_origami, fmt='%i')
    sample_origami = sample_origami.astype(np.bool)
    # sample_origami = np.ones_like(sample_origami, dtype=np.int).astype(np.bool)
    unique_origamies = {}
    # create the unique origami
    for origami_id, single_origami in enumerate(sample_origami):
        single_origami_x, single_origami_y = mesh_x[single_origami], mesh_y[single_origami]
        # Move center of the mass to the origin
        if config.origami_mean:
            single_origami_x = single_origami_x - np.mean(single_origami_x)
            single_origami_y = single_origami_y - np.mean(single_origami_y)
        # Convert pixel to nanometer
        single_origami_x = single_origami_x / config.Camera_Pixelsize
        single_origami_y = single_origami_y / config.Camera_Pixelsize

        unique_origamies[origami_id] = {}
        unique_origamies[origami_id]["x_cor"] = single_origami_x
        unique_origamies[origami_id]["y_cor"] = single_origami_y

    # Taking the random num_total_origamies here.
    # but later decided to pick random origami later
    # Keeping this Piece of code for future references
    """
    random_idx = np.random.choice(np.arange(num_unique_origami), num_total_origami)
    origami_counter = Counter(random_idx)
    unique_origami_x = np.asarray(unique_origami_x, dtype=np.object)
    unique_origami_y = np.asarray(unique_origami_y, dtype=np.object)

    total_origami_x = unique_origami_x[random_idx].ravel()
    total_origami_y = unique_origami_y[random_idx].ravel()
    """
    return unique_origamies


if __name__ == '__main__':
    from read_config import Config
    configuration = Config("config.yaml")

    unique_origami = get_unique_origami(configuration)
    print(unique_origami)
