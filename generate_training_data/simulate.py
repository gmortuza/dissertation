"""
    picasso.simulate
    ~~~~~~~~~~~~~~~~

    Simulate single molecule fluorescence data

    :author: Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import numpy as np
import io_modified as _io
import torch

magnification_factor = 0.79


def saveInfo(filename, info):
    _io.save_info(filename, [info], default_flow_style=True)


def noisy(image, mu, sigma):
    """
    Add gaussian noise to an image.
    """
    row, col = image.shape  # Variance for np.random is 1
    gauss = sigma * np.random.normal(0, 1, (row, col)) + mu
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy[noisy < 0] = 0
    return noisy, gauss


def noisy_p(image, mu):
    """
    # Add poissonian noise to an image or movie
    """
    poisson = np.random.poisson(mu, image.shape).astype(float)
    nosiy_image = image + poisson
    return nosiy_image, poisson


def add_noise(noise_type, movie, mu=None, sigma=None):
    # convert image to Tensor
    # TODO: Remove this later When everything will be based on torch
    image = torch.Tensor(movie)
    if noise_type == 'poisson':
        noise = torch.distributions.poisson.Poisson(image).sample()
    elif noise_type == 'gaussian':
        noise = mu + torch.randn_like(image) * sigma

    return (image + noise).numpy(), noise.numpy()


def check_type(movie):
    movie[movie >= (2 ** 16) - 1] = (2 ** 16) - 1
    movie = movie.astype("<u2")  # little-endian 16-bit unsigned int
    return movie


def generate_paint(
        mean_dark, mean_bright, frames, time, photon_rate, photon_rate_std, photon_budget, always_on
):
    """
    Paint-Generator:
    Generates on and off-traces for given parameters.
    Calculates the number of Photons in each frame for a binding site.
    """
    num_of_blinking_event = 4 * int(
        np.ceil(frames * time / (mean_dark + mean_bright))
    )  # This is an estimate for the total number of binding events
    if num_of_blinking_event < 10:
        num_of_blinking_event = num_of_blinking_event * 10

    if always_on > 0:
        return np.random.normal(always_on / frames, photon_rate_std, frames).clip(min=0), \
               np.zeros(frames, dtype=np.float64), [0] * 4

    dark_times = np.random.exponential(mean_dark, num_of_blinking_event)
    bright_times = np.random.exponential(mean_bright, num_of_blinking_event)

    events = np.vstack((dark_times, bright_times)).reshape(
        (-1,), order="F"
    )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    event_sum = np.cumsum(events)
    max_loc = np.argmax(
        event_sum > (frames * time)
    )  # Find the first event that exceeds the total integration time
    simulated_mean_dark = np.mean(events[:max_loc:2])

    simulated_mean_bright = np.mean(events[1:max_loc:2])

    # check trace
    if np.mod(max_loc, 2):  # uneven -> ends with an OFF-event
        on_events = int(np.floor(max_loc / 2))
    else:  # even -> ends with bright event
        on_events = int(max_loc / 2)
    bright_events = np.floor(max_loc / 2)  # number of bright_events

    photons_in_frame = np.zeros(
        int(frames + np.ceil(mean_bright / time * 20))
    )  # an on-event might be longer than the movie, so allocate more memory

    # calculate photon numbers
    for i in range(1, max_loc, 2):
        if photon_rate_std == 0:
            photons = np.round(photon_rate * time)
        else:
            photons = np.round(
                np.random.normal(photon_rate, photon_rate_std) * time
            )  # Number of Photons that are emitted in one frame

        if photons < 0:
            photons = 0

        tempFrame = int(
            np.floor(event_sum[i - 1] / time)
        )  # Get the first frame in which something happens in on-event
        on_frames = int(
            np.ceil((event_sum[i] - tempFrame * time) / time)
        )  # Number of frames in which photon emittance happens

        if photons * on_frames > photon_budget:
            on_frames = int(
                np.ceil(photon_budget / (photons * on_frames) * on_frames)
            )  # Reduce the number of on-frames if the photon_budget is reached

        for j in range(0, on_frames):
            if on_frames == 1:  # CASE 1: all photons are emitted in one frame
                photons_in_frame[1 + tempFrame] = int(
                    np.random.poisson(
                        ((tempFrame + 1) * time - event_sum[i - 1])
                        / time
                        * photons
                    )
                )
            elif (
                    on_frames == 2
            ):  # CASE 2: all photons are emitted in two frames
                if j == 1:  # photons in first on frame
                    photons_in_frame[1 + tempFrame] = int(
                        np.random.poisson(
                            ((tempFrame + 1) * time - event_sum[i - 1])
                            / time
                            * photons
                        )
                    )
                else:  # photons in second on frame
                    photons_in_frame[2 + tempFrame] = int(
                        np.random.poisson(
                            (event_sum[i] - (tempFrame + 1) * time)
                            / time
                            * photons
                        )
                    )
            else:  # CASE 3: all photons are emitted in three or more frames
                if j == 1:
                    photons_in_frame[1 + tempFrame] = int(
                        np.random.poisson(
                            ((tempFrame + 1) * time - event_sum[i - 1])
                            / time
                            * photons
                        )
                    )  # Indexing starts with 0
                elif j == on_frames:
                    photons_in_frame[on_frames + tempFrame] = int(
                        np.random.poisson(
                            (event_sum(i) - (tempFrame + on_frames - 1) * time)
                            / time
                            * photons
                        )
                    )
                else:
                    photons_in_frame[tempFrame + j] = int(
                        np.random.poisson(photons)
                    )

        total_photons = np.sum(
            photons_in_frame[1 + tempFrame: tempFrame + 1 + on_frames]
        )
        if total_photons > photon_budget:
            photons_in_frame[on_frames + tempFrame] = int(
                photons_in_frame[on_frames + tempFrame]
                - (total_photons - photon_budget)
            )

    photons_in_frame = photons_in_frame[0:frames]
    time_trace = events[0:max_loc]

    if on_events > 0:
        spot_kinetics = [
            on_events,
            sum(photons_in_frame > 0),
            simulated_mean_dark,
            simulated_mean_bright,
        ]
    else:
        spot_kinetics = [0, sum(photons_in_frame > 0), 0, 0]
    return photons_in_frame, time_trace, spot_kinetics


def distribute_photons(
        structures,
        integration_time,
        frames,
        taud,
        taub,
        photon_rate,
        photon_rate_std,
        photon_budget,
        always_on=0,
):
    """
    Distribute Photons
    """
    photons_in_frame, time_trace, spot_kinetics = generate_paint(
        int(taud),  # mean dark
        int(taub),  # mean bright
        frames,
        integration_time,
        photon_rate,
        photon_rate_std,
        photon_budget,
        always_on
    )

    return photons_in_frame, time_trace, spot_kinetics


def dist_photons_xy(runner, photon_dist, structures, psf):
    binding_sites_x = structures[0, :]
    binding_sites_y = structures[1, :]
    no_sites = len(binding_sites_x)  # number of binding sites in image

    temp_photons = np.array(photon_dist[:, runner]).astype(int)
    n_photons = np.sum(temp_photons)
    n_photons_step = np.cumsum(temp_photons)
    n_photons_step = np.insert(n_photons_step, 0, 0)

    # Allocate memory
    photon_pos_frame = np.zeros((n_photons, 2))
    # Positions where are putting some photons
    # indices that will have blinking event at this frame
    gt_position = np.argwhere(photon_dist[:, runner] > 0).flatten()
    for i in gt_position:
        photon_count = int(photon_dist[i, runner])
        # covariance matrix for the normal distribution
        cov = [[psf * psf, 0], [0, psf * psf]]
        mu = [binding_sites_x[i], binding_sites_y[i]]
        photon_pos = np.random.multivariate_normal(mu, cov, photon_count)
        photon_pos_frame[n_photons_step[i]: n_photons_step[i + 1], :] = photon_pos

    return photon_pos_frame, gt_position


def convertMovie(
        runner,
        photon_dist,
        config,
):
    edges = range(0, config["Height"] + 1)

    photon_pos_frame, gt_position = dist_photons_xy(runner, photon_dist, config["new_struct"], config["Imager_PSF"])

    if len(photon_pos_frame) == 0:
        # There is no photon allocated in this frame
        # So we will return empty image
        return np.zeros((config["Height"], config["Width"])), gt_position
    else:
        # Insert the drift in here
        x = photon_pos_frame[:, 0] + config["drift_x"][runner]
        y = photon_pos_frame[:, 1] + config["drift_y"][runner]
        simulated_frame, _, _ = np.histogram2d(y, x, bins=(edges, edges))
        return simulated_frame, gt_position
        # Because of this flip ground truth doesn't match with the render text
        # The origami might be fippled anyway. So this doesn't matter
        # Disabling this to have the exact match of the ground truth vs simulated data
        # simulated_frame = np.flipud(simulated_frame)  # to be consistent with render


def saveMovie(filename, movie, info):
    _io.save_raw(filename, movie, [info])


# Function to store the coordinates of a structure in a container.
# The coordinates wil be adjusted so that the center of mass is the origin
def defineStructure(
        structure_xx_px,
        structure_yy_px,
        structure_ex,
        structure3d,
        pixel_size,
        mean=True,
):
    if mean:
        structure_xx_px = structure_xx_px - np.mean(structure_xx_px)
        structure_yy_px = structure_yy_px - np.mean(structure_yy_px)
    # from px to nm
    structure_xx = []
    for x in structure_xx_px:
        structure_xx.append(x / pixel_size)
    structure_yy = []
    for x in structure_yy_px:
        structure_yy.append(x / pixel_size)

    structure = np.array(
        [structure_xx, structure_yy, structure_ex, structure3d]
    )  # FORMAT: x-pos,y-pos,exchange information

    return structure


def generatePositions(number, image_size, frame, arrangement):
    """
    Generate a set of positions where structures will be placed
    """
    if arrangement == 0:
        spacing = int(np.ceil((number ** 0.5)))
        lin_pos = np.linspace(frame, image_size - frame, spacing)
        [xx_grid_pos, yy_grid_pos] = np.meshgrid(lin_pos, lin_pos)
        xx_grid_pos = np.ravel(xx_grid_pos)
        yy_grid_pos = np.ravel(yy_grid_pos)
        xx_pos = xx_grid_pos[0:number]
        yy_pos = yy_grid_pos[0:number]
        grid_pos = np.vstack((xx_pos, yy_pos))
        grid_pos = np.transpose(grid_pos)
    else:
        grid_pos = (image_size - 2 * frame) * np.random.rand(number, 2) + frame

    return grid_pos


def rotateStructure(structure):
    """
    Rotate a structure randomly
    """
    angle_rad = np.random.rand(1) * 2 * np.pi
    new_structure = np.array(
        [
            (structure[0, :]) * np.cos(angle_rad)
            - (structure[1, :]) * np.sin(angle_rad),
            (structure[0, :]) * np.sin(angle_rad)
            + (structure[1, :]) * np.cos(angle_rad),
            structure[2, :],
            structure[3, :],
        ]
    )
    return new_structure


def incorporateStructure(structure, incorporation):
    """
    Returns a subset of the structure to reflect incorporation of staples
    """
    new_structure = structure[
                    :, (np.random.rand(structure.shape[1]) < incorporation)
                    ]
    return new_structure


def randomExchange(pos):
    """
    Randomly shuffle exchange parameters for random labeling
    """
    array_to_shuffle = pos[2, :]
    np.random.shuffle(array_to_shuffle)
    new_pos = np.array([pos[0, :], pos[1, :], array_to_shuffle, pos[3, :]])
    return new_pos


def prepareStructures(
        origamies, grid_pos, orientation, number, incorporation, exchange, frame_padding, height,
        num_gold_nano_particle=0
):
    """
    prepareStructures:
    Input positions, the structure definition consider rotation etc.
    """
    for i in range(0, len(grid_pos)):
        # for each grid position select a random origami and add that origami to that grid position
        # Origami id for this particular grid position
        origami = origamies[np.random.randint(0, len(origamies))]
        old_structure = np.array(
            [origami["x_cor"], origami["y_cor"], origami["labels"], origami["3d"]]
        )
        if orientation == 0:
            structure = old_structure
        else:
            structure = rotateStructure(old_structure)

        if incorporation == 1:
            pass
        else:
            structure = incorporateStructure(structure, incorporation)

        new_x = structure[0, :] + grid_pos[i, 0]
        new_y = structure[1, :] + grid_pos[i, 1]
        new_struct = np.array(
            [
                new_x,
                new_y,
                structure[2, :],
                structure[2, :] * 0 + i,
                structure[3, :],
            ]
        )
        if i == 0:
            new_pos = new_struct
        else:
            new_pos = np.concatenate((new_pos, new_struct), axis=1)

    # Choose some random position from the whole movie to put the always on event
    if num_gold_nano_particle > 0:
        fixed_structure = np.array(
            [
                np.random.uniform(low=frame_padding, high=height - frame_padding, size=num_gold_nano_particle),
                # considering height and width the same
                np.random.uniform(low=frame_padding, high=height - frame_padding, size=num_gold_nano_particle),
                np.ones(num_gold_nano_particle),
                np.zeros(num_gold_nano_particle),
                np.zeros(num_gold_nano_particle)
            ]
        )
        new_pos = np.concatenate((new_pos, fixed_structure), axis=1)

    if exchange == 1:
        new_pos = randomExchange(new_pos)
    return new_pos
