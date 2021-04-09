import torch
import numpy as np
from unique_origami import get_unique_origami
from histogram import histogramdd
from noise import extract_noise_from_frame
import torch.autograd.profiler as profiler

torch.manual_seed(1234)
np.random.seed(1234)
# torch.use_deterministic_algorithms(True)


def generate_binding_site_position(config):
    grid_position = generate_grid_pos(config)
    unique_origamies = get_unique_origami(config)

    return get_binding_site_pos_from_origamies(config, grid_position, unique_origamies)   # in nanometer


def get_binding_site_pos_from_origamies(config, grid_pos, origamies):
    """
    place_origamies:
    Input positions, the structure definition consider rotation etc.
    """
    structures = torch.empty(2, 0, device=config.device)
    for i in range(grid_pos.shape[0]):
        # for each grid position select a random origami and add that origami to that grid position
        # Origami id for this particular grid position
        structure = origamies[np.random.randint(0, len(origamies))]
        #
        structure = rotate_structure(structure) if config.origami_orientation else structure
        #
        structure = incorporate_structure(structure, config.binding_site_incorporation)

        # TODO: shape grid pos during creating. So that we don't need to reshape it in here
        structure += grid_pos[i]

        structures = torch.cat((structures, structure), axis=1)

    # Choose some random position from the whole movie to put the always on event
    if config.num_gold_nano_particle > 0:
        uniform_distribution = torch.distributions.uniform.Uniform(
            low=config.frame_padding, high=config.image_size - config.frame_padding)
        fixed_structure = uniform_distribution.sample((2, config.num_gold_nano_particle))
        structures = torch.cat((structures, fixed_structure), axis=1)

    return structures


def rotate_structure(structure):

    angle_rad = torch.rand(1) * 2 * np.pi
    structure = torch.stack(
        [
            (structure[0, :]) * torch.cos(angle_rad) - (structure[1, :]) * torch.sin(angle_rad),
            (structure[0, :]) * torch.sin(angle_rad) + (structure[1, :]) * torch.cos(angle_rad),
        ]
    )
    return structure


def incorporate_structure(structure, binding_site_incorporation):
    """
    Returns a subset of the structure to reflect incorporation of staples
    """
    if binding_site_incorporation == 1:
        return structure
    return structure[:, (np.random.rand(structure.shape[1]) < binding_site_incorporation)]


def generate_grid_pos(config) -> torch.Tensor:
    """
    Generate a set of positions where structures will be placed
    """
    number, image_size, frame, arrangement = convert_device([config.total_origami, config.image_size, config.frames,
                                                             config.origami_arrangement], config.device)

    if arrangement == 0:
        spacing = int(torch.ceil((number ** 0.5)))
        lin_pos = torch.linspace(frame, image_size - frame, spacing, device=config.device)
        [xx_grid_pos, yy_grid_pos] = torch.meshgrid(lin_pos, lin_pos)
        xx_grid_pos = torch.ravel(xx_grid_pos)
        yy_grid_pos = torch.ravel(yy_grid_pos)
        xx_pos = xx_grid_pos[0:number]
        yy_pos = yy_grid_pos[0:number]
        grid_pos = torch.vstack((xx_pos, yy_pos))
        grid_pos = torch.transpose(grid_pos, 0, 1)
    else:
        # TODO: Need to check if this is working or not
        grid_pos = (image_size - 2 * frame) * torch.rand(number, 2) + frame

    return grid_pos.view(-1, 2, 1)  # [total_origami, [x_pos], [y_pos]]


def distribute_photons_single_binding_site(binding_site_id, config, num_of_binding_site):
    # TODO: Convert numpy array to tensor
    mean_dark = config.tau_d
    mean_bright = config.PAINT_tau_b
    frames = config.frames
    time = config.Camera_integration_time
    photon_budget = config.Imager_Photonbudget
    photon_rate_std = config.Imager_photon_rate_std
    photon_rate = config.Imager_photon_rate

    # The last ids are for always on binding site
    always_on = config.photons_for_each_gold_nano_particle if \
        num_of_binding_site - binding_site_id <= int(config.num_gold_nano_particle) else 0

    # This method will be called from the multiprocess pool
    num_of_blinking_event = 4 * int(
        np.ceil(frames * time / (mean_dark + mean_bright))
    )  # This is an estimate for the total number of binding events
    if num_of_blinking_event < 10:
        num_of_blinking_event = num_of_blinking_event * 10

    if always_on > 0:
        # return it with id
        return binding_site_id, torch.distributions.normal.Normal\
            (torch.tensor(always_on / frames), torch.tensor((photon_rate_std))).sample((frames, )).clip(min=0)
        return
    dark_times = np.random.exponential(mean_dark, num_of_blinking_event)
    bright_times = np.random.exponential(mean_bright, num_of_blinking_event)

    events = np.vstack((dark_times, bright_times)).reshape(
        (-1,), order="F"
    )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    event_sum = np.cumsum(events)
    max_loc = np.argmax(
        event_sum > (frames * time)
    )  # Find the first event that exceeds the total integration time

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

    # Update the global variable
    return binding_site_id, torch.tensor(photons_in_frame[0:frames])


def dist_photons_xy(binding_site_position, distributed_photon, frame_id, frame_wise_noise, scale_tril):
    device = binding_site_position.device
    binding_sites_x = binding_site_position[0]
    binding_sites_y = binding_site_position[1]

    temp_photons = distributed_photon[:, frame_id]
    n_photons = torch.sum(temp_photons).item()  # Total photons for this frame
    n_photons_step = torch.cumsum(temp_photons, dim=0).to(torch.int)

    # Allocate memory
    photon_pos_frame = torch.zeros((int(n_photons), 2), device=device)
    # Positions where are putting some photons
    # indices that will have blinking event at this frame
    gt_positions = torch.where(distributed_photon[:, frame_id] > 0)[0].flatten()
    # This will contain the gt_id, x_mean, y_mean, photons, sx, sy, noise, x, y
    gt_infos = torch.zeros((len(gt_positions), 9), device=device)
    gt_infos[:, 0] = gt_positions[:]

    for i, gt_position in enumerate(gt_positions.tolist()):
        photon_count = int(distributed_photon[gt_position, frame_id])
        mu = torch.tensor([binding_sites_x[gt_position], binding_sites_y[gt_position]], device=device)
        multi_norm_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, scale_tril=scale_tril)
        photon_pos = multi_norm_dist.sample(sample_shape=torch.Size([photon_count]))
        gt_infos[i, 1:3] = photon_pos.mean(axis=0)
        gt_infos[i, 3] = distributed_photon[gt_position, frame_id]  # Photon count
        gt_infos[i, 4:6] = photon_pos.std(axis=0)
        # Extract the ground truth for this frame at this location
        gt_infos[i, 6] = frame_wise_noise[frame_id]
        # TODO: Add this into the configuration file
        # gt_infos[i, 6] = extract_noise_from_frame(self.movie[frame_id], position=gt_infos[i, 1:3])
        gt_infos[i, 7:9] = mu  # This is super exact position
        start = 0 if gt_position == 0 else n_photons_step[gt_position-1]
        photon_pos_frame[start: n_photons_step[gt_position], :] = photon_pos

    return photon_pos_frame, gt_infos


def convert_frame(frame_id, config, drifts, distributed_photon, frame_wise_noise, scale_tril, binding_site_position):
    edges = torch.arange(0, config.image_size + 1, device=config.device)
    photon_pos_frame, gt_infos = dist_photons_xy(binding_site_position, distributed_photon, frame_id, frame_wise_noise, scale_tril)

    if len(photon_pos_frame) == 0:
        # There is no photon allocated in this frame
        # So we will return empty image
        single_frame = torch.zeros((config.image_size, config.image_size), device=config.device)
    else:
        samples = photon_pos_frame + drifts[frame_id]
        # The implementation is not optimized for GPU.
        # So it is better to use CPU
        single_frame, _ = histogramdd(samples.T.roll(1, 0), bins=(edges, edges))
    return frame_id, single_frame, gt_infos


def get_scale_tril(config):
    cov = torch.tensor([[config.Imager_PSF * config.Imager_PSF, 0],
                        [0, config.Imager_PSF * config.Imager_PSF]]).to(config.device)
    return torch.linalg.cholesky(cov)


def convert_device(tensors, device):
    if not tensors:
        return tensors
    elif isinstance(tensors, (int, float)):
        return torch.tensor(tensors, device=device)
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (tuple, list)):
        return [convert_device(tensor, device) for tensor in tensors]

