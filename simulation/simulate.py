import torch
import numpy as np
from simulation.unique_origami import get_unique_origami
from utils import convert_device


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

        structure += grid_pos[i]

        structures = torch.cat(tensors=(structures, structure), dim=1)

    # Choose some random position from the whole movie to put the always on event
    if config.num_gold_nano_particle > 0:
        uniform_distribution = torch.distributions.uniform.Uniform(
            low=config.frame_padding, high=config.image_size - config.frame_padding)
        fixed_structure = uniform_distribution.sample((2, config.num_gold_nano_particle))
        structures = torch.cat(tensors=(structures, fixed_structure), dim=1)

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
    return structure[:, (np.random.rand(structure.shape[1]) < binding_site_incorporation)]


def generate_grid_pos(config) -> torch.Tensor:
    """
    Generate a set of positions where structures will be placed
    """
    number, image_size, frame_padding, arrangement = convert_device([config.total_origami, config.image_size,
                                                                     config.frame_padding, config.origami_arrangement],
                                                                    config.device)

    if arrangement == 0:
        spacing = int(torch.ceil((number ** 0.5)))
        lin_pos = torch.linspace(frame_padding, image_size - frame_padding, spacing, device=config.device)
        [xx_grid_pos, yy_grid_pos] = torch.meshgrid(lin_pos, lin_pos)
        xx_grid_pos = torch.ravel(xx_grid_pos)
        yy_grid_pos = torch.ravel(yy_grid_pos)
        xx_pos = xx_grid_pos[0:number]
        yy_pos = yy_grid_pos[0:number]
        grid_pos = torch.vstack((xx_pos, yy_pos))
        grid_pos = torch.transpose(grid_pos, 0, 1)
    else:
        # TODO: Need to check if this is working or not
        grid_pos = (image_size - 2 * frame_padding) * torch.rand(number, 2) + frame_padding

    return grid_pos.view(-1, 2, 1)  # [total_origami, [x_pos], [y_pos]]


def distribute_photons_single_binding_site(binding_site_id, config, num_of_binding_site):
    # TODO: Convert numpy array to tensor
    mean_dark = config.tau_d
    mean_bright = config.PAINT_tau_b
    frames = config.frame_to_generate
    time = config.Camera_integration_time
    photon_budget = config.Imager_Photonbudget
    photon_rate_std = config.Imager_photon_rate_std
    photon_rate = config.Imager_photon_rate

    # The last ids are for always on binding site -- gold nano particle
    always_on = config.photons_for_each_gold_nano_particle if \
        num_of_binding_site - binding_site_id <= int(config.num_gold_nano_particle) else 0

    # This method will be called from the multiprocess pool
    # total_acquisition_time = frame * integration_time
    # total time for a single binding event = mean_dark + mean_bright
    # total number of event = total_acquisition_time / total time for a single binding event
    num_of_blinking_event = 4 * int(
        np.ceil(frames * time / (mean_dark + mean_bright))
    )  # This is an estimate for the total number of binding events
    if num_of_blinking_event < 10:
        num_of_blinking_event = num_of_blinking_event * 10

    if always_on > 0:
        # return it with id
        return binding_site_id, torch.distributions.normal.\
            Normal(torch.tensor(always_on / frames), torch.tensor(photon_rate_std)).sample((frames, )).clip(min=0)
    dark_times = np.random.exponential(mean_dark, num_of_blinking_event)
    bright_times = np.random.exponential(mean_bright, num_of_blinking_event)

    # dark time will be followed by the bright times
    events = np.vstack((dark_times, bright_times)).reshape(
        (-1,), order="F"
    )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    # track the total time
    event_sum = np.cumsum(events)
    # Find the first event that exceeds the total integration time
    # will not use the event that surpass the total integration time
    max_loc = np.argmax(
        event_sum > (frames * time)
    )
    # an on-event might be longer than the movie, so allocate more memory
    photons_in_frame = np.zeros(
        int(frames + np.ceil(mean_bright / time * 20))
    )
    # we will only use events time that are within our total acquisition time (frames * integration time) limit
    # event sum is combine of dark and bright event. One binding event will complete by combining both these event
    for i in range(1, max_loc, 2):
        if photon_rate_std == 0:
            photons = np.round(photon_rate * time)
        else:
            photons = np.round(
                np.random.normal(photon_rate, photon_rate_std) * time
            )  # Number of Photons that are emitted in one frame

        photons = max(photons, 0)

        # use the event time stamp to see on which frame that event happened
        tempFrame = int(
            np.floor(event_sum[i - 1] / time)
        )
        # this particular blinking event will be active for #on_frames frames continuously
        # Number of frames in which photon emittance happens
        on_frames = int(
            np.ceil((event_sum[i] - tempFrame * time) / time)
        )
        # if total budget exceed for a particular binding event then reduce number of on frame
        if photons * on_frames > photon_budget:
            on_frames = int(
                np.ceil(photon_budget / (photons * on_frames) * on_frames)
            )  # Reduce the number of on-frames if the photon_budget is reached
        # assign photons for each of the on frames for this particular binding event
        for j in range(0, on_frames):
            # assign photons on each frame based on the remaining event sum for that frame
            if (on_frames < 3 and j == 0) or (on_frames >= 3 and j == on_frames - 1):
                # first frame
                frame_photons = ((tempFrame + 1) * time - event_sum[i - 1]) / time * photons
            elif (on_frames < 3 and j == 1) or (on_frames >= 3 and j == on_frames - 2):
                # second frame
                frame_photons = (event_sum[i] - (tempFrame + 1) * time) / time * photons
            else:
                # frame size is greater than 2 and at initial frames all the frames will be assigned entire photons
                # last two frame will have photon assigned based on remaining event sum
                frame_photons = photons

            photons_in_frame[tempFrame + j + 1] = int(np.random.poisson(frame_photons))

        total_photons = np.sum(
            photons_in_frame[1 + tempFrame: tempFrame + 1 + on_frames]
        )
        if total_photons > photon_budget:
            photons_in_frame[on_frames + tempFrame] = int(
                photons_in_frame[on_frames + tempFrame]
                - (total_photons - photon_budget)
            )

    return binding_site_id, torch.tensor(photons_in_frame[0:frames])


def get_binding_site_position_distribution(binding_site_position, config):
    scales = [i / 32 for i in [32, 63, 125, 249]]
    binding_site_distributions = {}
    binding_site_position = binding_site_position.T
    for scale in scales:
        binding_site_distribution = []
        scale_tril = get_scale_tril(config)
        scaled_binding_site_position = binding_site_position * scale
        for mu in scaled_binding_site_position:
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, scale_tril=scale_tril)
            binding_site_distribution.append(dist)
        binding_site_distributions[scale] = binding_site_distribution
    return binding_site_distributions


def dist_photons_xy(binding_site_position_distribution, distributed_photon, frame_id, frame_started,
                    frame_wise_noise, drifts):
    device = distributed_photon.device
    scales = [i / 32 for i in [32, 63, 125, 249]]
    temp_photons = distributed_photon[:, frame_id]  # Shape (number of binding event, )
    n_photons = torch.sum(temp_photons).item()  # Total photons for this frame
    n_photons_step = torch.cumsum(temp_photons, dim=0).to(torch.int)

    photons_pos_frame = {}
    for scale in scales:
        photons_pos_frame[scale] = torch.zeros((int(n_photons), 2), device=device)
    # Positions where are putting some photons
    # indices that will have blinking event at this frame
    gt_positions = torch.flatten(torch.where(distributed_photon[:, frame_id] > 0)[0])
    # 0,       , 1, 2, 3     , 4     , 5        , 6,       , 7,     , 8,   9,  10
    # frame_num, x, y, x_mean, y_mean, x_drifted, y_drifted, photons, s_x, s_y, noise
    # This will contain the gt_id, x_mean, y_mean, photons, sx, sy, noise, x, y
    gt_infos = torch.zeros((len(gt_positions), 11), device=device)
    gt_infos[:, 0] = torch.tensor([frame_id] * len(gt_positions))

    for i, gt_position in enumerate(gt_positions.tolist()):
        photon_count = int(distributed_photon[gt_position, frame_id])
        for scale, distribution in binding_site_position_distribution.items():
            multi_norm_dist = distribution[gt_position]
            start = 0 if gt_position == 0 else n_photons_step[gt_position - 1]
            photons_pos_frame[scale][start: n_photons_step[gt_position], :] = multi_norm_dist.sample(sample_shape=torch.Size([photon_count]))

        photon_pos = photons_pos_frame[1.0][start: n_photons_step[gt_position], :]
        # set x, y
        gt_infos[i, 1:3] = multi_norm_dist.mean  # This is super exact position

        gt_infos[i, 3:5] = photon_pos.mean(axis=0)
        gt_infos[i, 5:7] = gt_infos[i, 3:5] + drifts[frame_id]  # Will add drift in later method
        gt_infos[i, 7] = distributed_photon[gt_position, frame_id]  # Photon count
        gt_infos[i, 8:10] = photon_pos.std(axis=0)
        # Extract the ground truth for this frame at this location
        gt_infos[i, 10] = frame_wise_noise[frame_id - frame_started]

    return photons_pos_frame, gt_infos


def convert_frame(frame_id, frame_started, config, drifts, distributed_photon, frame_wise_noise,
                  binding_site_position_distribution):
    photon_pos_frames, gt_infos = dist_photons_xy(binding_site_position_distribution, distributed_photon, frame_id,
                                                 frame_started, frame_wise_noise, drifts)

    single_frames = []

    for scale, photon_pos_frame in photon_pos_frames.items():
        frame_size = int(32 * scale)
        if len(photon_pos_frame) == 0:
            # There is no photon allocated in this frame
            # So we will return empty image
            single_frame = torch.zeros((frame_size, frame_size), device=config.device)
            single_frames.append(single_frame)
        else:
            # TODO: fix drift
            samples = (photon_pos_frame + drifts[frame_id]).cpu().numpy()
            single_frame, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=(range(frame_size + 1),
                                                                                                  range(frame_size + 1)))
            single_frame = torch.from_numpy(single_frame).to(config.device)
            single_frames.append(single_frame)

    return frame_id, single_frames, gt_infos


def get_scale_tril(config):
    cov = torch.tensor([[config.Imager_PSF * config.Imager_PSF, 0],
                        [0, config.Imager_PSF * config.Imager_PSF]]).to(config.device)
    return torch.linalg.cholesky(cov)
