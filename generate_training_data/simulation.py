import torch
import numpy as np
from unique_origami import get_unique_origami

torch.manual_seed(1234)
np.random.seed(1234)


class Simulation:
    def __init__(self):
        pass

    def generate_binding_site_position(self):
        grid_position = self.generate_grid_pos(self.config.total_origami, self.config.image_size,
                                               self.config.frame_padding, self.config.origami_arrangement)
        unique_origamies = get_unique_origami(self.config)

        return self.get_binding_site_pos_from_origamies(grid_position, unique_origamies)   # in nanometer

    def get_binding_site_pos_from_origamies(self, grid_pos, origamies):
        """
        place_origamies:
        Input positions, the structure definition consider rotation etc.
        """
        structures = torch.empty(2, 0)
        for i in range(grid_pos.shape[0]):
            # for each grid position select a random origami and add that origami to that grid position
            # Origami id for this particular grid position
            origami = origamies[np.random.randint(0, len(origamies))]
            structure = torch.tensor([origami["x_cor"], origami["y_cor"]])
            #
            structure = self.rotate_structure(structure)
            #
            structure = self.incorporate_structure(structure, )

            structure[0] += grid_pos[i, 0]
            structure[1] += grid_pos[i, 1]

            structures = np.concatenate((structures, structure), axis=1)

        # Choose some random position from the whole movie to put the always on event
        if self.config.num_gold_nano_particle > 0:
            fixed_structure = np.array(
                [
                    np.random.uniform(low=self.config.frame_padding, high=self.config.image_size - self.config.frame_padding, size=self.config.num_gold_nano_particle),
                    # considering height and width the same
                    np.random.uniform(low=self.config.frame_padding, high=self.config.image_size - self.config.frame_padding, size=self.config.num_gold_nano_particle),
                ]
            )
            structures = np.concatenate((structures, fixed_structure), axis=1)

        return structures

    def rotate_structure(self, structure):
        if not self.config.origami_orientation:
            return structure

        angle_rad = np.random.rand(1) * 2 * np.pi
        structure = torch.tensor(
            [
                (structure[0, :]) * np.cos(angle_rad)
                - (structure[1, :]) * np.sin(angle_rad),
                (structure[0, :]) * np.sin(angle_rad)
                + (structure[1, :]) * np.cos(angle_rad),
                structure[2, :],
                structure[3, :],
            ]
        )
        return structure

    def incorporate_structure(self, structure):
        """
        Returns a subset of the structure to reflect incorporation of staples
        """
        if self.config.binding_site_incorporation == 1:
            return structure
        return structure[:, (np.random.rand(structure.shape[1]) < self.config.binding_site_incorporation)]

    @staticmethod
    def generate_grid_pos(number: int, image_size: int, frame: int, arrangement: int) -> torch.Tensor:
        """
        Generate a set of positions where structures will be placed
        """
        if arrangement == 0:
            spacing = int(np.ceil((number ** 0.5)))
            lin_pos = torch.linspace(frame, image_size - frame, spacing)
            [xx_grid_pos, yy_grid_pos] = torch.meshgrid(lin_pos, lin_pos)
            xx_grid_pos = torch.ravel(xx_grid_pos)
            yy_grid_pos = torch.ravel(yy_grid_pos)
            xx_pos = xx_grid_pos[0:number]
            yy_pos = yy_grid_pos[0:number]
            grid_pos = torch.vstack((xx_pos, yy_pos))
            grid_pos = torch.transpose(grid_pos, 0, 1)
        else:
            grid_pos = (image_size - 2 * frame) * torch.rand(number, 2) + frame

        return grid_pos

    def distribute_photons_single_binding_site(self, binding_site_id):
        # TODO: Convert numpy array to tensor
        mean_dark = self.config.tau_d
        mean_bright = self.config.PAINT_tau_b
        frames = self.config.Frames
        time = self.config.Camera_integration_time
        photon_budget = self.config.Imager_Photonbudget
        photon_rate_std = self.config.Imager_photon_rate_std
        photon_rate = self.config.Imager_photon_rate

        # The last ids are for always on binding site
        always_on = self.config.photons_for_each_gold_nano_particle if \
            self.num_of_binding_site - binding_site_id <= int(self.config.num_gold_nano_particle) else 0

        # This method will be called from the multiprocess pool
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
        self.distributed_photon[binding_site_id] = torch.tensor(photons_in_frame[0:frames])

    def dist_photons_xy(self, frame_id):
        binding_sites_x = self.binding_site_position[0]
        binding_sites_y = self.binding_site_position[1]

        temp_photons = np.array(self.distributed_photon[:, frame_id]).astype(int)
        n_photons = np.sum(temp_photons)  # Total photons for this frame
        n_photons_step = np.cumsum(temp_photons)
        n_photons_step = np.insert(n_photons_step, 0, 0)

        # Allocate memory
        photon_pos_frame = np.zeros((n_photons, 2))
        # Positions where are putting some photons
        # indices that will have blinking event at this frame
        gt_position = np.argwhere(self.distributed_photon[:, frame_id] > 0).flatten()
        for i in gt_position:
            photon_count = int(self.distributed_photon[i, frame_id])
            # covariance matrix for the normal distribution
            cov = [[self.config.Imager_PSF * self.config.Imager_PSF, 0], [0, self.config.Imager_PSF * self.config.Imager_PSF]]
            mu = [binding_sites_x[i], binding_sites_y[i]]
            photon_pos = np.random.multivariate_normal(mu, cov, photon_count)
            photon_pos_frame[n_photons_step[i]: n_photons_step[i + 1], :] = photon_pos

        return photon_pos_frame, gt_position

    def convert_frame(self, frame_id):
        edges = range(0, self.config.image_size + 1)

        photon_pos_frame, gt_position = self.dist_photons_xy(frame_id)

        # noise_for_this_frame = torch.rand_like(self.movie[frame_id]) * self.config.bg_model
        # noise_for_this_frame = torch.poisson(noise_for_this_frame)
        noise_distribution = torch.distributions.poisson.Poisson(self.config.bg_model)
        noise_for_this_frame = noise_distribution.sample((self.config.image_size, self.config.image_size))

        if len(photon_pos_frame) == 0:
            # There is no photon allocated in this frame
            # So we will return empty image
            # TODO: Add noise in here
            self.movie[frame_id, :, :] = torch.zeros_like(self.movie[frame_id]) + noise_for_this_frame
        else:
            # TODO: convert everything into tensor
            x = photon_pos_frame[:, 0] + self.drift_x[frame_id].numpy()
            y = photon_pos_frame[:, 1] + self.drift_y[frame_id].numpy()
            single_frame, _, _ = np.histogram2d(y, x, bins=(edges, edges), )

            # return simulated_frame, gt_position

            self.movie[frame_id, :, :] = torch.tensor(single_frame) + noise_for_this_frame
            # TODO: process gt_position. Some how store that information
            # Save the ground truth information
            self.gt_frame.extend([frame_id] * len(gt_position))
            if len(gt_position) == 1:
                self.gt_x_without_drift.append(self.binding_site_position[0, gt_position])
                self.gt_y_without_drift.append(self.binding_site_position[1, gt_position])

                self.gt_x_with_drift.append(self.binding_site_position[0, gt_position] + self.drift_x[frame_id].item())
                self.gt_y_with_drift.append(self.binding_site_position[1, gt_position] + self.drift_y[frame_id].item())
            else:
                self.gt_x_without_drift.extend(self.binding_site_position[0, gt_position].tolist())
                self.gt_y_without_drift.extend(self.binding_site_position[1, gt_position].tolist())

                self.gt_x_with_drift.extend((self.binding_site_position[0, gt_position] + self.drift_x[frame_id].numpy()).tolist())
                self.gt_y_with_drift.extend((self.binding_site_position[1, gt_position] + self.drift_y[frame_id].numpy()).tolist())
            self.gt_photon.extend(self.distributed_photon[gt_position, frame_id].tolist())
            # Add correct background
            self.gt_noise.extend([noise_for_this_frame.mean().item()] * len(gt_position))
            # TODO: save the ground truth for training purpose




if __name__ == '__main__':
    pass
