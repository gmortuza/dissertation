import torch
import numpy as np
from unique_origami import get_unique_origami


class Simulation:
    def __init__(self):
        pass

    def generate_blinking_event(self):
        grid_position = self.generate_grid_pos(self.config.total_origami, self.config.image_size,
                                               self.config.frame_padding, self.config.origami_arrangement)
        unique_origamies = get_unique_origami(self.config)

        blinking_positions = self.place_origamies(grid_position, unique_origamies)

    def place_origamies(self, grid_pos, origamies):
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


if __name__ == '__main__':
    pass
