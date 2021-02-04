from collections import Counter

import simulate
import numpy as _np
import io_modified as _io
from get_logger import get_logger
from tqdm import tqdm

# default variable
ADVANCED_MODE = 0  # 1 is with calibration of noise model
# IMAGER
POWER_DENSITY_CONVERSION = 20
POWERDENSITY_CONVERSION = 20
# NOISE MODEL
LASERC_DEFAULT = 0.012063
IMAGERC_DEFAULT = 0.003195
# Default\
STD_FACTOR = 1.82
logger = get_logger()


class Simulate:
    def __init__(self):
        self.config = self.loadSettings()
        self.simulate()

    def get_origami(self, config):
        """
        Generate the random origami
        :return: origami_counter --> How many times each of the origami appeared
        """
        _np.random.seed(100)

        x_distance, y_distance = config["Distance_x"], config["Distance_y"]
        row, column = config["origami_row"], config["origami_column"]
        num_total_origami, num_unique_origami = config["Structure.Number"], config["unique_origami"]

        x = _np.arange(0, x_distance * column, x_distance)
        y = _np.arange(0, y_distance * row, y_distance)
        mesh_x, mesh_y = _np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.ravel(), mesh_y.ravel()
        # adding seed for reproducibility
        sample_origami = _np.random.randint(2, size=(num_unique_origami, 48)).astype(_np.bool)
        unique_origami = {}
        # create the unique origami
        for origami_id, single_origami in enumerate(sample_origami):
            single_origami_x, single_origami_y = mesh_x[single_origami], mesh_y[single_origami]
            unique_origami[origami_id] = {}
            unique_origami[origami_id]["x_cor"] = single_origami_x
            unique_origami[origami_id]["y_cor"] = single_origami_y
            unique_origami[origami_id]["labels"] = _np.ones_like(single_origami_x)
            unique_origami[origami_id]["3d"] = _np.zeros_like(single_origami_x)

        # Taking the random num_total_origamies here.
        # but later decided to pick random origami later
        # Keeping this Piece of code for future references
        """
        random_idx = _np.random.choice(_np.arange(num_unique_origami), num_total_origami)
        origami_counter = Counter(random_idx)
        unique_origami_x = _np.asarray(unique_origami_x, dtype=_np.object)
        unique_origami_y = _np.asarray(unique_origami_y, dtype=_np.object)

        total_origami_x = unique_origami_x[random_idx].ravel()
        total_origami_y = unique_origami_y[random_idx].ravel()
        """
        return unique_origami

    def simulate(self):
        # Number of frames
        frames = self.config["Camera.Frames"]

        file_name = "test.raw"
        logger.info("Distributing photon")

        # Structure that will be used for this colors
        struct_partial = self.config["new_struct"]

        print("Distributing photons")

        no_sites = len(
            struct_partial[0, :]
        )  # number of binding sites in image
        # amount of photon for each of the binding sites on each frame
        photon_dist = _np.zeros((no_sites, frames), dtype=_np.int)
        spot_kinetics = _np.zeros((no_sites, 4), dtype=_np.float)

        time_trace = {}

        # This will populate the variable photon_dist
        for n_site in tqdm(range(0, no_sites), desc="Distributing photon"):
            #  For each site will assign the number of photon
            p_temp, t_temp, k_temp = simulate.distphotons(
                struct_partial,
                self.config["Camera.Integration Time"],
                self.config["Camera.Frames"],
                self.config["taud"],  # mean dark (ms)
                self.config["PAINT.taub"],  # mean bright (ms)
                self.config["Imager.Photonrate"],
                self.config["Imager.Photonrate Std"],
                self.config["Imager.Photonbudget"],
            )
            photon_dist[n_site, :] = p_temp
            spot_kinetics[n_site, :] = k_temp
            time_trace[n_site] = self.vectorToString(t_temp)
        # Converting into movie
        logger.info("Converting to image")

        movie = _np.zeros(shape=(frames, self.config["Camera.Image Size"], self.config["Camera.Image Size"]))
        for runner in tqdm(range(0, frames), desc="Converting into image"):
            movie[runner, :, :] = simulate.convertMovie(
                runner,
                photon_dist,
                struct_partial,
                self.config["Camera.Image Size"],
                frames,
                self.config["Imager.PSF"],
                self.config["Imager.Photonrate"],
                self.config["Imager.BackgroundLevel"],
                self.config["bgmodel"], # Noise
                int(self.config["Structure.3D"]),
                self.config["Structure.CX"],
                self.config["Structure.CY"],
            )
            # Add noise to this movie
        movie = simulate.noisy_p(movie, self.config["bgmodel"])
        # insert poisson noise
        movie = simulate.check_type(movie)
        logger.info("saving movie")
        # Convert newstruct and exhange_round to sring otherwise saving will have error
        del self.config["new_struct"]
        self.config["exchange_round"] = self.vectorToString(self.config["exchange_round"].tolist())
        simulate.saveMovie(file_name, movie, self.config)

    def loadSettings(self):  # TODO: re-write exceptions, check key

        path = "/Users/golammortuza/workspace/nam/dissertation/generate_training_data/simulate.yaml"
        config = _io.load_info(path)[0]
        # calculate taud
        config["taud"] = round(1 / (config["PAINT.k_on"] * config["PAINT.imager"] * 1 / 10 ** 9) * 1000)
        # Calculate photon parameter

        config["Imager.PhotonslopeStd"] = config["Imager.Photonslope"] / STD_FACTOR
        config["Imager.Photonrate"] = config["Imager.Photonslope"] * config["Imager.Laserpower"]
        config["Imager.Photonrate Std"] = config["Imager.PhotonslopeStd"] * config["Imager.Laserpower"]

        # Calculating the handle
        # handle x
        gridpos = simulate.generatePositions(
            int(config["Structure.Number"]), int(config["Camera.Image Size"]), int(config["Structure.Frame"]), int(config["Structure.Arrangement"])
        )
        structure_xx, structure_yy, structure_ex, structure_3d = self.readStructure(config)
        structure = simulate.defineStructure(
            structure_xx,
            structure_yy,
            structure_ex,
            structure_3d,
            config["Camera.Pixelsize"],
            mean=False,
        )
        new_struct = simulate.prepareStructures(
            structure,
            gridpos,
            int(config["Structure.Orientation"]),
            int(config["Structure.Number"]),
            int(config["Structure.Incorporation"]),
            exchange=0
        )
        config["new_struct"] = new_struct
        config["Structure.HandleX"] = self.vectorToString(new_struct[0])
        config["Structure.HandleY"] = self.vectorToString(new_struct[1])
        config["Structure.Handle3d"] = self.vectorToString(new_struct[4])
        config["Structure.HandleEx"] = self.vectorToString(new_struct[2])
        config["Structure.HandleStruct"] = self.vectorToString(new_struct[3])
        config["noexchangecolors"] = len(set(new_struct[2]))
        bgmodel = (
                (LASERC_DEFAULT + IMAGERC_DEFAULT * config["PAINT.imager"])
                * config["Imager.Laserpower"] * POWERDENSITY_CONVERSION
                * config["Camera.Integration Time"]
                * config["Imager.BackgroundLevel"]
        )
        config["bgmodel"] = int(bgmodel)
        config["exchange_round"] = _np.asarray(list(set(new_struct[2])), dtype=_np.int)
        return config

    def readStructure(self, config):
        """
        :param config:
        :return: structure_xx, structure_yy, structure_ex, structure_3d
        """
        # Generate origami
        # return self.get_origami(config)

        structure_xx = self.readLine(config["Structure.StructureX"])
        structure_yy = self.readLine(config["Structure.StructureY"])
        structure_ex = self.readLine(config["Structure.StructureEx"], "int")
        structure3d = self.readLine(config["Structure.Structure3D"])

        minlen = min(
            len(structure_ex),
            len(structure_xx),
            len(structure_yy),
            len(structure3d),
        )

        structure_xx = structure_xx[0:minlen]
        structure_yy = structure_yy[0:minlen]
        structure_ex = structure_ex[0:minlen]
        structure3d = structure3d[0:minlen]

        return structure_xx, structure_yy, structure_ex, structure3d

    def readLine(self, linetxt, type="float", textmode=True):
        if textmode:
            line = _np.asarray((linetxt).split(","))
        else:
            line = _np.asarray((linetxt.split(",")))

        values = []
        for element in line:
            try:
                if type == "int":
                    values.append(int(element))
                elif type == "float":
                    values.append(float(element))

            except ValueError:
                pass
        return values

    def vectorToString(self, x):
        x_arrstr = _np.char.mod("%f", x)
        x_str = ",".join(x_arrstr)
        return x_str


if __name__ == '__main__':
    s = Simulate()
