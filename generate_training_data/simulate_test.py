import time
from matplotlib import patches
import matplotlib.pyplot as plt
from scipy.stats import norm

import simulate
import numpy as _np
import io_modified as _io
import os.path as _ospath
from get_logger import get_logger
from tqdm import tqdm

# default variable
filename = "new_test_data.raw"
ADVANCED_MODE = 0  # 1 is with calibration of noise model
# IMAGER
POWER_DENSITY_CONVERSION = 20
POWERDENSITY_CONVERSION = 20
# NOISE MODEL
LASERC_DEFAULT = 0.012063
IMAGERC_DEFAULT = 0.003195
# STRUCTURE
NOISE = 0.5
TOTAL_HEIGHT = 16
TOTAL_WIDTH = 16
logger = get_logger()


class Simulate:
    def __init__(self):
        self.loadSettings()
        self.concatExchangeEdit = False
        self.conroundsEdit = 1
        self.current_round = 0
        self.simulate()

    def simulate(self):
        # Additional settings
        noise = NOISE
        # Exchange round to be simulated
        # default value is 1
        exchange_round_to_simulate = self.config["exchange_round"]

        no_exchange_color = len(set(exchange_round_to_simulate))
        exchange_colors = list(set(exchange_round_to_simulate))

        self.current_round += 1
        # structre read from he file
        struct = self.config["new_struct"]
        # Number of frames
        frames = self.config["Camera.Frames"]

        base_file_name = "simulation_noise_5_origami_9_in_16_16.raw"
        logger.info("Distributing photon")
        # time
        t0 = time.time()
        for n_color in range(0, no_exchange_color):
            # If we use multiple color then each colors blinking event will be on different file
            # This will create different file for each color
            # for now we are using only one color so this will run once
            if no_exchange_color > 1:
                # Update the file name according to that
                file_name = _io.multiple_filenames(base_file_name, n_color)
                # Don't take all the binding event. Only take the co-ordinate of binding event that is bound to this color
                struct_partial = struct[:, struct[2, :] == exchange_colors[n_color]]
            elif self.concatExchangeEdit:
                file_name = base_file_name
                struct_partial = struct[:, struct[2, :] == exchange_colors[self.current_round - 1], ]
            else:  # There is only one color
                file_name = base_file_name
                # Structure that will be used for this colors
                struct_partial = struct[:, struct[2, :] == exchange_colors[0]]

            print("Distributing photons")

            no_sites = len(
                struct_partial[0, :]
            )  # number of binding sites in image
            # amount of photon for each of the binding sites on each frame
            photon_dist = _np.zeros((no_sites, frames), dtype=_np.int)
            spot_kinetics = _np.zeros((no_sites, 4), dtype=_np.float)

            time_trace = {}

            # This will populate the variable photon_dist
            for n_site in range(0, no_sites):  # For each site will assign the number of photon
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
                outputmsg = (
                        "Distributing photons ... "
                        + str(_np.round(n_site / no_sites * 1000) / 10)
                        + " %"
                )
                print(outputmsg)
            # Converting into movie
            logger.info("Converting to image")

            movie = _np.zeros(shape=(frames, self.config["Camera.Image Size"], self.config["Camera.Image Size"]))
            for runner in range(0, frames):
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
                outputmsg = (
                        "Converting to Image ... "
                        + str(_np.round(runner / frames * 1000) / 10)
                        + " %"
                )

                print(outputmsg)

            movie = simulate.noisy_p(movie, self.config["bgmodel"])
            # insert poisson noise
            movie = simulate.check_type(movie)
            print("saving movie")
            simulate.saveMovie("simulation.raw", movie, self.config)

    def loadSettings(self):  # TODO: re-write exceptions, check key
        # Default\
        STDFACTOR = 1.82
        path = "/Users/golammortuza/workspace/nam/dissertation/generate_training_data/simulate.yaml"
        config = _io.load_info(path)[0]
        # calculate taud
        config["taud"] = round(1 / (config["PAINT.k_on"] * config["PAINT.imager"] * 1 / 10 ** 9) * 1000)
        self.taudEdit = config["taud"]
        # Calculate photon parameter

        config["Imager.PhotonslopeStd"] = config["Imager.Photonslope"] / STDFACTOR
        config["Imager.Photonrate"] = config["Imager.Photonslope"] * config["Imager.Laserpower"]
        config["Imager.Photonrate Std"] = config["Imager.PhotonslopeStd"] * config["Imager.Laserpower"]

        # Calculating the handle
        # handle x
        gridpos = simulate.generatePositions(
            int(config["Structure.Number"]), int(config["Camera.Image Size"]), int(config["Structure.Frame"]), int(config["Structure.Arrangement"])
        )
        structurexx, structureyy, structureex, structure3d = self.readStructure(config)
        structure = simulate.defineStructure(
            structurexx,
            structureyy,
            structureex,
            structure3d,
            config["Camera.Pixelsize"],
            mean=False,
        )
        newstruct = simulate.prepareStructures(
            structure,
            gridpos,
            int(config["Structure.Orientation"]),
            int(config["Structure.Number"]),
            int(config["Structure.Incorporation"]),
            exchange=0
        )
        config["new_struct"] = newstruct
        config["Structure.HandleX"] = self.vectorToString(newstruct[0])
        config["Structure.HandleY"] = self.vectorToString(newstruct[1])
        config["Structure.Handle3d"] = self.vectorToString(newstruct[4])
        config["Structure.HandleEx"] = self.vectorToString(newstruct[2])
        config["Structure.HandleStruct"] = self.vectorToString(newstruct[3])
        config["noexchangecolors"] = len(set(newstruct[2]))
        bgmodel = (
                (LASERC_DEFAULT + IMAGERC_DEFAULT * config["PAINT.imager"])
                * config["Imager.Laserpower"] * POWERDENSITY_CONVERSION
                * config["Camera.Integration Time"]
                * config["Imager.BackgroundLevel"]
        )
        config["bgmodel"] = int(bgmodel)
        self.backgroundframesimpleEdit = int(bgmodel)
        config["exchange_round"] = _np.asarray(list(set(newstruct[2])), dtype=_np.int)
        self.config = config

    def readStructure(self, config):
        structurexx = self.readLine(config["Structure.StructureX"])
        structureyy = self.readLine(config["Structure.StructureY"])
        structureex = self.readLine(config["Structure.StructureEx"], "int")
        structure3d = self.readLine(config["Structure.Structure3D"])

        minlen = min(
            len(structureex),
            len(structurexx),
            len(structureyy),
            len(structure3d),
        )

        structurexx = structurexx[0:minlen]
        structureyy = structureyy[0:minlen]
        structureex = structureex[0:minlen]
        structure3d = structure3d[0:minlen]

        return structurexx, structureyy, structureex, structure3d

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
