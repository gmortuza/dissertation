import sys
from collections import Counter

import simulate
import numpy as np
import io_modified as _io
from get_logger import get_logger
from tqdm import tqdm
import h5py

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

    def save_ground_truth(self, config):
        x = config["new_struct"][0]
        y = config["new_struct"][1]
        photons = config["Imager.Photonbudget"] / config["Imager.Photonslope"]
        ground_truth = np.rec.array(
            (
                x,
                y,
                np.full_like(x, photons),  # Photons
                np.zeros_like(x),  # background
                np.full_like(x, .009),  # lpx
                np.full_like(x, .009),  # lpy
            ), dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])
        with h5py.File(config["output_file"]+"_ground_truth_without_frame.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=ground_truth)

    def get_origami(self, config):
        """
        Generate the random origami
        :return: origami_counter --> How many times each of the origami appeared
        """
        # np.random.seed(100)

        x_distance, y_distance = config["distance_x"], config["distance_y"]
        row, column = config["origami_row"], config["origami_column"]
        num_total_origami, num_unique_origami = config["total_origami"], config["unique_origami"]

        x = np.arange(0, x_distance * column, x_distance)
        y = np.arange(0, y_distance * row, y_distance)
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.ravel(), mesh_y.ravel()
        sample_origami = np.random.randint(2, size=(num_unique_origami, 48), dtype=np.int)
        np.savetxt(config["output_file"] + "_gt-origami.txt", sample_origami, fmt='%i')
        sample_origami = sample_origami.astype(np.bool)
        # sample_origami = np.ones_like(sample_origami, dtype=np.int).astype(np.bool)
        unique_origami = {}
        # create the unique origami
        for origami_id, single_origami in enumerate(sample_origami):
            single_origami_x, single_origami_y = mesh_x[single_origami], mesh_y[single_origami]
            # Move center of the mass to the origin
            if config["origami_mean"]:
                single_origami_x = single_origami_x - np.mean(single_origami_x)
                single_origami_y = single_origami_y - np.mean(single_origami_y)
            # Convert pixel to nanometer
            single_origami_x = single_origami_x / config["Camera.Pixelsize"]
            single_origami_y = single_origami_y / config["Camera.Pixelsize"]

            unique_origami[origami_id] = {}
            unique_origami[origami_id]["x_cor"] = single_origami_x
            unique_origami[origami_id]["y_cor"] = single_origami_y
            unique_origami[origami_id]["labels"] = np.ones_like(single_origami_x)
            unique_origami[origami_id]["3d"] = np.zeros_like(single_origami_x)

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
        return unique_origami

    def simulate(self):
        # Number of frames
        frames = self.config["Frames"]

        file_name = self.config["output_file"]+".raw"
        logger.info("Distributing photon")

        # Structure that will be used for this colors

        no_sites = len(
            self.config["new_struct"][0, :]
        )  # number of binding sites in image
        # amount of photon for each of the binding sites on each frame
        photon_dist = np.zeros((no_sites, frames), dtype=np.int)
        spot_kinetics = np.zeros((no_sites, 4), dtype=np.float)

        time_trace = {}

        # This will populate the variable photon_dist
        for n_site in tqdm(range(0, no_sites), desc="Distributing photon"):
            always_on = float(self.config["photons_for_each_gold_nano_particle"]) \
                if no_sites - n_site <= int(self.config["num_gold_nano_particle"]) else 0
            if always_on > 0:
                print("ALways on")
            #  For each site will assign the number of photon
            p_temp, t_temp, k_temp = simulate.distribute_photons(
                self.config["new_struct"],
                self.config["Camera.Integration Time"],
                self.config["Frames"],
                self.config["taud"],  # mean dark (ms)
                self.config["PAINT.taub"],  # mean bright (ms)
                self.config["Imager.Photonrate"],
                self.config["Imager.Photonrate Std"],
                self.config["Imager.Photonbudget"],
                always_on=always_on
            )
            photon_dist[n_site, :] = p_temp
            spot_kinetics[n_site, :] = k_temp
            time_trace[n_site] = self.vectorToString(t_temp)
        # Converting into movie
        logger.info("Converting to image")
        ground_truth_frames = []
        ground_truth_x = np.asarray([])
        ground_truth_y = np.asarray([])
        ground_truth_x_with_drift = np.asarray([])
        ground_truth_y_with_drift = np.asarray([])
        photons = []

        movie = np.zeros(shape=(frames, self.config["Height"], self.config["Width"]))
        for runner in tqdm(range(0, frames), desc="Converting into image"):
            movie[runner, :, :], gt_pos = simulate.convertMovie(
                runner,
                photon_dist,
                self.config
            )
            # Generate the ground truth data
            # Not all the frames will have blinking event.
            # If this particular frame have any blinking event then we will record it's property
            if gt_pos.any():
                x_pos = self.config["new_struct"][0][gt_pos]
                y_pos = self.config["new_struct"][1][gt_pos]
                photons.extend(photon_dist[gt_pos, runner])
                ground_truth_frames.extend([runner+1] * len(gt_pos))
                ground_truth_x = np.concatenate((ground_truth_x, x_pos))
                ground_truth_y = np.concatenate((ground_truth_y, y_pos))
                ground_truth_x_with_drift = np.concatenate((ground_truth_x_with_drift, x_pos + self.config["drift_x"][runner]))
                ground_truth_y_with_drift = np.concatenate((ground_truth_y_with_drift, y_pos + self.config["drift_y"][runner]))
        # Add noise to this movie
        movie, added_noise = simulate.add_noise(self.config["noise_type"], movie)
        # Extract noise from each frame and for each binding site
        # Save the ground truth
        # Photons on each blinking event in each frame
        print("Here")

        ground_truth_frames = np.asarray(ground_truth_frames)
        content_for_yaml_file = f"""Box Size: 7\nPixelsize: {self.config["Camera.Pixelsize"]}\nFrames: {self.config["Frames"]}\nHeight: {self.config["Height"]}\nWidth: {self.config["Width"]}"""
        ground_truth_without_drift = np.rec.array(
            (
                ground_truth_frames,
                ground_truth_x,
                ground_truth_y,
                photons,
                np.zeros_like(ground_truth_x),  # background
                np.full_like(ground_truth_x, .009),  # lpx
                np.full_like(ground_truth_x, .009),  # lpy
            ), dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])
        with h5py.File(self.config["output_file"]+"_ground_truth_without_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=ground_truth_without_drift)
            with open(self.config["output_file"] + "_ground_truth_without_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)

        ground_truth_with_drift = np.rec.array(
            (
                ground_truth_frames,
                ground_truth_x_with_drift,
                ground_truth_y_with_drift,
                photons,
                np.zeros_like(ground_truth_x),  # background
                np.full_like(ground_truth_x, .009),  # lpx
                np.full_like(ground_truth_x, .009),  # lpy
            ), dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ])
        with h5py.File(self.config["output_file"]+"_ground_truth_with_drift.hdf5", "w") as locs_file:
            locs_file.create_dataset("locs", data=ground_truth_with_drift)
            # Creating yaml file for render purpose
            with open(self.config["output_file"] + "_ground_truth_with_drift.yaml", "w") as yaml_file:
                yaml_file.write(content_for_yaml_file)

        movie = simulate.check_type(movie)
        logger.info("saving movie")
        # Convert new struct and exchange_round to string otherwise saving will have error
        del self.config["new_struct"]
        del self.config["drift_x"]
        del self.config["drift_y"]
        self.config["exchange_round"] = self.vectorToString(self.config["exchange_round"].tolist())
        logger.info("Saving image")
        simulate.saveMovie(file_name, movie, self.config)

    def loadSettings(self):

        path = "config.yaml"
        config = _io.load_info(path)[0]
        # calculate taud
        config["taud"] = round(1 / (config["PAINT.k_on"] * config["PAINT.imager"] * 1 / 10 ** 9) * 1000)
        # Calculate photon parameter

        config["Imager.PhotonslopeStd"] = config["Imager.Photonslope"] / STD_FACTOR
        config["Imager.Photonrate"] = config["Imager.Photonslope"] * config["Imager.Laserpower"]
        config["Imager.Photonrate Std"] = config["Imager.PhotonslopeStd"] * config["Imager.Laserpower"]

        # Calculating the handle
        # handle x
        grid_pos = simulate.generatePositions(
            int(config["total_origami"]), int(config["Height"]), int(config["frame_padding"]),
            int(config["origami_arrangement"])
        )
        origamies = self.get_origami(config)

        new_struct = simulate.prepareStructures(
            origamies,
            grid_pos,
            int(config["origami_orientation"]),
            int(config["total_origami"]),
            int(config["binding_site_incorporation"]),
            exchange=0,
            frame_padding=int(config["frame_padding"]),
            height=int(config["Height"]),
            num_gold_nano_particle=int(config["num_gold_nano_particle"])
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
                * config["noise_level"]
        )
        config["bgmodel"] = int(bgmodel)
        config["exchange_round"] = np.asarray(list(set(new_struct[2])), dtype=np.int)
        config["drift_x"], config["drift_y"] = self.get_drift(config)

        self.save_ground_truth(config)

        return config

    def get_drift(self, config):
        drift_method = config["drift_method"]
        drift_x = config["drift_x"]
        drift_y = config["drift_y"]
        single_drift_x = drift_x / 1000
        single_drift_y = drift_y / 1000
        if drift_method == "linear":
            frame_drift_x, frame_drift_y = np.full(config["Frames"], single_drift_x), np.full(config["Frames"], single_drift_y)
        else:
            # Default value is random walk
            frame_drift_x = np.random.normal(0, single_drift_x, config["Frames"])
            frame_drift_y = np.random.normal(0, single_drift_y, config["Frames"])

        with open(config["output_file"]+"_drift_frame_relative.csv", "w") as drift_file:
            drift_file.write("dx, dy\n")
            np.savetxt(drift_file, np.transpose(np.vstack((frame_drift_x, frame_drift_y))), '%s', ',')

        frame_drift_x = np.cumsum(frame_drift_x)
        frame_drift_y = np.cumsum(frame_drift_y)
        # Save drift
        with open(config["output_file"]+"_drift.csv", "w") as drift_file:
            drift_file.write("dx, dy\n")
            np.savetxt(drift_file, np.transpose(np.vstack((frame_drift_x, frame_drift_y))), '%s', ',')
        return frame_drift_x, frame_drift_y

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
            line = np.asarray((linetxt).split(","))
        else:
            line = np.asarray((linetxt.split(",")))

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
        x_arrstr = np.char.mod("%f", x)
        x_str = ",".join(x_arrstr)
        return x_str


if __name__ == '__main__':
    s = Simulate()
