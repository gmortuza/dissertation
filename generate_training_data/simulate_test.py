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
        self.exchangeroundsEdit = "1"
        self.concatExchangeEdit = False
        self.conroundsEdit = 1
        self.currentround = 0
        self.simulate()

    def simulate(self):
        # Additional settings
        noise = NOISE
        # Exchange round to be simulated
        # default value is 1
        exchangeroundstoSim = _np.asarray(
            (self.exchangeroundsEdit).split(",")
        )
        # converting numpy int
        exchangeroundstoSim = exchangeroundstoSim.astype(_np.int)

        noexchangecolors = len(set(exchangeroundstoSim))
        exchangecolors = list(set(exchangeroundstoSim))

        # Checkbox
        if self.concatExchangeEdit:
            conrounds = noexchangecolors
        else:
            conrounds = self.conroundsEdit

        self.currentround += 1
        # structre read from he file
        struct = self.newstruct
        mode3Dstate = int(self.mode3DEdit)
        # Number of frames
        frames = self.framesEdit

        fileNameOld = "simulation_noise_5_origami_9_in_16_16.raw"
        logger.info("Distributing photon")
        pbar = tqdm(total=100, initial=0)
        # time
        t0 = time.time()
        for n_color in range(0, noexchangecolors):

            if noexchangecolors > 1:
                fileName = _io.multiple_filenames(fileNameOld, n_color)
                partstruct = struct[:, struct[2, :] == exchangecolors[n_color]]
            elif self.concatExchangeEdit:
                fileName = fileNameOld
                partstruct = struct[
                             :,
                             struct[2, :] == exchangecolors[self.currentround - 1],
                             ]
            else:
                fileName = fileNameOld
                # Structure that will be used for this colors
                partstruct = struct[:, struct[2, :] == exchangecolors[0]]

            print("Distributing photons")

            bindingsitesx = partstruct[0, :]

            nosites = len(
                bindingsitesx
            )  # number of binding sites in image
            # amount of photon for each of the binding sites on each frame
            photondist = _np.zeros((nosites, frames), dtype=_np.int)
            spotkinetics = _np.zeros((nosites, 4), dtype=_np.float)


            timetrace = {}

            # This will populate the variable photondist
            for n_site in range(0, nosites):  # For each site will assign the number of photon
                p_temp, t_temp, k_temp = simulate.distphotons(
                    partstruct,
                    self.integrationtimeEdit,
                    frames,
                    self.taudEdit,  # mean dark (ms)
                    self.taubEdit,  # mean bright (ms)
                    self.photonrateEdit,
                    self.photonratestdEdit,
                    self.photonbudgetEdit,
                )
                photondist[n_site, :] = p_temp
                spotkinetics[n_site, :] = k_temp
                timetrace[n_site] = self.vectorToString(t_temp)
                outputmsg = (
                        "Distributing photons ... "
                        + str(_np.round(n_site / nosites * 1000) / 10)
                        + " %"
                )
                pbar.update(2)
                # print(outputmsg)
                #self.mainpbar.setValue(_np.round(i / nosites * 1000) / 10)

            #self.statusBar().showMessage("Converting to image ... ")
            logger.info("Converting to image")
            #onevents = self.vectorToString(spotkinetics[:, 0])
            #localizations = self.vectorToString(spotkinetics[:, 1])
            #meandarksim = self.vectorToString(spotkinetics[:, 2])
            #meanbrightsim = self.vectorToString(spotkinetics[:, 3])

            movie = _np.zeros(shape=(frames, self.camerasizeEdit, self.camerasizeEdit))
            info = {}
            pbar = tqdm(100)
            if conrounds != 1:
                for runner in range(0, frames):
                    movie[runner, :, :] = simulate.convertMovie(
                        runner,
                        photondist,
                        partstruct,
                        self.camerasizeEdit,
                        frames,
                        self.psfEdit,
                        self.photonrateEdit,
                        self.backgroundlevelEdit,
                        noise,
                        mode3Dstate,
                        self.cx,
                        self.cy,
                    )
                    outputmsg = (
                            "Converting to Image ... "
                            + str(_np.round(runner / frames * 1000) / 10)
                            + " %"
                    )

                    self.statusBar().showMessage(outputmsg)
                    self.mainpbar.setValue(
                        _np.round(runner / frames * 1000) / 10
                    )
                    # app.processEvents()

                if self.currentround == 1:
                    self.movie = movie
                else:
                    movie = movie + self.movie
                    self.movie = movie

                self.statusBar().showMessage(
                    "Converting to image ... complete. Current round: "
                    + str(self.currentround)
                    + " of "
                    + str(conrounds)
                    + ". Please set and start next round."
                )
                if self.currentround == conrounds:
                    self.statusBar().showMessage(
                        "Adding noise to movie ..."
                    )
                    movie = simulate.noisy_p(movie, int(self.backgroundframesimpleEdit))
                    movie = simulate.check_type(movie)
                    self.statusBar().showMessage("Saving movie ...")

                    simulate.saveMovie(fileName, movie, info)
                    self.statusBar().showMessage(
                        "Movie saved to: " + fileName
                    )
                    dt = time.time() - t0
                    self.statusBar().showMessage(
                        "All computations finished. Last file saved to: "
                        + fileName
                        + ". Time elapsed: {:.2f} Seconds.".format(dt)
                    )
                    self.currentround = 0
                else:  # just save info file
                    # self.statusBar().showMessage('Saving yaml ...')
                    info_path = (
                            _ospath.splitext(fileName)[0]
                            + "_"
                            + str(self.currentround)
                            + ".yaml"
                    )
                    _io.save_info(info_path, [info])

                    if self.exportkinetics.isChecked():
                        # Export the kinetic data if this is checked
                        kinfo_path = (
                                _ospath.splitext(fileName)[0]
                                + "_"
                                + str(self.currentround)
                                + "_kinetics.yaml"
                        )
                        _io.save_info(kinfo_path, [timetrace])

                    self.statusBar().showMessage(
                        "Movie saved to: " + fileName
                    )

            else:
                for runner in range(0, frames):
                    movie[runner, :, :] = simulate.convertMovie(
                        runner,
                        photondist,
                        partstruct,
                        self.camerasizeEdit,
                        frames,
                        self.psfEdit,
                        self.photonrateEdit,
                        self.backgroundlevelEdit,
                        noise,
                        mode3Dstate,
                        self.cx,
                        self.cy,
                    )
                    # Add noise to this movie
                    # outputmsg = (
                    #         "Converting to Image ... "
                    #         + str(_np.round(runner / frames * 1000) / 10)
                    #         + " %"
                    # )

                    pbar.update(_np.round(runner / frames * 1000) / 10)
                    # print(outputmsg)
                    # self.statusBar().showMessage(outputmsg)
                    # self.mainpbar.setValue(
                    #    _np.round(runner / frames * 1000) / 10
                    # )

                # movie = simulate.noisy_p(movie, self.backgroundframesimpleEdit)
                # TODO: need to improve that functionality
                # insert poisson noise
                movie = movie + _np.random.poisson(100000 * noise, movie.shape)
                movie = simulate.check_type(movie)
                #self.mainpbar.setValue(100)
                #self.statusBar().showMessage(
                #    "Converting to image ... complete."
                #)
                #self.statusBar().showMessage("Saving movie ...")
                print("saving movie")
                simulate.saveMovie("simulation.raw", movie, self.info[0])
                #simulate.saveMovie(fileName, movie, info)
                #if self.exportkinetics.isChecked():
                # if False:
                #     # Export the kinetic data if this is checked
                #     kinfo_path = (
                #             _ospath.splitext(fileName)[0] + "_kinetics.yaml"
                #     )
                #     _io.save_info(kinfo_path, [timetrace])
                # print("Movie saved to: "+fileName)
                # #self.statusBar().showMessage("Movie saved to: " + fileName)
                # dt = time.time() - t0
                # print(
                #     "All computations finished. Last file saved to: "
                #     + fileName
                #     + ". Time elapsed: {:.2f} Seconds.".format(dt)
                # )
                # self.currentround = 0


    def changePaint(self):
        kon = self.konEdit
        imagerconcentration = self.imagerconcentrationEdit
        taud = round(1 / (kon * imagerconcentration * 1 / 10 ** 9) * 1000)
        self.taudEdit = (str(taud))
        self.changeNoise()

    def changeNoise(self):
        itime = self.integrationtimeEdit
        imagerconcentration = self.imagerconcentrationEdit
        laserpower = self.laserpowerEdit * POWER_DENSITY_CONVERSION
        bglevel = self.backgroundlevelEdit
        if ADVANCED_MODE:
            # NEW NOISE MODEL
            laserc = self.lasercEdit
            imagerc = self.imagercEdit
            bgoffset = self.BgoffsetEdit
            bgmodel = (
                laserc + imagerc * imagerconcentration
            ) * laserpower * itime + bgoffset
            equationA = self.EquationAEdit
            equationB = self.EquationBEdit
            equationC = self.EquationCEdit
            bgstdoffset = self.BgStdoffsetEdit
            bgmodelstd = (
                equationA * laserpower * itime
                + equationB * bgmodel
                + equationC
                + bgstdoffset * bglevel
            )
            self.backgroundframeEdit.setText(str(int(bgmodel)))
            self.noiseEdit.setText(str(int(bgmodelstd)))
        else:
            bgmodel = (
                (LASERC_DEFAULT + IMAGERC_DEFAULT * imagerconcentration)
                * laserpower
                * itime
                * bglevel
            )
            self.backgroundframesimpleEdit = int(bgmodel)

    def loadSettings(self):  # TODO: re-write exceptions, check key
        # Default\
        STDFACTOR = 1.82
        path = "/Users/golammortuza/workspace/nam/dissertation/generate_training_data/simulate.yaml"
        config = _io.load_info(path)[0]
        # calculate taud
        config["taud"] = round(1 / (config["PAINT.k_on"] * config["PAINT.imager"] * 1 / 10 ** 9) * 1000)
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
        config["Structure.HandleX"] = self.vectorToString(newstruct[0])
        config["Structure.HandleY"] = self.vectorToString(newstruct[1])
        config["Structure.Handle3d"] = self.vectorToString(newstruct[4])
        config["Structure.HandleEx"] = self.vectorToString(newstruct[2])
        config["Structure.HandleStruct"] = self.vectorToString(newstruct[3])
        config["noexchangecolors"] = len(set(newstruct[2]))


        self.config = config

        if path:
            # info = _io.load_info(path)
            info = [config]
            self.info = info
            self.framesEdit = (info[0]["Frames"])

            self.structureframeEdit = (info[0]["Structure.Frame"])
            self.structurenoEdit = (info[0]["Structure.Number"])
            self.structurexxEdit = (info[0]["Structure.StructureX"])
            self.structureyyEdit = (info[0]["Structure.StructureY"])
            self.structureexEdit = (info[0]["Structure.StructureEx"])
            try:
                self.structure3DEdit = (info[0]["Structure.Structure3D"])
                self.mode3DEdit = (info[0]["Structure.3D"])
                self.cx = (info[0]["Structure.CX"])
                self.cy = (info[0]["Structure.CY"])
            except Exception as e:
                print(e)
                pass
            try:
                self.photonslopemodeEdit = (
                    info[0]["Imager.Constant Photonrate Std"]
                )
            except Exception as e:
                print(e)
                pass

            try:
                self.backgroundlevelEdit = (
                    info[0]["Imager.BackgroundLevel"]
                )
            except Exception as e:
                print(e)
                pass
            self.structureIncorporationEdit = (
                info[0]["Structure.Incorporation"]
            )

            self.structurerandomEdit = (
                info[0]["Structure.Arrangement"]
            )
            self.structurerandomOrientationEdit = (
                info[0]["Structure.Orientation"]
            )

            self.konEdit = (info[0]["PAINT.k_on"])
            self.imagerconcentrationEdit = (info[0]["PAINT.imager"])
            self.taubEdit = (info[0]["PAINT.taub"])

            self.psfEdit = (info[0]["Imager.PSF"])
            self.photonrateEdit = (info[0]["Imager.Photonrate"])
            self.photonratestdEdit = (info[0]["Imager.Photonrate Std"])
            self.photonbudgetEdit = (info[0]["Imager.Photonbudget"])
            self.laserpowerEdit = (info[0]["Imager.Laserpower"])
            self.photonslopeEdit = (info[0]["Imager.Photonslope"])
            self.photonslopeStdEdit = (info[0]["Imager.PhotonslopeStd"])

            self.camerasizeEdit = (info[0]["Camera.Image Size"])
            self.integrationtimeEdit = (
                info[0]["Camera.Integration Time"]
            )
            self.framesEdit = (info[0]["Camera.Frames"])
            self.pixelsizeEdit = (info[0]["Camera.Pixelsize"])

            if ADVANCED_MODE:
                self.lasercEdit = (info[0]["Noise.Lasercoefficient"])
                self.imagercEdit = (info[0]["Noise.Imagercoefficient"])
                self.BgoffsetEdit = (info[0]["Noise.BackgroundOff"])

                self.EquationAEdit = (info[0]["Noise.EquationA"])
                self.EquationBEdit = (info[0]["Noise.EquationB"])
                self.EquationCEdit = (info[0]["Noise.EquationC"])
                self.BgStdoffsetEdit = (
                    info[0]["Noise.BackgroundStdOff"]
                )

            # SET POSITIONS
            handlexx = _np.asarray((info[0]["Structure.HandleX"]).split(","))
            handleyy = _np.asarray((info[0]["Structure.HandleY"]).split(","))
            handleex = _np.asarray((info[0]["Structure.HandleEx"]).split(","))
            handless = _np.asarray(
                (info[0]["Structure.HandleStruct"]).split(",")
            )

            handlexx = handlexx.astype(_np.float)
            handleyy = handleyy.astype(_np.float)
            handleex = handleex.astype(_np.float)
            handless = handless.astype(_np.float)

            handleex = handleex.astype(_np.int)
            handless = handless.astype(_np.int)

            handle3d = _np.asarray((info[0]["Structure.Handle3d"]).split(","))
            handle3d = handle3d.astype(_np.float)
            structure = _np.array(
                [handlexx, handleyy, handleex, handless, handle3d]
            )
            # 0 -> grid
            # 1 -> circle
            # 2 -> custom
            self.structurecombo = 2
            self.newstruct = structure
            # self.plotPositions()
            print("Settings loaded from: " + path)
        self.changePaint()

    def plotPositions(self):
        structurexx, structureyy, structureex, structure3d = (
            self.readStructure(self.config)
        )
        pixelsize = self.pixelsizeEdit
        structure = simulate.defineStructure(
            structurexx, structureyy, structureex, structure3d, pixelsize
        )

        number = self.structurenoEdit
        imageSize = self.camerasizeEdit
        frame = self.structureframeEdit
        arrangement = int(self.structurerandomEdit)
        # Get the position of each individual origami not the position of the blinking event of an origami
        # This is the position where each of the origami will be placed
        gridpos = simulate.generatePositions(
            number, imageSize, frame, arrangement
        )

        orientation = int(self.structurerandomOrientationEdit)
        incorporation = self.structureIncorporationEdit / 100
        exchange = 0

        # self.figure1.suptitle('Positions [Px]')
        self.figure1 = plt.figure()
        self.figure2 = plt.figure()
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        #ax1.hold(True)
        ax1.axis("equal")
        ax1.plot(self.newstruct[0, :], self.newstruct[1, :], "+")
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize - 2 * frame,
                imageSize - 2 * frame,
                linestyle="dashed",
                edgecolor="#000000",
                fill=False,  # remove background
            )
        )

        ax1.axes.set_xlim(0, imageSize)
        ax1.axes.set_ylim(0, imageSize)

        # PLOT first structure
        struct1 = self.newstruct[:, self.newstruct[3, :] == 0]

        noexchangecolors = len(set(struct1[2, :]))
        exchangecolors = list(set(struct1[2, :]))
        self.noexchangecolors = exchangecolors
        # self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        #ax1.hold(True)

        structurexx = struct1[0, :]
        structureyy = struct1[1, :]
        structureex = struct1[2, :]
        structurexx_nm = _np.multiply(
            structurexx - min(structurexx), pixelsize
        )
        structureyy_nm = _np.multiply(
            structureyy - min(structureyy), pixelsize
        )

        for i in range(0, noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0, len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx_nm[j])
                    plotyy.append(structureyy_nm[j])
            ax1.plot(plotxx, plotyy, "o")

            distx = round(1 / 10 * (max(structurexx_nm) - min(structurexx_nm)))
            disty = round(1 / 10 * (max(structureyy_nm) - min(structureyy_nm)))

            ax1.axes.set_xlim(
                (min(structurexx_nm) - distx, max(structurexx_nm) + distx)
            )
            ax1.axes.set_ylim(
                (min(structureyy_nm) - disty, max(structureyy_nm) + disty)
            )
        # self.canvas2.draw()
        self.figure2.show()

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
    print(s)
