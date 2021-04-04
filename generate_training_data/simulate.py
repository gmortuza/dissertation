"""
    picasso.simulate
    ~~~~~~~~~~~~~~~~

    Simulate single molcule fluorescence data

    :author: Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import numpy as _np
import io_modified as _io
from numba import njit

magfac = 0.79


@njit
def calculate_zpsf(z, cx, cy):
    z = z / magfac
    z2 = z * z
    z3 = z * z2
    z4 = z * z3
    z5 = z * z4
    z6 = z * z5
    wx = (
        cx[0] * z6
        + cx[1] * z5
        + cx[2] * z4
        + cx[3] * z3
        + cx[4] * z2
        + cx[5] * z
        + cx[6]
    )
    wy = (
        cy[0] * z6
        + cy[1] * z5
        + cy[2] * z4
        + cy[3] * z3
        + cy[4] * z2
        + cy[5] * z
        + cy[6]
    )
    return (wx, wy)


def test_calculate_zpsf():
    cx = _np.array([1, 2, 3, 4, 5, 6, 7])
    cy = _np.array([1, 2, 3, 4, 5, 6, 7])
    z = _np.array([1, 2, 3, 4, 5, 6, 7])
    wx, wy = calculate_zpsf(z, cx, cy)

    result = [4.90350522e+01,   7.13644987e+02,   5.52316597e+03,
              2.61621620e+04,   9.06621337e+04,   2.54548124e+05,
              6.14947219e+05]

    delta = wx - result
    assert sum(delta**2) < 0.001


def saveInfo(filename, info):
    _io.save_info(filename, [info], default_flow_style=True)


def noisy(image, mu, sigma):
    """
    Add gaussian noise to an image.
    """
    row, col = image.shape  # Variance for _np.random is 1
    gauss = sigma * _np.random.normal(0, 1, (row, col)) + mu
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy[noisy < 0] = 0
    return noisy


def noisy_p(image, mu):
    """
    # Add poissonian noise to an image or movie
    """
    poiss = _np.random.poisson(mu, image.shape).astype(float)
    noisy = image + poiss
    return noisy


def check_type(movie):
    movie[movie >= (2 ** 16) - 1] = (2 ** 16) - 1
    movie = movie.astype("<u2")  # little-endian 16-bit unsigned int
    return movie


def paintgen(
    meandark, meanbright, frames, time, photonrate, photonratestd, photonbudget, always_on
):
    """
    Paint-Generator:
    Generates on and off-traces for given parameters.
    Calculates the number of Photons in each frame for a binding site.
    """
    num_of_blinking_event = 4 * int(
        _np.ceil(frames * time / (meandark + meanbright))
    )  # This is an estimate for the total number of binding events
    if num_of_blinking_event < 10:
        num_of_blinking_event = num_of_blinking_event * 10

    if always_on > 0:
        print("Insdie paintgen always on")
        return _np.random.normal(always_on / frames, photonratestd, frames).clip(min=0), _np.zeros(frames,
                                                                                                   dtype=_np.float64), [
                   0] * 4

    dark_times = _np.random.exponential(meandark, num_of_blinking_event)
    bright_times = _np.random.exponential(meanbright, num_of_blinking_event)

    events = _np.vstack((dark_times, bright_times)).reshape(
        (-1,), order="F"
    )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
    eventsum = _np.cumsum(events)
    maxloc = _np.argmax(
        eventsum > (frames * time)
    )  # Find the first event that exceeds the total integration time
    simulatedmeandark = _np.mean(events[:maxloc:2])

    simulatedmeanbright = _np.mean(events[1:maxloc:2])

    # check trace
    if _np.mod(maxloc, 2):  # uneven -> ends with an OFF-event
        onevents = int(_np.floor(maxloc / 2))
    else:  # even -> ends with bright event
        onevents = int(maxloc / 2)
    bright_events = _np.floor(maxloc / 2)  # number of bright_events

    photonsinframe = _np.zeros(
        int(frames + _np.ceil(meanbright / time * 20))
    )  # an on-event might be longer than the movie, so allocate more memory

    # calculate photon numbers
    for i in range(1, maxloc, 2):
        if photonratestd == 0:
            photons = _np.round(photonrate * time)
        else:
            photons = _np.round(
                _np.random.normal(photonrate, photonratestd) * time
            )  # Number of Photons that are emitted in one frame

        if photons < 0:
            photons = 0

        tempFrame = int(
            _np.floor(eventsum[i - 1] / time)
        )  # Get the first frame in which something happens in on-event
        onFrames = int(
            _np.ceil((eventsum[i] - tempFrame * time) / time)
        )  # Number of frames in which photon emittance happens

        if photons * onFrames > photonbudget:
            onFrames = int(
                _np.ceil(photonbudget / (photons * onFrames) * onFrames)
            )  # Reduce the number of on-frames if the photonbudget is reached

        for j in range(0, (onFrames)):
            if onFrames == 1:  # CASE 1: all photons are emitted in one frame
                photonsinframe[1 + tempFrame] = int(
                    _np.random.poisson(
                        ((tempFrame + 1) * time - eventsum[i - 1])
                        / time
                        * photons
                    )
                )
            elif (
                onFrames == 2
            ):  # CASE 2: all photons are emitted in two frames
                emittedphotons = (
                    ((tempFrame + 1) * time - eventsum[i - 1]) / time * photons
                )
                if j == 1:  # photons in first onframe
                    photonsinframe[1 + tempFrame] = int(
                        _np.random.poisson(
                            ((tempFrame + 1) * time - eventsum[i - 1])
                            / time
                            * photons
                        )
                    )
                else:  # photons in second onframe
                    photonsinframe[2 + tempFrame] = int(
                        _np.random.poisson(
                            (eventsum[i] - (tempFrame + 1) * time)
                            / time
                            * photons
                        )
                    )
            else:  # CASE 3: all photons are mitted in three or more frames
                if j == 1:
                    photonsinframe[1 + tempFrame] = int(
                        _np.random.poisson(
                            ((tempFrame + 1) * time - eventsum[i - 1])
                            / time
                            * photons
                        )
                    )  # Indexing starts with 0
                elif j == onFrames:
                    photonsinframe[onFrames + tempFrame] = int(
                        _np.random.poisson(
                            (eventsum(i) - (tempFrame + onFrames - 1) * time)
                            / time
                            * photons
                        )
                    )
                else:
                    photonsinframe[tempFrame + j] = int(
                        _np.random.poisson(photons)
                    )

        totalphotons = _np.sum(
            photonsinframe[1 + tempFrame: tempFrame + 1 + onFrames]
        )
        if totalphotons > photonbudget:
            photonsinframe[onFrames + tempFrame] = int(
                photonsinframe[onFrames + tempFrame]
                - (totalphotons - photonbudget)
            )

    photonsinframe = photonsinframe[0:frames]
    timetrace = events[0:maxloc]

    if onevents > 0:
        spotkinetics = [
            onevents,
            sum(photonsinframe > 0),
            simulatedmeandark,
            simulatedmeanbright,
        ]
    else:
        spotkinetics = [0, sum(photonsinframe > 0), 0, 0]
    return photonsinframe, timetrace, spotkinetics


def distphotons(
    structures,
    itime,
    frames,
    taud,
    taub,
    photonrate,
    photonratestd,
    photonbudget,
    always_on=0,
):
    """
    Distrbute Photons
    """
    time = itime
    meandark = int(taud)
    meanbright = int(taub)

    bindingsitesx = structures[0, :]
    bindingsitesy = structures[1, :]
    nosites = len(bindingsitesx)

    photonposall = _np.zeros((2, 0))
    photonposall = [1, 1]

    photonsinframe, timetrace, spotkinetics = paintgen(
        meandark,
        meanbright,
        frames,
        time,
        photonrate,
        photonratestd,
        photonbudget,
        always_on
    )

    return photonsinframe, timetrace, spotkinetics


def dist_photons_xy(runner, photon_dist, structures, psf, mode3Dstate, cx, cy):

    binding_sites_x = structures[0, :]
    binding_sites_y = structures[1, :]
    binding_sites_z = structures[4, :]
    no_sites = len(binding_sites_x)  # number of binding sites in image

    temp_photons = _np.array(photon_dist[:, runner]).astype(int)
    n_photons = _np.sum(temp_photons)
    n_photons_step = _np.cumsum(temp_photons)
    n_photons_step = _np.insert(n_photons_step, 0, 0)

    # Allocate memory
    photon_pos_frame = _np.zeros((n_photons, 2))
    # Positions where are putting some photons
    gt_position = []
    for i in range(0, no_sites):
        photon_count = int(photon_dist[i, runner])
        if mode3Dstate:
            wx, wy = calculate_zpsf(binding_sites_z[i], cx, cy)
            cov = [[wx * wx, 0], [0, wy * wy]]
        else:
            # covariance matrix for the normal distribution
            cov = [[psf * psf, 0], [0, psf * psf]]

        if photon_count > 0:
            gt_position.append(i)
            mu = [binding_sites_x[i], binding_sites_y[i]]
            photon_pos = _np.random.multivariate_normal(mu, cov, photon_count)
            photon_pos_frame[
                n_photons_step[i]: n_photons_step[i + 1], :
            ] = photon_pos

    return photon_pos_frame, gt_position


def convertMovie(
    runner,
    photon_dist,
    config,
):
    edges = range(0, config["Height"] + 1)

    photon_pos_frame, gt_position = dist_photons_xy(
        runner, photon_dist, config["new_struct"], config["Imager.PSF"], config["origami_3d"],
        config["Structure.CX"], config["Structure.CY"]
    )

    if len(photon_pos_frame) == 0:
        # There is no photon allocated in this frame
        # So we will return empty image
        return _np.zeros((config["Height"], config["Width"])), gt_position
    else:
        # Insert the drift in here
        x = photon_pos_frame[:, 0] + config["drift_x"][runner]
        y = photon_pos_frame[:, 1] + config["drift_y"][runner]
        # TODO: Not sure why it is y, x need to check this
        simulated_frame, _, _ = _np.histogram2d(y, x, bins=(edges, edges))
        return simulated_frame, gt_position
        # Because of this flip ground truth doesn't match with the render text
        # The origami might be fippled anyway. So this doesn't matter
        # Disabling this to have the exact match of the ground truth vs simulated data
        # simulated_frame = _np.flipud(simulated_frame)  # to be consistent with render


def saveMovie(filename, movie, info):
    _io.save_raw(filename, movie, [info])


# Function to store the coordinates of a structure in a container.
# The coordinates wil be adjusted so that the center of mass is the origin
def defineStructure(
    structurexxpx,
    structureyypx,
    structureex,
    structure3d,
    pixelsize,
    mean=True,
):
    if mean:
        structurexxpx = structurexxpx - _np.mean(structurexxpx)
        structureyypx = structureyypx - _np.mean(structureyypx)
    # from px to nm
    structurexx = []
    for x in structurexxpx:
        structurexx.append(x / pixelsize)
    structureyy = []
    for x in structureyypx:
        structureyy.append(x / pixelsize)

    structure = _np.array(
        [structurexx, structureyy, structureex, structure3d]
    )  # FORMAT: x-pos,y-pos,exchange information

    return structure


def generatePositions(number, imagesize, frame, arrangement):
    """
    Generate a set of positions where structures will be placed
    """
    if arrangement == 0:
        spacing = int(_np.ceil((number ** 0.5)))
        linpos = _np.linspace(frame, imagesize - frame, spacing)
        [xxgridpos, yygridpos] = _np.meshgrid(linpos, linpos)
        xxgridpos = _np.ravel(xxgridpos)
        yygridpos = _np.ravel(yygridpos)
        xxpos = xxgridpos[0:number]
        yypos = yygridpos[0:number]
        gridpos = _np.vstack((xxpos, yypos))
        gridpos = _np.transpose(gridpos)
    else:
        gridpos = (imagesize - 2 * frame) * _np.random.rand(number, 2) + frame

    return gridpos


def rotateStructure(structure):
    """
    Rotate a structure randomly
    """
    angle_rad = _np.random.rand(1) * 2 * _np.pi
    newstructure = _np.array(
        [
            (structure[0, :]) * _np.cos(angle_rad)
            - (structure[1, :]) * _np.sin(angle_rad),
            (structure[0, :]) * _np.sin(angle_rad)
            + (structure[1, :]) * _np.cos(angle_rad),
            structure[2, :],
            structure[3, :],
        ]
    )
    return newstructure


def incorporateStructure(structure, incorporation):
    """
    Returns a subset of the strucutre to reflect incorporation of stpales
    """
    newstructure = structure[
        :, (_np.random.rand(structure.shape[1]) < incorporation)
    ]
    return newstructure


def randomExchange(pos):
    """
    Randomly shuffle exchange parameters for rnadom labeling
    """
    arraytoShuffle = pos[2, :]
    _np.random.shuffle(arraytoShuffle)
    newpos = _np.array([pos[0, :], pos[1, :], arraytoShuffle, pos[3, :]])
    return newpos


def prepareStructures(
    origamies, gridpos, orientation, number, incorporation, exchange, frame_padding, height, num_gold_nano_particle=0
):
    """
    prepareStructures:
    Input positions, the structure definition consider rotation etc.
    """
    for i in range(0, len(gridpos)):
        # for each grid position select a random origami and add that origami to that grid position
        # Origami id for this particular grid position
        origami = origamies[_np.random.randint(0, len(origamies))]
        old_structure = _np.array(
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

        newx = structure[0, :] + gridpos[i, 0]
        newy = structure[1, :] + gridpos[i, 1]
        newstruct = _np.array(
            [
                newx,
                newy,
                structure[2, :],
                structure[2, :] * 0 + i,
                structure[3, :],
            ]
        )
        if i == 0:
            newpos = newstruct
        else:
            newpos = _np.concatenate((newpos, newstruct), axis=1)

    # Choose some random position from the whole movie to put the always on event
    if num_gold_nano_particle > 0:
        fixed_structure = _np.array(
            [
                _np.random.uniform(low=frame_padding, high=height - frame_padding, size=num_gold_nano_particle),
                # considering height and width the same
                _np.random.uniform(low=frame_padding, high=height - frame_padding, size=num_gold_nano_particle),
                _np.ones(num_gold_nano_particle),
                _np.zeros(num_gold_nano_particle),
                _np.zeros(num_gold_nano_particle)
            ]
        )
        newpos = _np.concatenate((newpos, fixed_structure), axis=1)

    if exchange == 1:
        newpos = randomExchange(newpos)
    return newpos
