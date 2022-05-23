from picasso import io
from extract_location import localize
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import torch


file_path = '/data/golam/dnam_nn/simulated_data_multi/test/data_1_500_32_with_noise.pl'
# movie = np.fromfile(file_path)
# movie = movie.reshape(-1, 32, 32)

# movie = io.load_movie(file_path)
with open(file_path, 'rb') as f:
    movie = torch.load(f).cpu().numpy()

info = {
    'baseline': 100,
    'sensitive': 1.0,
    'gain': 1,
    'qe': .09,
    'sensitivity': 1,
}
parameters = {
    'Box Size': 7,
    'Min. Net Gradient': 7097,
    'Pixelsize': 109,
    'box': 7
}
# def localize(movie, info, parameters):
localized = localize.localize(movie[0], info, parameters)
df = pd.DataFrame(localized)
localized_arr = np.asarray(localized)
print("Done")



# import numpy as _np
#
# import extract_location.gaussmle as _gaussmle
#
# LOCS_DTYPE = [
#     ("frame", "u4"),
#     ("x", "f4"),
#     ("y", "f4"),
#     ("photons", "f4"),
#     ("sx", "f4"),
#     ("sy", "f4"),
#     ("bg", "f4"),
#     ("lpx", "f4"),
#     ("lpy", "f4"),
#     ("net_gradient", "f4"),
#     ("likelihood", "f4"),
#     ("iterations", "i4"),
# ]
#
# def local_maxima(frame, box):
#     """ Finds pixels with maximum value within a region of interest """
#     Y, X = frame.shape
#     maxima_map = _np.zeros(frame.shape, _np.uint8)
#     box_half = int(box / 2)
#     box_half_1 = box_half + 1
#     for i in range(box_half, Y - box_half_1):
#         for j in range(box_half, X - box_half_1):
#             local_frame = frame[
#                 i - box_half: i + box_half + 1,
#                 j - box_half: j + box_half + 1,
#             ]
#             flat_max = _np.argmax(local_frame)
#             i_local_max = int(flat_max / box)
#             j_local_max = int(flat_max % box)
#             if (i_local_max == box_half) and (j_local_max == box_half):
#                 maxima_map[i, j] = 1
#     y, x = _np.where(maxima_map)
#     return y, x
#
#
# def gradient_at(frame, y, x, i):
#     gy = frame[y + 1, x] - frame[y - 1, x]
#     gx = frame[y, x + 1] - frame[y, x - 1]
#     return gy, gx
#
#
# def net_gradient(frame, y, x, box, uy, ux):
#     box_half = int(box / 2)
#     ng = _np.zeros(len(x), dtype=_np.float32)
#     for i, (yi, xi) in enumerate(zip(y, x)):
#         for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
#             for l_index, m in enumerate(
#                 range(xi - box_half, xi + box_half + 1)
#             ):
#                 if not (k == yi and m == xi):
#                     gy, gx = gradient_at(frame, k, m, i)
#                     ng[i] += (
#                         gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
#                     )
#     return ng
#
#
# def identify_in_image(image, minimum_ng, box):
#     y, x = local_maxima(image, box)
#     box_half = int(box / 2)
#     # Now comes basically a meshgrid
#     ux = _np.zeros((box, box), dtype=_np.float32)
#     uy = _np.zeros((box, box), dtype=_np.float32)
#     for i in range(box):
#         val = box_half - i
#         ux[:, i] = uy[i, :] = val
#     unorm = _np.sqrt(ux ** 2 + uy ** 2)
#     ux /= unorm
#     uy /= unorm
#     ng = net_gradient(image, y, x, box, uy, ux)
#     positives = ng > minimum_ng
#     y = y[positives]
#     x = x[positives]
#     ng = ng[positives]
#     return y, x, ng
#
#
# def identify_in_frame(frame, minimum_ng, box, roi=None):
#     if roi is not None:
#         frame = frame[roi[0][0]: roi[1][0], roi[0][1]: roi[1][1]]
#     image = _np.float32(frame)  # otherwise numba goes crazy
#     y, x, net_gradient = identify_in_image(image, minimum_ng, box)
#     if roi is not None:
#         y += roi[0][0]
#         x += roi[0][1]
#     return y, x, net_gradient
#
#
# def identify_by_frame_number(movie, minimum_ng, box, frame_number, roi=None):
#     frame = movie[frame_number]
#     y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
#     frame = frame_number * _np.ones(len(x))
#     return _np.rec.array(
#         (frame, x, y, net_gradient),
#         dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
#     )
#
#
# def identify(movie, minimum_ng, box):
#     identifications = [
#         identify_by_frame_number(movie, minimum_ng, box, i)
#         for i in range(len(movie))
#     ]
#     return _np.hstack(identifications).view(_np.recarray)
#
# def _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box):
#     n_spots = len(ids_x)
#     r = int(box / 2)
#     spots = _np.zeros((n_spots, box, box), dtype=movie.dtype)
#     for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
#         spots[id] = movie[frame, yc - r: yc + r + 1, xc - r: xc + r + 1]
#     return spots
#
#
# def _cut_spots_frame(
#     frame, frame_number, ids_frame, ids_x, ids_y, r, start, N, spots
# ):
#     for j in range(start, N):
#         if ids_frame[j] > frame_number:
#             break
#         yc = ids_y[j]
#         xc = ids_x[j]
#         spots[j] = frame[yc - r: yc + r + 1, xc - r: xc + r + 1]
#     return j
#
#
# def _cut_spots(movie, ids, box):
#     if isinstance(movie, _np.ndarray):
#         return _cut_spots_numba(movie, ids.frame, ids.x, ids.y, box)
#     else:
#         """ Assumes that identifications are in order of frames! """
#         r = int(box / 2)
#         N = len(ids.frame)
#         spots = _np.zeros((N, box, box), dtype=movie.dtype)
#         start = 0
#         for frame_number, frame in enumerate(movie):
#             start = _cut_spots_frame(
#                 frame,
#                 frame_number,
#                 ids.frame,
#                 ids.x,
#                 ids.y,
#                 r,
#                 start,
#                 N,
#                 spots,
#             )
#         return spots
#
#
# def _to_photons(spots, camera_info):
#     spots = _np.float32(spots)
#     baseline = camera_info["baseline"]
#     sensitivity = camera_info["sensitivity"]
#     gain = camera_info["gain"]
#     qe = camera_info["qe"]
#     return (spots - baseline) * sensitivity / (gain * qe)
#
# def get_spots(movie, identifications, box, camera_info):
#     spots = _cut_spots(movie, identifications, box)
#     return _to_photons(spots, camera_info)
#
# def fit(
#     movie,
#     camera_info,
#     identifications,
#     box,
#     eps=0.001,
#     max_it=100,
#     method="sigma",
# ):
#     spots = get_spots(movie, identifications, box, camera_info)
#     theta, CRLBs, likelihoods, iterations = _gaussmle.gaussmle(
#         spots, eps, max_it, method=method
#     )
#     return locs_from_fits(
#         identifications, theta, CRLBs, likelihoods, iterations, box
#     )
#
#
# def locs_from_fits(
#     identifications, theta, CRLBs, likelihoods, iterations, box
# ):
#     box_offset = int(box / 2)
#     y = theta[:, 0] + identifications.y - box_offset
#     x = theta[:, 1] + identifications.x - box_offset
#     lpy = _np.sqrt(CRLBs[:, 0])
#     lpx = _np.sqrt(CRLBs[:, 1])
#     locs = _np.rec.array(
#         (
#             identifications.frame,
#             x,
#             y,
#             theta[:, 2],
#             theta[:, 5],
#             theta[:, 4],
#             theta[:, 3],
#             lpx,
#             lpy,
#             identifications.net_gradient,
#             likelihoods,
#             iterations,
#         ),
#         dtype=LOCS_DTYPE,
#     )
#     locs.sort(kind="mergesort", order="frame")
#     return locs
#
#
# def localize(movie, info, parameters):
#     print("localizing")
#     identifications = identify(movie, parameters)
#     return fit(movie, info, identifications, parameters["Box Size"])
#
#
# if __name__ == '__main__':
#
#     localize(movie, info, parameters)