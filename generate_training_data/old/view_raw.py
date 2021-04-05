import numpy as np
import matplotlib.pyplot as plt
# from io_modified import load_raw
import pandas as pd
import os
import h5py

# images = load_raw("simulation.raw")[0]
# plt.imshow(images[5], cmap="gray")
# plt.show()
# file = h5py.File('test_locs.hdf5', 'r')
df_without_drift = pd.DataFrame(np.array(h5py.File("generated_data/test_ground_truth_without_drift.hdf5", "r")["locs"]))
df_without_frame = pd.DataFrame(np.array(h5py.File("generated_data/test_ground_truth_without_frame.hdf5", "r")["locs"]))
df_localized = pd.DataFrame(np.array(h5py.File("generated_data/test_locs.hdf5", "r")["locs"]))
print("hi there")
# new_df = pd.DataFrame(np.array(h5py.File("test_locs_after_drop.hdf5", "r")["locs"]))
# Drop the column lpx, lpy, ellipticity
# df.drop(columns=['lpx', 'lpy', 'ellipticity'], inplace=True)
# locs = np.array(df)
# # what we actually need
# new_locs = np.rec.array(
#     (
#         # locs[:, 0],
#         locs[:, 1],  # x in nanometer
#         locs[:, 2],  # y[nm]
#         locs[:, 3],  # photons_count
#         # locs[:, 4],
#         # locs[:, 5],
#         np.zeros_like(locs[:, 6]),  # background
#         np.full_like(locs[:, 7], .009),  # lpx
#         np.full_like(locs[:, 8], .009),  # lpy
#         # locs[:, 10],
#     ), dtype=[
#                 # ("frame", "u4"),
#                 ("x", "f4"),
#                 ("y", "f4"),
#                 ("photons", "f4"),
#                 # ("sx", "f4"),
#                 # ("sy", "f4"),
#                 ("bg", "f4"),
#                 ("lpx", "f4"),
#                 ("lpy", "f4"),
#                 # ("net_gradient", "f4"),
#             ])
# # sanity check
# # no inf or nan:
# # locs = locs[
# #     np.all(
# #         np.array([np.isfinite(locs[_]) for _ in locs.dtype.names]),
# #         axis=0,
# #     )
# # ]
# # locs = locs[locs['x'] > 0]
# # locs = locs[locs['x'] > 0]
# # locs = locs[locs['x'] < 32]
# # locs = locs[locs['y'] < 32]
# # locs = locs[locs['lpx'] > 0]
# # locs = locs[locs['lpy'] > 0]
# with h5py.File("test_locs_after_drop.hdf5", "w") as locs_file:
#     locs_file.create_dataset("locs", data=new_locs)
# # df.to_csv("test_locs_after_drop.csv", index=False)
