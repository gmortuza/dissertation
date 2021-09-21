from datetime import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
import torch
from PIL import Image
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment


def crop(image, crop_size=32):
    row_partial = int(image.shape[0] / crop_size)
    col_partial = int(image.shape[1] / crop_size)
    cropped_empty_image = np.empty(shape=(row_partial, col_partial, crop_size, crop_size))
    for row in range(row_partial):
        for col in range(col_partial):
            x_start = row * crop_size
            y_start = col * crop_size
            cropped_empty_image[row][col] = image[x_start: x_start + crop_size, y_start: y_start + crop_size]
    return cropped_empty_image


def main():
    fname = '/Users/golammortuza/Downloads/20190913_All-Matrices_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM 13_42_03.spe'
    image_reader = imageio.get_reader(fname)
    single_image = np.asarray(image_reader.get_data(0))
    plt.imshow(single_image, cmap='gray')
    plt.show()
    crop_size = 32
    # cropped_image = crop(single_image, crop_size=32).reshape(-1, crop_size, crop_size)
    # image = Image.fromarray(cropped_image)
    images = []
    for i in range(100):
        images.append(np.asarray(image_reader.get_data(i)))
    images = np.asarray(images)
    images.tofile("images.raw")
    # imageio.mimwrite("test.tiff", cropped_image, format='TIFF')


# Counter
def counter_in_frame(localized_file):
    content = h5py.File(localized_file, 'r')
    frames = content['locs']['frame']
    # x = content['locs']['x']
    # y = content['locs']['y']
    # photons = content['locs']['photons']
    unique, count = np.unique(frames, return_counts=True)
    print(f"average on-event per frame is: {sum(count) / len(count)}")


def get_jaccard_index(prediction, label, radius):

    true_positive = 0
    # only nearby points in a single frame can be considered as true positive
    # iter over each frame
    for frame in np.unique(prediction[:, 0]):
        frame_prediction = prediction[prediction[:, 0] == frame+1][:, 1:] + [65, 65]
        frame_label = label[label[:, 0] == frame][:, 1:]
        if len(frame_prediction) > 0 and len(frame_label) > 0:
            pairwise_distance = pairwise_distances(frame_prediction, frame_label)
            prediction_ind, label_ind = linear_sum_assignment(pairwise_distance)
            for pred, lab in zip(prediction_ind, label_ind):
                if pairwise_distance[pred, lab] < radius:

                    true_positive += 1
                # print(f"x distance {frame_prediction[pred][0] - frame_label[lab][0]}")
                # print(f"y distance {frame_prediction[pred][1] - frame_label[lab][1]}")
    pred_len = prediction.shape[0]
    label_len = label.shape[0]
    return true_positive / (pred_len + label_len - true_positive)


def get_accuracy(prediction_file, label_file, radius=3):
    prediction = h5py.File(prediction_file, 'r')['locs']
    label = h5py.File(label_file, 'r')['locs']
    prediction = np.column_stack(np.asarray([prediction['frame'] + 1, prediction['x'] * 130, prediction['y'] * 130]))
    label = np.column_stack(np.asarray([label['frame'], label['x'] * 130, label['y'] * 130]))
    return get_jaccard_index(prediction, label, radius)


if __name__ == '__main__':
    # main()
    # counter_in_frame(localized_file='/Users/golammortuza/Desktop/Substack (1-200)_locs.hdf5')
    # counter_in_frame('/Users/golammortuza/workspace/nam/dnam_nn/simulated_data/test/data_1_2000_gt_without_drift.hdf5')
    # Get x, y points
    # prediction_file = '/Users/golammortuza/workspace/nam/dnam_nn/simulated_data/test/imagejresult_locs.hdf5'
    prediction_file = '/Users/golammortuza/workspace/nam/dnam_nn/simulated_data/test/data_1_4000_locs.hdf5'
    label_file = '/Users/golammortuza/workspace/nam/dnam_nn/simulated_data/test/data_1_4000_gt_without_drift.hdf5'
    for radius in range(1, 11):
        print(f"Accuracy for radius {radius} is: {round(get_accuracy(prediction_file, label_file, radius) * 100, 2)}%")