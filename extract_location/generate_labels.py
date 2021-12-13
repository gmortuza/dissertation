import os

import glob

import pickle
import torch
import torch.nn.functional as F
from read_config import Config


def if_intersect(x_start, x_end, y_start, y_end, points):
    # check if the box intersects with any of the points
    for idx, (x1, x2, y1, y2) in enumerate(points):
        x_left = max(x_start, x1)
        y_top = max(y_start, y1)
        x_right = min(x_end, x2)
        y_bottom = min(y_end, y2)
        if x_right < x_left or y_bottom < y_top:
            continue
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        if intersection_area > 0:
            return idx
    return -1


def extract_label_from_single_frame(frame, gts):
    patches = []
    patches_gt = []
    scale = 16.0
    points = []
    for point in gts:
        # extract the position from gt
        x_mean, y_mean = (point[[2, 1]] * scale).tolist()
        x_px, y_px = torch.round(point[[2, 1]] * scale).int().tolist()
        x_start, x_end = x_px - 5, x_px + 5
        y_start, y_end = y_px - 5, y_px + 5
        intersect_point = if_intersect(x_start, x_end, y_start, y_end, points)
        if intersect_point != -1:
            # intersect_point collide with current point so add them together
            x_start = min(points[intersect_point][0], x_start)
            x_end = max(points[intersect_point][1], x_end)
            y_start = min(points[intersect_point][2], y_start)
            y_end = max(points[intersect_point][3], y_end)
            # Make height and width even so that padding can be easier later
            if (x_end - x_start) % 2 != 0:
                x_start -= 1
            if (y_end - y_start) % 2 != 0:
                y_start -= 1
            # update the old point
            points[intersect_point] = (x_start, x_end, y_start, y_end)
            # Update existing points
            for label in patches_gt[intersect_point]:
                label[1] = (label[-4] - x_start) / (x_end - x_start)
                label[0] = (label[-3] - y_start) / (y_end - y_start)
            patches[intersect_point] = frame[x_start: x_end, y_start: y_end]
            # Add the current label in the gts
            x = (x_mean - x_start) / (x_end - x_start)
            y = (y_mean - y_start) / (y_end - y_start)
            label = [y, x, float(point[7] / 20000.), float(point[8]), float(point[9]), x_mean, y_mean, x_start, y_start]
            patches_gt[intersect_point].append(label)
        else:
            points.append((x_start, x_end, y_start, y_end))
            patch = frame[x_start: x_end, y_start: y_end]
            x = (x_mean - x_start) / (x_end - x_start)
            y = (y_mean - y_start) / (y_end - y_start)
            label = [y, x, float(point[7] / 20000.), float(point[8]), float(point[9]), x_mean, y_mean, x_start, y_start]
            patches.append(patch)
            patches_gt.append([label])
    return patches, patches_gt


def extract_label_from_folder(folder, config):
    file_names = glob.glob(f"{folder}/data_*_gt.pl")
    save_folder = os.path.join(folder, 'points')
    os.makedirs(save_folder, exist_ok=True)
    for file_name in file_names:
        patches = []
        locations = []
        start = int(file_name.split('_')[-3]) - 1
        gts = torch.load(file_name)
        data = torch.load(file_name.replace('_gt', '_' + str(config.resolution_slap[-1])), )
        # for each frame extract it's labels
        for frame_id, frame in enumerate(data, start=start):
            # get gt
            gt = gts[gts[:, 0] == frame_id]
            patch, location = extract_label_from_single_frame(frame, gt)
            # make all patches the same size
            # patch = F.pad(patch, (0, 0, 0, 0, 32, 32), mode='constant', value=0)
            for idx, (p, loc) in enumerate(zip(patch, location)):
                pad = ((20 - p.shape[1]) // 2, (20 - p.shape[1]) // 2, (20 - p.shape[0]) // 2, (20 - p.shape[0]) // 2)
                single_patch = F.pad(p, pad, mode='constant', value=0)
                # create label
                label = [1, loc[0][0], loc[0][1], loc[0][3], loc[0][4], loc[0][2]]
                if len(loc) > 1:
                    label.extend([1, loc[1][0], loc[1][1], loc[1][3], loc[1][4], loc[1][2]])
                else:
                    label.extend([0, 0, 0, 0, 0, 0])

                label = torch.Tensor(label)

                file_name = os.path.join(save_folder, f"p_{frame_id}_{idx}.pl")
                with open(file_name, 'wb') as f:
                    pickle.dump([single_patch, label], f)


def main(config):
    # Read data
    train_dir = os.path.join(config.input_dir, 'train')
    val_dir = os.path.join(config.input_dir, 'validation')
    extract_label_from_folder(train_dir, config)
    extract_label_from_folder(val_dir, config)
    # Extract location


if __name__ == '__main__':
    config_ = Config('../config.yaml')
    main(config_)
