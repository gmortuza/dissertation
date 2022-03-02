import os

import glob

import pickle
import torch
import torch.nn.functional as F
from read_config import Config

# Constants
SINGLE_EMITTER_WIDTH = 20
SINGLE_EMITTER_HEIGHT = 20
EXPORTED_TRAIN_WIDTH = 40
EXPORTED_TRAIN_HEIGHT = 40
SCALE = 16.0


# Performs a breadth-first search to find the connected pixels for each emitter
def get_emitter_start_end_pos(frame, point) -> tuple:
    x_px, y_px = torch.round(point[[2, 1]] * SCALE).int().tolist()
    x_start, x_end, y_start, y_end = (x_px, x_px, y_px, y_px)
    visited = set()  # set of visited pixels tuple(x, y)
    queue = [(x_px, y_px)]
    while queue:
        x, y = queue.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if frame[x, y] > 0:
            x_start = min(x_start, x)
            x_end = max(x_end, x)
            y_start = min(y_start, y)
            y_end = max(y_end, y)

            # up, down, left, right
            queue.append((x + 1, y))
            queue.append((x - 1, y))
            queue.append((x, y + 1))
            queue.append((x, y - 1))
            # up-left, up-right, down-left, down-right
            queue.append((x + 1, y + 1))
            queue.append((x - 1, y - 1))
            queue.append((x - 1, y + 1))
            queue.append((x + 1, y - 1))
    # if the emitter is too small, we ignore it
    if x_end - x_start < 5 or y_end - y_start < 5:
        return [None] * 4
    # Make height and width even so that padding can be easier later
    if (x_end - x_start) % 2 != 0:
        x_start -= 1
    if (y_end - y_start) % 2 != 0:
        y_start -= 1
    return x_start, x_end, y_start, y_end


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
    points = []
    for point in gts:
        # extract the position from gt
        x_start, x_end, y_start, y_end = get_emitter_start_end_pos(frame, point)
        if x_start is None:
            continue
        x_mean, y_mean = (point[[2, 1]] * SCALE).tolist()
        # x_px, y_px = torch.round(point[[2, 1]] * SCALE).int().tolist()
        # x_start, x_end = x_px - SINGLE_EMITTER_WIDTH // 2, x_px + SINGLE_EMITTER_WIDTH // 2
        # y_start, y_end = y_px - SINGLE_EMITTER_HEIGHT // 2, y_px + SINGLE_EMITTER_HEIGHT // 2
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
            patches[intersect_point] = frame[x_start: x_end, y_start: y_end]
            # Add the current label in the gts
            label = [float(point[7] / 20000.), float(point[8]), float(point[9]), x_mean, y_mean, x_start, y_start]
            patches_gt[intersect_point].append(label)
        else:
            points.append((x_start, x_end, y_start, y_end))
            patch = frame[x_start: x_end, y_start: y_end]
            label = [float(point[7] / 20000.), float(point[8]), float(point[9]), x_mean, y_mean, x_start, y_start]
            patches.append(patch)
            patches_gt.append([label])
    # patches --> list of tensors that contains the emitters in the frame
    # patches_gt --> list of labels for each patch
    #                    [photon_count, std_x, std_y, x_mean, y_mean, x_start, y_start]
    return patches, patches_gt


def extract_label_from_folder(folder, config):
    file_names = glob.glob(f"{folder}/data_*_gt.pl")
    save_folder = os.path.join(folder, 'points')
    os.makedirs(save_folder, exist_ok=True)
    max_ = 0
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
            # normalize the frame between 0 and 1
            if frame.max() > 0:
                frame = (frame - frame.min()) / (frame.max() - frame.min())
            patch, location = extract_label_from_single_frame(frame, gt)
            # make all patches the same size
            # patch = F.pad(patch, (0, 0, 0, 0, 32, 32), mode='constant', value=0)
            for idx, (p, loc) in enumerate(zip(patch, location)):
                max_ = max(max_, p.shape[0])
                pad = ((EXPORTED_TRAIN_WIDTH - p.shape[1]) // 2, (EXPORTED_TRAIN_WIDTH - p.shape[1]) // 2,
                       (EXPORTED_TRAIN_HEIGHT - p.shape[0]) // 2, (EXPORTED_TRAIN_HEIGHT - p.shape[0]) // 2)
                single_patch = F.pad(p, pad, mode='constant', value=0)
                # create label
                labels = []
                for l in loc:
                    # calculate the percentage
                    x_per = (l[4] - (l[6] - pad[0])) / EXPORTED_TRAIN_WIDTH
                    y_per = (l[3] - (l[5] - pad[2])) / EXPORTED_TRAIN_HEIGHT
                    labels.extend([1, y_per, x_per, l[0], l[1], l[2]])
                if len(loc) == 1:
                    labels.extend([0, 0, 0, 0, 0, 0])

                label = torch.Tensor(labels[:12])

                file_name = os.path.join(save_folder, f"p_{frame_id}_{idx}.pl")
                with open(file_name, 'wb') as f:
                    pickle.dump([single_patch, label], f)
    print(f"Max size is {max_}")


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
