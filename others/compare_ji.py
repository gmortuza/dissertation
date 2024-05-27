import pandas as pd
import collections
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np
# from ..metrics import metrics


#%%
def get_metrics(prediction_file):
    radiuses = range(10, 251, 10)
    ground_truth = pd.read_csv("/Users/gmortuza/Desktop/dnam_data/origami_data/test/combined.csv")
    predicted = pd.read_csv(prediction_file)
    # predicted['x'] += 20
    # predicted['y'] += 70
    true_positive = collections.defaultdict(lambda: 0)
    false_positive = collections.defaultdict(lambda: 0)
    ji = {}
    rmse = {}
    efficiency = {}
    alpha = .5
    distances_from_points = collections.defaultdict(lambda: [])
    total = 0

    for i in range(0, 20001):
        gt = ground_truth[ground_truth['frame'] == i]
        pred = predicted[predicted['frame'] == i]
        if len(gt) == 0 or len(pred) == 0:
            continue
        total += len(gt) + len(pred)
        distances = pairwise_distances(pred[['x', 'y']].values, gt[['x', 'y']].values)
        rec_ind, gt_ind = linear_sum_assignment(distances)
        assigned_distance = distances[rec_ind, gt_ind]
        for radius in radiuses:
            true_positive[radius] += np.sum(assigned_distance <= radius)
            false_positive[radius] += np.sum(assigned_distance > radius)
            distances_from_points[radius].extend(assigned_distance[assigned_distance <= radius].tolist())
    for radius in radiuses:
        ji[radius] = true_positive[radius] * 100 / (total - true_positive[radius])

        if len(distances_from_points[radius]) > 0:
            distances_from_point = np.asarray(distances_from_points[radius])
            rmse[radius] = np.sqrt(np.sum(distances_from_point ** 2) / len(distances_from_point))
        efficiency[radius] = 100 - ((100 - ji[radius]) ** 2 + (alpha ** 2 * rmse[radius] ** 2)) ** .5
    return ji, rmse, efficiency


our = "/Users/gmortuza/Desktop/dnam_data/origami_data/our/output_jun_7/output_picasso_512.csv"
thunderstorm = "/Users/gmortuza/Desktop/dnam_data/origami_data/thunderstorm/combined.csv"
picasso = "/Users/gmortuza/Desktop/dnam_data/origami_data/picasso/combined.csv"

metrics = get_metrics(thunderstorm)
results = []
for radius in metrics[0].keys():
    results.append([radius, metrics[0][radius], metrics[1][radius], metrics[2][radius]])

df = pd.DataFrame(results, columns=['radius', 'ji', 'rmse', 'efficiency'])
df.to_csv(thunderstorm.replace(".csv", "_compare.csv"), index=False)
print("Done")
