import torch
import numpy as np

import utils


def validation(model, data_loader, criterion, metrics, config) -> dict:
    model.eval()
    summary = []
    for inputs, labels in data_loader:
        inputs = utils.convert_device(inputs, config.device)
        labels = utils.convert_device(labels, config.device)
        # compute model output
        with torch.no_grad():
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)

        # Compute all metrics on this batch
        summary_batch = metrics(outputs, labels)
        summary_batch['loss'] = loss.item()
        summary.append(summary_batch)
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    config.logger.info("Eval metrics : " + metrics_string)

    return metrics_mean
