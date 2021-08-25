import torch
import numpy as np


def evaluate(model, loss_fn, data_loader, metrics, config) -> (float, dict):
    model.eval()
    summary = []
    for data_batch, labels_batch in data_loader:
        data_batch, labels_batch = data_batch.to(config.device), labels_batch.to(config.device)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        output_batch = output_batch.detach()
        labels_batch = labels_batch.detach()

        # Compute all metrics on this batch

        summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        summary_batch['loss'] = loss.item()
        summary.append(summary_batch)
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    config.logger.info("Eval metrics : " + metrics_string)

    return float(loss.item()), metrics_mean