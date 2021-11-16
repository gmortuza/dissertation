import torch

import utils

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# seed = 0
# torch.use_deterministic_algorithms(True)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.cuda.manual_seed_all(seed)


def train(model, data_loader, criterion, optimizer, metrics, config) -> dict:
    model.train()
    summary = []
    loss_avg = utils.RunningAverage()
    # train_batch, labels_batch = next(iter(train_data_loader))
    with tqdm(total=len(data_loader), disable=config.progress_bar_disable) as progress_bar:
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = utils.convert_device(inputs, config.device)
            labels = utils.convert_device(labels, config.device)

            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % config.save_summary_steps == 0:
                summary_batch = {metric: metrics[metric](outputs, labels)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summary.append(summary_batch)

                # update the average loss
            loss_avg.update(loss.item())

            progress_bar.set_postfix(loss='{:05.9f}'.format(loss_avg()))
            progress_bar.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    config.logger.info("Train metrics: " + metrics_string)

    return metrics_mean


def test():
    train()


if __name__ == '__main__':
    test()
