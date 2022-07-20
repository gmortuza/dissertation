import os
import random
import sys
sys.path.append("/data/golam/dnam_nn")

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train import train
from validation import validation
from metrics.metrics import get_metrics
from extract_location.loss import Loss
from extract_location.model import ExtractLocationModel

from read_config import Config
from extract_location.data_loader import fetch_data_loader
import utils





def main(config: Config):
    criterion = Loss(config)
    model = ExtractLocationModel(config).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, min_lr=1e-6)
    train_loader, val_loader = fetch_data_loader(config)
    best_val_acc = utils.load_checkpoint(model, config, optimizer, 'points')

    for epoch in range(1, config.num_epochs + 1):
        metrics = get_metrics(config, epoch, 'points')
        config.logger.info(f"Epoch {epoch}/{config.num_epochs}")
        train_metrics = train(model, train_loader, criterion, optimizer, metrics, config)
        for key, val in train_metrics.items():
            config.neptune["training/epoch/" + key].log(val)

        val_metrics = validation(model, val_loader, criterion, metrics, config)
        for key, val in val_metrics.items():
            config.neptune["validation/epoch/" + key].log(val)

        # scheduler.step(val_metrics['loss'])
        config.neptune['epoch/lr'].log(optimizer.param_groups[0]['lr'])


        best_val_acc = utils.save_checkpoint({'epoch': epoch,
                                              'state_dict': model.state_dict(),
                                              'optim_dict': optimizer.state_dict()},
                                             best_val_acc,
                                             config, val_metrics, 'points')
    best_val_acc = utils.save_checkpoint({'epoch': epoch,
                                          'state_dict': model.state_dict(),
                                          'optim_dict': optimizer.state_dict()},
                                         best_val_acc,
                                         config, train_metrics, 'points')


if __name__ == '__main__':
    from_borah = True if len(sys.argv) > 1 else False
    config_ = Config("../config.yaml", from_borah)
    config_.purpose = "train_point_extraction"
    if config_.use_seed:
        utils.set_seed(1)
    main(config_)
