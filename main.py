import torch
from torch.optim import Adam

from train import train
from validation import validation
from metrics.metrics import get_metrics
from losses.loss import Loss
from models.get_model import get_model
from read_config import Config
from data_loader import fetch_data_loader
import utils


def main(config: Config):
    criterion = Loss(config)
    model = get_model(config)
    config.log_param('model_params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    metrics = get_metrics(config)
    train_loader, val_loader = fetch_data_loader(config)
    best_val_acc = utils.load_checkpoint(model, config, optimizer)

    for epoch in range(1, config.num_epochs + 1):
        config.logger.info(f"Epoch {epoch}/{config.num_epochs}")
        train_metrics = train(model, train_loader, criterion, optimizer, metrics, config)
        for key, val in train_metrics.items():
            config.neptune["training/epoch/" + key].log(val)

        val_metrics = validation(model, val_loader, criterion, metrics, config)
        for key, val in val_metrics.items():
            config.neptune["validation/epoch/" + key].log(val)

        best_val_acc = utils.save_checkpoint({'epoch': epoch,
                                              'state_dict': model.state_dict(),
                                              'optim_dict': optimizer.state_dict()},
                                             best_val_acc,
                                             config, val_metrics)


if __name__ == '__main__':
    config_ = Config("config.yaml")
    main(config_)
