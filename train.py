import os
import torch
from read_config import Config
import torch.optim as optim
import utils
from models.get_model import get_model
from models.loss import dNamNNLoss
from models.metrics import metrics
from models.evaluate import evaluate
from data_loader import fetch_data_loader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np

scaler = torch.cuda.amp.GradScaler()


def train(model: torch.nn.Module, optimizer: torch.optim, loss_fn, train_data_loader: torch.utils.data.DataLoader,
          metrics, config: Config) -> (float, float):
    model.train()
    summary = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(train_data_loader)) as progress_bar:
        for i, (train_batch, labels_batch) in enumerate(train_data_loader):

            # Model output and it's loss
            with torch.cuda.amp.autocast(enabled=True):
                output_batch = model(train_batch)
                loss = loss_fn(output_batch, labels_batch)
            # loss = loss_fn(output_batch, train_batch)
            # clear previous gradients,  computer gradients of all variable wrt loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % config.save_summary_steps == 0:
                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
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

    return loss_avg(), metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, config):
    best_val_acc = float('-inf')
    if config.load_checkpoint:
        config.logger.info(f"Restoring parameters from {config.checkpoint_dir}")
        best_val_acc = utils.load_checkpoint(config.checkpoint_dir, model, config, optimizer)

    for epoch in range(config.num_epochs):
        config.logger.info("Epoch {}/{}".format(epoch + 1, config.num_epochs))
        train_loss, train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, config)

        # Write into tensorboard
        # config.tensor_board_writer.add_scalar("loss/train", train_loss, epoch)
        # config.tensor_board_writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)

        val_loss, val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, config)

        config.tensor_board_writer.add_scalars(f'loss', {
            'validation': val_loss,
            'training': train_loss,
        }, epoch)

        config.tensor_board_writer.add_scalars(f'accuracy', {
            'validation': val_metrics["accuracy"],
            'training': train_metrics["accuracy"],
        }, epoch)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=config.checkpoint_dir)

        # If best_eval, best_save_path
        if is_best:
            config.logger.info("Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                config.checkpoint_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            config.checkpoint_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    config = Config("config.yaml")
    config.tensor_board_writer = SummaryWriter(config.tensorflow_log_dir)
    model = get_model(config)
    loss_fn = dNamNNLoss(config)
    train_data_loader, val_data_loader = fetch_data_loader(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # train_and_evaluate(model, train_data_loader, val_data_loader, optimizer, loss_fn, metrics, config)
    from models.srgan.train import train_evaluation
    train_evaluation()
