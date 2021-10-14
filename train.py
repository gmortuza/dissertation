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
import neptune.new as neptune

from tqdm import tqdm
import numpy as np

# seed = 0
# torch.use_deterministic_algorithms(True)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.cuda.manual_seed_all(seed)

scaler = torch.cuda.amp.GradScaler()

neptune_run = neptune.init(
    project="golammdmortuza/dnam-nn",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMjQyYzgzOC04MWVjLTRkZTMtYTExZC1kMGEzMDllNDZmZTcifQ==",
    name="test name",
    tags=['normalizing factor'],
    mode="debug"  # debug stop tracking
)
# neptune_run.create_experiment(name="test name")

def train(model: torch.nn.Module, optimizer: torch.optim, loss_fn, train_data_loader: torch.utils.data.DataLoader,
          lr_scheduler, config: Config) -> (float, float):
    model.train()
    summary = []
    loss_avg = utils.RunningAverage()
    # train_batch, labels_batch = next(iter(train_data_loader))
    with tqdm(total=len(train_data_loader), disable=config.progress_bar_disable) as progress_bar:
        for i, (train_batch, labels_batch) in enumerate(train_data_loader):
            train_batch = [tb.to(config.device) for tb in train_batch]
            labels_batch = [lb.to(config.device) for lb in labels_batch]

            # Model output and it's loss
            # with torch.cuda.amp.autocast(enabled=True):
            output_batch = model(train_batch, labels_batch)
            loss = loss_fn(output_batch, labels_batch)
            # loss = loss_fn(output_batch, train_batch)
            # clear previous gradients,  computer gradients of all variable wrt loss

            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            if i % config.save_summary_steps == 0:
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
    metrics_string = " ; ".join("{}: {:05.9f}".format(k, v)
                                for k, v in metrics_mean.items())
    config.logger.info("Train metrics: " + metrics_string)

    return loss_avg(), metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, lr_scheduler, config):
    best_val_acc = utils.load_checkpoint(model, config, optimizer)

    for epoch in range(config.num_epochs):
        config.logger.info("Epoch {}/{}".format(epoch + 1, config.num_epochs))
        train_loss, train_metrics = train(model, optimizer, loss_fn, train_dataloader, lr_scheduler, config)

        # neptune_run["training/batch/loss"].log(train_loss)
        for key, val in train_metrics.items():
            neptune_run["training/batch/" + key].log(val)


        # Write into tensorboard
        config.tensor_board_writer.add_scalar("loss/train", train_loss, epoch)
        config.tensor_board_writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)

        # val_loss, val_metrics = evaluate(model, loss_fn, val_dataloader, config)
        #
        # config.tensor_board_writer.add_scalars(f'loss', {
        #     'validation': val_loss,
        #     'training': train_loss,
        # }, epoch)
        #
        # config.tensor_board_writer.add_scalars(f'accuracy', {
        #     'validation': val_metrics["accuracy"],
        #     'training': train_metrics["accuracy"],
        # }, epoch)
        #
    val_metrics = {
        'accuracy': .95
    }
    best_val_acc = utils.save_checkpoint({'epoch': epoch + 1,
                                          'state_dict': model.state_dict(),
                                          'optim_dict': optimizer.state_dict()},
                                         best_val_acc,
                                         config, val_metrics)


if __name__ == '__main__':
    config = Config("config.yaml")
    config.tensor_board_writer = SummaryWriter(config.tensorflow_log_dir)
    model = get_model(config)
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {model_param}")
    loss_fn = dNamNNLoss(config)
    train_data_loader, val_data_loader = fetch_data_loader(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)
    lr_scheduler = None
    neptune_param = {
        "learning_rate": config.learning_rate,
        "final_activation": 'relu',
        "epochs": config.num_epochs,
        "total_training_example": config.total_training_example,
        "criterion": str(loss_fn),
        "upsample_method": 'transposed',
        "upsampled_activation": "relu",
        "model_param": model_param,
        "input_normalize_factor": -1,
        "output_normalize_factor": 10.,
        "threshold": 50.
    }
    neptune_run['config/hyperparameters'] = neptune_param
    train_and_evaluate(model, train_data_loader, val_data_loader, optimizer, loss_fn, lr_scheduler, config)
    neptune_run.stop()
    # from models.srgan.train import train_evaluation
    # train_evaluation()
