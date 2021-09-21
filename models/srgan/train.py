import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.srgan.discriminator import Discriminator
from models.srgan.generator import Generator
from models.srgan.loss import fetch_disc_loss, fetch_gen_loss
from torch import optim
from data_loader import fetch_data_loader
from read_config import Config
import matplotlib.pyplot as plt
from models.metrics import normalized_cross_correlation
import utils
import numpy as np

config = Config('config.yaml')
config.tensor_board_writer = SummaryWriter(config.tensorflow_log_dir)

def train(gen, disc: nn.Module, disc_optim: torch.optim, gen_optim: torch.optim, gen_loss_fn, disc_loss_fn, train_data_loader: torch.utils.data.DataLoader):
    gen_loss_history = utils.RunningAverage()
    disc_loss_history = utils.RunningAverage()
    accuracy_history = utils.RunningAverage()
    summary = []

    with tqdm(total=len(train_data_loader)) as progress_bar:
        for idx, (low_res, high_res) in enumerate(train_data_loader):
            low_res = low_res.to(config.device)
            high_res = high_res.to(config.device)

            fake_img = gen(low_res)

            # plt.imshow(low_res.detach().cpu()[0][0], cmap='gray')
            # plt.title("input")
            # plt.show()
            # plt.imshow(fake_img.detach().cpu()[0][0], cmap='gray')
            # plt.title("fake")
            # plt.show()

            # Train Discriminator
            disc_real = disc(high_res)
            disc_fake = disc(fake_img.detach())
            disc_loss = disc_loss_fn(disc_fake, disc_real)

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            # train Generator
            disc_fake = disc(fake_img)
            gen_loss = gen_loss_fn(disc_fake, fake_img, high_res)
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            gen_loss_history.update(gen_loss.item())
            disc_loss_history.update(disc_loss.item())
            accuracy_history.update(normalized_cross_correlation(fake_img, high_res))

            bar_text = f'gen : {round(gen_loss_history(), 3)}, disc : {round(disc_loss_history(), 3)} ' \
                       f'acc {round(accuracy_history(), 3)}'
            # if idx % config.save_summary_steps == 0:
                # compute all metrics on this batch
                # summary_batch = {metric: metrics[metric](fake_img, high_res)
                #                  for metric in metrics}
                # summary_batch['gen_loss'] = gen_loss_history()
                # summary_batch['disc_loss'] = disc_loss_history()
                # summary.append(summary_batch)
            progress_bar.set_postfix(loss=bar_text)
            progress_bar.update()

    return gen_loss_history(), disc_loss_history(), accuracy_history()


def evaluation(gen, disc, gen_loss_fn, disc_loss_fn, eval_data_loader):
    gen.eval()
    disc.eval()
    gen_loss_history = utils.RunningAverage()
    disc_loss_history = utils.RunningAverage()
    accuracy_history = utils.RunningAverage()
    for low_res, high_res in eval_data_loader:
        low_res = low_res.to(config.device)
        high_res = high_res.to(config.device)
        with torch.no_grad():
            fake_img = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake_img.detach())
            disc_loss = disc_loss_fn(disc_fake, disc_real)
            gen_loss = gen_loss_fn(disc_fake, fake_img, high_res)
            accuracy = normalized_cross_correlation(fake_img.detach(), high_res.detach())

            gen_loss_history.update(gen_loss.item())
            disc_loss_history.update(disc_loss.item())
            accuracy_history.update(accuracy)

    return gen_loss_history(), disc_loss_history(), accuracy_history()


def train_evaluation():
    best_val_acc = float('-inf')
    gen = Generator(in_channel=1).to(config.device)
    disc = Discriminator(in_channel=1).to(config.device)
    gen_optim = optim.Adam(gen.parameters(), lr=config.learning_rate)
    disc_optim = optim.Adam(disc.parameters(), lr=config.learning_rate)
    if config.load_checkpoint:
        config.logger.info(f"Restoring parameters from {config.checkpoint_dir}")
        best_val_acc = utils.load_checkpoint(config.checkpoint_dir, gen, config, gen_optim, name='gen.')
        best_val_acc = utils.load_checkpoint(config.checkpoint_dir, disc, config, disc_optim, name='disc.')

    gen_loss_fn = fetch_gen_loss()
    disc_loss_fn = fetch_disc_loss()
    # Data loader
    train_data_loader, val_data_loader = fetch_data_loader(config)
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        train_loss = train(gen, disc, disc_optim, gen_optim, gen_loss_fn, disc_loss_fn, train_data_loader)
        print(f"Train loss: \n \t Generator: {train_loss[0]} \n \t Discriminator: {train_loss[1]} \n \t Accuracy: {train_loss[2]}")
        torch.cuda.empty_cache()
        eval_loss = evaluation(gen, disc, gen_loss_fn, disc_loss_fn, val_data_loader)
        print(f"Eval loss: \n \t Generator: {eval_loss[0]} \n \t Discriminator: {eval_loss[1]} \n \t Accuracy: {eval_loss[2]}")
        torch.cuda.empty_cache()

        config.tensor_board_writer.add_scalars(f'gen_loss', {
            'validation': eval_loss[0],
            'training': train_loss[0],
        }, epoch)

        config.tensor_board_writer.add_scalars(f'disc_loss', {
            'validation': eval_loss[1],
            'training': train_loss[1],
        }, epoch)

        config.tensor_board_writer.add_scalars(f'accuracy', {
            'validation': eval_loss[2],
            'training': train_loss[2],
        }, epoch)

        val_acc = eval_loss[2]
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': gen.state_dict(),
                               'optim_dict': gen_optim.state_dict()},
                              is_best=is_best,
                              checkpoint=config.checkpoint_dir,
                              name='gen.')

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': disc.state_dict(),
                               'optim_dict': disc_optim.state_dict()},
                              is_best=is_best,
                              checkpoint=config.checkpoint_dir,
                              name='disc.')

        # If best_eval, best_save_path
        if is_best:
            config.logger.info("Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                config.checkpoint_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json({'accuracy': val_acc}, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            config.checkpoint_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json({'accuracy': val_acc}, last_json_path)


if __name__ == '__main__':
    config = Config('../../config.yaml')
    config.tensor_board_writer = SummaryWriter(config.tensorflow_log_dir)
    train_evaluation()
