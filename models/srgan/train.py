import torch
import torch.nn as nn
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from loss import fetch_disc_loss, fetch_gen_loss
from torch import optim
from data_loader import fetch_data_loader
from read_config import Config
import matplotlib.pyplot as plt
from models.metrics import normalized_cross_correlation
import utils


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

    return sum(gen_loss_history) / len(gen_loss_history), sum(disc_loss_history) / len(disc_loss_history)


def evaluation(gen, disc, gen_loss_fn, disc_loss_fn, eval_data_loader):
    gen.eval()
    disc.eval()
    gen_loss_history = []
    disc_loss_history = []
    for low_res, high_res in eval_data_loader:
        low_res = low_res.to(config.device)
        high_res = high_res.to(config.device).to_dense()
        fake_img = gen(low_res)

        disc_real = disc(high_res)
        disc_fake = disc(fake_img.detach())
        disc_loss = disc_loss_fn(disc_fake, disc_real)

        disc_fake = disc(fake_img)
        gen_loss = gen_loss_fn(disc_fake, fake_img, high_res)

        gen_loss_history.append(gen_loss.item())
        disc_loss_history.append(disc_loss.item())
    return sum(gen_loss_history) / len(gen_loss_history), sum(disc_loss_history) / len(disc_loss_history)


def train_evaluation():
    gen = Generator(in_channel=1).to(config.device)
    disc = Discriminator(in_channel=1).to(config.device)
    gen_optim = optim.Adam(gen.parameters(), lr=config.learning_rate)
    disc_optim = optim.Adam(disc.parameters(), lr=config.learning_rate)
    gen_loss_fn = fetch_gen_loss()
    disc_loss_fn = fetch_disc_loss()
    # Data loader
    train_data_loader, val_data_loader = fetch_data_loader(config)
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        train_loss = train(gen, disc, disc_optim, gen_optim, gen_loss_fn, disc_loss_fn, train_data_loader)
        print(f"Train loss: \n \t Generator: {train_loss[0]} \n \t Discriminator: {train_loss[1]}")
        eval_loss = evaluation(gen, disc, gen_loss_fn, disc_loss_fn, val_data_loader)
        print(f"Eval loss: \n \t Generator: {eval_loss[0]} \n \t Discriminator: {eval_loss[1]}")


if __name__ == '__main__':
    config = Config('../../config.yaml')
    train_evaluation()
