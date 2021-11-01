
from models.srgan.train import train_evaluation
from read_config import Config
from torch.utils.tensorboard import SummaryWriter

config = Config('config.yaml')
config.tensor_board_writer = SummaryWriter(config.tensorflow_log_dir)
train_evaluation()
