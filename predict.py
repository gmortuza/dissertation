import torch
from read_config import Config
from utils import load_checkpoint
from models import get_model
from PIL import Image
import numpy as np


def predict(config):
    original_file = "/Users/golammortuza/workspace/nam/Substack_10.tif"
    im = Image.open(original_file)
    im_arr = np.asarray(im)
    print(im_arr.shape)
    pass


if __name__ == '__main__':
    config = Config("config.yaml")
    predict(config)

