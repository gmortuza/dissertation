from models.resnet34 import ResNet34
from models.base_model import BaseModel
from models.unet import UNet


def get_model(config):
    if config.model_type == 'resnet34':
        model = ResNet34(config)
    elif config.model_type == 'base_model':
        model = BaseModel(config)
    elif config.model_type == 'unet':
        model = UNet(config)

    return model.to(config.device)
