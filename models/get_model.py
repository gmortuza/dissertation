from models.resnet34 import ResNet34
from models.base_model import BaseModel


def get_model(config):
    if config.model_type == 'resnet34':
        return ResNet34(config).to(config.device)
    elif config.model_type == 'base_model':
        return BaseModel(config).to(config.device)
