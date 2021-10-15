from models.resnet34 import ResNet34
from models.base_model import BaseModel
from models.unet import UNet
from models.custom import Custom

def get_model(config):
    if config.model_type == 'resnet34':
        model = ResNet34(config)
    elif config.model_type == 'base_model':
        model = BaseModel(config)
    elif config.model_type == 'unet':
        model = UNet(config, in_channel=1, out_channel=1)
    elif config.model_type == 'custom_model':
        model = Custom(config)

    # log num parameters to neptune
    config.log_param('model_params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model.to(config.device)
