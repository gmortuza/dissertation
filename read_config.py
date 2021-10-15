import os
import yaml
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import neptune.new as neptune

torch.manual_seed(1234)


class Config:
    """Class that loads hyper parameters from a json file.
    Example:
    ```
    params = Params(config_file_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, config_file_path):
        self.update(config_file_path)
        self._additional_parameter()

        # Initially it will be false
        # In the hyper parameter searching file it will be set to True
        self.is_hyper_parameter_searching = False
        # This file is in the root directory of the project.

        # Delete .DS_Store file from all the directory
        # so that tensorflow/pytorch data pipeline won't have to deal with that
        command = "find " + self.base_dir + " -name '*.DS_Store' -type f -delete"
        os.system(command)
        self._make_absolute_directory()
        # set logger
        self.logger = self.get_logger()
        self.verbose = 1 if self.log_level == 'info' else 0
        self.logger.info("Finish reading the configuration file")
        self.neptune = self._setup_neptune()

    def _additional_parameter(self):
        """
        Parameters that will be calculated from the given parameter
        :return:
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device(self.device)
        # Calculate background model
        self.bg_model = torch.tensor(
                (self.laserc_default + self.imagec_default * self.PAINT_imager)
                * self.Imager_Laserpower * self.power_density_conversion
                * self.Camera_integration_time * self.noise_level, device=self.device
        )
        # Calculate tau_d
        self.tau_d = round(1 / (self.PAINT_k_on * self.PAINT_imager * 1 / 10 ** 9) * 1000)

        # Calculate photon parameter
        self.Imager_photon_slope_std = self.Imager_Photonslope / self.std_factor
        self.Imager_photon_rate = self.Imager_Photonslope * self.Imager_Laserpower
        self.Imager_photon_rate_std = self.Imager_photon_slope_std * self.Imager_Laserpower


    def _make_absolute_directory(self):
        # prepend this base directory with other parameter so that we won't get any error for the path
        # As those directory will be accessed from different file. which are in different location
        self.input_dir = os.path.join(self.base_dir, self.input_dir)
        self.output_dir = os.path.join(self.base_dir, self.output_dir)

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.log_dir = os.path.join(self.output_dir, "logs")
        # Create a directory for this run
        self.tensorflow_log_dir = os.path.join(self.log_dir, time.strftime("%l:%M%p - %b %d, %Y"))
        if not os.path.exists(self.tensorflow_log_dir):
            os.makedirs(self.tensorflow_log_dir)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            yaml.dump(self.__dict__, f, indent=4)

    def update(self, config_file_path):
        with open(config_file_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    def _setup_neptune(self):
        neptune_run = neptune.init(
            project=self.neptune_project,
            api_token=self.neptune_api_key,
            name=self.neptune_name,
            tags=self.neptune_tags,
            description=self.neptune_description,
            source_files=self.neptune_code_snapshot,
            mode=self.neptune_mode  # debug stop tracking
        )
        # Setup the hyper-parameters
        neptune_run['parameters'] = {
            "learning_rate": self.learning_rate,
            "final_activation": 'relu',
            "epochs": self.num_epochs,
            "upsample_method": 'transposed',
            "upsampled_activation": "relu",
            "input_normalize_factor": -1,
            "output_normalize_factor": self.last_layer_normalization_factor,
            "threshold": self.detection_threshold,
            "last_layer_input": "output"
        }
        return neptune_run

    def log_param(self, key, val):
        self.neptune['parameters'][key] = val


    def get_logger(self):

        logger = logging.getLogger(__name__)
        # we will either write the log information to file or console
        # Usually we don't need to log in both location
        if self.log_to == "file":
            handler = logging.FileHandler(os.path.join(self.log_dir, "logging.log"))
            handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                                   datefmt='%Y-%m-%d %H:%M:%S'))
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(handler)
        # set the log level
        if self.log_level == 'info':
            logger.setLevel(logging.INFO)
        elif self.log_level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif self.log_level == 'warning':
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.ERROR)

        return logger


if __name__ == '__main__':
    config = Config("config.yaml")
