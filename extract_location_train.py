import sys

from extract_location import main
from read_config import Config
import utils


if __name__ == '__main__':
    from_borah = True if len(sys.argv) > 1 else False
    config_ = Config("config.yaml", from_borah)
    if config_.use_seed:
        utils.set_seed(1)
    main.main(config_)