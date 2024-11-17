
from envs.smacv2.smacv2_logger import SMACv2Logger

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


LOGGER_REGISTRY = {
    "smacv2": SMACv2Logger,
}


