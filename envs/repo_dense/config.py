import numpy as np

from typing import Dict
from bravo_7_gym.envs.bravo7_env_client import DefaultEnvConfig


class RepoDenseConfig(DefaultEnvConfig):
    """ Unique config for reposition with dense reward """
    GOAL_REWARD = 100.0
    RANDOM_RESET = True
    RANDOM_X_RANGE = 0.015
    RANDOM_Y_RANGE = 0.015
    RANDOM_Z_RANGE = 0.00

    RANDOM_RX_RANGE = 0.00
    RANDOM_RY_RANGE = 0.00
    RANDOM_RZ_RANGE = 0.01
    REALSENSE_CAMERAS: Dict = {
        "wrist": "913422070891",
        "world": "913422070922",
    }
    USE_CAMERAS = True
    SAVE_VIDEO = True
    