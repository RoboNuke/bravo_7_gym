from bravo_7_gym.envs.bravo7_env_client import DefaultEnvConfig

import numpy as np
from typing import Dict

class FixedPegInsertConfig(DefaultEnvConfig):
    # reset vars
    RESET_POSE = np.array([0.5494, 0.0033, 0.4362, -0.1519, 0.4307, -0.2859, 0.8424])
    RANDOM_RESET = True
    RANDOM_X_RANGE = 0.015
    RANDOM_Y_RANGE = 0.015
    RANDOM_Z_RANGE = 0.00

    RANDOM_RX_RANGE = 0.00
    RANDOM_RY_RANGE = 0.00
    RANDOM_RZ_RANGE = 0.01
    Z_PRERESET_MOVE = 0.01
    Z_SAFETY_DISTANCE = 0.05
    FT_TARE_FREQ = 90 # ~20 sec per episode => 30 min to tare ft sensor

    # camera vars
    REALSENSE_CAMERAS: Dict = {
        "wrist": "913422070891",
        "world": "913422070922",
    }
    USE_CAMERAS = True
    SAVE_VIDEO = True

    # goal vars
    TARGET_POSE: np.ndarray = np.array([0.436699, -0.05278, 0.27785, 0.74634, 0.64014, 0.14252, 0.11355])
    
    # other
    USE_FT_SENSOR = True