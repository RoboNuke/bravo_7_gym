from bravo_7_gym.envs.bravo7_env_client import DefaultEnvConfig

import numpy as np
from typing import Dict

ang_offset = 3.14 * 15.0/ 180.0
class FixedPegInsertConfig(DefaultEnvConfig):
    # reset vars
    RESET_POSE = np.array([0.4170, 0.0, 0.19, 1.0, 0.0, 0.0, 0.0])
    RANDOM_RESET = True
    RANDOM_X_RANGE = 0.08
    RANDOM_Y_RANGE = 0.15
    RANDOM_Z_RANGE = 0.00

    RANDOM_RX_RANGE = 0.00
    RANDOM_RY_RANGE = 0.00
    RANDOM_RZ_RANGE = ang_offset / 2.0
    Z_PRERESET_MOVE = 0.04
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
    TARGET_POSE: np.ndarray = np.array([0.4170, -0.0175, 0.02291, 1.0, 0.0, 0.0, 0.0])
    
    # other
    USE_FT_SENSOR = True

    ABS_POSE_LIMIT_HIGH = np.array([0.50, 0.16, 0.2, 3.1415 + ang_offset, ang_offset, ang_offset])
    ABS_POSE_LIMIT_LOW = np.array([0.32, -0.19, 0.022, 3.1415 - ang_offset, -ang_offset, -ang_offset])