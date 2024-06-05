from bravo_7_gym.envs.bravo7_env_client import DefaultEnvConfig

import numpy as np
from typing import Dict

ang_offset = 3.14 * 15.0/ 180.0
#ang_offset = np.pi / 6
class FixedPegInsertConfig(DefaultEnvConfig):
    # reset vars
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.2]) # x, y, z, angle
    
    RANDOM_RESET = True
    RANDOM_X_RANGE = 0.05
    RANDOM_Y_RANGE = 0.05
    RANDOM_Z_RANGE = 0.00

    RANDOM_RX_RANGE = 0.00
    RANDOM_RY_RANGE = 0.00
    RANDOM_RZ_RANGE = ang_offset / 2
    Z_PRERESET_MOVE = 0.075
    Z_SAFETY_DISTANCE = 0.05
    FT_TARE_FREQ = 90 # ~20 sec per episode => 30 min to tare ft sensor

    # camera vars
    REALSENSE_CAMERAS: Dict = {
        #"wrist": "913422070891",
        "world": "913422070922",
    }
    USE_CAMERAS = True
    SAVE_VIDEO = True
    SAVE_PATH: str = "/home/hunter/serl_tests/test3_data/obs/"
    ACTION_SCALE = np.array([0.02, 0.1, 1.0])

    # goal vars
    TARGET_POSE: np.ndarray = np.array([0.424, -0.0147, 0.042, 1.0, 0.0, 0.0, 0.0])
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    # added 19mm + 0.02291
    # other
    USE_FT_SENSOR = False
    
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_X_RANGE, 
            TARGET_POSE[1] - RANDOM_Y_RANGE, 
            TARGET_POSE[2] + 0.023, 
            3.1415 - ang_offset, 
            -ang_offset, 
            -ang_offset
        ]
    )

    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_X_RANGE, 
            TARGET_POSE[1] + RANDOM_Y_RANGE, 
            TARGET_POSE[2] + 0.1, 
            3.1415 + ang_offset, 
            ang_offset, 
            ang_offset
         ]
    )