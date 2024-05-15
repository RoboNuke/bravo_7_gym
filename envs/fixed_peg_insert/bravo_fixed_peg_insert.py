import gymnasium as gym
import numpy as np
import requests
import time
import copy 

from bravo_7_gym.envs.bravo7_env_client import Bravo7Env
from bravo_7_gym.envs.fixed_peg_insert.config import FixedPegInsertConfig

from bravo_7_gym.utils.rotations import euler_2_quat, quat_2_euler

class Bravo7FixedPegInsert(Bravo7Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=FixedPegInsertConfig)

    def go_to_rest(self):
        """
        SAME AS DEFAULT EXCEPT MOVES UP IN Z BEFORE BEGINING
        """

        # move up in z
        self._update_currpos()
        prereset_pos = copy.deepcopy(self.currpos)
        if prereset_pos[2] <= self.config.TARGET_POSE[2] + self.config.Z_SAFETY_DISTANCE:
            requests.post(self.url + "stopCC")
            time.sleep(0.5)
            prereset_pos[2] += self.config.Z_PRERESET_MOVE
            self._send_pos_command(prereset_pos)

        super().go_to_rest()


    