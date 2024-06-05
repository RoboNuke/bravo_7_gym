import gymnasium as gym
import numpy as np
import requests
import time
import copy 

from scipy.spatial.transform import Rotation as R
from bravo_7_gym.envs.bravo7_env_client import Bravo7Env
from bravo_7_gym.envs.fixed_peg_insert.config import FixedPegInsertConfig

from bravo_7_gym.utils.rotations import euler_2_quat, quat_2_euler
from math import sqrt

class Bravo7FixedPegInsert(Bravo7Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=FixedPegInsertConfig)

    def clip_safety_box(self, pose: np.ndarray):
        """Clip the pose to be within the safety box."""
        pose[:2] = np.clip(
            pose[:2], self.xyz_bounding_box.low[:2], self.xyz_bounding_box.high[:2]
        )
        r = sqrt( (pose[0] - self.config.TARGET_POSE[0]) **2 +
                 (pose[1] - self.config.TARGET_POSE[1]) ** 2)
        if r < 0.031: # radius of the hole
            # over the hole
            pose[2] = np.clip(pose[2], 
                              self.xyz_bounding_box.low[2] - 0.03, 
                              self.xyz_bounding_box.high[2])
        else:
            pose[2] = np.clip(pose[2], 
                              self.xyz_bounding_box.low[2], 
                              self.xyz_bounding_box.high[2])
        

        euler = R.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = R.from_euler("xyz", euler).as_quat()

        return pose


    def go_to_rest(self):
        """
        SAME AS DEFAULT EXCEPT MOVES UP IN Z BEFORE BEGINING
        """
        requests.post(self.url + "startCC")
        time.sleep(0.5)

        # move up in z
        self._update_currpos()
        prereset_pos = copy.deepcopy(self.currpos)
        if prereset_pos[2] <= self.config.TARGET_POSE[2] + self.config.Z_SAFETY_DISTANCE:
            #requests.post(self.url + "stopCC")
            #time.sleep(0.5)
            prereset_pos[2] += self.config.Z_PRERESET_MOVE
            self._send_pos_command(prereset_pos)
            time.sleep(4.0)

        #requests.post(self.url + "toNamedPose", data={"name":"rest", "wait":True, "retry":True})
        # Perform Carteasian reset
        
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[0] += self.getRand(self.config.RANDOM_X_RANGE)
            reset_pose[1] += self.getRand(self.config.RANDOM_Y_RANGE)
            reset_pose[2] += self.getRand(self.config.RANDOM_Z_RANGE)

            rR = R.from_quat(reset_pose[3:])
            xR = R.from_euler('x', self.getRand(self.config.RANDOM_RX_RANGE))
            yR = R.from_euler('y', self.getRand(self.config.RANDOM_RY_RANGE))
            zR = R.from_euler('z', self.getRand(self.config.RANDOM_RZ_RANGE))
            reset_pose[3:] = (zR * yR * xR * rR).as_quat()
            self._send_pos_command(reset_pose)

        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)

        time.sleep(5.0)



    