import gymnasium as gym
import numpy as np

from bravo_7_gym.envs.bravo7_env_client import Bravo7Env
from bravo_7_gym.envs.repo_dense.config import RepoDenseConfig

class Bravo7RepoDense(Bravo7Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=RepoDenseConfig)

    def _denseReward(self, old_reward, obs):
        if old_reward:
            return self.config.GOAL_REWARD
        else:
            current_pose = obs["state"]["tcp_pose"]
            # get quaternion distance
            q_cur = self.pos2Quat(current_pose)
            q_goal = self.pos2Quat(self._TARGET_POSE)

            delta = np.zeros((4,)) # x, y, z, angle
            delta[:3] = np.abs(current_pose[:3] - self._TARGET_POSE[:3])
            dq = q_goal.inverse * q_cur
            delta[3] = dq.angle # gets the magnitude of the angle offset
            dist = delta.dot(delta)

            return 1.0 / dist

    def step(self, action: np.ndarray) -> tuple:
        obs, rew, done, a, info = super().step(action)

        done = self.curr_path_length >= self.max_episode_length

        
        rew = self._denseReward(rew, obs)

        return obs, rew, done, a, info