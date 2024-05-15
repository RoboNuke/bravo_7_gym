""" Gym Interface for Bravo 7 """
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
import cv2
import requests
import time
import copy
from typing import Dict
from datetime import datetime
from collections import OrderedDict
import threading
import queue

from scipy.spatial.transform import Rotation as R
from bravo_7_gym.utils.rotations import euler_2_quat, quat_2_euler
from pyquaternion import Quaternion
from bravo_7_gym.camera.rs_capture import RSCapture
from bravo_7_gym.camera.video_capture import VideoCapture

from mlsocket import MLSocket
import threading

class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


##############################################################################

class DefaultEnvConfig:
    """ Default Configurations for Bravo7 env. """
    REALSENSE_CAMERAS: Dict = {
        "wrist": "913422070891",
        "world": "913422070922",
    }
    TARGET_POSE: np.ndarray = np.array([0.436699, -0.05278, 0.27785, 0.74634, 0.64014, 0.14252, 0.11355])
    REWARD_THRESHOLD: np.ndarray = np.array([0.005, 0.005, 0.005, 0.1]) # x, y, z, angle
    ACTION_SCALE = np.ones((3,))
    RESET_POSE = np.array([0.5494, 0.0033, 0.4362, -0.1519, 0.4307, -0.2859, 0.8424])
    RANDOM_RESET = False
    RANDOM_X_RANGE = 0.015
    RANDOM_Y_RANGE = 0.015
    RANDOM_Z_RANGE = 0.00
    RANDOM_RX_RANGE = 0.00
    RANDOM_RY_RANGE = 0.00
    RANDOM_RZ_RANGE = 0.01
    ABS_POSE_LIMIT_HIGH = np.array([1.0, 0.5, 0.65, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_LOW = np.array([0.15, -0.5, 0.01, -3.14, -3.14, -3.14])
    USE_CAMERAS = False
    USE_FT_SENSOR = False
    FT_TARE_FREQ = 90 # ~20 sec per episode => 30 min to tare ft sensor
    SERVER_URL: str = "http://127.0.0.1:5000/"
    SAVE_VIDEO = True
    SAVE_PATH: str = "obs/"

class Bravo7Env(gym.Env):
    def __init__(
            self,
            hz=10,
            fake_env=False,
            config  = DefaultEnvConfig(),
            max_episode_length=100):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = max_episode_length
        self.url = self.config.SERVER_URL

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        if len(config.RESET_POSE) == 6:
            self.resetpos = np.concatenate(
                [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
            )
        else:
            self.resetpos = config.RESET_POSE
        
        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((6,))
        self.dq = np.zeros((6,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))

        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.hz = hz
        self.save_video = config.SAVE_VIDEO
        self.recording_frames = []
        self.save_path = config.SAVE_PATH
        if config.SAVE_VIDEO:
            print("Saving videos at: " + self.save_path)

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((6,), dtype=np.float32) * -1,
            np.ones((6,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    }
                )
            }
        )
        if config.USE_FT_SENSOR:
            self.ft_cycles = 0
            self.observation_space["state"]["tcp_force"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,))
            self.observation_space["state"]["tcp_torque"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,))

        if config.USE_CAMERAS:
            self.observation_space["images"] = gym.spaces.Dict(
                {
                    "wrist": gym.spaces.Box(
                        0, 255, shape=(128, 128, 3), dtype=np.uint8
                    ),
                    "world": gym.spaces.Box(
                        0, 255, shape=(128, 128, 3), dtype=np.uint8
                    ),
                }
            )
            if fake_env:
                return
            self.cap = None
            self.init_cameras(config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()
        print("Initialized Bravo 7 Env")


    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])*
            Rotation.from_quat(self.currpos[3:]) 
        ).as_quat()
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward

        info = {}
        info['found_goal'] = reward
        return ob, int(reward), done, False, info
    
    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # get quaternion distance
        q_cur = self.pos2Quat(current_pose)
        q_goal = self.pos2Quat(self._TARGET_POSE)

        delta = np.zeros((4,)) # x, y, z, angle
        delta[:3] = np.abs(current_pose[:3] - self._TARGET_POSE[:3])
        dq = q_goal.inverse * q_cur
        delta[3] = dq.angle # gets the magnitude of the angle offset
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            return False
        
    def pos2Quat(self, pos):
        return Quaternion(pos[6], pos[3], pos[4], pos[5])
    
    def _send_pos_command(self, pos: np.ndarray):
        # pose pose command
        #print("Sending:", pos)
        arr = np.array(pos).astype(np.float64)
        data = {"arr":arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _update_currpos(self):
        """
        Internal functions to update state information from
        the robot, gripper and force-torque sensor
        """
        ps = requests.post(self.url + "getstate", json={})
        ps = ps.json()
        self.currpos[:] = np.array(ps["pose"], dtype="float32")
        self.currvel[:] = np.array(ps["vel"], dtype="float32")

        self.currforce[:] = np.array(ps["force"], dtype="float32")
        self.currtorque[:] = np.array(ps["torque"], dtype="float32")

        self.q[:] = np.array(ps["q"], dtype="float32")
        self.dq[:] = np.array(ps["dq"], dtype="float32")

    def _get_obs(self) -> dict:
        if self.config.USE_FT_SENSOR:
            state_obs = {
                "tcp_pose": self.currpos.astype("float32"),
                "tcp_vel": self.currvel.astype("float32"),
                "tcp_force": self.currforce.astype("float32"),
                "tcp_torque": self.currtorque.astype("float32"), 
            }
        else:
            state_obs = {
                "tcp_pose": self.currpos.astype("float32"),
                "tcp_vel": self.currvel.astype("float32"),
            }

        if self.config.USE_CAMERAS:
            images = self.get_im()

            return copy.deepcopy(dict(images=images, state=state_obs))
        else:
            return copy.deepcopy(dict(state=state_obs))

    def reset(self, **kwargs):
        #print("RESETING")
        if self.save_video:
            self.save_video_recording()

        self.go_to_rest()
        self.curr_path_length = 0

        if self.config.USE_FT_SENSOR:
            # tare ft sensor every so often 
            self.ft_cycles += 1
            if self.ft_cycles == self.config.FT_TARE_FREQ:
                self.ft_cycles = 0
                requests.post(self.url + "stopCC")
                time.sleep(0.5)
                requests.post(self.url + "tareFTSensor")
                requests.post(self.url + "startCC")
                time.sleep(0.5)
        
        self._update_currpos()
        obs = self._get_obs()

        #print("RESET")
        return obs, {}

    def getRand(self, r):
        return np.random.uniform(-r, r)
    def go_to_rest(self):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        requests.post(self.url + "stopCC")
        time.sleep(0.5)

        requests.post(self.url + "toNamedPose", data={"name":"rest", "wait":True, "retry":True})
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

        time.sleep(0.5)
        requests.post(self.url + "startCC")
        time.sleep(0.5)

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

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
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.crop_image(key, rgb)
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in self.cap], axis=0)
        )
        self.img_queue.put(display_images)
        return images

    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name == "wrist":
            return image[:, 80:560, :]
        elif name == "world":
            return image[:, 80:560, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    self.save_path + f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, depth=False)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def close(self):
        self.close_cameras()
        super().close()



