""" Gym Interface for Bravo 7 """
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
import cv2

import time
import copy
from typing import Dict
from datetime import datetime
from collections import OrderedDict
import threading
import queue

from bravo_7_gym.utils.rotations import euler_2_quat, quat_2_euler
from bravo_7_gym.camera.rs_capture import RSCapture
from bravo_7_gym.camera.video_capture import VideoCapture

#import rospy
#from std_msgs.msg import Float64MultiArray
#from geometry_msgs.msg import PoseStamped
#from bravo_7_gym.msg import Bravo7State

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
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    #TARGET_POSE: np.ndarray = np.zeros((6,))
    TARGET_POSE: np.ndarray = np.array([0.437, -0.053, 0.278, -3.127, 0.365, -1.421])
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    #RESET_POSE = np.zeros((6,))
    RESET_POSE = np.array([0.232, 0.0, 0.547, 0.0, -0.01, -0.511])
    #RESET_POSE = np.array([0.5494, 0.0033, 0.436, -0.0164, 0.949, -0.646])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    EE_FRAME = "ee_link"
    USE_CAMERAS = False
    USE_FT_SENSOR = False
    CMD_PORT = 65432
    EVAL_CMD_PORT = 65000
    POS_PORT = 53269
    EVAL_POS_PORT = 55269
    HOST = "127.0.0.1"

class Bravo7Env(gym.Env):
    def __init__(
            self,
            kwargs, # "actor", "learner", "eval"
            hz=10,
            save_video = False,
            #config:DefaultEnvConfig = None,
            config  = DefaultEnvConfig(),
            max_episode_length=100):
        print("In env init")
        self.env_type = kwargs
        print(self.env_type)
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = max_episode_length

        self.ee_frame = config.EE_FRAME
        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )

        #self.stateHolder = Bravo7State()
        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))

        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

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
            self.observation_space["state"]["tcp_force"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,))
            self.observation_space["state"]["tcp_torque"] = gym.spaces.Box(-np.inf, np.inf, shape=(3,))

        if config.USE_CAMERAS:
            self.observation_space["images"] = gym.spaces.Dict(
                {
                    "wrist_1": gym.spaces.Box(
                        0, 255, shape=(128, 128, 3), dtype=np.uint8
                    ),
                    "wrist_2": gym.spaces.Box(
                        0, 255, shape=(128, 128, 3), dtype=np.uint8
                    ),
                }
            )

            self.cap = None
            self.init_cameras(config.REALSENSE_CAMERAS)
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()
        #self.b7CC_pub = rospy.Publisher(config.BRAVO_7_CC_TOPIC, Float64MultiArray, queue_size=1)
        #self.b7state_sub = rospy.Subscriber(config.BRAVO_7_STATE_TOPIC, Bravo7State, self.stateCB)

        #pos_socket = MLSocket().connect((config.HOST, config.POS_PORT))
        if not self.env_type == "learner":
            self.cmd_socket = MLSocket()
            self.lookToConnect()
            self.posThread = threading.Thread(target=self.posCB)
            self.posThread.start()

        print("Initialized Bravo 7 Env")

    #def stateCB(self, msg):
    #    self.stateHolder = msg
    def lookToConnect(self):
        connected = False
        print(f"{self.env_type}:Connecting on cmd socket")
        while not connected:
            try:
                if self.env_type == "actor":
                    self.cmd_socket.connect((self.config.HOST, self.config.CMD_PORT))
                else:
                    self.cmd_socket.connect((self.config.HOST, self.config.EVAL_CMD_PORT))
                connected = True
            except Exception as e:
                pass
        print(f"{self.env_type}:Printed to cmd socket")

    def posCB(self):
        with MLSocket() as s:
            if self.env_type == "actor":
                s.bind((self.config.HOST, self.config.POS_PORT))
            elif self.env_type == "eval":
                s.bind((self.config.HOST, self.config.EVAL_POS_PORT))
            s.listen()
            conn, address = s.accept()
            print(f"{self.env_type}:Connected by ", address)
            with conn:
                while True:
                    self.currpos = conn.recv(1024)
                    self.currvel = conn.recv(1024)
                    self.q = conn.recv(1024)
                    self.dq = conn.recv(1024)
                    self.currforce = conn.recv(1024)
                    self.currtorque = conn.recv(1024)


    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward
        return ob, int(reward), done, False, {}
    
    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        euler_angles = quat_2_euler(current_pose[3:])
        euler_angles = np.abs(euler_angles)
        current_pose = np.hstack([current_pose[:3], euler_angles])
        delta = np.abs(current_pose - self._TARGET_POSE)
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
            return False

    def _send_pos_command(self, pos: np.ndarray):
        # publish to command pose topic
        #cmd = Float64MultiArray()
        #cmd.data = pos
        #self.b7CC_pub.publish(cmd)
        try:
            self.cmd_socket.send(pos)
        except Exception as e:
            print(e)
            self.lookToConnect()

    def _update_currpos(self):
        """
        Internal functions to update state information from
        the robot, gripper and force-torque sensor
        """
        """sh = self.stateHolder.copy() # copy should prevent overright issues?
        self.q[:] = sh.q
        self.dq[:] = sh.dq

        self.currpos[:] = sh.pose
        self.currvel[:] = sh.vel

        self.currforce[:] = sh.force
        self.currtorque[:] = sh.torque
        """
        return

    def _get_obs(self) -> dict:
        if self.config.USE_FT_SENSOR:
            state_obs = {
                "tcp_pose": self.currpos,
                "tcp_vel": self.currvel,
                "tcp_force": self.currforce,
                "tcp_torque": self.currtorque, 
            }
        else:
            state_obs = {
                "tcp_pose": self.currpos,
                "tcp_vel": self.currvel,
            }

        if self.config.USE_CAMERAS:
            images = self.get_im()
            return copy.deepcopy(dict(images=images, state=state_obs))
        else:
            return copy.deepcopy(dict(state=state_obs))

    def reset(self, **kwargs):

        self.go_to_rest()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        return obs, {}

    def go_to_rest(self):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._TARGET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1.5)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=10.0)

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self._update_currpos()

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
        if name == "wrist_1":
            return image[:, 80:560, :]
        elif name == "wrist_2":
            return image[:, 80:560, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
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

    """
    def np_to_poseStamped(self, pos: np.ndarray, frame="ee_link"):
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x = pos[0]
        ps.pose.position.y = pos[1]
        ps.pose.position.z = pos[2]
        ps.pose.orientation.x = pos[3]
        ps.pose.orientation.y = pos[4]
        ps.pose.orientation.z = pos[5]
        ps.pose.orientation.w = pos[6]
        return ps
    """

