
"""
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
"""

from typing import Dict
import threading
import cv2
import numpy as np
import queue
import sys
sys.path.insert(0, "/home/hunter/catkin_ws/src/")
from bravo_7_gym.camera.rs_capture import RSCapture
from bravo_7_gym.camera.video_capture import VideoCapture
from collections import OrderedDict

REALSENSE_CAMERAS = {
        "wrist": "913422070891",
        "world": "913422070922",
    }

class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            #print("getin array")
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow("RealSense Cameras", frame)
            #print("showing img")
            cv2.waitKey(1)


import gymnasium as gym

def crop_image(name, image) -> np.ndarray:
    """Crop realsense images to be a square."""
    if name == "wrist":
        return image[:, 80:560, :]
    elif name == "world":
        return image[:, 80:560, :]
    else:
        return ValueError(f"Camera {name} not recognized in cropping")

if __name__=="__main__":
    cap = OrderedDict()
    for cam_name, cam_serial in REALSENSE_CAMERAS.items():
        print(cam_name, cam_serial)
        cam = VideoCapture(
            RSCapture(name=cam_name, serial_number=cam_serial, depth=False)
        )
        cap[cam_name] = cam
    observation_space = gym.spaces.Dict()
    observation_space["images"] = gym.spaces.Dict(
                {
                    "wrist": gym.spaces.Box(
                        0, 255, shape=(512, 512, 3), dtype=np.uint8
                    ),
                    "world": gym.spaces.Box(
                        0, 255, shape=(512, 512, 3), dtype=np.uint8
                    ),
                }
            )

    img_queue = queue.Queue()
    displayer = ImageDisplayer(img_queue)
    displayer.start()

    recording_frames = []
    while True:
        images = {}
        display_images = {}
        for key, capy in cap.items():
            try:
                rgb = capy.read()
                cropped_rgb = crop_image(key, rgb)
                resized = cv2.resize(
                    cropped_rgb, observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )

        recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in cap], axis=0)
        )
        img_queue.put(display_images)