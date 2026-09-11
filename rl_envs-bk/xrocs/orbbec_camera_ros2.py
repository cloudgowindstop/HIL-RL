from __future__ import annotations

import time
import cv2
import cv_bridge
import numpy as np
import numpy.typing as npt
from threading import Lock
from typing import Dict, Optional, Tuple, Union

from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

from rl_envs.xrocs.camera_utils import CameraUtils
from loguru import logger
import time
from rclpy.qos import qos_profile_sensor_data


class OrbbecCameraRos2:

    def __init__(self, node: Node, camera_config: Dict) -> None:
        """Initialize the Orbbec camera ROS 2 driver.
        
        Args:
            node: ROS 2 node.
            camera_config: The camera configuration. For example:
                {
                    "head": {
                        "enable": True,  # Reserved parameter for robot station initialization.
                        "type" = "OrbbecCameraRos2",  # Reserved parameter for robot station initialization.
                        "camera_name": "camera",
                        "use_compress_rgb": True,
                        "use_compress_depth": False
                    }
                }.
        """
        self.node = node
        self.cfg_dict = camera_config
        self._qos_profile = 10
        self._slop = 0.1  # The time difference between the two images.

        self.rgb_msg, self.depth_msg = None, None
        self.data_lock = Lock()

        self._cv_bridge = cv_bridge.CvBridge()

        self._camera_name = self.cfg_dict.get("camera_name", "camera")
        self._use_compress_rgb = self.cfg_dict.get("use_compress_rgb", False)
        self._use_compress_depth = self.cfg_dict.get("use_compress_depth", False)

        self._rgb_topic = f"/{self._camera_name}/color/image_raw"
        self._depth_topic = f"/{self._camera_name}/depth/image_raw"

        self._rgb_compress_topic = f"/{self._camera_name}/color/image_raw/compressed"
        self._depth_compress_topic = f"/{self._camera_name}/depth/image_raw/compressedDepth"

        self.color_topic_name = ""
        self.depth_topic_name = ""

        qos_best_effort = qos_profile_sensor_data

        if self._use_compress_rgb:
            self.rgb_suber = Subscriber(self.node, CompressedImage, self._rgb_compress_topic, qos_profile=qos_best_effort)
            self.color_topic_name = self._rgb_compress_topic
        else:
            self.rgb_suber = Subscriber(self.node, Image, self._rgb_topic, qos_profile=qos_best_effort)
            self.color_topic_name = self._rgb_topic

        if self._use_compress_depth:
            self.depth_suber = Subscriber(self.node, CompressedImage, self._depth_compress_topic, qos_profile=qos_best_effort)
            self.depth_topic_name = self._depth_compress_topic
        else:
            self.depth_suber = Subscriber(self.node, Image, self._depth_topic, qos_profile=qos_best_effort)
            self.depth_topic_name = self._depth_topic
            
        self.ats = ApproximateTimeSynchronizer(
            [self.rgb_suber, self.depth_suber], queue_size=self._qos_profile, slop=self._slop)
        self.ats.registerCallback(self._image_callback)
        
        logger.success("Orbbec camera ROS 2 driver initialized successfully.")

    def _image_callback(self, rgb_msg: Union[Image, CompressedImage], depth_msg: Union[Image, CompressedImage]) -> None:
        """Callback function for synchronized image messages.

        This function will be called when the image message is received.

        Args:
            rgb_msg: The RGB image message.
            depth_msg: The depth image message.
        """
        with self.data_lock:
            self.rgb_msg = rgb_msg
            self.depth_msg = depth_msg
    
    def read(self, img_size: Optional[Tuple[int, int]] = None, timeout: float = 2.0, check_interval: float = 0.5) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]:
        """Read a camera frame (includes both RGB and depth images).

        Note that the RGB and depth images will be compressed (refer to the 'CameraUtils.encode_rgb_image' and 'CameraUtils.encode_depth_image') before being returned.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            timeout: The timeout in seconds.
            check_interval: The check interval in seconds.

        Returns:
            The encoded color and depth image.
        """
        rgb_encode_data, depth_encode_data = None, None
        rgb_msg, depth_msg = None, None

        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.data_lock:
                valid_rgb = self.rgb_msg is not None
                valid_depth = self.depth_msg is not None

                if not valid_rgb:
                    logger.warning(
                        f"No {self._camera_name} RGB image received.\n"
                        f"无法订阅到相机 RGB 数据，当前订阅的话题为 '{self.color_topic_name}'，请使用 'ros2 topic list' 查看该话题是否存在或修改 'configuration.toml' 配置文件以适配所需要订阅的话题名称。\n"
                        f"Cannot subscribe to camera RGB data; current topic is '{self.color_topic_name}'; use 'ros2 topic list' to check if it exists or edit 'configuration.toml' to match the desired topic name."
                    )

                if not valid_depth:
                    logger.warning(
                        f"No {self._camera_name} depth image received.\n"
                        f"无法订阅到相机深度数据，当前订阅的话题为 '{self.depth_topic_name}'，请使用 'ros2 topic list' 查看该话题是否存在或修改 'configuration.toml' 配置文件以适配所需要订阅的话题名称。\n"
                        f"Cannot subscribe to camera depth data; current topic is '{self.depth_topic_name}'; use 'ros2 topic list' to check if it exists or edit 'configuration.toml' to match the desired topic name."
                    )

                if valid_rgb and valid_depth:
                    rgb_msg = self.rgb_msg
                    depth_msg = self.depth_msg
                    break
                else:
                    time.sleep(check_interval)

        if not valid_rgb or not valid_depth:
            logger.error(
                f"Failed to read images within the allowed time frame {timeout} seconds.\n"
                f"在规定时间内无法获取到相机数据，请从硬件连接、ROS 2 通信以及话题名称等方面排查。\n"
                f"Could not get camera data within the timeout; check hardware connection, ROS 2 communication, and topic names."
            )
            return None, None

        try:
            if isinstance(rgb_msg, CompressedImage):
                # The data in CompressedImage is already compressed.
                bgr_encode_data = np.frombuffer(rgb_msg.data, np.uint8)  # return BGR compressed data
            else:
                rgb_cv = self._cv_bridge.imgmsg_to_cv2(rgb_msg, "passthrough")  # RGB format
                bgr_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for correct encoding

                if img_size is not None:
                    bgr_cv = cv2.resize(bgr_cv, img_size)
                
                bgr_encode_data = CameraUtils.encode_rgb_image(bgr_cv)  # return BGR compressed data
        
        except Exception as e:
            logger.error(f"Failed to process RGB image: {e}.")
            return None, None

        try:
            if isinstance(depth_msg, CompressedImage):
                # The data in CompressedImage is already compressed.
                depth_encode_data = np.frombuffer(depth_msg.data[12:], np.uint8)
            else:
                depth_cv = self._cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

                if img_size is not None:
                    depth_cv = cv2.resize(depth_cv, img_size)
                
                depth_encode_data = CameraUtils.encode_depth_image(depth_cv)
        
        except Exception as e:
            logger.error(f"Failed to process depth image: {e}.")
            return None, None

        return bgr_encode_data, depth_encode_data
