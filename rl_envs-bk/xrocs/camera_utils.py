from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
from typing import Literal


class CameraUtils:
    
    @staticmethod
    def encode_rgb_image(rgb_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        _, encoded_rgb_image = cv2.imencode(".jpg", rgb_image)
        return encoded_rgb_image
    
    @staticmethod
    def encode_depth_image(depth_image: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.uint16]:
        depth_uint16_image = depth_image.clip(0.0, 65535.0).astype(np.uint16)
        _, encoded_depth_uint16_image = cv2.imencode(".png", depth_uint16_image)
        return encoded_depth_uint16_image

    @staticmethod
    def decode_color_image(
        encoded_color_image: npt.NDArray[np.uint8], 
        output_format: Literal["rgb", "bgr"] = "rgb"
    ) -> npt.NDArray[np.uint8]:
        decoded_bgr_color_image = cv2.imdecode(encoded_color_image, cv2.IMREAD_COLOR)  # cv2.imdecode needs BGR format and returns BGR.

        if output_format == "rgb":
            return cv2.cvtColor(decoded_bgr_color_image, cv2.COLOR_BGR2RGB)
        else:
            return decoded_bgr_color_image

    @staticmethod
    def decode_depth_image(encode_depth_uint16_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
        depth_uint16_image = cv2.imdecode(encode_depth_uint16_image, cv2.IMREAD_UNCHANGED)
        return depth_uint16_image

    @staticmethod
    def apply_depth_colormap(
        depth_image: npt.NDArray[np.floating | np.integer],
        color_map: int = cv2.COLORMAP_JET,
        normalize: bool = True
    ) -> npt.NDArray[np.uint8]:
        if normalize:
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_uint8 = depth_normalized.astype(np.uint8)
        else:
            depth_uint8 = depth_image.astype(np.uint8)
        
        depth_colormap = cv2.applyColorMap(depth_uint8, color_map)
        return depth_colormap
