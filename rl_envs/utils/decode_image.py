import cv2
import numpy as np
from typing import Union, Optional, Tuple


def decode_image(
    camera_rgb_images: Union[np.ndarray, list], 
    camera_depth_images: Optional[Union[np.ndarray, list]] = None, 
    bgr2rgb: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if type(camera_rgb_images[0]) is np.uint8:
        rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
        if bgr2rgb and rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if camera_depth_images is not None:
            depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        else:
            depth = np.asarray([])
        return rgb, depth
    else:
        rgb_images = []
        depth_images = []
        for idx, camera_rgb_image in enumerate(camera_rgb_images):
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            
            if bgr2rgb and rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb)
            depth_images.append(depth)
        rgb_images = np.asarray(rgb_images)
        depth_images = np.asarray(depth_images)
        return rgb_images, depth_images