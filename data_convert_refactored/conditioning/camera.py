"""相机配置、HDF5解码、裁剪缩放及Cosmos 33帧原始视频布局。"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image

if TYPE_CHECKING:
    from ..config import CosmosRuntimeConfig


COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4
COSMOS_CONDITIONING_FPS = 16

RAW_FRAME_SLICES = {
    "current_wrist": slice(5, 9),
    "current_primary": slice(9, 13),
    "future_wrist": slice(21, 25),
    "future_primary": slice(25, 29),
}

# VAE时间压缩后固定得到9个latent位置；低维条件占用索引1、4、5、8。
LATENT_INDICES: dict[str, int] = {
    "current_proprio_latent_idx": 1,
    "current_wrist_image_latent_idx": 2,
    "current_image_latent_idx": 3,
    "action_latent_idx": 4,
    "future_proprio_latent_idx": 5,
    "future_wrist_image_latent_idx": 6,
    "future_image_latent_idx": 7,
    "value_latent_idx": 8,
}


class CameraState(str, Enum):
    """相机安装/标定状态；当前只支持已验证的normal方案。"""

    NORMAL = "normal"


@dataclass(frozen=True)
class CameraProcessingConfig:
    """由语义相机状态解析出的具体像素裁剪参数。"""

    wrist_crop_mode: str
    wrist_crop_fraction: float
    wrist_left_center_offset_x: int
    wrist_right_center_offset_x: int
    head_crop_top_pixels: int


def resolve_camera_processing(state: CameraState) -> CameraProcessingConfig:
    """把相机状态解析为确定性crop参数，允许metadata记录并复现实验。"""
    if state is CameraState.NORMAL:
        return CameraProcessingConfig(
            wrist_crop_mode="center_width",
            wrist_crop_fraction=0.75,
            wrist_left_center_offset_x=64,
            wrist_right_center_offset_x=64,
            head_crop_top_pixels=100,
        )
    raise ValueError(f"unsupported camera state: {state}")


def duplicate_array(arr: np.ndarray, total_num_copies: int = 4) -> np.ndarray:
    """沿新时间维复制单帧；默认4份对应Cosmos时间压缩因子。"""
    return np.stack([arr] * total_num_copies)


def decode_camera_image(h5: h5py.File, cam_name: str, t: int) -> np.ndarray:
    """读取一帧RGB图像，同时兼容压缩字节和未压缩H×W×3数组。"""
    old_path = f"camera_observations/color_images/{cam_name}"
    fallback_path = f"/observations/images/{cam_name}"
    path = old_path if old_path in h5 else fallback_path
    data = h5[path][t]
    if isinstance(data, (bytes, np.void)):
        data = np.frombuffer(data, np.uint8)
    if data.ndim == 1:
        image = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {cam_name} frame {t}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if data.ndim == 3:
        return data
    raise ValueError(f"unsupported image shape for {cam_name}: {data.shape}")


def prepare_images_for_model(
    images: list[np.ndarray],
    config: CosmosRuntimeConfig,
    flip_images: bool = False,
) -> list[np.ndarray]:
    """执行与训练配置一致的JPEG模拟、224缩放和可选中心增强。"""
    images_array = np.stack(images, axis=0)
    if images_array.dtype != np.uint8 or images_array.ndim != 4 or images_array.shape[-1] != 3:
        raise ValueError(
            f"expected uint8 images shaped (N,H,W,3), got {images_array.dtype} {images_array.shape}"
        )
    if flip_images:
        images_array = np.flipud(images_array)
    if getattr(config, "use_jpeg_compression", False):
        for index in range(len(images_array)):
            buffer = io.BytesIO()
            Image.fromarray(images_array[index]).save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            images_array[index] = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    resized = [
        np.asarray(Image.fromarray(image).resize((config.image_size, config.image_size)))
        for image in images_array
    ]
    images_array = np.stack(resized, axis=0)
    if getattr(config, "trained_with_image_aug", False):
        _, height, width, _ = images_array.shape
        if height != width:
            raise ValueError(f"Cosmos image transform requires square images, got {height}x{width}")
        crop_size = int(height * 0.9**0.5)
        images_tensor = torch.from_numpy(images_array).permute(0, 3, 1, 2)
        images_array = torch.stack(
            [
                TVF.resize(TVF.center_crop(image, crop_size), [height, width], antialias=True)
                for image in images_tensor
            ]
        ).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return list(images_array)


def prepare_wrist_image(
    left_img: np.ndarray,
    right_img: np.ndarray | None,
    is_dual: bool,
    target_size: int = 224,
    crop_mode: str = "center_width",
    crop_fraction: float = 0.75,
    left_center_offset_x: int = 64,
    right_center_offset_x: int = 64,
) -> np.ndarray:
    """生成224×224腕部视图。

    双臂时左右图保持完整高度，各做水平中心crop并缩放为224×112，随后上下拼接；
    单臂不应用双臂crop，直接缩放单路腕相机。
    """
    if is_dual and right_img is not None:
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError(f"crop_fraction must be in (0, 1], got {crop_fraction}")

        def crop_one(image: np.ndarray, center_offset_x: int) -> np.ndarray:
            height, width = image.shape[:2]
            if crop_mode == "center_width":
                crop_width = max(1, int(round(width * crop_fraction)))
                center_x = width // 2 + center_offset_x
                x_start = int(np.clip(center_x - crop_width // 2, 0, width - crop_width))
                return image[:, x_start:x_start + crop_width]
            if crop_mode == "bottom":
                y_start = int(height * (1.0 - crop_fraction))
                return image[y_start:, :]
            if crop_mode == "none":
                return image
            raise ValueError(
                f"unsupported wrist crop mode {crop_mode!r}; expected 'center_width', 'bottom', or 'none'"
            )

        if crop_mode == "none":
            fused = np.concatenate([left_img, right_img], axis=0)
            return cv2.resize(fused, (target_size, target_size))
        left_crop = crop_one(left_img, left_center_offset_x)
        right_crop = crop_one(right_img, right_center_offset_x)
        half_h = target_size // 2
        left_resized = cv2.resize(left_crop, (target_size, half_h))
        right_resized = cv2.resize(right_crop, (target_size, half_h))
        return np.concatenate([left_resized, right_resized], axis=0)
    return cv2.resize(left_img, (target_size, target_size))


def prepare_primary_image(
    image: np.ndarray,
    camera_name: str,
    target_size: int = 224,
    head_crop_top_pixels: int = 100,
) -> np.ndarray:
    """处理主视角；仅名称含head的相机从顶部删除指定像素。"""
    if "head" in camera_name.lower() and head_crop_top_pixels > 0:
        if head_crop_top_pixels >= image.shape[0]:
            raise ValueError(
                f"head_crop_top_pixels={head_crop_top_pixels} must be smaller than "
                f"image height={image.shape[0]}"
            )
        image = image[head_crop_top_pixels:, :]
    return cv2.resize(image, (target_size, target_size))


def build_nine_frame_sequence(
    wrist_img: np.ndarray,
    primary_img: np.ndarray,
) -> torch.Tensor:
    """构造VAE输入(1,3,33,224,224)。

    33个raw时间位置包含当前腕部/主相机、未来占位和空白区域。VAE按4倍时间压缩
    得到9个latent位置，后续在固定索引注入低维条件。
    """
    blank = np.zeros_like(primary_img)
    blank_dup = duplicate_array(blank.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    wrist_dup = duplicate_array(wrist_img, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_dup = duplicate_array(primary_img, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)

    sequence = [
        np.expand_dims(blank, axis=0),
        blank_dup,
        wrist_dup,
        primary_dup,
        blank_dup.copy(),
        blank_dup.copy(),
        wrist_dup.copy(),
        primary_dup.copy(),
        blank_dup.copy(),
    ]
    raw = np.concatenate(sequence, axis=0)
    raw = np.expand_dims(raw, axis=0)
    raw = np.transpose(raw, (0, 4, 1, 2, 3))
    return torch.from_numpy(raw).to(dtype=torch.uint8)
