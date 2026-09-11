"""把Episode转换为Cosmos transition字典。

低维action/proprio使用前序阶段已加载和归一化的数组；本模块再次打开HDF5只读取
图像，不重复读取机器人状态。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .camera import (
    COSMOS_CONDITIONING_FPS,
    build_nine_frame_sequence,
    decode_camera_image,
    prepare_images_for_model,
    prepare_primary_image,
    prepare_wrist_image,
)
from ..memory_monitor import memory_event
from ..preparation.episode import Episode, action_proprio_source_arrays
from ..preparation.rotation import get_action_dim
from ..shape_trace import require_shape, trace_shape
from .episode_labeling import reward_done_for_step

if TYPE_CHECKING:
    from ..config import CosmosRuntimeConfig


def cosmos_obs_to_transition_state(obs: dict) -> dict:
    """只保留可进入transition的数值字段，并统一转成CPU Tensor。"""
    state = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            state[key] = torch.from_numpy(value)
        elif isinstance(value, (int, float)):
            state[key] = torch.tensor(value)
    return state


def sanitize_info_for_transition(info: dict) -> dict:
    """移除无法被LeRobot序列化的补充信息，避免metadata对象泄漏到frame。"""
    safe_info = {}
    for key, value in info.items():
        try:
            if isinstance(value, torch.Tensor):
                safe_info[key] = value
            elif isinstance(value, np.ndarray):
                safe_info[key] = torch.from_numpy(value)
            elif isinstance(value, (int, float, bool)):
                safe_info[key] = value
        except Exception:
            pass
    return safe_info


def select_cameras(cameras: tuple[str, ...]) -> tuple[str, str | None, str | None, str]:
    """按命名规则选出主视角、左右腕视角和单臂fallback腕视角。"""
    if any("head" in camera for camera in cameras):
        primary_cam = next(camera for camera in cameras if "head" in camera)
    elif "camera_wrist" in cameras:
        primary_cam = next((camera for camera in cameras if camera != "camera_wrist"), cameras[0])
    else:
        primary_cam = cameras[0]
    left_cam = next((camera for camera in cameras if "left" in camera), None)
    right_cam = next(
        (camera for camera in cameras if "right" in camera and camera != primary_cam),
        None,
    )
    if left_cam is None and right_cam is None and "camera_wrist" in cameras:
        right_cam = "camera_wrist"
    wrist_cam = left_cam or right_cam or cameras[0]
    return primary_cam, left_cam, right_cam, wrist_cam


def build_transition_list_for_episode(
    episode: Episode,
    robot_type: str,
    t5_embedding: torch.Tensor | None,
    config: CosmosRuntimeConfig,
    *,
    normalized_actions: np.ndarray,
    normalized_proprio: np.ndarray,
    euler_actions: np.ndarray,
    rotation_6d_actions: np.ndarray,
    auxiliary_fields: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict], dict]:
    """构造一条episode的T个transition。

    `normalized_actions`为(T,D)，`normalized_proprio`为(T,P)。每步video是
    (1,3,33,224,224) uint8模板；未来图像、action chunk、future proprio和value
    在VAE编码阶段按时间关系补入。
    """
    memory_event("build_transition_start", hdf5_path=str(episode.path))

    actions = np.asarray(normalized_actions, dtype=np.float32)
    proprio = np.asarray(normalized_proprio, dtype=np.float32)
    euler_control = np.asarray(euler_actions, dtype=np.float32)
    rotation_6d_control = np.asarray(rotation_6d_actions, dtype=np.float32)
    auxiliary = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in (auxiliary_fields or {}).items()
    }
    if actions.ndim != 2 or len(actions) != episode.length:
        raise ValueError(
            f"normalized action shape must be (episode_length, action_dim), got {actions.shape}"
        )
    if proprio.ndim != 2 or len(proprio) != episode.length:
        raise ValueError(
            f"normalized proprio shape must be (episode_length, proprio_dim), got {proprio.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"normalized action contains NaN or Inf: {episode.path}")
    if not np.all(np.isfinite(proprio)):
        raise ValueError(f"normalized proprio contains NaN or Inf: {episode.path}")

    action_dim = get_action_dim(robot_type, config.action_encoding)
    proprio_dim = 16 if episode.is_dual_arm else 8
    arm_count = 2 if episode.is_dual_arm else 1
    require_shape("normalized_actions", actions, (episode.length, action_dim))
    require_shape("normalized_proprio", proprio, (episode.length, proprio_dim))
    require_shape("euler_actions", euler_control, (episode.length, 7 * arm_count))
    require_shape(
        "rotation_6d_actions", rotation_6d_control, (episode.length, 10 * arm_count)
    )
    if not np.all(np.isfinite(euler_control)) or not np.all(np.isfinite(rotation_6d_control)):
        raise ValueError(f"auxiliary control action contains NaN or Inf: {episode.path}")
    for name, values in auxiliary.items():
        if not name.startswith(("action.", "observation.")):
            raise ValueError(
                f"auxiliary field must start with action. or observation.: {name}"
            )
        if values.ndim != 2 or values.shape[0] != episode.length:
            raise ValueError(
                f"{name} must have shape (episode_length,D), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf: {episode.path}")
    source_arrays = action_proprio_source_arrays(episode, config.action_source)
    trace_shape("NORMALIZED", actions=actions, proprio=proprio)

    import h5py

    primary_cam, left_cam, right_cam, wrist_cam = select_cameras(episode.camera_names)
    transitions = []
    progress_step = max(100, episode.length // 10)

    with h5py.File(episode.path, "r") as h5_file:
        for timestep in range(episode.length):
            # 图像在本循环按帧解码；Episode对象本身只常驻低维数据。
            img_primary = decode_camera_image(h5_file, primary_cam, timestep)
            if episode.is_dual_arm and right_cam:
                img_left = decode_camera_image(h5_file, left_cam, timestep) if left_cam else img_primary
                img_right = decode_camera_image(h5_file, right_cam, timestep)
                wrist_raw = prepare_wrist_image(
                    img_left,
                    img_right,
                    True,
                    crop_mode=config.wrist_crop_mode,
                    crop_fraction=config.wrist_crop_fraction,
                    left_center_offset_x=config.wrist_left_center_offset_x,
                    right_center_offset_x=config.wrist_right_center_offset_x,
                )
            else:
                img_wrist = (
                    decode_camera_image(h5_file, wrist_cam, timestep)
                    if wrist_cam != primary_cam
                    else img_primary
                )
                wrist_raw = prepare_wrist_image(img_wrist, None, False)

            primary_raw = prepare_primary_image(
                img_primary,
                primary_cam,
                target_size=config.image_size,
                head_crop_top_pixels=config.head_crop_top_pixels,
            )
            wrist_processed, primary_processed = prepare_images_for_model(
                [wrist_raw, primary_raw],
                config,
                flip_images=False,
            )
            video_tensor = build_nine_frame_sequence(wrist_processed, primary_processed)
            require_shape("transition.video", video_tensor, (1, 3, 33, 224, 224))
            if timestep == 0:
                trace_shape(
                    "TRANSITION",
                    video=video_tensor,
                    action=actions[timestep],
                    proprio=proprio[timestep],
                )

            data_batch = {
                "video": video_tensor,
                "proprio": torch.from_numpy(proprio[timestep]).reshape(1, -1).to(torch.bfloat16),
                "fps": torch.tensor([COSMOS_CONDITIONING_FPS], dtype=torch.bfloat16),
                "padding_mask": torch.zeros(1, 1, 224, 224, dtype=torch.bfloat16),
                "num_conditional_frames": 4,
                "current_proprio_latent_idx": torch.tensor([1], dtype=torch.int64),
                "current_wrist_image_latent_idx": torch.tensor([2], dtype=torch.int64),
                "current_image_latent_idx": torch.tensor([3], dtype=torch.int64),
                "action_latent_idx": torch.tensor([4], dtype=torch.int64),
                "future_proprio_latent_idx": torch.tensor([5], dtype=torch.int64),
                "future_wrist_image_latent_idx": torch.tensor([6], dtype=torch.int64),
                "future_image_latent_idx": torch.tensor([7], dtype=torch.int64),
                "value_latent_idx": torch.tensor([8], dtype=torch.int64),
            }
            if t5_embedding is not None:
                # T5 embedding属于模型条件，不作为独立Parquet列保存。
                data_batch["t5_text_embeddings"] = t5_embedding.to(torch.bfloat16)

            reward, done = reward_done_for_step(timestep, episode.length, config.episode_outcome)
            transitions.append(
                {
                    "state": cosmos_obs_to_transition_state(data_batch),
                    "action": torch.from_numpy(actions[timestep]).float(),
                    "action.euler_control": torch.from_numpy(
                        euler_control[timestep]
                    ).float(),
                    "action.rotation_6d_control": torch.from_numpy(
                        rotation_6d_control[timestep]
                    ).float(),
                    **{
                        name: torch.from_numpy(values[timestep]).float()
                        for name, values in auxiliary.items()
                    },
                    **{
                        name: torch.from_numpy(values[timestep]).float()
                        for name, values in source_arrays.items()
                    },
                    "reward": reward,
                    "next_state": {},
                    "done": done,
                    "truncated": False,
                    "complementary_info": sanitize_info_for_transition({"is_intervention": True}),
                }
            )
            if (timestep + 1) % progress_step == 0 or timestep + 1 == episode.length:
                memory_event(
                    "build_transition_progress",
                    frames_built=timestep + 1,
                    total_frames=episode.length,
                )

    episode_info = {
        "num_steps": episode.length,
        "is_dual_arm": episode.is_dual_arm,
        "action_dim": int(actions.shape[1]),
        "proprio_dim": int(proprio.shape[1]),
        "euler_action_dim": int(euler_control.shape[1]),
        "rotation_6d_action_dim": int(rotation_6d_control.shape[1]),
        "auxiliary_fields": {
            name: int(values.shape[1]) for name, values in auxiliary.items()
        },
        "source_fields": {name: int(values.shape[1]) for name, values in source_arrays.items()},
        "low_dim_source": "data_convert_refactored.preparation.episode",
        "image_source": "hdf5",
        "robot_type": robot_type,
    }
    memory_event("build_transition_complete", transition_count=len(transitions))
    return transitions, episode_info
