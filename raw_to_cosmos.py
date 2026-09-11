#!/usr/bin/env python3
"""
Raw HDF5 → Cosmos Policy 训练数据转换

从 BOS 下载的 raw HDF5 轨迹文件转换为一整个 LeRobot 数据集，
video 字段已 VAE 编码为 latent (16, 9, 28, 28)。

用法:
  python convert_raw_to_cosmos.py \
      --input  dataset_raw/plug_in_ethernet/success_episodes/ \
      --output dataset_cosmos/plug_in_ethernet_cosmos/ \
      --task "Plug in the ethernet cable type-c and usb"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 内联工具函数（避免 cosmos_policy 的重依赖链）
# 直接复制核心逻辑，和原始函数完全一致
# ---------------------------------------------------------------------------

def duplicate_array(arr: np.ndarray, total_num_copies: int = 4) -> np.ndarray:
    """沿新 axis=0 堆叠 arr N 次。来源: cosmos_policy/utils/utils.py"""
    return np.stack([arr] * total_num_copies)

def rescale_proprio(proprio: np.ndarray, dataset_stats: dict,
                    non_negative_only: bool = False, scale_multiplier: float = 1.0) -> np.ndarray:
    """Min-max 归一化到 [-1,+1]。来源: cosmos_utils.py:657"""
    curr_min = dataset_stats["proprio_min"]
    curr_max = dataset_stats["proprio_max"]
    if not non_negative_only:
        arr = 2 * ((proprio - curr_min) / (curr_max - curr_min)) - 1
    else:
        arr = (proprio - curr_min) / (curr_max - curr_min)
    return scale_multiplier * arr

def prepare_images_for_model(images: List[np.ndarray], cfg,
                             flip_images: bool = False) -> List[np.ndarray]:
    """图像预处理: resize 224 + 可选 JPEG 压缩模拟 + 中心裁剪。来源: cosmos_utils.py:532"""
    import io as _io
    images = np.stack(images, axis=0)  # (N, H, W, C)
    if flip_images:
        images = np.flipud(images)
    if getattr(cfg, "use_jpeg_compression", False):
        for i in range(len(images)):
            img = images[i].astype(np.uint8)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            images[i] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if getattr(cfg, "trained_with_image_aug", False):
        # 90% 面积中心裁剪 → resize 回原尺寸
        _, H, W, _ = images.shape
        crop_sz = int(H * 0.9**0.5)
        y1 = (H - crop_sz) // 2; x1 = (W - crop_sz) // 2
        cropped = images[:, y1:y1 + crop_sz, x1:x1 + crop_sz, :]
        result = []
        for i in range(len(cropped)):
            result.append(cv2.resize(cropped[i], (W, H), interpolation=cv2.INTER_LINEAR))
        images = np.array(result)
    return list(images)

def compute_monte_carlo_returns(num_steps: int, terminal_reward: float, gamma: float) -> np.ndarray:
    """MC 折现回报 → 缩放至 [-1, 1]。来源: dataset_common.py:25"""
    rewards = np.zeros(num_steps, dtype=np.float32)
    rewards[-1] = terminal_reward
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(num_steps)):
        G = rewards[t] + gamma * G
        returns[t] = G
    if terminal_reward > 0:
        returns = 2 * returns / terminal_reward - 1
    else:
        returns = 2 * returns - 1
    return returns

def get_action_chunk_with_padding(actions: np.ndarray, relative_step_idx: int,
                                   chunk_size: int, num_steps: int) -> np.ndarray:
    """动作序列中切 chunk，末尾不足则用最后一帧填充。来源: dataset_common.py:58"""
    remaining = num_steps - relative_step_idx
    if remaining >= chunk_size:
        return actions[relative_step_idx:relative_step_idx + chunk_size]
    chunk = np.zeros((chunk_size, actions.shape[1]), dtype=actions.dtype)
    chunk[:remaining] = actions[relative_step_idx:]
    if remaining > 0:
        chunk[remaining:] = actions[-1]
    return chunk

# Lerobot 数据集（只需这一项）
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 从 collect_data_cosmos.py 内联（避免导入 pynput 触发 X display 错误）
def cosmos_obs_to_transition_state(obs: dict) -> dict:
    """和 collect_data_cosmos.py:250 完全一致"""
    state = {}
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            state[key] = val.detach().cpu()
        elif isinstance(val, np.ndarray):
            state[key] = torch.from_numpy(val)
        elif isinstance(val, (int, float)):
            state[key] = torch.tensor(val)
    return state

def cosmos_encoded_state_to_frame(state: dict) -> dict:
    """和 collect_data_cosmos.py:211 完全一致"""
    frame_state = {}
    for key, val in state.items():
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()
            if val.ndim > 0 and val.shape[0] == 1:
                val = val.squeeze(0)
            arr = val.float().numpy()
        else:
            arr = np.asarray(val)
        if arr.ndim == 0:
            arr = np.array([arr.item()], dtype=arr.dtype)
        frame_state[key] = arr
    return frame_state

def sanitize_info_for_transition(info: dict) -> dict:
    """和 collect_data_cosmos.py:159 完全一致"""
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

COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ConversionConfig:
    """转换全量参数"""
    input_dir: str                    # success_episodes/ 目录
    output_dir: str                   # LeRobot 数据集输出目录
    task_description: str             # 任务语言描述

    # Cosmos 模型
    cosmos_config_path: str = "train_config_cosmos.json"
    t5_embeddings_path: str = ""      # T5 embeddings pkl，留空则实时编码

    # 图像
    image_size: int = 224
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True

    # 动作
    chunk_size: int = 16              # action chunk 长度
    action_scale: Tuple[float, float, float] = field(
        default_factory=lambda: (0.05, 0.5, 1.0)
    )  # 默认值，推荐用 compute_action_scale() 从数据统计

    # VAE 编码
    encode_batch_size: int = 1      # GPU VAE 编码 micro-batch (64 会 OOM)
    batch_size: int = 1               # 训练 batch_size (对齐配置)

    # 输出
    output_fps: int = 16

    # 控制
    max_episodes: int = 0             # 0 = 全部
    resume: bool = True               # 跳过已处理的 episode

    # 运行时长
    _start_time: float = field(default_factory=time.time)

# ---------------------------------------------------------------------------
# HDF5 读取
# ---------------------------------------------------------------------------

def discover_trajectories(input_dir: Path) -> List[Path]:
    """递归发现所有 trajectory.hdf5"""
    trajs = sorted(input_dir.rglob("trajectory.hdf5"))
    return [t for t in trajs if t.is_file()]

def read_hdf5_metadata(h5_path: Path) -> dict:
    """读取 HDF5 的元数据和字段列表，不加载图像"""
    with h5py.File(h5_path, "r") as f:
        puppet_keys = list(f["puppet"].keys())
        cam_keys = list(f["camera_observations/color_images"].keys())

        # 检测单臂还是双臂
        has_left_ee = "end_effector_left_pose_align" in puppet_keys
        has_right_ee = "end_effector_right_pose_align" in puppet_keys
        is_dual = has_left_ee and has_right_ee

        T = len(f["puppet/end_effector_left_pose_align/data"]
                if has_left_ee
                else f["puppet/end_effector/data"])

        return {
            "num_steps": T,
            "is_dual_arm": is_dual,
            "cameras": cam_keys,
            "puppet_keys": puppet_keys,
            "has_left_ee": has_left_ee,
            "has_right_ee": has_right_ee,
            "has_delta_ee": "delta_end_effector" in puppet_keys,
        }

def detect_robot_type(meta: dict) -> str:
    """推断 : 'dual' | 'single_delta' | 'single_absolute'"""
    if meta["is_dual_arm"]:
        return "dual"
    if meta["has_delta_ee"]:
        return "single_delta"
    return "single_absolute"

# ---------------------------------------------------------------------------
# 数据提取（一帧）
# ---------------------------------------------------------------------------

def extract_ee_pose_and_gripper(
    h5: h5py.File, t: int, robot_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (pose, gripper)
      dual:          pose=(14,) [left7, right7],  gripper=(2,) [left, right]
      single_delta:  pose=(7,)  xyz+qxyzw,         gripper=(1,)
      single_absolute: 同上
    """
    if robot_type == "dual":
        left_ee = np.asarray(h5["puppet/end_effector_left_pose_align/data"][t], dtype=np.float32)
        right_ee = np.asarray(h5["puppet/end_effector_right_pose_align/data"][t], dtype=np.float32)
        left_grip = np.asarray(h5["puppet/end_effector_left_position_align/data"][t], dtype=np.float32).reshape(1)
        right_grip = np.asarray(h5["puppet/end_effector_right_position_align/data"][t], dtype=np.float32).reshape(1)
        return (np.concatenate([left_ee, right_ee]),
                np.concatenate([left_grip, right_grip]))
    else:
        ee = np.asarray(h5["puppet/end_effector/data"][t], dtype=np.float32)
        grip = np.asarray(h5["puppet/hand_joint_position/data"][t], dtype=np.float32).reshape(1)
        return ee, grip

def decode_camera_image(h5: h5py.File, cam_name: str, t: int) -> np.ndarray:
    """解码一帧 JPEG → RGB uint8 (H, W, 3)"""
    jpeg_bytes = h5[f"camera_observations/color_images/{cam_name}"][t]
    buf = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"JPEG decode failed: {cam_name} frame {t}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------
# 动作转换
# ---------------------------------------------------------------------------

def pose_7d_to_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """xyz(3) + quat(4) → 4×4 齐次矩阵"""
    xyz = pose_7d[:3].astype(np.float64)
    quat = pose_7d[3:7].astype(np.float64)
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

def compute_delta_action_single(
    pose_curr_7d: np.ndarray,
    pose_future_7d: np.ndarray,
    grip_curr: float,
    grip_future: float,
    action_scale: Tuple[float, float, float],
) -> np.ndarray:
    """单臂：绝对位姿 → 增量欧拉角动作 (dx,dy,dz, rx,ry,rz, grip)"""
    T_curr = pose_7d_to_matrix(pose_curr_7d)
    T_future = pose_7d_to_matrix(pose_future_7d)
    T_delta = np.linalg.inv(T_curr) @ T_future

    dpos = T_delta[:3, 3].astype(np.float32)
    drot = Rotation.from_matrix(T_delta[:3, :3]).as_euler("xyz").astype(np.float32)
    dgrip = np.float32(grip_future - grip_curr)

    action = np.concatenate([
        dpos / action_scale[0],      # (3,) 平移
        drot / action_scale[1],      # (3,) 旋转
        [dgrip / action_scale[2]],   # (1,) 夹爪
    ])  # (7,)
    return np.clip(action, -1.0, 1.0)

def compute_delta_action_dual(
    pose_curr_14d: np.ndarray,
    pose_future_14d: np.ndarray,
    grip_curr: np.ndarray,
    grip_future: np.ndarray,
    action_scale: Tuple[float, float, float],
) -> np.ndarray:
    """双臂：左(7) + 右(7) → 14D"""
    left = compute_delta_action_single(
        pose_curr_14d[:7], pose_future_14d[:7],
        grip_curr[0], grip_future[0], action_scale,
    )
    right = compute_delta_action_single(
        pose_curr_14d[7:], pose_future_14d[7:],
        grip_curr[1], grip_future[1], action_scale,
    )
    return np.concatenate([left, right])  # (14,)

def compute_action_scale(
    traj_files: List[Path],
    robot_type: str,
    chunk_size: int = 16,
    sample_limit: int = 5,
) -> Tuple[float, float, float]:
    """
    从全量 episode 统计增量分布，计算 action_scale。

    对每个 episode 的每对 (t, t+chunk_size) 计算增量位姿，
    取 99 分位数作为 scale，保证 99% 的增量 ∈ [-1, +1]。
    sample_limit: 最多采样几个 episode（加快速度）。
    """
    all_dpos = []
    all_drot = []
    all_dgrip = []

    for h5_path in traj_files[:sample_limit]:
        with h5py.File(h5_path, "r") as f:
            if robot_type == "dual":
                left_key = "puppet/end_effector_left_pose_align/data"
                right_key = "puppet/end_effector_right_pose_align/data"
                grip_l_key = "puppet/end_effector_left_position_align/data"
                grip_r_key = "puppet/end_effector_right_position_align/data"
            else:
                left_key = "puppet/end_effector/data"
                grip_l_key = "puppet/hand_joint_position/data"
                right_key = None

            T = len(f[left_key])
            for t in range(0, T - chunk_size, chunk_size):
                # 左臂（或唯一臂）
                T_curr = pose_7d_to_matrix(f[left_key][t])
                T_fut = pose_7d_to_matrix(f[left_key][min(t + chunk_size, T - 1)])
                T_delta = np.linalg.inv(T_curr) @ T_fut
                all_dpos.append(np.abs(T_delta[:3, 3]))
                all_drot.append(np.abs(Rotation.from_matrix(T_delta[:3, :3]).as_euler("xyz")))
                all_dgrip.append(np.abs(
                    f[grip_l_key][min(t + chunk_size, T - 1)] - f[grip_l_key][t]
                ))

                # 右臂
                if right_key and robot_type == "dual":
                    T_curr_r = pose_7d_to_matrix(f[right_key][t])
                    T_fut_r = pose_7d_to_matrix(f[right_key][min(t + chunk_size, T - 1)])
                    T_delta_r = np.linalg.inv(T_curr_r) @ T_fut_r
                    all_dpos.append(np.abs(T_delta_r[:3, 3]))
                    all_drot.append(np.abs(Rotation.from_matrix(T_delta_r[:3, :3]).as_euler("xyz")))
                    all_dgrip.append(np.abs(
                        f[grip_r_key][min(t + chunk_size, T - 1)] - f[grip_r_key][t]
                    ))

    scale_pos = float(np.percentile(np.concatenate(all_dpos), 99))
    scale_rot = float(np.percentile(np.concatenate(all_drot), 99))
    scale_grip = float(np.percentile(np.concatenate(all_dgrip), 99))

    # 防止为零
    scale_pos = max(scale_pos, 0.001)
    scale_rot = max(scale_rot, 0.001)
    scale_grip = max(scale_grip, 0.01)

    print(f"[INFO] computed action_scale: pos={scale_pos:.4f}  rot={scale_rot:.4f}  grip={scale_grip:.4f}")
    return (scale_pos, scale_rot, scale_grip)
# ---------------------------------------------------------------------------

def compute_proprio_stats(
    traj_files: List[Path],
    robot_type: str,
    sample_limit: int = 5,
) -> Dict[str, np.ndarray]:
    """
    统计所有 episode 的 proprio 分布，返回 min/max 用于 rescale_proprio。
    遍历所有 episode，收集 absolute EE pose + gripper 的值范围。
    """
    all_vals = []
    for h5_path in traj_files[:sample_limit]:
        with h5py.File(h5_path, "r") as f:
            if robot_type == "dual":
                left_ee = f["puppet/end_effector_left_pose_align/data"][:]
                right_ee = f["puppet/end_effector_right_pose_align/data"][:]
                left_grip = f["puppet/end_effector_left_position_align/data"][:]
                right_grip = f["puppet/end_effector_right_position_align/data"][:]
                all_vals.append(np.concatenate([left_ee, right_ee, left_grip, right_grip], axis=1))
            else:
                ee = f["puppet/end_effector/data"][:]
                grip = f["puppet/hand_joint_position/data"][:]
                all_vals.append(np.concatenate([ee, grip], axis=1))

    all_vals = np.concatenate(all_vals, axis=0)  # (N, D)
    return {
        "proprio_min": all_vals.min(axis=0).astype(np.float32),
        "proprio_max": all_vals.max(axis=0).astype(np.float32),
    }

def build_proprio(
    pose: np.ndarray,
    grip: np.ndarray,
    dataset_stats: dict | None = None,
) -> np.ndarray:
    """
    末端位姿 + 夹爪拼接为 proprio，归一化到 [-1, +1]。
    dataset_stats 必须包含 "proprio_min" / "proprio_max"。
    """
    raw = np.concatenate([pose, grip]).astype(np.float32)
    if dataset_stats:
        return rescale_proprio(raw, dataset_stats, non_negative_only=False)
    return raw

# ---------------------------------------------------------------------------
# 图像预处理 & 序列构造
# ---------------------------------------------------------------------------

def prepare_wrist_image(
    left_img: np.ndarray,       # (480, 640, 3) or pre-resized
    right_img: np.ndarray | None,
    is_dual: bool,
    target_size: int = 224,
) -> np.ndarray:
    """处理腕部图像：双臂先上下拼接(960×640)再 resize→224"""
    if is_dual and right_img is not None:
        # 在原始分辨率拼接，保留更多细节
        fused = np.concatenate([left_img, right_img], axis=0)  # (960, 640, 3)
        return cv2.resize(fused, (target_size, target_size))
    else:
        return cv2.resize(left_img, (target_size, target_size))

def build_nine_frame_sequence(
    wrist_img: np.ndarray,      # (224, 224, 3) uint8
    primary_img: np.ndarray,    # (224, 224, 3) uint8
) -> torch.Tensor:
    """和 CosmosWrapper.observation() 完全一致的 9 帧构造 → (1, 3, 33, 224, 224) uint8"""
    blank = np.zeros_like(primary_img)
    blank_dup = duplicate_array(blank.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    wrist_dup = duplicate_array(wrist_img, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_dup = duplicate_array(primary_img, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)

    sequence = [
        np.expand_dims(blank, axis=0),    # f0: blank (1 帧)
        blank_dup,                         # f1-f4: proprio 占位
        wrist_dup,                         # f5-f8: wrist
        primary_dup,                       # f9-f12: primary
        blank_dup.copy(),                  # f13-f16: action 占位
        blank_dup.copy(),                  # f17-f20: future_proprio 占位
        wrist_dup.copy(),                  # f21-f24: future_wrist
        primary_dup.copy(),                # f25-f28: future_primary
        blank_dup.copy(),                  # f29-f32: value 占位
    ]

    raw = np.concatenate(sequence, axis=0)               # (33, 224, 224, 3)
    raw = np.expand_dims(raw, axis=0)                    # (1, 33, 224, 224, 3)
    raw = np.transpose(raw, (0, 4, 1, 2, 3))             # (1, 3, 33, 224, 224)
    return torch.from_numpy(raw).to(dtype=torch.uint8)

# ---------------------------------------------------------------------------
# Episode 级别：构造 Transition 列表
# ---------------------------------------------------------------------------

def build_transition_list_for_episode(
    h5_path: Path,
    robot_type: str,
    t5_embedding: torch.Tensor,
    cfg: ConversionConfig,
    proprio_stats: dict | None = None,
) -> tuple[List[dict], dict]:
    """
    读一个 episode 的 HDF5 → 返回 transition_list 和 episode 元信息。

    每个 transition 的 state 格式与 CosmosWrapper 输出的 data_batch 一致。
    """
    with h5py.File(h5_path, "r") as f:
        meta = read_hdf5_metadata(h5_path)
        T = meta["num_steps"]
        is_dual = meta["is_dual_arm"]
        cameras = meta["cameras"]

        # 确定相机映射
        # camera_head / camera_top → primary
        # camera_left → wrist1, camera_right → wrist2
        primary_cam = next((c for c in cameras if "head" in c or "top" in c), cameras[0])
        left_cam = next((c for c in cameras if "left" in c), None)
        right_cam = next((c for c in cameras if "right" in c), None)
        wrist_cam = left_cam or right_cam or cameras[0]

        transition_list = []

        for t in range(T):
            # ── 图像 ──
            img_primary = decode_camera_image(f, primary_cam, t)
            img_left = decode_camera_image(f, left_cam, t) if left_cam else img_primary

            if is_dual and right_cam:
                img_right = decode_camera_image(f, right_cam, t)
            else:
                img_right = None

            # 预处理
            wrist_raw = prepare_wrist_image(img_left, img_right, is_dual)
            primary_raw = cv2.resize(img_primary, (224, 224))

            all_imgs = prepare_images_for_model(
                [wrist_raw, primary_raw],
                cfg,
                flip_images=False,
            )
            wrist_processed = all_imgs[0]
            primary_processed = all_imgs[1]

            video_tensor = build_nine_frame_sequence(wrist_processed, primary_processed)

            # ── Proprio ──
            pose, grip = extract_ee_pose_and_gripper(f, t, robot_type)
            raw_proprio = build_proprio(pose, grip, dataset_stats=proprio_stats)
            proprio_t = torch.from_numpy(raw_proprio).reshape(1, -1).to(torch.bfloat16)

            # ── data_batch ──
            data_batch = {
                "video": video_tensor,
                "proprio": proprio_t,
                "t5_text_embeddings": t5_embedding.unsqueeze(0).to(torch.bfloat16),
                "fps": torch.tensor([cfg.output_fps], dtype=torch.bfloat16),
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

            state = cosmos_obs_to_transition_state(data_batch)

            # ── Action (chunk) ──
            future_t = min(t + cfg.chunk_size, T - 1)
            pose_f, grip_f = extract_ee_pose_and_gripper(f, future_t, robot_type)

            if is_dual:
                action = compute_delta_action_dual(
                    pose, pose_f, grip, grip_f, cfg.action_scale,
                )
            else:
                action = compute_delta_action_single(
                    pose, pose_f, grip[0], grip_f[0], cfg.action_scale,
                )

            transition_list.append({
                "state": state,
                "action": torch.from_numpy(action).float(),
                "reward": 0.0,
                "next_state": {},  # encode_episode_state 内填充
                "done": (t == T - 1),
                "truncated": False,
                "complementary_info": sanitize_info_for_transition(
                    {"is_intervention": True}
                ),
            })

    episode_info = {
        "num_steps": T,
        "is_dual_arm": is_dual,
        "action_dim": int(action.shape[0]) if "action" in dir() else 7,
    }
    return transition_list, episode_info

# ---------------------------------------------------------------------------
# 编码 + 写入
# ---------------------------------------------------------------------------

def encode_and_write_episode(
    transition_list: List[dict],
    episode_index: int,
    dataset: LeRobotDataset,
    policy: Any,
    cosmos_cfg: Any,
    cfg: ConversionConfig,
    task_description: str,
) -> int:
    """VAE 编码 episode 的 transition，写入 LeRobot 数据集。返回成功写入的帧数。"""
    # 移动 policy 到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode_device = device

    t0 = time.time()
    # 清空模型加载和上一 round 的显存碎片
    torch.cuda.empty_cache()
    # 核心调用 — 和 collect_data_cosmos.py 完全一致
    encode_world_size = int(
        os.environ.get("ENCODE_WORLD_SIZE", str(max(1, torch.cuda.device_count())))
    )
    encoded = policy.encode_episode_state(
        transition_list,
        cfg.encode_batch_size,
        encode_device,
        cosmos_cfg,
        cfg.batch_size,
        world_size=encode_world_size,
    )
    encode_time = time.time() - t0
    print(f"    VAE encode: {encode_time:.1f}s ({len(transition_list)} frames)")

    # 清空 GPU 缓存，防止跨 episode 内存累积
    torch.cuda.empty_cache()

    written = 0
    for transition in encoded:
        if not transition.get("complementary_info", {}).get("is_intervention", True):
            continue

        state = cosmos_encoded_state_to_frame(transition["state"])

        action_val = transition["action"]
        if isinstance(action_val, torch.Tensor):
            action_val = action_val.detach().cpu().numpy()

        reward_val = float(transition.get("reward", 0.0))
        done_val = bool(transition.get("done", False))

        frame = {
            **state,
            "action": action_val.astype(np.float32),
            "next.reward": np.array([reward_val], dtype=np.float32),
            "next.done": np.array([done_val], dtype=bool),
            "complementary_info.is_intervention": np.array([True], dtype=bool),
        }

        dataset.add_frame(frame, task=task_description)
        written += 1

    dataset.save_episode()
    return written

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_output_features(action_dim: int, proprio_dim: int) -> dict:
    """LeRobot 特征定义"""
    return {
        "video": {
            "dtype": "float32",
            "shape": (16, 9, 28, 28),
            "names": None,
        },
        "proprio": {
            "dtype": "float32",
            "shape": (proprio_dim,),
            "names": None,
        },
        "future_proprio": {
            "dtype": "float32",
            "shape": (proprio_dim,),
            "names": None,
        },
        "value_function_return": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.is_intervention": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_intervention"],
        },
    }

def init_cosmos_policy(cfg: ConversionConfig, action_dim: int = 7, proprio_dim: int = 8):
    """初始化 CosmosPolicy。只从 JSON 读必要字段，不依赖 draccus 严格校验。"""
    import json
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.cosmos.configuration_cosmos import CosmosConfig
    from lerobot.policies.cosmos.modeling_cosmos import CosmosPolicy

    # 1. 从 JSON 只读我们关心的字段
    SCRIPT_DIR = Path(__file__).resolve().parent
    cosmos_config_path = str(SCRIPT_DIR / cfg.cosmos_config_path)
    with open(cosmos_config_path) as f:
        raw = json.load(f)

    chunk_size = raw.get("chunk_size", 16)
    config_file = raw.get("config_file", "cosmos_policy/config/config.py")
    use_jpeg_compression = raw.get("use_jpeg_compression", True)
    trained_with_image_aug = raw.get("trained_with_image_aug", True)
    use_proprio = raw.get("use_proprio", True)
    normalize_proprio = raw.get("normalize_proprio", True)
    flip_images = raw.get("flip_images", False)

    # 2. 直接构造 CosmosConfig（不经过 TrainRLServerPipelineConfig/draccus）
    # cosmos_config_module 和 cosmos_experiment 使用默认值，与 train_config_cosmos.json 一致
    policy_cfg = CosmosConfig(
        input_features={
            "video":              PolicyFeature(type=FeatureType.VISUAL, shape=(3, 33, 224, 224)),
            "proprio":            PolicyFeature(type=FeatureType.STATE,  shape=(proprio_dim,)),
            "t5_text_embeddings": PolicyFeature(type=FeatureType.ENV,    shape=(512, 4096)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        use_torch_compile=False,  # 禁用 torch.compile，只用 VAE 不需要编译 DiT
    )

    # 3. 从 cosmos 实验配置加载 world config，并覆盖我们关心的字段
    cosmos_cfg = policy_cfg.load_world_config()
    cosmos_cfg.chunk_size = chunk_size
    cosmos_cfg.use_jpeg_compression = use_jpeg_compression
    cosmos_cfg.trained_with_image_aug = trained_with_image_aug
    cosmos_cfg.use_proprio = use_proprio
    cosmos_cfg.normalize_proprio = normalize_proprio
    cosmos_cfg.flip_images = flip_images

    # 4. 初始化模型
    print("[INFO] Loading CosmosPolicy model...")
    policy = CosmosPolicy(config=policy_cfg, cosmos_cfg=cosmos_cfg)
    policy.to("cuda")
    policy.on_train_start()
    policy.eval()
    print("[INFO] CosmosPolicy loaded.")

    return policy, cosmos_cfg

def main():
    parser = argparse.ArgumentParser(description="Raw HDF5 → Cosmos LeRobot 转换")
    parser.add_argument("--input", required=True, help="success_episodes/ 目录")
    parser.add_argument("--output", required=True, help="LeRobot 输出目录")
    parser.add_argument("--task", required=True, help="任务语言描述")
    parser.add_argument("--t5_embeddings", default="", help="T5 embeddings pkl 路径")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--encode_batch_size", type=int, default=64,
                        help="GPU VAE 编码 micro-batch 大小 (默认 64，A100 80G)")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    cfg = ConversionConfig(
        input_dir=args.input,
        output_dir=args.output,
        task_description=args.task,
        t5_embeddings_path=args.t5_embeddings,
        max_episodes=args.max_episodes,
        encode_batch_size=args.encode_batch_size,
        resume=not args.no_resume,
    )

    input_dir = Path(cfg.input_dir).expanduser().resolve()
    output_dir = Path(cfg.output_dir).expanduser().resolve()

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 1. 发现轨迹文件
    print("=" * 50)
    print("[STAGE 1/5] 扫描 HDF5 文件...")
    traj_files = discover_trajectories(input_dir)
    print(f"[STAGE 1/5] 发现 {len(traj_files)} 个 trajectory.hdf5 ✅")

    if cfg.max_episodes > 0:
        traj_files = traj_files[:cfg.max_episodes]

    if not traj_files:
        print("[ERROR] 未找到任何 trajectory.hdf5")
        sys.exit(1)

    # 2. 检测机器人类型 + 统计 action_scale + proprio stats
    print("=" * 50)
    print("[STAGE 2/5] 分析数据统计（机器人类型、action_scale、proprio）...")
    first_meta = read_hdf5_metadata(traj_files[0])
    robot_type = detect_robot_type(first_meta)
    is_dual = robot_type == "dual"
    action_dim = 14 if is_dual else 7
    proprio_dim = 16 if is_dual else 8

    print(f"[INFO] robot_type={robot_type}  action_dim={action_dim}  proprio_dim={proprio_dim}")
    print(f"[INFO] T={first_meta['num_steps']}  cameras={first_meta['cameras']}")

    print("[INFO] 统计 action_scale...")
    action_scale = compute_action_scale(traj_files, robot_type, cfg.chunk_size)
    cfg.action_scale = action_scale

    print("[INFO] 统计 proprio 范围...")
    proprio_stats = compute_proprio_stats(traj_files, robot_type)

    # 3. 初始化 CosmosPolicy
    print("=" * 50)
    print("[STAGE 3/5] 初始化 CosmosPolicy（加载 VAE + DiT 模型）...")
    policy, cosmos_cfg = init_cosmos_policy(cfg, action_dim=action_dim, proprio_dim=proprio_dim)
    print("[STAGE 3/5] CosmosPolicy 加载完成 ✅")

    # 4. T5 文本嵌入
    print("=" * 50)
    print("[STAGE 4/5] 准备文本嵌入...")
    if cfg.t5_embeddings_path and Path(cfg.t5_embeddings_path).exists():
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_t5_embedding_from_cache, init_t5_text_embeddings_cache)
        init_t5_text_embeddings_cache(cfg.t5_embeddings_path)
        t5_emb = get_t5_embedding_from_cache(cfg.task_description)
    else:
        print("[INFO] 无 T5 embeddings，使用零向量（单任务训练不影响）")
        t5_emb = torch.zeros(512, 4096, dtype=torch.bfloat16)

    print("[STAGE 4/5] 文本嵌入准备完成 ✅")
    print("=" * 50)
    print("[STAGE 5/5] 开始逐 episode 转换（含 VAE 编码）...")
    print(f"  共 {len(traj_files)} 个 episode")

    # 5. 创建输出数据集
    features = build_output_features(action_dim, proprio_dim)
    dataset = LeRobotDataset.create(
        f"cosmos_{output_dir.name}",
        fps=cfg.output_fps,
        features=features,
        root=str(output_dir),
        use_videos=False,  # video 存为 latent tensor，不是 mp4
    )
    print(f"[INFO] 输出数据集: {output_dir}")

    # 6. 逐 episode 处理
    total_written = 0
    for ep_idx, h5_path in enumerate(tqdm(traj_files, desc="转换")):
        try:
            print(f"\n  [{ep_idx+1}/{len(traj_files)}] 读取 HDF5: {h5_path.name}")
            transition_list, ep_info = build_transition_list_for_episode(
                h5_path, robot_type, t5_emb, cfg, proprio_stats,
            )
            print(f"    frames={len(transition_list)}  action_dim={ep_info['action_dim']}  dual_arm={ep_info['is_dual_arm']}")

            print(f"    VAE 编码中...")
            written = encode_and_write_episode(
                transition_list, ep_idx, dataset, policy, cosmos_cfg, cfg,
                cfg.task_description,
            )
            print(f"    VAE 编码完成 ✅, 写入 {written} 帧")
            total_written += written
        except Exception as exc:
            print(f"\n[FAIL] episode {ep_idx}: {h5_path.name}")
            traceback.print_exc()
            continue

    # 7. 完成
    elapsed = time.time() - cfg._start_time
    print(f"\n[DONE] {len(traj_files)} episodes → {total_written} frames")
    print(f"[DONE] 耗时 {elapsed:.0f}s ({elapsed/len(traj_files):.1f}s/ep)")
    print(f"[DONE] 输出: {output_dir}")

if __name__ == "__main__":
    main()
