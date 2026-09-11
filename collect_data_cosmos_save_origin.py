      
# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import sys
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    TransitionKey,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)

from pynput import keyboard

from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD, TRUNCATED
from lerobot.utils.utils import log_say
import hydra
import draccus
from lerobot.configs.default import DatasetConfig as LeRobotDatasetConfig
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.silri.configuration_silri import SiLRIConfig  # noqa: F401
from lerobot.configs.types import PolicyFeature, FeatureType

from lerobot.robots import so100_follower  # noqa: F401
from lerobot.scripts.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.queue import get_last_item_from_queue
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)
from make_env import make_env
from rl_envs.shared_state import shared_state
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk, replace_latent_with_proprio
logging.basicConfig(level=logging.INFO)
from omegaconf import OmegaConf
from lerobot.policies.factory import make_policy
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.libero.run_libero_eval import validate_config as validate_config_cosmos
from cosmos_policy.experiments.robot.cosmos_utils import load_dataset_stats


# 不改 TransitionKey：用普通字符串键挂原始观测（create_transition 返回的是 dict）
OBSERVATION_ORIGIN_KEY = "observation_origin"


# 创建过渡数据结构：用于存储单步交互的所有信息
def create_transition(
    observation=None, observation_origin=None, action=None, reward=None, done=None, truncated=None, info=None, complementary_data=None
):
    """Helper to create an EnvTransition dictionary."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        OBSERVATION_ORIGIN_KEY: observation_origin,
    }

# 数据集配置
@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None                   # 数据集保存路径
    num_episodes_to_record: int = 5           # 要记录的episode数
    replay_episode: int | None = None         # 要重放的episode索引
    push_to_hub: bool = False                 # 是否推送到hugging face hub上
    # cosmos:获取原始数据，与 LIBERODataset 对齐的 sample_dict 构造（可选 T5 表，键为语言指令字符串）
    t5_text_embeddings_path: str | None = None
    libero_chunk_size: int = 8
    libero_final_image_size: int = 224
    libero_num_duplicates_per_image: int = 4

# 主配置类：包含环境配置和数据集配置
@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def on_press(key):
    try:
        if str(key) == 'Key.scroll_lock':
            print("----------------set human intervention key to {}!----------------".format(shared_state.human_intervention_key))
            shared_state.human_intervention_key = not shared_state.human_intervention_key
            time.sleep(0.5)
        if str(key) == 'Key.space' or str(key) == 'Key.pause':
            print("----------------set terminate to true!----------------")
            shared_state.terminate = True
            time.sleep(0.5)
    except AttributeError:
        pass
try:
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
except Exception as e:
    print("error in keyboard listener:", e)
    exit(0)


def sanitize_info_for_transition(info: dict) -> dict:
    """Sanitize info to only include types supported by Transition.complementary_info.

    Allowed types per downstream consumer: torch.Tensor, float, int, bool.
    - np.ndarray -> torch.from_numpy(...)
    - list/tuple of numbers/bools -> torch.tensor([...])
    - np.bool_/np.integer/np.floating -> Python scalar via .item()
    - torch.Tensor -> kept as is
    Unsupported types are skipped with a warning to avoid runtime errors.
    """
    safe_info = {}
    for key, value in info.items():
        try:
            if isinstance(value, torch.Tensor):
                safe_info[key] = value
            elif isinstance(value, np.ndarray):
                # Convert arrays directly to tensor; device will be handled later
                safe_info[key] = torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                # If it's a sequence of numbers/bools, convert to tensor
                if all(isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)) for v in value):
                    safe_info[key] = torch.tensor([v.item() if isinstance(v, (np.integer, np.floating, np.bool_)) else v for v in value])
                else:
                    logging.warning(f"Dropping complementary_info[{key}] due to unsupported list/tuple contents type: {type(value)}")
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                safe_info[key] = value.item()
            elif isinstance(value, (int, float, bool)):
                safe_info[key] = value
            else:
                logging.warning(f"Dropping complementary_info[{key}] of unsupported type: {type(value)}")
        except Exception as e:
            logging.warning(f"Failed to sanitize complementary_info[{key}] ({type(value)}): {e}")
    return safe_info


def to_numpy_value(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert tensors/scalars to numpy for LeRobot dataset frames (always on CPU)."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            arr = np.array([value.item()])
        else:
            arr = value.numpy()
    else:
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = np.array([arr.item()])
    if dtype is not None:
        return arr.astype(dtype)
    return arr


def cosmos_encoded_state_to_frame(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert encoded cosmos transition state tensors to numpy arrays for dataset.add_frame."""
    frame_state = {}
    for key, val in state.items():
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()
            if val.ndim > 0 and val.shape[0] == 1:
                val = val.squeeze(0)
            arr = val.float().numpy()
        else:
            arr = np.asarray(val)
        # LeRobot scalar features use shape (1,), not ()
        if arr.ndim == 0:
            arr = np.array([arr.item()], dtype=arr.dtype)
        frame_state[key] = arr
    return frame_state


# 将环境返回的原始观测转换为策略网络的输入格式
def make_policy_obs(obs: dict, device: torch.device, robot_type: str) -> dict:
    # 先将numpy数组转换为Tensor，再调整维度顺序
    policy_obs = {}
    for keys in obs.keys():
        if "state" not in keys:
            img = torch.from_numpy(obs[keys]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            new_key = "observation.images." + keys
            policy_obs[new_key] = img
        else:
            state = torch.from_numpy(obs[keys]).float().unsqueeze(0).to(device)
            new_key = "observation.state"
            policy_obs[new_key] = state
    return policy_obs
    # 转换后的输出：
    # {
    #     "observation.images.camera_0": torch.Tensor (1, 3, H, W),  # 归一化到 [0,1]
    #     "observation.images.camera_1": torch.Tensor (1, 3, H, W),
    #     "observation.state": torch.Tensor (1, state_dim)
    # }
    
def cosmos_obs_to_transition_state(obs: dict) -> dict[str, torch.Tensor]:
    """Keep only tensor fields from a CosmosWrapper batch for RL transitions."""
    # 将不同数据类型的观测值统一转换为 torch.Tensor
    state: dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            state[key] = val.detach().cpu()
        elif isinstance(val, np.ndarray):
            state[key] = torch.from_numpy(val)
        elif isinstance(val, (int, float)):
            state[key] = torch.tensor(val)
        # 字符串仍然忽略
    return state


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        # # numpy 不支持 bfloat16；落盘/打印用 float32
        # if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        #     t = t.to(dtype=torch.float32)
        return t.numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray | None:
    """Convert CHW/HWC image array to HWC uint8. Returns None if not image-like."""
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    # CHW -> HWC
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] not in (1, 3, 4):
        return None
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        arr = (arr * 255.0) if max_v <= 1.5 else arr
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _safe_name(key: str) -> str:
    return key.replace("/", "_").replace(".", "_")


def _save_image_value(value: Any, out_path_prefix: Path) -> list[str]:
    """Save image-like tensors/arrays; return list of written file paths."""
    arr = _to_numpy(value)
    if arr is None:
        return []
    written: list[str] = []
    # video: (B, C, T, H, W) or (C, T, H, W)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] in (1, 3, 4):
        # (C, T, H, W)
        c, t, h, w = arr.shape
        for ti in range(t):
            img = _to_hwc_uint8(arr[:, ti, :, :])
            if img is None:
                continue
            path = Path(f"{out_path_prefix}_t{ti:02d}.png")
            Image.fromarray(img.squeeze(-1) if img.shape[-1] == 1 else img).save(path)
            written.append(str(path))
        return written
    img = _to_hwc_uint8(arr)
    if img is None:
        return []
    path = Path(f"{out_path_prefix}.png")
    Image.fromarray(img.squeeze(-1) if img.shape[-1] == 1 else img).save(path)
    written.append(str(path))
    return written


def _format_value_for_txt(value: Any) -> str:
    arr = _to_numpy(value)
    if arr is not None:
        if arr.size <= 64:
            return np.array2string(arr, precision=6, separator=", ", max_line_width=120)
        return f"ndarray(shape={arr.shape}, dtype={arr.dtype})"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    return repr(value)


def save_origin_obs_step(
    dump_root: Path,
    episode_idx: int,
    step_idx: int,
    obs_origin: dict,
    next_obs_origin: dict,
    proprio: Any,
    action: Any,
    reward: Any,
    done: Any,
    truncated: Any,
    complementary_info: dict | None,
) -> Path:
    """
    Save one step under: dump_root/episode_XXXX/step_YYYY/
      - obs_origin_*.png / next_obs_origin_*.png for image-like fields
      - step_YYYY.txt for non-image fields + action/reward/...
    """
    step_dir = dump_root / f"episode_{episode_idx:04d}" / f"step_{step_idx:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # 只落盘 obs.wrist / obs.right；其余字段写进 txt
    image_keys_keep = ("obs.wrist", "obs.right")
    image_keys = set()
    for prefix, obs_dict in (("obs_origin", obs_origin), ("next_obs_origin", next_obs_origin)):
        if not isinstance(obs_dict, dict):
            continue
        for key in image_keys_keep:
            val = obs_dict.get(key)
            if val is None:
                continue
            written = _save_image_value(val, step_dir / f"{prefix}_{_safe_name(key)}")
            if written:
                image_keys.add((prefix, key))

    lines = [
        f"episode_idx={episode_idx}",
        f"step_idx={step_idx}",
        f"reward={_format_value_for_txt(reward)}",
        f"done={_format_value_for_txt(done)}",
        f"truncated={_format_value_for_txt(truncated)}",
        f"proprio={_format_value_for_txt(proprio)}",
        f"action={_format_value_for_txt(action)}",
        "",
        "[complementary_info]",
    ]
    for key, val in (complementary_info or {}).items():
        lines.append(f"{key}={_format_value_for_txt(val)}")

    for prefix, obs_dict in (("obs_origin", obs_origin), ("next_obs_origin", next_obs_origin)):
        lines.append("")
        lines.append(f"[{prefix} non-image fields]")
        if not isinstance(obs_dict, dict):
            lines.append(f"(not a dict: {type(obs_dict)})")
            continue
        for key, val in obs_dict.items():
            if (prefix, key) in image_keys:
                continue
            if key == "obs.wrist" or key == "obs.right":
                lines.append(f"{key}={_format_value_for_txt(val)}")
            else:
                continue

    txt_path = step_dir / f"step_{step_idx:04d}.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return step_dir

# 执行一步环境交互，并处理过渡数据结构
def step_env_and_process_transition(
    env,
    action: torch.Tensor,
    env_cfg: any,
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    流程：
    1. env.step(action) → 获取 obs, reward, terminated, truncated, info
    2. make_policy_obs() → 将 obs 转换为策略网络输入格式
    3. create_transition() → 打包成过渡数据字典
    """
    # 执行一步动作，获取环境反馈，并处理成统一
    device = torch.device("cpu")
    action[2] = 1.0
    obs_origin, reward, terminated, truncated, info = env.step(action)
    # obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
    if env_cfg.policy_type != "cosmos":
        obs = make_policy_obs(obs_origin, device, env_cfg.robot_config.robot_type)
    else:
        obs = cosmos_obs_to_transition_state(obs_origin)

    new_transition = create_transition(
        observation=obs,
        observation_origin=obs_origin,
        action=action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        complementary_data=info,
    )
    return new_transition


# 主控制循环：数据录制
def control_loop(
    env: gym.Env,
    cfg: any,
    env_cfg: any,
    policy_cfg: any,
    cosmos_cfg: any,
    dataset_stats_cosmos: dict,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     cfg: gym_manipulator configuration
    主控制循环，负责：
    1. 创建 LeRobot 数据集
    2. 与环境交互，采集数据
    3. 将每一帧添加到数据集
    4. 回合结束时保存回合数据
    """
    policy = make_policy(
        cfg=policy_cfg.policy,
        env_cfg=policy_cfg.env,
    )

    device = torch.device("cpu")
    # Reset environment and processors
    obs_origin, info = env.reset()
    print('after reset')

    # 处理初始观测
    complementary_data = {
        "discrete_penalty": 0.0,
    }
    # obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
    if env_cfg.policy_type != "cosmos":
        obs = make_policy_obs(obs_origin, device, env_cfg.robot_config.robot_type)
    else:
        obs = cosmos_obs_to_transition_state(obs_origin)
    transition = create_transition(observation=obs, observation_origin=obs_origin, info=info, complementary_data=complementary_data)
    use_gripper = not env_cfg.robot_config.fix_gripper

    # 创建LeRobot数据集
    dataset = None
    if cfg.mode == "record":
        # 获取动作维度
        action_feature = cfg.env.features['action']
        # 定义数据集的特征（字段名称、数据类型、形状）
        features = {
            ACTION: {"dtype": "float32", "shape": (action_feature.shape[0]+1,), "names": None},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        # if use_gripper:
        features["complementary_info.discrete_penalty"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        }

        features["complementary_info.is_intervention"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_intervention"],
        }

        # 添加观测字段（图像和状态）
        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "observation.images" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }
            # cosmos: add video proprio future_proprio value_function_return
            if key == "video":
                features[key] = {
                    "dtype": "float32",
                    # "shape": value.squeeze(0).shape,    # lerobot规定写入的单帧shape是无batch的
                    "shape": (16, 9, 28, 28),
                    "names": None,
                }
            
            if key == "proprio":
                features[key] = {
                    # "dtype": "bfloat16",
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
                features["future_" + key] = {
                    # "dtype": "bfloat16",
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
                features["value_function_return"] = {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": None,
                }
        # Create dataset
        # 创建LeRobot数据集
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )
        print('cfg.dataset.root:', cfg.dataset.root)
        print('dataset.root.absolute():', dataset.root.absolute())
        input("press Enter to continue...")
        logging.info(f"Dataset will be saved to: {dataset.root.absolute()}")
        # print("==============================>>> features.keys:", features.keys())
        # print("==============================>>> features['video'].shape:", features['video']['shape'])
        

    # 数据采集循环
    # 初始化采样索引、episode索引、episode步数、episode开始时间
    libero_sample_idx = 0
    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()
    # Handle both PolicyFeature objects and dictionaries
    # 获取动作维度
    action_feature = cfg.env.features['action']
    if isinstance(action_feature, PolicyFeature):
        continuous_action_dim = action_feature.shape[0]
    else:
        continuous_action_dim = action_feature['shape'][0]
    
    terminate_count = 0
    episode_length_list = []
    success_frame_count = 0
    # cosmos: 仅保留“最近一个已完成 episode”的逐步记录，键为 step_idx
    completed_episode_steps: dict[int, dict[str, Any]] = {}
    transition_list = []
    origin_obs_dump_root = Path(cfg.dataset.root) / "origin_obs_dumps" if getattr(cfg.dataset, "root", None) else Path("origin_obs_dumps")
    origin_obs_dump_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"Origin obs dumps will be saved under: {origin_obs_dump_root.resolve()}")

    current_episode_steps: dict[int, dict[str, Any]] = {}
    # 采集指定数量的回合
    while episode_idx < cfg.dataset.num_episodes_to_record:
        # 创建全零动作，实际应该是从策略获得或者人工干预
        neutral_action = torch.tensor([0.0] * (continuous_action_dim + 1), dtype=torch.float32)

        # Use the new step function
        # 执行一步，获取过渡数据
        print("before step")
        transition = step_env_and_process_transition(
            env=env,
            action=neutral_action,
            env_cfg=env_cfg,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        # 将帧数据添加到数据集
        if cfg.mode == "record":
            # 提取观测数据（去掉批次维度，移到cpu）
            print("episode_step:", episode_step)
            # print("transition[TransitionKey.OBSERVATION]:", transition[TransitionKey.OBSERVATION].keys())
            if env_cfg.policy_type != "cosmos":
                next_obs = {
                    k: v.squeeze(0).cpu()
                    for k, v in transition[TransitionKey.OBSERVATION].items()
                    if isinstance(v, torch.Tensor)
                }
            else:
                next_obs = {
                    k: v.cpu()
                    for k, v in transition[TransitionKey.OBSERVATION].items()
                    if isinstance(v, torch.Tensor)
                }
            # Use teleop_action if available, otherwise use the action from the transition
            reward = transition[TransitionKey.REWARD]
            action_to_record = transition[TransitionKey.ACTION]
            if "is_intervention" in transition[TransitionKey.COMPLEMENTARY_DATA] and transition[TransitionKey.COMPLEMENTARY_DATA]["is_intervention"]:
                print("get intervention")
                if env_cfg.robot_config.robot_type == "sim":
                    action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
                else:
                    action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA]["intervene_action"]
                    print("action_to_record:", action_to_record)
                    print("action_to_record.shape:", action_to_record.shape)
                    print("action_to_record.dtype:", action_to_record.dtype)
                    action_to_record = torch.from_numpy(action_to_record)
            else:
                print('No intervention!!!!!!!!!!!!!!!!!!!')

            complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

            # cosmos: 记录每个transition需要的内容，这部分同actor cloud处理
            transition_list.append(
                Transition(
                    state=obs,
                    action=action_to_record,
                    reward=reward,
                    next_state=next_obs,
                    done=terminated,
                    truncated=truncated,
                    complementary_info=sanitize_info_for_transition(complementary),
                )
            )
            
            # next_obs_origin = {
            #     k: v.cpu() if isinstance(v, torch.Tensor) else v
            #     for k, v in transition[OBSERVATION_ORIGIN_KEY].items()
            # }
            next_obs_origin = transition[OBSERVATION_ORIGIN_KEY]
            # 按 step 落盘：图像 png + 其余字段 txt，目录为 episode_XXXX/step_YYYY/
            save_origin_obs_step(
                dump_root=origin_obs_dump_root,
                episode_idx=episode_idx,
                step_idx=episode_step,
                obs_origin=obs_origin,
                next_obs_origin=next_obs_origin,
                proprio=obs_origin["obs.state"],
                action=action_to_record,
                reward=reward,
                done=terminated,
                truncated=truncated,
                complementary_info=sanitize_info_for_transition(complementary),
            )

            # 添加到数据集
            if dataset is not None and complementary["is_intervention"] and policy_cfg.policy.type != "cosmos":
                # 构造单帧数据
                frame = {
                    **obs,
                    ACTION: action_to_record.cpu() if isinstance(action_to_record, torch.Tensor) else action_to_record,
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated], dtype=bool),
                    # TRUNCATED: np.array([truncated], dtype=bool),
                }
                frame["complementary_info.discrete_penalty"] = np.array([complementary["discrete_penalty"]], dtype=np.float32)
                frame["complementary_info.is_intervention"] = np.array([complementary["is_intervention"]], dtype=bool)
                # frame["task"] = cfg.dataset.task
                if frame["next.reward"] > 0:
                    print("add one success frame into dataset")
                    success_frame_count += 1
                dataset.add_frame(frame, task=env_cfg.task_name)

            obs = next_obs
            obs_origin = next_obs_origin
            
        episode_step += 1
        # 检查终止条件
        terminated = terminated or shared_state.terminate or truncated
        if terminated :
            terminate_count += 1 
        print('terminated:', terminated, '; terminate_count:', terminate_count)
        # Handle episode termination
        if terminated:
            # 只保留当前完成的 episode；下一回合开始后会重新累计 current_episode_steps
            completed_episode_steps = dict(current_episode_steps)
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            logging.info(
                "Saved current completed episode with %d steps (episode_idx=%d)",
                len(completed_episode_steps),
                episode_idx,
            )
            episode_length_list.append(episode_step)
            episode_step = 0
            episode_idx += 1

            # cosmos: encode + add frame into dataset
            if policy_cfg.policy.type == "cosmos":
                dataset, success_frame_count = encode_and_add_frame(policy, transition_list, dataset, env_cfg, policy_cfg, cosmos_cfg, device, dataset_stats_cosmos)
            
            # 保存当前回合数据，此时保存的是
            if dataset is not None:
                logging.info(f"Saving episode {episode_idx} with {success_frame_count} success frames")
                dataset.save_episode()
            print(f"------------------------>>> success save {episode_idx} episode")

            # Reset for new episode
            # 重置环境，准备下一个回合
            obs_origin, info = env.reset()
            print("------------  already reset env, in next episode  ------------")
            if env_cfg.policy_type != "cosmos":
                obs = make_policy_obs(obs_origin, device, env_cfg.robot_config.robot_type)
            else:
                obs = cosmos_obs_to_transition_state(obs_origin)
            transition = create_transition(observation=obs, info=info)
            # encode 会把 video 写成 latent (C=16, T=9)；下一回合必须清空，否则会与新的 uint8 video (C=3, T=33) 混用
            transition_list = []
            terminate_count = 0
            shared_state.terminate = False
            success_frame_count = 0
            
    episode_length_list = np.array(episode_length_list)
    print("episode_length_mean:", np.mean(episode_length_list)) 
    # 完成采集，保存并上传到hugging face hub
    if dataset is not None:
        logging.info(f"Dataset saved to: {dataset.root.absolute()}")
        input("Press Enter to continue...")
        if cfg.dataset.push_to_hub:
            logging.info("Pushing dataset to hub")
            dataset.push_to_hub()
    

def encode_and_add_frame(
    policy: Any,
    transition_list: list, 
    dataset: LeRobotDataset, 
    env_cfg: any, 
    policy_cfg: any, 
    cosmos_cfg: any,
    device: torch.device,
    dataset_stats_cosmos: dict,
) -> tuple[LeRobotDataset, int]:
    # cosmos处理一个episode的数据，encode
    # 这里的batch_size是为了和训练时候保持一致
    batch_size = policy_cfg.batch_size
    encode_device = get_safe_torch_device(try_device=policy_cfg.policy.device, log=True)
    encode_batch_size = 16
    success_frame_count = 0
    before_encode_time = time.time()
    print("transition_list[0]['action']:", transition_list[0]['action'])
    encode_world_size = int(
        os.environ.get("ENCODE_WORLD_SIZE", str(max(1, torch.cuda.device_count())))
    )
    transition_list = policy.encode_episode_state(
        transition_list,
        encode_batch_size,
        encode_device,
        cosmos_cfg,
        batch_size,
        dataset_stats_cosmos,
        world_size=encode_world_size,
    )
    after_encode_time = time.time()
    print("outside encode time: ", after_encode_time - before_encode_time)

    transition_step = 0
    for transition in transition_list:
        if dataset is not None and transition["complementary_info"]["is_intervention"]:
        # if dataset is not None:
            # if dataset is not None and transition["complementary_info"]["is_intervention"] and policy_cfg.policy.type == "cosmos":
            transition = move_transition_to_device(transition=transition, device=device)
            # tensor去除batch维度
            state = cosmos_encoded_state_to_frame(transition["state"])
            print("state['video'].max:", state['video'].max(), "state['video'].min:", state['video'].min())
            action = transition["action"].squeeze(0) if isinstance(transition["action"], torch.Tensor) else transition["action"]
            reward = transition["reward"].squeeze(0) if isinstance(transition["reward"], torch.Tensor) else transition["reward"]
            done = transition["done"] if isinstance(transition["done"], torch.Tensor) else transition["done"]
            complementary_info = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in transition["complementary_info"].items()
            }       
            # print("action.shape:", action.shape)
            # print("state['video'].shape:", state['video'].shape)
            # print("state['value_function_return'].dtype:", state['value_function_return'].dtype)
            # print("complementary_info.keys:", complementary_info.keys())  ['succeed', 'curr_pose_euler', 'is_intervention']

            transition_step += 1
            print("cosmos add frame into dataset")
            frame = {
                **state,
                ACTION: np.array(action, dtype=np.float32),
                REWARD: np.array([reward], dtype=np.float32),
                DONE: np.array([done], dtype=bool),
            }
            if policy_cfg.policy.use_gripper_penalty:
                frame["complementary_info.discrete_penalty"] = np.array([complementary_info["discrete_penalty"]], dtype=np.float32)
            else:
                frame["complementary_info.discrete_penalty"] = np.array([0], dtype=np.float32)  # 因为不经过reward wrapper暂时设置为0
            frame["complementary_info.is_intervention"] = np.array([complementary_info["is_intervention"]], dtype=bool)
            # if frame.get(REWARD, np.array([0]))[0] > 0:
            if frame["next.reward"] > 0:
                print("add one success frame into dataset")
                success_frame_count += 1
            dataset.add_frame(frame, task=env_cfg.task_name)
            print(f"------------------------------->>> add {transition_step} frame into dataset")
    
    return dataset, success_frame_count


# 回放模式：执行已录制的轨迹
def replay_trajectory(
    env, cfg
) -> None:
    """Replay recorded trajectory on robot environment."""
    # 从数据集加载并回放指定的回合
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"
    # 加载数据集
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    # 过滤出指定回合的数据
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    # 获取该回合的所有动作
    actions = episode_frames.select_columns(ACTION)
    # 重置环境
    _, info = env.reset()

    # 逐个执行动作
    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        # transition = action_processor(transition)
        action = transition[TransitionKey.ACTION]
        env.step(transition[TransitionKey.ACTION])



@hydra.main(config_path="./cfg", config_name="config", version_base=None) 
def main(env_cfg):
    if "franka" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_collect_data.json"
        policy_config_path = "../../train_config_silri_franka.json"
    elif "ur" in env_cfg.robot_config.robot_type:
        lerobot_config_path = "../../train_config_collect_data.json"
        policy_config_path = "../../train_config_silri_ur.json"
    else:
        raise ValueError(f"Invalid robot type: {env_cfg.robot_type}")

    with draccus.config_type("json"):
        cfg = draccus.parse(GymManipulatorConfig, lerobot_config_path, args=[f"--dataset.task={env_cfg.task_name}"])
    
    # 加入policy_cfg用于之后引入policy
    with draccus.config_type("json"):
        if not env_cfg.fix_gripper:
            policy_cfg = draccus.parse(TrainRLServerPipelineConfig, policy_config_path, args=[f"--policy.type={env_cfg.policy_type}", f"--policy.num_discrete_actions=2"])
        else:
            policy_cfg = draccus.parse(TrainRLServerPipelineConfig, policy_config_path, args=[f"--policy.type={env_cfg.policy_type}"])
    if env_cfg.dataset is not None:
        dataset_obj = OmegaConf.to_object(env_cfg.dataset)
        policy_cfg.dataset = LeRobotDatasetConfig(**dataset_obj)
    else:
        policy_cfg.dataset = None
    policy_cfg.validate()
    policy_cfg.job_name = env_cfg.task_name
    policy_cfg.env.features["observation.state"].shape = [14] if env_cfg.use_force else [8]
    policy_cfg.policy.input_features["observation.state"].shape = [14] if env_cfg.use_force else [8]
    
    # 加入cosmos_cfg
    cosmos_cfg = None
    if env_cfg.policy_type == "cosmos":
        with draccus.config_type("json"):
            cosmos_config_path = "../../train_config_cosmos.json"
            cosmos_cfg = draccus.parse(PolicyEvalConfig, cosmos_config_path, args=[])
            validate_config_cosmos(cosmos_cfg)
    if env_cfg.policy_type == "cosmos":
        dataset_stats_cosmos = load_dataset_stats(env_cfg.dataset_stats_path)
    else:
        dataset_stats_cosmos = None
    """Main entry point for gym manipulator script."""
    try:
        # # env = make_env(env_cfg, fake_env=False, use_human_intervention=env_cfg.use_human_intervention, classifier=True, use_gripper_penalty=True)
        # env = make_env(env_cfg, fake_env=True, use_human_intervention=env_cfg.use_human_intervention, classifier=True, use_gripper_penalty=True)
        env = make_env(
            env_cfg,
            fake_env=False,
            # fake_env=True,
            use_human_intervention=env_cfg.use_human_intervention,
            classifier=True,
            use_gripper_penalty=policy_cfg.policy.use_gripper_penalty,
            cfg=policy_cfg,
            cosmos_cfg=cosmos_cfg,
        )
    except Exception as e:
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    print('success make env')
    if cfg.mode == "replay":
        print("begin to replay trajectory")
        replay_trajectory(env, cfg)
        exit(0)

    control_loop(env, cfg, env_cfg, policy_cfg, cosmos_cfg, dataset_stats_cosmos)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"In collect_data.py: [{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)

    