# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Offline ReplayBuffer construction for Cosmos LIBERODataset (non-LeRobot schema).

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.buffer import ReplayBuffer
from lerobot.utils.transition import Transition

from cosmos_policy.datasets.libero_dataset import LIBERODataset


def _feat_entry(cfg: TrainRLServerPipelineConfig, key: str) -> dict[str, Any] | Any:
    feats = cfg.policy.input_features
    return feats[key] if isinstance(feats, dict) else getattr(feats, key)


def _feat_type_and_shape(cfg: TrainRLServerPipelineConfig, key: str) -> tuple[str, list[int]]:
    fe = _feat_entry(cfg, key)
    if isinstance(fe, dict):
        t = str(fe["type"])
        shape = list(fe["shape"])
    else:
        t = str(getattr(fe, "type", "STATE"))
        shape = list(getattr(fe, "shape"))
    return t, shape


def _select_image_hwc(key: str, episode: dict) -> np.ndarray:
    """Map policy observation key to LIBERO episode image array (T, H, W, 3) uint8."""
    lk = key.lower()
    if "wrist" in lk:
        return episode["wrist_images"]
    if "right" in lk or "left" in lk or "external" in lk or lk.endswith(".image") or "agent" in lk:
        return episode["images"]
    raise KeyError(
        f"Cannot map state key {key!r} to LIBERODataset images / wrist_images. "
        "Extend _select_image_hwc() or rename keys in policy.input_features."
    )


def _hwc_frame_to_chw1(
    frame_hwc_uint8: np.ndarray,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Single frame (H, W, 3) uint8 -> float32 (1, 3, out_h, out_w) in [0, 1]."""
    x = torch.from_numpy(np.asarray(frame_hwc_uint8, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float()
    x = x / 255.0
    if x.shape[-2] != out_h or x.shape[-1] != out_w:
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x


def _state_dict_for_timestep(
    episode: dict,
    t: int,
    state_keys: list[str],
    cfg: TrainRLServerPipelineConfig,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in state_keys:
        typ, shape = _feat_type_and_shape(cfg, key)
        if typ == "VISUAL":
            _, h, w = int(shape[0]), int(shape[1]), int(shape[2])
            arr_t = _select_image_hwc(key, episode)
            frame = arr_t[t]
            out[key] = _hwc_frame_to_chw1(frame, h, w)
        elif typ == "STATE":
            prop = episode["proprio"]
            row = prop[t]
            if isinstance(row, torch.Tensor):
                v = row.detach().float().flatten()
            else:
                v = torch.as_tensor(row, dtype=torch.float32).flatten()
            dim = int(shape[0])
            if v.numel() != dim:
                raise ValueError(
                    f"Proprio size mismatch for {key!r}: episode has {v.numel()} dims, "
                    f"policy.input_features expects {dim}. Align observation.state shape with LIBERO (e.g. 9)."
                )
            out[key] = v.unsqueeze(0)
        else:
            raise NotImplementedError(f"Unsupported feature type {typ!r} for key {key!r}")
    return out


def _action_tensor_1ba(episode: dict, t: int) -> torch.Tensor:
    a = episode["actions"][t]
    if isinstance(a, torch.Tensor):
        v = a.detach().float().flatten()
    else:
        v = torch.as_tensor(a, dtype=torch.float32).flatten()
    return v.unsqueeze(0)


def _libero_episode_transitions(
    episode: dict,
    state_keys: list[str],
    cfg: TrainRLServerPipelineConfig,
) -> list[Transition]:
    """One LIBERODataset demo episode -> LeRobot-style transitions (sparse terminal reward)."""
    T = int(episode["num_steps"])
    out: list[Transition] = []
    for t in range(T):
        state = _state_dict_for_timestep(episode, t, state_keys, cfg)
        action = _action_tensor_1ba(episode, t)
        done = t == T - 1
        reward = 1.0 if done else 0.0
        if done:
            next_state = {k: v.clone() for k, v in state.items()}
        else:
            next_state = _state_dict_for_timestep(episode, t + 1, state_keys, cfg)
        out.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=False,
                complementary_info=None,
            )
        )
    return out


def offline_replay_buffer_from_cosmos_libero(
    dataset: LIBERODataset,
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
    capacity: int | None = None,
    optimize_memory: bool = True,
    use_drq: bool = True,
) -> ReplayBuffer:
    """
    Build a ReplayBuffer from an instantiated LIBERODataset (Cosmos), using only in-memory
    demonstration episodes in ``dataset.data`` (same episodes Cosmos pretraining uses).

    State keys and tensor shapes must match ``cfg.policy.input_features`` (e.g. LIBERO
    proprio dim 9, not Franka 14, unless you pad/change config).
    """
    state_keys = list(cfg.policy.input_features.keys())
    if capacity is None:
        capacity = int(cfg.policy.offline_buffer_capacity)

    index_pairs: list[tuple[int, int]] = []
    for ep_idx in sorted(dataset.data.keys()):
        ep = dataset.data[ep_idx]
        n = int(ep["num_steps"])
        for t in range(n):
            index_pairs.append((ep_idx, t))

    if capacity < len(index_pairs):
        index_pairs = random.sample(index_pairs, capacity)

    transitions: list[Transition] = []
    for ep_idx, t in tqdm(index_pairs, desc="cosmos offline: episodes->transitions"):
        ep = dataset.data[ep_idx]
        T = int(ep["num_steps"])
        state = _state_dict_for_timestep(ep, t, state_keys, cfg)
        action = _action_tensor_1ba(ep, t)
        done = t == T - 1
        reward = 1.0 if done else 0.0
        if done:
            next_state = {k: v.clone() for k, v in state.items()}
        else:
            next_state = _state_dict_for_timestep(ep, t + 1, state_keys, cfg)
        transitions.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=False,
                complementary_info=None,
            )
        )

    return ReplayBuffer.from_transition_list(
        transitions,
        device=device,
        state_keys=state_keys,
        capacity=capacity,
        storage_device=storage_device,
        optimize_memory=optimize_memory,
        use_drq=use_drq,
    )
