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
"""
CosmosPolicy — LeRobot 侧封装，对接 NVIDIA ``cosmos_policy``（Predict2 + Policy）。

与上游代码的对应关系（本文件只 import / 调用，不修改 ``cosmos-policy`` 仓库）：

- **VAE**：``cosmos_policy.tokenizers_cosmos.wan2pt1`` 中的 ``Wan2pt1VAEInterface`` /
  ``CosmosPolicyWanVAE`` 由扩散模型内部 tokenizer 配置实例化，负责像素视频与潜变量的编解码。
- **训练**：``cosmos_policy.models.policy_text2world_model.CosmosPolicyDiffusionModel``（及
  ``CosmosPolicyVideo2WorldModel`` 子类）的 ``training_step``；内部经基类
  ``cosmos_policy._src.predict2.models.text2world_model.DiffusionModel`` 调用
  ``draw_training_sigma_and_epsilon``、去噪网络，并在策略子类中通过
  ``compute_loss_with_epsilon_and_sigma`` 计算损失。
- **实验配置**：``cosmos_policy.config.experiment.cosmos_policy_experiment_configs.cosmos_predict2_2b_480p_libero``
  定义了 LIBERO 上的 ``state_t``、tokenizer、WAN 重复帧数等；本模块对未写在
  ``CosmosConfig``（``configuration_cosmos.py`` 保持不变）里的字段使用与之对齐的默认布局，
  并允许运行时 ``setattr(cfg.policy, \"...\", ...)`` 覆盖。

数据流：LeRobot batch → ``_lerobot_batch_to_cosmos`` → ``cosmos_model.training_step``；
推理：``generate_samples_from_batch`` + ``cosmos_utils.extract_action_chunk_from_latent_sequence``。
"""
from __future__ import annotations

import logging
import pickle
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from cosmos_policy._src.predict2.tokenizers.base_vae import BaseVAE
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.cosmos.configuration_cosmos import CosmosConfig

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 与 ``cosmos_predict2_2b_480p_libero`` / LIBERO 数据集一致的默认布局
# （``configuration_cosmos.py`` 不含这些字段时由此提供）
# ---------------------------------------------------------------------------

_LIBERO_SLOT_ORDER: tuple[str, ...] = (
    "blank",
    "current_proprio",
    "current_wrist",
    "current_primary",
    "action",
    "future_proprio",
    "future_wrist",
    "future_primary",
    "value",
)

_LIBERO_LATENT_INDICES: dict[str, int] = {
    "blank": 0,
    "current_proprio": 1,
    "current_wrist": 2,
    "current_primary": 3,
    "action": 4,
    "future_proprio": 5,
    "future_wrist": 6,
    "future_primary": 7,
    "value": 8,
    "current_wrist2": -1,
    "future_wrist2": -1,
}
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")

# 初始化LeRobot格式的数据集
lerobot_dataset = L(LeRobotDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"), # 成功完成任务的专家演示数据
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl" # 预计算好的 T5 文本嵌入（任务指令编码，避免运行时重复计算）
    ),
    chunk_size=16, # 动作序列长度：模型一次预测 16 步连续动作（Cosmos 标准设置）
    use_image_aug=True, # 启用基础图像增强（亮度/对比度/饱和度/色相随机），提升泛化能力
    use_wrist_images=True, # 使用手腕相机图像（第一视角手部图像，帮助精细操作）
    use_proprio=True, # 使用本体感知信息（关节角度、夹爪状态等机器人自身状态）
    normalize_proprio=True, # 对本体感知数据做归一化（缩放到标准分布，让训练更稳定）
    normalize_actions=True, # 对动作数据做归一化（缩放到 [-1,1]，模型更容易学习）
    num_duplicates_per_image=4, # WAN 2.1 VAE 要求：4 张原始图像 → 生成 1 个潜在帧（必须设置为 4）
    use_stronger_image_aug=True, # 启用更强的数据增强（随机裁剪、透视变换、噪声等）
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),
    demonstration_sampling_prob=0.5, # 训练时采样专家演示数据的概率（50% 概率学动作）
    success_rollout_sampling_prob=0.5, # 训练时采样成功 rollout 数据的概率（50% 概率学未来状态）
    return_value_function_returns=True, # 计算并返回价值函数回报（reward-to-go），用于训练价值网络
    gamma=0.99, # 强化学习折扣因子，用于计算累积回报
)



@dataclass(frozen=True)
class _CosmosLeRobotLayout:
    """从 ``CosmosConfig`` + 可选 ``setattr`` 解析出的 Cosmos 侧超参。"""

    image_size: int
    chunk_size: int
    action_dim: int
    num_duplicates_per_image: int
    primary_image_key: str
    wrist_image_key: str | None
    wrist_image2_key: str | None
    use_proprio: bool
    latent_indices: dict[str, int]
    slot_order: tuple[str, ...]
    task_description: str
    t5_max_length: int
    t5_embedding_dim: int
    t5_text_embeddings_path: str
    num_denoising_steps: int
    n_action_steps: int


def _cosmos_layout_from_config(config: CosmosConfig) -> _CosmosLeRobotLayout:
    feats = list(config.input_features.keys())
    img_feats = [k for k in feats if k.startswith("observation.image")]
    primary = getattr(config, "primary_image_key", None) or (img_feats[0] if img_feats else "observation.image")
    wrist = getattr(config, "wrist_image_key", None)
    if wrist is None and len(img_feats) > 1:
        wrist = img_feats[1]
    wrist2 = getattr(config, "wrist_image2_key", None)
    if wrist2 is None and len(img_feats) > 2:
        wrist2 = img_feats[2]

    action_dim = int(config.output_features[ACTION].shape[0])
    chunk = int(getattr(config, "chunk_size", 16))
    image_size = int(getattr(config, "image_size", 224))
    nd = int(getattr(config, "num_duplicates_per_image", 4))
    use_proprio = bool(getattr(config, "use_proprio", True))

    latent_indices = getattr(config, "latent_indices", None) or dict(_LIBERO_LATENT_INDICES)
    slot_order = tuple(getattr(config, "_slot_order", _LIBERO_SLOT_ORDER))

    return _CosmosLeRobotLayout(
        image_size=image_size,
        chunk_size=chunk,
        action_dim=action_dim,
        num_duplicates_per_image=nd,
        primary_image_key=primary,
        wrist_image_key=wrist,
        wrist_image2_key=wrist2,
        use_proprio=use_proprio,
        latent_indices=latent_indices,
        slot_order=slot_order,
        task_description=str(getattr(config, "task_description", "complete the manipulation task")),
        t5_max_length=int(getattr(config, "t5_max_length", 512)),
        t5_embedding_dim=int(getattr(config, "t5_embedding_dim", 1024)),
        t5_text_embeddings_path=str(getattr(config, "t5_text_embeddings_path", "") or ""),
        num_denoising_steps=int(getattr(config, "num_denoising_steps", 9)),
        n_action_steps=int(getattr(config, "n_action_steps", getattr(config, "n_obs_steps", 1))),
    )


def _ensure_megatron_initialized() -> None:
    """单卡 / 已初始化 DDP 时按需初始化 Megatron 并行状态（Cosmos 内部可能依赖）。"""
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state  # type: ignore[import]

        if not dist.is_initialized():
            logger.info(
                "Skip megatron model-parallel init because torch.distributed is not initialized yet."
            )
            return

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
    except ImportError:
        logger.warning(
            "megatron-core is not importable. COSMOS may fail if context-parallel is required."
        )


def _load_t5_cache(cache_path: str) -> dict:
    with open(cache_path, "rb") as fh:
        return pickle.load(fh)


def _compute_t5_embedding_online(
    text: str, max_length: int, embedding_dim: int, device: str
) -> Tensor:
    from transformers import AutoTokenizer, T5EncoderModel  # type: ignore[import]

    model_name = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = encoder(**inputs)

    emb = outputs.last_hidden_state[0]
    if emb.shape[0] < max_length:
        pad = torch.zeros(max_length - emb.shape[0], emb.shape[1], device=device)
        emb = torch.cat([emb, pad], dim=0)
    else:
        emb = emb[:max_length]

    return emb.float()


def _resize_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    if frame.shape[0] == target_size and frame.shape[1] == target_size:
        return frame
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)


def _tensor_chw_to_hwc_uint8(t: Tensor) -> np.ndarray:
    arr = (t.cpu().float().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return arr


def _build_video_sequence(
    layout: _CosmosLeRobotLayout,
    current_primary: np.ndarray,
    current_wrist: np.ndarray | None,
    current_wrist2: np.ndarray | None,
    future_primary: np.ndarray | None,
    future_wrist: np.ndarray | None,
    future_wrist2: np.ndarray | None,
) -> np.ndarray:
    """与 LIBERO 构造一致：首帧 blank 仅 1 张，其余 slot 各 ``num_duplicates_per_image`` 张。"""
    H = W = layout.image_size
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    nd = layout.num_duplicates_per_image

    def _slot_tiled(frame: np.ndarray | None) -> np.ndarray:
        f = frame if frame is not None else blank
        f = _resize_frame(f, layout.image_size)
        return np.tile(f[np.newaxis], (nd, 1, 1, 1))

    frames: list[np.ndarray] = []
    for slot_name in layout.slot_order:
        if slot_name == "blank":
            frames.append(np.expand_dims(_resize_frame(blank, layout.image_size), axis=0))
        elif slot_name == "current_proprio":
            frames.append(_slot_tiled(None))
        elif slot_name == "current_wrist":
            frames.append(_slot_tiled(current_wrist))
        elif slot_name == "current_wrist2":
            frames.append(_slot_tiled(current_wrist2))
        elif slot_name == "current_primary":
            frames.append(_slot_tiled(current_primary))
        elif slot_name == "action":
            frames.append(_slot_tiled(None))
        elif slot_name == "future_proprio":
            frames.append(_slot_tiled(None))
        elif slot_name == "future_wrist":
            frames.append(_slot_tiled(future_wrist))
        elif slot_name == "future_wrist2":
            frames.append(_slot_tiled(future_wrist2))
        elif slot_name == "future_primary":
            frames.append(_slot_tiled(future_primary))
        elif slot_name == "value":
            frames.append(_slot_tiled(None))
        else:
            raise ValueError(f"Unknown slot name: {slot_name!r}")

    return np.concatenate(frames, axis=0)


def _make_latent_idx_tensor(value: int, batch_size: int, device: torch.device) -> Tensor:
    return torch.tensor([value] * batch_size, dtype=torch.int64, device=device)


def _fix_proprio_branch(
    layout: _CosmosLeRobotLayout,
    batch: dict[str, Tensor],
    B: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    state_key = "observation.state"
    if layout.use_proprio and state_key in batch:
        s = batch[state_key]
        if s.ndim == 3:
            s = s[:, -1]
        proprio = s.to(device).float()
        return proprio, proprio.clone()

    proprio_dim = 1
    if state_key in batch:
        proprio_dim = int(batch[state_key].shape[-1])
    return (
        torch.zeros(B, proprio_dim, device=device),
        torch.zeros(B, proprio_dim, device=device),
    )


def _lerobot_batch_to_cosmos(
    layout: _CosmosLeRobotLayout,
    batch: dict[str, Tensor],
    t5_embedding: Tensor,
    device: torch.device,
    *,
    is_inference: bool = False,
) -> dict[str, Tensor]:
    B = next(iter(batch.values())).shape[0]
    lidx = layout.latent_indices

    def _get_image(key: str | None) -> Tensor | None:
        if not key or key not in batch:
            return None
        t = batch[key]
        if t.ndim == 5:
            t = t[:, -1]
        return t

    prim_t = _get_image(layout.primary_image_key)
    if prim_t is None:
        raise KeyError(
            f"Cosmos batch requires primary images at key {layout.primary_image_key!r} "
            f"(setattr on config.policy, e.g. primary_image_key=...)."
        )
    wrist_t = _get_image(layout.wrist_image_key)
    wrist2_t = _get_image(layout.wrist_image2_key)

    def _get_future_image(key: str | None) -> Tensor | None:
        if not key or key not in batch:
            return None
        t = batch[key]
        if t.ndim == 5 and t.shape[1] >= 2:
            return t[:, -2]
        return None

    fut_prim_t = _get_future_image(layout.primary_image_key)
    fut_wrist_t = _get_future_image(layout.wrist_image_key)
    fut_wrist2_t = _get_future_image(layout.wrist_image2_key)

    H = W = layout.image_size

    def _t_to_hwc(t_bchw: Tensor | None, idx: int) -> np.ndarray | None:
        if t_bchw is None:
            return None
        return _tensor_chw_to_hwc_uint8(t_bchw[idx])

    video_list = []
    for b in range(B):
        seq = _build_video_sequence(
            layout,
            current_primary=_t_to_hwc(prim_t, b),
            current_wrist=_t_to_hwc(wrist_t, b),
            current_wrist2=_t_to_hwc(wrist2_t, b),
            future_primary=_t_to_hwc(fut_prim_t, b),
            future_wrist=_t_to_hwc(fut_wrist_t, b),
            future_wrist2=_t_to_hwc(fut_wrist2_t, b),
        )
        seq_t = torch.from_numpy(seq).permute(3, 0, 1, 2)
        video_list.append(seq_t)

    video = torch.stack(video_list, dim=0).to(device)
    proprio, future_proprio = _fix_proprio_branch(layout, batch, B, device)

    if not is_inference and ACTION in batch:
        act = batch[ACTION]
        if act.shape[1] >= layout.chunk_size:
            action_chunk = act[:, : layout.chunk_size].to(device).float()
        else:
            pad = act[:, -1:].expand(-1, layout.chunk_size - act.shape[1], -1)
            action_chunk = torch.cat([act, pad], dim=1).to(device).float()
    else:
        action_chunk = torch.zeros(B, layout.chunk_size, layout.action_dim, device=device)

    t5_emb = t5_embedding.unsqueeze(0).expand(B, -1, -1).to(device)
    t5_mask = torch.ones(B, layout.t5_max_length, dtype=torch.int64, device=device)

    def _idx(name: str) -> Tensor:
        v = int(lidx.get(name, -1))
        return _make_latent_idx_tensor(v, B, device)

    return {
        "video": video,
        "t5_text_embeddings": t5_emb.to(torch.bfloat16),
        "t5_text_mask": t5_mask,
        "fps": 16,
        "padding_mask": torch.zeros(B, 1, H, W, device=device),
        "image_size": layout.image_size * torch.ones(4, device=device),
        "actions": action_chunk,
        "proprio": proprio,
        "future_proprio": future_proprio,
        "rollout_data_mask": torch.zeros(B, dtype=torch.long, device=device),
        "world_model_sample_mask": torch.zeros(B, dtype=torch.long, device=device),
        "value_function_sample_mask": torch.zeros(B, dtype=torch.long, device=device),
        "value_function_return": torch.zeros(B, device=device),
        "action_latent_idx": _idx("action"),
        "current_proprio_latent_idx": _idx("current_proprio"),
        "current_wrist_image_latent_idx": _idx("current_wrist"),
        "current_wrist_image2_latent_idx": _idx("current_wrist2"),
        "current_image_latent_idx": _idx("current_primary"),
        "future_proprio_latent_idx": _idx("future_proprio"),
        "future_wrist_image_latent_idx": _idx("future_wrist"),
        "future_wrist_image2_latent_idx": _idx("future_wrist2"),
        "future_image_latent_idx": _idx("future_primary"),
        "value_latent_idx": _idx("value"),
    }


def _numpy_or_tensor_hwc_uint8(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        arr = (t.numpy() * 255).clip(0, 255).astype(np.uint8) if t.max() <= 1.0 + 1e-3 else t.numpy().astype(np.uint8)
        return arr
    arr = np.asarray(x)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.uint8)


class CosmosPolicy(
    PreTrainedPolicy
):
    """
    LeRobot 策略封装：训练时委托 ``CosmosPolicyVideo2WorldModel.training_step``；
    潜空间编解码由模型内 ``Wan2pt1VAEInterface``（见 ``cosmos_policy.tokenizers_cosmos.wan2pt1``）完成。
    """
    config_class = CosmosConfig
    name = "cosmos_policy"

    def __init__(
        self,
        config: CosmosConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._layout = _cosmos_layout_from_config(config) # 从config中解析出Cosmos侧超参？

        raw_stats = getattr(config, "dataset_stats", None)
        if dataset_stats is None and raw_stats is not None:
            dataset_stats = {
                feat_key: {
                    stat_key: torch.tensor(v, dtype=torch.float32) if isinstance(v, (list, tuple)) else v
                    for stat_key, v in stat_dict.items()
                }
                for feat_key, stat_dict in raw_stats.items()
            }

        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        _ensure_megatron_initialized()

        self.cosmos_model: nn.Module | None = None
        self._policy_eval_cfg: Any = None
        self._dataset_stats: dict[str, Any] | None = None
        self._training_iteration = 0

        # 加载T5文本嵌入缓存
        self._t5_cache: dict | None = None
        t5_path = self._layout.t5_text_embeddings_path
        if t5_path:
            self._t5_cache = _load_t5_cache(t5_path)
            logger.info("Loaded T5 cache with %d entries.", len(self._t5_cache))
        self._t5_embedding: Tensor | None = None

        self._queues: dict[str, deque] = {}
        self.reset()

    @property
    def _cosmos_model(self) -> nn.Module | None:
        return self.cosmos_model

    def set_cosmos_model(self, model: nn.Module | None) -> None:
        """由 ``cosmos_utils.get_model`` 等外部逻辑注入 ``CosmosPolicyVideo2WorldModel``。"""
        self.cosmos_model = model

    def attach_cosmos_eval(
        self,
        diffusion_model: nn.Module,
        policy_eval_cfg: Any,
        dataset_stats: dict[str, Any],
    ) -> None:
        """
        与官方 ``get_action`` 评估路径对齐：挂载 ``cosmos_model``、``PolicyEvalConfig``、数据集统计量。
        """
        self.set_cosmos_model(diffusion_model)
        self._policy_eval_cfg = policy_eval_cfg
        self._dataset_stats = dataset_stats

    def _load_cosmos_checkpoint(self, model: nn.Module, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if hasattr(model, "restore_checkpoint"):
            try:
                model.restore_checkpoint(str(ckpt_path))
                logger.info("Loaded COSMOS checkpoint via restore_checkpoint: %s", ckpt_path)
                return
            except Exception as e:
                logger.warning("restore_checkpoint failed (%s), falling back to torch.load.", e)

        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = state.get("model", state.get("state_dict", state))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s…", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s…", unexpected[:10])
        logger.info("Loaded COSMOS checkpoint via torch.load: %s", ckpt_path)

    def _get_t5_embedding(self) -> Tensor:
        if self._t5_embedding is not None:
            return self._t5_embedding

        device = self.config.device or "cpu"
        task = self._layout.task_description

        if self._t5_cache is not None and task in self._t5_cache:
            emb = self._t5_cache[task]
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            if emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb[0]
            self._t5_embedding = emb.float()
            return self._t5_embedding

        logger.info("Computing T5 embedding online for: '%s'", task)
        self._t5_embedding = _compute_t5_embedding_online(
            task,
            self._layout.t5_max_length,
            self._layout.t5_embedding_dim,
            device,
        )
        return self._t5_embedding

    def _task_label(self, policy_obs: dict[str, Any]) -> str:
        t = policy_obs.get("task")
        if t is not None and str(t).strip():
            return str(t)
        return self._layout.task_description

    def _batch_to_cosmos_obs(self, policy_obs: dict[str, Any]) -> dict[str, Any]:
        """供 ``cosmos_utils.get_action`` 使用的观测字典（numpy HWC + proprio）。"""
        out: dict[str, Any] = {}
        if "primary_image" in policy_obs:
            out["primary_image"] = _numpy_or_tensor_hwc_uint8(policy_obs["primary_image"])
        if "wrist_image" in policy_obs:
            out["wrist_image"] = _numpy_or_tensor_hwc_uint8(policy_obs["wrist_image"])
        if "proprio" in policy_obs:
            p = policy_obs["proprio"]
            out["proprio"] = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
        return out

    def get_optim_params(self) -> dict:
        if self.cosmos_model is None:
            raise RuntimeError("CosmosPolicy: cosmos_model is not set. Call set_cosmos_model / attach_cosmos_eval.")
        return {"actor": list(self.cosmos_model.parameters())}

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self._layout.n_action_steps)}
        for key in self.config.input_features:
            if key not in self._queues:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        if self.cosmos_model is None:
            raise RuntimeError("CosmosPolicy: cosmos_model is not set.")

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        device = next(self.cosmos_model.parameters()).device
        t5_emb = self._get_t5_embedding().to(device)

        cosmos_batch = _lerobot_batch_to_cosmos(
            self._layout,
            batch,
            t5_emb,
            device,
            is_inference=False,
        )
        output_dict, loss = self.cosmos_model.training_step(cosmos_batch, self._training_iteration)
        self._training_iteration += 1

        log_dict = {
            k: v.item() if isinstance(v, Tensor) and v.numel() == 1 else v
            for k, v in output_dict.items()
            if isinstance(v, (Tensor, float, int))
            and (not isinstance(v, Tensor) or v.numel() == 1)
        }
        return loss, log_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        from cosmos_policy.experiments.robot.cosmos_utils import (  # type: ignore[import]
            extract_action_chunk_from_latent_sequence,
        )

        if self.cosmos_model is None:
            raise RuntimeError("CosmosPolicy: cosmos_model is not set.")

        device = next(self.cosmos_model.parameters()).device
        t5_emb = self._get_t5_embedding().to(device)

        cosmos_batch = _lerobot_batch_to_cosmos(
            self._layout,
            batch,
            t5_emb,
            device,
            is_inference=True,
        )

        B = cosmos_batch["video"].shape[0]
        action_latent_idx = int(self._layout.latent_indices["action"])

        generated_latent = self.cosmos_model.generate_samples_from_batch(
            cosmos_batch,
            n_sample=B,
            num_steps=self._layout.num_denoising_steps,
            seed=0,
            is_negative_prompt=False,
        )

        action_indices = torch.full(
            (B,),
            action_latent_idx,
            dtype=torch.int64,
            device=generated_latent.device,
        )
        actions = extract_action_chunk_from_latent_sequence(
            generated_latent,
            action_shape=(self._layout.chunk_size, self._layout.action_dim),
            action_indices=action_indices,
        ).float()

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if any(k.startswith("observation.") for k in batch):
            if ACTION in batch:
                batch = {k: v for k, v in batch.items() if k != ACTION}
            batch = self.normalize_inputs(batch)
            self._queues = populate_queues(self._queues, batch)
            if len(self._queues[ACTION]) == 0:
                obs_batch = {
                    k: torch.stack(list(self._queues[k]), dim=1)
                    for k in batch
                    if k in self._queues
                }
                actions = self.predict_action_chunk(obs_batch)
                self._queues[ACTION].extend(
                    actions[:, : self._layout.n_action_steps].transpose(0, 1)
                )
            return self._queues[ACTION].popleft()

        if self._policy_eval_cfg is not None and self.cosmos_model is not None and self._dataset_stats is not None:
            from cosmos_policy.experiments.robot.cosmos_utils import get_action  # type: ignore[import]

            _raw = {k: v for k, v in batch.items() if k != ACTION}
            obs_cosmos = self._batch_to_cosmos_obs(_raw)
            task = self._task_label(_raw)
            out = get_action(
                self._policy_eval_cfg,
                self.cosmos_model,
                self._dataset_stats,
                obs_cosmos,
                task,
                seed=int(getattr(self.config, "seed", 0)),
                randomize_seed=bool(getattr(self.config, "randomize_seed", False)),
                num_denoising_steps_action=int(
                    getattr(self.config, "num_denoising_steps_action", self._layout.num_denoising_steps)
                ),
                generate_future_state_and_value_in_parallel=bool(
                    getattr(self.config, "generate_future_state_and_value_in_parallel", True)
                ),
            )
            actions = out["actions"]
            if not isinstance(actions, list) or len(actions) < 1:
                raise RuntimeError(f"get_action returned invalid actions: {type(actions)}")
            step = torch.as_tensor(actions[0], dtype=torch.float32, device=self.config.device)
            if step.ndim == 1:
                step = step.unsqueeze(0)
            return step

        raise RuntimeError(
            "CosmosPolicy.select_action: expected LeRobot observation batch or attach_cosmos_eval + raw obs."
        )

    def as_cosmos_policy_submodules(self) -> Cosmos_Policy:
        """返回仅含 VAE + DiT 的子模块视图，便于与 SAC ``Policy(encoder=..., network=...)`` 对照调试。"""
        if self.cosmos_model is None:
            raise RuntimeError("CosmosPolicy.as_cosmos_policy_submodules: cosmos_model is not set.")
        return Cosmos_Policy.from_cosmos_diffusion_model(self.cosmos_model)


    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Cosmos_Policy(
            vae=self.vae,
            network=self.net,
            sigma_data=self.sigma_data,
        )



class Cosmos_Policy(nn.Module):
    """
    与 ``lerobot.policies.sac.modeling_sac.Policy`` 对照的命名拆分：SAC 里 ``encoder`` + ``network`` 产生动作分布；
    此处 ``encoder`` 对应 cosmos 训练管线里的 **VAE（tokenizer）**，``network`` 对应 **DiT**（``net``）。

    组件均从 ``cosmos_policy`` 侧已有模块引用，不重复实现编解码与 Transformer 逻辑：

    - **VAE**：``cosmos_policy._src.predict2.tokenizers.base_vae`` 与
      ``cosmos_policy.tokenizers_cosmos.wan2pt1`` 等，在完整模型上为 ``model.tokenizer``。
    - **DiT**：``cosmos_policy._src.predict2.networks.minimal_v1_lvg_dit`` 等，在完整模型上为 ``model.net``。

    潜变量缩放与 ``cosmos_policy._src.predict2.models.text2world_model.DiffusionModel`` 的
    ``encode`` / ``decode``（``sigma_data``）一致。

    若 ``_as_view=True``（``from_cosmos_diffusion_model`` 默认），子模块**不**再次注册到本类，
    避免与 ``CosmosPolicy.cosmos_model`` 争抢同一 ``tokenizer``/``net``；此时 ``Cosmos_Policy.parameters()``
    为空，优化仍应对完整 ``cosmos_model`` 进行。
    """

    def __init__(
        self,
        vae: nn.Module,
        network: nn.Module,
        sigma_data: float = 0.5,
        *,
        _as_view: bool = False,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)
        if _as_view:
            object.__setattr__(self, "encoder", vae)
            object.__setattr__(self, "network", network)
        else:
            self.add_module("encoder", vae)
            self.add_module("network", network)

    @classmethod
    def from_cosmos_diffusion_model(cls, model: nn.Module) -> Cosmos_Policy:
        """
        从已实例化的 Cosmos 扩散模型（如 ``CosmosPolicyVideo2WorldModel``）拆出 ``tokenizer`` 与 ``net``。

        使用轻量**视图**模式，不再把同一 ``nn.Module`` 挂到第二棵树底下。

        Raises:
            ValueError: 模型上不存在 ``tokenizer`` / ``net`` 属性时。
        """
        tokenizer = getattr(model, "tokenizer", None)
        net = getattr(model, "net", None)
        if tokenizer is None or net is None:
            raise ValueError(
                "Cosmos_Policy.from_cosmos_diffusion_model: model must have ``tokenizer`` (VAE) and "
                "``net`` (DiT), same as ``DiffusionModel`` in cosmos_policy."
            )
        sigma_data = float(
            getattr(model, "sigma_data", getattr(getattr(model, "config", None), "sigma_data", 0.5))
        )
        return cls(vae=tokenizer, network=net, sigma_data=sigma_data, _as_view=True)

    @torch.no_grad()
    def encode(self, state: Tensor) -> Tensor:
        """像素视频/图像 → 潜变量（与 ``DiffusionModel.encode`` 一致：乘 ``sigma_data``）。"""
        return self.encoder.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """潜变量 → 像素（与 ``DiffusionModel.decode`` 一致：先除 ``sigma_data`` 再 VAE 解码）。"""
        return self.encoder.decode(latent / self.sigma_data)

    def forward(
        self,
        xt_B_C_T_H_W: Tensor,
        timesteps_B_T: Tensor,
        **condition_kwargs: Any,
    ) -> Tensor:
        """
        单步 DiT 前向，签名与 ``DiffusionModel.denoise`` 内部 ``self.net(...)`` 调用对齐
        （不含 EDM 的 ``c_in`` / ``c_skip`` / ``c_out``；若需要完整去噪一步请直接用 ``cosmos_model.denoise``）。
        """
        vae_enc = self.vae.encode(xt_B_C_T_H_W)
        return self.network(vae_enc, timesteps_B_T, **condition_kwargs)


