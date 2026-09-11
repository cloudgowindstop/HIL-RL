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
CosmosPolicy — 面向 LeRobot 框架的 NVIDIA Cosmos Predict2 视频扩散策略封装类

架构概述
--------------------
COSMOS 模型将机器人控制任务视为**视频生成问题**：

1. 由当前相机图像、用于未来观测/动作块/价值函数的空白占位帧，拼接成一段原始帧构成的「视频序列」；
2. 通过 WAN 2.1 VAE 编码器将该序列压缩为紧凑的隐空间张量，形状为 (B, C, T', H', W')；
3. 由一个大型视频扩散 Transformer 对隐空间向量进行去噪，模型条件依赖于：
   - T5 文本嵌入（任务描述指令）
   - 视频条件帧（当前观测数据）
4. 对去噪后的隐向量进行解码，并从指定的「动作槽位（action slot）」中提取出动作块。

训练流程
--------
COSMOS 模型的 `training_step` 方法内部已完整实现所有训练逻辑：
- 通过 `replace_latent_with_action_chunk` / `replace_latent_with_proprio` 方法，将真实动作、本体感知、未来状态注入隐空间；
- 采样噪声等级，并计算 EDM 损失。

本类的作用是：将 LeRobot 标准格式数据构建为 COSMOS 所需格式的 `data_batch`，并将训练逻辑委托给 `cosmos_model.training_step()` 执行。

推理流程
---------
1. 构建推理批次数据（当前帧 + 空白的未来帧/动作帧）；
2. 调用 `cosmos_model.generate_samples_from_batch()` 执行反向扩散采样；
3. 使用 `extract_action_chunk_from_latent_sequence()` 从去噪后的隐空间序列中提取动作块；
4. 对动作执行反归一化，并通过动作队列逐步输出可执行动作。
"""
from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.constants import ACTION
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
# from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel
# from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats
from lerobot.policies.cosmos.configuration_cosmos import CosmosConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Megatron / distributed initialisation helper

def _ensure_megatron_initialized() -> None:
    """
    Initialise Megatron-Core in trivial (1-GPU) parallel mode if it has not
    been done yet.  COSMOS internals depend on `megatron.core.parallel_state`
    even for single-GPU runs.
    初始化 Megatron-Core 并行环境（即使单 GPU 运行），因为 Cosmos 内部依赖megatron.core.parallel_state模块，即使单卡也需要初始化模型并行状态。
    """
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state  # type: ignore[import]

        # In single-process eval, torch.distributed is often not initialized.
        # Calling initialize_model_parallel() in that state asserts in megatron.
        if not dist.is_initialized():
            logger.info(
                "Skip megatron model-parallel init because torch.distributed "
                "is not initialized yet."
            )
            return

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
    except ImportError:
        logger.warning(
            "megatron-core is not importable.  COSMOS training / inference may "
            "fail if the model uses context-parallel or model-parallel features."
        )


# T5 embedding helpers

def _load_t5_cache(cache_path: str) -> dict:
    """Load a pre-computed T5 embedding cache (pickle dict: str → Tensor).
    加载预计算的 T5 文本嵌入缓存（Pickle 格式），避免重复计算耗时的 Flan-T5-XXL 嵌入
    """
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)
    return cache


def _compute_t5_embedding_online(
    text: str, max_length: int, embedding_dim: int, device: str
) -> Tensor:
    """
    Compute a single T5 (Flan-T5-XXL) text embedding on the fly.

    Returns
    -------
    Tensor of shape (max_length, embedding_dim) on `device`.
    在线计算单条文本的 Flan-T5-XXL 嵌入（当无缓存时使用），返回符合 Cosmos 要求维度的嵌入张量
    """
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

    # outputs.last_hidden_state: (1, seq_len, hidden_dim)
    emb = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

    # Pad or truncate to (max_length, embedding_dim)
    if emb.shape[0] < max_length:
        pad = torch.zeros(max_length - emb.shape[0], emb.shape[1], device=device)
        emb = torch.cat([emb, pad], dim=0)
    else:
        emb = emb[:max_length]

    return emb.float()


# Image-sequence construction utilities

def _resize_frame(
    frame: np.ndarray, target_size: int
) -> np.ndarray:
    """
    Resize a single HWC uint8 frame to (target_size × target_size).
    将单张 HWC 格式的 uint8 图像帧缩放到target_size × target_size（Cosmos 要求固定尺寸输入）
    """
    if frame.shape[0] == target_size and frame.shape[1] == target_size:
        return frame
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)


def _tensor_chw_to_hwc_uint8(t: Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) float tensor in [0, 1] to HWC uint8.
    将 LeRobot 的图像张量（CHW 格式、float 型、值域 [0,1]）转换为 OpenCV 兼容的 HWC 格式 uint8 数组
    """
    arr = (t.cpu().float().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return arr


def _build_video_sequence(
    config: CosmosPolicyConfig,
    current_primary: np.ndarray,
    current_wrist: np.ndarray | None,
    current_wrist2: np.ndarray | None,
    future_primary: np.ndarray | None,
    future_wrist: np.ndarray | None,
    future_wrist2: np.ndarray | None,
) -> np.ndarray:
    """
    Assemble the raw video tensor expected by the COSMOS tokenizer.

    The layout mirrors the ALOHA dataset's `__getitem__` construction.
    Every logical "slot" is repeated `num_duplicates_per_image` times so that
    the WAN 2.1 temporal tokenizer maps each slot to exactly one latent frame.

    Parameters
    ----------
    current_*  : HWC uint8 frames for the current timestep.
    future_*   : HWC uint8 frames for the future timestep (blank if None).

    Returns
    -------
    np.ndarray of shape (T_raw, H, W, C) uint8
    按照 Cosmos 的 slot 布局规则，组装原始视频序列（T_raw, H, W, C），是 LeRobot 图像到 Cosmos 视频张量的核心转换步骤。
    """
    H = W = config.image_size
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    nd = config.num_duplicates_per_image

    def _slot(frame: np.ndarray | None) -> np.ndarray:
        f = frame if frame is not None else blank
        f = _resize_frame(f, config.image_size)
        return np.tile(f[np.newaxis], (nd, 1, 1, 1))  # (nd, H, W, C)

    frames = []
    for slot_name in config._slot_order:
        if slot_name == "blank":
            frames.append(_slot(None))
        elif slot_name == "current_proprio":
            frames.append(_slot(None))          # proprio injected into latent
        elif slot_name == "current_wrist":
            frames.append(_slot(current_wrist))
        elif slot_name == "current_wrist2":
            frames.append(_slot(current_wrist2))
        elif slot_name == "current_primary":
            frames.append(_slot(current_primary))
        elif slot_name == "action":
            frames.append(_slot(None))          # action injected into latent
        elif slot_name == "future_proprio":
            frames.append(_slot(None))          # future proprio injected
        elif slot_name == "future_wrist":
            frames.append(_slot(future_wrist))
        elif slot_name == "future_wrist2":
            frames.append(_slot(future_wrist2))
        elif slot_name == "future_primary":
            frames.append(_slot(future_primary))
        elif slot_name == "value":
            frames.append(_slot(None))          # value injected into latent
        else:
            raise ValueError(f"Unknown slot name: {slot_name!r}")

    return np.concatenate(frames, axis=0)  # (T_raw, H, W, C)


# LeRobot → COSMOS batch conversion

def _make_latent_idx_tensor(
    value: int, batch_size: int, device: torch.device
) -> Tensor:
    """
    创建批量的潜在帧索引张量（Cosmos 需要指定每个 slot 对应的潜在帧位置）
    """
    return torch.tensor([value] * batch_size, dtype=torch.int64, device=device)


def _lerobot_batch_to_cosmos(
    config: CosmosConfig,
    batch: dict[str, Tensor],
    t5_embedding: Tensor,
    device: torch.device,
    *,
    is_inference: bool = False,
) -> dict:
    """
    Convert a LeRobot-format batch to the dict expected by COSMOS.

    LeRobot batch keys used
    -----------------------
    observation.<primary_image_key>       (B, [T,] C, H, W)  float [0,1]
    observation.<wrist_image_key>         optional
    observation.<wrist_image2_key>        optional
    observation.state                     (B, [T,] state_dim)
    action                                (B, T_action, action_dim)  — training only

    COSMOS output keys
    ------------------
    video                  (B, C, T_raw, H, W)  uint8
    t5_text_embeddings     (B, max_length, emb_dim)
    t5_text_mask           (B, max_length)
    fps                    scalar int
    padding_mask           (B, 1, H, W)
    actions                (B, chunk_size, action_dim)  — training
    proprio                (B, proprio_dim)
    future_proprio         (B, proprio_dim)
    rollout_data_mask      (B,)  all zeros (demo data)
    world_model_sample_mask(B,)  all zeros
    value_function_sample_mask (B,) all zeros
    value_function_return  (B,)  all zeros
    *_latent_idx           (B,)  scalar indices
    将 LeRobot 标准批量数据转换为 Cosmos 模型所需的输入字典，是数据格式转换的核心函数
    """
    B = next(iter(batch.values())).shape[0]
    lidx = config.latent_indices

    # Extract image frames from LeRobot batch
    def _get_image(key: str) -> np.ndarray | None:
        """
        Return the LAST obs-step frame as HWC uint8 numpy, or None.
        从 LeRobot batch 中提取指定键的图像张量（形状(B, [T,] C, H, W)），取最后一个时间步，转为 HWC uint8 numpy 数组
        """
        if key not in batch:
            return None
        t = batch[key]           # (B, [T,] C, H, W)
        if t.ndim == 5:
            t = t[:, -1]         # take last timestep → (B, C, H, W)
        # t: (B, C, H, W) float [0,1]  —  convert first sample; loop for each
        return t                 # keep as tensor, convert per-sample below

    prim_t = _get_image(config.primary_image_key)         # (B, C, H, W)
    wrist_t = _get_image(config.wrist_image_key)
    wrist2_t = _get_image(config.wrist_image2_key)

    # For future frames: use second-to-last obs step if available, else blank
    def _get_future_image(key: str) -> Tensor | None:
        # 提取未来帧图像，如果当前帧有多个时间步，则取倒数第二个时间步作为未来帧
        if key not in batch:
            return None
        t = batch[key]
        if t.ndim == 5 and t.shape[1] >= 2:
            return t[:, -2]      # second-to-last → acts as "future" in action delta
        return None              # will use blank

    fut_prim_t = _get_future_image(config.primary_image_key)
    fut_wrist_t = _get_future_image(config.wrist_image_key)
    fut_wrist2_t = _get_future_image(config.wrist_image2_key)

    # Build per-sample video sequences and stack
    # 构建每个样本的视频序列，并堆叠成一个张量（B, C, T, H, W）
    H = W = config.image_size

    def _t_to_hwc(t_bchw: Tensor | None, idx: int) -> np.ndarray | None:
        if t_bchw is None:
            return None
        return _tensor_chw_to_hwc_uint8(t_bchw[idx])

    video_list = []
    for b in range(B):
        seq = _build_video_sequence(
            config,
            current_primary=_t_to_hwc(prim_t, b),
            current_wrist=_t_to_hwc(wrist_t, b),
            current_wrist2=_t_to_hwc(wrist2_t, b),
            future_primary=_t_to_hwc(fut_prim_t, b),
            future_wrist=_t_to_hwc(fut_wrist_t, b),
            future_wrist2=_t_to_hwc(fut_wrist2_t, b),
        )
        # seq: (T_raw, H, W, C) uint8 → (C, T_raw, H, W)
        seq_t = torch.from_numpy(seq).permute(3, 0, 1, 2)  # (C, T, H, W)
        video_list.append(seq_t)

    video = torch.stack(video_list, dim=0).to(device)  # (B, C, T, H, W) uint8

    # Proprio                                                               #
    state_key = "observation.state"
    if config.use_proprio and state_key in batch:
        s = batch[state_key]
        if s.ndim == 3: # 取最后一个时间步的state作为proprio？？？
            s = s[:, -1]          # (B, state_dim)
        proprio = s.to(device).float()
        future_proprio = proprio.clone()   # fallback: same as current
    else:
        proprio_dim = (
            config.input_features.get(state_key).shape[0]
            if state_key in config.input_features
            else 1
        )
        proprio = torch.zeros(B, proprio_dim, device=device)
        future_proprio = torch.zeros(B, proprio_dim, device=device)

    # Action chunk  (training only)
    # 提取动作序列，如果当前动作序列长度小于 chunk_size，则用最后一个动作填充
    if not is_inference and ACTION in batch:
        act = batch[ACTION]      # (B, T_action, action_dim)
        if act.shape[1] >= config.chunk_size:
            action_chunk = act[:, : config.chunk_size].to(device).float()
        else:
            # Pad with last action
            pad = act[:, -1:].expand(-1, config.chunk_size - act.shape[1], -1)
            action_chunk = torch.cat([act, pad], dim=1).to(device).float()
    else:
        action_chunk = torch.zeros(B, config.chunk_size, config.action_dim, device=device)

    # T5 embeddings
    # t5_embedding将单样本T5嵌入扩展为批量维度: (max_length, emb_dim)  →  broadcast to batch（B，max_length, emd_dim）
    t5_emb = t5_embedding.unsqueeze(0).expand(B, -1, -1).to(device)   # (B, L, D)
    t5_mask = torch.ones(B, config.t5_max_length, dtype=torch.int64, device=device) # 构建T5掩码（全1向量，形状（B, t5_max_length））

    # Latent indices
    def _idx(name: str) -> Tensor:
        # 构建潜在索引，根据config的latent_indices生成各slot的索引张量
        v = lidx.get(name, -1)
        return _make_latent_idx_tensor(v, B, device) # 构建潜在索引张量（形状（B,），值为v）

    # 组装cosmos模型所需的输入字典
    cosmos_batch = {
        "video": video,
        "t5_text_embeddings": t5_emb.to(torch.bfloat16),
        "t5_text_mask": t5_mask,
        "fps": 16,
        "padding_mask": torch.zeros(B, 1, H, W, device=device),
        "image_size": config.image_size * torch.ones(4, device=device),
        # Action / proprio (training targets / conditioning)
        "actions": action_chunk,
        "proprio": proprio,
        "future_proprio": future_proprio,
        # Masking (demo data: all zeros → no rollout / value function)
        "rollout_data_mask": torch.zeros(B, dtype=torch.long, device=device),
        "world_model_sample_mask": torch.zeros(B, dtype=torch.long, device=device),
        "value_function_sample_mask": torch.zeros(B, dtype=torch.long, device=device),
        "value_function_return": torch.zeros(B, device=device),
        # Latent frame indices for each slot
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
    return cosmos_batch


# Main policy class

class CosmosPolicy(
    PreTrainedPolicy
):
    """
    LeRobot policy wrapper around the NVIDIA Cosmos Predict2 video-diffusion model.

    Initialization
    --------------
    The COSMOS model is loaded via Imaginaire's lazy-config system from the file
    pointed to by ``config.cosmos_config_path``.  Pre-trained weights are loaded
    from ``config.cosmos_checkpoint_path`` if provided.

    Requires
    --------
    * ``cosmos-predict2`` (NVIDIA) must be installed.
    * ``megatron-core`` must be importable (single-GPU trivial init is handled
      automatically).
    * A pre-computed T5-embedding cache **or** a network-accessible Flan-T5-XXL
      model for online embedding computation.
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

        # # Convert any list values in dataset_stats to torch.Tensor so that
        # # Normalize / Unnormalize modules can consume them.
        # raw_stats = self.config.dataset_stats
        # if raw_stats is not None:
        #     dataset_stats = {
        #         feat_key: {
        #             stat_key: torch.tensor(v, dtype=torch.float32)
        #             if isinstance(v, (list, tuple))
        #             else v
        #             for stat_key, v in stat_dict.items()
        #         }
        #         for feat_key, stat_dict in raw_stats.items()
        #     }
        # else:
        #     dataset_stats = None

        # # Determine action dimension and initialize all components
        # continuous_action_dim = config.output_features["action"].shape[0]
        # self.continuous_action_dim = continuous_action_dim

        # # Normalisation layers (for state / action)     
        # # NOTE: COSMOS handles its own image normalisation internally.  
        # # We still normalise state / action so the model receives values    
        # # in a predictable range.                                           
        # # 对输入的observation.state归一化
        # self.normalize_inputs = Normalize(
        #     config.input_features, config.normalization_mapping, dataset_stats
        # )
        # # 对输出的action归一化
        # self.normalize_targets = Normalize(
        #     config.output_features, config.normalization_mapping, dataset_stats
        # )
        # # 对输出的action反归一化
        # self.unnormalize_outputs = Unnormalize(
        #     config.output_features, config.normalization_mapping, dataset_stats
        # )

        # # Megatron (needed even for single GPU)
        # _ensure_megatron_initialized()

        # # COSMOS model
        # # 构建cosmos模型
        # self.cosmos_model = self._build_cosmos_model() 
        # self._training_iteration = 0  # 训练迭代计数器

        # # T5 embeddings
        # # 加载T5嵌入缓存，如果缓存存在，则加载缓存，否则计算T5嵌入
        # self._t5_cache: dict | None = None
        # if config.t5_text_embeddings_path: 
        #     self._t5_cache = _load_t5_cache(config.t5_text_embeddings_path)
        #     logger.info(f"Loaded T5 cache with {len(self._t5_cache)} entries.")
        # self._t5_embedding: Tensor | None = None   # lazy-computed

        # # Inference action queue
        # self._queues: dict[str, deque] = {}
        # self.reset()

    # Model construction

    def _build_cosmos_model(self) -> nn.Module | None:
        """
        Returns None as a placeholder.
        The real model is loaded externally via get_model(cosmos_cfg) and assigned
        to policy.cosmos_model after make_policy() completes.
        """
        return None

    def set_cosmos_model(self, model: nn.Module) -> None:
        """Inject the cosmos model loaded externally (e.g. via cosmos_utils.get_model)."""
        self.cosmos_model = model

    def _load_cosmos_checkpoint(self, model: nn.Module, checkpoint_path: str) -> None:
        """
        Load a COSMOS checkpoint into `model`.

        Tries the Imaginaire checkpoint API first (``model.restore_checkpoint``);
        falls back to a plain PyTorch ``load_state_dict``.
        """
        # 加载 Cosmos 预训练权重（兼容 Imaginaire 和 PyTorch 两种 checkpoint 格式）
        ckpt_path = Path(checkpoint_path)
        if hasattr(model, "restore_checkpoint"):
            try:
                model.restore_checkpoint(str(ckpt_path))
                logger.info(f"Loaded COSMOS checkpoint via restore_checkpoint: {ckpt_path}")
                return
            except Exception as e:
                logger.warning(f"restore_checkpoint failed ({e}), falling back to torch.load.")

        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = state.get("model", state.get("state_dict", state))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing[:10]}…")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected[:10]}…")
        logger.info(f"Loaded COSMOS checkpoint via torch.load: {ckpt_path}")

    # T5 embedding helpers

    def _get_t5_embedding(self) -> Tensor:
        """
        Return the T5 embedding for the configured task description.

        Caches the result after the first call.
        """
        # 获取配置的任务描述对应的 T5 嵌入（优先用缓存，无缓存则在线计算），并缓存结果避免重复计算
        if self._t5_embedding is not None:
            return self._t5_embedding

        device = self.config.device or "cpu"
        task = self.config.task_description

        if self._t5_cache is not None and task in self._t5_cache:
            emb = self._t5_cache[task]
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            self._t5_embedding = emb.float()
            return self._t5_embedding

        # Compute online
        logger.info(f"Computing T5 embedding online for: '{task}'")
        self._t5_embedding = _compute_t5_embedding_online(
            task,
            self.config.t5_max_length,
            self.config.t5_embedding_dim,
            device,
        )
        return self._t5_embedding

    # LeRobot PreTrainedPolicy interface

    def get_optim_params(self) -> dict:
        # 实现 LeRobot PreTrainedPolicy接口，返回 Cosmos 模型的可优化参数
        return self.cosmos_model.parameters()

    def reset(self) -> None:
        """Clear action queue.  Call on env.reset()."""
        # 重置推理时的动作队列和观测历史队列（环境重置时调用）
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # Also keep observation history queues
        for key in self.config.input_features:
            if key not in self._queues:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """
        Training forward pass.

        Converts the LeRobot batch to COSMOS format, runs the COSMOS
        ``training_step``, and returns the scalar loss.
        """
        # 实现 LeRobot 训练前向传播接口，将 LeRobot batch 转为 Cosmos 格式，调用 Cosmos 的training_step计算损失。
        # Normalise state / action in LeRobot space
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        device = next(self.cosmos_model.parameters()).device
        t5_emb = self._get_t5_embedding().to(device)

        cosmos_batch = _lerobot_batch_to_cosmos(
            self.config,
            batch,
            t5_emb,
            device,
            is_inference=False,
        )
        # todo : training_step 计算损失???
        output_dict, loss = self.cosmos_model.training_step(
            cosmos_batch, self._training_iteration
        )
        self._training_iteration += 1

        # Collect scalar log values from output_dict
        # 从output_dict中收集标量日志值（损失值、学习率等）
        log_dict = {
            k: v.item() if isinstance(v, Tensor) and v.numel() == 1 else v
            for k, v in output_dict.items()
            if isinstance(v, (Tensor, float, int)) and (
                not isinstance(v, Tensor) or v.numel() == 1
            )
        }

        return loss, log_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Run one complete reverse-diffusion pass and return the full action chunk.

        The batch should already contain the stacked observation history
        (queued in ``select_action``).

        Returns
        -------
        Tensor of shape (B, chunk_size, action_dim)
        """
        # 推理时执行完整的反向扩散流程，生成批量的动作块（chunk_size, action_dim）
        from cosmos_policy.experiments.robot.cosmos_utils import (  # type: ignore[import]
            extract_action_chunk_from_latent_sequence,
        )

        device = next(self.cosmos_model.parameters()).device
        t5_emb = self._get_t5_embedding().to(device)

        cosmos_batch = _lerobot_batch_to_cosmos(
            self.config,
            batch,
            t5_emb,
            device,
            is_inference=True,
        )

        B = cosmos_batch["video"].shape[0]
        action_latent_idx = self.config.latent_indices["action"]

        # Run reverse diffusion
        generated_latent = self.cosmos_model.generate_samples_from_batch(
            cosmos_batch,
            n_sample=B,
            num_steps=self.config.num_denoising_steps,
            seed=0,
            is_negative_prompt=False,
        )  # (B, C', T', H', W')

        # Extract the predicted action chunk from the designated latent slot
        action_indices = torch.full(
            (B,), action_latent_idx, dtype=torch.int64, device=generated_latent.device
        )
        actions = extract_action_chunk_from_latent_sequence(
            generated_latent,
            action_shape=(self.config.chunk_size, self.config.action_dim),
            action_indices=action_indices,
        ).float()  # (B, chunk_size, action_dim)

        # Un-normalise back to robot-space
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    # @torch.no_grad()
    # def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    #     """
    #     Return a single action to execute in the environment.

    #     Manages an internal cache:
    #     - ``n_obs_steps`` recent observation frames are queued.
    #     - ``n_action_steps`` actions are pre-computed each time the queue
    #       empties, by calling ``predict_action_chunk``.
    #     - Actions are served one step at a time (FIFO).
    #     """
    #     # 返回单步执行的动作（从预生成的动作块中逐步弹出单步动作）
    #     self.eval()

    #     # Remove action from batch if present (offline eval scenario)
    #     if ACTION in batch:
    #         batch = {k: v for k, v in batch.items() if k != ACTION}

    #     batch = self.normalize_inputs(batch)

    #     # Stack observation history
    #     self._queues = populate_queues(self._queues, batch)

    #     if len(self._queues[ACTION]) == 0:
    #         # Build a batch with stacked obs history
    #         obs_batch = {
    #             k: torch.stack(list(self._queues[k]), dim=1)
    #             for k in batch
    #             if k in self._queues
    #         }
    #         actions = self.predict_action_chunk(obs_batch)   # (B, chunk, action_dim)
    #         # Enqueue all n_action_steps actions
    #         self._queues[ACTION].extend(
    #             actions[:, : self.config.n_action_steps].transpose(0, 1)
    #         )  # deque of tensors (B, action_dim)

    #     return self._queues[ACTION].popleft()  # (B, action_dim)
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Return a single action to execute in the environment.

        Manages an internal cache:
        - ``n_obs_steps`` recent observation frames are queued.
        - ``n_action_steps`` actions are pre-computed each time the queue
          empties, by calling ``predict_action_chunk``.
        - Actions are served one step at a time (FIFO).
        """
        # 返回单步执行的动作（从预生成的动作块中逐步弹出单步动作）
        self.eval()

        # Remove action from batch if present (offline eval scenario)
        if ACTION in batch:
            batch = {k: v for k, v in batch.items() if k != ACTION}

        batch = self.normalize_inputs(batch)

        # Stack observation history
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            # Build a batch with stacked obs history
            obs_batch = {
                k: torch.stack(list(self._queues[k]), dim=1)
                for k in batch
                if k in self._queues
            }
            actions = self.predict_action_chunk(obs_batch)   # (B, chunk, action_dim)
            # Enqueue all n_action_steps actions
            self._queues[ACTION].extend(
                actions[:, : self.config.n_action_steps].transpose(0, 1)
            )  # deque of tensors (B, action_dim)

        return self._queues[ACTION].popleft()  # (B, action_dim)
