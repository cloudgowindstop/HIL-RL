#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

import copy
import math
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal
import attrs
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution
import os
import cv2
import time
from einops import rearrange
from lerobot.policies.normalize import NormalizeBuffer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.cosmos.configuration_cosmos import CosmosConfig, is_image_feature
from lerobot.policies.utils import get_device_from_parameters
from typing import Optional, Tuple, Dict, Any
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel, CosmosPolicyVideo2WorldConfig
from cosmos_policy._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.predict2.conditioner import BooleanFlag, DataType, ReMapkey, TextAttr
from cosmos_policy._src.predict2.configs.video2world.defaults.net import COSMOS_V1_2B_NET_MININET
from cosmos_policy._src.predict2.models.text2world_model import EMAConfig
from cosmos_policy._src.predict2.models.video2world_model import ConditioningStrategy, HighSigmaStrategy
from cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldCondition, Video2WorldConditioner
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.tokenizers_cosmos.wan2pt1 import Wan2pt1VAEInterface
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.policies.sac.modeling_sac import SACObservationEncoder, CriticHead, CriticEnsemble, DiscreteCritic, MLP
from cosmos_policy.models.policy_text2world_model import (
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_policy.experiments.robot.cosmos_utils import (  # type: ignore[import]
    extract_action_chunk_from_latent_sequence,
    rescale_proprio,
    unnormalize_actions,
)
from lerobot.utils.transition import move_transition_to_device
from cosmos_policy.datasets.dataset_common import (
    compute_monte_carlo_returns,
    get_action_chunk_with_padding,
)
from cosmos_policy.experiments.robot.cosmos_utils import load_dataset_stats
from cosmos_policy._src.predict2.conditioner import DataType
logger = logging.getLogger(__name__)


DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4
COSMOS_IMAGE_SIZE = 224  # Standard image size expected by Cosmos policies
CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
}

# Latent-frame layout (aligned with ``rl_envs/wrappers.py``).
LATENT_INDEX_KEYS = (
    "current_proprio_latent_idx",
    "current_wrist_image_latent_idx",
    "current_image_latent_idx",
    "action_latent_idx",
    "future_proprio_latent_idx",
    "future_wrist_image_latent_idx",
    "future_image_latent_idx",
    "value_latent_idx",
)
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
SPECIFIC_LATENT_INDICES_START: dict[str, int] = {
    "current_proprio_latent_idx": 1,
    "current_wrist_image_latent_idx": 5,
    "current_image_latent_idx": 9,
    "action_latent_idx": 13,
    "future_proprio_latent_idx": 17,
    "future_wrist_image_latent_idx": 21,
    "future_image_latent_idx": 25,
    "value_latent_idx": 29,
}

class _CompositeCosmosConfig:
    """Expose both policy config and world-model config from `self.config`."""

    def __init__(self, policy_config: CosmosConfig, world_config: CosmosPolicyVideo2WorldConfig) -> None:
        object.__setattr__(self, "_policy_config", policy_config)
        object.__setattr__(self, "_world_config", world_config)

    def __getattr__(self, name: str):
        policy_cfg = object.__getattribute__(self, "_policy_config")
        world_cfg = object.__getattribute__(self, "_world_config")
        if hasattr(policy_cfg, name):
            return getattr(policy_cfg, name)
        if hasattr(world_cfg, name):
            return getattr(world_cfg, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        policy_cfg = object.__getattribute__(self, "_policy_config")
        world_cfg = object.__getattribute__(self, "_world_config")
        if hasattr(policy_cfg, name):
            setattr(policy_cfg, name, value)
            return
        if hasattr(world_cfg, name):
            setattr(world_cfg, name, value)
            return
        object.__setattr__(self, name, value)

class CosmosPolicy(
    PreTrainedPolicy,
    CosmosPolicyVideo2WorldModel
):
    config_class = CosmosConfig
    name = "cosmos"

    def __init__(
        self,
        config: CosmosConfig | None = None,
        cosmos_cfg: CosmosPolicyVideo2WorldConfig  | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        if config is None:
            raise ValueError("CosmosPolicy requires a valid CosmosConfig.")
        if cosmos_cfg is None:
            cosmos_cfg = config.load_world_config()

        CosmosPolicyVideo2WorldModel.__init__(self, cosmos_cfg)
       
        config.validate_features()
        self.policy_config = config
        self.cosmos_cfg = cosmos_cfg
        self.config = _CompositeCosmosConfig(self.policy_config, self.cosmos_cfg)

        self.dataset_stats = self.config.dataset_stats

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features["action"].shape[0]
        self.continuous_action_dim = continuous_action_dim
        # self._init_normalization(dataset_stats)  # 不在这里初始化了，但是需要初始化critic和encoder，初始化optimizer时需要用到
        # # 初始化观测编码器（Actor与Critic可共享或独立）
        # self._init_encoders()  
        # # # 初始化Critic网络（连续动作+可选离散动作）
        # self._init_critics(continuous_action_dim)
        # # 初始化Actor网络（输出连续动作分布）
        self._init_actor(continuous_action_dim)

        # DiT 在 build_net 中为 fp32；denoise() 会把输入 cast 到 config.precision（bf16）。
        # 官方在 on_train_start 里把 net 对齐到同一 dtype，推理/在线采集也必须调用。
        self.to("cuda")  # type: ignore
        # 模型执行训练开始前的准备工作（如EMA初始化、梯度检查点设置等）
        #  _CompositeCosmosConfig 把 LeRobot 的 use_torch_compile=True 传给了 DiT，导致 torch.compile 无法追踪 Transformer Engine 的 C++ 算子
        self.on_train_start()
        self.t5_text_embeddings = None


    def get_optim_params(self) -> dict:
        """获取各模块的可优化参数，用于构建优化器"""
        optim_params = {
            # "actor": [
                # p
                # for n, p in self.actor.named_parameters()
                # 若共享编码器，Actor不优化编码器参数（避免梯度冲突）
                # if not n.startswith("encoder") or not self.shared_encoder
            # ],
            "actor": self.net.parameters(),
            
            # "critic": self.critic_ensemble.parameters(),
            # "temperature": self.log_alpha,
        }
        # 若有离散动作，添加离散Critic参数
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        return optim_params
    
    def make_optimizers_and_scheduler(self):
        """Make optimizers and scheduler"""
        opt_cfg, sch_cfg = self.config.load_optimizer_scheduler_configs()
        optimizers = {}
        optimizers["actor"], scheduler = self.init_optimizer_scheduler(opt_cfg, sch_cfg)
        return optimizers, scheduler

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    def _move_cosmos_batch_to_device(self, batch: dict) -> dict:
        """Move cosmos inference batch tensors to the same device as the policy."""
        device = get_device_from_parameters(self)
        print("device: ", device)
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], cosmos_cfg: Any, dataset_stats_cosmos: dict) -> Tensor:
        """Select action for inference/evaluation"""
        """
        推理/评估阶段选择动作
        Args:
            batch: 观测字典（含图像（left、wrist）、状态）
        Returns:
            最终动作张量（连续动作 + 可选离散动作拼接）
        """
        # observations_features = None
        
        # # 若共享编码器且含图像，缓存图像特征（避免重复编码，提升速度）
        # if self.shared_encoder and self.actor.encoder.has_images:
        #     # Cache and normalize image features

        #     observations_features = self.actor.encoder.get_cached_image_features(batch, normalize=True)
        # actions, _, _ = self.actor(batch, observations_features)


        # # 若有离散动作，离散Critic输出各动作价值，选价值最大的动作
        # if self.config.num_discrete_actions is not None:
        #     discrete_action_value = self.discrete_critic(batch, observations_features)
        #     discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)

        #     actions = torch.cat([actions, discrete_action], dim=-1)
        # return actions, {}

        # # 处理获得的batch数据:batch_size=1
        # batch = self.prepare_infer_data(batch, command=command, cosmos_cfg=self.cosmos_eval_cfg)
        # raw_state, latent_state_infer = self.change_to_latent(data_batch=batch)

        # condition, uncondition, num_conditional_frames = self._init_conditioner(batch, latent_state_infer)
        # condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        # if uncondition is not None:
        #     uncondition = uncondition.edit_for_inference(
        #         is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        #     )
        
        # # 本体感知注入到条件掩码中
        # proprio = batch["proprio"]
        # current_proprio_latent_idx = batch["current_proprio_latent_idx"]
        # batch_indices = torch.arange(latent_state_infer.shape[0], device=proprio.device)
        # # 推理的时候将proprio注入到条件掩码中，训练的时候已经在set_video_condition中注入到条件掩码中
        # condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, current_proprio_latent_idx, :, :] = 1
        # condition.gt_frames = replace_latent_with_proprio(
        #     condition.gt_frames, 
        #     proprio, 
        #     proprio_indices=current_proprio_latent_idx
        # )
        batch = self._move_cosmos_batch_to_device(batch)
        batch_size = 1
        num_denoising_steps_action = 5
        if cosmos_cfg.randomize_seed:  # false
            import secrets  
            seed = secrets.randbits(32) % 256
        else:
            seed = 1

        # add normalize proprio for inference
        batch["proprio"] = self._rescale_tensor_or_array(
            batch["proprio"],
            rescale_proprio,
            dataset_stats_cosmos,
        )
        generated_latent_with_action, orig_clean_latent_frames = self.generate_samples_from_batch(
            data_batch=batch,
            n_sample=batch_size,  # Generate samples
            num_steps=num_denoising_steps_action,
            seed=seed,
            is_negative_prompt=False,  # Negative prompt is for CFG
            use_variance_scale=cosmos_cfg.use_variance_scale,  # Whether to vary the magnitude of the initial noise - increases diversity slightly in generations
            return_orig_clean_latent_frames=True,  # Return the original (pre-injection) latent frames - needed for future image visualizations
        )  # (B, C'=16, T', H'=28, W'=28)
        action_latent_idx = LATENT_INDICES["action_latent_idx"]
        # print("action_latent_idx: ", action_latent_idx)
        # 模仿学习包含夹爪维度，所以需要+1
        action_dim = self.continuous_action_dim + 1

        action_indices = torch.full(
            (batch_size,),
            action_latent_idx,
            dtype=torch.int64,
            device=generated_latent_with_action.device,
        )
        actions = (
            extract_action_chunk_from_latent_sequence(
                generated_latent_with_action, action_shape=(cosmos_cfg.chunk_size, action_dim), action_indices=action_indices
            )
            .to(torch.float32)
            .cpu()
            .numpy()
        )  # (batch_size, chunk_size, action_dim)
        if cosmos_cfg.unnormalize_actions:
            # print('actions.shape: ', actions.shape)
            actions = unnormalize_actions(actions, dataset_stats_cosmos)
            # print('predict actions: ', actions[0][0])
        # exit(0)
        actions = actions[0]    # (chunk_size, action_dim)
        actions = [actions[i] for i in range(len(actions))]
        # 是否需要在推理的时候预测未来（训练的时候用的），应该是不需要
        # generate_future_state_and_value_in_parallel=not (
        #     cosmos_cfg.ar_future_prediction or cosmos_cfg.ar_value_prediction or cosmos_cfg.ar_qvalue_prediction
        # )
        # if generate_future_state_and_value_in_parallel:
        #     future_image_predictions = get_future_images_from_generated_samples(
        #         model,
        #         generated_latent_with_action.clone(),
        #         cfg,
        #         orig_clean_latent_frames,
        #         INDICES_TO_REPLACE,
        #         future_wrist_image_latent_idx if cfg.use_wrist_image else -1,
        #         future_wrist_image2_latent_idx if cfg.use_wrist_image and cfg.num_wrist_images == 2 else -1,
        #     )
        return actions, {}

    def rescale_action(self, action, dataset_stats, non_negative_only=False, scale_multiplier=1.0):
        """
        Rescale (normalize) proprio to the range [-1,+1] or [0,+1], with optional scaling by scale_multiplier.

        Args:
            proprio (np.ndarray): Proprio to be rescaled
            dataset_stats (dict): Dataset statistics needed for rescaling formula
            non_negative_only (bool): Whether to use [0,+1] range (True) or [-1,+1] range (False)
            scale_multiplier (float): Multiplier to adjust final scale

        Returns:
            np.ndarray: Rescaled proprio
        """
        arr = action
        curr_min = dataset_stats["actions_min"]
        curr_max = dataset_stats["actions_max"]
        # First, scale to [-1,+1] or [0,+1]:
        # - For [-1,+1]: x_new = 2 * ((x - curr_min) / (curr_max - curr_min)) - 1
        # - For [0,+1]: x_new = (x - curr_min) / (curr_max - curr_min)
        if not non_negative_only:  # [-1,+1]
            rescaled_arr = 2 * ((arr - curr_min) / (curr_max - curr_min)) - 1
        else:  # [0,+1]
            rescaled_arr = (arr - curr_min) / (curr_max - curr_min)
        # Scale to [-scale_multiplier,+scale_multiplier] or [0,+scale_multiplier]
        rescaled_arr = scale_multiplier * rescaled_arr
        action = rescaled_arr
        return action

    def encode_episode_state(
        self,
        transition_list: list,
        encode_batch_size: int,
        device: str,
        cosmos_cfg: Any,
        batch_size: int,
        dataset_stats_cosmos: dict = None,
        world_size: int = 1,
        encode_device_ids: list[int] | None = None,
    ) -> list:
        """Encode an episode, inject ground-truth signals, and write latents back into each transition.

        Args:
            world_size: Number of GPUs used to shard VAE encode along the episode dim
                (``cuda:0 .. cuda:world_size-1``). Ignored when ``encode_device_ids`` is set.
            encode_device_ids: Explicit GPU indices for encode (process-local, after
                ``CUDA_VISIBLE_DEVICES`` remapping), e.g. ``[1, 2, 3]``.
        """
        transition_video_list = []
        transition_action_chunk_list = []
        transition_proprio_list = []
        transition_future_proprio_list = []
        transition_value_function_return_list = []
        episode_length = len(transition_list)
        chunk_size = cosmos_cfg.chunk_size
        print("--------------------------------- episode_length:", episode_length)
        print("transition_list[-1][done]:", transition_list[-1]["done"])

        # normalize proprio and actions(non_negative_only=False,[-1,1])
        transition_list = self.normalizer(transition_list, dataset_stats_cosmos)

        actions = np.stack(
            [
                t["action"].detach().cpu().numpy()
                if isinstance(t["action"], torch.Tensor)
                else np.asarray(t["action"], dtype=np.float32)
                for t in transition_list
            ],
            axis=0,
        )
        print("actions.shape: ", actions.shape)
        

        terminal_reward = 1.0 if transition_list[-1]["done"] else 0.0
        # terminal_reward = 0.0
        gamma = getattr(cosmos_cfg, "gamma", 0.99)
        returns = compute_monte_carlo_returns(
            episode_length, terminal_reward=terminal_reward, gamma=gamma
        )
        # print("returns: ", returns)
        # print("returns: ", returns.dtype)    # float32
        for i, transition in enumerate(transition_list):
            transition = move_transition_to_device(transition=transition, device=device)
            future = min(i + chunk_size, episode_length - 1)
            future_transition = move_transition_to_device(transition=transition_list[future], device=device)
            # Inference-mode tensors (e.g. from env rollout) 不允许在模式外部对这些推理张量进行原地修改
            if isinstance(transition["state"].get("video"), torch.Tensor):
                transition["state"]["video"] = transition["state"]["video"].clone()
            # # fix4:device指定
            video_tensor = future_transition["state"]["video"]
            # # 获取所有元素的最小值和最大值
            # min_val = video_tensor.min()
            # max_val = video_tensor.max()
            # print("min_val: ", min_val, "max_val: ", max_val)   # [0,255]
            # print("type(video_tensor.min().item()): ", type(video_tensor.min().item()))  # <class 'int'>

            # if video_tensor.device != transition["state"]["video"].device:
            #     video_tensor = video_tensor.to(transition["state"]["video"].device)

            # copy future wrist
            # debug for latent
            future_wrist_start_idx = SPECIFIC_LATENT_INDICES_START["future_wrist_image_latent_idx"]
            wrist_start_idx = SPECIFIC_LATENT_INDICES_START["current_wrist_image_latent_idx"]
            wrist_duplicated_frames = video_tensor[0, :, wrist_start_idx:wrist_start_idx + 4, :, :].clone()
            transition["state"]["video"][0, :, future_wrist_start_idx:future_wrist_start_idx + 4, :, :] = (
                wrist_duplicated_frames
            )

            # copy future primary
            future_primary_start_idx = SPECIFIC_LATENT_INDICES_START["future_image_latent_idx"]
            primary_start_idx = SPECIFIC_LATENT_INDICES_START["current_image_latent_idx"]
            primary_duplicated_frames = video_tensor[0, :, primary_start_idx:primary_start_idx + 4, :, :].clone()
            transition["state"]["video"][0, :, future_primary_start_idx:future_primary_start_idx + 4, :, :] = (
                primary_duplicated_frames
            )
            transition_list[i] = transition
            transition_video_list.append(transition["state"]["video"])

            action_chunk = get_action_chunk_with_padding(
                actions=actions,
                relative_step_idx=i,
                chunk_size=chunk_size,
                num_steps=episode_length,
            )
            transition_action_chunk_list.append(
                torch.from_numpy(action_chunk).to(device=device, dtype=torch.float32)
            )

            proprio = transition["state"]["proprio"]
            transition_proprio_list.append(proprio if proprio.dim() > 1 else proprio.unsqueeze(0))

            future_proprio = future_transition["state"]["proprio"]
            if isinstance(future_proprio, torch.Tensor) and future_proprio.device != torch.device(device):
                future_proprio = future_proprio.to(device)
            transition_future_proprio_list.append(
                future_proprio if future_proprio.dim() > 1 else future_proprio.unsqueeze(0)
            )
            transition["state"]["value_function_return"] = returns[future]
            transition_value_function_return_list.append(float(returns[future]))    # (num_steps,) -> (B, )

        episode_state_video = torch.cat(transition_video_list, dim=0)    # (B, T, H, W, C)
        episode_state_action = torch.stack(transition_action_chunk_list, dim=0)    # (B, T, D)
        episode_state_proprio = torch.cat(transition_proprio_list, dim=0)
        episode_state_future_proprio = torch.cat(transition_future_proprio_list, dim=0)  # (B, T, D)
        episode_state_value_function_return = torch.tensor(
            transition_value_function_return_list,
            device=torch.device(device),
            # dtype=torch.float32,
        )    # (B, )

        action_latent_idx_batch = torch.full(
            (episode_length,), LATENT_INDICES["action_latent_idx"], dtype=torch.int64, device=device
        )
        proprio_latent_idx_batch = torch.full(
            (episode_length,), LATENT_INDICES["current_proprio_latent_idx"], dtype=torch.int64, device=device
        )
        future_proprio_latent_idx_batch = torch.full(
            (episode_length,), LATENT_INDICES["future_proprio_latent_idx"], dtype=torch.int64, device=device
        )
        value_latent_idx_batch = torch.full(
            (episode_length,), LATENT_INDICES["value_latent_idx"], dtype=torch.int64, device=device
        )

        encode_start_time = time.time()
        print("before encode")

        # 强制设置batch_size = self.config.batch_size
        sample_batch_size = batch_size
        episode_batch = {
            "video": episode_state_video,
            "action": episode_state_action,
            "proprio": episode_state_proprio,
            "future_proprio": episode_state_future_proprio,
            "value_function_return": episode_state_value_function_return,
            "fps": torch.tensor([16] * sample_batch_size, dtype=torch.bfloat16),
            "padding_mask": torch.zeros((sample_batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16), 
            "t5_text_embeddings": transition_list[0]["state"]["t5_text_embeddings"].repeat(sample_batch_size, 1, 1),
        }

        # 需要视频归一化和图像增强
        print("self.input_data_key: ", self.input_data_key)
        print("self.input_image_key: ", self.input_image_key)
        self._normalize_video_databatch_inplace(episode_batch, input_key='video')   # 只归一化video
        self._augment_image_dim_inplace(episode_batch)   # 这里实际没有进行图像增强，只对输入单帧图像进行处理
        print("====================== after normalize and augment")

        latent_state = self._encode_episode_state_on_device(
            episode_state=episode_batch["video"],
            encode_batch_size=encode_batch_size,
            world_size=world_size,
            encode_device_ids=encode_device_ids,
        )
        print("after encode")
        encode_end_time = time.time()
    
        print(f"encode time: {encode_end_time - encode_start_time:.2f}s, latent_state.shape: {latent_state.shape}")
        print("before inject")
        batch_indices = torch.arange(latent_state.shape[0], device=latent_state.device)
        c_latent, h_latent, w_latent = latent_state.shape[1], latent_state.shape[3], latent_state.shape[4]

        # encode之后使用的索引是action_latent_idx_batch=4
        latent_state = replace_latent_with_action_chunk(
            latent_state,
            episode_state_action,
            action_indices=action_latent_idx_batch,
        )
        # 注入proprio
        print("inject proprio")
        latent_state = replace_latent_with_proprio(
            latent_state,
            episode_state_proprio,
            proprio_indices=proprio_latent_idx_batch,
        )
        # 注入future_proprio
        print("inject future_proprio")
        latent_state = replace_latent_with_proprio(
            latent_state,
            episode_state_future_proprio,
            proprio_indices=future_proprio_latent_idx_batch,
        )
        # 注入value_function_return
        latent_state[batch_indices, :, value_latent_idx_batch, :, :] = (
            episode_state_value_function_return.reshape(-1, 1, 1, 1)
            .expand(-1, c_latent, h_latent, w_latent)
            .to(latent_state.dtype)
        )
        print("episode_state_value_function_return.dtype: ", episode_state_value_function_return.dtype)
        print("latent_state.dtype: ", latent_state.dtype)
        self.t5_text_embeddings = episode_batch["t5_text_embeddings"][:1].detach()
        # print("self.t5_text_embeddings.shape: ", self.t5_text_embeddings.shape)
        # print("episode_batch['t5_text_embeddings'].shape: ", episode_batch["t5_text_embeddings"].shape)
        return self._assign_episode_latent_to_transitions(transition_list, latent_state, episode_batch, cosmos_cfg)

    @staticmethod
    def _assign_episode_latent_to_transitions(
        transition_list: list,
        latent_state: torch.Tensor,
        episode_batch: Dict,
        cosmos_cfg: Any,
    ) -> list:
        """Write per-step latents back on CPU; keep only ``video`` and ``t5_text_embeddings``."""
        episode_length = len(transition_list)
        for i, transition in enumerate(transition_list):
            chunk_size = cosmos_cfg.chunk_size
            future = min(i + chunk_size, episode_length - 1)
            future_transition = transition_list[future]
            step_latent = latent_state[i : i + 1]
            next_idx = min(i + 1, episode_length - 1)
            next_step_latent = latent_state[next_idx : next_idx + 1]

            # 1. 所有索引最后变成一个全局变量； 2. 注意device是什么
            transition["state"] = {
                "video": step_latent,
                "proprio": transition["state"]["proprio"].detach(),
                "future_proprio": future_transition["state"]["proprio"].detach(),
                "value_function_return": episode_batch["value_function_return"][i].detach(),    # (B, ) ——> 标量
            }
            # transition["next_state"] = {
            #     "video": next_step_latent,
            #     # "t5_text_embeddings": t5_embeddings,
            # }
            transition["action"] = transition["action"].detach()
            transition_list[i] = transition
        return transition_list

    def _ensure_tokenizer_replicas(self, devices: list[str]) -> None:
        """Cache a tokenizer copy on each encode device (VAE only, not the full DiT)."""
        if not hasattr(self, "_tokenizer_replicas"):
            self._tokenizer_replicas: dict[str, nn.Module] = {}
        try:
            primary_device = str(next(self.tokenizer.parameters()).device)
        except StopIteration:
            primary_device = "cpu"

        for device_str in devices:
            if device_str in self._tokenizer_replicas:
                continue
            if device_str == primary_device:
                self._tokenizer_replicas[device_str] = self.tokenizer
                continue
            logger.info("Cloning tokenizer to %s for multi-GPU encode", device_str)
            tok = copy.deepcopy(self.tokenizer).to(device_str).eval()
            for p in tok.parameters():
                p.requires_grad_(False)
            self._tokenizer_replicas[device_str] = tok

    @staticmethod
    def _shard_lengths(num_items: int, world_size: int) -> list[int]:
        """Split ``num_items`` into ``world_size`` contiguous, non-overlapping shard lengths."""
        base, rem = divmod(num_items, world_size)
        return [base + (1 if i < rem else 0) for i in range(world_size)]

    @staticmethod
    def _resolve_encode_devices(
        world_size: int,
        encode_device_ids: list[int] | None,
        num_steps: int,
    ) -> list[str]:
        """Resolve process-local CUDA devices for sharded encode.

        ``encode_device_ids`` take precedence (e.g. ``[1, 2, 3]`` -> ``cuda:1,2,3``).
        Otherwise use ``cuda:0 .. cuda:world_size-1``. Indices are after
        ``CUDA_VISIBLE_DEVICES`` remapping.
        """
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available <= 0 or num_steps <= 0:
            return []

        if encode_device_ids is not None:
            ids = [int(i) for i in encode_device_ids]
            if not ids:
                raise ValueError("encode_device_ids is empty")
            bad = [i for i in ids if i < 0 or i >= available]
            if bad:
                raise ValueError(
                    f"encode_device_ids {bad} out of range; visible cuda devices: 0..{available - 1}"
                )
            # Keep order, drop duplicates
            seen: set[int] = set()
            uniq: list[int] = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            if len(uniq) > num_steps:
                uniq = uniq[:num_steps]
            return [f"cuda:{i}" for i in uniq]

        world_size = max(1, min(int(world_size), available, num_steps))
        return [f"cuda:{i}" for i in range(world_size)]

    def _encode_episode_state_on_device(
        self,
        episode_state: torch.Tensor,
        encode_batch_size: int = 1,
        world_size: int = 1,
        encode_device_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Encode episode videos in micro-batches; optionally shard across GPUs.

        Frames along dim=0 are partitioned without overlap onto the resolved encode
        devices, encoded in parallel, then concatenated in the original order on
        ``episode_state.device``.

        Prefer ``encode_device_ids`` to pick specific cards; otherwise ``world_size``
        selects ``cuda:0 .. cuda:world_size-1``.
        """
        num_steps = episode_state.shape[0]
        devices = self._resolve_encode_devices(world_size, encode_device_ids, num_steps)
        world_size = max(1, len(devices)) if devices else 1

        primary_device = episode_state.device
        print(
            f"encode devices={devices or ['<single/self.encode>']}, world_size={world_size}, "
            f"num_steps={num_steps}, encode_batch_size={encode_batch_size}, "
            f"primary_device={primary_device}"
        )

        # Single-device path: keep using policy.encode (tokenizer already on policy device).
        if world_size == 1:
            if devices and devices[0] != str(primary_device) and devices[0].startswith("cuda"):
                # Encode on a specific card that may differ from primary_device.
                self._ensure_tokenizer_replicas(devices)
                sigma_data = self.sigma_data
                tok = self._tokenizer_replicas[devices[0]]
                latent_chunks: list[torch.Tensor] = []
                with torch.inference_mode(), torch.cuda.device(devices[0]):
                    for start in range(0, num_steps, encode_batch_size):
                        batch = episode_state[start : start + encode_batch_size].to(
                            devices[0], non_blocking=True
                        )
                        latent = (tok.encode(batch) * sigma_data).contiguous().float()
                        latent_chunks.append(latent.to(primary_device))
                        del batch, latent
                return torch.cat(latent_chunks, dim=0)

            latent_chunks = []
            for start in range(0, num_steps, encode_batch_size):
                batch = episode_state[start : start + encode_batch_size]
                latent = self.encode(batch).contiguous().float()
                latent_chunks.append(latent)
                del batch, latent
            return torch.cat(latent_chunks, dim=0)

        self._ensure_tokenizer_replicas(devices)
        lengths = self._shard_lengths(num_steps, world_size)
        shards: list[torch.Tensor] = []
        offset = 0
        for length in lengths:
            shards.append(episode_state[offset : offset + length])
            offset += length

        sigma_data = self.sigma_data

        def _encode_shard(device_str: str, shard: torch.Tensor) -> torch.Tensor:
            if shard.numel() == 0 or shard.shape[0] == 0:
                return torch.empty(0)
            tok = self._tokenizer_replicas[device_str]
            outs: list[torch.Tensor] = []
            with torch.inference_mode(), torch.cuda.device(device_str):
                for start in range(0, shard.shape[0], encode_batch_size):
                    batch = shard[start : start + encode_batch_size].to(device_str, non_blocking=True)
                    latent = (tok.encode(batch) * sigma_data).contiguous().float()
                    outs.append(latent)
                    del batch, latent
            return torch.cat(outs, dim=0)

        with ThreadPoolExecutor(max_workers=world_size) as pool:
            parts = list(pool.map(_encode_shard, devices, shards))

        nonempty = [p for p in parts if p.numel() > 0]
        if not nonempty:
            raise RuntimeError("multi-GPU encode produced no latents")
        return torch.cat([p.to(primary_device) for p in nonempty], dim=0)
    
    # encode变成latent形式，但是此时未来状态和图像、动作均未知
    def change_to_latent(
        self, 
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        skip_vae_encoding: bool = False,
        previous_generated_latent: torch.Tensor = None
    ):
        x0_fn, orig_clean_latent_frames = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt=is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,  # skip_vae_encoding自回归使用的时候才会用到这个
                return_orig_clean_latent_frames=True,
            )
        
        # skip_vae_encoding支持自回归，我们不需要所以忽略？
        # _, x0, _ = self.get_data_and_condition(data_batch)
        is_image_batch = self.is_image_batch(data_batch) # False
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state_infer = self.encode(raw_state).contiguous().float()
        # print("latent_state.shape: ", latent_state_infer.shape)  # torch.Size([1, 16, 9, 28, 28])
        
        return raw_state, latent_state_infer

    # 当训练和推理都需要conditioner的时候使用
    # def _init_conditioner(
    #     self, data_batch: Dict, latent_state: torch.Tensor, is_negative_prompt: bool = False
    # ) -> tuple[Video2WorldCondition, Video2WorldCondition | None]:
    #     is_image_batch = self.is_image_batch(data_batch)
    #     # conditioner1：获取条件和无条件状态：文本嵌入和任务指令
    #     if is_negative_prompt:
    #         condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
    #     else:
    #         condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
    #     # 标记是图像还是视频
    #     condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
    #     if uncondition is not None:
    #         uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

    #     num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None) # 注意训练的时候设置一个None!!!!!!!!!!!!
        
    #     # condition2: 训练的时候固定为config里面的设置random_min_num_conditional_frames；推理的时候num_conditional_frames需要等于1
    #     condition = condition.set_video_condition(
    #         gt_frames=latent_state.to(**self.tensor_kwargs),  # 真实潜在状态作为目标帧
    #         random_min_num_conditional_frames=self.config.min_num_conditional_frames,  # 最少条件帧数
    #         random_max_num_conditional_frames=self.config.max_num_conditional_frames,  # 最多条件帧数
    #         num_conditional_frames=num_conditional_frames,  # 外部指定的条件帧数
    #         conditional_frames_probs=self.config.conditional_frames_probs,  # 条件帧数的概率分布
    #     )
    #     if uncondition is not None:
    #         uncondition = uncondition.set_video_condition(
    #             gt_frames=latent_state.to(**self.tensor_kwargs),
    #             random_min_num_conditional_frames=self.config.min_num_conditional_frames,
    #             random_max_num_conditional_frames=self.config.max_num_conditional_frames,
    #             num_conditional_frames=num_conditional_frames,
    #         )

    #     return condition, uncondition, num_conditional_frames

    def _init_conditioner(self, data_batch: Dict) -> Video2WorldCondition:
        is_image_batch = False
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return condition

    def set_conditioner(self, condition: Video2WorldCondition, latent_state: torch.Tensor, data_batch: Dict):
        # 设置视频条件（条件帧数量控制）：训练的时候固定为config里面的设置random_min_num_conditional_frames；推理的时候num_conditional_frames需要等于1
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),  # 真实潜在状态作为目标帧
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,  # 最少条件帧数
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,  # 最多条件帧数
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),  # 外部指定的条件帧数
            conditional_frames_probs=self.config.conditional_frames_probs,  # 条件帧数的概率分布
        )
        batch_indices = torch.arange(latent_state.shape[0], device=latent_state.device)
        _, C_latent, _, H_latent, W_latent = condition.gt_frames.shape
        batch_size = latent_state.shape[0]
        device = latent_state.device
        proprio = data_batch["proprio"].to(device)
        future_proprio = data_batch["future_proprio"].to(device)
        value_function_return = data_batch["value_function_return"].to(device)

        # proprio注入condition中
        condition.gt_frames = replace_latent_with_proprio(
            condition.gt_frames,
            proprio,
            proprio_indices=torch.full(
                (batch_size,), LATENT_INDICES["current_proprio_latent_idx"], dtype=torch.int64, device=device
            ),
        )
        # future_proprio注入到condition中，但是推理没有这个键，是否可以在推理之后放入然后保存
        condition.gt_frames = replace_latent_with_proprio(
            condition.gt_frames,
            future_proprio,
            proprio_indices=torch.full(
                (batch_size,), LATENT_INDICES["future_proprio_latent_idx"], dtype=torch.int64, device=device
            ),
        )
        # value 注入到condition中
        condition.gt_frames[batch_indices, :, LATENT_INDICES["value_latent_idx"], :, :] = (
            value_function_return
            .reshape(-1, 1, 1, 1)  # (B,) -> (B, 1, 1, 1)
            .expand(-1, C_latent, H_latent, W_latent)  # 扩展到 (B, C, H, W)
            .to(condition.gt_frames.dtype)
        )
        return condition
        

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
        env_cfg: any = None,
    ) -> dict[str, Tensor]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict
                - next_state: Next observations tensor dict
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features
                - next_observation_feature: Optional pre-computed next observation features
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch["action"]
        observations: dict[str, Tensor] = batch["state"]    # 这里应该是step_latent
        observation_features: Tensor = batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")

            loss_critic = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            next_observation_features: Tensor = batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic, 'loss_bc': loss_discrete_critic.clone().detach(), "loss_q": loss_discrete_critic.clone().detach()}
        if model == "actor":
            loss_actor,future_image_l1_loss, future_wrist_image_l1_loss, future_proprio_l1_loss, action_l1_loss, value_l1_loss = self.compute_loss_actor(
                observations=observations,
                observation_features=observation_features,
                actions=actions,
                env_cfg=env_cfg,
            )
            return {
                "loss_actor": loss_actor,
                "bc_loss": loss_actor.clone().detach(),
                "min_q_preds": loss_actor.clone().detach(),
                "future_image_l1_loss": future_image_l1_loss,
                "future_wrist_image_l1_loss": future_wrist_image_l1_loss,
                "future_proprio_l1_loss": future_proprio_l1_loss,
                "action_l1_loss": action_l1_loss,
                "value_l1_loss": value_l1_loss,
            }

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations, next_observation_features)

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * next_log_probs)

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        # if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
        actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None

        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )

            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)
    

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )
         

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)


            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards

            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties

            
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q
            # print("target_discrete_q:", target_discrete_q)
        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)
        
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)

        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            _, log_probs, _ = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
        actions: Tensor | None = None,
        env_cfg: any = None,
    ) -> Tensor:
        """Actor loss via Cosmos diffusion denoising (``self.net`` / ``self.actor``)."""
        del observation_features  # DiT backbone does not use SAC observation features.
        x0_B_C_T_H_W = observations["video"]

        # 失败数据不对action反向传播梯度
        returns = observations["value_function_return"]   # (B, )
        # print("returns:", returns.shape, returns)
        # fail_B = torch.isclose(returns.float(), torch.full_like(returns.float(), -1.0))  # (B,)
        fail_B = returns.float() == -1.0
        # print("fail_B:", fail_B.shape, fail_B)
        
        device = x0_B_C_T_H_W.device
        batch_size = x0_B_C_T_H_W.shape[0]
        if self.t5_text_embeddings is None:
            from cosmos_policy.experiments.robot.cosmos_utils import (
                get_t5_embedding_from_cache,
                init_t5_text_embeddings_cache,
            )
            task_description = env_cfg.task_description
            t5_text_embeddings_path = env_cfg.t5_text_embeddings_path
            init_t5_text_embeddings_cache(t5_text_embeddings_path, worker_id=0)
            self.t5_text_embeddings = get_t5_embedding_from_cache(task_description)
        # print("=========================>>> before init conditioner")
        condition = self._init_conditioner(
            {
                "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16, device=device),
                "padding_mask": torch.zeros(
                    (batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "t5_text_embeddings": self.t5_text_embeddings.to(device).expand(batch_size, -1, -1),
            }
        )
        condition = self.set_conditioner(condition, x0_B_C_T_H_W, observations)
        # print("=========================>>> after set conditioner")
        # denoise:
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)
        mean_B_C_T_H_W, std_B_T = self.sde.marginal_prob(x0_B_C_T_H_W, sigma_B_T)
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")  # xt = 干净数据 + 噪声强度 × 随机噪声 
        # print("=========================>>> before denoise")
        model_pred = self.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # print("=========================>>> after denoise")
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
        # 计算VAE编码后的潜在特征图的各种损失：MSE损失、EDM损失、Kendall损失
        # print("x0_B_C_T_H_W:", x0_B_C_T_H_W.shape, "model_pred.x0:", model_pred.x0.shape)
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2  # MSE

        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")  # EDM损失：MSE损失 × 噪声权重
        # print("weights_per_sigma_B_T:", weights_per_sigma_B_T)
        action_idx = LATENT_INDICES["action_latent_idx"]
        kendall_loss = edm_loss_B_C_T_H_W  # Kendall损失
        # print("kendall_loss:", kendall_loss.shape, kendall_loss[:, :, action_idx])

        kendall_loss[fail_B, :, action_idx] = 0   # 只清失败样本的 action 帧
        # print("kendall_loss:", kendall_loss.shape, kendall_loss[:, :, action_idx])
        
        batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)

        # print("before calculate future losses")
        # future image loss
        future_image_idx = LATENT_INDICES["future_image_latent_idx"]
        future_image_diff = (
            x0_B_C_T_H_W[batch_indices, :, future_image_idx, :, :]
            - model_pred.x0[batch_indices, :, future_image_idx, :, :]
        )
        future_image_loss = (future_image_diff**2).mean().detach()
        future_image_l1_loss = torch.abs(future_image_diff).mean().detach()
        # print("real future image:", x0_B_C_T_H_W[batch_indices, :, future_image_idx, :, :].max(), "min:", x0_B_C_T_H_W[batch_indices, :, future_image_idx, :, :].min())
        # print("pred future image:", model_pred.x0[batch_indices, :, future_image_idx, :, :].max(), "min:", model_pred.x0[batch_indices, :, future_image_idx, :, :].min())
        # print("future_image_loss:", future_image_loss, "future_image_l1_loss:", future_image_l1_loss)
        # future wrist image loss
        future_wrist_image_idx = LATENT_INDICES["future_wrist_image_latent_idx"]
        future_wrist_image_diff = (
            x0_B_C_T_H_W[batch_indices, :, future_wrist_image_idx, :, :]
            - model_pred.x0[batch_indices, :, future_wrist_image_idx, :, :]
        )
        future_wrist_image_loss = (future_wrist_image_diff**2).mean().detach()
        future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff).mean().detach()
        # print("future_wrist_image_loss:", future_wrist_image_loss, "future_wrist_image_l1_loss:", future_wrist_image_l1_loss)

        # future proprio loss
        future_proprio_idx = LATENT_INDICES["future_proprio_latent_idx"]
        future_proprio_diff = (
            x0_B_C_T_H_W[batch_indices, :, future_proprio_idx, :, :]
            - model_pred.x0[batch_indices, :, future_proprio_idx, :, :]
        )
        future_proprio_loss = (future_proprio_diff**2).mean().detach()
        future_proprio_l1_loss = torch.abs(future_proprio_diff).mean().detach()
        # print("future_proprio_loss:", future_proprio_loss, "future_proprio_l1_loss:", future_proprio_l1_loss)

        # action loss
        action_diff = (
            x0_B_C_T_H_W[batch_indices, :, action_idx, :, :]
            - model_pred.x0[batch_indices, :, action_idx, :, :]
        )
        action_loss = (action_diff**2).mean().detach()
        action_l1_loss = torch.abs(action_diff).mean().detach()
        # print("GT latent action:", x0_B_C_T_H_W[batch_indices, :, action_idx, :, :])
        # print("pred latent action:", model_pred.x0[batch_indices, :, action_idx, :, :])
        # # decode for debug
        # action_latent_idx = LATENT_INDICES["action_latent_idx"]
        # action_dim = self.continuous_action_dim + 1
        # action_indices = torch.full(
        #     (batch_size,),
        #     action_latent_idx,
        #     dtype=torch.int64,
        #     device=x0_B_C_T_H_W.device,
        # )
        # gt_actions = (
        #     extract_action_chunk_from_latent_sequence(
        #         x0_B_C_T_H_W, action_shape=(16, action_dim), action_indices=action_indices
        #     )
        #     .to(torch.float32)
        #     .cpu()
        #     .numpy()
        # )  # (batch_size, chunk_size, action_dim)
        # pred_actions = (
        #     extract_action_chunk_from_latent_sequence(
        #         model_pred.x0, action_shape=(16, action_dim), action_indices=action_indices
        #     )
        #     .to(torch.float32)
        #     .detach().cpu()
        #     .numpy()
        # )  # (batch_size, chunk_size, action_dim)
        # dataset_stats_cosmos = load_dataset_stats(env_cfg.dataset_stats_path)
        # print("actions shape:", actions.shape, gt_actions.shape)
        # gt_actions = unnormalize_actions(gt_actions, dataset_stats_cosmos)
        # pred_actions = unnormalize_actions(pred_actions, dataset_stats_cosmos)
        # input_actions = unnormalize_actions(actions.cpu().float(), dataset_stats_cosmos)
        # print("GT actions:", gt_actions[0])
        # print("input actions:", input_actions[0])
        # print("pred actions:", pred_actions[0])
        # exit(0)
        
        # print("action_loss:", action_loss, "action_l1_loss:", action_l1_loss)

        # value loss
        value_idx = LATENT_INDICES["value_latent_idx"]
        value_diff = (
            x0_B_C_T_H_W[batch_indices, :, value_idx, :, :]
            - model_pred.x0[batch_indices, :, value_idx, :, :]
        )
        value_loss = (value_diff**2).mean().detach()
        value_l1_loss = torch.abs(value_diff).mean().detach()
        # print("value_loss:", value_loss, "value_l1_loss:", value_l1_loss)

        # print("kendall_loss:", kendall_loss)
        if self.loss_reduce == "mean":
            # print("self.loss_reduce:", self.loss_reduce)
            # print("self.loss_scale:", self.loss_scale)
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        # print("already 前向传播, return kendall_loss")
        return kendall_loss, future_image_l1_loss, future_wrist_image_l1_loss, future_proprio_l1_loss, action_l1_loss, value_l1_loss


    # def _init_normalization(self, dataset_stats, batch: dict[str, Tensor], non_negative_only=False, scale_multiplier=1.0):
    #     """Initialize input/output normalization modules."""
    #     self.normalize_inputs = nn.Identity()
    #     self.normalize_targets = nn.Identity()
    #     if self.config.dataset_stats is not None:
    #         params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
    #         self.normalize_inputs = NormalizeBuffer(
    #             self.config.input_features, self.config.normalization_mapping, params
    #         )
    #         stats = dataset_stats or params
    #         self.normalize_targets = NormalizeBuffer(
    #             self.config.output_features, self.config.normalization_mapping, stats
    #         )

    #         # normalize proprio：因为cosmos设置non_negative_only为False，所以范围是[-1,1]
    #         if not non_negative_only:
    #             normalized_batch = self.normalize_inputs(batch)  # 返回归一化后的字典
    #             normalized_proprio = normalized_batch[OBS_STATE]
    #             normalized_proprio = scale_multiplier * normalized_proprio
           
    #     return normalized_proprio
    
    def _init_normalization(self, dataset_stats):
        """Initialize input/output normalization modules."""
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        if self.config.dataset_stats is not None:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = NormalizeBuffer(
                self.config.input_features, self.config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = NormalizeBuffer(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def normalizer(self, transition_list, dataset_stats_cosmos):
        if dataset_stats_cosmos is not None:
            for t in transition_list:
                # rescale_* mixes with numpy stats; bfloat16 tensors cannot .numpy()
                t["state"]["proprio"] = self._rescale_tensor_or_array(
                    t["state"]["proprio"],
                    rescale_proprio,
                    dataset_stats_cosmos,
                )
                t["action"] = self._rescale_tensor_or_array(
                    t["action"],
                    rescale_action,
                    dataset_stats_cosmos,
                )
        else:
            raise ValueError("dataset_stats_cosmos is None")
        return transition_list

    @staticmethod
    def _rescale_tensor_or_array(x, rescale_fn, dataset_stats_cosmos):
        if isinstance(x, torch.Tensor):
            device, dtype = x.device, x.dtype
            x_np = x.detach().float().cpu().numpy()
            out = rescale_fn(x_np, dataset_stats_cosmos, non_negative_only=False, scale_multiplier=1.0)
            return torch.from_numpy(np.asarray(out, dtype=np.float32)).to(device=device, dtype=dtype)
        out = rescale_fn(
            np.asarray(x, dtype=np.float32),
            dataset_stats_cosmos,
            non_negative_only=False,
            scale_multiplier=1.0,
        )
        return out


    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config, self.normalize_inputs)
        self.encoder_actor = (
            self.encoder_critic
            if self.shared_encoder
            else SACObservationEncoder(self.config, self.normalize_inputs)
        )

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=heads, output_normalization=self.normalize_targets
        )
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(
            encoder=self.encoder_critic, ensemble=target_heads, output_normalization=self.normalize_targets
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # # NOTE: The actor select only the continuous action part
        # self.actor = Policy(
        #     encoder=self.encoder_actor,
        #     network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
        #     action_dim=continuous_action_dim,
        #     encoder_is_shared=self.shared_encoder,
        #     **asdict(self.config.policy_kwargs),
        # )
        self.actor = self.net

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()

    def init_global_t5_text_embedding(
        self,
        task_description: str,
        t5_text_embeddings_path: str,
        gpu_id: int = 0,
    ) -> torch.Tensor:
        """Initialize T5 cache once and reuse the same text embedding globally."""
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_t5_embedding_from_cache,
            init_t5_text_embeddings_cache,
        )
        init_t5_text_embeddings_cache(t5_text_embeddings_path, worker_id=gpu_id)
        text_embedding = get_t5_embedding_from_cache(task_description)
        return text_embedding


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        output_normalization (nn.Module): normalization layer for actions.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        output_normalization: nn.Module,
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.output_normalization = output_normalization
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items()}
        # NOTE: We normalize actions it helps for sample efficiency
        actions: dict[str, torch.tensor] = {"action": actions}
        # NOTE: Normalization layer took dict in input and outputs a dict that why

        # print("actions before normalization:", actions)
        actions = self.output_normalization(actions)["action"]
        # print("actions after normalization:", actions)
        
        actions = actions.to(device)

        obs_enc = self.encoder(observations, cache=observation_features)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)

        return q_values


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)  # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations



def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings

        Args:
            height: Spatial height of input features
            width: Spatial width of input features
            channel: Number of input channels
            num_features: Number of output embedding dimensions
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))

        nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding

        Args:
            features: Input tensor of shape [B, C, H, W] where B is batch size,
                     C is number of channels, H is height, and W is width
        Returns:
            Output tensor of shape [B, C*F] where F is the number of features
        """

        features_expanded = features.unsqueeze(-1)  # [B, C, H, W, 1]
        kernel_expanded = self.kernel.unsqueeze(0)  # [1, C, H, W, F]

        # Element-wise multiplication and spatial reduction
        output = (features_expanded * kernel_expanded).sum(dim=(2, 3))  # Sum over H,W dimensions

        # Reshape to combine channel and feature dimensions
        output = output.view(output.size(0), -1)  # [B, C*F]

        return output


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x

# 将normalization_params转换为tensor
def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    converted_params = {}
    for outer_key, inner_dict in normalization_params.items():
        converted_params[outer_key] = {}
        for key, value in inner_dict.items():
            converted_params[outer_key][key] = torch.tensor(value)
            if "image" in outer_key:
                converted_params[outer_key][key] = converted_params[outer_key][key].view(3, 1, 1)

    return converted_params

def rescale_action(action, dataset_stats, non_negative_only=False, scale_multiplier=1.0):
    """
    Rescale (normalize) action to the range [-1,+1] or [0,+1], with optional scaling by scale_multiplier.

    Args:
        action (np.ndarray): Action to be rescaled
        dataset_stats (dict): Dataset statistics needed for rescaling formula
        non_negative_only (bool): Whether to use [0,+1] range (True) or [-1,+1] range (False)
        scale_multiplier (float): Multiplier to adjust final scale

    Returns:
        np.ndarray: Rescaled action
    """
    arr = action
    curr_min = dataset_stats["actions_min"]
    curr_max = dataset_stats["actions_max"]
    # First, scale to [-1,+1] or [0,+1]:
    # - For [-1,+1]: x_new = 2 * ((x - curr_min) / (curr_max - curr_min)) - 1
    # - For [0,+1]: x_new = (x - curr_min) / (curr_max - curr_min)
    if not non_negative_only:  # [-1,+1]
        rescaled_arr = 2 * ((arr - curr_min) / (curr_max - curr_min)) - 1
    else:  # [0,+1]
        rescaled_arr = (arr - curr_min) / (curr_max - curr_min)
    # Scale to [-scale_multiplier,+scale_multiplier] or [0,+scale_multiplier]
    rescaled_arr = scale_multiplier * rescaled_arr
    action = rescaled_arr
    return action