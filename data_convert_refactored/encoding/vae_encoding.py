"""CPU microbatch视频归一化、Cosmos VAE编码及latent条件注入。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cosmos_policy.datasets.dataset_common import (
    compute_monte_carlo_returns,
    get_action_chunk_with_padding,
)

from ..conditioning.camera import LATENT_INDICES, RAW_FRAME_SLICES
from ..shape_trace import require_shape, trace_shape


CLEAN_RESTORE_INDICES = (1, 4, 5, 8)


def normalize_video_batch_cpu(video_batch: torch.Tensor) -> torch.Tensor:
    """按microbatch把uint8 [0,255]视频转为float32 [-1,1]，避免整条episode膨胀。"""
    normalized = video_batch.float()
    normalized.div_(127.5).sub_(1.0)
    return normalized


def encode_normalized_video_batch(
    policy: Any,
    batch_cpu: torch.Tensor,
    *,
    single_device: torch.device,
    encode_batch_size: int,
    encode_world_size: int = 1,
    encode_device_ids: tuple[int, ...] | None = None,
    tokenizer_config: Any = None,
) -> torch.Tensor:
    """编码一个CPU microbatch，按配置选择单GPU或缓存的多GPU VAE池。

    输入为(B,3,33,224,224)，返回CPU上的(B,16,9,28,28) float32 latent。
    """
    if batch_cpu.device.type != "cpu":
        raise ValueError(f"normalized video batch must stay on CPU, got {batch_cpu.device}")
    if encode_world_size < 1:
        raise ValueError("encode_world_size must be >= 1")

    multi_gpu_requested = encode_world_size > 1 or encode_device_ids is not None
    if not multi_gpu_requested:
        batch_gpu = batch_cpu.to(single_device)
        try:
            latent = policy.encode(batch_gpu).contiguous().float().cpu()
        finally:
            del batch_gpu
    else:
        if tokenizer_config is None:
            raise ValueError("multi-GPU VAE encode requires tokenizer_config")
        from .multi_gpu_vae import get_or_create_vae_replica_pool

        pool = get_or_create_vae_replica_pool(
            policy=policy,
            tokenizer_config=tokenizer_config,
            encode_world_size=encode_world_size,
            encode_device_ids=encode_device_ids,
            encode_batch_size=encode_batch_size,
        )
        latent = pool.encode(batch_cpu, sigma_data=policy.sigma_data)

    if latent.shape[0] != batch_cpu.shape[0]:
        raise RuntimeError(
            f"VAE latent batch length {latent.shape[0]} does not match "
            f"input batch length {batch_cpu.shape[0]}"
        )
    return latent


def encode_episode_cpu_friendly(
    transition_list: list[dict],
    encode_batch_size: int,
    device: torch.device,
    cosmos_cfg: Any,
    batch_size: int,
    policy: Any,
    encode_world_size: int = 1,
    encode_device_ids: tuple[int, ...] | None = None,
    save_clean_restore_latent: bool = False,
) -> list[dict]:
    """编码完整episode并注入低维监督条件。

    action chunk形状为(T,chunk_size,D)，默认chunk_size=16；future proprio和未来图像
    使用`min(t+chunk_size,T-1)`。输出仍是T个transition，每步video为
    (1,16,9,28,28)。
    """
    from cosmos_policy.models.policy_text2world_model import (
        replace_latent_with_action_chunk,
        replace_latent_with_proprio,
    )

    from ..memory_monitor import memory_event, tensor_nbytes

    episode_length = len(transition_list)
    memory_event("encode_start", episode_length=episode_length)
    chunk_size = cosmos_cfg.chunk_size

    actions = np.stack(
        [
            transition["action"].detach().cpu().numpy()
            if isinstance(transition["action"], torch.Tensor)
            else np.asarray(transition["action"], dtype=np.float32)
            for transition in transition_list
        ],
        axis=0,
    )
    memory_event("after_actions_stack", actions_bytes=actions.nbytes)
    # value沿用Cosmos原生Monte Carlo return：成功episode终值1，失败episode终值0。
    terminal_reward = 1.0 if transition_list[-1]["done"] else 0.0
    gamma = getattr(cosmos_cfg, "gamma", 0.99)
    returns = compute_monte_carlo_returns(
        episode_length, terminal_reward=terminal_reward, gamma=gamma,
    )

    transition_video_list: list[torch.Tensor] = []
    transition_action_chunk_list: list[torch.Tensor] = []
    transition_proprio_list: list[torch.Tensor] = []
    transition_future_proprio_list: list[torch.Tensor] = []
    transition_value_list: list[float] = []

    memory_event("before_transition_clone_loop")
    progress_step = max(100, episode_length // 10)
    for index, transition in enumerate(transition_list):
        future = min(index + chunk_size, episode_length - 1)
        future_transition = transition_list[future]

        if isinstance(transition["state"].get("video"), torch.Tensor):
            transition["state"]["video"] = transition["state"]["video"].clone()

        video_tensor = future_transition["state"]["video"]
        wrist_dup = video_tensor[0, :, RAW_FRAME_SLICES["current_wrist"], :, :].clone()
        transition["state"]["video"][0, :, RAW_FRAME_SLICES["future_wrist"], :, :] = wrist_dup

        primary_dup = video_tensor[0, :, RAW_FRAME_SLICES["current_primary"], :, :].clone()
        transition["state"]["video"][0, :, RAW_FRAME_SLICES["future_primary"], :, :] = primary_dup

        transition_list[index] = transition
        transition_video_list.append(transition["state"]["video"])

        # 取action[t:t+16]；越过episode尾部时由官方函数复制末值补齐。
        action_chunk = get_action_chunk_with_padding(actions, index, chunk_size, episode_length)
        transition_action_chunk_list.append(torch.from_numpy(action_chunk).to(dtype=torch.float32))

        proprio = transition["state"]["proprio"]
        transition_proprio_list.append(proprio if proprio.dim() > 1 else proprio.unsqueeze(0))

        future_proprio = future_transition["state"]["proprio"]
        transition_future_proprio_list.append(
            future_proprio if future_proprio.dim() > 1 else future_proprio.unsqueeze(0)
        )

        transition["state"]["value_function_return"] = returns[future]
        transition_value_list.append(float(returns[future]))
        if (index + 1) % progress_step == 0 or index + 1 == episode_length:
            memory_event("transition_clone_progress", frames_processed=index + 1)

    memory_event("after_transition_clone_loop")

    episode_state_action = torch.stack(transition_action_chunk_list, dim=0)
    episode_state_proprio = torch.cat(transition_proprio_list, dim=0)
    episode_state_future_proprio = torch.cat(transition_future_proprio_list, dim=0)
    episode_state_value = torch.tensor(transition_value_list, dtype=torch.float32)
    action_dim = int(actions.shape[1])
    proprio_dim = int(episode_state_proprio.shape[1])
    require_shape(
        "episode_state_action",
        episode_state_action,
        (episode_length, chunk_size, action_dim),
    )
    require_shape(
        "episode_state_proprio", episode_state_proprio, (episode_length, proprio_dim)
    )
    require_shape(
        "episode_state_future_proprio",
        episode_state_future_proprio,
        (episode_length, proprio_dim),
    )
    require_shape("episode_state_value", episode_state_value, (episode_length,))
    trace_shape(
        "CONDITIONING",
        action_chunk=episode_state_action,
        proprio=episode_state_proprio,
        future_proprio=episode_state_future_proprio,
        value=episode_state_value,
    )

    latent_chunks: list[torch.Tensor] = []
    # CPU只对当前microbatch执行float32归一化；GPU峰值由encode_batch_size控制。
    for start in range(0, episode_length, encode_batch_size):
        end = min(start + encode_batch_size, episode_length)
        memory_event(
            "microbatch_cat_start",
            batch_start=start,
            batch_end=end,
            latent_chunk_count=len(latent_chunks),
        )
        batch_cpu = torch.cat(transition_video_list[start:end], dim=0)
        memory_event(
            "microbatch_cat_complete",
            batch_start=start,
            batch_end=end,
            tensor_bytes=tensor_nbytes(batch_cpu),
        )
        memory_event("microbatch_float_start", batch_start=start, batch_end=end)
        batch_cpu = normalize_video_batch_cpu(batch_cpu)
        current_batch_size = end - start
        require_shape(
            "vae_input", batch_cpu, (current_batch_size, 3, 33, 224, 224)
        )
        if start == 0 or end == episode_length:
            trace_shape(f"VAE_INPUT frames={start}:{end}", video=batch_cpu)
        memory_event(
            "microbatch_normalize_complete",
            batch_start=start,
            batch_end=end,
            tensor_bytes=tensor_nbytes(batch_cpu),
            tensor_dtype=str(batch_cpu.dtype),
        )
        latent = encode_normalized_video_batch(
            policy,
            batch_cpu,
            single_device=device,
            encode_batch_size=encode_batch_size,
            encode_world_size=encode_world_size,
            encode_device_ids=encode_device_ids,
            tokenizer_config=cosmos_cfg.tokenizer,
        )
        require_shape(
            "vae_output", latent, (current_batch_size, 16, 9, 28, 28)
        )
        if start == 0 or end == episode_length:
            trace_shape(f"VAE_OUTPUT frames={start}:{end}", latent=latent)
        memory_event(
            "microbatch_vae_encode_complete",
            batch_start=start,
            batch_end=end,
            encode_world_size=encode_world_size,
            encode_device_ids=list(encode_device_ids) if encode_device_ids is not None else None,
        )
        latent_chunks.append(latent)
        del batch_cpu, latent
        torch.cuda.empty_cache()
        memory_event("vae_batch_complete", batch_start=start, latent_chunk_count=len(latent_chunks))

    memory_event("before_latent_cat", latent_chunk_count=len(latent_chunks))
    latent_state = torch.cat(latent_chunks, dim=0)
    require_shape("latent_state", latent_state, (episode_length, 16, 9, 28, 28))
    memory_event(
        "after_latent_cat",
        tensor_bytes=tensor_nbytes(latent_state),
        tensor_shape=list(latent_state.shape),
    )
    del latent_chunks, transition_video_list
    memory_event("after_delete_video_and_chunks")

    channels, height, width = latent_state.shape[1], latent_state.shape[3], latent_state.shape[4]
    batch_indices = torch.arange(latent_state.shape[0])

    action_latent_idx = torch.full(
        (episode_length,), LATENT_INDICES["action_latent_idx"], dtype=torch.int64
    )
    proprio_latent_idx = torch.full(
        (episode_length,), LATENT_INDICES["current_proprio_latent_idx"], dtype=torch.int64
    )
    future_proprio_latent_idx = torch.full(
        (episode_length,), LATENT_INDICES["future_proprio_latent_idx"], dtype=torch.int64
    )
    value_latent_idx = torch.full(
        (episode_length,), LATENT_INDICES["value_latent_idx"], dtype=torch.int64
    )

    clean_restore_latent = None
    if save_clean_restore_latent:
        clean_restore_latent = latent_state[:, :, list(CLEAN_RESTORE_INDICES)].to(torch.float16)
        require_shape(
            "clean_restore_latent",
            clean_restore_latent,
            (episode_length, 16, len(CLEAN_RESTORE_INDICES), 28, 28),
        )
        memory_event(
            "after_clean_restore_extract",
            tensor_bytes=tensor_nbytes(clean_restore_latent),
            restore_indices=list(CLEAN_RESTORE_INDICES),
        )

    memory_event("before_latent_injection")
    # 固定latent时间位置：action=4、current proprio=1、future proprio=5、value=8。
    latent_state = replace_latent_with_action_chunk(latent_state, episode_state_action, action_latent_idx)
    memory_event("after_action_injection")
    latent_state = replace_latent_with_proprio(latent_state, episode_state_proprio, proprio_latent_idx)
    memory_event("after_current_proprio_injection")
    latent_state = replace_latent_with_proprio(
        latent_state, episode_state_future_proprio, future_proprio_latent_idx
    )
    memory_event("after_future_proprio_injection")
    latent_state[batch_indices, :, value_latent_idx, :, :] = (
        episode_state_value.reshape(-1, 1, 1, 1)
        .expand(-1, channels, height, width)
        .to(latent_state.dtype)
    )
    memory_event("after_value_injection")
    require_shape("injected_latent", latent_state, (episode_length, 16, 9, 28, 28))
    trace_shape("LATENT_INJECTED", latent=latent_state)

    for index, transition in enumerate(transition_list):
        future = min(index + chunk_size, episode_length - 1)
        future_transition = transition_list[future]
        step_latent = latent_state[index:index + 1]

        transition["state"] = {
            "video": step_latent,
            "proprio": transition["state"]["proprio"].detach().cpu(),
            "future_proprio": future_transition["state"]["proprio"].detach().cpu(),
            "value_function_return": torch.tensor([transition_value_list[index]], dtype=torch.float32),
        }
        # 保存实际传给replace_latent_with_action_chunk的同一份dataset-normalized
        # action chunk，便于从Parquet逐元素核对latent时间索引4。
        transition["action.latent_chunk"] = episode_state_action[index].detach().cpu()
        if clean_restore_latent is not None:
            transition["state"]["clean_restore_latent"] = clean_restore_latent[index:index + 1]
        transition["action"] = transition["action"].detach().cpu()
        transition_list[index] = transition

    memory_event("after_assign_latent_to_transitions")
    return transition_list
