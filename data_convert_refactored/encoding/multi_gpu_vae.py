"""转换侧GPU配置和多GPU VAE wrapper池。

Cosmos的`Wan2pt1VAEInterface`不是`nn.Module`，不能直接调用`to()`复制。因此每张
编码卡初始化完整wrapper并缓存；episode处理及Parquet写入仍由主进程串行执行。
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Callable

import torch


TokenizerFactory = Callable[[Any, torch.device], Any]
_POOL_ATTRIBUTE = "_conversion_vae_replica_pool"
_NORMALIZATION_TENSORS = (
    "mean",
    "std",
    "img_mean",
    "img_std",
    "video_mean",
    "video_std",
)


def parse_encode_device_ids(raw: str | None) -> tuple[int, ...] | None:
    """解析CUDA_VISIBLE_DEVICES映射后的进程内编号，例如`0,2,3`。"""
    if raw is None or not raw.strip():
        return None
    try:
        device_ids = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("encode device IDs must be comma-separated integers") from exc
    validate_encode_settings(1, device_ids)
    return device_ids or None


def validate_encode_settings(
    world_size: int, device_ids: tuple[int, ...] | None
) -> None:
    """拒绝空、负数或重复GPU编号，并保证world size为正数。"""
    if world_size < 1:
        raise ValueError("encode_world_size must be >= 1")
    if device_ids is None:
        return
    if not device_ids:
        raise ValueError("encode_device_ids must not be empty")
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError("encode device IDs must be non-negative")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("encode device IDs must not contain duplicates")


def resolve_encode_settings(
    cli_world_size: int | None,
    cli_device_ids: str | None,
) -> tuple[int, tuple[int, ...] | None]:
    """解析GPU参数；CLI优先于ENCODE_*环境变量，显式device IDs优先于world size。"""
    raw_device_ids = (
        cli_device_ids
        if cli_device_ids is not None
        else os.environ.get("ENCODE_CUDA_DEVICES")
    )
    device_ids = parse_encode_device_ids(raw_device_ids)
    if cli_world_size is not None:
        world_size = cli_world_size
    else:
        raw_world_size = os.environ.get("ENCODE_WORLD_SIZE", "1")
        try:
            world_size = int(raw_world_size)
        except ValueError as exc:
            raise ValueError("ENCODE_WORLD_SIZE must be an integer") from exc
    validate_encode_settings(world_size, device_ids)
    return world_size, device_ids


def _tokenizer_backend(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "model", None)
    model = getattr(backend, "model", None)
    if backend is None or not isinstance(model, torch.nn.Module):
        raise TypeError(
            "expected a Wan2pt1VAEInterface-like tokenizer with "
            "tokenizer.model.model as torch.nn.Module"
        )
    return backend


def tokenizer_device(tokenizer: Any) -> torch.device:
    """从Wan VAE内部nn.Module参数确定wrapper所在设备。"""
    backend = _tokenizer_backend(tokenizer)
    try:
        return next(backend.model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("VAE tokenizer model has no parameters") from exc


def validate_tokenizer_replica(tokenizer: Any, expected_device: torch.device) -> None:
    """确认模型权重和全部归一化Tensor都位于目标GPU。"""
    expected_device = torch.device(expected_device)
    actual_device = tokenizer_device(tokenizer)
    if actual_device != expected_device:
        raise RuntimeError(
            f"VAE replica expected on {expected_device}, got {actual_device}"
        )

    backend = _tokenizer_backend(tokenizer)
    for name in _NORMALIZATION_TENSORS:
        tensor = getattr(backend, name, None)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"VAE replica normalization field {name!r} is not a tensor")
        if tensor.device != expected_device:
            raise RuntimeError(
                f"VAE replica field {name} expected on {expected_device}, "
                f"got {tensor.device}"
            )


def instantiate_tokenizer_replica(
    tokenizer_config: Any,
    device: torch.device,
) -> Any:
    """在目标GPU上调用Cosmos原生lazy-config构造一个完整VAE wrapper。"""
    from cosmos_policy._src.imaginaire.lazy_config import (
        instantiate as lazy_instantiate,
    )

    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError(f"VAE replica device must be an indexed CUDA device, got {device}")

    replica_config = copy.deepcopy(tokenizer_config)
    with torch.cuda.device(device):
        replica = lazy_instantiate(replica_config)
        torch.cuda.synchronize(device)
    validate_tokenizer_replica(replica, device)
    return replica


class VAEReplicaPool:
    """每张编码卡持有一个完整、可跨episode复用的Wan VAE wrapper。"""

    def __init__(
        self,
        *,
        policy: Any,
        tokenizer_config: Any,
        devices: list[str],
        encode_batch_size: int,
        tokenizer_factory: TokenizerFactory = instantiate_tokenizer_replica,
    ) -> None:
        if not devices:
            raise ValueError("VAE replica pool requires at least one device")
        if encode_batch_size < 1:
            raise ValueError("encode_batch_size must be >= 1")

        self.policy = policy
        self.devices = tuple(str(torch.device(device)) for device in devices)
        self.encode_batch_size = encode_batch_size
        self.replicas: dict[str, Any] = {}

        primary_tokenizer = policy.tokenizer
        primary_device = str(tokenizer_device(primary_tokenizer))
        if primary_device not in self.devices:
            raise RuntimeError(
                f"primary VAE is on {primary_device}, but requested devices are "
                f"{list(self.devices)}"
            )
        validate_tokenizer_replica(primary_tokenizer, torch.device(primary_device))
        self.replicas[primary_device] = primary_tokenizer
        print(f"[VAE] replica ready: {primary_device} (reused primary wrapper)")

        # 顺序初始化，避免并发读取checkpoint及临时模型同时占用大量CPU/GPU内存。
        for device_str in self.devices:
            if device_str in self.replicas:
                continue
            device = torch.device(device_str)
            print(f"[VAE] initializing complete wrapper on {device_str}...")
            replica = tokenizer_factory(tokenizer_config, device)
            validate_tokenizer_replica(replica, device)
            self.replicas[device_str] = replica
            print(f"[VAE] replica ready: {device_str}")

    @property
    def signature(self) -> tuple[tuple[str, ...], int]:
        return self.devices, self.encode_batch_size

    def encode(self, batch_cpu: torch.Tensor, *, sigma_data: float) -> torch.Tensor:
        """按batch第一维分片到多卡，并按原顺序拼回CPU latent。"""
        if batch_cpu.device.type != "cpu":
            raise ValueError(f"VAE replica pool expects a CPU batch, got {batch_cpu.device}")
        if batch_cpu.shape[0] < 1:
            raise ValueError("VAE replica pool cannot encode an empty batch")

        active_devices = self.devices[: min(len(self.devices), batch_cpu.shape[0])]
        shard_lengths = self.policy._shard_lengths(
            batch_cpu.shape[0], len(active_devices)
        )
        shards: list[torch.Tensor] = []
        offset = 0
        for length in shard_lengths:
            shards.append(batch_cpu[offset : offset + length])
            offset += length

        def _encode_shard(item: tuple[str, torch.Tensor]) -> torch.Tensor:
            device_str, shard = item
            device = torch.device(device_str)
            tokenizer = self.replicas[device_str]
            outputs: list[torch.Tensor] = []
            context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), context:
                for start in range(0, shard.shape[0], self.encode_batch_size):
                    batch_gpu = shard[start : start + self.encode_batch_size].to(
                        device, non_blocking=True
                    )
                    latent_gpu = (
                        tokenizer.encode(batch_gpu) * sigma_data
                    ).contiguous().float()
                    outputs.append(latent_gpu.cpu())
                    del batch_gpu, latent_gpu
            return torch.cat(outputs, dim=0)

        with ThreadPoolExecutor(max_workers=len(active_devices)) as executor:
            parts = list(executor.map(_encode_shard, zip(active_devices, shards)))

        latent = torch.cat(parts, dim=0)
        if latent.shape[0] != batch_cpu.shape[0]:
            raise RuntimeError(
                f"VAE latent batch length {latent.shape[0]} does not match "
                f"input batch length {batch_cpu.shape[0]}"
            )
        return latent


def get_or_create_vae_replica_pool(
    *,
    policy: Any,
    tokenizer_config: Any,
    encode_world_size: int,
    encode_device_ids: tuple[int, ...] | None,
    encode_batch_size: int,
    tokenizer_factory: TokenizerFactory = instantiate_tokenizer_replica,
) -> VAEReplicaPool:
    """按设备集合和batch size创建或复用Policy上的VAE池。"""
    resolver = getattr(policy, "_resolve_encode_devices", None)
    if resolver is None:
        raise RuntimeError("CosmosPolicy does not provide _resolve_encode_devices")
    devices = resolver(
        encode_world_size,
        list(encode_device_ids) if encode_device_ids is not None else None,
        encode_batch_size,
    )
    if not devices:
        raise RuntimeError("no CUDA devices are available for multi-GPU VAE encode")

    requested_signature = (
        tuple(str(torch.device(device)) for device in devices),
        encode_batch_size,
    )
    existing = getattr(policy, _POOL_ATTRIBUTE, None)
    if existing is not None:
        if existing.signature != requested_signature:
            raise RuntimeError(
                f"cached VAE replica pool signature {existing.signature} does not "
                f"match requested {requested_signature}"
            )
        return existing

    pool = VAEReplicaPool(
        policy=policy,
        tokenizer_config=tokenizer_config,
        devices=devices,
        encode_batch_size=encode_batch_size,
        tokenizer_factory=tokenizer_factory,
    )
    setattr(policy, _POOL_ATTRIBUTE, pool)
    return pool
