"""为离线转换初始化只保留VAE的CosmosPolicy。"""

from __future__ import annotations

from pathlib import Path
import torch

from ..config import CosmosRuntimeConfig


def init_cosmos_policy(
    config: CosmosRuntimeConfig, action_dim: int = 7, proprio_dim: int = 8
):
    """构造与当前D/P维度匹配的Policy schema并加载VAE checkpoint。

    DiT和EMA网络加载后立即删除；单GPU复用主VAE，多GPU额外创建wrapper池。返回的
    `cosmos_config.chunk_size`决定action chunk及future conditioning时间偏移。
    """
    import json

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.cosmos.configuration_cosmos import CosmosConfig
    from lerobot.policies.cosmos.modeling_cosmos import CosmosPolicy

    project_dir = Path(__file__).resolve().parents[2]
    requested_config = Path(config.cosmos_config_path).expanduser()
    cosmos_config_path = str(
        requested_config if requested_config.is_absolute() else project_dir / requested_config
    )
    with open(cosmos_config_path) as file:
        raw = json.load(file)

    chunk_size = raw.get("chunk_size", 16)
    use_jpeg_compression = raw.get("use_jpeg_compression", True)
    trained_with_image_aug = raw.get("trained_with_image_aug", True)
    use_proprio = raw.get("use_proprio", True)
    normalize_proprio = raw.get("normalize_proprio", True)
    flip_images = raw.get("flip_images", False)

    policy_config = CosmosConfig(
        # 这里是Policy运行schema，不是最终Parquet schema。
        input_features={
            "video": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 33, 224, 224)),
            "proprio": PolicyFeature(type=FeatureType.STATE, shape=(proprio_dim,)),
            "future_proprio": PolicyFeature(type=FeatureType.STATE, shape=(proprio_dim,)),
            "value_function_return": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "t5_text_embeddings": PolicyFeature(type=FeatureType.ENV, shape=(512, 1024)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        use_torch_compile=False,
    )

    cosmos_config = policy_config.load_world_config()
    cosmos_config.chunk_size = chunk_size
    cosmos_config.use_jpeg_compression = use_jpeg_compression
    cosmos_config.trained_with_image_aug = trained_with_image_aug
    cosmos_config.use_proprio = use_proprio
    cosmos_config.normalize_proprio = normalize_proprio
    cosmos_config.flip_images = flip_images

    tokenizer_checkpoint = (
        project_dir.parent
        / "cosmos-policy"
        / "cosmos_policy"
        / "models"
        / "Cosmos-Policy-LIBERO-Predict2-2B"
        / "tokenizer"
        / "tokenizer.pth"
    ).resolve()
    if not tokenizer_checkpoint.is_file():
        raise FileNotFoundError(
            "Cosmos VAE tokenizer checkpoint not found at the path resolved from "
            f"the current repository: {tokenizer_checkpoint}"
        )
    cosmos_config.tokenizer.vae_pth = str(tokenizer_checkpoint)
    print(f"[INFO] VAE tokenizer checkpoint: {tokenizer_checkpoint}")

    visible_device_count = torch.cuda.device_count()
    requested_device_ids = (
        list(config.encode_device_ids)
        if config.encode_device_ids is not None
        else list(range(min(config.encode_world_size, visible_device_count)))
    )
    invalid_device_ids = [
        device_id
        for device_id in requested_device_ids
        if device_id < 0 or device_id >= visible_device_count
    ]
    if invalid_device_ids:
        raise ValueError(
            f"encode device ids {invalid_device_ids} are not visible; "
            f"visible CUDA device count is {visible_device_count}"
        )
    primary_device_id = requested_device_ids[0] if requested_device_ids else 0
    primary_device = torch.device(f"cuda:{primary_device_id}")
    torch.cuda.set_device(primary_device)

    print("[INFO] Loading CosmosPolicy model...")
    policy = CosmosPolicy(config=policy_config, cosmos_cfg=cosmos_config)
    if hasattr(policy, "net"):
        # 数据转换只调用tokenizer.encode，不需要扩散DiT权重常驻显存。
        del policy.net
    if hasattr(policy, "net_ema"):
        del policy.net_ema
    policy.eval()
    policy.to(primary_device)
    torch.cuda.empty_cache()
    print(
        "[INFO] CosmosPolicy loaded (DiT unloaded, VAE only); "
        f"primary_device={primary_device}, encode_world_size={config.encode_world_size}, "
        f"encode_device_ids={config.encode_device_ids}"
    )

    if config.encode_world_size > 1 or config.encode_device_ids is not None:
        from .multi_gpu_vae import get_or_create_vae_replica_pool

        get_or_create_vae_replica_pool(
            policy=policy,
            tokenizer_config=cosmos_config.tokenizer,
            encode_world_size=config.encode_world_size,
            encode_device_ids=config.encode_device_ids,
            encode_batch_size=config.encode_batch_size,
        )
        print("[INFO] Multi-GPU VAE replica pool initialized and cached")

    return policy, cosmos_config
