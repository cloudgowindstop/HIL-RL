"""Official Cosmos model loading and pre-encoded latent EDM forward."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

from .action_spec import get_action_spec
from .evaluation.policy import action_metrics
from .losses import edm_losses


def checkpoint_hash(path: Path) -> str:
    if not path.is_file():
        return "directory"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_action_chunk(
    latent: torch.Tensor,
    action_index: int = 4,
    action_dimension: int = 20,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Recover a repeated action chunk from one injected latent frame."""
    frame = latent[:, :, action_index].reshape(latent.shape[0], -1)
    elements = int(chunk_size) * int(action_dimension)
    chunks = frame.shape[1] // elements
    if chunks < 1:
        raise ValueError(f"action latent has {frame.shape[1]} elements, requires at least {elements}")
    return frame[:, : chunks * elements].reshape(
        latent.shape[0], chunks, chunk_size, action_dimension
    ).mean(dim=1)


def load_model(config: dict[str, Any], device: torch.device):
    """Use official loader; never silently continue with random weights."""
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model_config = config["model"]
    requested_lr = float(config["training"]["learning_rate"])
    path = Path(model_config["checkpoint_path"]).expanduser().resolve()
    tokenizer_path = Path(model_config["tokenizer_path"]).expanduser().resolve()
    print(f"[MODEL] experiment={model_config['experiment']}")
    print(f"[MODEL] checkpoint={path}")
    print(f"[MODEL] tokenizer={tokenizer_path}")
    digest = config.get("_base_checkpoint_sha256") or checkpoint_hash(path)
    print(f"[MODEL] checkpoint_sha256={digest}")
    model, official_config = load_model_from_checkpoint(
        experiment_name=model_config["experiment"],
        s3_checkpoint_dir=str(path),
        config_file=model_config.get("config_file", "cosmos_policy/config/config.py"),
        load_ema_to_reg=False,
        instantiate_ema=False,
        to_device=str(device),
        experiment_opts=[
            f"model.config.tokenizer.vae_pth={tokenizer_path}",
            f"optimizer.lr={requested_lr}",
        ],
    )
    model.train()
    return model, official_config


class TextEmbeddingProvider:
    def __init__(self, path: str, *, worker_id: int, expected_tasks: set[str]):
        from cosmos_policy.experiments.robot import cosmos_utils

        self.path = path
        self._cosmos_utils = cosmos_utils
        cosmos_utils.init_t5_text_embeddings_cache(path, worker_id=worker_id)
        missing = sorted(expected_tasks - set(cosmos_utils.t5_text_embeddings_cache))
        if missing:
            raise ValueError(
                f"T5 cache 缺少 {len(missing)} 个 task key，禁止训练时在线计算: {missing}"
            )

    def get(self, tasks: list[str], device: torch.device) -> torch.Tensor:
        values = [self._cosmos_utils.t5_text_embeddings_cache[task].squeeze(0) for task in tasks]
        return torch.stack(values).to(device=device, dtype=torch.bfloat16)


def build_preencoded_condition(
    model,
    x0: torch.Tensor,
    temporal_condition_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    conditioner_data_type,
):
    """Build the official condition object for an injected latent sequence."""
    batch_size, _, _, height, width = x0.shape
    conditioner_batch = {
        "fps": torch.full((batch_size,), 16, device=x0.device, dtype=torch.bfloat16),
        "padding_mask": torch.zeros(
            (batch_size, 1, height * 8, width * 8),
            device=x0.device,
            dtype=torch.bfloat16,
        ),
        "t5_text_embeddings": text_embeddings,
    }
    condition = model.conditioner(conditioner_batch).edit_data_type(conditioner_data_type)
    condition = condition.set_video_condition(
        gt_frames=x0,
        random_min_num_conditional_frames=4,
        random_max_num_conditional_frames=4,
        num_conditional_frames=4,
        conditional_frames_probs=None,
    )
    condition.condition_video_input_mask_B_C_T_H_W = temporal_condition_mask[
        :, None, :, None, None
    ].expand(-1, 1, -1, height, width).to(
        condition.condition_video_input_mask_B_C_T_H_W.dtype
    )
    return condition


def preencoded_edm_forward(
    model,
    batch: dict[str, Any],
    temporal_condition_mask: torch.Tensor,
    temporal_loss_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    action_min: torch.Tensor | None = None,
    action_max: torch.Tensor | None = None,
    sample_types: torch.Tensor | None = None,
    action_metric_mask: torch.Tensor | None = None,
    action_encoding: str = "cosmos_rotation_6d",
    action_dimension: int = 20,
    action_chunk_size: int = 16,
    euler_rotation_scale: float = 1.0,
    fixed_sigma: float | None = None,
    loss_reduction: str = "masked_mean",
    conditioner_data_type: Any | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """EDM forward matching official model, skipping VAE because input is latent."""
    if conditioner_data_type is None:
        from cosmos_policy._src.predict2.conditioner import DataType

        conditioner_data_type = DataType.VIDEO

    x0 = batch["video"].to(**model.tensor_kwargs)
    batch_size = x0.shape[0]
    condition = build_preencoded_condition(
        model, x0, temporal_condition_mask, text_embeddings, conditioner_data_type
    )
    if fixed_sigma is None:
        sigma, epsilon = model.draw_training_sigma_and_epsilon(x0.size(), condition)
    else:
        sigma = torch.full(
            (batch_size, x0.shape[2]),
            float(fixed_sigma),
            device=x0.device,
            dtype=torch.float32,
        )
        epsilon = torch.randn_like(x0)
    active_mask = (temporal_condition_mask | temporal_loss_mask)[:, None, :, None, None]
    diffusion_x0 = torch.where(active_mask, x0, torch.zeros_like(x0))
    # An unused slot must carry pure noise, not a noisy copy of ground truth.
    # This matters for inverse dynamics, where value slot 8 is intentionally absent.
    mean, std = model.sde.marginal_prob(diffusion_x0, sigma)
    noisy = mean + epsilon * rearrange(std, "b t -> b 1 t 1 1")
    prediction = model.denoise(noisy, sigma, condition).x0
    sigma_weights = model.get_per_sigma_loss_weights(sigma=sigma)
    loss, metrics = edm_losses(
        prediction,
        x0,
        sigma_weights,
        temporal_loss_mask,
        sample_types=sample_types,
        reduction=loss_reduction,
    )
    if action_min is not None and action_max is not None:
        spec = get_action_spec(action_encoding)
        if action_dimension != spec.dimension:
            raise ValueError(
                f"{action_encoding} action_dimension must be {spec.dimension}, got {action_dimension}"
            )
        predicted_action = extract_action_chunk(
            prediction, action_dimension=action_dimension, chunk_size=action_chunk_size
        )
        target_action = extract_action_chunk(
            x0, action_dimension=action_dimension, chunk_size=action_chunk_size
        )
        minimum = action_min.view(1, 1, action_dimension)
        maximum = action_max.view(1, 1, action_dimension)
        predicted_action = 0.5 * (predicted_action + 1) * (maximum - minimum) + minimum
        target_action = 0.5 * (target_action + 1) * (maximum - minimum) + minimum
        if action_metric_mask is not None:
            predicted_action = predicted_action[action_metric_mask]
            target_action = target_action[action_metric_mask]
        gripper_indices = spec.gripper_indices(number_of_arms=2)
        thresholds = tuple(
            float(0.5 * (action_min[index] + action_max[index]))
            for index in gripper_indices
        )
        metrics.update(
            {
                name: value.detach()
                for name, value in action_metrics(
                    predicted_action,
                    target_action,
                    gripper_thresholds=thresholds,
                    action_encoding=action_encoding,
                    euler_rotation_scale=euler_rotation_scale,
                ).items()
            }
        )
    value_valid = temporal_loss_mask[:, 8]
    metrics["value_scalar_mae"] = torch.zeros((), device=x0.device)
    metrics["value_scalar_mse"] = torch.zeros((), device=x0.device)
    predicted_value = prediction[:, :, 8].flatten(1).mean(1)
    if value_valid.any():
        target_value = x0[:, :, 8].flatten(1).mean(1)[value_valid]
        metrics["value_scalar_mae"] = (
            (predicted_value[value_valid] - target_value).abs().mean().detach()
        )
        metrics["value_scalar_mse"] = (
            (predicted_value[value_valid] - target_value).square().mean().detach()
        )
    if "value" in batch:
        metrics["value_rank_pred"] = predicted_value.detach()
        metrics["value_rank_target"] = batch["value"].to(
            device=predicted_value.device, dtype=torch.float32
        ).reshape(-1).detach()
        metrics["value_rank_valid"] = value_valid.detach()
    return loss, metrics
