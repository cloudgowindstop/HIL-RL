"""Full diffusion sampling and safe VAE decoding for pre-encoded data."""

from __future__ import annotations

import torch

from ..model import build_preencoded_condition


@torch.no_grad()
def sample_preencoded(
    model,
    x0: torch.Tensor,
    condition_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    num_steps: int = 10,
    guidance: float = 1.0,
    seed: int = 1,
    solver_option: str = "2ab",
    conditioner_data_type=None,
) -> torch.Tensor:
    """Generate a complete latent sequence from noise with inpainting conditions."""
    if conditioner_data_type is None:
        from cosmos_policy._src.predict2.conditioner import DataType

        conditioner_data_type = DataType.VIDEO
    x0 = x0.to(**model.tensor_kwargs)
    condition = build_preencoded_condition(
        model, x0, condition_mask, text_embeddings, conditioner_data_type
    )
    uncondition = build_preencoded_condition(
        model, x0, condition_mask, torch.zeros_like(text_embeddings), conditioner_data_type
    )

    def x0_fn(noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        conditional = model.denoise(noisy, sigma, condition).x0
        if guidance == 1.0:
            return conditional
        unconditional = model.denoise(noisy, sigma, uncondition).x0
        return conditional + guidance * (conditional - unconditional)

    generator = torch.Generator(device=x0.device).manual_seed(seed)
    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=torch.float32)
    initial = noise * float(model.sde.sigma_max)
    return model.sampler(
        x0_fn,
        initial,
        num_steps=num_steps,
        sigma_max=float(model.sde.sigma_max),
        sigma_min=float(model.sde.sigma_min),
        solver_option=solver_option,
    )


@torch.no_grad()
def decode_generated_video(
    model,
    generated_latent: torch.Tensor,
    *,
    clean_restore_latent: torch.Tensor | None,
    restore_indices: tuple[int, ...] = (1, 4, 5, 8),
) -> torch.Tensor:
    """Restore injection slots and decode RGB in [0,1]."""
    if clean_restore_latent is None:
        raise ValueError(
            "clean_restore_latent is required; injected Parquet latent cannot be decoded safely"
        )
    restored = generated_latent.clone()
    if clean_restore_latent.ndim != 5:
        raise ValueError(
            "clean_restore_latent must be [B,C,4,H,W] or [B,C,T,H,W], got "
            f"{tuple(clean_restore_latent.shape)}"
        )
    if clean_restore_latent.shape[2] == len(restore_indices):
        payload = clean_restore_latent
    elif clean_restore_latent.shape[2] == generated_latent.shape[2]:
        payload = clean_restore_latent[:, :, list(restore_indices)]
    else:
        raise ValueError(
            "clean_restore_latent temporal size must equal 4 or generated latent size, got "
            f"{clean_restore_latent.shape[2]}"
        )
    restored[:, :, list(restore_indices)] = payload.to(
        device=restored.device, dtype=restored.dtype
    )
    decoder = getattr(model, "decode", None)
    if decoder is None:
        decoder = model.tokenizer.decode
    return ((decoder(restored).float() + 1.0) * 0.5).clamp(0.0, 1.0)
