"""Measure VAE encode/decode throughput on the target GPU."""

from __future__ import annotations

import torch


@torch.no_grad()
def _benchmark(callable_, argument: torch.Tensor, *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        callable_(argument)
    torch.cuda.synchronize(argument.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(argument.device)
    start.record()
    for _ in range(repeats):
        callable_(argument)
    end.record()
    torch.cuda.synchronize(argument.device)
    seconds = start.elapsed_time(end) / 1000.0
    return {
        "seconds": seconds,
        "calls_per_second": repeats / seconds,
        "items_per_second": repeats * argument.shape[0] / seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated(argument.device) / 2**30,
    }


@torch.no_grad()
def benchmark_vae(
    model,
    *,
    normalized_video: torch.Tensor,
    latent: torch.Tensor,
    warmup: int = 2,
    repeats: int = 10,
) -> dict[str, dict[str, float]]:
    encoder = getattr(model, "encode", None) or model.tokenizer.encode
    decoder = getattr(model, "decode", None) or model.tokenizer.decode
    return {
        "encode": _benchmark(encoder, normalized_video, warmup=warmup, repeats=repeats),
        "decode": _benchmark(decoder, latent, warmup=warmup, repeats=repeats),
    }

