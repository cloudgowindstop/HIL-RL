"""RGB reconstruction and perceptual metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def abs_error_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-pixel absolute error; optionally scale each batch item by its max."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"abs_error_image shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    error = (prediction.float() - target.float()).abs()
    if normalize:
        reduce_dims = tuple(range(1, error.ndim))
        peak = error.amax(dim=reduce_dims, keepdim=True)
        error = error / peak.clamp_min(torch.finfo(error.dtype).eps)
    return error.clamp(0.0, 1.0)


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR for tensors shaped [N,C,H,W]."""
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
    maximum = torch.tensor(data_range * data_range, device=mse.device)
    return 10.0 * torch.log10(maximum / mse.clamp_min(torch.finfo(mse.dtype).eps))


def _gaussian_window(channels: int, device, dtype, size: int = 11, sigma: float = 1.5):
    coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coordinates * coordinates) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    window = kernel[:, None] * kernel[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image local-window SSIM for [N,C,H,W]."""
    prediction, target = prediction.float(), target.float()
    channels = prediction.shape[1]
    window = _gaussian_window(channels, prediction.device, prediction.dtype)
    padding = window.shape[-1] // 2
    mean_x = functional.conv2d(prediction, window, padding=padding, groups=channels)
    mean_y = functional.conv2d(target, window, padding=padding, groups=channels)
    mean_x2, mean_y2 = mean_x.square(), mean_y.square()
    covariance = functional.conv2d(prediction * target, window, padding=padding, groups=channels) - mean_x * mean_y
    variance_x = functional.conv2d(prediction.square(), window, padding=padding, groups=channels) - mean_x2
    variance_y = functional.conv2d(target.square(), window, padding=padding, groups=channels) - mean_y2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x2 + mean_y2 + c1) * (variance_x + variance_y + c2)
    ).clamp_min(torch.finfo(prediction.dtype).tiny)
    return score.flatten(1).mean(1)


class LPIPSMetric:
    """Optional LPIPS wrapper; dependency and weights must be installed explicitly."""

    def __init__(self, device: torch.device):
        try:
            import lpips
        except ModuleNotFoundError as error:
            raise RuntimeError("LPIPS unavailable: install the 'lpips' package") from error
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects RGB in [-1, 1].
        return self.model(prediction * 2 - 1, target * 2 - 1).flatten()
