"""Decoded Cosmos RGB frame slices. Copied from convert camera layout."""

from __future__ import annotations

import torch

DECODED_RGB_FRAMES = 33
DECODED_RGB_CHANNELS = 3

# Matches data_convert_refactored/conditioning/camera.py RAW_FRAME_SLICES.
# Copied here so evaluation does not import the conversion package.
RAW_FRAME_SLICES = {
    "current_wrist": slice(5, 9),
    "current_primary": slice(9, 13),
    "future_wrist": slice(21, 25),
    "future_primary": slice(25, 29),
}

FUTURE_CAMERA_NAMES = ("future_wrist", "future_primary")


def slice_bounds(name: str) -> tuple[int, int]:
    frame_slice = RAW_FRAME_SLICES[name]
    return int(frame_slice.start), int(frame_slice.stop)


def require_decoded_rgb(video: torch.Tensor) -> torch.Tensor:
    """Accept [C,T,H,W] or [B,C,T,H,W] and require 33 RGB frames."""
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            f"decoded RGB must be [C,T,H,W] or [B,C,T,H,W], got {tuple(video.shape)}"
        )
    if video.shape[1] != DECODED_RGB_CHANNELS:
        raise ValueError(
            f"decoded RGB channel size must be {DECODED_RGB_CHANNELS}, got {video.shape[1]}"
        )
    if video.shape[2] != DECODED_RGB_FRAMES:
        raise ValueError(
            f"decoded RGB temporal size must be {DECODED_RGB_FRAMES}, got {video.shape[2]}"
        )
    return video


def slice_camera(video: torch.Tensor, name: str) -> torch.Tensor:
    if name not in RAW_FRAME_SLICES:
        raise KeyError(f"unknown camera slice {name!r}")
    video = require_decoded_rgb(video)
    return video[:, :, RAW_FRAME_SLICES[name]]


def future_camera_views(video: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return future wrist/primary clips shaped [B,C,4,H,W]."""
    return {name: slice_camera(video, name) for name in FUTURE_CAMERA_NAMES}


def flatten_clip_frames(clip: torch.Tensor) -> torch.Tensor:
    """Turn [B,C,T,H,W] into [B*T,C,H,W] for per-frame PSNR/SSIM."""
    if clip.ndim != 5:
        raise ValueError(f"clip must be [B,C,T,H,W], got {tuple(clip.shape)}")
    batch, channels, time, height, width = clip.shape
    return clip.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)


def camera_strip(clip: torch.Tensor) -> torch.Tensor:
    """Concatenate time frames along width. Input [B,C,T,H,W], output [C,H,T*W]."""
    if clip.ndim != 5 or clip.shape[0] != 1:
        raise ValueError(f"camera strip expects batch=1 [1,C,T,H,W], got {tuple(clip.shape)}")
    frames = clip[0].unbind(dim=1)
    return torch.cat(frames, dim=2)


def contact_sheet(
    ground_truth: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    error: dict[str, torch.Tensor],
    *,
    cameras: tuple[str, ...] = FUTURE_CAMERA_NAMES,
    gap: int = 4,
) -> torch.Tensor:
    """Build GT | Pred | |Error| rows for each future camera. Output [C,H,W]."""
    gap_value = 0.25
    rows: list[torch.Tensor] = []
    for name in cameras:
        cells = [
            camera_strip(ground_truth[name]),
            camera_strip(prediction[name]),
            camera_strip(error[name]),
        ]
        height = cells[0].shape[1]
        spacer = cells[0].new_full((cells[0].shape[0], height, gap), gap_value)
        row = cells[0]
        for cell in cells[1:]:
            row = torch.cat((row, spacer, cell), dim=2)
        rows.append(row)
    width = rows[0].shape[2]
    spacer = rows[0].new_full((rows[0].shape[0], gap, width), gap_value)
    stacked = rows[0]
    for row in rows[1:]:
        stacked = torch.cat((stacked, spacer, row), dim=1)
    return stacked.clamp(0.0, 1.0)
