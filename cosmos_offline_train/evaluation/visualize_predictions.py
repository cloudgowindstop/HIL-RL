"""World-conditioned RGB visualization for frozen visualization samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..checkpoint import load_checkpoint_for_evaluation
from ..config import ConfigError, load_config
from ..dataset import CosmosParquetDataset, EpisodeRecord
from ..distributed import cleanup, initialize
from ..masks import WORLD, temporal_masks
from ..model import TextEmbeddingProvider, checkpoint_hash, load_model
from .rgb_layout import (
    FUTURE_CAMERA_NAMES,
    contact_sheet,
    flatten_clip_frames,
    future_camera_views,
    slice_bounds,
)
from .sampling import decode_generated_video, sample_preencoded
from .video import abs_error_image, psnr, ssim


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def load_frozen_sample_specs(path: Path, encoding: str | None = None) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and payload.get("samples"):
        samples = list(payload["samples"])
        if encoding:
            samples = [item for item in samples if item.get("encoding", encoding) == encoding]
        if not samples:
            raise ValueError(f"no visualization samples for encoding={encoding}: {path}")
        return samples
    if not isinstance(payload, dict) or not payload.get("episodes"):
        raise ValueError(f"visualization list must contain samples or episodes: {path}")
    if not encoding:
        raise ValueError("protocol visualization_samples.json requires --encoding")
    samples: list[dict[str, Any]] = []
    for episode in payload["episodes"]:
        paths = episode.get("episode_paths") or {}
        episode_path = paths.get(encoding)
        if not episode_path:
            raise ValueError(f"{episode.get('source_id')} missing episode_paths.{encoding}")
        for fraction, row_index in zip(
            payload.get("row_fractions", [None] * len(episode["row_indices"])),
            episode["row_indices"],
        ):
            samples.append(
                {
                    "sample_id": f"{episode_path}:{row_index}",
                    "source_id": episode["source_id"],
                    "episode_path": episode_path,
                    "row_index": int(row_index),
                    "row_fraction": fraction,
                    "rows": episode["rows"],
                    "encoding": encoding,
                    "task": episode.get("task", ""),
                    "outcome": episode.get("outcome", "success"),
                }
            )
    return samples


def require_clean_restore(sample: dict[str, Any], spec: dict[str, Any]) -> torch.Tensor:
    clean = sample.get("clean_restore_latent")
    if clean is None:
        raise ValueError(
            f"{spec['episode_path']}:{spec['row_index']} has no clean_restore_latent; "
            "refuse to decode injected Parquet latent"
        )
    return clean


def load_parquet_sample(spec: dict[str, Any], action_dimension: int) -> dict[str, Any]:
    episode_path = Path(spec["episode_path"]).expanduser().resolve()
    dataset_root = episode_path.parents[2]
    task = str(spec.get("task") or "").strip()
    if not task:
        metadata_path = dataset_root / "cosmos_dataset_metadata.json"
        if metadata_path.is_file():
            task = str(load_json(metadata_path).get("task_description") or "")
    record = EpisodeRecord(
        path=str(episode_path),
        dataset_root=str(dataset_root),
        group=str(spec.get("source_id", "")).split("/", 1)[0],
        task=task or "unknown",
        outcome=str(spec.get("outcome") or "success"),
        rows=int(spec["rows"]),
        source_id=str(spec.get("source_id", "")),
    )
    dataset = CosmosParquetDataset([record], action_dimension=action_dimension)
    sample = dataset[int(spec["row_index"])]
    require_clean_restore(sample, spec)
    return sample


def sample_stem(spec: dict[str, Any]) -> str:
    source = str(spec.get("source_id") or "sample").replace("/", "_")
    return f"{source}_row{int(spec['row_index'])}"


def _to_uint8_image(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.asarray(np.rint(array * 255.0), dtype=np.uint8)


def write_png(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_image(image), mode="RGB").save(path)


def write_mp4(path: Path, clip: torch.Tensor, fps: int = 4) -> None:
    """Write [1,C,T,H,W] RGB in [0,1] as mp4."""
    if clip.ndim != 5 or clip.shape[0] != 1:
        raise ValueError(f"mp4 clip must be [1,C,T,H,W], got {tuple(clip.shape)}")
    frames = (
        clip[0]
        .detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 3, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    from torchvision.io import write_video

    write_video(str(path), frames, fps=fps)


def future_metrics(
    prediction_views: dict[str, torch.Tensor],
    target_views: dict[str, torch.Tensor],
) -> dict[str, float]:
    values: dict[str, float] = {}
    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    for name in FUTURE_CAMERA_NAMES:
        pred_frames = flatten_clip_frames(prediction_views[name])
        target_frames = flatten_clip_frames(target_views[name])
        values[f"{name}_psnr"] = float(psnr(pred_frames, target_frames).mean().item())
        values[f"{name}_ssim"] = float(ssim(pred_frames, target_frames).mean().item())
        all_pred.append(pred_frames)
        all_target.append(target_frames)
    values["future_cameras_psnr"] = float(psnr(torch.cat(all_pred), torch.cat(all_target)).mean().item())
    values["future_cameras_ssim"] = float(ssim(torch.cat(all_pred), torch.cat(all_target)).mean().item())
    return values


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [key for key in rows[0] if key.endswith("_psnr") or key.endswith("_ssim")]
    return {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in keys}


@torch.no_grad()
def visualize_one(
    *,
    model,
    embeddings: TextEmbeddingProvider,
    spec: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
    num_steps: int,
    seed: int,
    guidance: float,
    output_dir: Path,
) -> dict[str, Any]:
    video = sample["video"].unsqueeze(0).to(device=device)
    clean = sample["clean_restore_latent"].unsqueeze(0).to(device=device)
    types = torch.tensor([WORLD], device=device)
    condition_mask, _ = temporal_masks(types, time=video.shape[2])
    text = embeddings.get([str(sample["task"])], device)
    generated = sample_preencoded(
        model,
        video,
        condition_mask,
        text,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )
    ground_truth_rgb = decode_generated_video(model, video, clean_restore_latent=clean)
    prediction_rgb = decode_generated_video(model, generated, clean_restore_latent=clean)
    target_views = future_camera_views(ground_truth_rgb)
    pred_views = future_camera_views(prediction_rgb)
    error_views = {
        name: abs_error_image(pred_views[name], target_views[name], normalize=True)
        for name in FUTURE_CAMERA_NAMES
    }
    metrics = future_metrics(pred_views, target_views)
    stem = sample_stem(spec)
    sample_dir = output_dir / "samples" / stem
    sheet = contact_sheet(target_views, pred_views, error_views)
    write_png(sample_dir / "contact.png", sheet)
    write_mp4(
        sample_dir / "future.mp4",
        torch.cat(
            [
                target_views["future_wrist"],
                pred_views["future_wrist"],
                target_views["future_primary"],
                pred_views["future_primary"],
            ],
            dim=4,
        ),
    )
    metrics.update(
        {
            "sample_id": spec["sample_id"],
            "source_id": spec.get("source_id"),
            "row_index": int(spec["row_index"]),
            "row_fraction": spec.get("row_fraction"),
            "contact_png": str(sample_dir / "contact.png"),
            "future_mp4": str(sample_dir / "future.mp4"),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World RGB visualization for frozen samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--encoding")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--base", action="store_true")
    parser.add_argument("--mode", default="world", choices=("world",))
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def _required_file(path_value: Any, name: str) -> Path:
    if not path_value:
        raise ConfigError(f"{name} 未配置")
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"{name} 不存在: {path}")
    return path


def main() -> None:
    args = parse_args()
    if args.num_steps < 1:
        raise SystemExit("--num-steps must be >= 1")
    if args.guidance != 1.0:
        raise SystemExit("World visualization requires --guidance 1.0")
    config = load_config(args.config)
    data = config["data"]
    model_cfg = config["model"]
    evaluation = config.get("evaluation", {})
    _required_file(model_cfg.get("checkpoint_path"), "model.checkpoint_path")
    _required_file(model_cfg.get("tokenizer_path"), "model.tokenizer_path")
    t5_path = _required_file(data.get("t5_embeddings_path"), "data.t5_embeddings_path")
    samples_path = Path(
        args.samples or evaluation.get("visualization_samples") or ""
    ).expanduser()
    if not samples_path.is_file():
        raise SystemExit("visualization sample list missing; pass --samples")
    encoding = args.encoding or data.get("action_encoding")
    specs = load_frozen_sample_specs(samples_path, encoding=encoding)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise SystemExit("--max-samples must be >= 1")
        specs = specs[: args.max_samples]
    output_dir = Path(
        args.output_dir or config.get("runtime", {}).get("output_dir") or ""
    ).expanduser().resolve()
    if not output_dir.as_posix():
        raise SystemExit("pass --output-dir or set runtime.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else evaluation.get("noise_seed", 20260820))
    action_dimension = int(data["action_dimension"])
    parquet_samples = [load_parquet_sample(spec, action_dimension) for spec in specs]
    context = None
    try:
        context = initialize()
        model, _ = load_model(config, context.device)
        model.eval()
        checkpoint_path = Path(model_cfg["checkpoint_path"]).expanduser().resolve()
        checkpoint_name = "base"
        state = {"epoch": 0, "global_step": 0}
        if not args.base:
            checkpoint_path = Path(args.checkpoint).expanduser().resolve()
            checkpoint_name = str(checkpoint_path)
            state = load_checkpoint_for_evaluation(
                checkpoint_path, model=model, device=context.device
            )
        tasks = {str(sample["task"]) for sample in parquet_samples}
        embeddings = TextEmbeddingProvider(
            str(t5_path), worker_id=context.local_rank, expected_tasks=tasks
        )
        rows: list[dict[str, Any]] = []
        for spec, sample in zip(specs, parquet_samples):
            print(
                f"[VIZ] {spec.get('source_id')} row={spec['row_index']} "
                f"mode={args.mode} steps={args.num_steps} seed={seed}",
                flush=True,
            )
            rows.append(
                visualize_one(
                    model=model,
                    embeddings=embeddings,
                    spec=spec,
                    sample=sample,
                    device=context.device,
                    num_steps=int(args.num_steps),
                    seed=seed,
                    guidance=float(args.guidance),
                    output_dir=output_dir,
                )
            )
        protocol = {
            "mode": args.mode,
            "num_steps": int(args.num_steps),
            "guidance": float(args.guidance),
            "seed": seed,
            "solver_option": "2ab",
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": checkpoint_hash(checkpoint_path),
            "checkpoint_step": int(state["global_step"]),
            "config": config.get("_config_path"),
            "samples_path": str(samples_path.resolve()),
            "encoding": encoding,
            "action_encoding": data.get("action_encoding"),
            "action_dimension": action_dimension,
            "sample_ids": [spec["sample_id"] for spec in specs],
            "future_slices": {
                name: list(slice_bounds(name)) for name in FUTURE_CAMERA_NAMES
            },
        }
        summary = {
            "sample_count": len(rows),
            "metrics": mean_metrics(rows),
        }
        (output_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / "per_sample_metrics.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[VIZ] wrote {output_dir / 'summary.json'}", flush=True)
    finally:
        if context is not None:
            cleanup()


if __name__ == "__main__":
    main()
