#!/usr/bin/env python3
"""Compatibility CLI for full raw-HDF5 to Cosmos LeRobot conversion.

The implementation lives in data_convert_refactored. This file stays as the
single user-facing Python entry point and preserves old underscore arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
for path in (PROJECT_DIR, PROJECT_DIR / "lerobot" / "src", WORKSPACE_DIR / "cosmos-policy"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_convert_refactored.conditioning.camera import CameraState
from data_convert_refactored.conditioning.episode_labeling import EpisodeOutcome
from data_convert_refactored.config import (
    ActionEncoding,
    ActionScale,
    ActionSource,
    ConversionConfig,
    StatsMode,
)
from data_convert_refactored.encoding.multi_gpu_vae import resolve_encode_settings
from data_convert_refactored.pipeline import ConversionPipeline


def _path(value: str | Path | None) -> Path | None:
    return None if not value else Path(value).expanduser().resolve()


def parse_args() -> tuple[ConversionConfig, bool]:
    parser = argparse.ArgumentParser(description="Raw HDF5 -> Cosmos LeRobot conversion")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", "--task-description", dest="task", required=True)
    parser.add_argument(
        "--episode-outcome",
        "--episode_outcome",
        dest="episode_outcome",
        required=True,
        choices=[value.value for value in EpisodeOutcome],
    )
    parser.add_argument(
        "--stats-mode",
        "--stats_mode",
        dest="stats_mode",
        choices=[value.value for value in StatsMode],
        default=StatsMode.OFFICIAL.value,
    )
    parser.add_argument(
        "--dataset-stats",
        "--dataset_stats",
        dest="dataset_stats",
        type=Path,
        help=(
            "Stats path. With --stats-mode generated this is generated stats; "
            "with --stats-mode official this is treated as official stats for old commands."
        ),
    )
    parser.add_argument(
        "--official-dataset-stats",
        "--official_dataset_stats",
        dest="official_dataset_stats",
        type=Path,
    )
    parser.add_argument(
        "--action-encoding",
        "--action_encoding",
        dest="action_encoding",
        choices=[value.value for value in ActionEncoding],
        default=ActionEncoding.ROTATION_6D.value,
    )
    parser.add_argument(
        "--action-source",
        "--action_source",
        dest="action_source",
        choices=[value.value for value in ActionSource],
        default=ActionSource.PUPPET_NEXT_FRAME.value,
    )
    parser.add_argument("--translation-scale", "--translation_scale", dest="translation_scale", type=float, default=0.02)
    parser.add_argument("--rotation-scale", "--rotation_scale", dest="rotation_scale", type=float, default=0.06)
    parser.add_argument("--gripper-scale", "--gripper_scale", dest="gripper_scale", type=float, default=1.0)
    parser.add_argument("--kinematics-config", "--kinematics_config", dest="kinematics_config", type=Path)
    parser.add_argument("--fk-max-position-error-m", "--fk_max_position_error_m", dest="fk_max_position_error_m", type=float, default=0.005)
    parser.add_argument("--fk-max-rotation-error-deg", "--fk_max_rotation_error_deg", dest="fk_max_rotation_error_deg", type=float, default=1.0)
    parser.add_argument("--t5-embeddings", "--t5_embeddings", dest="t5_embeddings", type=Path)
    parser.add_argument("--skip-t5", "--skip_t5", dest="skip_t5", action="store_true")
    parser.add_argument(
        "--episode-manifest",
        "--episode_manifest",
        dest="episode_manifest",
        type=Path,
        help="JSON source list used to freeze identical episodes and ordering across encodings",
    )
    parser.add_argument("--max-episodes", "--max_episodes", dest="max_episodes", type=int, default=0)
    parser.add_argument("--encode-batch-size", "--encode_batch_size", dest="encode_batch_size", type=int, default=16)
    parser.add_argument("--encode-world-size", "--encode_world_size", dest="encode_world_size", type=int)
    parser.add_argument("--encode-device-ids", "--encode_device_ids", dest="encode_device_ids")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument(
        "--save-clean-restore-latent",
        "--save_clean_restore_latent",
        dest="save_clean_restore_latent",
        action="store_true",
        help="Save compact clean VAE slots required for safe RGB evaluation.",
    )
    parser.add_argument(
        "--cosmos-config",
        "--cosmos_config",
        dest="cosmos_config",
        type=Path,
        default=Path("train_config_cosmos.json"),
    )
    parser.add_argument(
        "--camera-state",
        "--camera_state",
        dest="camera_state",
        choices=[value.value for value in CameraState],
        default=CameraState.NORMAL.value,
    )
    parser.add_argument(
        "--wrist-crop-mode",
        "--wrist_crop_mode",
        dest="wrist_crop_mode",
        choices=("center_width", "bottom", "none"),
    )
    parser.add_argument(
        "--wrist-crop-fraction",
        "--wrist_crop_fraction",
        "--wrist-crop-bottom",
        dest="wrist_crop_fraction",
        type=float,
    )
    parser.add_argument("--wrist-left-center-offset-x", "--wrist_left_center_offset_x", dest="wrist_left_center_offset_x", type=int)
    parser.add_argument("--wrist-right-center-offset-x", "--wrist_right_center_offset_x", dest="wrist_right_center_offset_x", type=int)
    parser.add_argument("--head-crop-top-pixels", "--head_crop_top_pixels", dest="head_crop_top_pixels", type=int)
    parser.add_argument("--monitor-memory", "--monitor_memory", dest="monitor_memory", action="store_true")
    parser.add_argument("--memory-sample-interval", "--memory_sample_interval", dest="memory_sample_interval", type=float, default=2.0)
    parser.add_argument("--preflight-only", "--preflight_only", dest="preflight_only", action="store_true")
    parser.add_argument(
        "--no-resume",
        "--no_resume",
        dest="no_resume",
        action="store_true",
        help="Accepted for old commands; refactored pipeline is idempotent by output files.",
    )
    args = parser.parse_args()

    stats_mode = StatsMode(args.stats_mode)
    dataset_stats = None
    official_dataset_stats = None
    if stats_mode is StatsMode.GENERATED:
        dataset_stats = _path(args.dataset_stats)
        if args.official_dataset_stats:
            parser.error("--official-dataset-stats is only valid with --stats-mode official")
    else:
        official_dataset_stats = _path(args.official_dataset_stats or args.dataset_stats)
    if dataset_stats is None and official_dataset_stats is None:
        flag = "--dataset-stats" if stats_mode is StatsMode.GENERATED else "--official-dataset-stats"
        parser.error(f"{flag} is required")

    encode_world_size, encode_device_ids = resolve_encode_settings(
        args.encode_world_size, args.encode_device_ids
    )
    config = ConversionConfig(
        input_dir=args.input.expanduser().resolve(),
        output_dir=args.output.expanduser().resolve(),
        task=args.task,
        episode_outcome=EpisodeOutcome(args.episode_outcome),
        stats_mode=stats_mode,
        dataset_stats=dataset_stats,
        official_dataset_stats=official_dataset_stats,
        action_encoding=ActionEncoding(args.action_encoding),
        action_source=ActionSource(args.action_source),
        action_scale=ActionScale(
            translation_m=args.translation_scale,
            rotation_rad=args.rotation_scale,
            gripper=args.gripper_scale,
        ),
        kinematics_config=_path(args.kinematics_config),
        t5_embeddings=_path(args.t5_embeddings),
        skip_t5=args.skip_t5,
        episode_manifest=_path(args.episode_manifest),
        max_episodes=args.max_episodes,
        encode_batch_size=args.encode_batch_size,
        encode_world_size=encode_world_size,
        encode_device_ids=encode_device_ids,
        batch_size=args.batch_size,
        save_clean_restore_latent=args.save_clean_restore_latent,
        cosmos_config_path=args.cosmos_config.expanduser(),
        camera_state=CameraState(args.camera_state),
        wrist_crop_mode=args.wrist_crop_mode,
        wrist_crop_fraction=args.wrist_crop_fraction,
        wrist_left_center_offset_x=args.wrist_left_center_offset_x,
        wrist_right_center_offset_x=args.wrist_right_center_offset_x,
        head_crop_top_pixels=args.head_crop_top_pixels,
        fk_max_position_error_m=args.fk_max_position_error_m,
        fk_max_rotation_error_deg=args.fk_max_rotation_error_deg,
        monitor_memory=args.monitor_memory,
        memory_sample_interval=args.memory_sample_interval,
    )
    return config, args.preflight_only


def main() -> None:
    config, preflight_only = parse_args()
    ConversionPipeline(config).run(preflight_only=preflight_only)


if __name__ == "__main__":
    main()
