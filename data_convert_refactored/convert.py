#!/usr/bin/env python3
"""HDF5到Cosmos LeRobot转换的Python CLI入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .conditioning.camera import CameraState
from .config import ActionEncoding, ActionScale, ActionSource, ConversionConfig, StatsMode
from .conditioning.episode_labeling import EpisodeOutcome
from .encoding.multi_gpu_vae import resolve_encode_settings
from .pipeline import ConversionPipeline


def parse_config() -> tuple[ConversionConfig, bool]:
    """解析CLI并生成类型化配置；不读取HDF5，也不加载模型。"""
    parser = argparse.ArgumentParser(description="Modular HDF5 to Cosmos Policy conversion")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--episode-outcome",
        required=True,
        choices=[value.value for value in EpisodeOutcome],
        help="Explicit label applied to every episode under --input",
    )
    parser.add_argument(
        "--stats-mode", choices=[value.value for value in StatsMode], required=True
    )
    stats_group = parser.add_mutually_exclusive_group(required=True)
    stats_group.add_argument(
        "--dataset-stats", type=Path, help="Generated shared stats with semantic sidecar"
    )
    stats_group.add_argument(
        "--official-dataset-stats",
        type=Path,
        help="Official collect_data_cosmos.py stats (no semantic sidecar required)",
    )
    parser.add_argument(
        "--action-encoding",
        choices=[value.value for value in ActionEncoding],
        default=ActionEncoding.ROTATION_6D.value,
    )
    parser.add_argument(
        "--action-source",
        choices=[value.value for value in ActionSource],
        default=ActionSource.PUPPET_NEXT_FRAME.value,
    )
    parser.add_argument("--translation-scale", type=float, default=0.02)
    parser.add_argument("--rotation-scale", type=float, default=0.06)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--kinematics-config", type=Path)
    parser.add_argument("--fk-max-position-error-m", type=float, default=0.005)
    parser.add_argument("--fk-max-rotation-error-deg", type=float, default=1.0)
    parser.add_argument("--t5-embeddings", type=Path)
    parser.add_argument("--skip-t5", action="store_true")
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        help="JSON source list used to freeze identical episodes and ordering across encodings",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument(
        "--encode-world-size",
        type=int,
        default=None,
        help="Number of visible GPUs used for VAE encode (or ENCODE_WORLD_SIZE; default 1)",
    )
    parser.add_argument(
        "--encode-device-ids",
        default=None,
        help=(
            "Comma-separated process-local CUDA IDs (or ENCODE_CUDA_DEVICES); "
            "explicit IDs take precedence over world size"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--save-clean-restore-latent",
        action="store_true",
        help=(
            "save the four clean VAE latent slots overwritten by robot conditioning; "
            "required for safe RGB reconstruction metrics"
        ),
    )
    parser.add_argument("--cosmos-config", type=Path, default=Path("train_config_cosmos.json"))
    parser.add_argument(
        "--camera-state",
        choices=[value.value for value in CameraState],
        default=CameraState.NORMAL.value,
        help="Semantic camera preprocessing profile (currently only: normal)",
    )
    parser.add_argument(
        "--wrist-crop-mode",
        choices=("center_width", "bottom", "none"),
        default=None,
    )
    parser.add_argument(
        "--wrist-crop-fraction",
        "--wrist-crop-bottom",
        dest="wrist_crop_fraction",
        type=float,
        default=None,
        help="Crop fraction; --wrist-crop-bottom is a deprecated alias",
    )
    parser.add_argument("--wrist-left-center-offset-x", type=int, default=None)
    parser.add_argument("--wrist-right-center-offset-x", type=int, default=None)
    parser.add_argument("--head-crop-top-pixels", type=int, default=None)
    parser.add_argument(
        "--monitor-memory", action="store_true",
        help="record process/cgroup memory at key stages and on a timer",
    )
    parser.add_argument("--memory-sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--monitor-storage", action="store_true",
        help=(
            "record compact filesystem/process samples to output and /tmp, and "
            "perform real fsync probes before each episode and Parquet save"
        ),
    )
    parser.add_argument("--storage-sample-interval", type=float, default=5.0)
    parser.add_argument("--storage-probe-mib", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate an existing output and continue after its last committed episode",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate HDF5/FK/actions without loading VAE or writing a dataset",
    )
    args = parser.parse_args()
    encode_world_size, encode_device_ids = resolve_encode_settings(
        args.encode_world_size, args.encode_device_ids
    )

    config = ConversionConfig(
        input_dir=args.input.expanduser().resolve(),
        output_dir=args.output.expanduser().resolve(),
        task=args.task,
        episode_outcome=EpisodeOutcome(args.episode_outcome),
        stats_mode=StatsMode(args.stats_mode),
        dataset_stats=(args.dataset_stats.expanduser().resolve() if args.dataset_stats else None),
        official_dataset_stats=(
            args.official_dataset_stats.expanduser().resolve()
            if args.official_dataset_stats
            else None
        ),
        action_encoding=ActionEncoding(args.action_encoding),
        action_source=ActionSource(args.action_source),
        action_scale=ActionScale(
            translation_m=args.translation_scale,
            rotation_rad=args.rotation_scale,
            gripper=args.gripper_scale,
        ),
        kinematics_config=(
            args.kinematics_config.expanduser().resolve() if args.kinematics_config else None
        ),
        t5_embeddings=(args.t5_embeddings.expanduser().resolve() if args.t5_embeddings else None),
        skip_t5=args.skip_t5,
        episode_manifest=(
            args.episode_manifest.expanduser().resolve() if args.episode_manifest else None
        ),
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
        monitor_storage=args.monitor_storage,
        storage_sample_interval=args.storage_sample_interval,
        storage_probe_mib=args.storage_probe_mib,
        resume=args.resume,
    )
    return config, args.preflight_only


def main() -> None:
    """启动顶层ConversionPipeline。"""
    config, preflight_only = parse_config()
    ConversionPipeline(config).run(preflight_only=preflight_only)


if __name__ == "__main__":
    main()
