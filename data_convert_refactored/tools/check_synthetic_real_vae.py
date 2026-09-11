#!/usr/bin/env python3
"""Run a real-VAE synthetic conversion and compare every stored value.

The reference path computes low-dimensional values with the independent NumPy
oracle, encodes the prepared videos directly with the real Cosmos VAE, and
injects conditions with an independent NumPy implementation. The candidate
path runs the complete production ConversionPipeline and reads its Parquet.
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from data_convert_refactored.conditioning.camera import RAW_FRAME_SLICES
from data_convert_refactored.conditioning.episode_labeling import EpisodeOutcome
from data_convert_refactored.conditioning.transition_builder import (
    build_transition_list_for_episode,
)
from data_convert_refactored.config import (
    ActionEncoding,
    ActionScale,
    ActionSource,
    ConversionConfig,
    StatsMode,
)
from data_convert_refactored.cosmos_backend import CosmosBackend
from data_convert_refactored.encoding.policy_loader import init_cosmos_policy
from data_convert_refactored.encoding.vae_encoding import (
    encode_normalized_video_batch,
    normalize_video_batch_cpu,
)
from data_convert_refactored.pipeline import ConversionPipeline
from data_convert_refactored.preparation.episode import load_episode
from data_convert_refactored.tests.synthetic.hdf5_generator import (
    create_synthetic_fixture,
)
from data_convert_refactored.tests.synthetic.expected_values import (
    build_expected_values,
    inject_all_conditions,
)


def _column(table, name: str, dtype=None) -> np.ndarray:
    array = np.asarray(table[name].to_pylist())
    return array.astype(dtype) if dtype is not None else array


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape differs: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    print(f"[MATCH] {name}: shape={actual.shape} max_abs_error={error:.9g}")
    return error


def _build_config(root: Path, stats_path: Path, cosmos_config: Path, encode_batch_size: int):
    return ConversionConfig(
        input_dir=root / "input",
        output_dir=root / "output",
        task="synthetic predictable motion",
        episode_outcome=EpisodeOutcome.SUCCESS,
        stats_mode=StatsMode.OFFICIAL,
        official_dataset_stats=stats_path,
        action_encoding=ActionEncoding.LEGACY_EULER,
        action_source=ActionSource.PUPPET_NEXT_FRAME,
        action_scale=ActionScale(0.02, 0.06, 1.0),
        skip_t5=True,
        encode_batch_size=encode_batch_size,
        encode_world_size=1,
        cosmos_config_path=cosmos_config,
        use_jpeg_compression=False,
        trained_with_image_aug=False,
    )


def _direct_real_vae_reference(config: ConversionConfig, hdf5_path: Path) -> np.ndarray:
    expected = build_expected_values()
    episode = load_episode(hdf5_path)
    runtime = CosmosBackend(config)._runtime_config()
    if runtime.chunk_size != 16:
        raise AssertionError(
            f"synthetic oracle expects chunk_size=16, got {runtime.chunk_size}"
        )
    transitions, _ = build_transition_list_for_episode(
        episode,
        "single_absolute",
        None,
        runtime,
        normalized_actions=expected.normalized_actions,
        normalized_proprio=expected.normalized_proprio,
        euler_actions=expected.first_stage_actions,
        rotation_6d_actions=expected.rotation_6d_actions,
    )
    videos = torch.cat(
        [transition["state"]["video"].clone() for transition in transitions], dim=0
    )
    for timestep in range(len(transitions)):
        future = min(timestep + runtime.chunk_size, len(transitions) - 1)
        videos[timestep, :, RAW_FRAME_SLICES["future_wrist"]] = videos[
            future, :, RAW_FRAME_SLICES["current_wrist"]
        ]
        videos[timestep, :, RAW_FRAME_SLICES["future_primary"]] = videos[
            future, :, RAW_FRAME_SLICES["current_primary"]
        ]

    policy, cosmos_config = init_cosmos_policy(runtime, action_dim=7, proprio_dim=8)
    if not np.isclose(float(getattr(cosmos_config, "gamma", 0.99)), 0.99):
        raise AssertionError(
            f"synthetic oracle expects gamma=0.99, got {cosmos_config.gamma}"
        )
    device = next(policy.parameters()).device
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(videos), config.encode_batch_size):
            batch = normalize_video_batch_cpu(videos[start : start + config.encode_batch_size])
            chunks.append(
                encode_normalized_video_batch(
                    policy,
                    batch,
                    single_device=device,
                    encode_batch_size=config.encode_batch_size,
                    tokenizer_config=cosmos_config.tokenizer,
                )
            )
    base_latent = torch.cat(chunks).float().numpy()
    del chunks, videos, transitions, policy, cosmos_config
    gc.collect()
    torch.cuda.empty_cache()
    return inject_all_conditions(base_latent, expected)


def run(root: Path, cosmos_config: Path, encode_batch_size: int) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("real VAE check requires an available CUDA device")
    root.mkdir(parents=True, exist_ok=False)
    hdf5_path, stats_path = create_synthetic_fixture(root)
    config = _build_config(root, stats_path, cosmos_config, encode_batch_size)
    expected = build_expected_values()

    print("[REFERENCE] direct real VAE encode")
    expected_latent = _direct_real_vae_reference(config, hdf5_path)
    print("[CANDIDATE] complete production conversion")
    ConversionPipeline(config).run()

    parquet_files = sorted((root / "output" / "data").rglob("*.parquet"))
    if len(parquet_files) != 1:
        raise AssertionError(f"expected one Parquet file, got {parquet_files}")
    parquet_path = parquet_files[0]
    table = pq.read_table(parquet_path)

    expected_proprio = (
        torch.from_numpy(expected.normalized_proprio).to(torch.bfloat16).float().numpy()
    )
    expected_future_proprio = (
        torch.from_numpy(expected.future_proprio).to(torch.bfloat16).float().numpy()
    )
    _assert_close("action", _column(table, "action", np.float32), expected.normalized_actions)
    _assert_close(
        "action.euler_control",
        _column(table, "action.euler_control", np.float32),
        expected.first_stage_actions,
    )
    _assert_close(
        "action.rotation_6d_control",
        _column(table, "action.rotation_6d_control", np.float32),
        expected.rotation_6d_actions,
    )
    _assert_close(
        "source.puppet.pose_xyzw",
        _column(table, "source.puppet.pose_xyzw", np.float32),
        expected.poses_xyzw,
    )
    _assert_close(
        "source.puppet.gripper",
        _column(table, "source.puppet.gripper", np.float32).reshape(-1),
        expected.gripper,
    )
    _assert_close("proprio", _column(table, "proprio", np.float32), expected_proprio)
    _assert_close(
        "future_proprio",
        _column(table, "future_proprio", np.float32),
        expected_future_proprio,
    )
    _assert_close(
        "value_function_return",
        _column(table, "value_function_return", np.float32).reshape(-1),
        expected.future_returns,
    )
    _assert_close(
        "next.reward",
        _column(table, "next.reward", np.float32).reshape(-1),
        expected.rewards,
    )
    np.testing.assert_array_equal(
        _column(table, "next.done", bool).reshape(-1), expected.dones
    )
    print(f"[MATCH] next.done: shape={expected.dones.shape}")
    _assert_close("video", _column(table, "video", np.float32), expected_latent)
    print(f"[PASS] real-VAE synthetic conversion: {parquet_path}")
    return parquet_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--cosmos-config", type=Path, default=Path("train_config_cosmos.json")
    )
    parser.add_argument("--encode-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.encode_batch_size < 1:
        parser.error("--encode-batch-size must be positive")
    root = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else Path(tempfile.gettempdir()) / f"cosmos_synthetic_real_vae_{__import__('os').getpid()}"
    )
    try:
        run(root, args.cosmos_config.expanduser(), args.encode_batch_size)
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[ARTIFACTS] {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
