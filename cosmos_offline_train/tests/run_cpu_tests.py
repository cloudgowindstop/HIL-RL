"""Dependency-light CPU correctness suite invoked by ``run.sh test-cpu``."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DistributedSampler, TensorDataset

from ..action_spec import get_action_spec
from ..checkpoint import _cpu_byte_rng_states, load_checkpoint, save_checkpoint
from ..config import ConfigError, resolve_data_layout, validation_batch_limit
from ..dataset import (
    CosmosParquetDataset,
    EpisodeRecord,
    build_pre_split_index,
    load_manifest,
    manifest_sha256,
    save_manifest,
    select_episode_subset,
    split_episode_index,
)
from ..dataloader import DistributedEvalSampler, EpisodeBalancedSampler
from ..evaluation.datasets import InverseFutureShuffleDataset
from ..evaluation.policy import action_metrics
from ..evaluation.rgb_layout import (
    RAW_FRAME_SLICES,
    contact_sheet,
    flatten_clip_frames,
    future_camera_views,
    require_decoded_rgb,
    slice_bounds,
    slice_camera,
)
from ..evaluation.sampling import decode_generated_video
from ..evaluation.statistics import (
    aggregate_episode_metrics,
    average_precision,
    binary_auroc,
    bootstrap_confidence_interval,
    expected_calibration_error,
    mean_absolute_error,
    mean_squared_error,
    spearman_correlation,
)
from ..evaluation.value import (
    extract_value_scalar,
    pooled_value_spearman,
    pop_value_rank_pairs,
    value_metrics,
)
from ..evaluation.video import abs_error_image, psnr, ssim
from ..evaluation.visualize_predictions import (
    load_frozen_sample_specs,
    mean_metrics,
    require_clean_restore,
    sample_stem,
)
from ..losses import edm_losses, masked_mean
from ..masks import (
    INVERSE_DYNAMICS, POLICY, VALUE, WORLD, apply_failed_action_mask, temporal_masks,
)
from ..metrics import JsonlLogger
from ..model import extract_action_chunk, preencoded_edm_forward
from ..monitoring import PrometheusExporter, WandBExporter
from ..plot_metrics import deduplicate_records, metric_series
from ..preflight import run_preflight
from ..rotation_6d import geodesic_error_degrees, matrix_to_rotation_6d, rotation_6d_to_matrix
from ..tools.prepare_legacy_dataset import prepare as prepare_legacy_dataset
from ..tools.select_visualization_samples import selected_row_indices
from ..tools.summarize_compare_matrix import (
    collect_world_rgb,
    extract_id_table,
    extract_policy_row,
    overlay_summary,
    select_end_validation,
    summarize,
)
from ..trainer import gather_value_rank_arrays, should_run_periodic, smoke_loss_window_means


def _summarize_compare_matrix_test() -> None:
    mid = {
        "mode": "validation",
        "global_step": 1000,
        "sigma_0.5/policy/left_rotation_geodesic_deg": 0.5,
        "sigma_0.5/policy/right_rotation_geodesic_deg": 1.2,
        "sigma_0.5/policy/left_translation_mae": 0.11,
        "sigma_0.5/policy/right_translation_mae": 0.04,
        "sigma_0.5/policy/left_gripper_accuracy": 0.95,
        "sigma_0.5/policy/right_gripper_accuracy": 1.0,
        "sigma_0.5/policy/left_horizon_16_rotation_geodesic_deg": 0.3,
        "sigma_0.5/policy/right_horizon_16_rotation_geodesic_deg": 1.6,
        "sigma_0.5/policy/left_horizon_16_translation_mae": 0.12,
        "sigma_0.5/policy/right_horizon_16_translation_mae": 0.03,
    }
    full = {
        **mid,
        "sigma_0.5/world/future_image_l1": 0.12,
        "sigma_0.5/world/future_wrist_image_l1": 0.15,
        "sigma_0.5/world/future_proprio_l1": 0.08,
        "sigma_0.5/world/value_l1": 0.04,
        "sigma_0.5/value/value_scalar_mae": 0.04,
        "sigma_0.5/value/value_l1": 0.06,
    }
    chosen = select_end_validation(
        [
            {"mode": "train", "global_step": 1000, "sigma_0.5/policy/left_rotation_geodesic_deg": 99},
            {"mode": "validation", "global_step": 500, **mid},
            {"mode": "validation", "global_step": 1000, **mid},
            {"mode": "validation", "global_step": 1000, **full},
        ]
    )
    assert chosen["sigma_0.5/world/future_image_l1"] == 0.12
    row = extract_policy_row("6D-P", chosen)
    assert row["left_rotation_geodesic_deg"] == 0.5
    assert row["sigma_0.5/world/future_image_l1"] == 0.12
    merged = overlay_summary(
        full,
        None,
    )
    assert merged is full or merged["sigma_0.5/world/future_image_l1"] == 0.12
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = {
            "sigma_0.5/policy/left_rotation_geodesic_deg": 0.0,
            "sigma_0.5/inverse_dynamics/normal/action_physical_mae": 0.02,
            "sigma_0.5/inverse_dynamics/future_shuffle/action_physical_mae": 0.03,
            "sigma_0.5/inverse_dynamics/future_information_gain/action_physical_mae": 0.01,
            "sigma_0.5/inverse_dynamics/normal/left_rotation_geodesic_deg": 0.1,
            "sigma_0.5/inverse_dynamics/future_shuffle/left_rotation_geodesic_deg": 0.2,
            "sigma_0.5/inverse_dynamics/future_information_gain/left_rotation_geodesic_deg": 0.1,
            "sigma_0.5/inverse_dynamics/normal/right_rotation_geodesic_deg": 0.2,
            "sigma_0.5/inverse_dynamics/future_shuffle/right_rotation_geodesic_deg": 0.1,
            "sigma_0.5/inverse_dynamics/future_information_gain/right_rotation_geodesic_deg": -0.1,
            "sigma_0.5/inverse_dynamics/normal/left_translation_mae": 0.01,
            "sigma_0.5/inverse_dynamics/future_shuffle/left_translation_mae": 0.02,
            "sigma_0.5/inverse_dynamics/future_information_gain/left_translation_mae": 0.01,
            "sigma_0.5/inverse_dynamics/normal/right_translation_mae": 0.01,
            "sigma_0.5/inverse_dynamics/future_shuffle/right_translation_mae": 0.015,
            "sigma_0.5/inverse_dynamics/future_information_gain/right_translation_mae": 0.005,
            "sigma_0.5/inverse_dynamics/normal/left_horizon_16_rotation_geodesic_deg": 0.1,
            "sigma_0.5/inverse_dynamics/future_shuffle/left_horizon_16_rotation_geodesic_deg": 0.2,
            "sigma_0.5/inverse_dynamics/future_information_gain/left_horizon_16_rotation_geodesic_deg": 0.1,
            "sigma_0.5/inverse_dynamics/normal/right_horizon_16_rotation_geodesic_deg": 0.2,
            "sigma_0.5/inverse_dynamics/future_shuffle/right_horizon_16_rotation_geodesic_deg": 0.3,
            "sigma_0.5/inverse_dynamics/future_information_gain/right_horizon_16_rotation_geodesic_deg": 0.1,
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        overlaid = overlay_summary(dict(full), summary_path)
        assert overlaid["sigma_0.5/policy/left_rotation_geodesic_deg"] == 0.5
        assert overlaid["sigma_0.5/inverse_dynamics/normal/action_physical_mae"] == 0.02
        id_rows = extract_id_table(overlaid)
        gain = {row["metric"]: row["delta_future"] for row in id_rows}
        assert gain["action_physical_mae"] == 0.01
        assert gain["right_rotation_geodesic_deg"] == -0.1
        for name, folder in (
            ("E-P", "euler_14d_policy"),
            ("6D-P", "rotation6d_20d_policy"),
            ("6D-J", "rotation6d_20d_joint"),
            ("6D-J-ID", "rotation6d_20d_joint_id"),
        ):
            run_dir = root / folder
            run_dir.mkdir()
            record = dict(full)
            if name == "6D-J-ID":
                record.update(summary)
            (run_dir / "metrics.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        out = root / "compare_matrix"
        viz = root / "visualize" / "euler_14d_policy"
        viz.mkdir(parents=True)
        (viz / "summary.json").write_text(
            json.dumps(
                {
                    "sample_count": 16,
                    "metrics": {
                        "future_cameras_psnr": 12.5,
                        "future_primary_psnr": 11.0,
                        "future_wrist_psnr": 14.0,
                        "future_cameras_ssim": 0.4,
                        "future_primary_ssim": 0.35,
                        "future_wrist_ssim": 0.45,
                    },
                }
            ),
            encoding="utf-8",
        )
        assert collect_world_rgb(root)[0]["future_cameras_psnr"] == 12.5
        payload = summarize(root, out)
        assert (out / "compare_matrix.md").is_file()
        assert payload["table1_e_p_vs_6d_p"][0]["run"] == "E-P"
        assert payload["table3_id_delta_future"][0]["delta_future"] == 0.01
        assert payload["table4_world_rgb"][0]["run"] == "E-P"


def _visualization_sample_selection_test() -> None:
    assert selected_row_indices(984, [0.4, 0.7]) == [(0.4, 393), (0.7, 688)]
    assert selected_row_indices(1663, [0.4, 0.7]) == [(0.4, 664), (0.7, 1163)]
    assert selected_row_indices(1056, [0.4, 0.7]) == [(0.4, 422), (0.7, 738)]
    assert selected_row_indices(1, [0.4, 0.7]) == [(0.4, 0)]


def _rotation_tests() -> None:
    identity = torch.eye(3).repeat(8, 1, 1)
    representation = matrix_to_rotation_6d(identity)
    reconstructed = rotation_6d_to_matrix(representation)
    assert torch.allclose(reconstructed, identity, atol=1e-6)
    assert torch.allclose(torch.det(reconstructed), torch.ones(8), atol=1e-6)
    assert torch.max(geodesic_error_degrees(representation, representation)) < 1e-3


def _mask_loss_tests() -> None:
    types = torch.tensor([POLICY, WORLD, VALUE, INVERSE_DYNAMICS])
    condition, loss = temporal_masks(types)
    assert condition[0].tolist() == [True, True, True, True, False, False, False, False, False]
    assert condition[1].tolist() == [True, True, True, True, True, False, False, False, False]
    assert condition[2].tolist() == [True, True, True, True, True, True, True, True, False]
    assert condition[3].tolist() == [True, True, True, True, False, True, True, True, False]
    assert loss[3].tolist() == [False, False, False, False, True, False, False, False, False]
    current_condition, _ = temporal_masks(
        torch.tensor([INVERSE_DYNAMICS]), inverse_condition_mode="current_only"
    )
    future_condition, _ = temporal_masks(
        torch.tensor([INVERSE_DYNAMICS]), inverse_condition_mode="future_only"
    )
    assert current_condition[0].tolist() == [True, True, True, True, False, False, False, False, False]
    assert future_condition[0].tolist() == [False, False, False, False, False, True, True, True, False]
    failed_mask = apply_failed_action_mask(
        loss, types, torch.tensor([True, True, True, True])
    )
    assert not failed_mask[0, 4] and failed_mask[1, 5] and failed_mask[2, 8]
    assert failed_mask[3, 4]
    values = torch.tensor([1.0, 100.0])
    assert masked_mean(values, torch.tensor([True, False])).item() == 1.0
    target = torch.zeros(4, 2, 9, 2, 2)
    prediction = torch.ones_like(target)
    total, metrics = edm_losses(prediction, target, torch.ones(4, 9), loss, sample_types=types)
    assert torch.allclose(total, torch.tensor(1.0))
    assert math.isfinite(metrics["action_edm_loss"].item())
    assert all(
        f"sample_{name}_edm_loss" in metrics
        for name in ("policy", "world", "value", "inverse_dynamics")
    )
    assert metrics["sample_policy_edm_loss"].item() == 1.0
    assert metrics["sample_world_edm_loss"].item() == 1.0
    assert metrics["sample_value_edm_loss"].item() == 1.0
    assert metrics["sample_inverse_dynamics_edm_loss"].item() == 1.0
    learner_total, learner_metrics = edm_losses(
        prediction,
        target,
        torch.ones(4, 9),
        loss,
        sample_types=types,
        reduction="learner_actor_full_tensor_mean",
    )
    expected_ratio = loss.float().mean()
    assert torch.allclose(learner_total, expected_ratio)
    assert torch.allclose(learner_metrics["policy_masked_edm_loss"], torch.tensor(1.0))


def _validation_limit_tests() -> None:
    assert validation_batch_limit({"evaluation": {"max_validation_batches": 200}}) == 200
    assert validation_batch_limit({"evaluation": {"max_validation_batches": None}}) is None
    try:
        validation_batch_limit({"evaluation": {"max_validation_batches": 0}})
    except ConfigError:
        pass
    else:
        raise AssertionError("zero validation batch limit was accepted")

    class FakeEpisodeDataset:
        records = [SimpleNamespace(rows=3), SimpleNamespace(rows=4), SimpleNamespace(rows=5)]
        _offsets = np.asarray([0, 3, 7, 12])

    sampler = EpisodeBalancedSampler(FakeEpisodeDataset(), seed=7, max_samples=6)
    indices = list(sampler)
    assert len(indices) == 6 and len(set(indices)) == 6
    selected_episodes = {
        int(np.searchsorted(FakeEpisodeDataset._offsets, index, side="right") - 1)
        for index in indices
    }
    assert selected_episodes == {0, 1, 2}
    episode_sequence = [
        int(np.searchsorted(FakeEpisodeDataset._offsets, index, side="right") - 1)
        for index in indices
    ]
    assert sum(left != right for left, right in zip(episode_sequence, episode_sequence[1:])) <= 2
    assert indices == list(
        EpisodeBalancedSampler(FakeEpisodeDataset(), seed=7, max_samples=6)
    )
    shards = [list(DistributedEvalSampler(range(11), rank=rank, world_size=4)) for rank in range(4)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))
    assert should_run_periodic(500, 500)
    assert not should_run_periodic(0, 500)
    assert not should_run_periodic(499, 500)


def _plot_metric_tests() -> None:
    records = [
        {
            "mode": "train", "global_step": 1,
            "action_edm_loss": 0.0, "policy_samples": 0,
            "inverse_dynamics_samples": 0,
        },
        {
            "mode": "train", "global_step": 2,
            "action_edm_loss": 2.0, "policy_samples": 0,
            "inverse_dynamics_samples": 1,
        },
    ]
    steps, values = metric_series(records, "action_edm_loss")
    assert steps.tolist() == [2.0] and values.tolist() == [2.0]
    deduplicated = deduplicate_records(
        [
            {"mode": "train", "global_step": 5, "loss_actor_learner": 1.0},
            {"mode": "train", "global_step": 5, "loss_actor_learner": 0.5},
        ]
    )
    assert len(deduplicated) == 1
    assert deduplicated[0]["loss_actor_learner"] == 0.5
    eval_steps, eval_values = metric_series(
        [{"mode": "eval", "global_step": 500, "eval_loss_actor": 0.5}],
        "eval_loss_actor",
        modes=frozenset({"eval"}),
    )
    assert eval_steps.tolist() == [500.0] and eval_values.tolist() == [0.5]


def _prometheus_exporter_test() -> None:
    exporter = PrometheusExporter(
        enabled=True, host="127.0.0.1", port=8000, path="/metrics", serve=False
    )
    try:
        assert exporter.enabled
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlLogger(Path(directory), exporter=exporter)
            logger.log(
                {
                    "mode": "train",
                    "global_step": 5,
                    "loss_actor_learner": 0.25,
                    "gradient_norm": 1.5,
                }
            )
            logger.log(
                {
                    "mode": "eval",
                    "global_step": 100,
                    "eval_loss_actor": 0.2,
                    "eval_action_l1_loss": 0.1,
                }
            )
            assert len(logger.path.read_text(encoding="utf-8").splitlines()) == 2
        body = exporter.render_metrics()
        assert "cosmos_global_step 100" in body
        assert "cosmos_train_loss_actor 0.25" in body
        assert "cosmos_gradient_norm 1.5" in body
        assert "cosmos_eval_loss_actor 0.2" in body
        assert "cosmos_eval_action_l1_loss 0.1" in body
    finally:
        exporter.close()
    assert not exporter.enabled
    non_main = PrometheusExporter.from_config(
        {"monitoring": {"prometheus_enabled": True}}, is_main=False
    )
    assert not non_main.enabled


def _wandb_exporter_test() -> None:
    class FakeRun:
        name = "E-P"
        url = "https://wandb.test/run"

        def __init__(self):
            self.logs: list[dict] = []
            self.finished = False

        def log(self, data, **kwargs):
            self.logs.append({"data": data, **kwargs})

        def finish(self):
            self.finished = True

    run = FakeRun()
    exporter = WandBExporter(enabled=False, project="unused")
    assert not exporter.enabled
    exporter.update({"mode": "train", "global_step": 1, "total_edm_loss": 0.5})
    exporter = WandBExporter.__new__(WandBExporter)
    exporter.enabled = True
    exporter._wandb = run
    exporter.update(
        {
            "mode": "train",
            "global_step": 5,
            "total_edm_loss": 0.25,
            "sample_id": "skip-me",
        }
    )
    exporter.update(
        {"mode": "eval", "global_step": 500, "eval_loss_actor": 0.2}
    )
    assert run.logs[0]["data"]["global_step"] == 5
    assert run.logs[0]["data"]["train/total_edm_loss"] == 0.25
    assert "train/sample_id" not in run.logs[0]["data"]
    assert run.logs[1]["data"]["eval/eval_loss_actor"] == 0.2
    assert run.logs[1]["data"]["global_step"] == 500
    exporter.close()
    assert run.finished and not exporter.enabled
    non_main = WandBExporter.from_config(
        {
            "monitoring": {
                "wandb_enabled": True,
                "wandb_project": "cosmos-offline-compare",
            },
            "runtime": {"output_dir": "/tmp"},
        },
        is_main=False,
    )
    assert not non_main.enabled


def _evaluation_metric_tests() -> None:
    prediction = np.asarray([0.0, 1.0, 2.0])
    target = np.asarray([0.0, 2.0, 2.0])
    assert mean_absolute_error(prediction, target) == 1 / 3
    assert mean_squared_error(prediction, target) == 1 / 3
    assert binary_auroc([0.1, 0.9], [0, 1]) == 1.0
    assert average_precision([0.1, 0.9], [0, 1]) == 1.0
    assert spearman_correlation([1, 2, 3], [2, 4, 6]) == 1.0
    assert spearman_correlation([3, 2, 1], [1, 2, 3]) == -1.0
    assert math.isnan(spearman_correlation([1, 1, 1], [0, 2, 4]))
    assert math.isnan(spearman_correlation([1], [2]))
    spearman, count = pooled_value_spearman([0.1, 0.4, 0.9], [1.0, 2.0, 3.0])
    assert spearman == 1.0 and count == 3
    ranking = pop_value_rank_pairs(
        {
            "value_rank_pred": torch.tensor([0.1, 0.2, 0.3]),
            "value_rank_target": torch.tensor([1.0, 2.0, 3.0]),
            "value_rank_valid": torch.tensor([True, False, True]),
        }
    )
    assert ranking is not None
    assert torch.allclose(ranking[0], torch.tensor([0.1, 0.3]))
    assert torch.allclose(ranking[1], torch.tensor([1.0, 3.0]))
    assert pop_value_rank_pairs({"value_scalar_mae": torch.zeros(())}) is None
    latent = torch.zeros(2, 4, 9, 2, 2)
    latent[0, :, 8] = 2.0
    latent[1, :, 8] = 4.0
    assert extract_value_scalar(latent).tolist() == [2.0, 4.0]
    report = value_metrics([0.1, 0.9], [0.0, 1.0])
    assert math.isclose(report["value_spearman"], 1.0, rel_tol=0, abs_tol=1e-12)
    gathered_pred, gathered_target = gather_value_rank_arrays(
        [torch.tensor([0.2]), torch.tensor([0.8])],
        [torch.tensor([1.0]), torch.tensor([3.0])],
        SimpleNamespace(enabled=False, world_size=1),
    )
    assert np.allclose(gathered_pred, [0.2, 0.8])
    assert np.allclose(gathered_target, [1.0, 3.0])
    assert expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0
    lower, upper = bootstrap_confidence_interval([1.0, 1.0], samples=20)
    assert lower == upper == 1.0
    report = aggregate_episode_metrics(
        [
            {"episode_path": "a", "metrics": {"x": 1.0}},
            {"episode_path": "a", "metrics": {"x": 3.0}},
            {"episode_path": "b", "metrics": {"x": 4.0}},
        ],
        bootstrap_samples=20,
    )
    assert report["x"]["episodes"] == 2 and report["x"]["mean"] == 3.0


def _video_metric_tests() -> None:
    target = torch.zeros(2, 3, 16, 16)
    identical = target.clone()
    changed = target.clone()
    changed[:, :, 4:12, 4:12] = 1
    assert torch.all(psnr(identical, target) > 60)
    assert torch.allclose(ssim(identical, target), torch.ones(2), atol=1e-5)
    assert torch.all(psnr(changed, target) < psnr(identical, target))
    assert torch.all(ssim(changed, target) < ssim(identical, target))
    error = abs_error_image(changed, target, normalize=True)
    assert error.shape == changed.shape
    assert torch.isclose(error.amax(), torch.tensor(1.0))
    raw = abs_error_image(changed, target, normalize=False)
    assert torch.allclose(raw.amax(), torch.tensor(1.0))


def _rgb_layout_tests() -> None:
    assert slice_bounds("current_wrist") == (5, 9)
    assert slice_bounds("current_primary") == (9, 13)
    assert slice_bounds("future_wrist") == (21, 25)
    assert slice_bounds("future_primary") == (25, 29)
    assert RAW_FRAME_SLICES["future_wrist"] == slice(21, 25)
    video = torch.zeros(1, 3, 33, 8, 8)
    video[:, :, 21:25] = 0.25
    video[:, :, 25:29] = 0.75
    views = future_camera_views(video)
    assert views["future_wrist"].shape == (1, 3, 4, 8, 8)
    assert torch.all(views["future_wrist"] == 0.25)
    assert torch.all(views["future_primary"] == 0.75)
    assert flatten_clip_frames(views["future_wrist"]).shape == (4, 3, 8, 8)
    assert slice_camera(video[0], "current_wrist").shape == (1, 3, 4, 8, 8)
    try:
        require_decoded_rgb(torch.zeros(1, 3, 16, 8, 8))
    except ValueError as error:
        assert "33" in str(error)
    else:
        raise AssertionError("decoded RGB with 16 frames was accepted")
    sheet = contact_sheet(views, views, views)
    assert sheet.ndim == 3 and sheet.shape[0] == 3
    assert sample_stem({"source_id": "0804_am_success/0804_094025", "row_index": 393}) == (
        "0804_am_success_0804_094025_row393"
    )


def _visualization_sample_list_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "samples.json"
        path.write_text(
            json.dumps(
                {
                    "row_fractions": [0.4, 0.7],
                    "episodes": [
                        {
                            "source_id": "0804_am_success/0804_094025",
                            "rows": 984,
                            "row_indices": [393, 688],
                            "episode_paths": {
                                "cosmos_rotation_6d": "/tmp/6d.parquet",
                                "legacy_euler": "/tmp/euler.parquet",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        specs = load_frozen_sample_specs(path, encoding="cosmos_rotation_6d")
        assert [item["row_index"] for item in specs] == [393, 688]
        assert specs[0]["episode_path"] == "/tmp/6d.parquet"
        try:
            load_frozen_sample_specs(path)
        except ValueError:
            pass
        else:
            raise AssertionError("protocol list without --encoding was accepted")
        try:
            require_clean_restore({}, {"episode_path": "/tmp/x.parquet", "row_index": 0})
        except ValueError as error:
            assert "clean_restore_latent" in str(error)
        else:
            raise AssertionError("missing clean_restore_latent was accepted")
        averaged = mean_metrics(
            [
                {"future_cameras_psnr": 8.0, "future_cameras_ssim": 0.2, "row_fraction": 0.4},
                {"future_cameras_psnr": 10.0, "future_cameras_ssim": 0.4, "row_fraction": 0.7},
            ]
        )
        assert averaged.keys() == {"future_cameras_psnr", "future_cameras_ssim"}
        assert abs(averaged["future_cameras_psnr"] - 9.0) < 1e-9
        assert abs(averaged["future_cameras_ssim"] - 0.3) < 1e-9


def _inverse_shuffle_dataset_test() -> None:
    class FakeDataset:
        records = [SimpleNamespace(rows=2), SimpleNamespace(rows=2)]
        _offsets = np.asarray([0, 2, 4])

        def __len__(self):
            return 4

        def _locate(self, index):
            return (0, index) if index < 2 else (1, index - 2)

        def __getitem__(self, index):
            return {
                "video": torch.full((16, 9, 2, 2), float(index)),
                "sample_id": str(index),
            }

    dataset = InverseFutureShuffleDataset(FakeDataset())
    sample = dataset[0]
    assert sample["shuffled_future_sample_id"] == "2"
    assert torch.all(sample["shuffled_future_video"] == 2)


def _safe_decode_test() -> None:
    class FakeModel:
        def __init__(self):
            self.decoded_input = None

        def decode(self, value):
            self.decoded_input = value.clone()
            return torch.zeros(value.shape[0], 3, 33, 4, 4)

    model = FakeModel()
    generated = torch.ones(1, 16, 9, 2, 2)
    clean = torch.zeros(1, 16, 4, 2, 2)
    decoded = decode_generated_video(model, generated, clean_restore_latent=clean)
    assert decoded.min() == decoded.max() == 0.5
    assert torch.all(model.decoded_input[:, :, [1, 4, 5, 8]] == 0)
    try:
        decode_generated_video(model, generated, clean_restore_latent=None)
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe decode without clean restore latent was accepted")


def _split_tests() -> None:
    records = [
        EpisodeRecord(
            f"/{task}/{episode}.parquet", "/", f"g{episode % 2}", task, "success", 2
        )
        for task in ("a", "b", "c")
        for episode in range(20)
    ]
    train, val, test = split_episode_index(records, seed=7)
    paths = [{record.path for record in split} for split in (train, val, test)]
    assert paths[0].isdisjoint(paths[1])
    assert paths[0].isdisjoint(paths[2])
    assert paths[1].isdisjoint(paths[2])
    assert set().union(*paths) == {record.path for record in records}
    assert [len(split) for split in (train, val, test)] == [48, 6, 6]
    for split in (train, val, test):
        assert {record.task for record in split} == {"a", "b", "c"}
    again = split_episode_index(records, seed=7)
    assert [[r.path for r in x] for x in (train, val, test)] == [[r.path for r in x] for x in again]

    tiny = [
        EpisodeRecord(f"/tiny/{episode}.parquet", "/", "one-group", "tiny", "success", 2)
        for episode in range(2)
    ]
    tiny_train, tiny_val, tiny_test = split_episode_index(tiny, seed=7)
    assert [len(split) for split in (tiny_train, tiny_val, tiny_test)] == [1, 1, 0]
    train_only = split_episode_index(tiny, seed=7, ratios=(1.0, 0.0, 0.0))
    assert [len(split) for split in train_only] == [2, 0, 0]

    one_task_records = [record for record in records if record.task == "a"]
    smoke_records = select_episode_subset(one_task_records, max_episodes=10, seed=42)
    smoke_train, smoke_val, smoke_test = split_episode_index(
        smoke_records, seed=42, ratios=(0.8, 0.2, 0.0)
    )
    assert [len(split) for split in (smoke_train, smoke_val, smoke_test)] == [8, 2, 0]
    again = select_episode_subset(one_task_records, max_episodes=10, seed=42)
    assert [record.path for record in smoke_records] == [record.path for record in again]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.json"
        save_manifest(smoke_train, path)
        digest = manifest_sha256(path)
        assert load_manifest(path) == smoke_train
        try:
            save_manifest(smoke_train, path)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing manifest was overwritten")
        save_manifest(smoke_train, path, overwrite=True)
        assert manifest_sha256(path) == digest

    dataset = TensorDataset(torch.arange(20))
    rank0 = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=7, drop_last=True)
    rank1 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=7, drop_last=True)
    first0, first1 = list(rank0), list(rank1)
    assert set(first0).isdisjoint(first1) and set(first0 + first1) == set(range(20))
    rank0.set_epoch(1)
    assert list(rank0) != first0


def _lazy_dataset_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "episode_000000.parquet"
        video = np.zeros((16, 9, 28, 28), dtype=np.float32).tolist()
        table = pa.Table.from_pylist([
            {
                "video": video,
                "action": [0.0] * 20,
                "proprio": [0.0] * 16,
                "future_proprio": [0.0] * 16,
                "value_function_return": 1.0,
                "next.reward": 0.0,
                "next.done": False,
            }
        ])
        pq.write_table(table, path)
        dataset = CosmosParquetDataset([
            EpisodeRecord(str(path), directory, "g", "task", "success", 1)
        ])
        assert len(dataset) == 1 and len(dataset._cache) == 0
        sample = dataset[0]
        assert sample["video"].shape == (16, 9, 28, 28)
        assert sample["action"].shape == (20,)
        assert len(dataset._cache) == 1

        table14 = table.set_column(
            table.schema.get_field_index("action"),
            "action",
            pa.array([[0.0] * 14]),
        )
        path14 = Path(directory) / "episode_000001.parquet"
        pq.write_table(table14, path14)
        dataset14 = CosmosParquetDataset(
            [EpisodeRecord(str(path14), directory, "g", "task", "success", 1)],
            action_dimension=14,
        )
        assert dataset14[0]["action"].shape == (14,)

        two_group = Path(directory) / "episode_000002.parquet"
        first = np.zeros((16, 9, 28, 28), dtype=np.float32)
        second = np.ones((16, 9, 28, 28), dtype=np.float32)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {**table.to_pylist()[0], "video": first.tolist(), "action": [1.0] * 20},
                    {**table.to_pylist()[0], "video": second.tolist(), "action": [2.0] * 20},
                ]
            ),
            two_group,
            row_group_size=1,
        )
        grouped = pq.ParquetFile(two_group)
        assert grouped.num_row_groups == 2
        dataset2 = CosmosParquetDataset(
            [EpisodeRecord(str(two_group), directory, "g", "task", "success", 2)]
        )
        sample1 = dataset2[1]
        assert sample1["video"].shape == (16, 9, 28, 28)
        assert torch.equal(sample1["action"], torch.full((20,), 2.0))
        assert list(dataset2._cache) == [(str(two_group), 1)]


def _action_compatibility_tests() -> None:
    euler = get_action_spec("legacy_euler")
    rotation_6d = get_action_spec("cosmos_rotation_6d")
    assert euler.dimension == 14 and euler.gripper_indices() == (6, 13)
    assert rotation_6d.dimension == 20 and rotation_6d.gripper_indices() == (9, 19)

    expected = torch.arange(2 * 16 * 14, dtype=torch.float32).reshape(2, 16, 14)
    flat = expected.flatten(1)
    plane_elements = 16 * 28 * 28
    repeats = math.ceil(plane_elements / flat.shape[1])
    plane = flat.repeat(1, repeats)[:, :plane_elements]
    latent = torch.zeros(2, 16, 9, 28, 28)
    latent[:, :, 4] = plane.reshape(2, 16, 28, 28)
    actual = extract_action_chunk(latent, action_dimension=14)
    assert torch.equal(actual, expected)

    target = torch.zeros(1, 16, 14)
    prediction = target.clone()
    prediction[..., 5] = math.pi / 18
    metrics = action_metrics(
        prediction, target, action_encoding="legacy_euler"
    )
    assert torch.allclose(
        metrics["left_rotation_geodesic_deg"], torch.tensor(10.0), atol=1e-3
    )
    assert torch.isfinite(metrics["right_rotation_geodesic_deg"])


def _pre_split_layout_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for split in ("train", "eval"):
            split_root = root / split
            data_dir = split_root / "data" / "chunk-000"
            data_dir.mkdir(parents=True)
            (split_root / "cosmos_dataset_metadata.json").write_text(
                '{"task_description":"task","episode_labeling":{"outcome":"success"}}',
                encoding="utf-8",
            )
            pq.write_table(
                pa.Table.from_pylist([{"action": [0.0] * 14}]),
                data_dir / "episode_000000.parquet",
            )
        splits = build_pre_split_index(root / "train", root / "eval")
        assert len(splits.train) == 1 and len(splits.validation) == 1
        assert not splits.test

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        nested = root / "datasets" / "task_a"
        nested.mkdir(parents=True)
        (nested / "cosmos_dataset_metadata.json").write_text("{}", encoding="utf-8")
        assert resolve_data_layout({"layout": "auto"}, root) == "unified"


def _legacy_prepare_tool_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        row = {
            "video": np.zeros((16, 9, 28, 28), dtype=np.float32).tolist(),
            "action": [0.0] * 14,
            "proprio": [0.0] * 16,
            "future_proprio": [0.0] * 16,
            "value_function_return": 0.0,
            "next.reward": 0.0,
            "next.done": False,
        }
        info = {
            "total_episodes": 1,
            "features": {
                "video": {"shape": [16, 9, 28, 28]},
                "action": {"shape": [14]},
                "proprio": {"shape": [16]},
                "future_proprio": {"shape": [16]},
            },
        }
        for split in ("train", "eval"):
            split_root = root / split
            data_dir = split_root / "data" / "chunk-000"
            meta_dir = split_root / "meta"
            data_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)
            (meta_dir / "info.json").write_text(
                __import__("json").dumps(info), encoding="utf-8"
            )
            pq.write_table(
                pa.Table.from_pylist([row]), data_dir / "episode_000000.parquet"
            )
        statistics_path = root / "dataset_statistics.json"
        statistics_path.write_text(
            __import__("json").dumps(
                {"actions_min": [-1.0] * 14, "actions_max": [1.0] * 14}
            ),
            encoding="utf-8",
        )
        prepare_legacy_dataset(
            SimpleNamespace(
                root=str(root), task="install handle",
                statistics_path=str(statistics_path), action_encoding="legacy_euler",
                action_source="puppet_next_frame", outcome="success", group=None,
                chunk_size=16, rotation_scale=0.06,
                train_directory="train", val_directory="eval",
                test_directory=None, overwrite=False, dry_run=False,
            )
        )
        assert (root / "train" / "cosmos_dataset_metadata.json").is_file()
        assert len(load_manifest(root / "manifests" / "train.json")) == 1
        report = run_preflight(
            {
                "data": {
                    "layout": "pre_split", "root": str(root),
                    "action_encoding": "legacy_euler", "action_dimension": 14,
                    "rotation_scale": 0.06,
                    "action_includes_gripper": True, "number_of_arms": 2,
                    "chunk_size": 16, "statistics_path": str(statistics_path),
                }
            }
        )
        assert report.passed, report.errors


def _smoke_loss_trend_test() -> None:
    first, last, window = smoke_loss_window_means(
        [1.00, 0.96, 1.02, 0.94, 0.82, 0.85, 0.78, 0.80]
    )
    assert window == 2
    assert last < first

    first, last, _ = smoke_loss_window_means(
        [0.70, 0.72, 0.75, 0.78, 0.80, 0.84, 0.86, 0.90]
    )
    assert last > first

    try:
        smoke_loss_window_means([1.0, 0.9, 0.8])
    except ValueError:
        pass
    else:
        raise AssertionError("short smoke loss history must be rejected")


def _checkpoint_cuda_rng_restore_test() -> None:
    gpu_like = torch.zeros(8, dtype=torch.uint8)
    fake_cuda = gpu_like.new_tensor([1, 2, 3, 4, 5, 6, 7, 8])
    restored = _cpu_byte_rng_states([fake_cuda, fake_cuda.clone()])
    assert len(restored) == 2
    for state in restored:
        assert state.device.type == "cpu"
        assert state.dtype == torch.uint8
        assert state.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert _cpu_byte_rng_states(None) == []
    assert len(_cpu_byte_rng_states(fake_cuda)) == 1


def _checkpoint_test() -> None:
    class Holder:
        def __init__(self):
            self.net = nn.Linear(2, 1)

    with tempfile.TemporaryDirectory() as directory:
        model = Holder()
        optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        original = {name: value.detach().clone() for name, value in model.net.state_dict().items()}
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=2, batch_in_epoch=3, global_step=7, samples_seen=32, config={"x": 1},
            data_identity={"train_manifest_sha256": "abc"},
        )
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.add_(10)
        state = load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device="cpu",
            expected_identity={"train_manifest_sha256": "abc"},
        )
        assert state == {"epoch": 2, "global_step": 7, "samples_seen": 32, "batch_in_epoch": 3}
        for name, value in model.net.state_dict().items():
            assert torch.equal(value, original[name])
        try:
            load_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, device="cpu",
                expected_identity={"train_manifest_sha256": "changed"},
            )
        except RuntimeError as error:
            assert "数据身份不匹配" in str(error)
        else:
            raise AssertionError("checkpoint identity mismatch was accepted")


def _preencoded_forward_test() -> None:
    class Condition:
        def __init__(self):
            self.condition_video_input_mask_B_C_T_H_W = torch.zeros(2, 1, 9, 28, 28)

        def edit_data_type(self, _):
            return self

        def set_video_condition(self, gt_frames, **_):
            self.gt_frames = gt_frames
            return self

    class SDE:
        @staticmethod
        def marginal_prob(x0, sigma):
            return x0, torch.zeros_like(sigma)

    class FakeModel:
        tensor_kwargs = {"device": torch.device("cpu"), "dtype": torch.float32}
        sde = SDE()

        @staticmethod
        def conditioner(_):
            return Condition()

        @staticmethod
        def draw_training_sigma_and_epsilon(size, condition):
            return torch.ones(size[0], size[2]), torch.zeros(size)

        @staticmethod
        def denoise(noisy, sigma, condition):
            return SimpleNamespace(x0=noisy * 0.9)

        @staticmethod
        def get_per_sigma_loss_weights(sigma):
            return torch.ones_like(sigma)

    video = torch.zeros(2, 16, 9, 28, 28)
    # Identity rotation in both arms inside the action latent's flattened payload.
    action_chunk = torch.zeros(2, 16, 20)
    action_chunk[..., 3] = action_chunk[..., 6] = 1
    action_chunk[..., 13] = action_chunk[..., 16] = 1
    flat = action_chunk.reshape(2, -1)
    action_frame = torch.zeros(2, 16 * 28 * 28)
    action_frame[:, : flat.shape[1]] = flat
    video[:, :, 4] = action_frame.reshape(2, 16, 28, 28)
    types = torch.tensor([POLICY, WORLD])
    condition, loss_mask = temporal_masks(types)
    loss, metrics = preencoded_edm_forward(
        FakeModel(), {"video": video, "value": torch.tensor([0.2, 0.8])},
        condition, loss_mask,
        torch.zeros(2, 512, 4096), torch.full((20,), -1.0), torch.ones(20),
        conditioner_data_type=object(),
    )
    assert "value_rank_pred" in metrics
    assert torch.allclose(metrics["value_rank_target"], torch.tensor([0.2, 0.8]))
    assert pop_value_rank_pairs(metrics) is not None
    assert "value_rank_pred" not in metrics
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["left_rotation_geodesic_deg"])
    assert "left_horizon_16_rotation_geodesic_deg" in metrics

    euler_video = torch.zeros(2, 16, 9, 28, 28)
    euler_action = torch.zeros(2, 16, 14)
    euler_action[..., 5] = 0.1
    euler_action[..., 12] = -0.1
    flat = euler_action.flatten(1)
    repeats = math.ceil((16 * 28 * 28) / flat.shape[1])
    euler_video[:, :, 4] = flat.repeat(1, repeats)[:, : 16 * 28 * 28].reshape(
        2, 16, 28, 28
    )
    euler_loss, euler_metrics = preencoded_edm_forward(
        FakeModel(), {"video": euler_video}, condition, loss_mask,
        torch.zeros(2, 512, 4096), torch.full((14,), -1.0), torch.ones(14),
        action_encoding="legacy_euler", action_dimension=14,
        conditioner_data_type=object(),
    )
    assert torch.isfinite(euler_loss)
    assert torch.isfinite(euler_metrics["left_rotation_geodesic_deg"])


def run() -> None:
    tests = [
        _summarize_compare_matrix_test,
        _visualization_sample_selection_test,
        _rgb_layout_tests, _visualization_sample_list_test,
        _rotation_tests, _mask_loss_tests, _validation_limit_tests, _plot_metric_tests,
        _prometheus_exporter_test, _wandb_exporter_test,
        _evaluation_metric_tests, _video_metric_tests, _inverse_shuffle_dataset_test,
        _safe_decode_test, _split_tests, _lazy_dataset_test,
        _action_compatibility_tests, _pre_split_layout_test, _legacy_prepare_tool_test,
        _smoke_loss_trend_test, _checkpoint_cuda_rng_restore_test, _checkpoint_test,
        _preencoded_forward_test,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[TEST] {len(tests)} CPU test groups passed")


if __name__ == "__main__":
    run()
