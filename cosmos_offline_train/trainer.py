"""Pure offline train/validation loop."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import load_checkpoint, save_checkpoint
from .config import validation_batch_limit
from .distributed import DistributedContext, reduce_metrics
from .masks import (
    INVERSE_DYNAMICS, OBJECTIVE_NAMES, POLICY, VALUE, WORLD,
    apply_failed_action_mask, sample_types, temporal_masks,
)
from .metrics import JsonlLogger, MeanMetrics
from .model import TextEmbeddingProvider, preencoded_edm_forward
from .monitoring import PrometheusExporter, WandBExporter
from .evaluation.statistics import aggregate_episode_metrics
from .evaluation.value import pooled_value_spearman, pop_value_rank_pairs


def smoke_loss_window_means(losses: list[float]) -> tuple[float, float, int]:
    """Compare early and late loss windows while tolerating single-step noise."""
    if len(losses) < 4:
        raise ValueError("smoke optimization check requires at least 4 steps")
    window = max(2, len(losses) // 4)
    first_mean = sum(losses[:window]) / window
    last_mean = sum(losses[-window:]) / window
    return first_mean, last_mean, window


def should_run_periodic(step: int, interval: int) -> bool:
    """Return whether a positive optimizer step hits a configured interval."""
    return interval > 0 and step > 0 and step % interval == 0


def gather_value_rank_arrays(
    pred_chunks: list[torch.Tensor],
    target_chunks: list[torch.Tensor],
    context: DistributedContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate local ranking pairs, then gather across ranks."""
    if pred_chunks:
        local_pred = torch.cat(pred_chunks).numpy()
        local_target = torch.cat(target_chunks).numpy()
    else:
        local_pred = np.zeros((0,), dtype=np.float64)
        local_target = np.zeros((0,), dtype=np.float64)
    if not context.enabled:
        return local_pred, local_target
    gathered: list[tuple[np.ndarray, np.ndarray] | None] = [
        None for _ in range(context.world_size)
    ]
    dist.all_gather_object(gathered, (local_pred, local_target))
    return (
        np.concatenate([item[0] for item in gathered if item is not None]),
        np.concatenate([item[1] for item in gathered if item is not None]),
    )


@contextmanager
def unwrapped_ddp_net(model):
    """Run uneven validation shards without DDP forward collectives."""
    net = model.net
    if isinstance(net, DDP):
        model.net = net.module
    try:
        yield
    finally:
        model.net = net


class CosmosOfflineTrainer:
    SAMPLE_COUNT_METRICS = frozenset(
        {
            "policy_samples", "world_samples", "value_samples",
            "inverse_dynamics_samples", "failure_samples",
        }
    )

    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loader,
        config: dict[str, Any],
        context: DistributedContext,
        resume_path: Path | None = None,
        data_identity: dict[str, str] | None = None,
        evaluation_only: bool = False,
        inverse_shuffle_loader=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.context = context
        self.training = config["training"]
        self.runtime = config["runtime"]
        self.output_dir = Path(self.runtime["output_dir"]).expanduser().resolve()
        self.logger: JsonlLogger | None = None
        self.prometheus: PrometheusExporter | None = None
        self.data_identity = dict(data_identity or {})
        self.evaluation_only = evaluation_only
        self.inverse_shuffle_loader = inverse_shuffle_loader
        expected_tasks = {
            record.task
            for loader in (train_loader, val_loader)
            for record in loader.loader.dataset.records
        }
        self.embeddings = TextEmbeddingProvider(
            config["data"]["t5_embeddings_path"],
            worker_id=context.local_rank,
            expected_tasks=expected_tasks,
        )
        statistics_path = config["data"].get("statistics_path")
        if not statistics_path:
            raise ValueError("data.statistics_path 未配置")
        statistics = json.loads(Path(statistics_path).expanduser().read_text(encoding="utf-8"))
        self.action_min = torch.tensor(statistics["actions_min"], device=context.device, dtype=torch.float32)
        self.action_max = torch.tensor(statistics["actions_max"], device=context.device, dtype=torch.float32)
        self.action_dimension = int(config["data"]["action_dimension"])
        self.action_encoding = str(config["data"]["action_encoding"])
        self.action_chunk_size = int(config["data"]["chunk_size"])
        self.euler_rotation_scale = float(config["data"].get("rotation_scale", 1.0))
        if (
            self.action_min.numel() != self.action_dimension
            or self.action_max.numel() != self.action_dimension
        ):
            raise ValueError(
                f"action statistics must contain {self.action_dimension} dimensions"
            )
        objectives = config["objectives"]
        self.training_mode = str(objectives["training_mode"])
        self.loss_reduction = str(objectives.get("loss_reduction", "masked_mean"))
        if self.training_mode == "policy_only":
            self.probabilities = (1.0, 0.0, 0.0, 0.0)
        elif self.training_mode == "inverse_dynamics_only":
            self.probabilities = (0.0, 0.0, 0.0, 1.0)
        elif self.training_mode == "base_joint":
            self.probabilities = (0.5, 0.25, 0.25, 0.0)
        elif self.training_mode == "joint_with_inverse":
            self.probabilities = (0.4, 0.2, 0.2, 0.2)
        else:
            self.probabilities = (
                float(objectives["policy_probability"]),
                float(objectives["world_probability"]),
                float(objectives["value_probability"]),
                float(objectives.get("inverse_dynamics_probability", 0.0)),
            )
        self.epoch = 0
        self.global_step = 0
        self.samples_seen = 0
        self.batch_in_epoch = 0
        self.validation_episode_summary: dict[str, Any] = {}
        self._last_value_rank: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last_validation_step = -1
        self.last_checkpoint_step = -1
        if resume_path is not None:
            state = load_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=context.device,
                expected_identity=self.data_identity,
            )
            self.epoch = state["epoch"]
            self.global_step = state["global_step"]
            self.samples_seen = state["samples_seen"]
            self.batch_in_epoch = state["batch_in_epoch"]
            if context.is_main:
                print(
                    f"[CKPT] resumed step={self.global_step} epoch={self.epoch} "
                    f"batch_in_epoch={self.batch_in_epoch} samples_seen={self.samples_seen}",
                    flush=True,
                )
        self.prometheus = PrometheusExporter.from_config(
            config, is_main=context.is_main, evaluation_only=evaluation_only
        )
        self.wandb = WandBExporter.from_config(
            config, is_main=context.is_main, evaluation_only=evaluation_only
        )
        if context.is_main:
            self.logger = JsonlLogger(
                self.output_dir, exporters=[self.prometheus, self.wandb]
            )

    def close(self) -> None:
        if self.prometheus is not None:
            self.prometheus.close()
        if getattr(self, "wandb", None) is not None:
            self.wandb.close()

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.context.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _forward(
        self,
        batch: dict[str, Any],
        *,
        fixed_sigma: float | None = None,
        forced_objective: str | None = None,
        condition_ablation: str = "normal",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = self._move(batch)
        if forced_objective is not None:
            try:
                objective_type = OBJECTIVE_NAMES.index(forced_objective)
            except ValueError as error:
                raise ValueError(f"unknown objective: {forced_objective}") from error
            types = torch.full(
                (batch["video"].shape[0],), objective_type,
                device=self.context.device, dtype=torch.long,
            )
        else:
            types = sample_types(batch["video"].shape[0], self.probabilities, self.context.device)
        if condition_ablation == "future_shuffle":
            if "shuffled_future_video" not in batch:
                raise ValueError("future_shuffle requires shuffled_future_video in batch")
            video = batch["video"].clone()
            video[:, :, 5:8] = batch["shuffled_future_video"]
            batch["video"] = video
        condition_mask, loss_mask = temporal_masks(
            types,
            time=batch["video"].shape[2],
            inverse_condition_mode=condition_ablation,
        )
        failed = torch.tensor(
            [str(value).lower() == "failure" for value in batch["outcome"]],
            device=self.context.device,
            dtype=torch.bool,
        )
        loss_mask = apply_failed_action_mask(loss_mask, types, failed)
        embeddings = self.embeddings.get(list(batch["task"]), self.context.device)
        loss, metrics = preencoded_edm_forward(
            self.model, batch, condition_mask, loss_mask, embeddings,
            self.action_min, self.action_max,
            sample_types=types,
            action_metric_mask=((types == POLICY) & ~failed) | (types == INVERSE_DYNAMICS),
            action_encoding=self.action_encoding,
            action_dimension=self.action_dimension,
            action_chunk_size=self.action_chunk_size,
            euler_rotation_scale=self.euler_rotation_scale,
            fixed_sigma=fixed_sigma,
            loss_reduction=self.loss_reduction,
        )
        metrics["policy_samples"] = (types == POLICY).sum().detach()
        metrics["world_samples"] = (types == WORLD).sum().detach()
        metrics["value_samples"] = (types == VALUE).sum().detach()
        metrics["inverse_dynamics_samples"] = (types == INVERSE_DYNAMICS).sum().detach()
        metrics["failure_samples"] = failed.sum().detach()
        self._last_value_rank = pop_value_rank_pairs(metrics)
        return loss, metrics

    def train_one_epoch(self, epoch: int, max_steps: int | None = None) -> bool:
        if self.evaluation_only:
            raise RuntimeError("evaluation-only trainer 禁止训练")
        self.model.train()
        self.train_loader.set_epoch(epoch)
        accumulation = int(self.training["gradient_accumulation_steps"])
        self.optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        starting_samples = self.samples_seen
        previous_log_time = start
        previous_log_step = self.global_step
        number_of_batches = len(self.train_loader.loader)
        skip_until = self.batch_in_epoch if epoch == self.epoch else 0
        if self.context.is_main:
            print(
                f"[TRAIN] epoch={epoch} step={self.global_step} "
                f"batches={number_of_batches} skip={skip_until} accum={accumulation}",
                flush=True,
            )
        final_group_size = number_of_batches % accumulation or accumulation
        group_metric_sums: dict[str, torch.Tensor] = {}
        group_local_samples = 0
        for batch_index, batch in enumerate(self.train_loader.loader):
            if skip_until and batch_index < skip_until:
                if self.context.is_main:
                    print(
                        f"[RESUME] skip batch {batch_index + 1}/{skip_until}",
                        flush=True,
                    )
                continue
            sync_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == number_of_batches
            divisor = final_group_size if batch_index >= number_of_batches - final_group_size else accumulation
            no_sync = (
                self.model.net.no_sync()
                if isinstance(self.model.net, DDP) and not sync_step
                else nullcontext()
            )
            track_memory = torch.cuda.is_available()
            if track_memory:
                torch.cuda.reset_peak_memory_stats(self.context.device)
                mem_before_fwd = torch.cuda.memory_allocated(self.context.device) / 2**20
            with no_sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = self._forward(batch)
                scaled_loss = loss / divisor
            local_batch_size = int(batch["video"].shape[0])
            group_local_samples += local_batch_size
            for name, value in metrics.items():
                contribution = value.detach().float()
                if name not in self.SAMPLE_COUNT_METRICS:
                    contribution = contribution * local_batch_size
                group_metric_sums[name] = group_metric_sums.get(name, 0) + contribution
            if track_memory:
                mem_after_fwd = torch.cuda.memory_allocated(self.context.device) / 2**20
                peak_fwd = torch.cuda.max_memory_allocated(self.context.device) / 2**20
                torch.cuda.reset_peak_memory_stats(self.context.device)
            self.scaler.scale(scaled_loss).backward()
            if track_memory:
                mem_after_bwd = torch.cuda.memory_allocated(self.context.device) / 2**20
                peak_bwd = torch.cuda.max_memory_allocated(self.context.device) / 2**20
            if sync_step:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.net.parameters(), float(self.training.get("gradient_clip_norm", 10.0))
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.batch_in_epoch = batch_index + 1
                update_samples = group_local_samples * self.context.world_size
                self.samples_seen += update_samples
                if should_run_periodic(
                    self.global_step, int(self.training.get("log_every_steps", 10))
                ):
                    now = time.monotonic()
                    logged_steps = max(1, self.global_step - previous_log_step)
                    total_samples = torch.tensor(
                        float(group_local_samples), device=self.context.device
                    )
                    if self.context.enabled:
                        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
                    reduced: dict[str, float] = {}
                    for name, value in group_metric_sums.items():
                        tensor = value.clone()
                        if self.context.enabled:
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        if name not in self.SAMPLE_COUNT_METRICS:
                            tensor /= total_samples.clamp_min(1)
                        reduced[name] = tensor.item()
                    reduced["gradient_norm"] = float(grad_norm)
                    reduced["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
                    reduced["loss_scale"] = float(self.scaler.get_scale())
                    reduced["step_time_seconds"] = (now - previous_log_time) / logged_steps
                    if track_memory:
                        reduced.update(
                            gpu_memory_allocated_gib=torch.cuda.max_memory_allocated(
                                self.context.device
                            ) / 2**30,
                            gpu_fwd_delta_mib=mem_after_fwd - mem_before_fwd,
                            gpu_fwd_peak_mib=peak_fwd,
                            gpu_bwd_delta_mib=mem_after_bwd - mem_after_fwd,
                            gpu_bwd_peak_mib=peak_bwd,
                        )
                    reduced.update(
                        mode="train", epoch=epoch, global_step=self.global_step,
                        samples_seen=self.samples_seen,
                        samples_per_second=(self.samples_seen - starting_samples)
                        / max(now - start, 1e-6),
                    )
                    previous_log_time = now
                    previous_log_step = self.global_step
                    if self.logger:
                        self.logger.log(reduced)
                        print(f"[TRAIN] step={self.global_step} loss={reduced['total_edm_loss']:.6f}")
                group_metric_sums = {}
                group_local_samples = 0
                checkpoint_every = int(self.training.get("checkpoint_every_steps", 1000))
                if should_run_periodic(self.global_step, checkpoint_every):
                    if self.context.is_main:
                        self.save(epoch, self.batch_in_epoch)
                    self.last_checkpoint_step = self.global_step
                    if self.context.enabled:
                        dist.barrier()
                validation_every = int(self.training.get("validation_every_steps", 0))
                if should_run_periodic(self.global_step, validation_every):
                    self.run_step_validation(epoch)
                if max_steps is not None and self.global_step >= max_steps:
                    return True
        return False

    @torch.no_grad()
    def validate_learner_actor(
        self, epoch: int, max_batches: int | None = None
    ) -> dict[str, float]:
        """Whole-split Policy validation matching learner_copy_dist loss semantics."""
        was_training = self.model.training
        self.model.eval()
        evaluation = self.config["evaluation"]
        seed = int(evaluation.get("noise_seed", 20260827)) + self.context.rank
        progress_every = max(1, int(evaluation.get("progress_every_batches", 100)))
        source_metrics = {
            "eval_loss_actor": "loss_actor_learner",
            "eval_policy_masked_edm_loss": "policy_masked_edm_loss",
            "eval_future_image_l1_loss": "future_image_l1",
            "eval_future_wrist_image_l1_loss": "future_wrist_image_l1",
            "eval_future_proprio_l1_loss": "future_proprio_l1",
            "eval_action_l1_loss": "action_l1",
            "eval_value_l1_loss": "value_l1",
        }
        local_sums = torch.zeros(
            len(source_metrics), dtype=torch.float64, device=self.context.device
        )
        local_count = 0
        available = len(self.val_loader.loader)
        limit = available if max_batches is None else min(max_batches, available)
        started = time.monotonic()
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[self.context.device.index]), unwrapped_ddp_net(
                self.model
            ):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                for batch_index, batch in enumerate(self.val_loader.loader):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, metrics = self._forward(batch, forced_objective="policy")
                    batch_size = int(batch["video"].shape[0])
                    for metric_index, source in enumerate(source_metrics.values()):
                        local_sums[metric_index] += metrics[source].detach().double() * batch_size
                    local_count += batch_size
                    completed = batch_index + 1
                    if self.context.is_main and (
                        completed % progress_every == 0 or completed == limit
                    ):
                        elapsed = time.monotonic() - started
                        print(
                            f"[EVAL-LEARNER] step={self.global_step} "
                            f"batch={completed}/{limit} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
        finally:
            np.random.set_state(numpy_state)
            self.model.train(was_training)
        stats = torch.cat(
            [local_sums, torch.tensor([local_count], dtype=torch.float64, device=self.context.device)]
        )
        if self.context.enabled:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        sample_count = stats[-1].item()
        if sample_count <= 0:
            raise RuntimeError("learner_actor validation split is empty")
        values = {
            name: stats[index].item() / sample_count
            for index, name in enumerate(source_metrics)
        }
        values["total_edm_loss"] = values["eval_loss_actor"]
        self.last_validation_step = self.global_step
        if self.logger:
            self.logger.log(
                {
                    "mode": "eval",
                    "protocol": "learner_actor",
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "validation_samples": int(sample_count),
                    **values,
                }
            )
            print(
                f"[VAL] step={self.global_step} samples={int(sample_count)} "
                f"loss_actor={values['eval_loss_actor']:.6f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        return values

    def run_configured_validation(self, epoch: int) -> dict[str, float]:
        protocol = str(self.config["evaluation"].get("protocol", "fixed_suite"))
        limit = validation_batch_limit(self.config)
        if protocol == "learner_actor":
            return self.validate_learner_actor(epoch, max_batches=limit)
        return self.validate(epoch, max_batches=limit)

    def run_step_validation(self, epoch: int) -> dict[str, float]:
        """Use learner full split or the configured lightweight fixed probe."""
        evaluation = self.config["evaluation"]
        if str(evaluation.get("protocol", "fixed_suite")) == "learner_actor":
            return self.validate_learner_actor(
                epoch, max_batches=validation_batch_limit(self.config)
            )
        original_sigmas = evaluation["sigma_values"]
        original_objectives = evaluation.get("objectives")
        was_training = self.model.training
        try:
            evaluation["sigma_values"] = evaluation["step_validation_sigma_values"]
            evaluation["objectives"] = evaluation["step_validation_objectives"]
            return self.validate(
                epoch, max_batches=int(evaluation["step_validation_max_batches"])
            )
        finally:
            evaluation["sigma_values"] = original_sigmas
            evaluation["objectives"] = original_objectives
            self.model.train(was_training)

    @torch.no_grad()
    def validate(self, epoch: int, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        values: dict[str, float] = {}
        sigma_values = [float(value) for value in self.config["evaluation"]["sigma_values"]]
        validation_seed = int(self.config["evaluation"].get("noise_seed", 20260820))
        total_losses: list[float] = []
        collect_episode_metrics = bool(
            self.config["evaluation"].get("episode_aggregation", False)
        )
        episode_rows: list[dict[str, Any]] = []
        evaluation_objectives = self.config["evaluation"].get("objectives")
        if evaluation_objectives is None:
            evaluation_objectives = list(OBJECTIVE_NAMES)
        # Each sigma uses fixed samples and a fixed noise stream. Validation does
        # not consume training RNG and is reproducible across checkpoints/resume.
        inverse_ablations = [
            str(value)
            for value in self.config["evaluation"].get(
                "inverse_condition_ablations", ["normal"]
            )
        ]
        condition_total = len(sigma_values) * sum(
            len(inverse_ablations) if objective == "inverse_dynamics" else 1
            for objective in evaluation_objectives
        )
        condition_index = 0
        evaluation_start = time.monotonic()
        progress_every = max(
            1, int(self.config["evaluation"].get("progress_every_batches", 10))
        )
        for sigma_index, sigma in enumerate(sigma_values):
            for objective_index, objective in enumerate(evaluation_objectives):
                ablations = inverse_ablations if objective == "inverse_dynamics" else ["normal"]
                ablation_results: dict[str, dict[str, float]] = {}
                for ablation_index, ablation in enumerate(ablations):
                    condition_index += 1
                    active_loader = self.val_loader
                    if ablation == "future_shuffle":
                        if self.inverse_shuffle_loader is None:
                            raise ValueError(
                                "future_shuffle requested without inverse shuffle loader"
                            )
                        active_loader = self.inverse_shuffle_loader
                    available_batches = len(active_loader.loader)
                    condition_batches = (
                        available_batches
                        if max_batches is None
                        else min(max_batches, available_batches)
                    )
                    condition_start = time.monotonic()
                    if self.context.is_main:
                        print(
                            f"[EVAL] condition={condition_index}/{condition_total} "
                            f"step={self.global_step} sigma={sigma:g} "
                            f"objective={objective} ablation={ablation} "
                            f"batches={condition_batches}",
                            flush=True,
                        )
                    accumulator = MeanMetrics()
                    value_pred_chunks: list[torch.Tensor] = []
                    value_target_chunks: list[torch.Tensor] = []
                    with torch.random.fork_rng(devices=[self.context.device.index]):
                        seed = (
                            validation_seed + sigma_index * 100
                            + objective_index * 10 + self.context.rank
                        )
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        for batch_index, batch in enumerate(active_loader.loader):
                            if max_batches is not None and batch_index >= max_batches:
                                break
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                _, metrics = self._forward(
                                    batch,
                                    fixed_sigma=sigma,
                                    forced_objective=str(objective),
                                    condition_ablation=ablation,
                                )
                            reduced = reduce_metrics(
                                metrics, self.context, sum_names=self.SAMPLE_COUNT_METRICS
                            )
                            accumulator.update(reduced, weight=float(batch["video"].shape[0]))
                            if (
                                objective == "value"
                                and self._last_value_rank is not None
                                and int(self._last_value_rank[0].numel()) > 0
                            ):
                                value_pred_chunks.append(self._last_value_rank[0])
                                value_target_chunks.append(self._last_value_rank[1])
                            completed_batches = batch_index + 1
                            if self.context.is_main and (
                                completed_batches % progress_every == 0
                                or completed_batches == condition_batches
                            ):
                                elapsed = time.monotonic() - condition_start
                                rate = completed_batches / max(elapsed, 1e-6)
                                eta = (condition_batches - completed_batches) / max(rate, 1e-6)
                                print(
                                    f"[EVAL] condition={condition_index}/{condition_total} "
                                    f"batch={completed_batches}/{condition_batches} "
                                    f"rate={rate:.2f} batch/s elapsed={elapsed:.1f}s "
                                    f"ETA={eta:.1f}s",
                                    flush=True,
                                )
                            if collect_episode_metrics:
                                if int(batch["video"].shape[0]) != 1:
                                    raise ValueError(
                                        "episode aggregation requires batch_size_per_rank=1"
                                    )
                                prefix = f"sigma_{sigma:g}/{objective}"
                                if objective == "inverse_dynamics":
                                    prefix += f"/{ablation}"
                                episode_rows.append(
                                    {
                                        "episode_path": str(batch["episode_path"][0]),
                                        "metrics": {
                                            f"{prefix}/{name}": value
                                            for name, value in reduced.items()
                                        },
                                    }
                                )
                    objective_metrics = accumulator.compute()
                    if objective == "value":
                        rank_pred, rank_target = gather_value_rank_arrays(
                            value_pred_chunks, value_target_chunks, self.context
                        )
                        spearman, count = pooled_value_spearman(rank_pred, rank_target)
                        objective_metrics["value_spearman"] = spearman
                        objective_metrics["value_spearman_n"] = float(count)
                    ablation_results[ablation] = objective_metrics
                    prefix = f"sigma_{sigma:g}/{objective}"
                    if objective == "inverse_dynamics":
                        prefix += f"/{ablation}"
                    values.update(
                        {f"{prefix}/{name}": value for name, value in objective_metrics.items()}
                    )
                    if "total_edm_loss" in objective_metrics:
                        total_losses.append(objective_metrics["total_edm_loss"])
                    if self.context.is_main:
                        print(
                            f"[EVAL] condition={condition_index}/{condition_total} complete "
                            f"loss={objective_metrics.get('total_edm_loss', float('nan')):.6f} "
                            f"elapsed={time.monotonic() - condition_start:.1f}s",
                            flush=True,
                        )
                if objective == "inverse_dynamics" and {
                    "normal", "future_shuffle"
                }.issubset(ablation_results):
                    normal = ablation_results["normal"]
                    shuffled = ablation_results["future_shuffle"]
                    for name in sorted(set(normal) & set(shuffled)):
                        if name.startswith(("action_", "left_", "right_")):
                            values[
                                f"sigma_{sigma:g}/inverse_dynamics/future_information_gain/{name}"
                            ] = shuffled[name] - normal[name]
        values["total_edm_loss"] = (
            sum(total_losses) / len(total_losses) if total_losses else float("nan")
        )
        if collect_episode_metrics:
            if self.context.enabled:
                gathered: list[list[dict[str, Any]] | None] = [
                    None for _ in range(self.context.world_size)
                ]
                dist.all_gather_object(gathered, episode_rows)
                episode_rows = [
                    row for rank_rows in gathered if rank_rows is not None for row in rank_rows
                ]
            self.validation_episode_summary = aggregate_episode_metrics(
                episode_rows,
                bootstrap_samples=int(
                    self.config["evaluation"].get("bootstrap_samples", 2000)
                ),
            )
        if self.logger:
            self.logger.log({"mode": "validation", "epoch": epoch, "global_step": self.global_step, **values})
            print(
                f"[VAL] epoch={epoch} loss={values.get('total_edm_loss', float('nan')):.6f} "
                f"elapsed={time.monotonic() - evaluation_start:.1f}s",
                flush=True,
            )
        return values

    def save(self, epoch: int, batch_in_epoch: int = 0) -> None:
        if self.evaluation_only:
            raise RuntimeError("evaluation-only trainer 禁止保存训练checkpoint")
        path = self.output_dir / "checkpoints" / f"step_{self.global_step:09d}.pt"
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            global_step=self.global_step,
            samples_seen=self.samples_seen,
            config=self.config,
            data_identity=self.data_identity,
        )
        latest = path.parent / "latest.json"
        temporary = latest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"checkpoint": path.name, "global_step": self.global_step}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(latest)

    def smoke_overfit(self) -> None:
        """Repeat one batch with fixed diffusion noise; fail if basic learning is broken."""
        self.model.train()
        self.train_loader.set_epoch(0)
        batch = next(iter(self.train_loader.loader))
        steps = int(self.runtime.get("smoke_max_train_steps", 10))
        losses: list[float] = []
        self.optimizer.zero_grad(set_to_none=True)
        for step in range(steps):
            torch.manual_seed(1234 + self.context.rank)
            torch.cuda.manual_seed_all(1234 + self.context.rank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = self._forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"smoke loss is not finite at step {step}: {loss.item()}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.net.parameters(), float(self.training.get("gradient_clip_norm", 10.0))
            )
            grad_norm_value = float(grad_norm)
            if not math.isfinite(grad_norm_value):
                raise RuntimeError(
                    f"smoke gradient is not finite at step {step}: {grad_norm_value}"
                )
            if grad_norm_value <= 0:
                raise RuntimeError(f"smoke gradient is zero at step {step}")
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise RuntimeError(f"smoke learning rate is invalid: {learning_rate}")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self.samples_seen += int(batch["video"].shape[0]) * self.context.world_size
            reduced = reduce_metrics(
                metrics, self.context, sum_names=self.SAMPLE_COUNT_METRICS
            )
            reduced["gradient_norm"] = grad_norm_value
            reduced["learning_rate"] = learning_rate
            losses.append(reduced["total_edm_loss"])
            if self.context.is_main:
                if self.logger:
                    self.logger.log(
                        {
                            "mode": "smoke_train",
                            "training_mode": self.training_mode,
                            "global_step": self.global_step,
                            "samples_seen": self.samples_seen,
                            **reduced,
                        }
                    )
                print(
                    f"[SMOKE] step={step + 1}/{steps} loss={losses[-1]:.6f} "
                    f"lr={learning_rate:.3e} grad_norm={grad_norm_value:.4f}"
                )
        first_mean, last_mean, window = smoke_loss_window_means(losses)
        if last_mean >= first_mean:
            raise RuntimeError(
                "fixed-batch mean loss did not decrease: "
                f"first_{window}_mean={first_mean:.6f}, "
                f"last_{window}_mean={last_mean:.6f}"
            )
        if self.context.is_main:
            relative_drop = (first_mean - last_mean) / max(abs(first_mean), 1e-12)
            print(
                f"[SMOKE] OPTIMIZATION PASS window={window} "
                f"first_mean={first_mean:.6f} last_mean={last_mean:.6f} "
                f"relative_drop={relative_drop:.2%}"
            )

    def fit(self, smoke: bool = False) -> None:
        if smoke:
            self.smoke_overfit()
            smoke_batches = int(self.runtime.get("smoke_max_validation_batches", 2))
            protocol = str(self.config["evaluation"].get("protocol", "fixed_suite"))
            if protocol == "learner_actor":
                self.validate_learner_actor(0, max_batches=smoke_batches)
            else:
                self.validate(0, max_batches=smoke_batches)
            if self.context.is_main:
                self.save(0, 0)
                print("[SMOKE] VALIDATION/CHECKPOINT PASS")
            return
        epochs = int(self.training["epochs"])
        configured_max = self.training.get("max_steps")
        max_steps = int(configured_max) if configured_max is not None else None
        validation_every = int(self.training.get("validation_every_epochs", 1))
        if self.global_step == 0 and bool(
            self.config["evaluation"].get("evaluate_base_before_training", False)
        ):
            self.run_configured_validation(self.epoch)
        completed_epoch = self.epoch
        for epoch in range(self.epoch, epochs):
            stopped = self.train_one_epoch(epoch, max_steps=max_steps)
            completed_epoch = epoch + 1
            if validation_every > 0 and (epoch + 1) % validation_every == 0:
                self.run_configured_validation(epoch)
            if stopped and bool(self.training.get("validate_at_end", True)):
                if self.last_validation_step != self.global_step:
                    self.run_configured_validation(epoch)
            self.batch_in_epoch = 0
            if bool(self.training.get("save_every_epochs", True)):
                if self.context.is_main:
                    self.save(epoch + 1, 0)
                self.last_checkpoint_step = self.global_step
                if self.context.enabled:
                    dist.barrier()
            if stopped:
                break
        if bool(self.training.get("checkpoint_at_end", True)):
            if self.last_checkpoint_step != self.global_step:
                if self.context.is_main:
                    self.save(completed_epoch, self.batch_in_epoch)
                if self.context.enabled:
                    dist.barrier()
        if self.context.is_main and bool(self.runtime.get("plot_after_training", False)):
            from .plot_metrics import generate

            generate(self.logger.path, self.output_dir / "plots")
