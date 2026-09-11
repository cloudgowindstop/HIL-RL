"""Offline training command entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import (
    ConfigError,
    load_config,
    resolve_pre_split_directories,
    validate_training_config,
    validation_batch_limit,
)
from .checkpoint import load_checkpoint_for_evaluation
from .components.optimization import build_grad_scaler, build_optimizer_scheduler
from .dataloader import build_loader
from .dataset import (
    CosmosParquetDataset,
    build_pre_split_index,
    build_episode_index,
    load_manifest,
    manifest_sha256,
    save_manifest,
    select_episode_subset,
    split_episode_index,
)
from .distributed import cleanup, initialize
from .evaluation.datasets import InverseFutureShuffleDataset
from .model import checkpoint_hash, load_model
from .metrics import JsonlLogger
from .preflight import print_report, run_preflight
from .pipeline.trainer import CosmosOfflineTrainer
from .pipeline.train_pipeline import prepare_splits as run_prepare_splits
from .pipeline.train_pipeline import run as run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "preflight", "prepare-splits", "test-cpu", "smoke-single",
            "smoke-ddp", "train", "resume", "validate", "validate-base",
        ),
    )
    parser.add_argument("--resume")
    return parser.parse_args()


def _persist_manifests(
    splits: tuple[list, list, list],
    paths: tuple[Path, Path, Path | None],
    *,
    overwrite: bool,
) -> tuple[list, list, list]:
    """Write new manifests, or reuse complete manifests from an earlier run."""
    required_paths = [path for path in paths if path is not None]
    existing = [path for path in required_paths if path.exists()]
    if existing and len(existing) != len(required_paths) and not overwrite:
        raise ConfigError(f"manifest 只存在一部分，拒绝覆盖: {existing}")
    if existing and not overwrite:
        loaded = [load_manifest(path) if path is not None else [] for path in paths]
        return tuple(loaded)
    for records, path in zip(splits, paths):
        if path is not None:
            save_manifest(records, path, overwrite=overwrite)
    return splits


def _records(config: dict, output_dir: Path):
    data = config["data"]
    names = ("train_manifest", "val_manifest", "test_manifest")
    configured = [data.get(name) for name in names]
    train_path = Path(configured[0]).expanduser().resolve() if configured[0] else None
    val_path = Path(configured[1]).expanduser().resolve() if configured[1] else None
    test_path = Path(configured[2]).expanduser().resolve() if configured[2] else None
    if train_path and val_path and train_path.is_file() and val_path.is_file():
        if test_path and not test_path.is_file():
            raise ConfigError(f"test manifest 已配置但不存在: {test_path}")
        return (
            load_manifest(train_path),
            load_manifest(val_path),
            load_manifest(test_path) if test_path else [],
        )
    existing = [path for path in (train_path, val_path, test_path) if path and path.exists()]
    if existing:
        raise ConfigError(f"manifest 只存在一部分，拒绝自动覆盖: {existing}")

    split = config["split"]
    if data.get("layout", "unified") == "pre_split":
        root = Path(data["root"]).expanduser().resolve()
        train_root, val_root, test_root = resolve_pre_split_directories(data, root)
        splits = build_pre_split_index(train_root, val_root, test_root)
        seed = int(split.get("seed", config["training"]["seed"]))
        train = select_episode_subset(
            splits.train, int(data.get("max_train_episodes", 0)), seed
        )
        val = select_episode_subset(
            splits.validation, int(data.get("max_val_episodes", 0)), seed
        )
        test = select_episode_subset(
            splits.test, int(data.get("max_test_episodes", 0)), seed
        )
        split_dir = output_dir / "splits"
        paths = (
            train_path or split_dir / "train.json",
            val_path or split_dir / "val.json",
            test_path or (split_dir / "test.json" if test else None),
        )
        train, val, test = _persist_manifests(
            (train, val, test), paths, overwrite=bool(split.get("overwrite", False))
        )
        data["train_manifest"] = str(paths[0])
        data["val_manifest"] = str(paths[1])
        data["test_manifest"] = str(paths[2]) if paths[2] else None
        return train, val, test

    if not bool(split.get("create_if_missing", False)):
        raise ConfigError("manifest 不存在且 split.create_if_missing=false")
    records = build_episode_index(data["root"])
    seed = int(split.get("seed", config["training"]["seed"]))
    selected = select_episode_subset(records, int(split["max_episodes"]), seed)
    ratios = tuple(float(value) for value in split["ratios"])
    train, val, test = split_episode_index(selected, seed=seed, ratios=ratios)
    split_dir = output_dir / "splits"
    paths = (
        train_path or split_dir / "train.json",
        val_path or split_dir / "val.json",
        test_path or (split_dir / "test.json" if ratios[2] > 0 else None),
    )
    train, val, test = _persist_manifests(
        (train, val, test), paths, overwrite=bool(split.get("overwrite", False))
    )
    data["train_manifest"] = str(paths[0])
    data["val_manifest"] = str(paths[1])
    data["test_manifest"] = str(paths[2]) if paths[2] else None
    return train, val, test


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_fingerprint(root: str | Path) -> str:
    root = Path(root).expanduser().resolve()
    manifests = sorted(root.rglob("meta/conversion_manifest.json"))
    values: list[str] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.append(str(payload.get("fingerprint_sha256", _file_sha256(path))))
    if not values:
        metadata = sorted(root.rglob("cosmos_dataset_metadata.json"))
        values = [_file_sha256(path) for path in metadata]
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _build_identity(config: dict) -> dict[str, str]:
    data = config["data"]
    identity = {
        "train_manifest_sha256": manifest_sha256(data["train_manifest"]),
        "val_manifest_sha256": manifest_sha256(data["val_manifest"]),
        "dataset_fingerprint": _dataset_fingerprint(data["root"]),
        "statistics_sha256": _file_sha256(data["statistics_path"]),
        "t5_sha256": _file_sha256(data["t5_embeddings_path"]),
        "action_encoding": str(data["action_encoding"]),
        "action_dimension": str(data["action_dimension"]),
        "data_layout": str(data.get("layout", "unified")),
    }
    if data.get("test_manifest"):
        identity["test_manifest_sha256"] = manifest_sha256(data["test_manifest"])
    base_hash = checkpoint_hash(Path(config["model"]["checkpoint_path"]).expanduser().resolve())
    identity["base_checkpoint_sha256"] = base_hash
    identity["tokenizer_sha256"] = _file_sha256(config["model"]["tokenizer_path"])
    config["_base_checkpoint_sha256"] = base_hash
    return identity


def _loader_common(config: dict, context, *, workers: int) -> dict:
    data = config["data"]
    training = config["training"]
    return dict(
        batch_size=int(training["batch_size_per_rank"]),
        workers=workers,
        seed=int(training["seed"]),
        distributed=context.enabled,
        rank=context.rank,
        world_size=context.world_size,
        pin_memory=bool(data["pin_memory"]),
        persistent_workers=bool(data["persistent_workers"]),
        prefetch_factor=int(data.get("prefetch_factor", 2)),
    )


def _run_training(config: dict, args: argparse.Namespace) -> None:
    validate_training_config(config)
    report = run_preflight(config)
    if int(os.environ.get("RANK", 0)) == 0:
        print_report(report)
    if not report.passed:
        raise ConfigError("数据 preflight 未通过，拒绝启动训练")
    context = initialize()
    trainer = None
    try:
        seed = int(config["training"]["seed"]) + context.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # 性能开关：固定 shape 的 DiT 训练（matmul 为主）下 TF32 加速矩阵乘；
        # cudnn.benchmark 为卷积类算子自动选择最快算法。
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        runtime = config["runtime"]
        output_value = runtime["output_dir"]
        if args.mode in ("smoke-single", "smoke-ddp") and runtime.get("smoke_output_dir"):
            output_value = runtime["smoke_output_dir"]
            runtime["output_dir"] = output_value
        output_dir = Path(output_value).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        train_records, val_records, _ = _records(config, output_dir)
        if not train_records or not val_records:
            raise ConfigError(
                f"split 为空: train episodes={len(train_records)}, val episodes={len(val_records)}。"
                "至少需要 2 个 episode 以生成 train/validation split。"
            )
        data_identity = _build_identity(config)
        config["identity"] = data_identity
        if context.is_main:
            (output_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=False),
                encoding="utf-8",
            )
            print(f"[DATA] train_episodes={len(train_records)} val_episodes={len(val_records)}")
        action_dimension = int(config["data"]["action_dimension"])
        train_dataset = CosmosParquetDataset(
            train_records, action_dimension=action_dimension
        )
        val_dataset = CosmosParquetDataset(
            val_records, action_dimension=action_dimension
        )
        data_config = config["data"]
        training = config["training"]
        workers = int(data_config["workers_per_rank"])
        if args.mode in ("smoke-single", "smoke-ddp"):
            workers = 0
        common = _loader_common(config, context, workers=workers)
        train_loader = build_loader(train_dataset, train=True, **common)
        val_loader = build_loader(
            val_dataset,
            train=False,
            distributed_eval_no_padding=(
                str(config["evaluation"].get("protocol", "fixed_suite"))
                == "learner_actor"
            ),
            **common,
        )
        inverse_shuffle_loader = None
        if "future_shuffle" in config["evaluation"].get(
            "inverse_condition_ablations", []
        ):
            inverse_shuffle_loader = build_loader(
                InverseFutureShuffleDataset(val_dataset),
                train=False,
                distributed_eval_no_padding=(
                    str(config["evaluation"].get("protocol", "fixed_suite"))
                    == "learner_actor"
                ),
                **common,
            )
        model, official_config = load_model(config, context.device)
        if context.enabled:
            model.net = DDP(
                model.net,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                find_unused_parameters=False,
            )
        optimizer, scheduler = build_optimizer_scheduler(model, official_config)
        if context.is_main:
            print(
                f"[OPTIMIZER] base_lr={float(training['learning_rate']):.8g} "
                f"initial_scheduled_lr={optimizer.param_groups[0]['lr']:.8g}"
            )
        # BF16 training must not enter FP16 GradScaler's CUDA unscale kernel.
        scaler = build_grad_scaler()
        resume = Path(args.resume).expanduser().resolve() if args.resume else None
        trainer = CosmosOfflineTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            context=context,
            resume_path=resume,
            data_identity=data_identity,
            inverse_shuffle_loader=inverse_shuffle_loader,
        )
        trainer.fit(smoke=args.mode in ("smoke-single", "smoke-ddp"))
    finally:
        if trainer is not None:
            trainer.close()
        cleanup()


def _prepare_validation(config: dict):
    """Build validation data, model, VAE/tokenizer, T5 provider, and trainer once."""
    validate_training_config(config)
    report = run_preflight(config)
    print_report(report)
    if not report.passed:
        raise ConfigError("数据 preflight 未通过，拒绝验证")
    context = initialize()
    output_dir = Path(config["runtime"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, val_records, _ = _records(config, output_dir)
    if not val_records:
        raise ConfigError("validation split 为空")
    data_identity = _build_identity(config)
    config["identity"] = data_identity
    # Episode-level bootstrap needs one sample per batch. Training compare
    # configs use batch_size_per_rank=2; keep mean metrics and skip aggregation.
    config["evaluation"]["episode_aggregation"] = (
        int(config["training"]["batch_size_per_rank"]) == 1
    )
    common = _loader_common(config, context, workers=0)
    learner_protocol = (
        str(config["evaluation"].get("protocol", "fixed_suite")) == "learner_actor"
    )
    common["episode_balanced"] = not learner_protocol
    common["distributed_eval_no_padding"] = learner_protocol
    validation_limit = validation_batch_limit(config)
    if validation_limit is not None:
        common["episode_balanced_max_samples"] = (
            validation_limit
            * int(config["training"]["batch_size_per_rank"])
            * context.world_size
        )
    val_dataset = CosmosParquetDataset(
        val_records,
        cache_episodes=int(config["evaluation"].get("cache_episodes", 2)),
        action_dimension=int(config["data"]["action_dimension"]),
    )
    val_loader = build_loader(val_dataset, train=False, **common)
    inverse_shuffle_loader = None
    if "future_shuffle" in config["evaluation"].get(
        "inverse_condition_ablations", []
    ):
        inverse_shuffle_loader = build_loader(
            InverseFutureShuffleDataset(val_dataset), train=False, **common
        )
    model, _ = load_model(config, context.device)
    trainer = CosmosOfflineTrainer(
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        train_loader=val_loader,
        val_loader=val_loader,
        config=config,
        context=context,
        data_identity=data_identity,
        evaluation_only=True,
        inverse_shuffle_loader=inverse_shuffle_loader,
    )
    return context, model, trainer, data_identity


def _write_validation_outputs(
    trainer: CosmosOfflineTrainer,
    values: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / f"validation_summary_step_{trainer.global_step:09d}.json"
    summary.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[VAL] summary={summary}")
    episode_summary = (
        output_dir / f"validation_episode_summary_step_{trainer.global_step:09d}.json"
    )
    episode_summary.write_text(
        json.dumps(trainer.validation_episode_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[VAL] episode_summary={episode_summary}")


def _run_validation(config: dict, args: argparse.Namespace) -> None:
    evaluate_base = args.mode == "validate-base"
    if not evaluate_base and not args.resume:
        raise ConfigError("validate 需要 --resume CHECKPOINT")
    context = None
    try:
        context, model, trainer, data_identity = _prepare_validation(config)
        output_dir = Path(config["runtime"]["output_dir"]).expanduser().resolve()
        if evaluate_base:
            state = {"epoch": 0, "global_step": 0}
        else:
            state = load_checkpoint_for_evaluation(
                Path(args.resume).expanduser().resolve(),
                model=model,
                device=context.device,
                expected_identity=data_identity,
            )
        trainer.global_step = state["global_step"]
        values = trainer.validate(state["epoch"], max_batches=validation_batch_limit(config))
        if context.is_main:
            _write_validation_outputs(trainer, values, output_dir)
    finally:
        if context is not None:
            cleanup()


def _run_validation_suite(
    config: dict,
    checkpoints: list[tuple[str, Path]],
    *,
    include_base: bool,
) -> None:
    """Evaluate Base and checkpoints while reusing one model/VAE/T5/data runtime."""
    if not include_base and not checkpoints:
        raise ConfigError("validation suite requires Base or at least one checkpoint")
    context = None
    try:
        context, model, trainer, data_identity = _prepare_validation(config)
        output_root = Path(config["runtime"]["output_dir"]).expanduser().resolve()
        targets: list[tuple[str, Path | None]] = []
        if include_base:
            targets.append(("base", None))
        targets.extend(checkpoints)
        for index, (name, checkpoint) in enumerate(targets, start=1):
            output_dir = output_root / name
            if checkpoint is None:
                state = {"epoch": 0, "global_step": 0}
                source = "Base"
            else:
                state = load_checkpoint_for_evaluation(
                    checkpoint,
                    model=model,
                    device=context.device,
                    expected_identity=data_identity,
                )
                source = str(checkpoint)
            trainer.global_step = state["global_step"]
            trainer.logger = JsonlLogger(output_dir) if context.is_main else None
            print(
                f"[SUITE] model={index}/{len(targets)} name={name} "
                f"step={trainer.global_step} source={source}",
                flush=True,
            )
            values = trainer.validate(
                state["epoch"], max_batches=validation_batch_limit(config)
            )
            if context.is_main:
                _write_validation_outputs(trainer, values, output_dir)
            torch.cuda.empty_cache()
    finally:
        if context is not None:
            cleanup()


def _prepare_splits(config: dict) -> None:
    validate_training_config(config)
    report = run_preflight(config)
    print_report(report)
    if not report.passed:
        raise ConfigError("数据 preflight 未通过，拒绝划分")
    output_dir = Path(config["runtime"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = _records(config, output_dir)
    identity = {
        "train_manifest_sha256": manifest_sha256(config["data"]["train_manifest"]),
        "val_manifest_sha256": manifest_sha256(config["data"]["val_manifest"]),
        "dataset_fingerprint": _dataset_fingerprint(config["data"]["root"]),
    }
    if config["data"].get("test_manifest"):
        identity["test_manifest_sha256"] = manifest_sha256(config["data"]["test_manifest"])
    print(f"[SPLIT] train={len(train)} validation={len(val)} test={len(test)}")
    print(json.dumps(identity, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.mode == "preflight":
            report = run_preflight(config)
            print_report(report)
            if not report.passed:
                raise SystemExit(2)
            return
        if args.mode == "test-cpu":
            from .tests.run_cpu_tests import run

            run()
            return
        if args.mode == "prepare-splits":
            run_prepare_splits(config)
            return
        if args.mode in {"validate", "validate-base"}:
            _run_validation(config, args)
            return
        run_training_pipeline(config, args)
    except ConfigError as error:
        raise SystemExit(f"[ERROR] {error}") from None


if __name__ == "__main__":
    main()
