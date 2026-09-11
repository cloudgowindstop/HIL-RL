"""Configuration loading and checks that do not require CUDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .action_spec import action_spec_from_data


class ConfigError(ValueError):
    """User-facing experiment configuration error."""


def validation_batch_limit(config: dict[str, Any]) -> int | None:
    """Return configured validation cap; ``None`` explicitly means full split."""
    evaluation = require_mapping(config, "evaluation")
    value = evaluation.get("max_validation_batches")
    if value is None:
        return None
    limit = int(value)
    if limit < 1:
        raise ConfigError("evaluation.max_validation_batches 必须 >= 1 或 null")
    return limit


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"配置不存在: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigError(f"配置根节点必须是 mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"缺少配置段: {key}")
    return value


def resolve_data_layout(data: dict[str, Any], root: Path) -> str:
    """Resolve unified or already-split layout without changing old defaults."""
    requested = str(data.get("layout", "unified"))
    if requested not in {"unified", "pre_split", "auto"}:
        raise ConfigError("data.layout 必须是 unified、pre_split 或 auto")
    if requested != "auto":
        return requested
    split_names = {"train", "eval", "val", "validation", "test"}
    has_pre_split = (root / "train").is_dir() and any(
        (root / name).is_dir() for name in ("eval", "val", "validation")
    )
    metadata_paths = root.rglob("cosmos_dataset_metadata.json")
    has_unified = any(
        path.relative_to(root).parts[0] not in split_names
        for path in metadata_paths
    )
    if has_unified == has_pre_split:
        raise ConfigError(
            "data.layout=auto 无法唯一判断布局；请显式设置 unified 或 pre_split"
        )
    return "unified" if has_unified else "pre_split"


def resolve_pre_split_directories(
    data: dict[str, Any], root: Path
) -> tuple[Path, Path, Path | None]:
    """Resolve train/validation/test aliases for a pre-split dataset."""
    train = root / str(data.get("train_directory", "train"))
    configured_val = data.get("val_directory")
    if configured_val:
        validation = root / str(configured_val)
    else:
        validation = next(
            (root / name for name in ("eval", "val", "validation") if (root / name).is_dir()),
            root / "eval",
        )
    configured_test = data.get("test_directory")
    test = root / str(configured_test) if configured_test else None
    if not train.is_dir():
        raise ConfigError(f"pre_split train目录不存在: {train}")
    if not validation.is_dir():
        raise ConfigError(f"pre_split validation目录不存在: {validation}")
    if test is not None and not test.is_dir():
        raise ConfigError(f"pre_split test目录不存在: {test}")
    return train, validation, test


def validate_preflight_config(config: dict[str, Any]) -> Path:
    data = require_mapping(config, "data")
    root = data.get("root")
    source_root = data.get("source_root")
    active_root = root or source_root
    if not active_root:
        raise ConfigError("data.source_root 和 data.root 至少需要配置一个")
    root_path = Path(str(active_root)).expanduser().resolve()
    if not root_path.is_dir():
        name = "data.root" if root else "data.source_root"
        raise ConfigError(f"{name} 目录不存在: {root_path}")

    try:
        action_spec = action_spec_from_data(data)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if data.get("number_of_arms") != 2:
        raise ConfigError(f"data.number_of_arms 必须是 2，当前为 {data.get('number_of_arms')!r}")
    if data.get("chunk_size") != 16:
        raise ConfigError(f"data.chunk_size 必须是 16，当前为 {data.get('chunk_size')!r}")
    if data.get("action_includes_gripper") is not True:
        raise ConfigError("data.action_includes_gripper 必须是 true")
    if action_spec.encoding == "legacy_euler":
        rotation_scale = float(data.get("rotation_scale", 0.0))
        if rotation_scale <= 0:
            raise ConfigError("legacy_euler要求data.rotation_scale为正数")
    layout = resolve_data_layout(data, root_path)
    data["layout"] = layout
    if layout == "pre_split":
        resolve_pre_split_directories(data, root_path)
    return root_path


def validate_training_config(config: dict[str, Any]) -> None:
    validate_preflight_config(config)
    model = require_mapping(config, "model")
    training = require_mapping(config, "training")
    objectives = require_mapping(config, "objectives")
    runtime = require_mapping(config, "runtime")
    checkpoint = model.get("checkpoint_path")
    if not checkpoint:
        raise ConfigError("model.checkpoint_path 未配置；禁止随机初始化 2B 模型")
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint_path.exists():
        raise ConfigError(f"模型 checkpoint 不存在: {checkpoint_path}")
    tokenizer = model.get("tokenizer_path")
    if not tokenizer:
        raise ConfigError("model.tokenizer_path 未配置；预编码训练仍需 tokenizer 配置初始化模型")
    tokenizer_path = Path(str(tokenizer)).expanduser().resolve()
    if not tokenizer_path.is_file():
        raise ConfigError(f"模型 tokenizer checkpoint 不存在: {tokenizer_path}")
    data = require_mapping(config, "data")
    if not data.get("root"):
        raise ConfigError("data.root 未配置；训练不能直接读取 source_root raw HDF5")
    for key in ("t5_embeddings_path", "statistics_path"):
        path = data.get(key)
        if not path:
            raise ConfigError(f"data.{key} 未配置")
        if not Path(str(path)).expanduser().is_file():
            raise ConfigError(f"data.{key} 不存在: {path}")
    if int(training.get("batch_size_per_rank", 0)) < 1:
        raise ConfigError("training.batch_size_per_rank 必须 >= 1")
    if int(training.get("gradient_accumulation_steps", 0)) < 1:
        raise ConfigError("training.gradient_accumulation_steps 必须 >= 1")
    validation_every_steps = int(training.get("validation_every_steps", 0))
    if validation_every_steps < 0:
        raise ConfigError("training.validation_every_steps 必须 >= 0")
    validation_every_epochs = int(training.get("validation_every_epochs", 1))
    if validation_every_epochs < 0:
        raise ConfigError("training.validation_every_epochs 必须 >= 0")
    if int(data.get("workers_per_rank", -1)) < 0:
        raise ConfigError("data.workers_per_rank 必须 >= 0")
    if int(data.get("prefetch_factor", 0)) < 1:
        raise ConfigError("data.prefetch_factor 必须 >= 1")
    mode = str(objectives.get("training_mode", "")).strip()
    supported_modes = {
        "policy_only", "inverse_dynamics_only", "base_joint",
        "joint_with_inverse", "custom_mix",
    }
    if mode not in supported_modes:
        raise ConfigError(
            "objectives.training_mode 必须是 policy_only、inverse_dynamics_only、"
            "base_joint、joint_with_inverse 或 custom_mix"
        )
    if mode == "policy_only" and objectives.get("policy_auxiliary_targets") is not True:
        raise ConfigError(
            "policy_only 基线必须设置 objectives.policy_auxiliary_targets=true；"
            "该模式预测 action、future state 和 value"
        )
    loss_reduction = str(objectives.get("loss_reduction", "masked_mean"))
    if loss_reduction not in {"masked_mean", "learner_actor_full_tensor_mean"}:
        raise ConfigError(
            "objectives.loss_reduction 必须是 masked_mean 或 "
            "learner_actor_full_tensor_mean"
        )
    if mode == "custom_mix":
        probabilities = [
            float(objectives.get("policy_probability", 0)),
            float(objectives.get("world_probability", 0)),
            float(objectives.get("value_probability", 0)),
            float(objectives.get("inverse_dynamics_probability", 0)),
        ]
        if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
            raise ConfigError(
                "custom_mix 的 policy/world/value/inverse_dynamics probability "
                "必须非负且总和为 1"
            )

    split = require_mapping(config, "split")
    max_episodes = int(split.get("max_episodes", 0))
    if max_episodes < 0 or (data["layout"] == "unified" and max_episodes < 2):
        raise ConfigError("split.max_episodes 在unified布局必须 >= 2；pre_split可设0表示全部")
    if data["layout"] == "unified":
        ratios = split.get("ratios")
        if not isinstance(ratios, list) or len(ratios) != 3:
            raise ConfigError("split.ratios 必须是 [train, validation, test]")
        ratios = [float(value) for value in ratios]
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ConfigError("split.ratios 必须非负且总和为 1")
        if ratios[0] <= 0 or ratios[1] <= 0:
            raise ConfigError("split 必须包含非空 train 和 validation 比例")

    evaluation = require_mapping(config, "evaluation")
    protocol = str(evaluation.get("protocol", "fixed_suite"))
    if protocol not in {"fixed_suite", "learner_actor"}:
        raise ConfigError("evaluation.protocol 必须是 fixed_suite 或 learner_actor")
    sigma_values = evaluation.get("sigma_values")
    if protocol == "fixed_suite":
        if not isinstance(sigma_values, list) or not sigma_values:
            raise ConfigError("evaluation.sigma_values 必须是非空列表")
        if any(not 0 < float(value) <= 1 for value in sigma_values):
            raise ConfigError("evaluation.sigma_values 必须位于 (0, 1]")
    elif mode != "policy_only":
        raise ConfigError("evaluation.protocol=learner_actor 要求 policy_only 训练")
    validation_batch_limit(config)
    evaluation_objectives = evaluation.get("objectives")
    if evaluation_objectives is not None:
        valid_objectives = {"policy", "world", "value", "inverse_dynamics"}
        if not isinstance(evaluation_objectives, list) or not evaluation_objectives:
            raise ConfigError("evaluation.objectives 必须是非空列表")
        unknown = set(map(str, evaluation_objectives)) - valid_objectives
        if unknown:
            raise ConfigError(f"evaluation.objectives 包含未知目标: {sorted(unknown)}")
    if validation_every_steps and protocol == "fixed_suite":
        if int(evaluation.get("step_validation_max_batches", 0)) < 1:
            raise ConfigError(
                "启用 training.validation_every_steps 时，"
                "evaluation.step_validation_max_batches 必须 >= 1"
            )
        step_objectives = evaluation.get("step_validation_objectives", [])
        valid_objectives = {"policy", "world", "value", "inverse_dynamics"}
        if not isinstance(step_objectives, list) or not step_objectives:
            raise ConfigError("evaluation.step_validation_objectives 必须是非空列表")
        unknown = set(map(str, step_objectives)) - valid_objectives
        if unknown:
            raise ConfigError(
                f"evaluation.step_validation_objectives 包含未知目标: {sorted(unknown)}"
            )
        step_sigmas = evaluation.get("step_validation_sigma_values", [])
        if not isinstance(step_sigmas, list) or not step_sigmas:
            raise ConfigError("evaluation.step_validation_sigma_values 必须是非空列表")
    inverse_ablations = evaluation.get("inverse_condition_ablations", ["normal"])
    valid_ablations = {"normal", "current_only", "future_only", "future_shuffle"}
    if not isinstance(inverse_ablations, list) or not inverse_ablations:
        raise ConfigError("evaluation.inverse_condition_ablations 必须是非空列表")
    unknown_ablations = set(map(str, inverse_ablations)) - valid_ablations
    if unknown_ablations:
        raise ConfigError(
            f"evaluation.inverse_condition_ablations 包含未知模式: {sorted(unknown_ablations)}"
        )
    monitoring = config.get("monitoring", {})
    if not isinstance(monitoring, dict):
        raise ConfigError("monitoring 必须是 mapping")
    prometheus_enabled = monitoring.get("prometheus_enabled", False)
    if not isinstance(prometheus_enabled, bool):
        raise ConfigError("monitoring.prometheus_enabled 必须是 boolean")
    port = int(monitoring.get("port", 8000))
    if not 1 <= port <= 65535:
        raise ConfigError("monitoring.port 必须位于 [1, 65535]")
    host = str(monitoring.get("host", "0.0.0.0")).strip()
    if not host:
        raise ConfigError("monitoring.host 不能为空")
    metrics_path = str(monitoring.get("path", "/metrics"))
    if not metrics_path.startswith("/"):
        raise ConfigError("monitoring.path 必须以 / 开头")
    wandb_enabled = monitoring.get("wandb_enabled", False)
    if not isinstance(wandb_enabled, bool):
        raise ConfigError("monitoring.wandb_enabled 必须是 boolean")
    if wandb_enabled:
        project = str(monitoring.get("wandb_project") or "").strip()
        if not project:
            raise ConfigError("启用 WandB 时 monitoring.wandb_project 不能为空")
        wandb_mode = str(monitoring.get("wandb_mode", "online")).strip().lower()
        if wandb_mode not in {"online", "offline", "disabled"}:
            raise ConfigError("monitoring.wandb_mode 必须是 online、offline 或 disabled")
    output_dir = runtime.get("output_dir")
    if not output_dir:
        raise ConfigError("runtime.output_dir 未配置")


def resolve_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value).expanduser().resolve()
