"""Resolve camera keys from task / robot config for policy and reward classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


def _as_list(keys: Any) -> list[str]:
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return [str(k) for k in keys]


def resolve_image_keys(cfg: Any) -> list[str]:
    """Return ``robot_config.image_keys`` from a Hydra env cfg or nested dict."""
    if cfg is None:
        return []
    robot_config = getattr(cfg, "robot_config", None)
    if robot_config is None and isinstance(cfg, Mapping):
        robot_config = cfg.get("robot_config", cfg)
    if robot_config is None:
        return []
    keys = getattr(robot_config, "image_keys", None)
    if keys is None and isinstance(robot_config, Mapping):
        keys = robot_config.get("image_keys")
    return _as_list(keys)


def resolve_classifier_camera_keys(cfg: Any) -> list[str]:
    """Cameras for the reward classifier.

    Prefer ``classifier_cfg.classifier_keys``; fall back to ``robot_config.image_keys``.
    """
    classifier_cfg = getattr(cfg, "classifier_cfg", None)
    if classifier_cfg is None and isinstance(cfg, Mapping):
        classifier_cfg = cfg.get("classifier_cfg")
    if classifier_cfg is not None:
        keys = getattr(classifier_cfg, "classifier_keys", None)
        if keys is None and isinstance(classifier_cfg, Mapping):
            keys = classifier_cfg.get("classifier_keys")
        keys = _as_list(keys)
        if keys:
            return keys
    return resolve_image_keys(cfg)


def load_task_yaml(task_name: str, cfg_dir: str | Path | None = None) -> dict:
    """Load ``cfg/task/{task_name}.yaml`` relative to HIL-RL-icml root."""
    if cfg_dir is None:
        cfg_dir = Path(__file__).resolve().parents[1] / "cfg" / "task"
    else:
        cfg_dir = Path(cfg_dir)
    path = cfg_dir / f"{task_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Task yaml not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data)}")
    return data


def observation_image_key(camera: str) -> str:
    return f"observation.images.{camera}"




def filter_features_to_cameras(
    features: Mapping[str, Any],
    camera_keys: Sequence[str],
    *,
    keep_non_image: bool = True,
) -> dict[str, Any]:
    """Keep only selected image cameras (plus optional non-image features)."""
    allowed = {observation_image_key(k) for k in camera_keys}
    out: dict[str, Any] = {}
    for key, value in features.items():
        is_image = key.startswith("observation.images.") or key.startswith("observation.image.")
        if is_image:
            if key in allowed:
                out[key] = value
        elif keep_non_image:
            out[key] = value
    missing = [k for k in allowed if k not in features]
    if missing:
        raise KeyError(
            f"Requested camera features missing from dataset/config: {missing}. "
            f"Available: {[k for k in features if 'image' in k]}"
        )
    return out


def _feature_shape(ft: Any, default: Sequence[int]) -> tuple[int, ...]:
    if ft is None:
        return tuple(default)
    shape = getattr(ft, "shape", None)
    if shape is None and isinstance(ft, Mapping):
        shape = ft.get("shape")
    return tuple(shape) if shape is not None else tuple(default)


def apply_use_state_to_lerobot_cfg(cfg: Any, use_state: bool) -> None:
    """Include or drop ``observation.state`` from policy / env feature specs.

    When ``use_state`` is False, the policy is image-only (SAC already supports
    ``has_state`` gated on ``observation.state`` in ``input_features``).
    """
    state_key = "observation.state"
    if use_state:
        return

    policy_input = getattr(getattr(cfg, "policy", None), "input_features", None)
    if isinstance(policy_input, dict) and state_key in policy_input:
        cfg.policy.input_features = {k: v for k, v in policy_input.items() if k != state_key}

    env_features = getattr(getattr(cfg, "env", None), "features", None)
    if isinstance(env_features, dict) and state_key in env_features:
        cfg.env.features = {k: v for k, v in env_features.items() if k != state_key}

    features_map = getattr(getattr(cfg, "env", None), "features_map", None)
    if isinstance(features_map, dict) and state_key in features_map:
        cfg.env.features_map = {k: v for k, v in features_map.items() if k != state_key}

    dataset_stats = getattr(getattr(cfg, "policy", None), "dataset_stats", None)
    if isinstance(dataset_stats, dict) and state_key in dataset_stats:
        cfg.policy.dataset_stats = {k: v for k, v in dataset_stats.items() if k != state_key}


def apply_image_keys_to_lerobot_cfg(
    cfg: Any,
    image_keys: Sequence[str],
    *,
    image_shape: Sequence[int] = (128, 128, 3),
    state_shape: Sequence[int] | None = None,
    include_state: bool = True,
) -> list[str]:
    """Rebuild ``cfg.env.features`` / ``features_map`` and ``cfg.policy.input_features`` from image_keys.

    Args:
        include_state: If True, keep ``observation.state`` in policy inputs; if False, image-only.

    Returns the normalized image_keys list.
    """
    image_keys = _as_list(image_keys)
    if not image_keys:
        raise ValueError("image_keys must be a non-empty list")

    # Preserve existing state / action shapes when present.
    existing_features = getattr(getattr(cfg, "env", None), "features", None) or {}
    if state_shape is None:
        state_shape = _feature_shape(existing_features.get("observation.state"), (8,))
        if state_shape == (8,):
            policy_state = getattr(getattr(cfg, "policy", None), "input_features", {}) or {}
            if "observation.state" in policy_state:
                state_shape = _feature_shape(policy_state["observation.state"], (8,))

    action_shape = _feature_shape(existing_features.get("action"), (6,))

    from lerobot.configs.types import FeatureType, PolicyFeature

    env_features: dict[str, PolicyFeature] = {}
    features_map: dict[str, str] = {}
    policy_input: dict[str, PolicyFeature] = {}

    # env.features use HWC; policy.input_features use CHW.
    if len(image_shape) == 3 and image_shape[0] == 3:
        chw = tuple(image_shape)
        hwc = (image_shape[1], image_shape[2], image_shape[0])
    else:
        hwc = tuple(image_shape)
        chw = (image_shape[2], image_shape[0], image_shape[1])

    for cam in image_keys:
        key = observation_image_key(cam)
        env_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=hwc)
        features_map[key] = key
        policy_input[key] = PolicyFeature(type=FeatureType.VISUAL, shape=chw)

    if include_state:
        env_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=tuple(state_shape))
        features_map["observation.state"] = "observation.state"
        policy_input["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=tuple(state_shape))

    env_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(action_shape))
    features_map["action"] = "action"

    cfg.env.features = env_features
    cfg.env.features_map = features_map
    cfg.policy.input_features = policy_input

    # Drop stale camera / state stats so Normalize does not expect removed keys.
    dataset_stats = getattr(cfg.policy, "dataset_stats", None)
    if isinstance(dataset_stats, dict):
        cfg.policy.dataset_stats = {
            k: v
            for k, v in dataset_stats.items()
            if k in policy_input
            or (
                not str(k).startswith("observation.images.")
                and k != "observation.state"
            )
        }

    return image_keys


def resolve_pose_action_mask(env_cfg: Any) -> np.ndarray:
    """Read pose_action_mask [x,y,z,roll,pitch,yaw] from task robot_config."""
    rc = getattr(env_cfg, "robot_config", env_cfg)
    mask = getattr(rc, "pose_action_mask", None)
    if mask is None and isinstance(rc, Mapping):
        mask = rc.get("pose_action_mask")
    if mask is None:
        mask = getattr(env_cfg, "pose_action_mask", None)
    if mask is None:
        mask = [1, 1, 1, 1, 1, 1]
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if mask.shape[0] != 6:
        raise ValueError(f"pose_action_mask must have 6 entries, got {mask.shape[0]}")
    return mask


def policy_action_dim_from_env_cfg(env_cfg: Any) -> int:
    """Continuous policy action dim = sum(pose_action_mask) * num_arms."""
    mask = resolve_pose_action_mask(env_cfg)
    rc = getattr(env_cfg, "robot_config", env_cfg)
    dual = getattr(rc, "dual_arm", False)
    if isinstance(rc, Mapping):
        dual = rc.get("dual_arm", dual)
    n_arms = 2 if dual else 1
    return int(mask.sum()) * n_arms


def apply_pose_action_dim_to_lerobot_cfg(cfg: Any, env_cfg: Any) -> int:
    """Override ``env.features['action'].shape`` from task yaml pose_action_mask.

    Keeps learner / actor policy output dims aligned with the task mask.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    action_dim = policy_action_dim_from_env_cfg(env_cfg)
    action_shape = (action_dim,)
    mask = resolve_pose_action_mask(env_cfg)

    env_features = getattr(getattr(cfg, "env", None), "features", None)
    if isinstance(env_features, dict):
        env_features = dict(env_features)
        env_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=action_shape)
        cfg.env.features = env_features

    policy_out = getattr(getattr(cfg, "policy", None), "output_features", None)
    if isinstance(policy_out, dict):
        policy_out = dict(policy_out)
        policy_out["action"] = PolicyFeature(type=FeatureType.ACTION, shape=action_shape)
        cfg.policy.output_features = policy_out

    print(
        f"Policy action dim from task pose_action_mask={mask.tolist()} -> shape={action_shape}"
    )
    return action_dim
