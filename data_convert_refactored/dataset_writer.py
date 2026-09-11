"""LeRobot feature definition, episode writing, and conversion metadata."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .conditioning.camera import COSMOS_CONDITIONING_FPS, LATENT_INDICES, RAW_FRAME_SLICES
from .config import CosmosRuntimeConfig
from .memory_monitor import memory_event
from .preparation.rotation import COSMOS_ROTATION_6D, LEGACY_EULER, get_action_dim
from .preparation.episode import action_proprio_source_arrays
from .encoding.vae_encoding import CLEAN_RESTORE_INDICES, encode_episode_cpu_friendly
from .shape_trace import require_shape, trace_shape


def cosmos_encoded_state_to_frame(state: dict) -> dict:
    """把单步Tensor state转成LeRobot可序列化的NumPy字段。

    编码阶段保留batch维1；写Parquet前去掉该维。标量重新包装为(1,)，与feature
    schema保持一致。
    """
    frame_state = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim > 0 and value.shape[0] == 1:
                value = value.squeeze(0)
            array = value.numpy() if key == "clean_restore_latent" else value.float().numpy()
        else:
            array = np.asarray(value)
        if array.ndim == 0:
            array = np.array([array.item()], dtype=array.dtype)
        frame_state[key] = array
    return frame_state


def build_audit_feature_shapes(encoded_episode: Any) -> dict[str, tuple[int, ...]]:
    """根据action表示和实际source字段生成固定LeRobot附加schema。"""
    if encoded_episode.euler_actions is None or encoded_episode.rotation_6d_actions is None:
        raise ValueError("encoded episode is missing auxiliary action representations")
    source_arrays = action_proprio_source_arrays(
        encoded_episode.episode, encoded_episode.action_source.value
    )
    return {
        "action.euler_control": (int(encoded_episode.euler_actions.shape[1]),),
        "action.rotation_6d_control": (
            int(encoded_episode.rotation_6d_actions.shape[1]),
        ),
        **{
            name: (int(values.shape[1]),)
            for name, values in encoded_episode.auxiliary_fields.items()
        },
        **{name: (int(values.shape[1]),) for name, values in source_arrays.items()},
    }


def build_output_features(
    action_dim: int,
    proprio_dim: int,
    audit_feature_shapes: dict[str, tuple[int, ...]] | None = None,
    *,
    action_chunk_size: int = 16,
    save_clean_restore_latent: bool = False,
) -> dict:
    """声明Parquet逐行schema；video已经是注入低维条件后的Cosmos latent。"""
    features = {
        "video": {"dtype": "float32", "shape": (16, 9, 28, 28), "names": None},
        "proprio": {"dtype": "float32", "shape": (proprio_dim,), "names": None},
        "future_proprio": {"dtype": "float32", "shape": (proprio_dim,), "names": None},
        "value_function_return": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "action.latent_chunk": {
            "dtype": "float32",
            "shape": (action_chunk_size, action_dim),
            "names": None,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
        "complementary_info.is_intervention": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_intervention"],
        },
    }
    if save_clean_restore_latent:
        features["clean_restore_latent"] = {
            "dtype": "float16",
            "shape": (16, len(CLEAN_RESTORE_INDICES), 28, 28),
            "names": None,
        }
    for name, shape in (audit_feature_shapes or {}).items():
        features[name] = {"dtype": "float32", "shape": shape, "names": None}
    return features


def encode_and_write_episode(
    transition_list: list[dict],
    episode_index: int,
    dataset: Any,
    policy: Any,
    cosmos_cfg: Any,
    config: CosmosRuntimeConfig,
    task_description: str,
) -> int:
    """编码一个完整episode并逐帧写入LeRobot数据集。

    输入transition仍包含uint8视频条件；`encode_episode_cpu_friendly`将其转换为
    (T,16,9,28,28) latent，并注入action chunk、proprio、future proprio和value。
    返回本episode实际写入的帧数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gpu_before = torch.cuda.memory_allocated() / 1e9
    frame_count = len(transition_list)
    if frame_count == 0:
        raise ValueError("cannot encode an empty transition list")
    action_dim = int(transition_list[0]["action"].shape[-1])
    proprio_dim = int(transition_list[0]["state"]["proprio"].shape[-1])
    estimated_video_gb = frame_count * 3 * 33 * 224 * 224 / 1e9
    print(
        f"    GPU 编码前: {gpu_before:.1f} GB | 帧数={frame_count} | "
        f"预估 uint8 视频={estimated_video_gb:.1f} GB (在CPU)"
    )

    encoded = encode_episode_cpu_friendly(
        transition_list,
        config.encode_batch_size,
        device,
        cosmos_cfg,
        config.batch_size,
        policy,
        encode_world_size=config.encode_world_size,
        encode_device_ids=config.encode_device_ids,
        save_clean_restore_latent=config.save_clean_restore_latent,
    )

    encode_time = time.time() - start_time
    gpu_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"    VAE encode: {encode_time:.1f}s ({len(transition_list)} frames)")
    print(f"    GPU 峰值: {gpu_peak:.1f} GB (reserved: {torch.cuda.memory_reserved()/1e9:.1f} GB)")
    torch.cuda.empty_cache()

    memory_event("add_frame_start")
    written = 0
    progress_step = max(100, len(encoded) // 10)
    for index, transition in enumerate(encoded):
        # 当前离线转换全部是intervention；保留过滤条件以对齐在线采集的数据语义。
        if not transition.get("complementary_info", {}).get("is_intervention", True):
            continue

        state = cosmos_encoded_state_to_frame(transition["state"])
        action_value = transition["action"]
        if isinstance(action_value, torch.Tensor):
            action_value = action_value.detach().cpu().numpy()

        frame = {
            **state,
            "action": action_value.astype(np.float32),
            "next.reward": np.array([float(transition.get("reward", 0.0))], dtype=np.float32),
            "next.done": np.array([bool(transition.get("done", False))], dtype=bool),
            "complementary_info.discrete_penalty": np.array([0], dtype=np.float32),
            "complementary_info.is_intervention": np.array([True], dtype=bool),
        }
        for name, value in transition.items():
            if not name.startswith(("action.", "observation.", "source.")):
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().float().numpy()
            frame[name] = np.asarray(value, dtype=np.float32)
        require_shape("frame.video", frame["video"], (16, 9, 28, 28))
        require_shape("frame.action", frame["action"], (action_dim,))
        require_shape(
            "frame.action.latent_chunk",
            frame["action.latent_chunk"],
            (config.chunk_size, action_dim),
        )
        require_shape("frame.proprio", frame["proprio"], (proprio_dim,))
        require_shape("frame.future_proprio", frame["future_proprio"], (proprio_dim,))
        require_shape("frame.value_function_return", frame["value_function_return"], (1,))
        if config.save_clean_restore_latent:
            require_shape(
                "frame.clean_restore_latent",
                frame["clean_restore_latent"],
                (16, len(CLEAN_RESTORE_INDICES), 28, 28),
            )
            if frame["clean_restore_latent"].dtype != np.float16:
                raise ValueError(
                    "frame.clean_restore_latent must use float16, got "
                    f"{frame['clean_restore_latent'].dtype}"
                )
        require_shape("frame.next.reward", frame["next.reward"], (1,))
        require_shape("frame.next.done", frame["next.done"], (1,))
        for name, value in frame.items():
            if name == "action.latent_chunk":
                continue
            if name.startswith(("action.", "observation.", "source.")) and np.asarray(value).ndim != 1:
                raise ValueError(f"audit field {name} must be 1D, got {np.asarray(value).shape}")
        if index == 0 or index == len(encoded) - 1:
            trace_shape(
                f"PARQUET_FRAME index={index}",
                video=frame["video"],
                action=frame["action"],
                action_latent_chunk=frame["action.latent_chunk"],
                proprio=frame["proprio"],
                future_proprio=frame["future_proprio"],
                value=frame["value_function_return"],
                reward=frame["next.reward"],
                done=frame["next.done"],
            )
        dataset.add_frame(frame, task=task_description)
        written += 1
        if written % progress_step == 0 or written == len(encoded):
            memory_event("add_frame_progress", frames_added=written)

    memory_event("before_dataset_save_episode", frames_added=written)
    from .storage_monitor import storage_event, storage_failure, storage_probe

    storage_probe("before_dataset_save_probe")
    storage_event("before_dataset_save_episode", frames_added=written)
    # save_episode负责生成当前episode的Parquet及LeRobot episode metadata。
    try:
        dataset.save_episode()
    except OSError as exc:
        storage_failure("dataset_save_episode_failed", exc, frames_added=written)
        raise
    memory_event("after_dataset_save_episode", frames_added=written)
    storage_event("after_dataset_save_episode", frames_added=written)

    rows_before_clear = len(dataset.hf_dataset)
    memory_event("before_hf_dataset_clear", hf_dataset_rows=rows_before_clear)
    # Parquet落盘后立即释放writer累积的Arrow行，避免多episode时CPU内存持续增长。
    dataset.hf_dataset = dataset.hf_dataset.select([])
    rows_after_clear = len(dataset.hf_dataset)
    memory_event("after_hf_dataset_clear", hf_dataset_rows=rows_after_clear)
    if rows_after_clear != 0:
        raise RuntimeError(
            f"failed to clear in-memory Hugging Face dataset: {rows_after_clear} rows remain"
        )
    return written


def proprio_order(robot_type: str) -> list[str]:
    """返回metadata中的proprio通道顺序；每臂为pose quaternion加夹爪，共8维。"""
    arm = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    if robot_type == "dual":
        return [f"left_{name}" for name in arm] + [f"right_{name}" for name in arm]
    return arm


def action_order(robot_type: str, action_encoding: str) -> list[str]:
    """返回metadata中的action通道顺序，确保训练和实机反归一化可解释。"""
    prefix = ["left", "right"] if robot_type == "dual" else ["arm"]
    if action_encoding == LEGACY_EULER:
        arm_fields = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
    elif action_encoding == COSMOS_ROTATION_6D:
        arm_fields = [
            "dx",
            "dy",
            "dz",
            "rot6d_col0_x",
            "rot6d_col0_y",
            "rot6d_col0_z",
            "rot6d_col1_x",
            "rot6d_col1_y",
            "rot6d_col1_z",
            "gripper",
        ]
    else:
        raise ValueError(f"unsupported action encoding: {action_encoding}")
    return [f"{arm}_{field}" for arm in prefix for field in arm_fields]


def save_conversion_metadata(
    output_dir: Path,
    config: CosmosRuntimeConfig,
    robot_type: str,
    proprio_stats: dict[str, np.ndarray],
    camera_names: list[str],
    detected_fps: int,
    fk_validation: dict | None = None,
    action_stats: dict[str, np.ndarray] | None = None,
    stats_range_report: dict | None = None,
) -> None:
    """保存可复现实验所需的stats、维度、时序、crop和action语义。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    proprio_constant_mask = proprio_stats.get(
        "proprio_constant_mask",
        np.zeros_like(proprio_stats["proprio_min"], dtype=bool),
    )
    stats = {
        "proprio_min": proprio_stats["proprio_min"].tolist(),
        "proprio_max": proprio_stats["proprio_max"].tolist(),
        "proprio_constant_mask": np.asarray(proprio_constant_mask).tolist(),
    }
    if "num_proprio_samples" in proprio_stats:
        stats["num_proprio_samples"] = int(proprio_stats["num_proprio_samples"])
    if action_stats is not None:
        stats.update(
            {
                "actions_min": action_stats["actions_min"].tolist(),
                "actions_max": action_stats["actions_max"].tolist(),
                "actions_constant_mask": np.asarray(
                    action_stats.get(
                        "actions_constant_mask",
                        np.zeros_like(action_stats["actions_min"], dtype=bool),
                    )
                ).tolist(),
            }
        )
        if "num_action_samples" in action_stats:
            stats["num_action_samples"] = int(action_stats["num_action_samples"])
    else:
        stats.update({"action_min": [-1.0], "action_max": [1.0]})

    output_stats_path = output_dir / "dataset_statistics.json"
    external_stats_path = Path(config.dataset_stats_path).expanduser().resolve()
    if external_stats_path.is_file():
        external_bytes = external_stats_path.read_bytes()
        external_stats_sha256 = hashlib.sha256(external_bytes).hexdigest()
        external_stats = json.loads(external_bytes)
        external_action_min = external_stats.get("actions_min", [])
        effective_action_min = action_stats.get("actions_min", []) if action_stats else []
        if len(external_action_min) == len(effective_action_min):
            output_stats_path.write_bytes(external_bytes)
        else:
            with output_stats_path.open("w", encoding="utf-8") as file:
                json.dump(stats, file, indent=2)
        external_sidecar = external_stats_path.with_suffix(".metadata.json")
        if config.stats_mode == "generated":
            output_sidecar = output_dir / "dataset_statistics.metadata.json"
            if not external_sidecar.is_file():
                raise FileNotFoundError(
                    f"generated stats semantic metadata is missing: {external_sidecar}"
                )
            output_sidecar.write_bytes(external_sidecar.read_bytes())
    else:
        with output_stats_path.open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2)
        external_stats_sha256 = hashlib.sha256(output_stats_path.read_bytes()).hexdigest()

    source_fields = ["source.puppet.pose_xyzw", "source.puppet.gripper"]
    if config.action_source == "master_same_frame":
        source_fields.extend(["source.master.pose_xyzw", "source.master.gripper"])
    elif config.action_source == "master_joint_fk_same_frame":
        source_fields.extend(
            [
                "source.puppet.joints",
                "source.master.joints",
                "source.master.gripper",
            ]
        )

    metadata = {
        "format_version": 2 if config.save_clean_restore_latent else 1,
        "task_description": config.task_description,
        "robot_type": robot_type,
        "episode_labeling": {
            "outcome": config.episode_outcome,
            "source": "explicit_cli",
            "success_tail_frames": 5,
            "reward_positive": 10.0,
            "reward_negative": -0.05,
            "done_dtype": "bool",
        },
        "language_conditioning": {
            "t5_embedding_included_during_conversion": not config.skip_t5,
            "t5_embeddings_path_used": (
                str(Path(config.t5_embeddings_path).expanduser().resolve())
                if not config.skip_t5
                else None
            ),
            "task_string_preserved_in_lerobot_metadata": True,
        },
        "action": {
            "training_field": "action",
            "encoding": config.action_encoding,
            "source": config.action_source,
            "dimension": get_action_dim(robot_type, config.action_encoding),
            "order": action_order(robot_type, config.action_encoding),
            "pose_semantics": "local_delta_inv_current_times_target",
            "current_pose": "puppet_same_frame",
            "target_pose": (
                "FK(master_joint_same_frame)"
                if config.action_source == "master_joint_fk_same_frame"
                else (
                    "master_same_frame"
                    if config.action_source == "master_same_frame"
                    else "puppet_next_frame_with_last_action_repeated"
                )
            ),
            "gripper_semantics": "absolute_target",
            "last_action_is_padding": config.action_source == "puppet_next_frame",
            "last_action_is_synthetic": config.action_source == "puppet_next_frame",
            "last_action_padding_strategy": (
                "repeat_last_valid" if config.action_source == "puppet_next_frame" else "none"
            ),
            "last_action_has_observed_target_frame": config.action_source != "puppet_next_frame",
            "translation_scale": config.action_scale[0],
            "rotation_representation": (
                "matrix_first_two_columns"
                if config.action_encoding == COSMOS_ROTATION_6D
                else "euler_xyz"
            ),
            "rotation_scale": (
                None if config.action_encoding == COSMOS_ROTATION_6D else config.action_scale[1]
            ),
            "gripper_scale": config.action_scale[2],
            "pose_range": [-1.0, 1.0],
            "gripper_range": [0.0, 1.0],
            "normalization": (
                "robot_control_scale_then_dataset_minmax_to_minus1_plus1"
                if action_stats is not None
                else "robot_control_scale_only"
            ),
            "statistics_file": "dataset_statistics.json",
            "inference_unnormalize_actions": action_stats is not None,
            "serialized_range": [-1.0, 1.0] if action_stats is not None else None,
            "injected_into_video_latent": True,
            "latent_index": LATENT_INDICES["action_latent_idx"],
            "chunk_size": config.chunk_size,
            "latent_chunk_field": "action.latent_chunk",
            "latent_chunk_shape": [
                config.chunk_size,
                get_action_dim(robot_type, config.action_encoding),
            ],
            "latent_chunk_dataset_normalized": True,
            "latent_chunk_padding_strategy": "repeat_last",
            "latent_chunk_flatten_order": "time_major_then_action_dimension",
            "auxiliary_fields": {
                "action.latent_chunk": {
                    "shape": [
                        config.chunk_size,
                        get_action_dim(robot_type, config.action_encoding),
                    ],
                    "semantics": "exact action chunk passed to latent injection",
                    "dataset_normalized": True,
                    "injected_into_video_latent": True,
                    "latent_index": LATENT_INDICES["action_latent_idx"],
                },
                "action.euler_control": {
                    "dimension": 14 if robot_type == "dual" else 7,
                    "rotation_representation": "euler_xyz_divided_by_rotation_scale",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.rotation_6d_control": {
                    "dimension": 20 if robot_type == "dual" else 10,
                    "rotation_representation": "matrix_first_two_columns",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.target_rotation_6d": {
                    "dimension": 12 if robot_type == "dual" else 6,
                    "semantics": "target absolute EE rotation",
                    "rotation_representation": "matrix_first_two_columns",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.delta_rotation_6d": {
                    "dimension": 12 if robot_type == "dual" else 6,
                    "semantics": "rotation of inv(T_current) @ T_target",
                    "rotation_representation": "matrix_first_two_columns",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.target_ee_pose_xyzw": {
                    "dimension": 14 if robot_type == "dual" else 7,
                    "semantics": "target absolute EE pose [xyz,qx,qy,qz,qw]",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.target_joint_position": {
                    "dimension": 14 if robot_type == "dual" else 7,
                    "semantics": "target absolute aligned arm joint position",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.delta_ee_xyz_euler": {
                    "dimension": 12 if robot_type == "dual" else 6,
                    "semantics": "local inv(T_current) @ T_target as [xyz,euler_xyz]",
                    "translation_unit": "meter",
                    "rotation_unit": "radian",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.delta_joint_position": {
                    "dimension": 14 if robot_type == "dual" else 7,
                    "semantics": "target_joint - current_joint",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
                "action.target_gripper": {
                    "dimension": 2 if robot_type == "dual" else 1,
                    "semantics": "target absolute gripper position",
                    "dataset_normalized": False,
                    "injected_into_video_latent": False,
                },
            },
        },
        "auxiliary_observation": {
            "semantics": "current puppet state at timestep t",
            "dataset_normalized": False,
            "injected_into_video_latent": False,
            "fields": {
                "observation.rotation_6d": {
                    "dimension": 12 if robot_type == "dual" else 6,
                    "rotation_representation": "matrix_first_two_columns",
                },
                "observation.ee_pose_xyzw": {
                    "dimension": 14 if robot_type == "dual" else 7,
                    "order_per_arm": ["x", "y", "z", "qx", "qy", "qz", "qw"],
                },
                "observation.joint_position": {
                    "dimension": 14 if robot_type == "dual" else 7,
                },
                "observation.gripper": {
                    "dimension": 2 if robot_type == "dual" else 1,
                },
            },
        },
        "proprio": {
            "order": proprio_order(robot_type),
            "normalization": "per_dimension_minmax_to_minus1_plus1",
            "constant_dimensions_map_to": 0.0,
            "clip": False,
            "statistics_file": "dataset_statistics.json",
        },
        "source_low_dim": {
            "semantics": "HDF5 aligned fields used by action/proprio computation",
            "dataset_normalized": False,
            "fields": source_fields,
            "dual_arm_order": "left_then_right" if robot_type == "dual" else None,
        },
        "statistics": {
            "mode": config.stats_mode,
            "source_path": str(external_stats_path),
            "sha256": hashlib.sha256(output_stats_path.read_bytes()).hexdigest(),
            "source_sha256": external_stats_sha256,
            "runtime_action_stats_adapter": (
                action_stats.get("action_stats_runtime_adapter")
                if action_stats is not None
                else None
            ),
            "semantic_validation": config.stats_mode == "generated",
            "semantic_metadata_path": (
                str(external_stats_path.with_suffix(".metadata.json"))
                if config.stats_mode == "generated"
                else None
            ),
            "copied_semantic_metadata_file": (
                "dataset_statistics.metadata.json" if config.stats_mode == "generated" else None
            ),
            "normalization_implementation": (
                "cosmos_utils.load_dataset_stats + CosmosPolicy.rescale_action + "
                "cosmos_utils.rescale_proprio"
            ),
            "normalization_clip": False,
            "range_report": stats_range_report,
        },
        "images": {
            "camera_names_in_hdf5": camera_names,
            "size": config.image_size,
            "camera_state": config.camera_state,
            "wrist_crop_mode": config.wrist_crop_mode,
            "wrist_crop_fraction": config.wrist_crop_fraction,
            "wrist_left_center_offset_x": config.wrist_left_center_offset_x,
            "wrist_right_center_offset_x": config.wrist_right_center_offset_x,
            "head_crop_top_pixels": config.head_crop_top_pixels,
            "use_jpeg_compression": config.use_jpeg_compression,
            "trained_with_image_aug": config.trained_with_image_aug,
            "raw_frame_slices": {
                key: [value.start, value.stop] for key, value in RAW_FRAME_SLICES.items()
            },
        },
        "vae_encode": {
            "parallelism": (
                "intra_episode_microbatch_sharding"
                if config.encode_world_size > 1 or config.encode_device_ids is not None
                else "single_gpu"
            ),
            "requested_world_size": config.encode_world_size,
            "requested_device_ids": (
                list(config.encode_device_ids) if config.encode_device_ids is not None else None
            ),
            "encode_batch_size": config.encode_batch_size,
            "episode_parallelism": False,
            "writer_processes": 1,
            "implementation": (
                "data_convert_refactored.encoding.multi_gpu_vae.VAEReplicaPool"
                if config.encode_world_size > 1 or config.encode_device_ids is not None
                else "CosmosPolicy.encode"
            ),
            "replica_strategy": (
                "independent_wrapper_initialization"
                if config.encode_world_size > 1 or config.encode_device_ids is not None
                else None
            ),
            "replica_cache_scope": (
                "conversion_process"
                if config.encode_world_size > 1 or config.encode_device_ids is not None
                else None
            ),
            "clean_restore_latent": {
                "enabled": config.save_clean_restore_latent,
                "field": "clean_restore_latent" if config.save_clean_restore_latent else None,
                "dtype": "float16" if config.save_clean_restore_latent else None,
                "shape": [16, len(CLEAN_RESTORE_INDICES), 28, 28]
                if config.save_clean_restore_latent
                else None,
                "restore_indices": list(CLEAN_RESTORE_INDICES),
                "purpose": "restore robot-conditioning slots before VAE decode",
            },
        },
        "latent_indices": LATENT_INDICES,
        "kinematics": (
            {
                "config_file": str(Path(config.kinematics_config_path).expanduser().resolve()),
                "fk_validation": fk_validation,
            }
            if config.action_source == "master_joint_fk_same_frame"
            else None
        ),
        "source_fps": detected_fps,
        "cosmos_conditioning_fps": COSMOS_CONDITIONING_FPS,
        "timing": {
            "source_fps": detected_fps,
            "lerobot_metadata_fps": detected_fps,
            "cosmos_conditioning_fps": COSMOS_CONDITIONING_FPS,
            "action_step_seconds": 1.0 / detected_fps,
            "future_offset_frames": config.chunk_size,
            "future_offset_seconds": config.chunk_size / detected_fps,
        },
        "reward_done_value_modified": True,
    }
    with (output_dir / "cosmos_dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
