#!/usr/bin/env python3
"""Extract real-unit robot replay data from a Cosmos rotation-6D dataset.

The action is read only from the converted Parquet dataset.  The HDF5 file is
used only to attach the robot reset state (initial TCP pose, joints, gripper).
The output is one ``.npy`` dictionary intended for explicit offline inspection
before it is used on hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from cosmos_rotation_6d import rotation_6d_to_matrix


def load_metadata(dataset: Path) -> dict:
    path = dataset / "cosmos_dataset_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Cosmos dataset metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    action = metadata.get("action", {})
    if action.get("encoding") != "cosmos_rotation_6d" or action.get("dimension") != 10:
        raise ValueError(
            "dataset must use 10D cosmos_rotation_6d actions; "
            f"got encoding={action.get('encoding')!r}, dimension={action.get('dimension')!r}"
        )
    return metadata


def load_episode_actions(dataset: Path, episode_index: int) -> tuple[Path, np.ndarray]:
    candidates = sorted(dataset.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one Parquet for episode {episode_index}, found {len(candidates)} under {dataset}"
        )
    parquet_path = candidates[0]
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table.column("action").combine_chunks().to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"expected action shape (T,10), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Parquet action contains NaN or Inf")
    return parquet_path, actions


def decode_metric_actions(
    actions_10d: np.ndarray,
    translation_scale: float,
    gripper_scale: float,
    drop_last_action: bool,
) -> tuple[np.ndarray, dict]:
    # Legacy puppet-next-frame data pads its final row by targeting itself.
    # Official master-same-frame data contains a valid control error at every row.
    replay_actions = actions_10d[:-1] if drop_last_action else actions_10d
    actions_7d = np.empty((len(replay_actions), 7), dtype=np.float32)
    rotation_errors = []
    determinants = []
    orthogonality_errors = []

    for index, action in enumerate(replay_actions):
        matrix = rotation_6d_to_matrix(action[3:9])
        euler_xyz = Rotation.from_matrix(matrix).as_euler("xyz")
        reconstructed = Rotation.from_euler("xyz", euler_xyz).as_matrix()
        error = matrix.T @ reconstructed

        actions_7d[index, :3] = action[:3] * translation_scale
        actions_7d[index, 3:6] = euler_xyz
        # Cosmos conversion stores an absolute gripper target, not a delta.
        actions_7d[index, 6] = action[9] * gripper_scale

        rotation_errors.append(
            np.degrees(np.linalg.norm(Rotation.from_matrix(error).as_rotvec()))
        )
        determinants.append(np.linalg.det(matrix))
        orthogonality_errors.append(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))

    validation = {
        "max_rotation_roundtrip_error_deg": float(max(rotation_errors, default=0.0)),
        "max_rotation_orthogonality_error": float(max(orthogonality_errors, default=0.0)),
        "min_rotation_determinant": float(min(determinants, default=1.0)),
        "max_rotation_determinant": float(max(determinants, default=1.0)),
    }
    return actions_7d, validation


def load_initial_state(hdf5_path: Path) -> dict:
    with h5py.File(hdf5_path, "r") as h5_file:
        pose_key = "puppet/end_effector_single_pose_align/data"
        joints_key = "puppet/arm_single_position_align/data"
        gripper_key = "puppet/end_effector_single_position_align/data"
        for key in (pose_key, joints_key, gripper_key):
            if key not in h5_file:
                raise ValueError(f"required single-arm initial-state field is missing: {key}")

        pose_xyzw = np.asarray(h5_file[pose_key][0], dtype=np.float32)
        joints = np.asarray(h5_file[joints_key][0], dtype=np.float32)
        gripper = float(np.asarray(h5_file[gripper_key][0]).reshape(-1)[0])
        trajectory_length = int(h5_file["metadata/trajectory_length"][()])

    if pose_xyzw.shape != (7,):
        raise ValueError(f"expected initial TCP pose shape (7,), got {pose_xyzw.shape}")
    if joints.shape != (6,):
        raise ValueError(f"expected initial joints shape (6,), got {joints.shape}")

    pose_xyz_euler = np.concatenate(
        (pose_xyzw[:3], Rotation.from_quat(pose_xyzw[3:7]).as_euler("xyz"))
    ).astype(np.float32)
    return {
        "initial_ee_pose_xyz_euler": pose_xyz_euler,
        "initial_ee_pose_xyz_quat_xyzw": pose_xyzw,
        "initial_joint_positions": joints,
        "initial_gripper": np.float32(gripper),
        "hdf5_trajectory_length": trajectory_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cosmos 10D rotation-6D -> real-unit 7D replay action"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    hdf5_path = args.hdf5.resolve()
    metadata = load_metadata(dataset)
    action_metadata = metadata["action"]
    translation_scale = float(action_metadata["translation_scale"])
    gripper_scale = float(action_metadata["gripper_scale"])

    parquet_path, actions_10d = load_episode_actions(dataset, args.episode_index)
    initial_state = load_initial_state(hdf5_path)
    if len(actions_10d) != initial_state["hdf5_trajectory_length"]:
        raise ValueError(
            "Parquet/HDF5 frame count mismatch: "
            f"{len(actions_10d)} != {initial_state['hdf5_trajectory_length']}"
        )

    metric_actions, validation = decode_metric_actions(
        actions_10d,
        translation_scale,
        gripper_scale,
        drop_last_action=action_metadata.get("source")
        not in ("master_same_frame", "master_joint_fk_same_frame"),
    )
    payload = {
        "format_version": 1,
        "action_units": "dx_dy_dz_meters__droll_dpitch_dyaw_radians__gripper_absolute",
        "action_frame": action_metadata.get("pose_semantics"),
        "gripper_semantics": action_metadata.get("gripper_semantics"),
        "source_dataset": str(dataset),
        "source_parquet": str(parquet_path.resolve()),
        "source_hdf5": str(hdf5_path),
        "episode_index": args.episode_index,
        "translation_scale_used": np.float32(translation_scale),
        "gripper_scale_used": np.float32(gripper_scale),
        "cosmos_action_10d": (
            actions_10d
            if action_metadata.get("source") in ("master_same_frame", "master_joint_fk_same_frame")
            else actions_10d[:-1]
        ).astype(np.float32),
        "metric_action_7d": metric_actions,
        **initial_state,
        "validation": validation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, payload, allow_pickle=True)
    print(f"saved: {args.output.resolve()}")
    print(f"metric_action_7d shape: {metric_actions.shape}")
    print(f"initial ee xyz+euler: {initial_state['initial_ee_pose_xyz_euler']}")
    print(f"initial joints: {initial_state['initial_joint_positions']}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
