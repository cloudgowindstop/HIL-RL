#!/usr/bin/env python3
"""Validate Tienyi-style dual-arm FK alignment and export 20D Cosmos actions without VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from convert_raw_to_cosmos import (
    encode_dual_arm_master_fk_action_6d,
    validate_fk_against_puppet_pose,
)
from robot_kinematics import DualArmKinematics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--kinematics-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--translation-scale", type=float, default=0.02)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--fk-max-position-error-m", type=float, default=0.005)
    parser.add_argument("--fk-max-rotation-error-deg", type=float, default=1.0)
    args = parser.parse_args()

    hdf5_path = args.hdf5.resolve()
    kinematics = DualArmKinematics(args.kinematics_config)
    fk_report = validate_fk_against_puppet_pose(
        hdf5_path,
        kinematics,
        args.fk_max_position_error_m,
        args.fk_max_rotation_error_deg,
    )
    with h5py.File(hdf5_path, "r") as h5_file:
        length = len(h5_file["puppet/arm_left_position_align/data"])
        actions = np.stack(
            [
                encode_dual_arm_master_fk_action_6d(
                    h5_file,
                    index,
                    kinematics,
                    args.translation_scale,
                    args.gripper_scale,
                )
                for index in range(length)
            ]
        )

    left_translation = actions[:, :3]
    right_translation = actions[:, 10:13]
    summary = {
        "frames": int(len(actions)),
        "action_shape": list(actions.shape),
        "action_source": "master_joint_fk_same_frame",
        "translation_scale": args.translation_scale,
        "gripper_scale": args.gripper_scale,
        "left_translation_clipped_frames": int(np.any(np.abs(left_translation) >= 1.0, axis=1).sum()),
        "right_translation_clipped_frames": int(np.any(np.abs(right_translation) >= 1.0, axis=1).sum()),
        "finite": bool(np.all(np.isfinite(actions))),
        "fk_validation": fk_report,
    }
    payload = {
        "format_version": 1,
        "source_hdf5": str(hdf5_path),
        "kinematics_config": str(kinematics.config_path),
        "cosmos_action_20d": actions.astype(np.float32),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, payload, allow_pickle=True)
    print(f"saved: {args.output.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
