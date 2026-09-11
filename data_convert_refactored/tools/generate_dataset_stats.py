#!/usr/bin/env python3
"""从多个训练目录生成共享action/proprio statistics及语义sidecar。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ActionEncoding, ActionScale, ActionSource
from ..preparation.actions import EncodedEpisode, make_action_encoder
from ..preparation.dataset_stats import compute_shared_stats, write_shared_stats
from ..preparation.episode import discover_episodes, load_episode
from ..preparation.kinematics import DualArmKinematics


def main() -> None:
    """合并全部输入episode计算stats，禁止混合单臂和双臂数据。"""
    parser = argparse.ArgumentParser(description="Generate shared Cosmos action/proprio statistics")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-source", choices=[x.value for x in ActionSource], required=True)
    parser.add_argument("--action-encoding", choices=[x.value for x in ActionEncoding], required=True)
    parser.add_argument("--translation-scale", type=float, default=0.02)
    parser.add_argument("--rotation-scale", type=float, default=0.06)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--kinematics-config", type=Path)
    args = parser.parse_args()

    inputs = [path.expanduser().resolve() for path in args.input]
    paths = sorted({path for root in inputs for path in discover_episodes(root)})
    if not paths:
        parser.error("no trajectory.hdf5 found under --input")
    episodes = [load_episode(path) for path in paths]
    dual = episodes[0].is_dual_arm
    if any(item.is_dual_arm != dual for item in episodes):
        parser.error("cannot mix single-arm and dual-arm episodes")

    source = ActionSource(args.action_source)
    encoding = ActionEncoding(args.action_encoding)
    scale = ActionScale(args.translation_scale, args.rotation_scale, args.gripper_scale)
    scale.validate(encoding)
    kinematics = None
    if source is ActionSource.MASTER_JOINT_FK_SAME_FRAME:
        if args.kinematics_config is None:
            parser.error("--kinematics-config is required for master_joint_fk_same_frame")
        kinematics = DualArmKinematics(args.kinematics_config.expanduser().resolve())
    encoder = make_action_encoder(source, encoding, scale, kinematics)
    encoded = [
        EncodedEpisode(item, encoder.encode_episode(item), source, encoding, encoder.last_action_is_padding)
        for item in episodes
    ]
    stats = compute_shared_stats(episodes, encoded)
    output, sidecar = write_shared_stats(
        args.output.expanduser().resolve(),
        stats,
        inputs=inputs,
        robot_type="dual" if dual else "single_absolute",
        action_source=source,
        action_encoding=encoding,
        action_scale=scale,
        action_dimension=encoded[0].action_dim,
        proprio_dimension=16 if dual else 8,
    )
    print(f"stats: {output}")
    print(f"metadata: {sidecar}")
    print(f"episodes: {len(episodes)}")


if __name__ == "__main__":
    main()
