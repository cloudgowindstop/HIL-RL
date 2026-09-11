#!/usr/bin/env python3
"""
验证 parquet action → 积分反推 HDF5 绝对位姿，逐分量报告误差。

用法:
  python3 validate_conversion.py \
      --parquet dataset_cosmos/{task}_cosmos/data/chunk-000/episode_000000.parquet \
      --hdf5-dir dataset_raw/{task}/
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def pose_7d_to_matrix(pose_7d):
    xyz = pose_7d[:3].astype(np.float64)
    quat = pose_7d[3:7].astype(np.float64)
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def extract_ee_pose_and_gripper(h5, t, robot_type):
    if robot_type == "dual":
        left_ee = np.asarray(h5["puppet/end_effector_left_pose_align/data"][t], dtype=np.float32)
        right_ee = np.asarray(h5["puppet/end_effector_right_pose_align/data"][t], dtype=np.float32)
        left_grip = np.asarray(h5["puppet/end_effector_left_position_align/data"][t], dtype=np.float32).reshape(1)
        right_grip = np.asarray(h5["puppet/end_effector_right_position_align/data"][t], dtype=np.float32).reshape(1)
        return (np.concatenate([left_ee, right_ee]),
                np.concatenate([left_grip, right_grip]))
    elif "end_effector_single_pose_align" in h5["puppet"]:
        ee = np.asarray(h5["puppet/end_effector_single_pose_align/data"][t], dtype=np.float32)
        grip = np.asarray(h5["puppet/end_effector_single_position_align/data"][t], dtype=np.float32).reshape(1)
        return ee, grip
    else:
        ee = np.asarray(h5["puppet/end_effector/data"][t], dtype=np.float32)
        grip = np.asarray(h5["puppet/hand_joint_position/data"][t], dtype=np.float32).reshape(1)
        return ee, grip


def _quat_diff(q1, q2):
    return min(np.abs(q1 - q2).max(), np.abs(q1 + q2).max())


def integrate_action(xyz_quat, grip, action, action_scale):
    scale_pos, scale_rot, scale_grip = action_scale
    dpos = action[:3] * scale_pos
    drot = action[3:6] * scale_rot
    dgrip = action[6] * scale_grip
    T_curr = pose_7d_to_matrix(xyz_quat)
    T_delta = np.eye(4)
    T_delta[:3, 3] = dpos
    T_delta[:3, :3] = Rotation.from_euler("xyz", drot).as_matrix()
    T_new = T_curr @ T_delta
    new_xyz = T_new[:3, 3].astype(np.float32)
    new_quat = Rotation.from_matrix(T_new[:3, :3]).as_quat().astype(np.float32)
    new_grip = np.clip(np.float32(grip + dgrip), 0.0, 1.0)
    return np.concatenate([new_xyz, new_quat]), new_grip


def main():
    parser = argparse.ArgumentParser(description="验证 parquet → HDF5 action 精度")
    parser.add_argument("--parquet", required=True, help="episode_*.parquet 路径")
    parser.add_argument("--hdf5-dir", required=True, help="HDF5 目录")
    parser.add_argument("--action-scale", default=None,
                        help="手动指定, 格式 pos,rot,grip")
    parser.add_argument("--robot-type", default="dual",
                        choices=["dual", "single_absolute", "single_delta"])
    parser.add_argument("--out", default=None,
                        help="输出目录 (保存重建轨迹 npy)")
    args = parser.parse_args()

    pq_path = Path(args.parquet).expanduser().resolve()
    hdf5_dir = Path(args.hdf5_dir).expanduser().resolve()
    if not pq_path.exists():
        print(f"[ERROR] parquet 不存在: {pq_path}"); sys.exit(1)
    if not hdf5_dir.exists():
        print(f"[ERROR] HDF5 目录不存在: {hdf5_dir}"); sys.exit(1)

    robot_type = args.robot_type
    is_dual = (robot_type == "dual")

    # ── 1. 读 parquet ──
    print(f"[INFO] 读取 parquet: {pq_path.name}")
    table = pq.read_table(str(pq_path))
    actions_raw = table["action"].combine_chunks().to_pylist()
    T_pq = len(actions_raw)
    actions = np.stack([np.array(a, dtype=np.float32) for a in actions_raw], axis=0)
    print(f"[INFO] parquet: {T_pq} frames, action shape={actions.shape}")

    # ── 2. HDF5 ──
    all_hdf5 = sorted(hdf5_dir.rglob("trajectory.hdf5"))
    ep_idx = int(pq_path.stem.split("_")[-1])
    h5_path = all_hdf5[ep_idx] if ep_idx < len(all_hdf5) else None
    if h5_path is None:
        print(f"[ERROR] episode {ep_idx} 超出范围 ({len(all_hdf5)} HDF5s)")
        sys.exit(1)

    # ── 3. action_scale ──
    if args.action_scale:
        parts = [float(x) for x in args.action_scale.split(",")]
        action_scale = (parts[0], parts[1], parts[2])
        print(f"[INFO] action_scale (手动) = ({action_scale[0]:.10f}, {action_scale[1]:.10f}, {action_scale[2]:.10f})")
    else:
        traj_files = all_hdf5[:5]
        all_dpos, all_drot, all_dgrip = [], [], []
        for h5f in traj_files:
            with h5py.File(h5f, "r") as f:
                if robot_type == "dual":
                    key_pairs = [("puppet/end_effector_left_pose_align/data",
                                  "puppet/end_effector_left_position_align/data"),
                                 ("puppet/end_effector_right_pose_align/data",
                                  "puppet/end_effector_right_position_align/data")]
                elif "end_effector_single_pose_align" in f["puppet"]:
                    key_pairs = [("puppet/end_effector_single_pose_align/data",
                                  "puppet/end_effector_single_position_align/data")]
                else:
                    key_pairs = [("puppet/end_effector/data", "puppet/hand_joint_position/data")]
                T_f = len(f[key_pairs[0][0]])
                for t in range(T_f - 1):
                    for pk, gk in key_pairs:
                        T_c = pose_7d_to_matrix(f[pk][t])
                        T_fu = pose_7d_to_matrix(f[pk][t + 1])
                        T_d = np.linalg.inv(T_c) @ T_fu
                        all_dpos.append(np.abs(T_d[:3, 3]))
                        all_drot.append(np.abs(Rotation.from_matrix(T_d[:3, :3]).as_euler("xyz")))
                        all_dgrip.append(np.abs(f[gk][t + 1] - f[gk][t]))
        action_scale = (max(float(np.percentile(np.concatenate(all_dpos), 99)), 0.001),
                        max(float(np.percentile(np.concatenate(all_drot), 99)), 0.001),
                        max(float(np.percentile(np.concatenate(all_dgrip), 99)), 0.01))
        print(f"[INFO] action_scale (from {len(traj_files)} HDF5s) = ({action_scale[0]:.10f}, {action_scale[1]:.10f}, {action_scale[2]:.10f})")

    # ── 4. 积分 + 逐分量误差 ──
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        T_h5 = h5["metadata/trajectory_length"][()]
        print(f"[INFO] HDF5: {T_h5} frames")

        pose0, grip0 = extract_ee_pose_and_gripper(h5, 0, robot_type)

        if is_dual:
            # 双臂: 分别追踪 left/right
            L_p = pose0[:7].copy()
            L_g = float(grip0[0])
            R_p = pose0[7:].copy()
            R_g = float(grip0[1])

            # 累积各分量误差
            accum = {
                "L_dx": [], "L_dy": [], "L_dz": [],
                "L_rx": [], "L_ry": [], "L_rz": [], "L_grip": [],
                "R_dx": [], "R_dy": [], "R_dz": [],
                "R_rx": [], "R_ry": [], "R_rz": [], "R_grip": [],
                "L_quat": [], "R_quat": [],
            }
            recon_poses = [np.concatenate([L_p, R_p, grip0])]  # (16,)

            # 逐 action 分量真实值 (从 HDF5 计算，用于对比)
            action_true = []

            for t in range(T_h5 - 1):
                act = actions[t]
                L_p, L_g = integrate_action(L_p, L_g, act[:7], action_scale)
                R_p, R_g = integrate_action(R_p, R_g, act[7:], action_scale)

                h5_pose, h5_grip = extract_ee_pose_and_gripper(h5, t + 1, robot_type)
                recon_poses.append(np.concatenate([L_p, R_p, np.array([L_g, R_g])]))

                # ── 重建的 delta action (反推) ──
                recon_L_dpos = (act[:3] * action_scale[0]).astype(np.float32)
                recon_L_drot = (act[3:6] * action_scale[1]).astype(np.float32)
                recon_L_dgrip = float(act[6] * action_scale[2])

                # ── 真实的 delta action (从 HDF5) ──
                T_c_L = pose_7d_to_matrix(pose0[:7] if t == 0 else h5["puppet/end_effector_left_pose_align/data"][t].astype(np.float64))
                T_f_L = pose_7d_to_matrix(h5["puppet/end_effector_left_pose_align/data"][t + 1].astype(np.float64))
                T_d_L = np.linalg.inv(T_c_L) @ T_f_L
                true_L_dpos = T_d_L[:3, 3].astype(np.float32)
                true_L_drot = Rotation.from_matrix(T_d_L[:3, :3]).as_euler("xyz").astype(np.float32)
                true_L_dgrip = float(h5["puppet/end_effector_left_position_align/data"][t + 1] -
                                     (h5["puppet/end_effector_left_position_align/data"][t] if t == 0 else h5["puppet/end_effector_left_position_align/data"][t]))

                T_c_R = pose_7d_to_matrix(pose0[7:] if t == 0 else h5["puppet/end_effector_right_pose_align/data"][t].astype(np.float64))
                T_f_R = pose_7d_to_matrix(h5["puppet/end_effector_right_pose_align/data"][t + 1].astype(np.float64))
                T_d_R = np.linalg.inv(T_c_R) @ T_f_R
                true_R_dpos = T_d_R[:3, 3].astype(np.float32)
                true_R_drot = Rotation.from_matrix(T_d_R[:3, :3]).as_euler("xyz").astype(np.float32)
                true_R_dgrip = float(h5["puppet/end_effector_right_position_align/data"][t + 1] -
                                     (h5["puppet/end_effector_right_position_align/data"][t] if t == 0 else h5["puppet/end_effector_right_position_align/data"][t]))

                # ── 累积各分量误差 ──
                accum["L_dx"].append(abs(recon_L_dpos[0] - true_L_dpos[0]))
                accum["L_dy"].append(abs(recon_L_dpos[1] - true_L_dpos[1]))
                accum["L_dz"].append(abs(recon_L_dpos[2] - true_L_dpos[2]))
                accum["L_rx"].append(abs(recon_L_drot[0] - true_L_drot[0]))
                accum["L_ry"].append(abs(recon_L_drot[1] - true_L_drot[1]))
                accum["L_rz"].append(abs(recon_L_drot[2] - true_L_drot[2]))
                accum["L_grip"].append(abs(recon_L_dgrip - true_L_dgrip))
                accum["L_quat"].append(_quat_diff(L_p[3:], h5_pose[3:7]))

                accum["R_dx"].append(abs(recon_R_dpos[0] - true_R_dpos[0]))
                accum["R_dy"].append(abs(recon_R_dpos[1] - true_R_dpos[1]))
                accum["R_dz"].append(abs(recon_R_dpos[2] - true_R_dpos[2]))
                accum["R_rx"].append(abs(recon_R_drot[0] - true_R_drot[0]))
                accum["R_ry"].append(abs(recon_R_drot[1] - true_R_drot[1]))
                accum["R_rz"].append(abs(recon_R_drot[2] - true_R_drot[2]))
                accum["R_grip"].append(abs(recon_R_dgrip - true_R_dgrip))
                accum["R_quat"].append(_quat_diff(R_p[3:], h5_pose[10:13]))

        else:
            # 单臂
            xyz_quat = pose0.copy()
            grip_val = float(grip0[0])
            accum = {
                "dx": [], "dy": [], "dz": [],
                "rx": [], "ry": [], "rz": [], "grip": [], "quat": [],
            }
            recon_poses = [np.concatenate([pose0, grip0])]

            for t in range(T_h5 - 1):
                act = actions[t]
                xyz_quat, grip_val = integrate_action(xyz_quat, grip_val, act, action_scale)
                h5_pose, h5_grip = extract_ee_pose_and_gripper(h5, t + 1, robot_type)
                recon_poses.append(np.concatenate([xyz_quat, np.array([grip_val])]))

                recon_dpos = (act[:3] * action_scale[0]).astype(np.float32)
                recon_drot = (act[3:6] * action_scale[1]).astype(np.float32)
                recon_dgrip = float(act[6] * action_scale[2])

                T_c = pose_7d_to_matrix(pose0 if t == 0 else h5_pose)
                T_f = pose_7d_to_matrix(h5["puppet/end_effector_single_pose_align/data" if "end_effector_single_pose_align" in h5["puppet"] else "puppet/end_effector/data"][t + 1].astype(np.float64))
                T_d = np.linalg.inv(T_c) @ T_f
                true_dpos = T_d[:3, 3].astype(np.float32)
                true_drot = Rotation.from_matrix(T_d[:3, :3]).as_euler("xyz").astype(np.float32)
                true_dgrip = float(h5_grip[0] - grip0[0] if t == 0 else h5_grip[0] - float(h5["puppet/end_effector_single_position_align/data" if "end_effector_single_position_align" in h5["puppet"] else "puppet/hand_joint_position/data"][t]))

                accum["dx"].append(abs(recon_dpos[0] - true_dpos[0]))
                accum["dy"].append(abs(recon_dpos[1] - true_dpos[1]))
                accum["dz"].append(abs(recon_dpos[2] - true_dpos[2]))
                accum["rx"].append(abs(recon_drot[0] - true_drot[0]))
                accum["ry"].append(abs(recon_drot[1] - true_drot[1]))
                accum["rz"].append(abs(recon_drot[2] - true_drot[2]))
                accum["grip"].append(abs(recon_dgrip - true_dgrip))
                accum["quat"].append(_quat_diff(xyz_quat[3:], h5_pose[3:7]))

    # ── 5. 保存重建轨迹 ──
    if out_dir:
        # 从 parquet 路径自动提取任务名: .../{task}_cosmos/... → {task}
        pq_parts = pq_path.parts
        task_name = "unknown"
        for p in pq_parts:
            if p.endswith("_cosmos"):
                task_name = p.replace("_cosmos", "")
                break
        task_out = out_dir / task_name
        task_out.mkdir(parents=True, exist_ok=True)

        recon_arr = np.stack(recon_poses, axis=0)  # (T, D)
        out_path = task_out / f"recon_ep{ep_idx:06d}.npy"
        np.save(str(out_path), recon_arr)
        print(f"[INFO] 重建轨迹保存到: {out_path}  shape={recon_arr.shape}")

        # 同时保存 HDF5 真实轨迹
        with h5py.File(h5_path, "r") as h5:
            true_poses = []
            for t in range(T_h5):
                p, g = extract_ee_pose_and_gripper(h5, t, robot_type)
                if is_dual:
                    true_poses.append(np.concatenate([p, g]))
                else:
                    true_poses.append(np.concatenate([p, g]))
            true_arr = np.stack(true_poses, axis=0)
        true_path = task_out / f"true_ep{ep_idx:06d}.npy"
        np.save(str(true_path), true_arr.astype(np.float32))
        print(f"[INFO] 真实轨迹保存到: {true_path}  shape={true_arr.shape}")

    # ── 6. 逐分量误差报告 ──
    print()
    print("=" * 80)
    print(f"{'Component':<10} {'Mean':>12} {'Max':>12} {'Std':>12} {'P99':>12}")
    print("-" * 80)
    for name, errs in accum.items():
        errs = np.array(errs)
        print(f"{name:<10} {errs.mean():>12.4e} {errs.max():>12.4e} {errs.std():>12.4e} "
              f"{np.percentile(errs, 99):>12.4e}")
    print("-" * 80)
    all_errs = np.concatenate([np.array(v) for v in accum.values()])
    print(f"{'OVERALL':<10} {all_errs.mean():>12.4e} {all_errs.max():>12.4e} {all_errs.std():>12.4e} "
          f"{np.percentile(all_errs, 99):>12.4e}")
    print("=" * 80)

    if all_errs.max() < 0.01:
        print("✅ 验证通过")
    else:
        print("⚠️  最大误差 > 0.01，请检查 (通常为夹爪浮动累积)")


if __name__ == "__main__":
    main()
