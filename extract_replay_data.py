#!/usr/bin/env python3
"""
提取 LeRobot + HDF5 数据 → 真机复现用文件 + 验证

输出 (per episode):
  pose_abs.txt           HDF5 绝对位姿 (T, 8) 空格分隔
  action_delta.npy       物理 delta (T-1, 7) 已反归一化
  验证打印

用法:
  python3 extract_replay_data.py \
      --parquet dataset_cosmos/pick_spoon_cosmos/data/chunk-000/episode_000000.parquet \
      --hdf5-dir dataset_raw/pick_spoon/ \
      --action-scale 0.006154,0.007708,0.164157 \
      --out pose_delta/pick_spoon/
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


def extract_ee_pose_and_gripper(h5, t):
    if "end_effector_single_pose_align" in h5["puppet"]:
        ee = np.asarray(h5["puppet/end_effector_single_pose_align/data"][t], dtype=np.float64)
        grip = float(h5["puppet/end_effector_single_position_align/data"][t])
    elif "end_effector_left_pose_align" in h5["puppet"]:
        left = np.asarray(h5["puppet/end_effector_left_pose_align/data"][t], dtype=np.float64)
        right = np.asarray(h5["puppet/end_effector_right_pose_align/data"][t], dtype=np.float64)
        Lg = float(h5["puppet/end_effector_left_position_align/data"][t])
        Rg = float(h5["puppet/end_effector_right_position_align/data"][t])
        return np.concatenate([left, right]), np.array([Lg, Rg])
    else:
        ee = np.asarray(h5["puppet/end_effector/data"][t], dtype=np.float64)
        grip = float(h5["puppet/hand_joint_position/data"][t])
    return ee, np.array([grip])


def _quat_diff(q1, q2):
    return min(np.abs(q1 - q2).max(), np.abs(q1 + q2).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--hdf5-dir", required=True)
    parser.add_argument("--action-scale", required=True,
                        help="pos,rot,grip 三个 scale, 逗号分隔")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parts = [float(x) for x in args.action_scale.split(",")]
    scale_pos, scale_rot, scale_grip = parts

    # ── 读 parquet ──
    pq_path = Path(args.parquet)
    ep_idx = int(pq_path.stem.split("_")[-1])
    table = pq.read_table(str(pq_path))
    actions_norm = np.stack([np.array(a, np.float32)
                             for a in table["action"].combine_chunks().to_pylist()])
    T_pq = len(actions_norm)
    print(f"[parquet] {T_pq} frames, action={actions_norm.shape}")

    # ── 定位 HDF5 ──
    all_hdf5 = sorted(Path(args.hdf5_dir).rglob("trajectory.hdf5"))
    h5_path = all_hdf5[ep_idx]
    with h5py.File(h5_path, "r") as h5:
        T_h5 = h5["metadata/trajectory_length"][()]
        lang = h5["metadata/language_instruction"][()]
        print(f"[HDF5] {h5_path.name}  {T_h5} frames  task={lang}")

        if T_pq != T_h5:
            print(f"ERROR: 帧数不一致"); sys.exit(1)

        is_dual = "end_effector_left_pose_align" in h5["puppet"]
        D = 16 if is_dual else 8
        D_action = 14 if is_dual else 7

        # 读取全部绝对位姿
        abs_poses = []
        for t in range(T_h5):
            ee, grip = extract_ee_pose_and_gripper(h5, t)
            abs_poses.append(np.concatenate([ee, grip.ravel()]))
        abs_poses = np.array(abs_poses, dtype=np.float32)  # (T, D)

        # ── 反归一化 action → 物理 delta ──
        actions_phys = np.zeros((T_pq - 1, D_action), dtype=np.float32)
        for t in range(T_pq - 1):
            a = actions_norm[t].astype(np.float64)
            if is_dual:
                n_per_arm = D_action // 2
                for arm_off in [0, n_per_arm]:
                    a[arm_off:arm_off+3] *= scale_pos
                    a[arm_off+3:arm_off+6] *= scale_rot
                    a[arm_off+6] *= scale_grip
            else:
                a[:3] *= scale_pos
                a[3:6] *= scale_rot
                a[6] *= scale_grip
            actions_phys[t] = a.astype(np.float32)

        # ── 积分验证 ──
        max_pos_err = 0.0
        max_quat_err = 0.0
        max_grip_err = 0.0
        recon = abs_poses[0].astype(np.float64).copy()

        for t in range(T_h5 - 1):
            a = actions_phys[t].astype(np.float64)
            if is_dual:
                n = D_action // 2
                for arm_idx, base in enumerate([0, n]):
                    T_c = pose_7d_to_matrix(recon[base:base+7])
                    T_d = np.eye(4)
                    T_d[:3, 3] = a[base:base+3]
                    T_d[:3, :3] = Rotation.from_euler("xyz", a[base+3:base+6]).as_matrix()
                    T_n = T_c @ T_d
                    recon[base:base+3] = T_n[:3, 3]
                    recon[base+3:base+7] = Rotation.from_matrix(T_n[:3, :3]).as_quat()
                    recon[7+arm_idx] = np.clip(recon[7+arm_idx] + a[base+6], 0.0, 1.0)
            else:
                T_c = pose_7d_to_matrix(recon[:7])
                T_d = np.eye(4)
                T_d[:3, 3] = a[:3]
                T_d[:3, :3] = Rotation.from_euler("xyz", a[3:6]).as_matrix()
                T_n = T_c @ T_d
                recon[:3] = T_n[:3, 3]
                recon[3:7] = Rotation.from_matrix(T_n[:3, :3]).as_quat()
                recon[7] = np.clip(recon[7] + a[6], 0.0, 1.0)

            true_pose = abs_poses[t + 1].astype(np.float64)
            pos_err = np.abs(recon[:3] - true_pose[:3]).max()
            if is_dual:
                pos_err = max(pos_err, np.abs(recon[7:10] - true_pose[7:10]).max())
            quat_err = _quat_diff(recon[3:7], true_pose[3:7])
            if is_dual:
                quat_err = max(quat_err, _quat_diff(recon[10:14], true_pose[10:14]))
            grip_err = np.abs(recon[7] - true_pose[7])
            if is_dual:
                grip_err = max(grip_err, np.abs(recon[14] - true_pose[14]))

            max_pos_err = max(max_pos_err, pos_err)
            max_quat_err = max(max_quat_err, quat_err)
            max_grip_err = max(max_grip_err, grip_err)

        # ── 保存 ──
        out_dir = Path(args.out) / f"ep_{ep_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(str(out_dir / "pose_abs.txt"), abs_poses, fmt="%.6f")
        np.save(str(out_dir / "action_delta.npy"), actions_phys)
        np.savetxt(str(out_dir / "action_delta.txt"), actions_phys,
                   fmt="%.6f", header="dx dy dz rx ry rz dgrip")

        # 初始位姿：xyz+欧拉角 + 关节角 + 夹爪
        init_xyz = abs_poses[0, :3]
        init_quat = abs_poses[0, 3:7]
        init_euler = Rotation.from_quat(init_quat).as_euler("xyz")
        init_grip = abs_poses[0, 7]
        init_joints = h5["puppet/arm_single_position_align/data" if "arm_single_position_align" in h5["puppet"]
                         else "puppet/arm_left_position_align/data"][0]
        if is_dual:
            init_joints_r = h5["puppet/arm_right_position_align/data"][0]
            init_joints_full = np.concatenate([init_joints, init_joints_r])
        else:
            init_joints_full = init_joints

        with open(str(out_dir / "init_pose.txt"), "w") as fh:
            fh.write(f"# xyz (m):       {' '.join(f'{v:.6f}' for v in init_xyz)}\n")
            fh.write(f"# euler (rad):   {' '.join(f'{v:.6f}' for v in init_euler)}\n")
            fh.write(f"# quat:          {' '.join(f'{v:.6f}' for v in init_quat)}\n")
            fh.write(f"# grip [0,1]:    {init_grip:.6f}\n")
            fh.write(f"# joints (rad):  {' '.join(f'{v:.6f}' for v in init_joints_full)}\n")

        print(f"\n{'='*60}")
        print(f"episode {ep_idx}: {T_h5} frames ({lang})")
        print(f"  绝对位姿 → {out_dir / 'pose_abs.txt'}  shape={abs_poses.shape}")
        print(f"  物理delta → {out_dir / 'action_delta.npy'}  shape={actions_phys.shape}")
        print(f"\n  积分验证:")
        print(f"    最大位置误差: {max_pos_err:.4e} m")
        print(f"    最大姿态误差: {max_quat_err:.4e}")
        print(f"    最大夹爪误差: {max_grip_err:.4e}")
        print(f"    {'✅ 通过' if max(max_pos_err, max_quat_err, max_grip_err) < 0.01 else '⚠️  超出阈值'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
