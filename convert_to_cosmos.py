#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert per-trajectory HDF5 files into OURDataset demo+rollout format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing trajectory folders (each has data/trajectory.hdf5).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root. Will create demo_data/ and rollout_data/ inside.",
    )
    parser.add_argument(
        "--demo_ratio",
        type=float,
        default=0.8,
        help="Split ratio for demo files. Remaining files become rollout files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before split.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="",
        help="Optional override for rollout task_description attr.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N trajectories after shuffle (0 means all).",
    )
    return parser.parse_args()


def collect_trajectory_files(input_dir: Path) -> list[Path]:
    files = []
    # files列表内每个都是path对象
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        candidate = sub / "data" / "trajectory.hdf5"
        if candidate.exists():
            files.append(candidate)
    return files


def decode_language_instruction(src_h5: h5py.File, fallback: str) -> str:
    if "language_instruction" not in src_h5:
        return fallback
    value = src_h5["language_instruction"][()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def has_jpeg_storage(image_ds: h5py.Dataset) -> bool:
    # JPEG压缩图像会存字节流，类型是对象 kind='0'
    # 普通RGB图像类型是数字数组，kind='f'或'u'
    return image_ds.dtype.kind == "O"


def read_actions_and_proprio(src_h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    puppet = src_h5["puppet"]

    if "delta_end_effector" in puppet:
        print("-----------------> use delta_end_effector as actions")
        actions = puppet["delta_end_effector"][:].astype(np.float32)
    elif "end_effector" in puppet:
        actions = puppet["end_effector"][:].astype(np.float32)
    else:
        raise KeyError("No action key found in puppet group (delta_end_effector/end_effector).")

    if "end_effector" in puppet and "hand_joint_position" in puppet:
        # arm = puppet["arm_joint_position"][:].astype(np.float32)
        print("-----------------> use end_effector+hand_joint_position as proprio")
        arm_end_effector = puppet["end_effector"][:].astype(np.float32)
        hand = puppet["hand_joint_position"][:].astype(np.float32)
        proprio = np.concatenate([arm_end_effector, hand], axis=-1)
    else:
        raise KeyError("No proprio key found in puppet group.")

    return actions, proprio


def write_vlen_uint8_dataset(group: h5py.Group, key: str, seq_obj_array: np.ndarray):
    dt = h5py.vlen_dtype(np.dtype("uint8"))
    ds = group.create_dataset(key, (len(seq_obj_array),), dtype=dt)
    for i, item in enumerate(seq_obj_array):
        ds[i] = np.asarray(item, dtype=np.uint8)


def write_demo_file(src_file: Path, dst_file: Path):
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w") as dst:
        print("src_file:", src)
        right_ds = src["observations"]["rgb_images"]["camera_right"]
        wrist_ds = src["observations"]["rgb_images"]["camera_wrist"]
        
        actions, proprio = read_actions_and_proprio(src)

        data_group = dst.create_group("data")
        demo_group = data_group.create_group("demo_0")
        obs_group = demo_group.create_group("obs")

        if has_jpeg_storage(right_ds):
            print("-----------------> use JPEG storage")
            write_vlen_uint8_dataset(obs_group, "agentview_rgb_jpeg", right_ds[:])
        else:
            print("-----------------> no JPEG storage")
            obs_group.create_dataset("agentview_rgb", data=right_ds[:], compression="gzip")

        if has_jpeg_storage(wrist_ds):
            print("-----------------> use JPEG storage")
            write_vlen_uint8_dataset(obs_group, "eye_in_hand_rgb_jpeg", wrist_ds[:])
        else:
            print("-----------------> no JPEG storage")
            obs_group.create_dataset("eye_in_hand_rgb", data=wrist_ds[:], compression="gzip")

        demo_group.create_dataset("actions", data=actions, compression="gzip")
        demo_group.create_dataset("robot_states", data=proprio, compression="gzip")


def write_rollout_file(src_file: Path, dst_file: Path, task_description_override: str):
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w") as dst:
        right_ds = src["observations"]["rgb_images"]["camera_right"]
        wrist_ds = src["observations"]["rgb_images"]["camera_wrist"]
        actions, proprio = read_actions_and_proprio(src)

        is_jpeg = has_jpeg_storage(right_ds)
        if is_jpeg:
            print("-----------------> use JPEG storage")
            write_vlen_uint8_dataset(dst, "primary_images_jpeg", right_ds[:])
            write_vlen_uint8_dataset(dst, "wrist_images_jpeg", wrist_ds[:])
        else:
            print("-----------------> no JPEG storage")
            dst.create_dataset("primary_images", data=right_ds[:], compression="gzip")
            dst.create_dataset("wrist_images", data=wrist_ds[:], compression="gzip")

        dst.create_dataset("actions", data=actions, compression="gzip")
        dst.create_dataset("proprio", data=proprio, compression="gzip")

        # default_instruction = src_file.parent.parent.name.replace("_", " ")
        # task_description = task_description_override or decode_language_instruction(src, default_instruction)
        # dst.attrs["task_description"] = task_description
        dst.attrs["success"] = True


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not (0.0 <= args.demo_ratio <= 1.0):
        raise ValueError("--demo_ratio must be in [0, 1]")

    all_files = collect_trajectory_files(input_dir)
    if len(all_files) == 0:
        raise RuntimeError(f"No trajectory.hdf5 found under: {input_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(all_files)

    if args.limit > 0:
        all_files = all_files[: args.limit]

    split_idx = int(len(all_files) * args.demo_ratio)
    split_idx = max(1, min(split_idx, len(all_files) - 1))

    demo_files = sorted(all_files[:split_idx], key=lambda x: x.as_posix())
    rollout_files = sorted(all_files[split_idx:], key=lambda x: x.as_posix())

    demo_dir = output_dir / "demo_data"
    rollout_dir = output_dir / "rollout_data"
    if args.demo_ratio == 1:
        demo_dir = output_dir / "all_data"
        
    print(f"Total trajectories: {len(all_files)}")
    print(f"Demo trajectories: {len(demo_files)} -> {demo_dir}")
    print(f"Rollout trajectories: {len(rollout_files)} -> {rollout_dir}")

    for src_file in tqdm(demo_files, desc="Writing demo files"):
        traj_id = src_file.parent.parent.name
        out_file = demo_dir / f"{traj_id}.hdf5"    # .../demo/0527_202114.hdf5
        write_demo_file(src_file, out_file)

    for src_file in tqdm(rollout_files, desc="Writing rollout files"):
        traj_id = src_file.parent.parent.name
        out_file = rollout_dir / f"{traj_id}.hdf5"
        write_rollout_file(src_file, out_file, args.task_description)

    print("Done.")


if __name__ == "__main__":
    main()
