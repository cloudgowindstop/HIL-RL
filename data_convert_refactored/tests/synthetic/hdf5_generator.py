"""Generate synthetic HDF5 by recursive transforms, independent of the oracle."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .scenario_spec import SCENARIO, SyntheticScenario


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def _generator_poses(scenario: SyntheticScenario) -> np.ndarray:
    """Generate source poses by multiplying one local transform per timestep."""
    poses = np.zeros((scenario.length, 7), dtype=np.float64)
    current = np.eye(4, dtype=np.float64)
    cosine = np.cos(scenario.rotation_step_rad)
    sine = np.sin(scenario.rotation_step_rad)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    delta[:3, 3] = [scenario.translation_step_m, 0.0, 0.0]

    for timestep in range(scenario.length):
        yaw = np.arctan2(current[1, 0], current[0, 0])
        poses[timestep, :3] = current[:3, 3]
        poses[timestep, 5] = np.sin(yaw / 2.0)
        poses[timestep, 6] = np.cos(yaw / 2.0)
        current = current @ delta
    return poses.astype(np.float32)


def _generator_gripper(scenario: SyntheticScenario) -> np.ndarray:
    """Generate source gripper values with a loop, not the oracle's NumPy formula."""
    values = np.empty(scenario.length, dtype=np.float32)
    for timestep in range(scenario.length):
        values[timestep] = timestep / (scenario.length - 1)
    return values


def synthetic_image(timestep: int, camera: str) -> np.ndarray:
    """Create RGB image with exact time marker and horizontal/vertical gradients."""
    camera_offset = 20 if camera == "camera_wrist" else 80
    image = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[..., 0] = camera_offset + timestep * 7
    image[..., 1] = np.arange(IMAGE_WIDTH, dtype=np.uint16)[None, :] % 256
    image[..., 2] = np.arange(IMAGE_HEIGHT, dtype=np.uint16)[:, None] % 256
    return image


def create_synthetic_fixture(
    root: Path, scenario: SyntheticScenario = SCENARIO
) -> tuple[Path, Path]:
    """Write HDF5 and non-identity per-channel stats without using the oracle."""
    episode_dir = root / "input" / "episode_000000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = episode_dir / "trajectory.hdf5"
    stats_path = root / "dataset_statistics.json"
    poses = _generator_poses(scenario)
    gripper = _generator_gripper(scenario)
    timestamps = np.arange(scenario.length, dtype=np.float64) / scenario.source_fps

    with h5py.File(hdf5_path, "w") as h5_file:
        h5_file.create_dataset(
            "puppet/end_effector_single_pose_align/data", data=poses
        )
        h5_file.create_dataset(
            "puppet/end_effector_single_position_align/data", data=gripper[:, None]
        )
        joints = np.stack(
            [
                0.1 * np.arange(scenario.length) + 0.01 * joint
                for joint in range(6)
            ],
            axis=1,
        ).astype(np.float32)
        h5_file.create_dataset("puppet/arm_single_position_align/data", data=joints)
        h5_file.create_dataset("camera_observations/timestamp", data=timestamps)
        images = h5_file.create_group("camera_observations/color_images")
        for camera in ("camera_right", "camera_wrist"):
            images.create_dataset(
                camera,
                data=np.stack(
                    [synthetic_image(timestep, camera) for timestep in range(scenario.length)]
                ),
                compression="gzip",
                compression_opts=1,
            )

    stats = {
        "actions_min": list(scenario.action_min),
        "actions_max": list(scenario.action_max),
        "proprio_min": list(scenario.proprio_min),
        "proprio_max": list(scenario.proprio_max),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return hdf5_path, stats_path

