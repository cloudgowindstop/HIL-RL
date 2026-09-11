"""Small dependency-free URDF FK adapter used by offline data conversion.

The adapter intentionally requires an explicit robot config.  Robot geometry,
joint order, zero offsets, signs, end-effector frame, and TCP must never be
guessed from an HDF5 trajectory.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _transform(xyz=None, rpy=None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if xyz is not None:
        result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    if rpy is not None:
        result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return result


def _parse_vector(value: str | None, default: list[float]) -> np.ndarray:
    return np.asarray(default if not value else [float(x) for x in value.split()], dtype=np.float64)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"invalid URDF joint axis: {axis}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(axis / norm * angle).as_matrix()
    return result


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class URDFKinematics:
    def __init__(self, config: dict, config_dir: Path):
        urdf_path = Path(config["urdf_path"]).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = config_dir / urdf_path
        self.urdf_path = urdf_path.resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF does not exist: {self.urdf_path}")

        self.base_link = str(config["base_link"])
        self.end_effector_link = str(config["end_effector_link"])
        self.joint_names = list(config["joint_names"])
        self.joint_units = str(config.get("joint_units", "radians"))
        if self.joint_units not in ("radians", "degrees"):
            raise ValueError("joint_units must be radians or degrees")
        n = len(self.joint_names)
        self.joint_signs = np.asarray(config.get("joint_signs", [1.0] * n), dtype=np.float64)
        self.joint_offsets = np.asarray(config.get("joint_offsets", [0.0] * n), dtype=np.float64)
        if self.joint_signs.shape != (n,) or self.joint_offsets.shape != (n,):
            raise ValueError("joint_signs and joint_offsets must match joint_names")
        self.tcp_transform = np.asarray(config.get("tcp_transform", np.eye(4)), dtype=np.float64)
        if self.tcp_transform.shape != (4, 4):
            raise ValueError("tcp_transform must be a 4x4 matrix")

        root = ET.parse(self.urdf_path).getroot()
        joints: dict[str, _Joint] = {}
        child_to_joint: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            xyz = _parse_vector(origin.get("xyz") if origin is not None else None, [0, 0, 0])
            rpy = _parse_vector(origin.get("rpy") if origin is not None else None, [0, 0, 0])
            axis_element = element.find("axis")
            axis = _parse_vector(axis_element.get("xyz") if axis_element is not None else None, [1, 0, 0])
            joint = _Joint(
                name=element.get("name", ""),
                kind=element.get("type", "fixed"),
                parent=element.find("parent").get("link"),
                child=element.find("child").get("link"),
                origin=_transform(xyz, rpy),
                axis=axis,
            )
            joints[joint.name] = joint
            child_to_joint[joint.child] = joint

        missing_names = [name for name in self.joint_names if name not in joints]
        if missing_names:
            raise ValueError(f"configured joints are absent from URDF: {missing_names}")

        reversed_chain: list[_Joint] = []
        link = self.end_effector_link
        while link != self.base_link:
            if link not in child_to_joint:
                raise ValueError(
                    f"no URDF chain from {self.base_link} to {self.end_effector_link}; stopped at {link}"
                )
            joint = child_to_joint[link]
            reversed_chain.append(joint)
            link = joint.parent
        self.chain = list(reversed(reversed_chain))
        chain_movable = [j.name for j in self.chain if j.kind in ("revolute", "continuous", "prismatic")]
        if set(chain_movable) != set(self.joint_names):
            raise ValueError(
                "configured joint_names must exactly match movable joints in the selected chain; "
                f"chain={chain_movable}, configured={self.joint_names}"
            )

    def forward(self, joints: np.ndarray) -> np.ndarray:
        values = np.asarray(joints, dtype=np.float64)
        if values.shape != (len(self.joint_names),):
            raise ValueError(f"expected {len(self.joint_names)} joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint vector contains NaN or Inf")
        if self.joint_units == "degrees":
            values = np.deg2rad(values)
        values = values * self.joint_signs + self.joint_offsets
        by_name = dict(zip(self.joint_names, values, strict=True))

        result = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            result = result @ joint.origin
            if joint.kind in ("revolute", "continuous"):
                result = result @ _axis_rotation(joint.axis, by_name[joint.name])
            elif joint.kind == "prismatic":
                motion = np.eye(4, dtype=np.float64)
                motion[:3, 3] = joint.axis / np.linalg.norm(joint.axis) * by_name[joint.name]
                result = result @ motion
            elif joint.kind != "fixed":
                raise ValueError(f"unsupported URDF joint type: {joint.kind}")
        return result @ self.tcp_transform


class DualArmKinematics:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"kinematics config does not exist: {self.config_path}")
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if config.get("robot") is None:
            raise ValueError("kinematics config must identify robot")
        self.robot = str(config["robot"])
        self.left = URDFKinematics(config["left"], self.config_path.parent)
        self.right = URDFKinematics(config["right"], self.config_path.parent)

    def forward_left(self, joints: np.ndarray) -> np.ndarray:
        return self.left.forward(joints)

    def forward_right(self, joints: np.ndarray) -> np.ndarray:
        return self.right.forward(joints)
