"""Action layouts shared by config, data loading, latent decoding, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    encoding: str
    dimension: int
    per_arm_dimension: int
    rotation_slice: slice
    gripper_index: int
    rotation_representation: str
    arm_fields: tuple[str, ...]

    def order(self, number_of_arms: int = 2) -> list[str]:
        arms = ("left", "right") if number_of_arms == 2 else ("arm",)
        return [f"{arm}_{field}" for arm in arms for field in self.arm_fields]

    def gripper_indices(self, number_of_arms: int = 2) -> tuple[int, ...]:
        return tuple(
            arm * self.per_arm_dimension + self.gripper_index
            for arm in range(number_of_arms)
        )


ACTION_SPECS = {
    "legacy_euler": ActionSpec(
        encoding="legacy_euler",
        dimension=14,
        per_arm_dimension=7,
        rotation_slice=slice(3, 6),
        gripper_index=6,
        rotation_representation="euler_xyz",
        arm_fields=("dx", "dy", "dz", "rx", "ry", "rz", "gripper"),
    ),
    "cosmos_rotation_6d": ActionSpec(
        encoding="cosmos_rotation_6d",
        dimension=20,
        per_arm_dimension=10,
        rotation_slice=slice(3, 9),
        gripper_index=9,
        rotation_representation="matrix_first_two_columns",
        arm_fields=(
            "dx", "dy", "dz",
            "rot6d_col0_x", "rot6d_col0_y", "rot6d_col0_z",
            "rot6d_col1_x", "rot6d_col1_y", "rot6d_col1_z",
            "gripper",
        ),
    ),
}


def get_action_spec(encoding: str) -> ActionSpec:
    try:
        return ACTION_SPECS[str(encoding)]
    except KeyError as error:
        raise ValueError(
            f"unsupported action encoding {encoding!r}; expected one of {sorted(ACTION_SPECS)}"
        ) from error


def action_spec_from_data(data: dict[str, Any]) -> ActionSpec:
    spec = get_action_spec(str(data.get("action_encoding", "")))
    dimension = int(data.get("action_dimension", -1))
    if dimension != spec.dimension:
        raise ValueError(
            f"{spec.encoding} requires action_dimension={spec.dimension}, got {dimension}"
        )
    return spec
