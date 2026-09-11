import os
import time
from threading import Thread, Event
from typing import Optional, Sequence, Union

import numpy as np

from xtele.common.common import SerialParams
from xtele.equipment.dynamixel.driver.dynamixel_driver import DynamixelDriver

TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 1158.73,
    "XM430_W210_T": 1000 / 2.69,
}


class DynamixelRobot:
    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Sequence[float] = None,
        joint_signs: Sequence[int] = None,
        port: str = SerialParams.PORT,
        baudrate: int = SerialParams.BAUDRATE,
        gripper_config: Sequence[int] = None,
        driver: Optional[DynamixelDriver] = None,
        dynamic_config: Optional[dict] = None,
    ):
        if gripper_config is None:
            self.gripper_open_close = None
            self.gripper_delta = None
        else:
            self.gripper_open_close = (
                gripper_config[0] * np.pi / 180,
                gripper_config[1] * np.pi / 180,
            )
            self.gripper_delta = self.gripper_open_close[1] - self.gripper_open_close[0]

        self._joint_ids = joint_ids
        self._last_joint = None
        self._alpha = 0.99

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(np.abs(self._joint_signs) == 1), (
            f"joint_signs: {self._joint_signs}"
        )

        if driver is None:
            self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
        else:
            self._driver = driver

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_dxl_position(self):
        pos = self._driver.get_position()
        return np.asarray([pos[i] for i in self._joint_ids])

    def get_dxl_velocity(self):
        vel = self._driver.get_velocity()
        return np.asarray([vel[i] for i in self._joint_ids])

    def get_frequency(self):
        return self._driver.get_frequency()

    def get_joints(self) -> np.ndarray:
        position = self.get_dxl_position()
        joints = position / 2048.0 * np.pi

        pos = (joints - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_joint is None:
            self._last_joint = pos
        else:
            # exponential smoothing
            pos = self._last_joint * (1 - self._alpha) + pos * self._alpha
            self._last_joint = pos
        return pos

    def reset_to_zero(self):
        # round pos to nearest multiple of 512
        pos = np.asarray(np.round(self.get_dxl_position() / 512) * 512, dtype=np.int32)
        self._driver.set_position(self._joint_ids, pos)
        # self.disable_torque()

    def enable_torque(self):
        self._driver.set_torque_mode(self._joint_ids, [True for _ in self._joint_ids])

    def disable_torque(self):
        self._driver.set_torque_mode(self._joint_ids, [False for _ in self._joint_ids])

    def close(self):
        self.disable_torque()

    @staticmethod
    def _radians_to_circles(radians):
        circles_per_radian = 4096 / (2 * np.pi)
        circles = radians * circles_per_radian
        return int(circles)

    def set_position(self, goal_positions: Union[Sequence[float], float]):
        if self.gripper_open_close:
            position_in_circle = self._radians_to_circles(
                self.gripper_open_close[0] + goal_positions * self.gripper_delta
            )
            self._driver.set_position(self._joint_ids, [position_in_circle])
        else:
            target_pos = np.array(goal_positions)
            current_pos = self.get_joints()
            max_delta = np.max(np.abs(target_pos - current_pos))
            step = np.clip(max_delta * 101 * 2 / np.pi, 2, 101).astype(int)

            trajectories = np.linspace(current_pos, target_pos, step, endpoint=True)[1:]
            for traj in trajectories:
                ids = self._joint_ids
                id_offset = min(ids)
                position_with_offsets_and_signs = []
                for i in range(self.num_dofs()):
                    joint_id = ids[i]
                    adjusted_position = (
                        traj[i] / self._joint_signs[joint_id - id_offset]
                        + self._joint_offsets[joint_id - id_offset]
                    )
                    position_with_offsets_and_signs.append(adjusted_position)

                position_in_circles = [
                    self._radians_to_circles(pos)
                    for pos in position_with_offsets_and_signs
                ]
                self._driver.set_position(ids, position_in_circles)

    def _torque_to_current(
        self,
        dxl_ids: Sequence[int],
        goal_torques: Sequence[float],
        signs: Sequence[float],
    ) -> np.ndarray:
        assert len(goal_torques) == len(dxl_ids), (
            "The length of torques must match the number of servos"
        )
        assert np.all(np.abs(signs)) == 1, "The signs must all be 1 or -1"

        goal_currents = [
            int(TORQUE_TO_CURRENT_MAPPING[self.servo_types[dxl_id]] * current * sign)
            for dxl_id, current, sign in zip(dxl_ids, goal_torques, signs)
        ]
        return np.clip(np.array(goal_currents), -900, 900)

    def set_torque(
        self, dxl_ids: Sequence[int], goal_torques: Sequence[float], sign: Sequence[int]
    ):
        self._driver.set_current(
            dxl_ids, self._torque_to_current(dxl_ids, goal_torques, sign)
        )

    def set_position_torque(
        self,
        dxl_ids: Sequence[int],
        goal_positions: Sequence[float],
        goal_torques: Sequence[float],
        signs: Sequence[int],
    ):
        target_pos = np.array(goal_positions)
        current_pos = self.get_joints()
        max_delta = np.max(np.abs(target_pos - current_pos))
        step = np.clip(max_delta * 101 * 2 / np.pi, 2, 101).astype(int)

        trajectories = np.linspace(current_pos, target_pos, step, endpoint=True)[1:]
        for traj in trajectories:
            ids = self._joint_ids
            id_offset = min(ids)
            position_with_offsets_and_signs = []
            for i in range(self.num_dofs()):
                joint_id = ids[i]
                adjusted_position = (
                    traj[i] / self._joint_signs[joint_id - id_offset]
                    + self._joint_offsets[joint_id - id_offset]
                )
                position_with_offsets_and_signs.append(adjusted_position)

            position_in_circles = [
                self._radians_to_circles(pos) for pos in position_with_offsets_and_signs
            ]

            self._driver.set_position_current(
                dxl_ids,
                position_in_circles,
                self._torque_to_current(dxl_ids, goal_torques, signs),
            )

    @staticmethod
    def _position_normalize(position: np.ndarray) -> np.ndarray:
        return position / 2048.0 * np.pi

    @staticmethod
    def _velocity_normalize(velocity: np.ndarray) -> np.ndarray:
        return velocity * 0.229 * 2 * np.pi / 60






