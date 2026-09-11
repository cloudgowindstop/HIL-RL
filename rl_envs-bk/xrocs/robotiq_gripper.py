from __future__ import annotations

import time

import numpy as np

from rl_envs.xrocs.rg_driver_opt import RobotiqGripperDriver

class RobotiqGripper:

    def __init__(self, robot_ip: str):
        self.gripper = RobotiqGripperDriver()
        self._robot_ip = robot_ip

    def num_dofs(self) -> int:
        return 1

    def connect(self) -> bool:
        self.gripper.connect(hostname=self._robot_ip, port=63352)
        return True

    def open(self) -> bool:
        self.sync_target_joint(0)
        return True

    def close(self) -> bool:
        self.sync_target_joint(1)
        return True

    def get_current_joint(self):
        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        position = gripper_pos / 255
        return np.array([position])

    def set_target_joint(self, target_joint: np.ndarray) -> None:
        self.sync_target_joint(target_joint)

    def sync_target_joint(self, target_joint: np.ndarray, force=10, speed=255) -> None:
        target_joint = float(target_joint)
        assert 0.0 <= target_joint <= 1.0, "Gripper control parameter must be between 0 and 1"
        gripper_pos = target_joint * 255
        self.gripper.move(int(gripper_pos), speed, force)
