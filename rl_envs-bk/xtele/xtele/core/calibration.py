#!/usr/bin/env python3
"""
Creator: Eric Xu
Developer
    - Jacob Ji
    - Shane Xie
First create: 2024-07-15
Last  modify: 2025-07-08

Version History:
v1.6.0 - Support for teleoperation product.

"""

import numpy as np

from xtele.core.config_manager import ConfigManager
from xtele.equipment.dynamixel.driver.dynamixel_driver import DynamixelDriver


class Calibration:
    def __init__(self):
        self.config_manager = ConfigManager()

    def calibrate(self):
        config = self.config_manager.config
        input("标定即将开始, 请将同构臂保持在Home位并按下[Enter]键:")

        for linker_type in config["xlinker"].keys():
            print(f"{linker_type} Linker:")
            for name in config["xlinker"][linker_type].keys():
                print(f"=> Linker for {name}:")
                if not config["home_pose"]:
                    raise ValueError("home_pose is not set!")
                home_pose = config["home_pose"][name]
                robot_config = config["xlinker"][linker_type][name]
                self.get_offset(robot_config, home_pose, name)

        print("标定完成，等待标定文件写入...")
        self.config_manager.write_config()
        print("标定文件写入完成！")

    @staticmethod
    def get_offset(robot_config, home_pose, name) -> None:
        driver = DynamixelDriver(
            robot_config["joint_ids"], robot_config["port"], baudrate=2000000
        )
        home_pose = np.array(home_pose)
        joint_signs = np.array(robot_config["joint_signs"])
        ids = robot_config["joint_ids"]

        def get_joints() -> np.ndarray:
            pos = driver.get_position()
            pos = np.asarray([pos[i] for i in ids])
            return np.array(pos / 2048.0 * np.pi)

        for _ in range(10):
            get_joints()  # warmup

        curr_joints = get_joints()
        if "gripper" in robot_config.keys():
            joint_offsets = curr_joints[:-1] - home_pose / joint_signs
            print(f"joint offsets: {np.around(joint_offsets, 3).tolist()}")
            while True:
                input(f"请将 {name} 夹爪保持张开状态并按下[Enter]键：")
                open_degrees = np.rad2deg(get_joints()[-1])
                input(f"请将 {name} 夹爪保持闭合状态并按下[Enter]键：")
                close_degrees = np.rad2deg(get_joints()[-1])
                print("gripper open (degrees)   ", open_degrees)
                print("gripper close (degrees)  ", close_degrees)
                if np.abs(close_degrees - open_degrees) > 10:
                    robot_config["gripper"] = [open_degrees, close_degrees]
                    break
                print(
                    "\033[33m夹爪闭合状态下标定结果与张开状态相近，即将重新进行夹爪标定...\033[0m"
                )
        else:
            joint_offsets = curr_joints - home_pose / joint_signs
            print(f"joint offsets: {np.around(joint_offsets, 3).tolist()}")
        robot_config["joint_offsets"] = joint_offsets.tolist()

        driver.close()


if __name__ == "__main__":
    calibration = Calibration()
    print(calibration.config_manager.config)
    calibration.calibrate()
