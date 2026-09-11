"""Example script demonstrating the usage of UR station driver.
This script shows how to:
1. Initialize and connect to UR station devices
2. Get robot state (arm, hand)
3. Execute robot actions (arm, hand)
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path to allow imports
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import time
import numpy as np
from rl_envs.xrocs.core.config_loader import ConfigLoader
from rl_envs.xrocs.core.station_loader import StationLoader

from rl_envs.xrocs.ur_station import URStation

def example_get_obs():
    # 读取状态接口
    # Read state interface
    '''
    obs = {
        "arm_joints": {
            "left": np.ndarray,      # dofs=7 radian
            "right": np.ndarray      # dofs=7,), radian
        },
        "arm_pose": {                # 同上，正解得到 TCP 位姿
                                    # Same as above; forward kinematics gives TCP pose
            "left": np.ndarray,      # dofs=7, xyz(m)+xyzw
            "right": np.ndarray      # dofs=7, xyz(m)+xyzw
        },
        "hand_joints": {             # 手部关节(夹爪为1dofs,灵巧手为6dofs)
                                    # Hand joints: gripper 1 DoF, dexterous hand 6 DoFs
            "left": np.ndarray,
            "right": np.ndarray
        },
        "images":{
            "wrist": np.ndarray,
            "top": np.ndarray
        },
        "depths":{
            "wrist": np.ndarray,
            "top": np.ndarray
        },
    } 
    '''   
    obs = robot_station.get_obs()
    print(obs.keys())
    print(f"obs:{obs}")
    return obs

def example_execute_action():
    # 控制接口
    # Control interface
    action = {}

    # 手臂控制：station 预期 {"arm": {"position": {"left": cmd, "right": cmd}}}
    # Arm control: station expects {"arm": {"position": {"left": cmd, "right": cmd}}}
    cur_sg_joint = robot_station.get_robot_state()["arm_joints"]["single"]
    sg_arm_cmd = cur_sg_joint + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    action["arm"] = {"position": {"single": sg_arm_cmd}}

    # 手部控制：station 预期 {"hand": {"position": {"left": cmd, "right": cmd}}}
    # Hand control: station expects {"hand": {"position": {"left": cmd, "right": cmd}}}
    hand_handles = robot_station.get_gripper_handle()
    if hand_handles:
        hand_cmd = {
            "single": np.array([0.0])
        }
        action["hand"] = {
            "position": hand_cmd
            }


    # 执行控制
    # Execute control
    if action:
        robot_station.execute_action(action)



if __name__ == "__main__":
    # 从配置文件加载配置
    # Load config from configuration file
    # Use script directory to find config file regardless of where script is run from
    config_path = script_dir / "xrocs" / "configuration.toml"
    config_path = str(config_path)
    cfg_loader = ConfigLoader(config_path)
    cfg_dict = cfg_loader.get_config()
    station_loader = StationLoader(cfg_dict)
    robot_station = station_loader.generate_station_handle()
    print('robot_station.connect')
    input("press Enter to continue...")
    robot_station.connect()




    time.sleep(1)
    robot = robot_station.get_robot_handle()["single"]

    obs = example_get_obs()

    # ==================== test the function get_ee_pose_from_joint ====================
    compute_ee_pose = robot.get_ee_pose_from_joint(obs['arm_joints']['single'])
    print(f"observed ee_pose {obs['arm_pose']['single']}")
    print(f"compute ee_pose:{compute_ee_pose}")
    
    # =========================== Go home ===========================
    input("Press to go home...")
    robot.reach_target_joint([3.82, -1.56, -1.56, -1.56, 1.56, 3.14])

    # =========================== test function step ===========================
    obs = example_get_obs()
    action = {}
    cur_sg_joint = obs["arm_joints"]["single"]
    sg_arm_cmd = cur_sg_joint + np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"cur_sg_joint:{cur_sg_joint}")
    print(f"sg_arm_cmd:{sg_arm_cmd}")
    action["arm"] = {"position": {"single": sg_arm_cmd}}
    action["hand"] = {
        "position": {
            "single": np.array([1.0])
        }
    }
    # 执行控制
    # Execute control
    input("Press to step...")
    obs = robot_station.step(action)

    # =========================== test function step_ee ===========================
    obs = robot_station.get_obs()
    cur_ee_pose = robot.get_tool_cartesian_pose_xyzrpy()
    tar_ee_pose = np.asarray(cur_ee_pose) + np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0])
    print(f"cur_ee_pose:{cur_ee_pose}")
    print(f"tar_ee_pose:{tar_ee_pose}")
    action = {}
    action["arm_pose"] = {
        "single": tar_ee_pose
    }
    action["hand_joints"] = {
        "single": np.array([0.0])
    }
    input("Press to step ee...")
    robot_station.step_ee(action)


    

    