from __future__ import annotations

from typing import Dict
from loguru import logger

from rl_envs.xrocs.Rate import Rate



class URStation:

    def __init__(self, robot_dict, camera_dict, hand_dict, control_rate_hz) -> None:
        self._robot_cfg = robot_dict
        self._camera_cfg = camera_dict
        self._hand_cfg = hand_dict

        self._robot_dict = self.init_robot()
        self._camera_dict = self.instantiate_cameras(self._camera_cfg)
        self._hand_dict = self.instantiate_hands(self._hand_cfg)

        self._rate = Rate(control_rate_hz)

        logger.success("TienKung2Ros2Station Started 🚀🚀🚀")

    def init_robot(self):
        instances = {}
        
        for name, cfg in self._robot_cfg.items():
            from rl_envs.xrocs.ur import URRobot
            instance = URRobot(robot_ip=cfg["ip"])

            if instance is not None:
                instances[name] = instance

        return instances

    def instantiate_cameras(self, camera_cfgs: Dict):
        instances = {}
        
        for name, cfg in camera_cfgs.items():
            from rl_envs.xrocs.ros2_node_manager import ros2_node_manager
            from rl_envs.xrocs.orbbec_camera_ros2 import OrbbecCameraRos2
            instance = OrbbecCameraRos2(node=ros2_node_manager.acquire(), camera_config=cfg)

            if instance is not None:
                instances[name] = instance

        return instances
    
    def instantiate_hands(self, hand_cfgs: Dict):
        instances = {}
        
        for name, cfg in hand_cfgs.items():
            from rl_envs.xrocs.robotiq_gripper import RobotiqGripper
            instance = RobotiqGripper(robot_ip=cfg["ip"])

            if instance is not None:
                instances[name] = instance

        return instances
    
    def connect(self):
        for arm in self._robot_dict.values():
            arm.connect()
        for hand in self._hand_dict.values():
            hand.connect()
        
        obs = self.get_obs()
        assert obs["arm_joints"] != {}, "Robot connection failed: arm_joints is empty"
        assert obs["images"] != {}, "Camera connection failed: images is empty"
        assert obs["hand_joints"] != {}, "Hand connection failed: hand_joints is empty"

    def get_robot_handle(self):
        return self._robot_dict

    def get_camera_handle(self):
        return self._camera_dict

    def get_gripper_handle(self):
        return self._hand_dict
    
    def execute_action(self, action: dict[str, dict]) -> None:
        if "arm" in action:
            if 'position' in action["arm"]:
                arm_mode = "position"
            elif 'pose' in action["arm"]:
                arm_mode = "pose"
            elif 'hybrid' in action["arm"]:
                arm_mode = "hybrid"
            for name, joints in action["arm"][arm_mode].items():
                if name in self._robot_dict:
                    self._robot_dict[name].sync_target_joint(joints)

        if "hand" in action:
            for name, joints in action["hand"]["position"].items():
                if name in self._hand_dict:
                    self._hand_dict[name].sync_target_joint(joints)
        self._rate.sleep()

    def step(self, action: dict[str, dict]):
        self.execute_action(action)
        return self.get_obs()

    def step_ee(self, robot_targets: dict[str, dict]):
        if "arm_pose" in robot_targets:
            for name, pose in robot_targets["arm_pose"].items():
                self._robot_dict[name].sync_tool_cartesian_pose(pose)
        if "hand_joints" in robot_targets:
            for name, joints in robot_targets["hand_joints"].items():
                self._hand_dict[name].sync_target_joint(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_robot_state(self):
        observations = {
            "arm_joints": {
                name: arm.get_current_joint() for name, arm in self._robot_dict.items()
            },
            "arm_pose": {
                name: arm.get_tool_cartesian_pose() for name, arm in self._robot_dict.items()
            },
            "hand_joints": {
                name: hand.get_current_joint() for name, hand in self._hand_dict.items()
            },
        }
        return observations

    def get_camera_state(self):
        images = {}
        depths = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            images[name] = image
            depths[name] = depth
        return images, depths

    def get_obs(self):
        """Get complete observations including robot state, camera images, and depths.

        Returns:
            Dictionary containing robot state, images, and depth data
        """
        images, depths = self.get_camera_state()
        observations = self.get_robot_state()
        observations["images"] = images
        observations["depths"] = depths
        return observations
    
    def get_ee_pose_from_joint(self, joint: np.ndarray):
        for name, robot in self._robot_dict.items():
            return robot.get_ee_pose_from_joint(joint)
        return None
