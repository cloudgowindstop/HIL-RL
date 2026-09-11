"""Gym Interface for Franka and UR"""
import os
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import gymnasium as gym
import zmq
import pickle
import cv2
import copy     
from scipy.spatial.transform import Rotation
import time
from typing import Dict
from pathlib import Path


from rl_envs.xrocs.core.config_loader import ConfigLoader
from rl_envs.xrocs.core.station_loader import StationLoader



##############################################################################
import traceback
import sys
from rl_envs.shared_state import shared_state

def decoder_image(camera_rgb_images, camera_depth_images, bgr2rgb=False):
    if type(camera_rgb_images[0]) is np.uint8:
        rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
        if bgr2rgb and rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if camera_depth_images is not None:
            depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        else:
            depth = np.asarray([])
        return rgb, depth
    else:
        rgb_images = []
        depth_images = []
        for idx, camera_rgb_image in enumerate(camera_rgb_images):
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            
            if bgr2rgb and rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb)
            depth_images.append(depth)
        rgb_images = np.asarray(rgb_images)
        depth_images = np.asarray(depth_images)
        return rgb_images, depth_images
    
def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class BaseEnv(gym.Env):
    def __init__(
        self,
        fake_env=False,
        config=None,
    ):
        self.config = config
        self._gripper_sleep = config.gripper_sleep
        self.joint_dim = config.joint_dim
        self._reset_joint = np.array(config.reset_joint)[0:self.joint_dim]
        # self._reset_pose = np.array(config.target_pose)
        self._random_xy_range = config.random_xy_range
        self._random_rz_range = config.random_rz_range
        self._random_reset = config.random_reset
        self._bgr2rgb = config.bgr2rgb
        self._image_keys = config.image_keys
        self.robot_type = config.robot_type
        self.action_scale = config.action_scale
        self.max_episode_length = config.max_episode_length
        self.control_mode = config.control_mode
        self.close_gripper = config.close_gripper
        self.fix_gripper = config.fix_gripper
        self.ego_mode = config.ego_mode
        assert self.control_mode in ["joint", "pose"], f'Not valid control mode: {self.control_mode}'
        
        self.fake_env = fake_env
        self.hz = config.hz

        image_resize = config.image_resize

        state_dict = {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "gripper_pose": gym.spaces.Box(0, 1, shape=(1,)),
                        "joints": gym.spaces.Box(
                            -np.inf, np.inf, shape=(self.joint_dim,)
                        ),
                        "ee_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(6,)
                        ),
                        "arm_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                    }

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=image_resize[key], dtype=np.uint8) 
                        for key in config.image_keys}
                ),
            }
        )

        
        if self.control_mode == "pose":
            # Action/Observation Space: action_space = (xyz + rpy + gripper)
            self.action_space = gym.spaces.Box(
                np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            )
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.joint_dim + 1,))


        if fake_env:
            print_green("fake env : not connect to robot")
            return 

        # 使用基于文件位置的路径，避免依赖工作目录
        # Use file-based path to avoid dependency on working directory
        base_dir = Path(__file__).parent.absolute()
        config_path = base_dir / "xrocs" / "configuration.toml"
        cfg_loader = ConfigLoader(str(config_path))
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
    
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        self._update_currpos()
        self.last_gripper_act = time.time()

        self.xyz_bounding_box = gym.spaces.Box(
            np.array(config.abs_pose_limit_low[:3]),
            np.array(config.abs_pose_limit_high[:3]),
            dtype=np.float64,
        )
        
        
        # self.image_crop = {}

        # if hasattr(config, 'image_crop') and config.image_crop is not None:
        #     for camera_key, crop_func in config.image_crop.items():
        #         if callable(crop_func):
        #             self.image_crop[camera_key] = crop_func
        #         else:
        #             def make_crop_func(crop_params):
        #                 crop_params = list(crop_params)
        #                 if isinstance(crop_params, (list, tuple)) and len(crop_params) == 2:
        #                     h_range, w_range = crop_params
        #                     return lambda img: img[h_range[0]:h_range[1], w_range[0]:w_range[1]]
        #                 else:
        #                     return lambda img: img
                    
        #             self.image_crop[camera_key] = make_crop_func(crop_func)
        
        self.save_path = None
        self.save_frame = False
        self.obs_pre = None
        self.last_gripper_value = 1.0 if self.close_gripper else 0.0



    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        return pose

    def pose_quat2euler(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_euler = Rotation.from_quat(pose_quat).as_euler("xyz")
        pose = np.hstack([pose_t, pose_euler])
        return pose
    

    def get_xtele(self) -> dict:
        if 'ur' in self.robot_type:
            self.tele_agent.switch_act()
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv 
        elif 'franka' in self.robot_type:
            self.tele_agent.exit_any_sync()
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv
        else:
            raise NotImplementedError("Unknown robot type")

    def init_xtele(self,):
        if 'ur' in self.robot_type or 'franka' in self.robot_type:
            from xtele.core.integrate_module import TeleCore
            self.tele_agent = TeleCore()
        else:   
            raise NotImplementedError("Unknown robot type")

    def sync_xtele(self, timeout: float = 5):
        goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)

        if 'ur' in self.robot_type:
            self.tele_agent.switch_reverse()
            self.tele_agent.sync_position(goal)
        elif 'franka' in self.robot_type:
            need_torque = False
            tele_cur_joints = self.tele_agent.act()
            tele_tar_joints = goal
            timeout = int(timeout // 0.02)
            if timeout <= 1:
                path = np.array([tele_tar_joints])
            else:
                path = np.linspace(tele_cur_joints, tele_tar_joints, timeout)
            for p in path:
                self.tele_agent.sync_position_torque(p)
                time.sleep(0.02)

        else:
            raise NotImplementedError("Unknown robot type")    

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        
        start_time = time.time()
        if (time.time() - self.last_gripper_act > self._gripper_sleep) and not self.fix_gripper:
            include_gripper = True
        else:
            include_gripper = False

        if self.control_mode == "joint":
            obs = self._send_joint_command(action, include_gripper) 
            curr_pose_euler = None
        elif self.control_mode == "pose":
            action = action.clip(-1, 1)
            
            action_t = action[0:3] * self.action_scale[0]
            action_euler = action[3:6] * self.action_scale[1]

            
            # Action transformation matrix
            action_mat = Rotation.from_euler("xyz", action_euler).as_matrix()
            action_pose = np.eye(4)
            action_pose[:3, :3] = action_mat
            action_pose[:3, 3] = action_t

            # 当前位置的变换矩阵
            currpos_pose = np.eye(4)
            currpos_mat = Rotation.from_quat(self.currpos[3:]).as_matrix()
            currpos_pose[:3, :3] = currpos_mat
            currpos_pose[:3, 3] = self.currpos[:3]
            
            # Calculate the new target pose (current pose × action transformation)
            tar_pose_new = currpos_pose @ action_pose
            # 计算新的目标位姿的欧拉角
            tar_euler_new = Rotation.from_matrix(tar_pose_new[:3, :3]).as_euler("xyz")
            
            # Calculate the current pose's euler angle
            cur_euler = self.pose_quat2euler(self.currpos)  

            # Calculate the new target pose (position + euler angle + gripper)
            next_pos = np.hstack([tar_pose_new[:3,3], tar_euler_new, action[-1] * self.action_scale[2]])
            next_pos = self.clip_safety_box(next_pos)
            obs = self._send_pos_command(next_pos, include_gripper) 
            curr_pose_euler = self.pose_quat2euler(obs['arm_pose']['single'])
        else:
            raise NotImplementedError(f"Not valid control mode: {self.control_mode}")

        if include_gripper:
            self.last_gripper_act = time.time()


        self.curr_path_length += 1

        end_time = time.time()
        sleep_time = max(0, 1/self.hz - (end_time - start_time))
        time.sleep(sleep_time)
        obs = self._get_obs()
        
        reward = 0.0
        terminated = False
        truncated = self.curr_path_length >= self.max_episode_length
        return obs, int(reward), terminated, truncated, {"succeed": terminated, "curr_pose_euler": curr_pose_euler, "is_intervention": False}


    def reset(self, **kwargs):
        self.last_gripper_act = time.time()
        if self.ego_mode:
            
            # provide intervention and reset from the only one person in the scene
            self.last_gripper_value = 1.0 if self.close_gripper else 0.0
            self.sync_xtele()
            while True:
                try:
                    input("Press Enter to continue...")
                    break  # Break the loop if input is successful
                except (EOFError, ValueError):
                    print("Input is temporarily unavailable, retrying...")
                    traceback.print_exc()
                    time.sleep(1)
                    continue  
            shared_state.terminate = False
            print("Reset the scene, press Space to continue...")
            while not shared_state.terminate:
                obs = self.get_xtele()
                xtele_joints = obs['joints']
                self._update_currpos()
                target_joint = xtele_joints.copy()
                self._send_joint_command(target_joint, include_gripper=True)
                time.sleep(1 / self.hz)
            shared_state.terminate = False
            print('go to reset!!!!!!!!!!')
            self.go_to_reset(joint_reset=True)    
        else:
            self.go_to_reset(joint_reset=True)      
            shared_state.terminate = False
            # print("重新摆放场景, 按空格继续: ")
            print("Reset the scene, press Space to continue...")
            while not shared_state.terminate:
                continue
            shared_state.terminate = False

        self.curr_path_length = 0
        self.last_gripper_act = time.time()
        self.last_gripper_value = 1.0 if self.close_gripper else 0.0
        obs = self._get_obs(obs=None)
        return obs, {"success": False, "is_intervention": False}


    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """        
        # perform joint reset if needed
        self._update_currpos()
        curr_pose = self.currpos.copy()
        curr_pose = self.pose_quat2euler(curr_pose)
        # reset_pose = self._reset_pose.copy()

        # if np.linalg.norm(curr_pose - reset_pose) > 0.15 or joint_reset:
        assert self._reset_joint.shape == (self.joint_dim,)
        # if "ur" in self.robot_type:
        #     arm_joints = np.append(self.curr_arm_joints, self.last_gripper_value)
        #     for _ in range(5):
        #         self._send_joint_command(arm_joints, include_gripper=False)
        #         time.sleep(1 / self.hz)
                
        #     for name, _robot in self.robot_station.get_robot_handle().items():
        #         goal_joints = Joints(self._reset_joint, num_of_dofs=self.joint_dim)
        #         try:
        #             return_val =_robot.reach_target_joint(goal_joints)
        #         except Exception as e:
        #             print(f"Error in reach_target_joint: {e}")
        #     for _ in range(5):
        #         self._send_joint_command(self._reset_joint, include_gripper=False)
        #         time.sleep(1 / self.hz)
        # else:
        goal_joints = self._reset_joint.copy()
        # Combine the current joints and gripper position 
        curr_joints = np.concatenate([self.curr_arm_joints, np.array([self.curr_gripper_joints])])
        # Combine the target joints and gripper position
        goal_joints = np.concatenate([goal_joints, np.array([self.last_gripper_value])])
        cnt = int(3 / (1 / self.hz))
        # Generate a linear interpolation path from the current joints to the target joints (smooth transition)
        path = np.linspace(curr_joints, goal_joints, cnt)
        for p in path:
            self._send_joint_command(p, include_gripper=False)
            time.sleep(1 / self.hz)

        self._update_currpos()
        curr_pose = self.currpos.copy()
        curr_pose = self.pose_quat2euler(curr_pose)
        
        # If random reset is enabled, add random perturbations to the xy plane and rotation angle
        if self._random_reset:  
            reset_pose = curr_pose.copy()
            # Add random offset to the xy plane
            reset_pose[:2] += np.random.uniform(
                -self._random_xy_range, self._random_xy_range, (2,)
            )
            # 获取旋转角
            axis_random = np.array(reset_pose[3:])
            assert axis_random.shape == (3,)
            # 在Z轴旋转角上添加随机扰动
            axis_random[-1] += np.random.uniform(
                -self._random_rz_range, self._random_rz_range
            )
            reset_pose[3:] = axis_random
            cnt = int(1 / (1 / self.hz))
            path = np.linspace(curr_pose, reset_pose, cnt)
            for p in path:
                self._send_pos_command(p, include_gripper=False)
                time.sleep(1 / self.hz)


    def _send_joint_command(self, joints: np.ndarray, include_gripper=False):
        gripper_value = joints[-1] if include_gripper else self.last_gripper_value
        gripper_value_binary = 1.0 if gripper_value >= 0.5 else 0.0
        if include_gripper:
            self.last_gripper_value = gripper_value_binary
        if "ur" in self.robot_type:
            robot_target = {
                "arm_joints": {
                    "single": joints[0:self.joint_dim]
                },
                "hand_joints": {"single": self.last_gripper_value}
            }
            obs = self.robot_station.step(robot_target)
            return obs
        elif "franka" in self.robot_type:
            robot_target = {
                "arm": {
                    "position": {
                        "single": np.append(joints[0:self.joint_dim], self.last_gripper_value),
                    }
                }
            }
            obs = self.robot_station.step(robot_target)
            return obs
        else:
            raise NotImplementedError("Unknown robot type")


    def _send_pos_command(self, pose: np.ndarray, include_gripper=False):
        if include_gripper:
            gripper_value_binary = 1.0 if pose[-1] >= 0.5 else 0.0
            self.last_gripper_value = gripper_value_binary

        if "ur" in self.robot_type:
            robot_target = {
                "arm_pose": {
                    "single": pose[0:6]
                },
                "hand_joints": {"single": self.last_gripper_value}
            }
            obs = self.robot_station.step_ee(robot_target)
        elif "franka" in self.robot_type:
            arm_pose = np.append(pose[0:6], self.last_gripper_value)
            robot_target = {
                "arm_pose": {
                    "single": arm_pose
                },
                "hand_joints": {}
            }
        else:
            raise NotImplementedError("Unknown robot type")
        return obs


    def _update_currpos(self, obs=None):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        if obs is None:
            obs = self._get_obs_from_robot()
        self.currpos = obs["arm_pose"]['single']
        self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
        self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
        return obs

    def _get_obs(self, obs=None) -> dict:
        if self.fake_env:
            images = self.get_fake_im()
            state_observation = self.get_fake_pose()
        else:
            if obs is None:
                obs = self._get_obs_from_robot()
            self.currpos = obs["arm_pose"]['single']
            self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
            self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
            images = {}
            for key, cap in obs["images"].items():
                if key not in self._image_keys:
                    continue
                rgb, _ = decoder_image(cap, None, bgr2rgb=self._bgr2rgb)
                images[key] = rgb 
                if not os.path.exists(f"online_image_{key}.png"):
                    cv2.imwrite(f"online_image_{key}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            state_observation = {
                "tcp_pose": self.currpos,
                "gripper_pose": self.curr_gripper_joints,
                "joints": self.curr_arm_joints,
            }

            
        return dict(images=images, state=state_observation)

    def _get_obs_from_robot(self) -> dict:
        obs = self.robot_station.get_obs()
        # standardize arm_pose
        arm_pose = obs['arm_pose']['single']
        arm_pose_t, arm_pose_quat = arm_pose[0:3], arm_pose[3:]
        arm_pose_quat = Rotation.from_quat(arm_pose_quat).as_quat(canonical=True)
        arm_pose = np.hstack([arm_pose_t, arm_pose_quat])
        obs['arm_pose'] = {'single': arm_pose}
        return obs

    def close(self):
        return

