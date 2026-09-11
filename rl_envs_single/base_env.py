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
import h5py
from hydra.core.hydra_config import HydraConfig





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


_FAKE_CAMERA_KEY_MAP = {
    "right": "camera_right",
    "wrist": "camera_wrist",
    "front": "camera_front",
    "left": "camera_left",
    "top": "camera_top",
}


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
        print(f"self._random_xy_range: {self._random_xy_range}")
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
        self.enable_rotation = config.enable_rotation
        assert self.control_mode in ["joint", "pose"], f'Not valid control mode: {self.control_mode}'
        
        self.fake_env = fake_env
        self.hz = config.hz

        image_resize = config.image_resize
        print_green(f'in BaseEnv image_resize: {image_resize}')

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
            if "tien" in self.robot_type:
                self.action_space = gym.spaces.Box( 
                    np.array([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            else:
                # Action/Observation Space: action_space = (xyz + rpy + gripper)
                self.action_space = gym.spaces.Box(
                    np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            self._apply_mask_to_action_space()

        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.joint_dim + 1,))


        if fake_env:
            print_green("fake env : not connect to robot")
            self._fake_episode_path = "/media/HIL-RL-Project/dataset/data/trajectory.hdf5"
            self._load_fake_episode()
            self._fake_step_idx = 0
            self._apply_fake_step_state(0)
            self.xyz_bounding_box = gym.spaces.Box(
                np.array(config.abs_pose_limit_low[:3]),
                np.array(config.abs_pose_limit_high[:3]),
                dtype=np.float64,
            )
            self.rpy_bounding_box = gym.spaces.Box(
                np.array(config.abs_pose_limit_low[3:]),
                np.array(config.abs_pose_limit_high[3:]),
                dtype=np.float64,
            )
            print("after init BaseEnv (fake_env)")
            return

        from xrocs.core.config_loader import ConfigLoader
        from xrocs.core.station_loader import StationLoader
        from xrocs.common.data_type import Joints
        cfg_loader = ConfigLoader("/home/eai/Documents/configuration.toml")
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()

        try:
            self.robot_station.connect()
        except Exception as e:
            print(f"[{type(e).__name__}] {e!r}")
            traceback.print_exc()          # full stacktrace
            sys.exit(1)
        
        self._update_currpos()
        self.last_gripper_act = time.time()

        self.xyz_bounding_box = gym.spaces.Box(
            np.array(config.abs_pose_limit_low[:3]),
            np.array(config.abs_pose_limit_high[:3]),
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array(config.abs_pose_limit_low[3:]),
            np.array(config.abs_pose_limit_high[3:]),
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
        print('after init BaseEnv')



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
            self.tele_agent.exit_any_sync(0)
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
        if self.fake_env:
            print("action:", action)
            return self._fake_step(action)

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
            if self.enable_rotation:
                    action_euler = action[3:6] * self.action_scale[1]
            else:
                action_euler = np.array([0, 0, 0])

            
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
            # print("cur_euler:", cur_euler)      

            # Calculate the new target pose (position + euler angle + gripper)
            next_pos = np.hstack([tar_pose_new[:3,3], tar_euler_new, action[-1] * self.action_scale[2]])
            next_pos = self.clip_safety_box(next_pos)
            if not self.enable_rotation:
                if "franka1" or "franka2" in self.robot_type:
                    next_pos[3:6] = [3.14, 0, 0]
                else:
                    raise NotImplementedError(f"Robot {self.robot_type} does not support disable_rotation mode")
            obs = self._send_pos_command(next_pos, include_gripper) 
            # print("close _send_pos_command for debugging")
            curr_pose_euler = self.pose_quat2euler(obs['arm_pose']['single'])
        else: 
            raise NotImplementedError(f"Not valid control mode: {self.control_mode}")

        if include_gripper:
            self.last_gripper_act = time.time()


        self.curr_path_length += 1

        end_time = time.time()
        sleep_time = max(0, 1/self.hz - (end_time - start_time))
        # print("===============================================base step sleep_time:", sleep_time)
        time.sleep(sleep_time)
        # print("===============================================base step get_obs time:", time.time() - start_time)
        obs = self._get_obs()
        
        reward = 0.0
        terminated = False
        truncated = self.curr_path_length >= self.max_episode_length
        return obs, int(reward), terminated, truncated, {"succeed": terminated, "curr_pose_euler": curr_pose_euler, "is_intervention": False}


    def reset(self, **kwargs):
        self.last_gripper_act = time.time()
        self.last_gripper_value = 1.0 if self.close_gripper else 0.0

        if self.fake_env:
            self.curr_path_length = 0
            self._fake_step_idx = 0
            self._apply_fake_step_state(0)
            shared_state.terminate = False
            shared_state.emergency_terminate = False
            obs = self._get_obs()
            print("curr_action:", self.curr_action)
            return obs, {"success": False, "is_intervention": False}

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
                    # time.sleep(1)
                    continue  
            shared_state.terminate = False
            # 每次reset前shared_state.emergency_terminate设置为False
            shared_state.emergency_terminate = False
            print("Reset the scene, press Pause to continue...")
            # print("Reset the scene, press Space to continue...")
            while not shared_state.terminate:
                obs = self.get_xtele()
                xtele_joints = obs['joints']
                self._update_currpos()
                target_joint = xtele_joints.copy()
                print("franka target_joint:", self.curr_arm_joints)
                print("xtele target_joint:", target_joint)
                # if self.fix_gripper:
                #     self._send_joint_command(target_joint, include_gripper=False)
                # else:
                self._send_joint_command(target_joint, include_gripper=True)
                time.sleep(1 / self.hz)
            shared_state.terminate = False
            print('go to reset!!!!!!!!!!')
            self.go_to_reset(joint_reset=True)    
        else:
            self.go_to_reset(joint_reset=True)      
            shared_state.terminate = False
            # 每次reset前shared_state.emergency_terminate设置为False
            shared_state.emergency_terminate = False
            # print("重新摆放场景, 按空格继续: ")
            print("Reset the scene, press Pause to continue...")
            # print("Reset the scene, press Space to continue...")
            while not shared_state.terminate:
                continue
            shared_state.terminate = False

        self.curr_path_length = 0
        self.last_gripper_act = time.time()
        
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
        if "ur" in self.robot_type:
            arm_joints = np.append(self.curr_arm_joints, self.last_gripper_value)
            for _ in range(5):
                self._send_joint_command(arm_joints, include_gripper=False)
                time.sleep(1 / self.hz)
                
            for name, _robot in self.robot_station.get_robot_handle().items():
                goal_joints = Joints(self._reset_joint, num_of_dofs=self.joint_dim)
                try:
                    return_val =_robot.reach_target_joint(goal_joints)
                except Exception as e:
                    print(f"Error in reach_target_joint: {e}")
            for _ in range(5):
                self._send_joint_command(self._reset_joint, include_gripper=False)
                time.sleep(1 / self.hz)
        else:
            goal_joints = self._reset_joint.copy()
            # Combine the current joints and gripper position 
            curr_joints = np.concatenate([self.curr_arm_joints, np.array([self.curr_gripper_joints])])
            # Combine the target joints and gripper position
            goal_joints = np.concatenate([goal_joints, np.array([self.last_gripper_value])])


            cnt = int(3 / (1 / self.hz))
            # Generate a linear interpolation path from the current joints to the target joints (smooth transition)
            path = np.linspace(curr_joints, goal_joints, cnt)
            print(f"self.last_gripper_value: {self.last_gripper_value}")
            if self.close_gripper:
                for p in path:
                    self._send_joint_command(p, include_gripper=True)
            else:
                for p in path:
                    self._send_joint_command(p, include_gripper=False)
                # time.sleep(1 / self.hz)

        self._update_currpos()
        reset_pose = self.currpos.copy()
        reset_pose = self.pose_quat2euler(reset_pose)
        
        # If random reset is enabled, add random perturbations to the xy plane and rotation angle
        if self._random_reset:  
            # Add random offset to the xy plane
            print(f"change reset for conrft regime: only randomly change the yz plane")
            reset_pose[1:3] += np.random.uniform(
                -self._random_xy_range, self._random_xy_range, (2,)
            )
            # # 获取旋转角
            # axis_random = np.array(reset_pose[3:])
            # assert axis_random.shape == (3,)
            # # 在Z轴旋转角上添加随机扰动
            # axis_random[-1] += np.random.uniform(
            #     -self._random_rz_range, self._random_rz_range
            # )
            # reset_pose[3:] = axis_random
            self._send_pos_command(reset_pose, include_gripper=False)


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
            # print(f"self.last_gripper_value: {self.last_gripper_value}")
            # print("joints:", joints)
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
        else:
            # print(f"self.last_gripper_value: {self.last_gripper_value}")
            arm_pose = np.append(pose[0:6], self.last_gripper_value)
            robot_target = {
                "arm_pose": {
                    "single": arm_pose
                },
                "hand_joints": {}
            }
            if "franka" in self.robot_type:
                currpose = self.currpos.copy()
                currpose_t, currpose_quat = currpose[0:3], currpose[3:]
                currpose_quat = Rotation.from_quat(currpose_quat).as_euler("xyz")
                currpose = np.hstack([currpose_t, currpose_quat])
                obs = self.robot_station.step_ee(robot_target)
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
            step_idx = self._fake_step_idx
            self._apply_fake_step_state(step_idx)
            if step_idx > self._fake_actions.shape[0] - 1:
                step_idx = self._fake_actions.shape[0] - 1
            self.curr_action = self._fake_actions[step_idx].astype(np.float64).copy()
            
            # images = {
            #     key: self._decode_fake_image(step_idx, key) for key in self._image_keys
            # }
            # state_observation = {
            #     "tcp_pose": self.currpos,
            #     "gripper_pose": np.atleast_1d(self.curr_gripper_joints),
            #     "joints": self.curr_arm_joints,
            # }

            image_path = "/media/HIL-RL-Project/HIL-RL/experiments/pick_toy/origin_obs_dumps/episode_0000/step_0019"
            # Match _decode_fake_image / decoder_image: cv2 loads BGR, optionally convert to RGB.
            images = {}
            for key in self._image_keys:
                bgr = cv2.imread(os.path.join(image_path, f"obs_origin_obs_{key}.png"))
                if bgr is None:
                    raise RuntimeError(
                        f"Failed to load fake image '{key}' from {image_path}"
                    )
                if self._bgr2rgb:
                    images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    images[key] = bgr
            txt_path = os.path.join(image_path, "step_0019.txt")
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            proprio = None
            for line in content.splitlines():
                if line.startswith("proprio="):
                    vals = line.split("=", 1)[1].strip().strip("[]")
                    proprio = np.array(
                        [float(x) for x in vals.split(",") if x.strip()],
                        dtype=np.float64,
                    )
                    break
            if proprio is None:
                raise ValueError(f"proprio not found in {txt_path}")
            # dump proprio = xyz(3) + quat(4) + gripper(1)
            state_observation = {
                "tcp_pose": proprio[:7],
                "gripper_pose": np.atleast_1d(proprio[7]),
                "joints": np.zeros(self.joint_dim),
            }
        else:
            if obs is None:
                obs = self._get_obs_from_robot()
            self.currpos = obs["arm_pose"]['single']
            self.curr_gripper_joints = np.array(obs["hand_joints"]['single']).squeeze()
            self.curr_arm_joints = np.array(obs["arm_joints"]['single'][0:self.joint_dim])
            # print('curr_arm_joints:', self.curr_arm_joints)
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

    def _load_fake_episode(self) -> None:
        """Load trajectory.hdf5 into memory for fake_env replay."""
        if not os.path.isfile(self._fake_episode_path):
            raise FileNotFoundError(f"fake_env episode not found: {self._fake_episode_path}")

        with h5py.File(self._fake_episode_path, "r") as f:
            self._fake_compress = bool(f.attrs.get("compress", True))
            puppet = f["puppet"]
            self._fake_num_steps = int(puppet["end_effector"].shape[0])
            self._fake_end_effector = puppet["end_effector"][:]
            self._fake_arm_joints = puppet["arm_joint_position"][:]
            self._fake_hand_joints = puppet["hand_joint_position"][:]
            self._fake_actions = np.concatenate([puppet["delta_end_effector"][0:-1], puppet["hand_joint_position"][1:]], axis=1) 
            self._fake_actions = np.concatenate([self._fake_actions, self._fake_actions[-2:]], axis=0)
            self._fake_rewards = puppet["reward"][:].astype(np.float32)
            self._fake_dones = puppet["done"][:]

            rgb_group = f["observations/rgb_images"]
            self._fake_rgb_bytes = {}
            for key in self._image_keys:
                camera_key = _FAKE_CAMERA_KEY_MAP.get(key, key)
                if camera_key not in rgb_group:
                    raise KeyError(
                        f"Image key '{key}' maps to '{camera_key}', "
                        f"but it was not found in trajectory file."
                    )
                self._fake_rgb_bytes[key] = rgb_group[camera_key][:]

        print_green(
            f"fake_env loaded {self._fake_num_steps} steps from {self._fake_episode_path}"
        )

    def _apply_fake_step_state(self, step_idx: int) -> None:
        step_idx = int(np.clip(step_idx, 0, self._fake_num_steps - 1))
        self.currpos = self._fake_end_effector[step_idx].astype(np.float64).copy()
        self.curr_action = self._fake_actions[step_idx].astype(np.float64).copy()
        self.curr_arm_joints = (
            self._fake_arm_joints[step_idx][: self.joint_dim].astype(np.float64).copy()
        )
        self.curr_gripper_joints = (
            np.array(self._fake_hand_joints[step_idx]).squeeze().astype(np.float64)
        )

    def _decode_fake_image(self, step_idx: int, image_key: str) -> np.ndarray:
        raw = self._fake_rgb_bytes[image_key][step_idx]
        rgb, _ = decoder_image(raw, None, bgr2rgb=self._bgr2rgb)
        if rgb is None:
            raise RuntimeError(
                f"Failed to decode fake image '{image_key}' at step {step_idx}"
            )
        return rgb

    def _fake_step(self, action: np.ndarray) -> tuple:
        del action
        start_time = time.time()
        self.curr_path_length += 1

        at_last_step = self._fake_step_idx >= self._fake_num_steps - 1
        # print("------------------------------->>> self._fake_num_steps:", self._fake_num_steps)
        if not at_last_step:
            self._fake_step_idx += 1

        obs = self._get_obs()
        reward = float(self._fake_rewards[self._fake_step_idx])
        terminated = at_last_step or bool(self._fake_dones[self._fake_step_idx])
        # truncated = self.curr_path_length >= self.max_episode_length
        truncated = self._fake_step_idx >= self._fake_num_steps - 1
        curr_pose_euler = self.pose_quat2euler(self.currpos)

        sleep_time = max(0, 1 / self.hz - (time.time() - start_time))
        time.sleep(sleep_time)

        return (
            obs,
            int(reward),
            terminated,
            truncated,
            {
                "succeed": terminated,
                "curr_pose_euler": curr_pose_euler,
                "is_intervention": False,
            },
        )





