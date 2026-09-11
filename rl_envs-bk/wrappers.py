import time
import numpy as np
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten
from rl_envs.shared_state import shared_state
import cv2
import traceback
import sys
import torch
from libero.libero import benchmark


class HumanIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.env.unwrapped.init_xtele() # init xtele
        self.control_mode = env.unwrapped.control_mode
    

    def reset(self, **kwargs):
        """Reset the environment and sync robot position."""
        obs, info = self.env.reset(**kwargs)
        shared_state.human_intervention_key = False
        self.env.unwrapped.sync_xtele(timeout=2)
        info["is_intervention"] = False
        return obs, info

    def pose2matrix(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = Rotation.from_quat(pose_quat).as_matrix()
        pose_matrix[:3, 3] = pose_t
        return pose_matrix
    

    def action(self, action: np.ndarray) -> np.ndarray:
        # intervened = True
        intervened = shared_state.human_intervention_key
        if intervened:
            try:
                obs = self.env.unwrapped.get_xtele()
                xtele_joints, xtele_pose = obs['joints'], obs['pose']
                # print("gripper_value:", len(xtele_joints), xtele_joints[-1])
                if self.control_mode == "joint":
                    expert_a = xtele_joints
                else:
                    curr_matrix = self.pose2matrix(self.env.unwrapped.currpos)
                    tar_matrix = self.pose2matrix(xtele_pose)
                    T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)

                    
                rel_rot = Rotation.from_matrix(T_diff_matrix[:3, :3]).as_euler("xyz")
                rel_pos = T_diff_matrix[:3, 3]
                expert_a = np.zeros(7, dtype=np.float32)
                expert_a[:3] = rel_pos / self.env.unwrapped.action_scale[0]
                expert_a[3:6] = rel_rot / self.env.unwrapped.action_scale[1]
                expert_a[6:] = xtele_joints[-1] / self.env.unwrapped.action_scale[2]
                
                # expert_a = np.clip(expert_a, [-1]*7, [1]*7)
                """
                intervention action 边缘裁剪
                """
                epsilon = 1e-6
                expert_a[0:6]= expert_a[0:6].clip(-1+epsilon, 1-epsilon)

                return expert_a, xtele_joints, True
            except Exception as e:
                print(f"Error in action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
        return action, None, False

    def step(self, action):
        action, xtele_joints,replaced = self.action(action)
        if replaced:
            obs, rew, terminated, truncated, info = self.env.step(action)
            info["intervene_action"] = action
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)        
            self.env.unwrapped.sync_xtele(timeout=0.1)
        

        info["is_intervention"] = replaced
        return obs, rew, terminated, truncated, info




class AugmentedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env

    def observation(self, obs):
        images = obs['images']
        env = self.env.unwrapped
        for key, img in images.items():
            if hasattr(env, 'image_crop'):
                cropped_rgb = env.image_crop[key](img) if key in env.image_crop else img
            else:
                cropped_rgb = img
            cropped_rgb = cv2.resize(
                cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = cropped_rgb

        return obs
    
    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info






class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )


        return observation


from collections import OrderedDict


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None, use_force=False):
        super().__init__(env)
        if use_force:
            self.proprio_keys = proprio_keys
        else:
            self.proprio_keys = proprio_keys[:2]

        print("proprio_keys:", self.proprio_keys)    

        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            OrderedDict((key, self.env.observation_space["state"][key]) for key in self.proprio_keys)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        from collections import OrderedDict
        obs = {
            "state": flatten(
                self.proprio_space,
                OrderedDict((key, obs["state"][key]) for key in self.proprio_keys),
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

  
def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs


class CosmosWrapper(gym.Wrapper):
    """
    CosmosWrapper：处理输入obs text proprio image
    保证step前和step后可以直接调用
    """

    # # 相机图像在数组里的索引对应，这部分放不放在wrapper里面
    # WRIST_IMAGE_IDX = 0
    # IMAGE_IDX = 1
    # 确定性重置种子
    DETERMINISTIC_RESET_SEED = 195

    # def __init__(self, env, cosmos_cfg, dataset_stats, task_description):
    def __init__(self, env, dataset_stats, task_description):
        super().__init__(env)
        from cosmos_policy.experiments.robot.cosmos_utils import (
            rescale_proprio,
        )
        from cosmos_policy.experiments.robot.libero.libero_utils import (
            get_libero_image,
            get_libero_wrist_image,
            get_libero_dummy_action,
        )
        # self.cosmos_cfg = cosmos_cfg # 不导入配置文件
        # from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
        # self.resize_size = get_image_resize_size("cosmos")
        self.resize_size = 224
        self.flip_images = True

        self.dataset_stats = dataset_stats
        # 预计算并缓存 text_embedding（与图像并列作为一种模态）
        # self.text_embedding = self._build_text_embedding(task_description)
        # self.text_embedding = get_t5_embedding_from_cache(task_description)

        self._rescale_proprio = rescale_proprio
        self._get_libero_image = get_libero_image
        self._get_libero_wrist_image = get_libero_wrist_image
        self._get_libero_dummy_action = get_libero_dummy_action
        

    # def _extract_raw_obs(self, obs):
    #     """从Libero环境中提取原始obs中的primary_image、wrist_image数据"""
    #     primary_image = self._get_libero_image(obs, self.flip_images)
    #     wrist_image = self._get_libero_wrist_image(obs, self.flip_images)
    #     return primary_image, wrist_image

    # def _process_images(self, primary_image, wrist_image):
    #     """
    #     按 Libero 顺序组合相机图像，并调用 prepare_images_for_model 进行预处理。
    #     返回 (wrist_img_processed, primary_img_processed)，与 WRIST_IMAGE_IDX / IMAGE_IDX 对应。
    #     """
    #     # Libero顺序：[wrist, primary]
    #     all_camera_images = [wrist_image, primary_image]
    #     processed_images = self._prepare_images_for_model(all_camera_images)

    #     # return processed[self.WRIST_IMAGE_IDX], processed[self.IMAGE_IDX]
    #     return processed_images[0], processed_images[1]

    # def _process_proprio(self, proprio):
    #     """对Libero环境中的proprio进行归一化。"""
    #     proprio = self._rescale_proprio(
    #         proprio,
    #         self.dataset_stats,
    #         non_negative_only=False,
    #         scale_multiplier=1.0,
    #     )
    #     return proprio


    def _process_text_embedding(self, task_description):
        """预计算并缓存text_embedding"""
        from cosmos_policy.experiments.robot.cosmos_utils import get_t5_embedding_from_cache
        # 输出为字符串，还需要这样判断吗
        if isinstance(task_description, str):
            self.text_embedding = get_t5_embedding_from_cache(task_description)
        elif isinstance(task_description, np.ndarray):
            self.text_embedding = torch.tensor(task_description, dtype=torch.bfloat16).cuda()
        else:
            raise TypeError(
                f"task_description 必须是 str 或 np.ndarray，得到 {type(task_description)}"
            )

    def _process_observation(self, obs):
        """处理Libero环境中原始obs，返回预处理后的observation"""
        # HIL-RL环境中的obs，定义为dict_keys(['observation.state', 'observation.images.right', 'observation.images.wrist'])
        # 从libero环境中提取obs，转化为HIL-RL环境中定义的obs格式
        # 原prepare_observation的实现：提取原始图像
        primary_image = self._get_libero_image(obs, self.flip_images)
        wrist_image = self._get_libero_wrist_image(obs, self.flip_images)
        proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))

        # 处理获得的图像：
        all_camera_images = self.prepare_images_for_model([wrist_image, primary_image], self.flip_images)
        # 处理获得的proprio：normalize_proprio
        proprio = self._rescale_proprio(proprio, self.dataset_stats, non_negative_only=False, scale_multiplier=1.0)

        print("observation.images.wrist", all_camera_images[0])
        print("observation.state", proprio)
        # 转化成HIL-RL环境中的observation格式
        # ！！！！确认一下环境转化的obs是什么样的！！！！
        observation = {
            "observation.state": proprio,
            "observation.images.right": all_camera_images[1],
            "observation.images.wrist": all_camera_images[0],
        }
        # 需要注意：1.image 需要除以255吗？ 2.to(device)吗？3.text_embedding需要to(device)
        return observation


    def make_policy_obs(self, obs, device: torch.device) -> dict:
        """libero环境与HIL-RL环境obs输出的接口"""
        policy_obs = {}
        # cosmos需要输出什么obs？
        


    
    def reset(self, deterministic_reset_seed=None):
        # 确定性重置种子
        reset_seed = deterministic_reset_seed if deterministic_reset_seed is not None else self.DETERMINISTIC_RESET_SEED
        from cosmos_policy.experiments.robot.robot_utils import set_seed_everywhere
        set_seed_everywhere(reset_seed)
        # 创建环境需要在哪里解决？——> 使用HIL-RL的dummy环境
        # benchmark_dict = benchmark.get_benchmark_dict()
        # task_suite = benchmark_dict[cosmos_cfg.task_suite_name]()
        # print(f"task_suite: {task_suite}")
        # num_tasks = task_suite.n_tasks
        
        # logging.info(f"Task suite: {cosmos_cfg.task_suite_name}")
        # logging.info(f"Number of tasks: {num_tasks}")
        
        # online_env, task_description = get_libero_env(task_suite.get_task(0), cosmos_cfg.model_family, resolution=cosmos_cfg.env_img_res)
        # print("----------------------------->>> already get libero env! ----------------------------->>>")
        self.env.reset()
        # 加载初始状态（只加载一次，供所有 episode 使用）
        # task_suite需要加载配置文件访问！！！！？
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_10"]() # 先固定任务集为libero_10
        initial_states, all_initial_states = self.load_initial_states(task_suite, 0)
        # 设置第一个 episode 的初始状态
        initial_state = initial_states[0] # 0表示当前任务执行第一个episode的初始状态
        if initial_state is not None:
            obs = self.env.set_init_state(initial_state)
        else:
            obs = self.env.get_observation()

        # 前十步稳定在初始状态
        for _ in range(10):
            obs, reward, done, info = self.env.step(self.get_libero_dummy_action("cosmos"))
        
        return obs, info


    def load_initial_states(self, task_suite, task_id: int):
        # Get default initial states
        # 改写load_initial_states函数，直接认定cfg.initial_states_path == "DEFAULT"，同时也没有all_initial_states返回
        initial_states = task_suite.get_task_init_states(task_id)
        return initial_states

    # # step函数待改
    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #     obs = self._build_policy_obs(obs)
    #     return obs, reward, done, info



    def prepare_observation(self, obs, flip_images: bool = False):
        """Prepare observation for policy input."""
        # Get preprocessed images
        img = self._get_libero_image(obs, self.flip_images)
        wrist_img = self._get_libero_wrist_image(obs, self.flip_images)

        # Prepare observations dict
        observation = {
            "primary_image": img,
            "wrist_image": wrist_img,
            "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
        }

        return observation  # Return processed observation


    def prepare_images_for_model(self, images: list[np.ndarray], flip_images: bool = False) -> list[np.ndarray]:
        """
        Prepare images for Cosmos model input by resizing and cropping as needed.

        Args:
            images (list[np.ndarray]): List of input images as numpy arrays
            flip_images (bool): Whether to flip images vertically across x-axis

        Returns:
            np.ndarray: Processed images ready for the model
        """
        from cosmos_policy.experiments.robot.cosmos_utils import (
            check_images_format,
            apply_image_transforms,
        )
        from cosmos_policy.datasets.dataset_utils import (
            resize_images,
            apply_jpeg_compression_np,
        )
        images = np.stack(images, axis=0)  # (T, H, W, C)
        # Check that the images have the right format
        check_images_format(images)
        # Flip images vertically across x-axis if needed (e.g., for LIBERO and RoboCasa)
        if flip_images:
            images = np.flipud(images)
        # Apply JPEG compression (if trained on JPEG-compressed images)
        images = apply_jpeg_compression_np(images, quality=95)
        # Resize images to match training distribution
        processed_images = resize_images(images, 224)
        # Apply image transformations if trained with image augmentations
        processed_images = apply_image_transforms(processed_images)

        return processed_images

    