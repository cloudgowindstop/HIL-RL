import time
import numpy as np
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten
# from xrocs.utils.logger.logger_loader import logger
from rl_envs.shared_state import shared_state
import cv2
import traceback
import sys
from collections import deque
from typing import Optional
import torch
COSMOS_IMAGE_SIZE = 224  # Standard image size expected by Cosmos policies
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4


class FakeHumanIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
    

    def reset(self, **kwargs):
        """Reset the environment and sync robot position."""
        obs, info = self.env.reset(**kwargs)
        info["is_intervention"] = False
        return obs, info


    def action(self, action: np.ndarray) -> np.ndarray:
        # intervened = True
        action = np.zeros(7, dtype=np.float32)
        return action, None, True

    def step(self, action):
        # print("========================= no sync_xtele")
        action, xtele_joints,replaced = self.action(action)
        obs, rew, terminated, truncated, info = self.env.step(action)
        info["intervene_action"] = action
        info["is_intervention"] = replaced
        return obs, rew, terminated, truncated, info

class HumanIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.env.unwrapped.init_xtele() # init xtele
        self.control_mode = env.unwrapped.control_mode
        self.enable_rotation = env.unwrapped.enable_rotation

    

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
    


    def transform_pose(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        curr_matrix = self.pose2matrix(current_pose)
        tar_matrix = self.pose2matrix(target_pose)


        T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)

        return T_diff_matrix

    def action(self, action: np.ndarray) -> np.ndarray:
        # intervened = True
        intervened = shared_state.human_intervention_key
        if intervened:
            # update1:不进行get_xtele
            # print("========================= no get_xtele")
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

                if not self.enable_rotation:
                    if "franka" in self.robot_type:
                        expert_a[3:6] = [0.0, 0.0, 0.0]
                    else:
                        raise NotImplementedError("Unknown robot type")

                return expert_a, xtele_joints, True
            except Exception as e:
                print(f"Error in action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
            # return action, None, False
        return action, None, False

    def step(self, action):
        # print("========================= no sync_xtele")
        action, xtele_joints,replaced = self.action(action)
        if replaced:
            obs, rew, terminated, truncated, info = self.env.step(action)
            info["intervene_action"] = action
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)   
            # # update2:不进行sync_xtele，但是is_intervention为True，并且干预时刻则terminated为True
            # print("------------------------->>> Human intervention detected, terminating current episode.")
            self.env.unwrapped.sync_xtele(timeout=0.1)
        info["is_intervention"] = replaced
        
        # intervened = shared_state.human_intervention_key
        # if intervened:
        #     info["is_intervention"] = True
        # else:
        #     info["is_intervention"] = False

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

def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()

def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: isinstance(x, list)
    )


class ChunkingWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon
            )

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon

        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
        return (stack_obs(self.current_obs), reward, done, trunc, info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


class CosmosWrapper(gym.ObservationWrapper):
    """
    Convert SERLObsWrapper output (flat state + camera images) into a Cosmos policy batch.
    Must be stacked after SERLObsWrapper, which already selects and flattens proprio keys.
    """

    def __init__(
        self,
        env,
        proprio_keys=None,
        use_force=False,
        cosmos_cfg=None,
        task_description: str | None = None,
        t5_text_embeddings_path: str | None = None,
    ):
        super().__init__(env)
        self.cosmos_cfg = cosmos_cfg
        self.observation_space = env.observation_space
        if not task_description:
            raise ValueError("CosmosWrapper requires `task_description` from task yaml config.")
        self.task_description = task_description
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.text_embedding = self.init_global_t5_text_embedding(
            self.task_description,
            self.t5_text_embeddings_path,
        )

    def observation(self, obs):
        from cosmos_policy.experiments.robot.cosmos_utils import (
            prepare_images_for_model,
        )
        from cosmos_policy.utils.utils import duplicate_array

        with torch.inference_mode():
            policy_obs = [
                obs["wrist"],
                obs["right"],
            ]
            policy_obs = prepare_images_for_model(policy_obs, self.cosmos_cfg)
            proprio = obs["state"]
  
            # 构造序列
            image_sequence = []
            current_sequence_idx = 0  # Used to track which index in the sequence of images we are on
            
            # Add blank placeholder image (special placeholder for 1+T temporal VAE compression)
            # 1. 空白帧
            primary_image = policy_obs[1]
            blank_image = np.zeros_like(primary_image)
            image_sequence.append(np.expand_dims(blank_image, axis=0))
            current_sequence_idx += 1

            # 2. proprio
            blank_image_duplicated = duplicate_array(
                blank_image.copy(), total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR
            )
            image_sequence.append(blank_image_duplicated)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            
            # 3. wrist image
            wrist_image = policy_obs[0]
            wrist_image_duplicated = duplicate_array(wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
            image_sequence.append(wrist_image_duplicated)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            # 4. primary image
            primary_image_duplicated = duplicate_array(
                primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR
            )
            image_sequence.append(primary_image_duplicated)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            # 5. action
            image_sequence.append(blank_image_duplicated.copy())
            action_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            # 6. future proprio
            image_sequence.append(blank_image_duplicated.copy())
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            # 7. future wrist image
            image_sequence.append(wrist_image_duplicated.copy())
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            # 8. future primary image
            image_sequence.append(primary_image_duplicated.copy())
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            # 9. value
            image_sequence.append(blank_image_duplicated.copy())
            value_latent_idx = current_sequence_idx
            current_sequence_idx += 1

            batch_size = 1
            raw_image_sequence = np.concatenate(image_sequence, axis=0)  # (T, H, W, C) 
            raw_image_sequence = np.expand_dims(raw_image_sequence, axis=0)  # (T, H, W, C) -> (1, T, H, W, C)
            raw_image_sequence = np.tile(raw_image_sequence, (batch_size, 1, 1, 1, 1))  # (1, T, H, W, C) -> (B, T, H, W, C)
            raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))  # (B, T, H, W, C) -> (B, C, T, H, W)



            raw_image_sequence = torch.from_numpy(raw_image_sequence).to(dtype=torch.uint8)
            # # fix2:device指定
            proprio_tensor = (
                torch.from_numpy(proprio).reshape(batch_size, -1).to(dtype=torch.bfloat16)
            )  # (B, proprio_dim)

            data_batch = {
                # "dataset_name": "video_data",
                "video": raw_image_sequence,  # (B, C, T, H, W)
                "t5_text_embeddings": self.text_embedding.repeat(batch_size, 1, 1).to(dtype=torch.bfloat16),
                "fps": torch.tensor(
                    [16] * batch_size, dtype=torch.bfloat16
                ),  # Just match the training config (always 16 FPS)
                "padding_mask": torch.zeros(
                    (batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16
                ),  # Padding mask (assume no padding here)
                # diff3: 推理时固定前几帧为条件帧，训练的时候由于需要作为世界模型/价值函数样本，所以需要动态调整，但是只有demo的时候相同
                # "num_conditional_frames": model.config.min_num_conditional_frames,  # Number of latent frames used as conditioning
                "num_conditional_frames": 4,
                "proprio": proprio_tensor,
                # Specify the indices of various elements in the latent diffusion sequence
                "current_proprio_latent_idx": (
                    torch.tensor([current_proprio_latent_idx] * batch_size, dtype=torch.int64)
                ),
                "current_wrist_image_latent_idx": (
                    torch.tensor([current_wrist_image_latent_idx] * batch_size, dtype=torch.int64)
                ),
                
                "current_image_latent_idx": (
                    torch.tensor([current_image_latent_idx] * batch_size, dtype=torch.int64)
                ),
                "action_latent_idx": torch.tensor([action_latent_idx] * batch_size, dtype=torch.int64),
                "future_proprio_latent_idx": (
                    torch.tensor([future_proprio_latent_idx] * batch_size, dtype=torch.int64)
                ),
                "future_wrist_image_latent_idx": (
                    torch.tensor([future_wrist_image_latent_idx] * batch_size, dtype=torch.int64)
                ),
                "future_image_latent_idx": (
                    torch.tensor([future_image_latent_idx] * batch_size, dtype=torch.int64)
                ),
                "value_latent_idx": torch.tensor([value_latent_idx] * batch_size, dtype=torch.int64),
                "obs.wrist": obs["wrist"],
                "obs.right": obs["right"],
                "obs.state": obs["state"],
            }
        return data_batch

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info


    def init_global_t5_text_embedding(
        self,
        task_description: str,
        t5_text_embeddings_path: str,
        gpu_id: int = 0,
    ) -> torch.Tensor:
        """Initialize T5 cache once and reuse the same text embedding globally."""
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_t5_embedding_from_cache,
            init_t5_text_embeddings_cache,
        )
        init_t5_text_embeddings_cache(t5_text_embeddings_path, worker_id=gpu_id)
        text_embedding = get_t5_embedding_from_cache(task_description)
        return text_embedding
