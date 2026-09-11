import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import jax
import numpy as np
from rl_envs.utils.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _flatten_dict_to_array(data_dict: dict) -> np.ndarray:
    """
    将嵌套字典扁平化为numpy数组
    例如: {'tcp_pose': array([...]), 'gripper_pose': array([...])} -> array([...])
    """
    flattened_values = []
    # print("=====================================================")
    # print(f"data_dict: {data_dict}")
    # print("=====================================================")
    for key in sorted(data_dict.keys()):  # 确保顺序一致
        value = data_dict[key]
        if isinstance(value, np.ndarray):
            flattened_values.append(value.flatten())
        else:
            flattened_values.append(np.array(value).flatten())
    
    # print("=====================================================")
    # print(f"np.concatenate(flattened_values): {np.concatenate(flattened_values)}")
    # print("=====================================================")
    return np.concatenate(flattened_values)


def _print_dataset_structure(dataset_dict, prefix="", max_depth=3, current_depth=0):
    """
    递归打印dataset_dict的结构和shape信息
    """
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(dataset_dict, dict):
        for k, v in dataset_dict.items():
            if isinstance(v, np.ndarray):
                print(f"{prefix}{k}: numpy.ndarray, shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"{prefix}{k}: dict")
                _print_dataset_structure(v, prefix + "  ", max_depth, current_depth + 1)
            else:
                print(f"{prefix}{k}: {type(v)}")
    elif isinstance(dataset_dict, np.ndarray):
        print(f"{prefix}numpy.ndarray, shape={dataset_dict.shape}, dtype={dataset_dict.dtype}")
    else:
        print(f"{prefix}{type(dataset_dict)}")


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):  
    if isinstance(dataset_dict, np.ndarray):
        
        if isinstance(data_dict, dict):
            data_dict = _flatten_dict_to_array(data_dict)
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        include_octo_embeddings: Optional[bool] = False,
        include_mc_returns: Optional[bool] = False,
        action_mask: Optional[np.ndarray] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        

        
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        
        if include_mc_returns:
            dataset_dict['mc_returns'] = np.empty((capacity,), dtype=np.float32)
        
        if include_octo_embeddings:
            dataset_dict['embeddings'] = np.empty((capacity, 384), dtype=np.float32)
            dataset_dict['next_embeddings'] = np.empty((capacity, 384), dtype=np.float32)

        if include_next_actions:
            dataset_dict['next_actions'] = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
            dataset_dict['next_intvn'] = np.empty((capacity,), dtype=bool)

        if include_label:
            dataset_dict['labels'] = np.empty((capacity,), dtype=int)

        if include_grasp_penalty:
            dataset_dict['grasp_penalty'] = np.empty((capacity,), dtype=np.float32)

        # print("=== Final Dataset Structure ===")
        # _print_dataset_structure(dataset_dict)
        # print("===================================")

        super().__init__(dataset_dict, action_mask=action_mask)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(
                    f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
