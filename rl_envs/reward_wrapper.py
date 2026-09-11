from pathlib import Path
import logging
import json 
import torch
try:
    from torch.amp import GradScaler
    _GRADSCALER_HAS_DEVICE_PARAM = True
except ImportError:
    from torch.cuda.amp import GradScaler
    _GRADSCALER_HAS_DEVICE_PARAM = False
import time
import gymnasium as gym
import os
import copy      
from tqdm import tqdm


from lerobot.utils.buffer import ReplayBuffer
from lerobot.datasets.factory import make_dataset
from lerobot.utils.utils import has_method
from lerobot.utils.transition import Transition
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
import cv2

from rl_envs.shared_state import shared_state

from typing import Dict, Tuple, Optional, Union, Any, List


CLASSIFIER_BUFFER_CAPACITY = 10000


def make_policy_obs(obs: dict, device: torch.device, robot_type: str) -> dict:
    # 先将numpy数组转换为Tensor，再调整维度顺序
    policy_obs = {}
    for keys in obs.keys():
        if "state" not in keys:
            img = torch.from_numpy(obs[keys]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
            new_key = "observation.images." + keys
            policy_obs[new_key] = img
        else:
            state = torch.from_numpy(obs[keys]).float().unsqueeze(0).to(device)
            new_key = "observation.state"
            policy_obs[new_key] = state
    return policy_obs


def print_green(x: Any) -> None:
    return print("\033[92m {}\033[00m".format(x))


def _classifier_image_keys(reward_classifier) -> list[str]:
    return [k for k in reward_classifier.config.input_features if "image" in k]


def _load_classifier_replay_buffer(
    dataset,
    *,
    device: torch.device,
    image_keys: list[str],
    capacity: int = CLASSIFIER_BUFFER_CAPACITY,
    cache_name: str = "classifier_replay",
    seed: int = 0,
) -> ReplayBuffer:
    """
    Fast path for reward-classifier buffers:
    - caps at ``capacity`` frames (random subset)
    - only loads image keys needed by the classifier (no next-state re-decode)
    - caches tensors under ``dataset.root`` so later startups skip video decode
    """
    root = Path(dataset.root)
    cache_path = root / f".{cache_name}_cap{capacity}.pt"
    n = len(dataset)
    take_n = min(capacity, n)

    if cache_path.is_file():
        print(f"[reward_wrapper] Loading cached classifier buffer: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        buffer = ReplayBuffer(
            capacity=capacity,
            device=device,
            state_keys=payload["state_keys"],
            storage_device="cpu",
            optimize_memory=True,
            use_drq=True,
        )
        first_state = {k: v[0:1] for k, v in payload["states"].items()}
        buffer._initialize_storage(state=first_state, action=payload["actions"][0:1], complementary_info=None)
        size = int(payload["size"])
        for key in buffer.state_keys:
            buffer.states[key][:size] = payload["states"][key][:size]
        buffer.actions[:size] = payload["actions"][:size]
        buffer.rewards[:size] = payload["rewards"][:size]
        buffer.dones[:size] = payload["dones"][:size]
        buffer.truncateds[:size] = payload["truncateds"][:size]
        buffer.size = size
        buffer.position = size % capacity
        buffer.initialized = True
        print(f"[reward_wrapper] Cached buffer ready: size={size}")
        return buffer

    if n > take_n:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=rng)[:take_n].tolist()
        indices.sort()
    else:
        indices = list(range(n))

    print(
        f"[reward_wrapper] Building classifier buffer from {root} "
        f"({take_n}/{n} frames, images only, no next-state decode)"
    )

    sample0 = dataset[indices[0]]
    action_dim = int(sample0["action"].numel())
    first_state = {k: sample0[k].unsqueeze(0).float() for k in image_keys}
    buffer = ReplayBuffer(
        capacity=capacity,
        device=device,
        state_keys=image_keys,
        storage_device="cpu",
        optimize_memory=True,
        use_drq=True,
    )
    buffer._initialize_storage(
        state=first_state,
        action=torch.zeros(1, action_dim),
        complementary_info=None,
    )

    dummy_action = torch.zeros(action_dim)
    for idx in tqdm(indices, desc=f"load {root.name}", leave=False):
        sample = dataset[idx]
        state = {k: sample[k].unsqueeze(0).float() for k in image_keys}
        reward = float(sample["next.reward"].item())
        # Classifier only needs (image, label); skip next-state to avoid a second video decode.
        buffer.add(
            state=state,
            action=dummy_action,
            reward=reward,
            next_state=state,
            done=False,
            truncated=False,
            complementary_info=None,
        )

    size = buffer.size
    torch.save(
        {
            "state_keys": list(image_keys),
            "states": {k: buffer.states[k][:size].contiguous() for k in image_keys},
            "actions": buffer.actions[:size].contiguous(),
            "rewards": buffer.rewards[:size].contiguous(),
            "dones": buffer.dones[:size].contiguous(),
            "truncateds": buffer.truncateds[:size].contiguous(),
            "size": size,
        },
        cache_path,
    )
    print(f"[reward_wrapper] Saved classifier buffer cache -> {cache_path} (size={size})")
    return buffer


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env, classifier_cfg, cfg=None):
        super().__init__(env)
        self.load_classifier = classifier_cfg.load_classifier
        observation_space = copy.deepcopy(env.observation_space)
        self.algorithm = classifier_cfg.algorithm
        self.robot_type = env.unwrapped.robot_type
        self.device = torch.device("cuda:0")
        
        if self.load_classifier:
            print("=============================attention! cfg is not used!=============================")
            # self.cfg = cfg
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
            self.reward_classifier = Classifier.from_pretrained(str(classifier_cfg.checkpoint_path)+'/pretrained_model')
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()
        else:
            self.classifier = None 
        self.time_step = 0
        self.reward_pos = classifier_cfg.reward_pos
        self.reward_neg = classifier_cfg.reward_neg
        self.classifier_keys = classifier_cfg.classifier_keys
        self.task_name = classifier_cfg.task_name
        self.train_epoch = 0
        self.batch_size = classifier_cfg.batch_size
        self.require_train = classifier_cfg.require_train


        if self.require_train and self.load_classifier:
            self.save_dir = os.path.join(os.getcwd(), classifier_cfg.checkpoint_path, "../../")
            print('------> save model to :', self.save_dir)
            from lerobot.optim.factory import make_optimizer_and_scheduler
            from lerobot.optim.optimizers import MultiAdamConfig

            # ============== reload optimizer and scheduler ==============
            self.reward_classifier.train()
            original_get_optim_params = self.reward_classifier.get_optim_params
            params = original_get_optim_params()
            params = list(params)    
            """
            "resume 实现
            """
            # optimizer_groups = cfg.optimizer.optimizer_groups
            params_dict = {"reward_classifier": params}
            self.reward_classifier.get_optim_params = lambda p=params_dict: p
            self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(cfg, self.reward_classifier)
            self.reward_classifier.get_optim_params = original_get_optim_params
            self.grad_scaler = GradScaler(enabled=True)
            self.train_epoch, self.optimizer, self.lr_scheduler = load_training_state(Path(classifier_cfg.checkpoint_path), self.optimizer, self.lr_scheduler)
            self.optimizer = self.optimizer['reward_classifier']

            assert self.train_epoch > 0, 'self.train_epoch is 0' 

            # ============ initialize params ===========
            self.grad_clip_norm = cfg.optimizer.grad_clip_norm
            #============= reload dataset ==============
            origin_root_path = os.path.dirname(classifier_cfg.dataset_path)
            task_name = os.path.basename(classifier_cfg.dataset_path)
            image_keys = _classifier_image_keys(self.reward_classifier)

            cfg.dataset.root = os.path.join(origin_root_path, task_name + "_success")
            success_dataset = make_dataset(cfg)
            self.pos_buffer = _load_classifier_replay_buffer(
                success_dataset,
                device=self.device,
                image_keys=image_keys,
                capacity=5000,
                cache_name="classifier_pos",
                seed=0,
            )

            cfg.dataset.root = os.path.join(origin_root_path, task_name + "_failure")
            failure_dataset = make_dataset(cfg)
            self.neg_buffer = _load_classifier_replay_buffer(
                failure_dataset,
                device=self.device,
                image_keys=image_keys,
                capacity=30000,
                cache_name="classifier_neg",
                seed=1,
            )

            # ==================== append logs ====================
            self.jsonl_file_path = os.path.join(classifier_cfg.checkpoint_path, "../../",  "training_results.jsonl")
            self.accuracy_sum = 0.0
            self.accuracy_count = 0 




    def reset(self, **kwargs):
        # todo: change here to False
        shared_state.terminate = False
        obs, info = self.env.reset(**kwargs)

        for key in obs.keys():
            if key != "state":
                if not os.path.exists(f"resize_online_image_{key}.png"):
                    bgr = cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"resize_online_image_{key}.png", bgr)

        reward_obs = copy.deepcopy(obs)
        reward_obs = make_policy_obs(reward_obs, self.device, self.robot_type)
        self.last_obs = reward_obs
        self.time_step = 0
        info['succeed'] = False
        self.accuracy_sum = 0.0
        self.accuracy_count = 0 
        return obs, info
    
    def step(self, action):
        self.time_step += 1
        obs, rew, _, truncated, info = self.env.step(action)
        # obs, rew, _, truncated, info = self.lp_wrapper(action)
        # self.lp.print_stats()
        reward_st_time = time.time()
        reward_obs = copy.deepcopy(obs)
        reward_obs = make_policy_obs(reward_obs, self.device, self.robot_type)

        if self.load_classifier:
            with torch.inference_mode():
                success = self.reward_classifier.predict_reward(reward_obs, threshold=0.9)
        else:
            success = False
        if success:
            rew = self.reward_pos
        else:
            rew = self.reward_neg
        terminated = success

        reward_end_time = time.time()
        reward_time = reward_end_time - reward_st_time
        # print(f"----------> reward_time: {reward_time}s")

        if truncated:
            print("任务超时，将自动重置")
            time.sleep(1)
            # print("Task timed out, will be automatically reset")
        classifier_need_update = False
        
        if self.time_step <= 10:
            terminated = False
            shared_state.terminate = False
            rew = self.reward_neg
        else:
            if terminated:   
                start_time = time.time()
                print_green("模型判断任务成功.请按PAUSE键确认，否则将在5秒后自动继续")
                # print("Model judged the task as successfully completed. If this is wrong, press the space bar to correct it; otherwise, the program will automatically continue after 5 seconds.")
                while True:
                    # If shared_state.terminate is used to flag failure, reset it to False after handling (task continues).
                    if shared_state.terminate:
                        rew = self.reward_neg
                        terminated = False
                        classifier_need_update = True
                        shared_state.terminate = False
                        break
                    if time.time() - start_time > 5:
                        break
            elif shared_state.terminate:
                print_green('人类判断任务成功')
                # print('Human judged the task as successfully completed.')
                # If the human sets shared_state.terminate = True to confirm success, do not reset it (task ends).
                classifier_need_update = True
                terminated = True
                rew = self.reward_pos
        
        if classifier_need_update and self.require_train and self.load_classifier:
            # Match offline buffer action dim (PoseActionPadWrapper may expand to full env action).
            action_t = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
            expected = int(self.pos_buffer.actions.shape[-1])
            if action_t.numel() > expected:
                action_t = action_t[:expected]
            elif action_t.numel() < expected:
                pad = torch.zeros(expected, dtype=torch.float32)
                pad[: action_t.numel()] = action_t
                action_t = pad
            transition = Transition(
                state=self.last_obs,
                action=action_t,
                reward=torch.tensor(rew),
                next_state=self.last_obs,
                done=torch.tensor(terminated),
                truncated=torch.tensor(truncated)
            )
            
            if rew > 0.5: 
                print('add to pos_buffer')
                self.pos_buffer.add(**transition)
            else:
                print('add to neg_buffer')
                self.neg_buffer.add(**transition)
            
            self.reward_classifier.train()

            pos_iterator = self.pos_buffer.get_iterator(
                batch_size=self.batch_size, async_prefetch=True, queue_size=2
            )
            neg_iterator = self.neg_buffer.get_iterator(
                batch_size=self.batch_size, async_prefetch=True, queue_size=2
            )

            total_loss = 0.0
            num_batches = 10
            
            for batch_idx in range(num_batches):
                # Sample equal number of positive and negative examples
                pos_sample = next(pos_iterator)
                neg_sample = next(neg_iterator)
                # Classifier only needs images + reward labels (skip full transition concat /
                # is_intervention, which our lightweight classifier buffers do not store).
                new_batch = {
                    key: torch.cat(
                        [pos_sample["state"][key], neg_sample["state"][key]], dim=0
                    )
                    for key in pos_sample["state"]
                }
                new_batch["next.reward"] = torch.cat(
                    [pos_sample["reward"], neg_sample["reward"]], dim=0
                )

                with torch.autocast(device_type=self.device.type):
                    loss, output_dict = self.reward_classifier.forward(new_batch)
                    if output_dict and "accuracy" in output_dict:
                        self.accuracy_sum += output_dict["accuracy"] / 100.0
                        self.accuracy_count += 1
                
                total_loss += loss.item()
                
                self.grad_scaler.scale(loss).backward()

                # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
                self.grad_scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.reward_classifier.parameters(),
                    self.grad_clip_norm,
                    error_if_nonfinite=False,
                )

                # self.reward_classifier.print_grad()

                # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                self.grad_scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                self.grad_scaler.update()

                self.optimizer.zero_grad()

                # Only step scheduler every few batches to avoid too fast decay
                # Step through pytorch scheduler at every batch instead of epoch
                if self.lr_scheduler is not None and batch_idx % 10 == 0:
                    self.lr_scheduler.step()

                if has_method(self.reward_classifier, "update"):
                    # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
                    self.reward_classifier.update()
                
                # Print training statistics
                avg_loss = total_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                avg_accuracy = self.accuracy_sum / self.accuracy_count if self.accuracy_count > 0 else 0.0
                
                print(f'Training completed - Epoch: {self.train_epoch}, '
                      f'Avg Loss: {avg_loss:.6f}, '
                      f'Avg Accuracy: {avg_accuracy:.4f}.')

            self.train_epoch += 1
            
            print("=============================attention! checkpoint is not saved!=============================")
            # if self.train_epoch % 10 == 0: 
            #     checkpoint_dir = get_step_checkpoint_dir(Path(self.save_dir), 10*self.train_epoch, self.train_epoch)
            #     logging.info(f"Checkpoint policy after step {self.train_epoch}, save at {checkpoint_dir}")
            #     save_checkpoint(checkpoint_dir, self.train_epoch, self.cfg, self.reward_classifier, self.optimizer, self.lr_scheduler)
            #     update_last_checkpoint(checkpoint_dir)

            #     if self.accuracy_count > 0:
            #         avg_accuracy = self.accuracy_sum / self.accuracy_count
            #         log_entry = {
            #             "epoch": self.train_epoch,
            #             "train_loss": f"{loss.item():.8f}",
            #             "train_accuracy": f"{avg_accuracy:.8f}"
            #         }

            #         with open(self.jsonl_file_path, "a") as f:
            #             f.write(json.dumps(log_entry) + "\n")
            #         self.accuracy_sum = 0.0
            #         self.accuracy_count = 0
            self.reward_classifier.eval()
                


        self.last_obs = reward_obs
        info['succeed'] = bool(terminated)

        return obs, rew, terminated, truncated, info


class EmergencyTerminateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        shared_state.emergency_terminate = False

    def reset(self, **kwargs):
        shared_state.emergency_terminate = False
        obs, info = self.env.reset(**kwargs)
        return obs, info
        
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)


        if shared_state.emergency_terminate:
            terminated = True
            rew = -5.0
            print("紧急终止，奖励为-5.0, 环境重置!")
        return obs, rew, terminated, truncated, info



class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (14,) or env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.env.unwrapped.curr_gripper_joints
        info['discrete_penalty'] = 0.0
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        action = copy.deepcopy(action)
        observation, reward, terminated, truncated, info = self.env.step(action)

        if "intervene_action" in info:
            action = info["intervene_action"]

        last_pos_val = self.last_gripper_pos
        if isinstance(last_pos_val, dict):
            unwrapped = self.env.unwrapped
            active = unwrapped.active_arm() if hasattr(unwrapped, "active_arm") else None
            if active is not None and active in last_pos_val:
                last_pos_val = last_pos_val[active]
            else:
                # dual-arm fallback: right then left
                last_pos_val = last_pos_val.get("right", next(iter(last_pos_val.values())))

        if hasattr(last_pos_val, '__getitem__') and hasattr(last_pos_val, '__len__'):
             if len(last_pos_val) > 0:
                last_pos_val = last_pos_val[0]

        if (action[-1] > 0.3 and last_pos_val < 0.3) or (
            action[-1] < 0.5 and last_pos_val > 0.5):
            info['discrete_penalty'] = self.penalty
        else:
            info['discrete_penalty'] = 0.0

        self.last_gripper_pos = self.env.unwrapped.curr_gripper_joints

        return observation, reward, terminated, truncated, info