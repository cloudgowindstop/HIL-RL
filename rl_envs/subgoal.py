"""Multi-subgoal policy / classifier helpers.

Episode flow when ``use_subgoals=true``:
  stage 0 → ... → classifier_0 success → reward_pos, advance (no robot reset)
  stage 1 → ... → classifier_1 success → reward_pos, episode terminate

Each stage has its own classifier (and optionally its own online fine-tune buffers).
Transitions are tagged with ``info['subgoal_id']`` for learner routing.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.train_utils import load_training_state
from lerobot.utils.transition import Transition
from lerobot.utils.utils import has_method

from rl_envs.reward_wrapper import (
    _GRADSCALER_HAS_DEVICE_PARAM,
    _classifier_image_keys,
    _load_classifier_replay_buffer,
    make_policy_obs,
    print_green,
)
from rl_envs.shared_state import shared_state

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


def find_subgoal_wrapper(env):
    """Walk wrapper stack for SubgoalRewardClassifierWrapper."""
    cur = env
    while cur is not None:
        if isinstance(cur, SubgoalRewardClassifierWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


def _to_plain(obj: Any) -> Any:
    """OmegaConf → plain Python containers."""
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


class _RewardClassifierUnit:
    """One independent reward classifier (+ optional online fine-tune state)."""

    def __init__(self, classifier_cfg, cfg, robot_type: str, name: str = ""):
        self.name = name or str(getattr(classifier_cfg, "task_name", "subgoal"))
        self.load_classifier = classifier_cfg.load_classifier
        self.robot_type = robot_type
        self.device = torch.device("cuda:0")
        self.reward_pos = classifier_cfg.reward_pos
        self.reward_neg = classifier_cfg.reward_neg
        self.classifier_keys = classifier_cfg.classifier_keys
        self.batch_size = classifier_cfg.batch_size
        self.require_train = classifier_cfg.require_train
        self.train_epoch = 0
        self.last_obs = None
        self.accuracy_sum = 0.0
        self.accuracy_count = 0

        if self.load_classifier:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            ckpt = str(classifier_cfg.checkpoint_path) + "/pretrained_model"
            print_green(f"[subgoal:{self.name}] load classifier from {ckpt}")
            self.reward_classifier = Classifier.from_pretrained(ckpt)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()
        else:
            self.reward_classifier = None

        if self.require_train and self.load_classifier:
            self.save_dir = os.path.join(os.getcwd(), classifier_cfg.checkpoint_path, "../../")
            print(f"[subgoal:{self.name}] classifier save_dir: {self.save_dir}")
            self.reward_classifier.train()
            original_get_optim_params = self.reward_classifier.get_optim_params
            params = list(original_get_optim_params())
            params_dict = {"reward_classifier": params}
            self.reward_classifier.get_optim_params = lambda p=params_dict: p
            self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(cfg, self.reward_classifier)
            self.reward_classifier.get_optim_params = original_get_optim_params
            if _GRADSCALER_HAS_DEVICE_PARAM:
                self.grad_scaler = GradScaler("cuda", enabled=True)
            else:
                self.grad_scaler = GradScaler(enabled=True)
            self.train_epoch, self.optimizer, self.lr_scheduler = load_training_state(
                Path(classifier_cfg.checkpoint_path), self.optimizer, self.lr_scheduler
            )
            self.optimizer = self.optimizer["reward_classifier"]
            assert self.train_epoch > 0, f"[{self.name}] self.train_epoch is 0"
            self.grad_clip_norm = cfg.optimizer.grad_clip_norm

            origin_root_path = os.path.dirname(classifier_cfg.dataset_path)
            task_name = os.path.basename(classifier_cfg.dataset_path)
            image_keys = _classifier_image_keys(self.reward_classifier)

            saved_root = cfg.dataset.root
            cfg.dataset.root = os.path.join(origin_root_path, task_name + "_success")
            success_dataset = make_dataset(cfg)
            self.pos_buffer = _load_classifier_replay_buffer(
                success_dataset,
                device=self.device,
                image_keys=image_keys,
                capacity=5000,
                cache_name=f"classifier_pos_{self.name}",
                seed=0,
            )
            cfg.dataset.root = os.path.join(origin_root_path, task_name + "_failure")
            failure_dataset = make_dataset(cfg)
            self.neg_buffer = _load_classifier_replay_buffer(
                failure_dataset,
                device=self.device,
                image_keys=image_keys,
                capacity=30000,
                cache_name=f"classifier_neg_{self.name}",
                seed=1,
            )
            cfg.dataset.root = saved_root
            self.jsonl_file_path = os.path.join(
                classifier_cfg.checkpoint_path, "../../", "training_results.jsonl"
            )
        else:
            self.optimizer = None
            self.lr_scheduler = None
            self.grad_scaler = None
            self.pos_buffer = None
            self.neg_buffer = None
            self.grad_clip_norm = None

    def predict_success(self, reward_obs) -> bool:
        if not self.load_classifier or self.reward_classifier is None:
            return False
        with torch.inference_mode():
            return bool(self.reward_classifier.predict_reward(reward_obs, threshold=0.9))

    def online_update(self, action, rew: float, terminated: bool):
        if not (self.require_train and self.load_classifier):
            return
        # PoseActionPadWrapper expands compact policy action (e.g. 4) to full env action (e.g. 7);
        # classifier buffers are built from offline data with compact dims — match that.
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
            truncated=torch.tensor(False),
        )
        if rew > 0.5:
            # print(f"[subgoal:{self.name}] add to pos_buffer")
            self.pos_buffer.add(**transition)
        else:
            # print(f"[subgoal:{self.name}] add to neg_buffer")
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
            pos_sample = next(pos_iterator)
            neg_sample = next(neg_iterator)
            new_batch = {
                key: torch.cat([pos_sample["state"][key], neg_sample["state"][key]], dim=0)
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
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.reward_classifier.parameters(),
                self.grad_clip_norm,
                error_if_nonfinite=False,
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None and batch_idx % 10 == 0:
                self.lr_scheduler.step()
            if has_method(self.reward_classifier, "update"):
                self.reward_classifier.update()

        avg_loss = total_loss / num_batches
        avg_accuracy = self.accuracy_sum / self.accuracy_count if self.accuracy_count > 0 else 0.0
        print(
            f"[subgoal:{self.name}] Training completed - Epoch: {self.train_epoch}, "
            f"Avg Loss: {avg_loss:.6f}, Avg Accuracy: {avg_accuracy:.4f}."
        )
        self.train_epoch += 1
        self.reward_classifier.eval()


class SubgoalRewardClassifierWrapper(gym.Wrapper):
    """Reward / terminate using the classifier of the active subgoal; advance on success.

    Each subgoal may override (applied to ``env.unwrapped`` on reset / stage advance):
      - ``hz``: BaseEnv step pacing
      - ``max_episode_length``: truncates when steps *within this subgoal* exceed the limit
      - ``action_scale``: [pos, rot, gripper] used by BaseEnv / HumanIntervention
      - ``abs_pose_limit_low`` / ``abs_pose_limit_high``: rebuild xyz/rpy boxes for clip_safety_box
    """

    def __init__(self, env, subgoals_cfg, cfg=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type
        self.units: list[_RewardClassifierUnit] = []
        self.subgoal_hz: list[float] = []
        self.subgoal_max_len: list[int] = []
        self.subgoal_action_scale: list[list[float]] = []
        self.subgoal_pose_limit_low: list[Any] = []
        self.subgoal_pose_limit_high: list[Any] = []

        for i, sg in enumerate(subgoals_cfg):
            name = str(getattr(sg, "name", f"sg{i}"))
            unit = _RewardClassifierUnit(sg.classifier_cfg, cfg, self.robot_type, name=name)
            self.units.append(unit)
            self.subgoal_hz.append(float(sg.hz))
            self.subgoal_max_len.append(int(sg.max_episode_length))
            self.subgoal_action_scale.append([float(x) for x in list(sg.action_scale)])

            low = sg.abs_pose_limit_low
            high = sg.abs_pose_limit_high
            self.subgoal_pose_limit_low.append(_to_plain(low))
            self.subgoal_pose_limit_high.append(_to_plain(high))

        assert len(self.units) >= 1
        # Whole-episode safety cap on base env (sum of stage budgets).
        total_max = int(sum(self.subgoal_max_len))
        env.unwrapped.max_episode_length = max(env.unwrapped.max_episode_length, total_max)
        self.current_subgoal = 0
        self.time_step = 0  # steps within current subgoal
        self.episode_step = 0
        self._apply_subgoal_runtime(0)

    @property
    def n_subgoals(self) -> int:
        return len(self.units)

    def _apply_pose_limits(self, unwrapped, low: Any, high: Any) -> None:
        """Rebuild xyz_bounding_box and rpy_bounding_box from abs_pose_limit dict/list."""
        if unwrapped.dual_arm and isinstance(low, dict):
            unwrapped.xyz_bounding_box = gym.spaces.Dict(
                {
                    arm: gym.spaces.Box(
                        np.asarray(low[arm][:3], dtype=np.float64),
                        np.asarray(high[arm][:3], dtype=np.float64),
                        dtype=np.float64,
                    )
                    for arm in low
                }
            )
            unwrapped.rpy_bounding_box = gym.spaces.Dict(
                {
                    arm: gym.spaces.Box(
                        np.asarray(low[arm][3:], dtype=np.float64),
                        np.asarray(high[arm][3:], dtype=np.float64),
                        dtype=np.float64,
                    )
                    for arm in low
                }
            )
            return

        if isinstance(low, dict):
            active = unwrapped.active_arm() if hasattr(unwrapped, "active_arm") else None
            key = active if active in low else next(iter(low))
            low, high = low[key], high[key]
        unwrapped.xyz_bounding_box = gym.spaces.Box(
            np.asarray(low[:3], dtype=np.float64),
            np.asarray(high[:3], dtype=np.float64),
            dtype=np.float64,
        )
        unwrapped.rpy_bounding_box = gym.spaces.Box(
            np.asarray(low[3:], dtype=np.float64),
            np.asarray(high[3:], dtype=np.float64),
            dtype=np.float64,
        )

    def _apply_subgoal_runtime(self, sg_id: int) -> None:
        """Switch hz / action_scale / abs pose limits for the active subgoal."""
        max_len = self.subgoal_max_len[sg_id]
        unwrapped = self.env.unwrapped
        unwrapped.hz = self.subgoal_hz[sg_id]
        unwrapped.action_scale = list(self.subgoal_action_scale[sg_id])

        # clip_safety_box 读 xyz/rpy_bounding_box；yaml 为 {arm: [x,y,z,r,p,y]}
        self._apply_pose_limits(
            unwrapped,
            self.subgoal_pose_limit_low[sg_id],
            self.subgoal_pose_limit_high[sg_id],
        )

        # print_green(
        #     f"[subgoal:{self.units[sg_id].name}] apply hz={self.subgoal_hz[sg_id]}, "
        #     f"max_episode_length={max_len}, action_scale={self.subgoal_action_scale[sg_id]}, "
        #     f"abs_pose_limit_low={self.subgoal_pose_limit_low[sg_id]}, "
        #     f"abs_pose_limit_high={self.subgoal_pose_limit_high[sg_id]}"
        # )

    def reset(self, **kwargs):
        shared_state.terminate = False
        self.current_subgoal = 0
        self.time_step = 0
        self.episode_step = 0
        self._apply_subgoal_runtime(0)
        obs, info = self.env.reset(**kwargs)
        for key in obs.keys():
            if key != "state" and not os.path.exists(f"resize_online_image_{key}.png"):
                bgr = cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"resize_online_image_{key}.png", bgr)

        reward_obs = make_policy_obs(copy.deepcopy(obs), self.units[0].device, self.robot_type)
        for unit in self.units:
            unit.last_obs = reward_obs
            unit.accuracy_sum = 0.0
            unit.accuracy_count = 0
        info["succeed"] = False
        info["subgoal_id"] = 0
        info["subgoal_advanced"] = False
        info["subgoal_name"] = self.units[0].name
        return obs, info

    def step(self, action):
        self.time_step += 1
        self.episode_step += 1
        obs, _rew, _terminated_env, truncated, info = self.env.step(action)

        sg_id = self.current_subgoal
        unit = self.units[sg_id]
        reward_obs = make_policy_obs(copy.deepcopy(obs), unit.device, self.robot_type)
        success = unit.predict_success(reward_obs)
        rew = unit.reward_pos if success else unit.reward_neg
        stage_success = bool(success)

        # Per-subgoal step budget (independent of whole-episode curr_path_length).
        stage_truncated = self.time_step >= self.subgoal_max_len[sg_id]
        truncated = bool(truncated) or stage_truncated
        if truncated:
            if stage_truncated:
                print(
                    f"[subgoal:{unit.name}] 子目标超时 "
                    f"({self.time_step}/{self.subgoal_max_len[sg_id]} steps, hz={self.subgoal_hz[sg_id]})，将自动重置"
                )
            else:
                print("任务超时，将自动重置")
            time.sleep(1)

        classifier_need_update = False
        if stage_success:
            start_time = time.time()
            print_green(
                f"[subgoal:{unit.name}] 模型判断子目标成功. 请按PAUSE键确认，否则将在5秒后自动继续"
            )
            while True:
                if shared_state.terminate:
                    rew = unit.reward_neg
                    stage_success = False
                    classifier_need_update = True
                    shared_state.terminate = False
                    break
                if time.time() - start_time > 5:
                    break
        if shared_state.terminate:
            print_green(f"[subgoal:{unit.name}] 人类判断子目标成功")
            classifier_need_update = True
            stage_success = True
            rew = unit.reward_pos

        if classifier_need_update:
            # Online update uses the stage that produced the label (before advance).
            unit.online_update(action, rew, terminated=False)

        subgoal_advanced = False
        terminated = False
        episode_succeed = False
        if stage_success and not truncated:
            if sg_id < self.n_subgoals - 1:
                # print_green(
                #     f"[subgoal] advance {unit.name} ({sg_id}) → {self.units[sg_id + 1].name} ({sg_id + 1})"
                # )
                self.current_subgoal = sg_id + 1
                self.time_step = 0
                subgoal_advanced = True
                terminated = False
                episode_succeed = False
                # 消费 Pause，避免采集/训练侧把中间子目标成功当成整局结束
                shared_state.terminate = False
                shared_state.human_intervention_key = False
                self._apply_subgoal_runtime(self.current_subgoal)
            else:
                terminated = True
                episode_succeed = True

        unit.last_obs = reward_obs
        # Keep other units' last_obs in sync for possible human override on advance edge.
        for u in self.units:
            u.last_obs = reward_obs

        info["succeed"] = bool(episode_succeed)
        info["subgoal_id"] = int(sg_id)  # id that produced this step's reward
        info["subgoal_advanced"] = bool(subgoal_advanced)
        info["subgoal_name"] = unit.name
        info["next_subgoal_id"] = int(self.current_subgoal)
        return obs, rew, terminated, truncated, info
