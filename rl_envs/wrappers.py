
import numpy as np
from gymnasium import Env, spaces
import gymnasium as gym
from scipy.spatial.transform import Rotation  # 用于旋转表示转换
from gymnasium.spaces import Box
from gymnasium.spaces import flatten_space, flatten  # 用于扁平化观察空间
from xrocs.utils.logger.logger_loader import logger
from .shared_state import shared_state
import cv2  # OpenCV 用于图像处理
import traceback
import sys
import time
from collections import OrderedDict  # 用于保持字典顺序

from typing import Optional
from pynput import keyboard
import torch

from rl_envs.shared_state import shared_state
COSMOS_IMAGE_SIZE = 224  # Standard image size expected by Cosmos policies
COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4


def print_green(x: any) -> None:
    return print("\033[92m {}\033[00m".format(x))

def on_press(key):
    try:
        
        if str(key) == 'Key.scroll_lock':
            print("----------------set human intervention key to {}!----------------".format(shared_state.human_intervention_key))
            shared_state.human_intervention_key = not shared_state.human_intervention_key
            time.sleep(0.5)
        # if str(key) == 'Key.space' or str(key) == 'Key.pause':
        if str(key) == 'Key.pause':
            print("----------------set terminate to true!----------------")
            shared_state.terminate = True
            time.sleep(0.5)
        if str(key) == 'Key.space':
            print("----------------set terminate to false!----------------")
            shared_state.emergency_terminate = True
            time.sleep(0.5)
    except AttributeError:
        pass
try:
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
except Exception as e:
    print("error in keyboard listener:", e)
    exit(0)

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
    def __init__(self, env, intervention_mode="joint", action_indices=None):
        super().__init__(env)
        self.robot_type = env.unwrapped.robot_type  # 机器人类型
        self.dual_arm = env.unwrapped.dual_arm
        self.free_left = env.unwrapped.free_left
        self.free_right = env.unwrapped.free_right
        self.env.unwrapped.init_xtele()  # 初始化遥操作设备（xtele）
        self.control_mode = env.unwrapped.control_mode  # 控制模式（joint 或 pose）
        self.intervention_mode = intervention_mode  # 干预模式（joint 或 pose）
        # self.enable_rotation = env.unwrapped.enable_rotation  # 是否启用旋转控制


    def reset(self, **kwargs):
        shared_state.human_intervention_key = False  # 重置人工干预标志
        # 在内层 reset 前读取：仅第一次且 free_arm_teleop=true 时允许同构控制 free 臂
        allow_free_teleop = (
            hasattr(self.env.unwrapped, "allow_free_arm_teleop_on_reset")
            and self.env.unwrapped.allow_free_arm_teleop_on_reset()
        )
        obs, info = self.env.reset(**kwargs)  # 重置内层环境

        has_free = self.free_left or self.free_right
        if has_free and not allow_free_teleop:
            # free_arm_teleop=false 或 之后的 reset：只同步 active；free 同构下电，机械臂 hold reset
            sync_arms = (
                self.env.unwrapped.active_arm_names()
                if hasattr(self.env.unwrapped, "active_arm_names")
                else None
            )
            self.env.unwrapped.sync_xtele(timeout=2, arms=sync_arms)
            if hasattr(self.env.unwrapped, "_power_off_xtele_arms"):
                self.env.unwrapped._power_off_xtele_arms()
            self.env.unwrapped._enable_free_arm_hold = True
        else:
            # 第一次 free_* reset（free_arm_teleop=true）或无 free：同构臂先与机械臂同构（含 free 臂）
            self.env.unwrapped.sync_xtele(timeout=2)

            # ===== SYNC CHECK DEBUG(mingbo debug) =====
            try:
                xtele_data = self.env.unwrapped.get_xtele()
                for name in self.env.unwrapped.curr_arm_joints.keys():
                    xt = xtele_data['joints'][name][:7]
                    rb = self.env.unwrapped.curr_arm_joints[name]
                    diff = xt - rb
                    print(f"[SYNC CHECK] {name:5s} xtele ={xt.round(3)}")
                    print(f"[SYNC CHECK] {name:5s} robot ={rb.round(3)}")
                    print(f"[SYNC CHECK] {name:5s} diff  ={diff.round(3)}  max_abs={abs(diff).max():.3f}")
            except Exception as e:
                print(f"[SYNC CHECK] failed: {e}")
            # ===== END SYNC CHECK =====

            # get_xtele 会 exit sync；重新同步 active 臂，并将 free 同构臂下电
            sync_arms = (
                self.env.unwrapped.active_arm_names()
                if hasattr(self.env.unwrapped, "active_arm_names")
                else None
            )
            self.env.unwrapped.sync_xtele(timeout=2, arms=sync_arms)
            if hasattr(self.env.unwrapped, "_power_off_xtele_arms"):
                self.env.unwrapped._power_off_xtele_arms()
            self.env.unwrapped._enable_free_arm_hold = True
            if has_free and hasattr(self.env.unwrapped, "_free_arms_initialized"):
                self.env.unwrapped._free_arms_initialized = True

        info["is_intervention"] = False  # 设置干预标志为 False
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

        # shared_state.human_intervention_key = True
        intervened = shared_state.human_intervention_key  # 从共享状态获取干预标志

        if intervened:
            try:
                obs = self.env.unwrapped.get_xtele()
                xtele_joints, xtele_pose = obs['joints'], obs['pose']
                # free_*：干预只使用被控制的臂
                if self.free_left:
                    xtele_joints = {"right": xtele_joints["right"]}
                    xtele_pose = {"right": xtele_pose["right"]}
                elif self.free_right:
                    xtele_joints = {"left": xtele_joints["left"]}
                    xtele_pose = {"left": xtele_pose["left"]}
                
                if self.control_mode == "joint":
                    expert_a = xtele_joints
                else:
                    epsilon = 1e-6

                    if "tienkung" in self.robot_type or "tienyi" in self.robot_type: # mingbo debug
                        expert_a = []
                        for name, target_pose in xtele_pose.items():
                            curr_pose = self.env.unwrapped.currpos[name]
                            curr_matrix = self.pose2matrix(curr_pose)
                            tar_matrix = self.pose2matrix(target_pose)
                            T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)
                            rel_rot = Rotation.from_matrix(T_diff_matrix[:3, :3]).as_euler("xyz")  # 相对旋转（欧拉角）
                            rel_pos = T_diff_matrix[:3, 3]  # 相对位置
                            
                            tmp_a = np.zeros(7, dtype=np.float64)  # ⚠️ 硬编码 7 维
                            tmp_a[:3] = rel_pos / self.env.unwrapped.action_scale[0]  # 位置增量（归一化）
                            tmp_a[3:6] = rel_rot / self.env.unwrapped.action_scale[1]  # 旋转增量（归一化）
                            tmp_a[6:] = xtele_joints[name][-1] / self.env.unwrapped.action_scale[2]  # 夹爪（归一化）
                            tmp_a = np.clip(tmp_a, [-1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, 0.0], [1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0])
                            
                            expert_a += tmp_a.tolist()
                    
                        expert_a = np.array(expert_a)
                    else:
                        curr_matrix = self.pose2matrix(self.env.unwrapped.currpos)
                        tar_matrix = self.pose2matrix(xtele_pose)
                        
                        # 2. 计算相对变换矩阵（从当前位姿到目标位姿）
                        T_diff_matrix = np.dot(np.linalg.inv(curr_matrix), tar_matrix)
                        
                        # 3. 提取相对旋转和相对位置
                        rel_rot = Rotation.from_matrix(T_diff_matrix[:3, :3]).as_euler("xyz")  # 相对旋转（欧拉角）
                        rel_pos = T_diff_matrix[:3, 3]  # 相对位置
                        
                        # 4. 构建动作数组（归一化到 [-1, 1]）
                        # ⚠️【机器人构型相关】硬编码动作维度为 7，应使用环境动作空间维度
                        expert_a = np.zeros(7, dtype=np.float64)  # ⚠️ 硬编码 7 维
                        expert_a[:3] = rel_pos / self.env.unwrapped.action_scale[0]  # 位置增量（归一化）
                        expert_a[3:6] = rel_rot / self.env.unwrapped.action_scale[1]  # 旋转增量（归一化）
                        expert_a[6:] = xtele_joints[-1] / self.env.unwrapped.action_scale[2]  # 夹爪（归一化）
                        
                        # 5. 裁剪到 [-1, 1] 范围
                        # ⚠️ 硬编码 7 维的上下界
                        expert_a = np.clip(expert_a, [-1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, -1.0+epsilon, 0.0], [1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0-epsilon, 1.0])


                return expert_a, xtele_joints, True  # 返回干预动作、遥操作关节、替换标志
            except Exception as e:
                print(f"Error in action: {e}")
                print(f"[{type(e).__name__}] {e!r}")
                traceback.print_exc()          # full stacktrace
                sys.exit(1)
        return action, None, False

    def step(self, action):
        """
        执行一步动作 - 如果启用人工干预，则使用干预动作
        
        功能：
        1. 检查是否需要人工干预
        2. 如果需要，替换动作为干预动作
        3. 执行动作
        4. 如果未干预，同步遥操作设备位置
        
        参数：
            action: 策略输出的动作
        
        返回：
            (obs, reward, terminated, truncated, info) 元组
        """
        # 处理动作（可能被替换为干预动作）
        start_time = time.time()
        action, xtele_joints, replaced = self.action(action)
        end_time = time.time()
        # print("human intervention action time:", end_time - start_time)
        # pose 模式下冻结 pitch/yaw 等（pose_action_mask），保证执行与记录一致
        unwrapped = self.env.unwrapped
        if (
            self.control_mode == "pose"
            and hasattr(unwrapped, "apply_pose_action_mask")
            and not isinstance(action, dict)
        ):
            action = unwrapped.apply_pose_action_mask(action)
        if replaced:
            # 如果被干预，使用干预动作
            obs, rew, terminated, truncated, info = self.env.step(action, xtele_joints=xtele_joints, intervention_mode=self.intervention_mode) # joint, pose
            info["intervene_action"] = action  # 记录干预动作（用于后续处理）
        else:
            # 如果未干预，使用原始动作，并同步遥操作设备位置
            obs, rew, terminated, truncated, info = self.env.step(action)
            # 仅同步 active 臂；free 臂同构保持下电
            sync_arms = None
            # if hasattr(self.env.unwrapped, "active_arm_names"):
            #     sync_arms = self.env.unwrapped.active_arm_names()
            # sync_st = time.perf_counter()
            self.env.unwrapped.sync_xtele(timeout=0.1, arms=sync_arms)
            # print(f"----------> sync_xtele_time: {time.perf_counter() - sync_st}s")

            # if hasattr(self.env.unwrapped, "_power_off_xtele_arms"):
            #     power_st = time.perf_counter()
            #     self.env.unwrapped._power_off_xtele_arms()

        # line_profiler.disable_by_count()
        # line_profiler.print_stats()

        info["is_intervention"] = replaced  # 设置干预标志
        return obs, rew, terminated, truncated, info


_FORCE_DIM_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


class EndEffectorForceInterventionWrapper(gym.Wrapper):
    """末端力/力矩超限时触发人类干预。

    - 上下限来自 task yaml 的 ``ee_force_limit_low/high``（每臂 6 维 Fx..Tz）
    - 每步结束后读 ``env.unwrapped.curr_ee_force``（由 ``_update_currpos`` 缓存）
    - 超限：置 ``shared_state.human_intervention_key = True``，下一拍由
      ``HumanIntervention`` 接管；本步已执行的动作无法撤回
    - 力回到范围内不会自动取消干预（需人工 ScrollLock 关闭）
    """

    FORCE_DIM = 6

    def __init__(self, env, config=None):
        super().__init__(env)
        if config is None:
            config = getattr(env.unwrapped, "config", None)
        low_cfg = getattr(config, "ee_force_limit_low", None) if config is not None else None
        high_cfg = getattr(config, "ee_force_limit_high", None) if config is not None else None
        if low_cfg is None and isinstance(config, dict):
            low_cfg = config.get("ee_force_limit_low")
            high_cfg = config.get("ee_force_limit_high")
        if low_cfg is None or high_cfg is None:
            raise ValueError(
                "EndEffectorForceInterventionWrapper requires ee_force_limit_low/high"
            )
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(low_cfg):
                low_cfg = OmegaConf.to_container(low_cfg, resolve=True)
            if OmegaConf.is_config(high_cfg):
                high_cfg = OmegaConf.to_container(high_cfg, resolve=True)
        except Exception:
            pass

        force_limits: dict = {}
        if hasattr(low_cfg, "keys") or isinstance(low_cfg, dict):
            for name in low_cfg.keys():
                if name not in high_cfg:
                    raise ValueError(f"ee_force_limit_high missing arm '{name}'")
                lo = np.asarray(low_cfg[name], dtype=np.float64).reshape(-1)
                hi = np.asarray(high_cfg[name], dtype=np.float64).reshape(-1)
                if lo.shape[0] != 6 or hi.shape[0] != 6:
                    raise ValueError(
                        f"ee_force_limit for '{name}' must be 6-dim [Fx,Fy,Fz,Tx,Ty,Tz], "
                        f"got low={lo.shape}, high={hi.shape}"
                    )
                if np.any(lo > hi):
                    raise ValueError(
                        f"ee_force_limit for '{name}': low > high on dims "
                        f"{np.where(lo > hi)[0].tolist()}"
                    )
                force_limits[str(name)] = (lo, hi)
        else:
            lo = np.asarray(low_cfg, dtype=np.float64).reshape(-1)
            hi = np.asarray(high_cfg, dtype=np.float64).reshape(-1)
            if lo.shape[0] != 6 or hi.shape[0] != 6:
                raise ValueError("ee_force_limit list form must be 6-dim")
            force_limits["single"] = (lo, hi)

        self.force_limits = force_limits
        self._pending_force_violation = False
        self.last_force_check = None
        print(
            "[EndEffectorForceIntervention] arms="
            + ", ".join(
                f"{name}: lo={lo.round(3).tolist()} hi={hi.round(3).tolist()}"
                for name, (lo, hi) in self.force_limits.items()
            )
        )

    def _arms_to_check(self) -> list:
        active = self.env.unwrapped.active_arm() if hasattr(self.env.unwrapped, "active_arm") else None
        if active is not None:
            if active in self.force_limits:
                return [active]
            # active 未配置 limits：不检查（避免误用 free 臂力）
            return []
        return list(self.force_limits.keys())

    def _read_ee_force(self) -> Optional[dict]:
        force = getattr(self.env.unwrapped, "curr_ee_force", None)
        if force is None:
            return None
        print_green(f"force: {force}")
        print_green(f"-----------------------todo debug here-----------------------")
        if isinstance(force, dict):
            print_green(f"force is a dict: {force}")
            return {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in force.items()}

    def check_force(self, force_dict: Optional[dict] = None):
        """返回 (violated: bool, detail: dict)。"""
        if force_dict is None:
            force_dict = self._read_ee_force()
        detail = {
            "violated": False,
            "arms": {},
            "checked_arms": self._arms_to_check(),
        }
        if force_dict is None:
            detail["error"] = "curr_ee_force unavailable"
            self.last_force_check = detail
            return False, detail

        violated = False
        for name in detail["checked_arms"]:
            if name not in force_dict:
                detail["arms"][name] = {"error": "missing force reading"}
                continue
            if name not in self.force_limits:
                continue
            lo, hi = self.force_limits[name]
            f = force_dict[name]
            if f.shape[0] < self.FORCE_DIM:
                detail["arms"][name] = {"error": f"force dim {f.shape[0]} < 6"}
                continue
            f6 = f[: self.FORCE_DIM]
            below = f6 < lo
            above = f6 > hi
            arm_violated = bool(np.any(below) | np.any(above))
            violated = violated or arm_violated
            dims = []
            for i, dim in enumerate(_FORCE_DIM_NAMES):
                if below[i] or above[i]:
                    dims.append(
                        {
                            "name": dim,
                            "value": float(f6[i]),
                            "low": float(lo[i]),
                            "high": float(hi[i]),
                        }
                    )
            detail["arms"][name] = {
                "force": f6.tolist(),
                "violated": arm_violated,
                "out_of_range": dims,
            }
        detail["violated"] = violated
        self.last_force_check = detail
        return violated, detail

    def reset(self, **kwargs):
        self._pending_force_violation = False
        obs, info = self.env.reset(**kwargs)
        violated, detail = self.check_force()
        info["force_violation"] = bool(violated)
        info["force_violation_detail"] = detail
        # reset 后若已超限，立即请求干预（下一步 / 首步策略前）
        if violated:
            shared_state.human_intervention_key = True
            self._pending_force_violation = True
            print(
                "[EndEffectorForceIntervention] force out of range at reset → "
                "request human intervention"
            )
            for name, arm in detail.get("arms", {}).items():
                for d in arm.get("out_of_range", []):
                    print(
                        f"  {name} {d['name']}: {d['value']:.4f} "
                        f"not in [{d['low']:.4f}, {d['high']:.4f}]"
                    )
        return obs, info

    def step(self, action):
        # 上一步超限：本步强制打开人类干预（HumanIntervention 在内层读 flag）
        if self._pending_force_violation:
            shared_state.human_intervention_key = True

        obs, reward, terminated, truncated, info = self.env.step(action)

        violated, detail = self.check_force()
        info["force_violation"] = bool(violated)
        info["force_violation_detail"] = detail
        if violated:
            info["violation"] = True
            self._pending_force_violation = True
            shared_state.human_intervention_key = True
            print_green(
                "[EndEffectorForceIntervention] force out of range → "
                "human intervention ON"
            )
            for name, arm in detail.get("arms", {}).items():
                for d in arm.get("out_of_range", []):
                    print(
                        f"  {name} {d['name']}: {d['value']:.4f} "
                        f"not in [{d['low']:.4f}, {d['high']:.4f}]"
                    )
        else:
            self._pending_force_violation = False
            # 不自动关闭 human_intervention_key，由人工 ScrollLock 退出

        return obs, reward, terminated, truncated, info


class AugmentedObservationWrapper(gym.ObservationWrapper):
    """
    观察增强包装器 - 对图像进行裁剪和缩放
    
    功能：
    1. 对图像进行 ROI 裁剪（如果配置了 image_crop）
    2. 将图像缩放到目标尺寸
    3. 保持观察空间的其他部分不变
    
    使用场景：
    - 减少计算量：只处理图像的重要区域
    - 标准化图像尺寸：确保所有图像具有相同的尺寸
    - 聚焦重要区域：裁剪掉不相关的背景
    """
    def __init__(self, env):
        """
        初始化观察增强包装器
        
        参数：
            env: 被包装的环境
        """
        super().__init__(env)
        self.observation_space = env.observation_space  # 保持观察空间不变
        self.env = env

    def observation(self, obs):
        """
        处理观察 - 对图像进行裁剪和缩放
        
        功能：
        1. 提取图像字典
        2. 对每个图像进行裁剪（如果配置了）
        3. 将图像缩放到目标尺寸
        4. 更新观察中的图像
        
        参数：
            obs: 原始观察字典
        
        返回：
            处理后的观察字典
        """
        t0 = time.time()
        images = obs['images']  # 提取图像字典
        env = self.env.unwrapped  # 获取底层环境
        
        # 对每个图像进行处理
        for key, img in images.items():
            # 1. 裁剪图像（如果配置了裁剪函数）
            if hasattr(env, 'image_crop'):
                cropped_rgb = env.image_crop[key](img) if key in env.image_crop else img
            else:
                cropped_rgb = img
            
            # 2. 缩放图像到目标尺寸
            # shape[:2][::-1] 表示 (height, width)，cv2.resize 需要 (width, height)
            cropped_rgb = cv2.resize(
                cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = cropped_rgb  # 更新图像

        # augment_time = time.time() - t0
        # print(f"augment_time: {augment_time}s ... ")
        return obs
    
    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info


class PoseActionPadWrapper(gym.Wrapper):
    """将 policy 的 compact 动作按 pose_action_mask 补齐为 env 全维动作。

    pose_action_mask: [x, y, z, roll, pitch, yaw]（不含 gripper）
    每臂 env 动作: [x, y, z, roll, pitch, yaw, gripper]

    - mask 只作用于 xyzrpy；gripper 不在 mask 范围
    - ``fix_gripper=True``：对外动作 = 仅 mask=1 的连续维
    - ``fix_gripper=False``：对外动作 = 连续维 + 末尾 1 维 gripper
      （与 policy ``num_discrete_actions`` 拼到 action 末尾一致）
    - ``step(compact)`` → 按 mask 散射到全维，再交给内层
    - 干预时 ``info['intervene_action']`` 压回同一 compact 维
    """

    POSE_DIM = 6
    ARM_DIM = 7  # xyzrpy + gripper
    GRIPPER_IDX = 6  # 每臂内 gripper 下标

    def __init__(self, env):
        super().__init__(env)
        unwrapped = env.unwrapped
        mask = getattr(unwrapped, "pose_action_mask", None)
        if mask is None:
            mask = np.ones(self.POSE_DIM, dtype=np.float32)
        self.pose_action_mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        if self.pose_action_mask.shape[0] != self.POSE_DIM:
            raise ValueError(
                f"pose_action_mask must have {self.POSE_DIM} entries, got {self.pose_action_mask.shape[0]}"
            )
        self.enabled_idx = np.flatnonzero(self.pose_action_mask > 0.5)
        self.n_arms = 2 if getattr(unwrapped, "dual_arm", False) else 1
        self.env_action_dim = int(np.prod(env.action_space.shape))
        expected = self.ARM_DIM * self.n_arms
        if getattr(unwrapped, "control_mode", "pose") != "pose":
            self._passthrough = True
            self.include_gripper = False
            self.cont_dim = self.env_action_dim
            self.policy_action_dim = self.env_action_dim
            return
        if self.env_action_dim != expected:
            raise ValueError(
                f"PoseActionPadWrapper expects env action dim {expected}, got {self.env_action_dim}"
            )
        self._passthrough = False
        # gripper 不在 mask 内：由 fix_gripper 决定是否进入对外动作维
        self.include_gripper = not bool(getattr(unwrapped, "fix_gripper", False))
        self.cont_dim = int(len(self.enabled_idx) * self.n_arms)
        self.policy_action_dim = self.cont_dim + (1 * self.n_arms if self.include_gripper else 0)
        low = np.full(self.cont_dim, -1.0, dtype=np.float32)
        high = np.full(self.cont_dim, 1.0, dtype=np.float32)
        if self.include_gripper:
            low = np.concatenate([low, np.array([0.0], dtype=np.float32)])
            high = np.concatenate([high, np.array([1.0], dtype=np.float32)])
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        print(
            f"[PoseActionPadWrapper] mask={self.pose_action_mask.tolist()} "
            f"enabled_idx={self.enabled_idx.tolist()} "
            f"cont_dim={self.cont_dim} include_gripper={self.include_gripper} "
            f"policy_dim={self.policy_action_dim} -> env_dim={self.env_action_dim}"
        )

    def expand_action(self, action: np.ndarray) -> np.ndarray:
        """compact (+ trailing gripper if not fix_gripper) → full env action."""
        if self._passthrough:
            return np.asarray(action, dtype=np.float32)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] < self.cont_dim:
            raise ValueError(
                f"PoseActionPadWrapper: action dim {a.shape[0]} < cont_dim {self.cont_dim}"
            )
        cont = a[: self.cont_dim]
        gripper = a[self.cont_dim :] if a.shape[0] > self.cont_dim else None
        full = np.zeros(self.env_action_dim, dtype=np.float32)
        offset = 0
        n = len(self.enabled_idx)
        for arm in range(self.n_arms):
            base = arm * self.ARM_DIM
            full[base + self.enabled_idx] = cont[offset : offset + n]
            offset += n
        # gripper 不走 mask：写入每臂 gripper 槽；单离散时各臂同值
        if self.include_gripper and gripper is not None and gripper.size > 0:
            if gripper.size >= self.n_arms:
                for arm in range(self.n_arms):
                    full[arm * self.ARM_DIM + self.GRIPPER_IDX] = float(gripper[arm])
            else:
                g = float(gripper[0])
                for arm in range(self.n_arms):
                    full[arm * self.ARM_DIM + self.GRIPPER_IDX] = g
        # 安全：再次强制 mask=0 的 xyzrpy 维为 0（不动 gripper）
        if hasattr(self.env.unwrapped, "apply_pose_action_mask"):
            full = self.env.unwrapped.apply_pose_action_mask(full)
        else:
            for arm in range(self.n_arms):
                base = arm * self.ARM_DIM
                full[base : base + self.POSE_DIM] *= self.pose_action_mask
        return full

    def compact_action(self, action: np.ndarray) -> np.ndarray:
        """full env action → compact（连续维 [+ gripper]）。"""
        if self._passthrough:
            return np.asarray(action, dtype=np.float32)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        parts = []
        for arm in range(self.n_arms):
            base = arm * self.ARM_DIM
            parts.append(a[base + self.enabled_idx])
        if self.include_gripper:
            # 与 policy 单离散维对齐：取最后一臂 gripper；多臂且需保留双爪时可传满 n_arms
            grips = [
                a[arm * self.ARM_DIM + self.GRIPPER_IDX] for arm in range(self.n_arms)
            ]
            parts.append(np.asarray(grips[-1:], dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def step(self, action):
        if self._passthrough:
            return self.env.step(action)
        full = self.expand_action(action)
        obs, reward, terminated, truncated, info = self.env.step(full)
        # 干预动作压回 compact，与 policy / dataset 维一致（含非 fix_gripper 时的 gripper）
        if info.get("is_intervention") and "intervene_action" in info:
            ia = info["intervene_action"]
            if not isinstance(ia, dict):
                info["intervene_action"] = self.compact_action(ia)
                info["intervene_action_full"] = np.asarray(ia, dtype=np.float32)
        return obs, reward, terminated, truncated, info


class Quat2EulerWrapper(gym.ObservationWrapper):

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

 
class SERLObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, proprio_keys=None, use_force=False):
        super().__init__(env)
        
        # ========== 选择本体感觉键 ==========
        if use_force:
            # 如果使用力传感器，包含所有指定的键
            self.proprio_keys = proprio_keys
        else:
            # 如果不使用力传感器，只使用前两个键（通常是 tcp_pose 和 gripper_pose）
            self.proprio_keys = proprio_keys[:2]

        print("proprio_keys:", self.proprio_keys)  

        # 如果未指定，使用所有可用的状态键
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        # ========== 定义本体感觉空间 ==========
        # 使用 OrderedDict 保持键的顺序
        self.proprio_space = gym.spaces.Dict(
            OrderedDict((key, self.env.observation_space["state"][key]) for key in self.proprio_keys)
        )
        
        # ========== 定义新的观察空间 ==========
        # 状态部分：扁平化为单个向量
        # 图像部分：保持不变
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),  # 扁平化状态空间
                **(self.env.observation_space["images"]),  # 保持图像空间不变
            }
        )

    def observation(self, obs):
        # t0 = time.time()
        from collections import OrderedDict
        # 扁平化状态：将选定的状态键组合成单个向量
        obs = {
            "state": flatten(
                self.proprio_space,  # 状态空间定义
                OrderedDict((key, obs["state"][key]) for key in self.proprio_keys),  # 选定的状态键
            ),
            **(obs["images"]),  # 保持图像不变
        }
        # flatten_time = time.time() - t0
        # print(f"flatten_time: {flatten_time}s ... ")
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info


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
        dual_arm=False,
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
        self.dual_arm = dual_arm

    def observation(self, obs):
        from cosmos_policy.experiments.robot.cosmos_utils import (
            prepare_images_for_model,
        )
        from cosmos_policy.utils.utils import duplicate_array

        with torch.inference_mode():
            if self.dual_arm:
                print("obs keys:", obs.keys())
                print("obs left:", obs["left"].shape)
                print("obs right:", obs["right"].shape)
                print("obs head:", obs["head"].shape)
                policy_obs = [
                    obs["head"],
                    obs["left"],
                    obs["right"],
                ]
                left_wrist_image = policy_obs[1]
                right_wrist_image = policy_obs[2]
                wrist_image = np.concatenate([left_wrist_image, right_wrist_image], axis=0)
                print("wrist_image shape:", wrist_image.shape)
                policy_obs = [
                    obs["head"],
                    wrist_image,
                ]
            else:
                policy_obs = [
                    obs["right"],
                    obs["wrist"],
                ]
                
            policy_obs = prepare_images_for_model(policy_obs, self.cosmos_cfg)
            proprio = obs["state"]
  
            # 构造序列
            image_sequence = []
            current_sequence_idx = 0  # Used to track which index in the sequence of images we are on
            
            # Add blank placeholder image (special placeholder for 1+T temporal VAE compression)
            # 1. 空白帧
            primary_image = policy_obs[0]
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
            wrist_image = policy_obs[1]
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
