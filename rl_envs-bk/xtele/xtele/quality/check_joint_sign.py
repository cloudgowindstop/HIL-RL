import tomli as tomllib
import os
import mujoco
import mujoco.viewer as mviewer

import tomli_w


import threading

import numpy as np
from pynput import keyboard
from xtele.common.common import Path
from xtele.core.integrate_module import TeleCore

ACT_MODE = 0
SYNC_MODE = 1

AGENT_MODE = {
    "ur": ["arm"]
}

ROBOT_SIM_STANDARD = {
    "ur": np.array([1, 1, 1, 1, 1, -1]),
}  # (仿真中)统一旋转规则，关节角度旋转的方向（增减情况）

ARMS_DIRECTION = {
    "ur": np.zeros(6),
}

HOME_POSE = {
    "ur": np.zeros(6),  # 6 joints for UR robot
}

ORIGIN_JOINT_SIGNS = {
    "ur": np.ones(6),
}

JOINT_INDICES = {
    "ur": list(range(0, 6)),  # UR arm joint indices
}

MODEL_PATH = {
    "ur": "model/ur_description/urdf/ur5_robot.xml"
}

class MujoCoBase:
    def __init__(self, model_path: str, args: str = "ur"):
        current_script = os.path.realpath(__file__)
        script_dir = os.path.dirname(current_script)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(script_dir, model_path))
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = HOME_POSE[args].copy()
        self.target_qpos = HOME_POSE[args].copy()
        self.arm_joint_indices = JOINT_INDICES[args]

        if not os.path.exists(Path.CONFIG_DIR):
            raise RuntimeError(
                "The configuration file does not exist, please create it."
            )
        with open(Path.CONFIG_DIR, "rb") as f:
            config = tomllib.load(f)

        self.tele_agent = TeleCore()
        self.mode = ACT_MODE
        self.args = args

        self.btn_enter = True
        self.btn_exit = False
        self.flag_rewrite = False

        if args == "ur":
            ORIGIN_JOINT_SIGNS[args] = config["xlinker"]["single"]["robot"]["joint_signs"]  #

        print(f"Default joint_signs: {ORIGIN_JOINT_SIGNS[args]}")

    def start_keyboard_listener(self) -> None:
        """Start the keyboard listener."""

        def _listener():
            with keyboard.Listener(on_press=self.on_press) as listener:
                listener.join()

        listener_thread = threading.Thread(target=_listener, daemon=True)
        listener_thread.start()

    def on_press(self, key) -> None:
        """Handle the keyboard press event."""
        try:
            if key == keyboard.Key.enter:
                self.btn_enter = False
            if key == keyboard.Key.esc:
                self.btn_exit = True
        except AttributeError:
            Warning("Keyboard AttributeError")

    def step(self, data):
        if not hasattr(self, "data") or self.data is None:
            self.data = mujoco.MjData(self.model)

        # 将关节角度赋值给 data.qpos（位置状态）
        # 注意：要根据实际机器人的关节结构调整索引范围
        self.data.qpos[self.arm_joint_indices] = data

        mujoco.mj_step(self.model, self.data)
        self.data.qvel = np.zeros(len(self.data.qvel))
        return self.data.qpos, self.data.qvel

    def check_franka_ur(self, robot_correct_sign):
        self.arm_agent = self.tele_agent.tele_agent._equips["dynamixel_arm"]
        
        ids = self.arm_agent._robot._joint_ids
        for id in ids:
            joint_pos = self.tele_agent.act_dict()[
                f"single_{id}"
            ]

            print(f"{AGENT_MODE[self.args]}关节{id}向下(左视图顺时针方向)/右旋(主视图逆时针方向)转后，按下Enter键")
            while self.btn_enter:
                robot_targets = self.tele_agent.act()
                self.set_arm_targets(robot_targets[:-1]) # no gripper
                self.step(self.target_qpos)
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                if self.btn_exit:
                    self.tele_agent.close()
                    return

            joint_pos_delta = (
                self.tele_agent.act_dict()[f"single_{id}"]
                - joint_pos
            )  #
            robot_correct_sign[id] = joint_pos_delta/abs(joint_pos_delta)
            if robot_correct_sign[id] != ROBOT_SIM_STANDARD[self.args][id]:
                ORIGIN_JOINT_SIGNS[self.args][id] *= -1
                self.flag_rewrite = True

            self.btn_enter = True
        return

    def run(self):
        self.start_keyboard_listener()
        
        with mviewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer

            print(f"{AGENT_MODE[self.args]}")
            while self.btn_enter:
                robot_targets = self.tele_agent.act()
                self.set_arm_targets(robot_targets[:-1]) # no gripper
                self.step(self.target_qpos)
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                if self.btn_exit:
                    self.tele_agent.close()
                    return
            self.btn_enter = True
            
            while not viewer.is_running():
                continue
            
            robot_correct_sign = ARMS_DIRECTION[self.args]
            if self.args == "ur":
                print("开始检查Franka/ur臂关节方向")
                self.check_franka_ur(robot_correct_sign)
            else:
                print(f"未知的机器人类型: {self.args}")
                return

            if self.flag_rewrite:
                self.write_config()
            else:
                print("Correct. ")

    def write_config(self):
        print(f"joint_signs: {ORIGIN_JOINT_SIGNS[self.args]}")

        if self.args == "ur":
            pass

        with open(Path.CONFIG_DIR, "wb") as f:
            tomli_w.dump(self.tele_agent.config, f)

    def set_arm_targets(self, arm_targets: np.ndarray):
        if arm_targets.size != len(self.arm_joint_indices):
            print(
                f"错误: 目标维度不匹配. 预期 {len(self.arm_joint_indices)}, 实际 {arm_targets.size}"
            )
            return

        # 只更新手臂关节的目标位置
        self.target_qpos[self.arm_joint_indices] = arm_targets

    def decompose_action(self, action: dict) -> dict:
        left_joints = np.zeros(7)
        right_joints = np.zeros(7)
        for i in range(7):
            left_joints[i] = action[f"left_joint_{i}"]
            right_joints[i] = action[f"right_joint_{i}"]

        return {"arm_joints": {"robot": np.concatenate([left_joints, right_joints])}}
