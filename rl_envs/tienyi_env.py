"""Tienkung Environment - Specialized environment for Tienkung dual-arm robot"""
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from xrocs.core.config_loader import ConfigLoader
from xrocs.core.station_loader import StationLoader
from rl_envs.base_env import BaseEnv

def print_green(x: any) -> None:
    return print("\033[92m {}\033[00m".format(x))


class TienYiEnv(BaseEnv):
    """
    Specialized environment for TienYi dual-arm robot.
    Inherits from BaseEnv and overrides specific methods for Tienkung robot.
    """
    
    def __init__(
        self,
        fake_env=False,
        config=None,
    ):
        """
        Initialize TienYi environment.
        
        Args:
            fake_env: If True, don't connect to real robot
            config: Configuration object containing robot parameters
        """
        #         self.config = config
        self._gripper_sleep = config.gripper_sleep
        self.joint_dim = config.joint_dim
        self.dual_arm = config.dual_arm
        self.free_left = config.free_left
        self.free_right = config.free_right
        # true: 第一次 reset 可用同构臂操控 free 臂；false: free 臂直接走到 reset 位姿
        self.free_arm_teleop = bool(getattr(config, "free_arm_teleop", True))
        if self.free_left or self.free_right:
            assert not self.dual_arm, (
                "free_left/free_right requires dual_arm=false so action dims match the active arm"
            )
            assert not (self.free_left and self.free_right), (
                "free_left and free_right cannot both be true"
            )
        if not self.dual_arm:
            assert self.free_left or self.free_right, (
                "tienyi with dual_arm=false requires free_left or free_right"
            )
        # free_*：仅第一次 reset 允许同构控制 free 臂（需 free_arm_teleop=true）；之后 hold + 同构下电
        self._enable_free_arm_hold = False
        self._free_arms_initialized = False

        # tienyi 始终保留 left/right 的 reset 关节（free 臂在 step 中保持该位姿）
        if hasattr(config.reset_joint, "keys"):
            self._reset_joint = {
                name: np.array(config.reset_joint[name])[0 : self.joint_dim]
                for name in config.reset_joint.keys()
            }
        else:
            self._reset_joint = np.array(config.reset_joint)[0 : self.joint_dim]

        # 可选：reset 时设定夹爪。free_right → reset_hands.right；free_left → reset_hands.left
        self._reset_hands = self._parse_reset_hands(config)

        self._random_xy_range = config.random_xy_range
        self._random_rz_range = config.random_rz_range
        self._random_reset = config.random_reset
        self._bgr2rgb = config.bgr2rgb
        self._image_keys = config.image_keys
        print("in TienYI self._image_keys:", self._image_keys)
        self.robot_type = config.robot_type
        self.action_scale = config.action_scale
        self.max_episode_length = config.max_episode_length
        self.control_mode = config.control_mode
        self.close_gripper = config.close_gripper
        self.fix_gripper = config.fix_gripper
        self.ego_mode = config.ego_mode
        self.use_cmd_pose = config.use_cmd_pose
        self.pose_action_mask = self._parse_pose_action_mask(config)
        
        assert self.control_mode in ["joint", "pose"], f'Not valid control mode: {self.control_mode}'
        
        self.fake_env = fake_env
        self.hz = config.hz
        # true: xrocs ros2controller；false: 原生 ros2 topic（状态仍走 xrocs station）
        self.use_ros2controller = bool(getattr(config, "use_ros2controller", True))

        xyz_limits = {}
        rpy_limits = {}

        for arm_name in config.abs_pose_limit_low.keys():
            low_vals = np.array(config.abs_pose_limit_low[arm_name])
            high_vals = np.array(config.abs_pose_limit_high[arm_name])

            # XYZ limits (前3维)
            xyz_limits[arm_name] = gym.spaces.Box(
                low=low_vals[:3],
                high=high_vals[:3],
                dtype=np.float64
            )

            # RPY limits (后3维)
            rpy_limits[arm_name] = gym.spaces.Box(
                low=low_vals[3:],
                high=high_vals[3:],
                dtype=np.float64
            )

        self.xyz_bounding_box = gym.spaces.Dict(xyz_limits)
        self.rpy_bounding_box = gym.spaces.Dict(rpy_limits)

        image_resize = config.image_resize
        tcp_pose_dim = 7  # 末端执行器位姿：xyz 位置(3) + 四元数旋转(4) = 7维
        if self.dual_arm:
            tcp_pose_dim = 2 * tcp_pose_dim  # 双臂，每臂7维
        
        gripper_dim = 1
        if self.dual_arm:
            gripper_dim = 2 * gripper_dim
        
        joint_dim = self.joint_dim
        if self.dual_arm:
            joint_dim = 2 * joint_dim
        
        state_dict = {
            "tcp_pose": gym.spaces.Box(
                -np.inf, np.inf, shape=(tcp_pose_dim,)
            ),  
            "gripper_pose": gym.spaces.Box(0, 1, shape=(gripper_dim,)),  # 夹爪开合度：0=完全打开，1=完全关闭
            "joints": gym.spaces.Box(
                -np.inf, np.inf, shape=(joint_dim,)
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
            if self.dual_arm:
                # action_space = left(xyz + rpy + gripper) + right(xyz + rpy + gripper)
                self.action_space = gym.spaces.Box( 
                    np.array([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            else:
                # action_space = (xyz + rpy + gripper)
                self.action_space = gym.spaces.Box(
                    np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            self._apply_mask_to_action_space()
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(joint_dim + gripper_dim,))

        if fake_env:
            print_green("fake env : not connect to robot")
            return 

        cfg_loader = ConfigLoader("/home/ubuntu/Documents/configuration.toml")  # 加载配置文件
        self.cfg_dict = cfg_loader.get_config()  # 获取配置字典
        # topic 模式不初始化 motion controller，避免依赖 /EAIHardware/set_arm_enable
        if not self.use_ros2controller:
            robot_ctrl = self.cfg_dict.setdefault("robot", {}).setdefault("controller", {})
            robot_ctrl["enable_direct_motion_controller"] = False
            robot_ctrl["enable_moveit_motion_controller"] = False
            print_green(
                "[TienYiEnv] use_ros2controller=false → disable direct/moveit MC; "
                "commands via ros2 topics, state via xrocs"
            )

        station_loader = StationLoader(self.cfg_dict)  # 创建站点加载器
        self.robot_station = station_loader.generate_station_handle()  # 生成机器人站点句柄
        self.robot_station.connect()  # 连接机器人硬件
        robot = self.robot_station.get_robot_handle()["robot"]

        self._active_controllers: list[str] | None = None
        self._cmd_pubs = {}
        self.ros2controller = None

        if self.use_ros2controller:
            self.ros2controller = robot.ros2controller
            # Crow HTTP(7890/7880) 在现场常未监听；走已就绪的 ROS2 service
            self.ros2controller.enable_arm(via_http=True, timeout_sec=3.0)
            self.ros2controller.set_arm_mode(3, via_http=True, timeout_sec=3.0)
            # joint → jointspace_*；pose → endpose_single_arm_qp_*
            # free 臂在 hold 开启后始终附带 jointspace，避免被误驱动
            self._ensure_controllers(self.control_mode, self.active_arm_names())
        else:
            # 只 pub topic；controller 需外部已激活，此处不调 controller_manager
            self._init_topic_command_interface(robot)

        self.left_gripper = self.robot_station.get_gripper_handle()["left"]
        self.right_gripper = self.robot_station.get_gripper_handle()["right"]

        if self.dual_arm and hasattr(config.abs_pose_limit_low, "keys"):
            xyz_limits = {}
            rpy_limits = {}

            for arm_name in config.abs_pose_limit_low.keys():
                low_vals = np.array(config.abs_pose_limit_low[arm_name])
                high_vals = np.array(config.abs_pose_limit_high[arm_name])

                # XYZ limits (前3维)
                xyz_limits[arm_name] = gym.spaces.Box(
                    low=low_vals[:3],
                    high=high_vals[:3],
                    dtype=np.float64
                )
                # RPY limits (后3维)
                rpy_limits[arm_name] = gym.spaces.Box(
                    low=low_vals[3:],
                    high=high_vals[3:],
                    dtype=np.float64
                )

            self.xyz_bounding_box = gym.spaces.Dict(xyz_limits)
            self.rpy_bounding_box = gym.spaces.Dict(rpy_limits)
        else:
            # 单臂 / free_*：limits 可为 list，或取 active 臂的 dict 项
            low = config.abs_pose_limit_low
            high = config.abs_pose_limit_high
            active = self.active_arm()
            if active is not None and hasattr(low, "keys") and active in low:
                low = low[active]
                high = high[active]
            self.xyz_bounding_box = gym.spaces.Box(
                np.array(low[:3]),
                np.array(high[:3]),
                dtype=np.float64,
            )
            self.rpy_bounding_box = gym.spaces.Box(
                np.array(low[3:]),
                np.array(high[3:]),
                dtype=np.float64,
            )
        
        
        self.image_crop = {}

        if hasattr(config, 'image_crop') and config.image_crop is not None:
            for camera_key, crop_func in config.image_crop.items():
                if callable(crop_func):
                    self.image_crop[camera_key] = crop_func
                else:
                    def make_crop_func(crop_params):
                        crop_params = list(crop_params)
                        if isinstance(crop_params, (list, tuple)) and len(crop_params) == 2:
                            h_range, w_range = crop_params
                            return lambda img: img[h_range[0]:h_range[1], w_range[0]:w_range[1]]
                        else:
                            return lambda img: img
                    
                    self.image_crop[camera_key] = make_crop_func(crop_func)

        # fields expected by BaseEnv methods
        self.episode_id = 0
        self.currpos = None
        self.trajectory_log = False
        self.trajectory_log_file = "trajectory_log.txt"
        self.replay_trajectory = getattr(config, "replay_trajectory", False)
        self.absolute_action = getattr(config, "absolute_action", False)
        self.last_gripper_value = {
            "left": 1.0 if self.close_gripper else 0.0,
            "right": 1.0 if self.close_gripper else 0.0,
        }
        self._apply_reset_hands_to_gripper()

    @staticmethod
    def _parse_reset_hands(config) -> dict | None:
        """有 reset_hands 则按配置锁定；无则不锁定 hand。"""
        try:
            from omegaconf import OmegaConf

            reset_hands = OmegaConf.select(
                config, "reset_hands", default=None, throw_on_resolution_failure=False
            )
        except Exception:
            reset_hands = getattr(config, "reset_hands", None)
        if reset_hands is None:
            return None
        if not hasattr(reset_hands, "keys"):
            return None
        parsed = {
            name: float(np.asarray(reset_hands[name]).flatten()[0])
            for name in reset_hands.keys()
        }
        return parsed or None

    def _apply_reset_hands_to_gripper(self) -> None:
        """仅当配置了 reset_hands 时，用其覆盖对应臂的 last_gripper_value。"""
        if not self._reset_hands:
            return
        if not isinstance(self.last_gripper_value, dict):
            return
        for name, val in self._reset_hands.items():
            self.last_gripper_value[name] = val

    def hands_locked(self) -> bool:
        """仅当存在 reset_hands，且第一次 free 操作结束后，才锁定 hand。"""
        return bool(self._reset_hands) and bool(
            getattr(self, "_free_arms_initialized", False)
        )

    def locked_hand_names(self) -> list:
        if not self.hands_locked():
            return []
        return list(self._reset_hands.keys())

    def close(self) -> None:
        if "tienyi" in self.robot_type and getattr(self, "tele_agent", None) is not None:
            self.tele_agent.close()
            self.tele_agent = None

    def _gripper_cmd_for_arm(self, name: str, include_gripper: bool, cmd_val=None, control_mode: str = "joint") -> list:
        """有 reset_hands 且已锁定时强制配置值；否则不锁定，走正常 gripper 逻辑。"""
        if name in self.locked_hand_names():
            val = float(self._reset_hands[name])
            self.last_gripper_value[name] = val
            return [val]
        if include_gripper and cmd_val is not None:
            # if control_mode == "joint":
            #     return [cmd_val]
            # elif control_mode == "pose":
            binary = 1.0 if float(cmd_val) >= 0.5 else 0.0
            self.last_gripper_value[name] = binary
            return [binary]
        return [float(self.last_gripper_value.get(name, 0.0))]

    # 与 xrocs TienYi2 mc_direct 示例一致：topic/API 与 controller 成对使用
    _ARM_CONTROLLERS = {
        "joint": {
            "left": "jointspace_arm_L_controller",
            "right": "jointspace_arm_R_controller",
        },
        "pose": {
            "left": "endpose_single_arm_qp_L_controller",
            "right": "endpose_single_arm_qp_R_controller",
        },
    }
    _TOPIC_ENDPOSE = {"left": "/endposetarget_L", "right": "/endposetarget_R"}
    _TOPIC_JOINTSPACE = {"left": "/jointspace_commands_L", "right": "/jointspace_commands_R"}

    def _init_topic_command_interface(self, robot) -> None:
        """不走 ros2controller 时，用 robot.node 创建 publisher。

        不在此处强制 switch_controllers（本机可能无 controller_manager）；
        首次发控制前由 _ensure_controllers 再确认/激活。
        """
        from std_msgs.msg import Float64MultiArray
        from controller_manager_msgs.srv import SwitchController, ListControllers

        node = getattr(robot, "node", None)
        if node is None:
            raise RuntimeError("robot.node is required for use_ros2controller=false")
        self._ros_node = node
        for arm, topic in self._TOPIC_ENDPOSE.items():
            self._cmd_pubs[f"endpose_{arm}"] = node.create_publisher(
                Float64MultiArray, topic, 10
            )
        for arm, topic in self._TOPIC_JOINTSPACE.items():
            self._cmd_pubs[f"joint_{arm}"] = node.create_publisher(
                Float64MultiArray, topic, 10
            )
        self._switch_controller_srv = "/controller_manager/switch_controller"
        self._list_controllers_srv = "/controller_manager/list_controllers"
        self._switch_controller_client = node.create_client(
            SwitchController, self._switch_controller_srv
        )
        self._list_controllers_client = node.create_client(
            ListControllers, self._list_controllers_srv
        )
        self._topic_controllers_user_confirmed = False
        print_green(
            "[TienYiEnv] topic pubs ready: "
            f"{list(self._TOPIC_ENDPOSE.values()) + list(self._TOPIC_JOINTSPACE.values())}"
        )

    def _format_switch_controllers_cli(self, activate: list[str], deactivate: list[str] | None = None) -> str:
        """生成与手工命令等价的提示。"""
        parts = ["ros2 control switch_controllers"]
        if activate:
            parts.append("--activate")
            parts.extend(activate)
        if deactivate:
            parts.append("--deactivate")
            parts.extend(deactivate)
        return " ".join(parts)

    def _list_controller_states(self, timeout_sec: float = 3.0) -> dict[str, str]:
        """返回 {controller_name: state}，如 active/inactive。"""
        from controller_manager_msgs.srv import ListControllers

        client = getattr(self, "_list_controllers_client", None)
        if client is None:
            raise RuntimeError("ListControllers client not initialized")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"service unavailable: {self._list_controllers_srv} "
                "(controller_manager 未起来？)"
            )
        future = client.call_async(ListControllers.Request())
        result = future.result(timeout=10.0)
        return {c.name: c.state for c in result.controller}

    def _switch_controllers_topic_mode(
        self, activate: list[str], deactivate: list[str] | None = None, timeout_sec: float = 3.0
    ) -> bool:
        """等价 ros2 control switch_controllers --activate/--deactivate。"""
        from controller_manager_msgs.srv import SwitchController

        client = getattr(self, "_switch_controller_client", None)
        if client is None:
            raise RuntimeError("SwitchController client not initialized")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"service unavailable: {self._switch_controller_srv} "
                "(controller_manager 未起来？请先 bringup ros2_control)"
            )
        req = SwitchController.Request()
        req.activate_controllers = list(activate or [])
        req.deactivate_controllers = list(deactivate or [])
        if hasattr(req, "start_controllers"):
            req.start_controllers = list(activate or [])
        if hasattr(req, "stop_controllers"):
            req.stop_controllers = list(deactivate or [])
        req.strictness = SwitchController.Request.BEST_EFFORT
        if hasattr(req, "activate_asap"):
            req.activate_asap = True
        if hasattr(req, "start_asap"):
            req.start_asap = True
        if hasattr(req, "timeout"):
            req.timeout.sec = int(timeout_sec)
            req.timeout.nanosec = 0
        future = client.call_async(req)
        result = future.result(timeout=15.0)
        return bool(getattr(result, "ok", False))

    def _activate_and_confirm_controllers(self, desired: list[str]) -> None:
        """activate desired controllers 并用 list_controllers 确认均为 active。"""
        all_ctrls = sorted(
            {
                *self._ARM_CONTROLLERS["joint"].values(),
                *self._ARM_CONTROLLERS["pose"].values(),
            }
        )
        deactivate = [c for c in all_ctrls if c not in desired]

        states = self._list_controller_states()
        print_green(f"[controllers] before switch: { {n: states.get(n,'?') for n in all_ctrls} }")

        already = [n for n in desired if states.get(n) == "active"]
        need = [n for n in desired if n not in already]
        if need or any(states.get(c) == "active" for c in deactivate):
            print_green(
                f"[controllers] switch --activate {desired} --deactivate {deactivate}"
            )
            ok = self._switch_controllers_topic_mode(desired, deactivate)
            if not ok:
                raise RuntimeError(
                    f"SwitchController failed for activate={desired}, deactivate={deactivate}"
                )

        states = self._list_controller_states()
        inactive = [n for n in desired if states.get(n) != "active"]
        print_green(f"[controllers] after switch: { {n: states.get(n,'?') for n in desired} }")
        if inactive:
            raise RuntimeError(
                f"controllers not active after switch: {inactive}; "
                f"states={ {n: states.get(n) for n in desired} }"
            )
        self._active_controllers = list(desired)
        print_green(f"[controllers] confirmed active: {desired}")

    def _confirm_controllers_before_control(self, desired: list[str]) -> None:
        """首次发控制前确认 controller 已激活。

        优先自动 switch+list；若无 controller_manager，则打印等价 CLI 并等待人工确认。
        """
        all_ctrls = sorted(
            {
                *self._ARM_CONTROLLERS["joint"].values(),
                *self._ARM_CONTROLLERS["pose"].values(),
            }
        )
        deactivate = [c for c in all_ctrls if c not in desired]
        cli = self._format_switch_controllers_cli(desired, deactivate)

        try:
            self._activate_and_confirm_controllers(desired)
            return
        except Exception as e:
            print_green(f"[controllers] auto switch/list unavailable: {e}")

        print_green(
            "[controllers] 请先在终端激活控制器（与下列命令等价），确认后再继续：\n"
            f"  {cli}\n"
            "示例:\n"
            "  ros2 control switch_controllers --activate endpose_single_arm_qp_L_controller"
        )
        # 若之后服务可用，再自动确认一次；否则依赖人工确认
        input("控制器已激活后按 Enter 继续...")
        try:
            states = self._list_controller_states(timeout_sec=2.0)
            inactive = [n for n in desired if states.get(n) != "active"]
            if inactive:
                raise RuntimeError(
                    f"仍未 active: {inactive}; states={ {n: states.get(n) for n in desired} }"
                )
            print_green(f"[controllers] confirmed active via list: {desired}")
        except Exception as e:
            print_green(
                f"[controllers] 无法自动 list 确认（{e}）；已按人工确认继续，"
                f"请确保已执行: {cli}"
            )
        self._active_controllers = list(desired)
        self._topic_controllers_user_confirmed = True

    def _pub_float64_array(self, pub_key: str, data) -> None:
        from std_msgs.msg import Float64MultiArray

        pub = self._cmd_pubs.get(pub_key)
        if pub is None:
            raise RuntimeError(f"missing publisher: {pub_key}")
        msg = Float64MultiArray()
        msg.data = [float(x) for x in np.asarray(data, dtype=np.float64).flatten().tolist()]
        pub.publish(msg)

    def _pub_joint_target(self, arm: str, target) -> None:
        """joint：topic 模式发 /jointspace_commands_{L|R}，7 维 Float64MultiArray。"""
        if self.use_ros2controller:
            if arm == "left":
                self.ros2controller.direct_ctrler.pub_jointspace_commands_l(target=target)
            else:
                self.ros2controller.direct_ctrler.pub_jointspace_commands_r(target=target)
        else:
            # 等价:
            # ros2 topic pub /jointspace_commands_L std_msgs/msg/Float64MultiArray \
            #   'data: [j1,...,j7]' --once
            q = np.asarray(target, dtype=np.float64).flatten()[: self.joint_dim]
            if q.size != self.joint_dim:
                raise ValueError(
                    f"joint target for {arm} must have {self.joint_dim} dims, got {q.size}"
                )
            self._pub_float64_array(f"joint_{arm}", q)

    def _pub_endpose_target(self, arm: str, pose_xyzrpy) -> None:
        """pose：topic 模式发 /endposetarget_{L|R}，[x,y,z,roll,pitch,yaw]。"""
        if self.use_ros2controller:
            target_pose = self.pose_euler2quat(pose_xyzrpy)
            kwargs = dict(
                from_frame="waist_yaw_link",
                target=target_pose.tolist(),
                offset=[0.0, 0.0, 0.0],
            )
            if arm == "left":
                self.ros2controller.direct_ctrler.pub_endposetarget_l(
                    to_frame="left_tcp_link", **kwargs
                )
            else:
                self.ros2controller.direct_ctrler.pub_endposetarget_r(
                    to_frame="right_tcp_link", **kwargs
                )
        else:
            # 等价:
            # ros2 topic pub /endposetarget_L std_msgs/msg/Float64MultiArray \
            #   'data: [x,y,z,roll,pitch,yaw]' --once
            xyzrpy = np.asarray(pose_xyzrpy, dtype=np.float64).flatten()[:6]
            if xyzrpy.size != 6:
                raise ValueError(
                    f"endpose target for {arm} must have 6 dims xyzrpy, got {xyzrpy.size}"
                )
            self._pub_float64_array(f"endpose_{arm}", xyzrpy)

    def free_arm_names(self) -> list:
        names = []
        if getattr(self, "free_left", False):
            names.append("left")
        if getattr(self, "free_right", False):
            names.append("right")
        return names

    def allow_free_arm_teleop_on_reset(self) -> bool:
        """True only on the first reset when free_* is set and free_arm_teleop=true.
        free_arm_teleop=false：不需要同构操控 free 臂，直接走 reset 位姿。
        """
        if not self.free_arm_names():
            return False
        if not getattr(self, "free_arm_teleop", True):
            return False
        return not self._free_arms_initialized

    def active_arm(self) -> str | None:
        """单臂 free 模式下返回被控制的臂；双臂返回 None。"""
        if self.free_left and not self.free_right:
            return "right"
        if self.free_right and not self.free_left:
            return "left"
        return None

    def active_arm_names(self) -> list:
        active = self.active_arm()
        if active is not None:
            return [active]
        return ["left", "right"]

    def _desired_controllers(self, mode: str, commanded_arms: list) -> list[str]:
        """commanded 臂用 mode 对应 controller；hold 中的 free 臂始终附带 jointspace。"""
        if mode not in self._ARM_CONTROLLERS:
            raise ValueError(f"Unknown control mode for controllers: {mode}")
        names: list[str] = []
        for arm in commanded_arms:
            if arm not in self._ARM_CONTROLLERS[mode]:
                raise ValueError(f"Unknown arm name: {arm}")
            names.append(self._ARM_CONTROLLERS[mode][arm])
        if getattr(self, "_enable_free_arm_hold", False):
            for arm in self.free_arm_names():
                free_ctrl = self._ARM_CONTROLLERS["joint"][arm]
                if free_ctrl not in names:
                    names.append(free_ctrl)
        return sorted(set(names))

    def _ensure_controllers(self, mode: str, commanded_arms: list | None = None) -> None:
        """按发送模式激活/确认 controller；集合未变则跳过。

        use_ros2controller=false：首次发控制前确认（自动 switch+list，失败则提示 CLI 并等待人工确认）。
        """
        if commanded_arms is None:
            commanded_arms = self.active_arm_names()
        desired = self._desired_controllers(mode, list(commanded_arms))
        if self._active_controllers == desired:
            return
        if self.use_ros2controller:
            ok = self.ros2controller.activate_controllers(
                desired, via_http=True, timeout_sec=3.0
            )
            if ok:
                self._active_controllers = desired
                print_green(f"[controllers] activated ({mode}): {desired}")
            else:
                self._active_controllers = None
                print(f"[controllers] FAILED to activate ({mode}): {desired}")
            return
        self._confirm_controllers_before_control(desired)

    def _hold_free_robot_arms(self) -> None:
        """step 阶段：将被 free 的机械臂保持在 reset 关节角。"""
        if not getattr(self, "_enable_free_arm_hold", False):
            return
        if not isinstance(self._reset_joint, dict):
            return
        free_names = self.free_arm_names()
        if not free_names:
            return
        # 只补齐 free 臂的 jointspace，不切换 active 臂当前 mode
        free_ctrls = [self._ARM_CONTROLLERS["joint"][a] for a in free_names]
        current = list(self._active_controllers or [])
        missing = [c for c in free_ctrls if c not in current]
        if missing:
            if current:
                desired = sorted(set(current + free_ctrls))
                if self.use_ros2controller:
                    ok = self.ros2controller.activate_controllers(
                        desired, via_http=True, timeout_sec=3.0
                    )
                    if ok:
                        self._active_controllers = desired
                        print_green(f"[controllers] merged free hold: {desired}")
                    else:
                        self._active_controllers = None
                else:
                    self._confirm_controllers_before_control(desired)
            else:
                self._ensure_controllers(self.control_mode, self.active_arm_names())
        for name in free_names:
            if name not in self._reset_joint:
                continue
            arm_target = np.asarray(self._reset_joint[name], dtype=np.float64).flatten()[
                : self.joint_dim
            ].tolist()
            self._pub_joint_target(name, arm_target)
            # 仅当配置了 reset_hands 且已锁定时，才强制保持 hand；否则不锁定 hand
            if not (self._reset_hands and name in self._reset_hands and self.hands_locked()):
                continue
            gripper = [float(self._reset_hands[name])]
            self.last_gripper_value[name] = gripper[0]
            if name == "left":
                self.left_gripper.sync_target_joint(gripper)
            elif name == "right":
                self.right_gripper.sync_target_joint(gripper)

    def _get_tele_equips(self) -> dict:
        return self.tele_agent.tele_agent._equips

    def _exit_xtele_sync_via_switch_mode(self) -> None:
        """用 TeleCore.switch_mode 恢复同构臂模式（替代 exit_any_sync，Universal station 无该方法）。"""
        last = getattr(self, "_xtele_last_mode", None)
        if not last:
            return
        for key, mode in list(last.items()):
            try:
                self.tele_agent.switch_mode(key, mode)
            except Exception as e:
                print(f"[xtele] switch_mode restore failed for {key}: {e}")

    def _sync_xtele_no_gripper(self, arms: dict) -> None:
        if arms is None:
            arms = self.curr_arm_joints

        if not self.tele_agent.get_sync_flag():
            self.tele_agent.set_sync_mode() 

        goal = []
        for name in ["left", "right"]:
            if name not in self.curr_arm_joints:
                continue
            arm_joints = self.curr_arm_joints[name]
            goal = goal + list(arm_joints)
        
        print("goal in _sync_xtele_no_gripper:", goal)
        input("Press Enter to Continue...")
        self.tele_agent.set_sync_pos(goal)

    # def _sync_xtele_arms(self, arm_names: list) -> None:
    #     """将指定同构臂同步到当前机械臂关节（进入位置模式并 set_position）。"""
    #     from xtele.equipment.dynamixel.driver.dynamixel_robot import OperatingMode

    #     equips = self._get_tele_equips()
    #     # station = self.tele_agent.tele_agent
    #     # exit_any_sync 依赖 station.Last_OperatingMode，必须与官方 set_position 路径一致
    #     # if not hasattr(station, "Last_OperatingMode"):
    #         # station.Last_OperatingMode = {}
    #     if not hasattr(self, "_xtele_last_mode"):
    #         self._xtele_last_mode = {}

    #     for name in arm_names:
    #         arm_key = f"dynamixel_{name}_arm"
    #         grip_key = f"dynamixel_{name}_gripper"
    #         arm_joints = list(
    #             np.asarray(self.curr_arm_joints[name], dtype=np.float64).flatten()[
    #                 : self.joint_dim
    #             ]
    #         )
    #         hand = np.asarray(self.curr_gripper_joints[name], dtype=np.float32).flatten()
    #         if hand.size == 0:
    #             hand_val = 0.01176
    #         else:
    #             hand_val = float(hand[0])
    #         # GripperAgent.set_position expects a sequence when gripper_limits is set
    #         grip_pos = [hand_val]

    #         ## sync无夹爪维度
    #         for key, pos in ((arm_key, arm_joints), (grip_key, grip_pos)):
    #             if key not in equips:
    #                 continue
    #             equip = equips[key]
    #             if (
    #                 hasattr(equip, "_robot")
    #                 and equip._robot.dynamic_mode != OperatingMode.Position
    #             ):
    #                 prev_mode = equip._robot.dynamic_mode
    #                 self._xtele_last_mode[key] = prev_mode
    #                 # station.Last_OperatingMode[key] = prev_mode
    #                 # equip.switch_operating_mode(OperatingMode.Position)
    #                 self.tele_agent.switch_mode(key, OperatingMode.Position)
    #             equip.set_position(pos)

    def _power_off_xtele_arms(self, arm_names: list | None = None) -> None:
        """同构臂下电（关闭力矩）。默认下电所有 free 臂。"""
        if arm_names is None:
            arm_names = self.free_arm_names()
        if not arm_names or not hasattr(self, "tele_agent"):
            return
        equips = self._get_tele_equips()
        for name in arm_names:
            for key in (f"dynamixel_{name}_arm", f"dynamixel_{name}_gripper"):
                if key in equips:
                    equips[key].deactivate_torque()
    
    # def get_xtele(self) -> dict:
    #     # 获取xtele的值；补齐 Last_OperatingMode，供 exit_any_sync 恢复非位置模式
    #     # station = self.tele_agent.tele_agent
    #     # if not hasattr(station, "Last_OperatingMode"):
    #         # station.Last_OperatingMode = {}
    #     # if hasattr(self, "_xtele_last_mode"):
    #         # for key, mode in self._xtele_last_mode.items():
    #             # station.Last_OperatingMode.setdefault(key, mode)
    #     # self.tele_agent.exit_any_sync(0)

    #     # 退出同构 Position 同步：Universal 无 exit_any_sync，改用 switch_mode 恢复原模式


    #     ## get_sync_flag状态机
    #     self._exit_xtele_sync_via_switch_mode()
    #     # [0:7]: left arm joints [7]: left gripper, [8:15]: right arm joints [15]: right gripper
    #     joints = self.tele_agent.act()

    #     pin_kine = self.robot_station._robot_dict.get("robot").pin_kine

    #     left_joints = joints["arm"]["position"]["left"]
    #     right_joints = joints["arm"]["position"]["right"]
    #     left_gripper = joints["hand"]["position"]["left"]
    #     right_gripper = joints["hand"]["position"]["right"]

    #     l_tcp_pose = pin_kine.get_left_tcp_pose(left_joints)
    #     r_tcp_pose = pin_kine.get_right_tcp_pose(right_joints)

    #     left_joints = np.concatenate([left_joints, [left_gripper]])
    #     right_joints = np.concatenate([right_joints, [right_gripper]])

    #     joints = {
    #         'left': left_joints,
    #         'right': right_joints
    #     }

    #     xtele_ee_pose = {
    #         'left': l_tcp_pose.get_xyz_m_xyzw_ndarray(),
    #         'right': r_tcp_pose.get_xyz_m_xyzw_ndarray()
    #     }

    #     recv = {
    #         'joints': joints,
    #         'pose': xtele_ee_pose,
    #     }

    #     return recv

    def get_xtele(self) -> dict:
        self._update_currpos()
        robot_pos = list(self.curr_arm_joints[key] for key in self.curr_arm_joints.keys())
        action = np.asarray(
            self.tele_agent.get_intervention_action(robot_pos), dtype=np.float64
        ).flatten()
        print("action in get_xtele:", action)
        input("Press Enter to Continue...")
        assert action.shape[0] == 17, f"expect 17-dim, got {action.shape}"
        # 布局: [0:7] 左臂 | [7] 左夹爪 | [8:15] 右臂 | [15] 右夹爪 | [16] 腰

        pin_kine = self.robot_station._robot_dict.get("robot").pin_kine
        l_tcp_pose = pin_kine.get_left_tcp_pose(action[0:7])
        r_tcp_pose = pin_kine.get_right_tcp_pose(action[8:15])

        return {
            "joints": {                       # 与 tienkung 保持 8 维/臂 的一致布局
                "left":  action[0:8],
                "right": action[8:16],
            },
            "pose": {
                "left":  l_tcp_pose.get_xyz_m_xyzw_ndarray(),
                "right": r_tcp_pose.get_xyz_m_xyzw_ndarray(),
            },
            "waist": float(action[16]),
        }

    def _send_joint_command(self, joints: np.ndarray | dict, include_gripper=False):
        '''
        joints: 双臂字典 {"left": ..., "right": ...}；
                单臂也可直接传 ndarray，内部包装为 active_arm（默认 right）。
        '''
        self._ensure_controllers("joint", list(joints.keys()))
        for name in joints.keys():
            cmd_val = joints[name][-1] if include_gripper else None
            target_gripper = self._gripper_cmd_for_arm(name, include_gripper, cmd_val, control_mode = "joint")

            arm_target = np.asarray(joints[name]).tolist()[0:self.joint_dim]
            self._pub_joint_target(name, arm_target)
            if name == "left":
                self.left_gripper.sync_target_joint(target_gripper)
            elif name == "right":
                self.right_gripper.sync_target_joint(target_gripper)
        return None
    
    def _send_pos_command(self, pose: np.ndarray | dict, include_gripper=False):
        '''
        pose: 双臂字典 {"left": ..., "right": ...}；
        '''
        self._ensure_controllers("pose", list(pose.keys()))
        for name in pose.keys():
            cmd_val = pose[name][-1] if include_gripper else None
            target_gripper = self._gripper_cmd_for_arm(name, include_gripper, cmd_val, control_mode = "pose")
            self._pub_endpose_target(name, pose[name])
            if name == "left":
                self.left_gripper.sync_target_joint(target_gripper)
            else:
                self.right_gripper.sync_target_joint(target_gripper)

        return None

    def _get_obs_from_robot(self) -> dict:
        obs = self.robot_station.get_obs()
        # print("obs from robot: ", obs)
        arm_pose = obs["arm_pose"]
        arm_pose_t = {name: arm_pose[name][0:3] for name in arm_pose}
        arm_pose_quat = {name: arm_pose[name][3:] for name in arm_pose}
        arm_pose_quat = {name: Rotation.from_quat(arm_pose_quat[name]).as_quat(canonical=True) for name in arm_pose_quat}
        arm_pose = {name: np.hstack([arm_pose_t[name], arm_pose_quat[name]]) for name in arm_pose}
        obs["arm_pose"] = arm_pose
        obs["hand_joints"] = self._transform_hand_joints(obs["hand_joints"])
        return obs


    def _transform_hand_joints(self, hand_joint_dict):
        #### todo debug
        # if self.robot_type == "tienkung":
        #     transform_joints_dict = {}
        #     for name, joints in hand_joint_dict.items():
        #         if np.array(joints).shape[0] == 1:
        #             target_joint = [-0.5 * joints[0] + 1] * 6
        #             target_joint[-1] = 0
        #         else:
        #             target_joint = [(1 - joints[0]) * 2]
        #         transform_joints_dict[name] = target_joint

        #     return transform_joints_dict
        if "tienyi" in self.robot_type:
            return hand_joint_dict
        else:
            raise NotImplementedError("Unknown robot type")
    