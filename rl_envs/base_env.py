
import os
import numpy as np
np.set_printoptions(precision=5, suppress=True)  # 设置 numpy 打印格式：5位小数，抑制科学计数法
import gymnasium as gym
import zmq  # ZeroMQ 用于进程间通信（franka1 使用）
import pickle  # 用于序列化数据
import cv2  # OpenCV 用于图像处理
from scipy.spatial.transform import Rotation  # 用于旋转表示转换（四元数、欧拉角等）
import time
from typing import Dict, Tuple, Optional, Union, Any, List


try:
    from xhil.rl_env.shared_state import shared_state
    from xhil.rl_env.utils.decode_image import decode_image
except Exception:
    from .shared_state import shared_state
    from .utils.decode_image import decode_image

def print_green(x: any) -> None:
    return print("\033[92m {}\033[00m".format(x))


class BaseEnv(gym.Env):
    def __init__(
        self,
        fake_env: bool = False,
        config: Optional[Any] = None,
    ) -> None:
        print('before init BaseEnv')
        self.config = config
        self._gripper_sleep = config.gripper_sleep  # 夹爪动作之间的最小时间间隔（秒）
        self.joint_dim = config.joint_dim  # 关节维度（通常为 7，对应 7 个关节）
        self.dual_arm = config.dual_arm  # 是否为双臂机器人
        print('config.reset_joint:', config.reset_joint)
        if self.dual_arm:
            self._reset_joint = {name: np.array(config.reset_joint[name])[0:self.joint_dim] for name in config.reset_joint.keys()}
        else:
            self._reset_joint = np.array(config.reset_joint)[0:self.joint_dim]  # 重置时的目标关节角度

        # 可选：reset 时设定夹爪（task.reset_hands → robot_config.reset_hands）
        self._reset_hands = None
        reset_hands = getattr(config, "reset_hands", None)
        if reset_hands is not None and hasattr(reset_hands, "keys"):
            self._reset_hands = {
                name: float(np.asarray(reset_hands[name]).flatten()[0])
                for name in reset_hands.keys()
            }

        
        self._random_xy_range = config.random_xy_range  # 随机重置时 xy 平面的随机范围
        self._random_rz_range = config.random_rz_range  # 随机重置时 z 轴旋转的随机范围
        self._random_reset = config.random_reset  # 是否启用随机重置（增加训练多样性）
        self._bgr2rgb = config.bgr2rgb  # 是否将 BGR 图像转换为 RGB（OpenCV 默认 BGR）
        self._image_keys = config.image_keys  # 图像键列表（如 ["right", "wrist"]）
        print("in BaseEnv self._image_keys:", self._image_keys)
        self.robot_type = config.robot_type  # 机器人类型（"franka1", "franka2", "ur_wrist"）
        self.robot_identify = config.robot_identify  # 机器人标识（如 "robot"）
        self.gripper_identify = config.gripper_identify  # 夹爪标识（如 ["left", "right"]）
        self.action_scale = config.action_scale  # 动作缩放因子 [位置缩放, 旋转缩放, 夹爪缩放]
        self.max_episode_length = config.max_episode_length  # 最大 episode 长度（步数）
        self.absolute_action = config.absolute_action  # 是否为绝对动作（True=绝对位姿，False=相对增量）
        self.control_mode = config.control_mode  # 控制模式（"joint"=关节空间，"pose"=笛卡尔空间）
        self.close_gripper = config.close_gripper  # 重置时是否关闭夹爪
        self.fix_gripper = config.fix_gripper  # 是否固定夹爪（不控制夹爪）
        self.ego_mode = config.ego_mode  # 是否启用自我模式（用于手动重置场景）
        # True: pose 模式下 tcp_pose/currpos 用指令位姿；False: 每步从真机 arm_pose 读取
        self.use_cmd_pose = config.use_cmd_pose
        # pose 动作掩码 [x,y,z,roll,pitch,yaw]；1=可动，0=冻结（delta 置 0）；夹爪由 fix_gripper 单独控制
        self.pose_action_mask = self._parse_pose_action_mask(config)
        self.currpos = None
        self.replay_trajectory = config.get("replay_trajectory", False)  # 是否回放轨迹 
        if self.replay_trajectory:
            self.joint_trajectory_file = config.joint_trajectory_file
            self.action_trajectory_file = config.action_trajectory_file

        # 轨迹日志：每回合 reset replay 与 step 时记录 joint / ee_pos
        self.trajectory_log = False  # 是否记录轨迹日志
        self.trajectory_log_file = "trajectory_log.txt"
        
        assert self.control_mode in ["joint", "pose"], f'Not valid control mode: {self.control_mode}'
        
        self.fake_env = fake_env  # 是否为假环境（True 时不连接真实机器人）
        self.hz = config.hz  # 控制频率（Hz），用于控制执行速度
        self._last_cmd_time = None  # 上次发指令的 wall time（perf_counter）

        image_resize = config.image_resize  # 图像尺寸字典，如 {"right": (128, 128, 3), "wrist": (128, 128, 3)}
        print_green(f'in BaseEnv image_resize: {image_resize}')

        tcp_pose_dim = 7  # 末端执行器位姿：xyz 位置(3) + 四元数旋转(4) = 7维
        if self.dual_arm:
            tcp_pose_dim = 2 * 7  # 双臂，每臂7维
        
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
        if "franka1" in self.robot_type:
            state_dict["ee_force"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))  # 末端力/力矩
            state_dict['arm_force'] = gym.spaces.Box(-np.inf, np.inf, shape=(7,))

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),  # 状态字典（位姿、关节、力等）
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=image_resize[key], dtype=np.uint8) 
                        for key in config.image_keys}  # 图像字典（每个相机一个图像）
                ),
            }
        )

        if self.control_mode == "pose":
            if "tienkung" in self.robot_type:
                self.action_space = gym.spaces.Box( 
                    np.array([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            else:
                self.action_space = gym.spaces.Box(
                    np.array([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
                    np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
                )
            self._apply_mask_to_action_space()
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.joint_dim + 1,))

        # if fake_env:
        #     print_green("fake env : not connect to robot")
        #     return 
        if fake_env:
            print_green("fake env : not connect to robot")
            print("in BaseEnv fake_env: give fake h5 data")
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

        if "franka1" in self.robot_type:
            # franka1 使用 ZeroMQ 进行进程间通信
            # 这种方式允许环境运行在一个进程中，机器人控制运行在另一个进程中
            host = "0.0.0.0"  # 监听所有网络接口
            self._context = zmq.Context()  # 创建 ZMQ 上下文
            self._socket = self._context.socket(zmq.REQ)  # 创建请求-响应模式的 socket
            self.addr = f"tcp://{host}:{4398}"  # 绑定地址和端口
            self._socket.bind(self.addr)  # 绑定 socket（等待机器人控制进程连接）
        else:
            # franka2 和 ur_wrist 使用 xrocs 框架
            # xrocs 是一个统一的机器人控制框架
            cfg_loader = ConfigLoader("/home/eai/Documents/configuration.toml")  # 加载配置文件
            self.cfg_dict = cfg_loader.get_config()  # 获取配置字典
            station_loader = StationLoader(self.cfg_dict)  # 创建站点加载器
            self.robot_station = station_loader.generate_station_handle()  # 生成机器人站点句柄
            self.robot_station.connect()  # 连接机器人硬件


        
        ### Test Code by Chris ###
        if "tienkung" or "tienyi" in self.robot_type:
            robot = self.robot_station.get_robot_handle()["robot"]
            self.ros2controller = robot.ros2controller

            self.ros2controller.set_arm_enable(True)
            self.ros2controller.set_arm_mode(0)
            # self.ros2controller.set_param(
            #     node_name='endpose_single_arm_qp_L_controller',
            #     parameter_name='vel_limits', 
            #     value=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            #     timeout_sec=10.0
            # )
            # self.ros2controller.set_param(
            #     node_name='endpose_single_arm_qp_R_controller',
            #     parameter_name='vel_limits', 
            #     value=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            #     timeout_sec=10.0
            # )
            # self.ros2controller.
            self.left_gripper = self.robot_station.get_gripper_handle()["left"]
            self.right_gripper = self.robot_station.get_gripper_handle()["right"]


        
        self._update_currpos()  # 更新当前位置（从机器人获取当前状态）
        self.last_gripper_act = time.time()  # 记录上次夹爪动作时间（用于限制夹爪动作频率）

        # # xyz 位置边界：限制末端执行器在安全的工作空间内
        # # todo 2: determine proper xyz bounding box for each arm
        # self.xyz_bounding_box = gym.spaces.Box(
        #     np.array(config.abs_pose_limit_low[:3]),  # 下界：[x_min, y_min, z_min]
        #     np.array(config.abs_pose_limit_high[:3]),  # 上界：[x_max, y_max, z_max]
        #     dtype=np.float64,
        # )
        # # rpy 旋转边界：限制末端执行器的旋转角度
        # self.rpy_bounding_box = gym.spaces.Box(
        #     np.array(config.abs_pose_limit_low[3:]),  # 下界：[roll_min, pitch_min, yaw_min]
        #     np.array(config.abs_pose_limit_high[3:]),  # 上界：[roll_max, pitch_max, yaw_max]
        #     dtype=np.float64,
        # )
   
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

        # 用于裁剪图像 ROI（感兴趣区域），减少计算量并聚焦重要区域
        self.image_crop = {}

        # 如果配置了图像裁剪，则创建裁剪函数
        if hasattr(config, 'image_crop') and config.image_crop is not None:
            for camera_key, crop_func in config.image_crop.items():
                if callable(crop_func):
                    # 如果已经是函数，直接使用
                    self.image_crop[camera_key] = crop_func
                else:
                    # 如果是参数列表，创建裁剪函数
                    def make_crop_func(crop_params: Union[list, tuple, Any]) -> Any:
                        crop_params = list(crop_params)
                        if isinstance(crop_params, (list, tuple)) and len(crop_params) == 2:
                            h_range, w_range = crop_params  # 高度范围和宽度范围
                            # print(h_range)
                            # 返回裁剪函数：img[h_min:h_max, w_min:w_max]
                            return lambda img: img[h_range[0]:h_range[1], w_range[0]:w_range[1]]
                        else:
                            # 如果参数格式不正确，返回恒等函数（不裁剪）
                            return lambda img: img
                    
                    self.image_crop[camera_key] = make_crop_func(crop_func)
        
        self.save_path = None  # 保存路径（用于保存数据，如果启用）
        self.save_frame = False  # 是否保存帧（用于调试）
        self.obs_pre = None  # 上一帧观察（可能用于某些算法）
        # self.last_gripper_value = 1.0 if self.close_gripper else 0.0  # 初始夹爪值（1.0=关闭，0.0=打开）
        self.last_gripper_value = {
            "left": 1.0 if self.close_gripper else 0.0,
            "right": 1.0 if self.close_gripper else 0.0,
        }
        self.last_gripper_act = time.time()  # 重置夹爪动作时间
        print('after init BaseEnv')
        self.episode_id = 0


    def _get_reset_waypoints(self) -> List[Dict[str, np.ndarray]]:
        """
        获取重置时的途经点列表。
        根据每个手臂当前的z值，独立选择桌面上或桌面下的路径。
        
        Returns:
            途经点列表，每个途经点是一个字典 {'left': array, 'right': array}
        """
        
        # ============ 获取当前末端z值并判断路径 ============
        z_threshold = -0.15  # z值阈值
        
        # 默认路径选择（如果currpos未初始化，走安全的桌面下路径）
        left_use_upper_path = False
        right_use_upper_path = False
        
        if self.currpos is not None:
            print("=============add tienyi debug=====================================================================")

            if self.dual_arm and "tienkung" in self. robot_type:
                left_z = self.currpos["left"][2]
                right_z = self.currpos["right"][2]
                
                # 判断每个手臂应该走哪条路径
                left_use_upper_path = (left_z >= z_threshold)
                right_use_upper_path = (right_z >= z_threshold)
                
                # 打印诊断信息
                print_green("=" * 60)
                print_green(f"[重置途经点] 当前末端Z值 - Left: {left_z:.4f}, Right: {right_z:.4f}")
                print_green(f"[路径选择] Left:  {'桌面上路径' if left_use_upper_path else '桌面下路径'}")
                print_green(f"[路径选择] Right: {'桌面上路径' if right_use_upper_path else '桌面下路径'}")
                print_green("=" * 60)
            else:
                print_green("[重置途经点] 当前机器人类型不支持获取双臂末端位置")
        else:
            print_green("[重置途经点] self.currpos 尚未初始化，使用默认路径（桌面下）")
        
        # #  ============ 定义路径途经点 ============

        ### todo debug

        # pushtest_tienkung_0113 路径 (自定义路径，适用于 pushtest_tienkung_0113 任务)
        # 桌面下路径（安全路径，避障）
        lower_path_waypoints = [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],           # 左臂：抬高到安全高度
            [0.0, 1.0, -0.25, -1.37, -0.9, 0.0, 0.0],       # 左臂：通过中间点
            [0.0, 0.1, -0.15, -1.77, 0.0, 0.0, 0.0]  # 左臂：到达最终位置
        ]
        
        # 桌面上路径（直接路径，适用于已在桌面上的情况）
        upper_path_waypoints = [
            [0.0, 0.1, -0.25, -1.37, 0.0, 0.0, 0.0],     # 左臂：直接通过中间点
            [0.0, 0.1, -0.15, -1.77, 0.0, 0.0, 0.0]  # 左臂：到达最终位置
        ]
        
        # 右臂路径（镜像对称）
        lower_path_waypoints_right = [
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.25, -1.37, 0.9, 0.0, 0.0],
            [0.0, -0.1, 0.15, -1.77, 0.0, 0.0, 0.0]
        ]
        
        upper_path_waypoints_right = [
            [0.0, -0.1, 0.25, -1.37, 0.0, 0.0, 0.0],
            [0.0, -0.1, 0.15, -1.77, 0.0, 0.0, 0.0]
        ]

        puzzle_path_waypoints_left = [
            [0.0, 0.1, -0.15, -1.77, 0.0, 0.0, 0.0]
        ]
        puzzle_path_waypoints_right = [
            [0.0, -0.1, 0.15, -1.77, 0.0, 0.0, 0.0]
        ]


        # # wear_scarf 路径 (自定义路径，适用于 wear_scarf 任务)
        # # 桌面下路径（安全路径，避障）
        # lower_path_waypoints = [
        #     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],           # 左臂：抬高到安全高度
        #     [0.0, 1.0, -0.25, -1.37, -0.9, 0.0, 0.0],       # 左臂：通过中间点
        #     [0.0, 0.1, -0.25, -1.37, -0.9, 0.0, 0.0]  # 左臂：到达最终位置
        # ]
        
        # # 桌面上路径（直接路径，适用于已在桌面上的情况）
        # upper_path_waypoints = [
        #     [0.0, 0.1, -0.25, -1.37, 0.0, 0.0, 0.0],     # 左臂：直接通过中间点
        #     [0.0, 0.1, -0.25, -1.37, -0.9, 0.0, 0.0]  # 左臂：到达最终位置
        # ]
        
        # # 右臂路径（镜像对称）
        # lower_path_waypoints_right = [
        #     [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     [0.0, -1.0, 0.25, -1.37, 0.9, 0.0, 0.0],
        #     [0.0, -0.1, 0.25, -1.37, 0.9, 0.0, 0.0]
        # ]
        
        # upper_path_waypoints_right = [
        #     [0.0, -0.1, 0.25, -1.37, 0.0, 0.0, 0.0],
        #     [0.0, -0.1, 0.25, -1.37, 0.9, 0.0, 0.0]
        # ]

        # ============ 根据判断结果组合路径 ============
        # 选择左臂路径
        # left_path = upper_path_waypoints if left_use_upper_path else lower_path_waypoints
        # 选择右臂路径
        # right_path = upper_path_waypoints_right if right_use_upper_path else lower_path_waypoints_right
        left_path = puzzle_path_waypoints_left
        right_path = puzzle_path_waypoints_right
        
        # 确保两条路径长度一致（补齐较短的路径）
        max_len = max(len(left_path), len(right_path))
        
        # 如果左臂路径较短，复制最后一个点
        while len(left_path) < max_len:
            left_path.append(left_path[-1])
        
        # 如果右臂路径较短，复制最后一个点
        while len(right_path) < max_len:
            right_path.append(right_path[-1])
        
        # ============ 转换为途经点列表 ============
        waypoints = []
        for left_joints, right_joints in zip(left_path, right_path):
            waypoint = {
                'left': np.array(left_joints[0:self.joint_dim],dtype=np.float64),
                'right': np.array(right_joints[0:self.joint_dim],dtype=np.float64)
            }
            waypoints.append(waypoint)
        
        print_green(f"[重置途经点] 生成了 {len(waypoints)} 个途经点")
        
        return waypoints
    
    def _move_through_waypoints(self, waypoints:  List[Dict[str, np.ndarray]], duration_per_waypoint: float = 3.0) -> None:
        """
        按照途经点列表移动机器人（直接执行途经点，不生成平滑路径）。
        
        Args:
            waypoints: 途经点列表，每个途经点是一个字典 {'left': array, 'right': array}
            duration_per_waypoint:  每个途经点之间的移动时间（秒）- 此参数在当前实现中不使用
        """
        if not self.dual_arm:
            print_green("单臂机器人不支持途经点功能")
            return
        
        if "tienkung" not in self.robot_type:
            print_green(f"机器人类型 {self.robot_type} 暂不支持途经点功能")
            return
        
        print_green("=" * 80)
        print_green(f"开始通过 {len(waypoints)} 个途经点移动（直接执行模式）")
        print_green("=" * 80)
        
        # 遍历所有途经点
        for waypoint_idx, waypoint in enumerate(waypoints):
            print_green(f"\n[途经点 {waypoint_idx + 1}/{len(waypoints)}]")
            
            # 打印当前状态和目标
            for name in waypoint. keys():
                curr_joints = self.curr_arm_joints[name]
                goal_joints = waypoint[name]
                
                print_green(f"  {name. upper()} 臂:")
                print_green(f"    当前:  {np.round(curr_joints, 4)}")
                print_green(f"    目标: {np.round(goal_joints, 4)}")
                print_green(f"    差值: {np.round(goal_joints - curr_joints, 4)}")
            
            # 直接发送目标关节角度
            print_green(f"\n  ⏩ 直接发送关节命令...")
            start_time = time.time()
            
            # 初始化夹爪为闭合
            # waypoint["left"] = np.append(waypoint["left"], [1.0 ])
            # waypoint["right"] = np.append(waypoint["right"], [1.0])

            # self._send_joint_command(waypoint, include_gripper=True)
            
            # waypoint["left"] = waypoint["left"][:-1]
            # waypoint["right"] = waypoint["right"][:-1]

            self._send_joint_command(waypoint, include_gripper=False)


            exec_time = time.time() - start_time
            # print_green(f"  ⏱  命令发送耗时: {exec_time:. 4f}s")
            
            # 等待机器人到达（可选，根据实际情况调整）
            time.sleep(duration_per_waypoint)
            
            # 更新当前位置
            self._update_currpos()
            
            # 打印到达后的实际位置和误差
            print_green(f"\n  [到达验证]")
            for name in waypoint.keys():
                actual_joints = self.curr_arm_joints[name]
                target_joints = waypoint[name]
                error = actual_joints - target_joints
                max_error = np.max(np.abs(error))
                
                print_green(f"    {name.upper()} 臂:")
                print_green(f"      实际: {np.round(actual_joints, 4)}")
                print_green(f"      误差: {np.round(error, 4)} (最大: {max_error:.4f})")
            
            print_green(f"\n  ✓ 已到达途经点 {waypoint_idx + 1}/{len(waypoints)}")
        
        print_green("\n" + "=" * 80)
        print_green("✓ 所有途经点已完成")
        print_green("=" * 80)



    def clip_safety_box(self, pose: np.ndarray, arm_name: str) -> np.ndarray:
        """Clip absolute pose [x,y,z,r,p,y,(gripper)] to abs_pose_limit boxes.

        dual_arm: Dict[arm_name -> Box]; single-arm / free_*: plain Box
        """
        pose = np.asarray(pose, dtype=np.float64).copy()

        xyz_bbox = self.xyz_bounding_box
        if isinstance(xyz_bbox, gym.spaces.Dict):
            xyz_bbox = xyz_bbox[arm_name]
        pose[:3] = np.clip(pose[:3], xyz_bbox.low, xyz_bbox.high)

        rpy_bbox = self.rpy_bounding_box
        if isinstance(rpy_bbox, gym.spaces.Dict):
            rpy_bbox = rpy_bbox[arm_name]
        pose[3:6] = np.clip(pose[3:6], rpy_bbox.low, rpy_bbox.high)

        return pose



    def get_xtele(self) -> Dict[str, Any]:
        if "franka1" in self.robot_type:  # ⚠️ 字符串匹配
            send_content = {"get_xtele": True}
            self._socket.send(pickle.dumps(send_content))
            recv = self._socket.recv()
            recv = pickle.loads(recv)
            return recv
        elif 'ur' in self.robot_type:
            self.tele_agent.switch_act()
            joints = self.tele_agent.act()
            joints = list(joints)
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv 
        elif 'franka2' in self.robot_type:
            # 应该不是在这里退出反向同构，这里是为了获取xtele的值
            self.tele_agent.exit_any_sync(0)
            # self.tele_agent.switch_act()
            joints = self.tele_agent.act()

            joints = list(joints)
            # print("type(joints):", type(joints))
            xtele_ee_pose = self.robot_station.get_ee_pose_from_joint(joints[0:self.joint_dim])
            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }
            return recv
        elif "tienkung" in self.robot_type:
            # 应该不是在这里退出反向同构，这里是为了获取xtele的值
            self.tele_agent.exit_any_sync(0)
            # [0:7]: left arm joints [7]: left gripper, [8:15]: right arm joints [15]: right gripper

            joints = self.tele_agent.act()

            pin_kine = self.robot_station._robot_dict.get("robot").pin_kine
            l_tcp_pose = pin_kine.get_left_tcp_pose(joints[0:7])
            r_tcp_pose = pin_kine.get_right_tcp_pose(joints[8:15])

            # left_joints = self.curr_arm_joints
            # l_tcp_pose = pin_kine.get_left_tcp_pose(self.curr_arm_joints['left'][0:7])
            # r_tcp_pose = pin_kine.get_right_tcp_pose(self.curr_arm_joints['right'][0:7])

            joints = {
                'left': joints[0:8],
                'right': joints[8:16]
            }

            xtele_ee_pose = {
                'left': l_tcp_pose.get_xyz_m_xyzw_ndarray(),
                'right': r_tcp_pose.get_xyz_m_xyzw_ndarray()
            }

            recv = {
                'joints': joints,
                'pose': xtele_ee_pose,
            }

            return recv
        else:
            raise NotImplementedError("Unknown robot type")

    def init_xtele(self) -> None:
        if "franka1" in self.robot_type:  # ⚠️ 字符串匹配
            recv_content = {"success": False}
            while not recv_content["success"]:
                send_content = {"init_xtele": True}
                self._socket.send(pickle.dumps(send_content))
                recv_content = self._socket.recv()
                recv_content = pickle.loads(recv_content)
        elif 'ur' in self.robot_type or 'franka2' in self.robot_type:
            from xtele.core.integrate_module import TeleCore
            self.tele_agent = TeleCore()
        elif "tienkung" in self.robot_type:
            from xtele.core.integrate_module import TeleCore
            self.tele_agent = TeleCore()
        # elif "tienyi" in self.robot_type:
        #     from xtele.core.integrate_module import TeleCore
        #     self.tele_agent = TeleCore()
        elif "tienyi" in self.robot_type:
            from xtele.equipment.tiansuo.tele_controller import TeleController
            robot_init_pos = [0.0] * 14  # 或 15 维（最后一位为腰关节）
            self.tele_agent = TeleController(
                config={"basic": {"tiansuo_ip": "192.168.41.100"}},
                robot_init_pos=list(robot_init_pos),
                skip_sync_check=False,
                freq=60,
            )
            input("Success Created TeleController, Press Enter to Continue...")
        else:
            raise NotImplementedError("Unknown robot type")

    def sync_xtele(self, timeout: float = 5, arms: Optional[List[str]] = None) -> None:
        if "franka1" in self.robot_type:
            goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)
            recv_content = {"success": False}

            while not recv_content["success"]:
                send_content = {"sync_xtele": goal, "timeout": timeout}
                self._socket.send(pickle.dumps(send_content))
                recv_content = self._socket.recv()
                recv_content = pickle.loads(recv_content)
        elif 'ur' in self.robot_type:
            goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)
            self.tele_agent.switch_reverse()
            self.tele_agent.sync_position(goal)
        elif 'franka2' in self.robot_type:
            goal = np.append(self.curr_arm_joints, self.curr_gripper_joints)
            need_torque = False
            """
            添加重力模式后再加上
            """
            # self.tele_agent.switch_mode("dynamixel_arm", OperatingMode.Position) 
            tele_cur_joints = self.tele_agent.act()
            tele_tar_joints = goal
            timeout = int(timeout // 0.02) - 5
            if timeout <= 0:
                path = np.array([tele_tar_joints])
            else:
                path = np.linspace(tele_cur_joints, tele_tar_joints, timeout)
            # for _ in range(5):
            #     path = np.append(path, [path[-1]], axis=0)
            for p in path:
                self.tele_agent.set_position(p)
                time.sleep(0.01)
        elif "tienkung" in self.robot_type:
            goal = []
            
            for name in self.curr_arm_joints.keys():
                arm_joints = self.curr_arm_joints[name]
                hand_joints = self.curr_gripper_joints[name]

                goal = goal + list(arm_joints)
                goal.append(hand_joints)
            self.tele_agent.set_position(goal)
        elif "tienyi" in self.robot_type:
            # arms=None: 同步全部（reset）；step 可只同步 active 臂
            if not hasattr(self, "curr_arm_joints") or self.curr_arm_joints is None:
                self._update_currpos()

            if arms is None:
                arms = self.curr_arm_joints

            if hasattr(self, "_sync_xtele_arms"):
                arms_key = list(arms.keys())
                self._sync_xtele_arms(arms_key)
            elif hasattr(self, "_sync_xtele_no_gripper"):
                self._sync_xtele_no_gripper(arms)
            else:
                goal = []
                for name in ["left", "right"]:
                    if name not in self.curr_arm_joints:
                        continue
                    arm_joints = self.curr_arm_joints[name]
                    hand_joints = self.curr_gripper_joints[name]
                    goal = goal + list(arm_joints)
                    goal.append(hand_joints)
                self.tele_agent.set_position(goal)
        else:
            raise NotImplementedError("Unknown robot type")

    @staticmethod
    def _parse_pose_action_mask(config) -> np.ndarray:
        """Parse pose action mask: [x,y,z,roll,pitch,yaw], 1=movable, 0=frozen.

        Gripper is not part of this mask (use fix_gripper / close_gripper).
        Falls back to enable_rotation: false -> freeze all of roll/pitch/yaw.
        """
        mask = getattr(config, "pose_action_mask", None)
        if mask is None:
            enable_rot = getattr(config, "enable_rotation", True)
            if not enable_rot:
                mask = [1, 1, 1, 0, 0, 0]
            else:
                mask = [1, 1, 1, 1, 1, 1]
        mask = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask.shape[0] != 6:
            raise ValueError(
                f"pose_action_mask must have 6 entries [x,y,z,roll,pitch,yaw], got {mask.shape[0]}"
            )
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"pose_action_mask entries must be 0 or 1, got {mask}")
        print_green(f"pose_action_mask [x,y,z,r,p,y]: {mask.tolist()}")
        return mask

    def apply_pose_action_mask(self, action: np.ndarray) -> np.ndarray:
        """Zero frozen xyz/rpy dims. Gripper dim is left unchanged (fix_gripper)."""
        if getattr(self, "control_mode", None) != "pose":
            return action
        mask = getattr(self, "pose_action_mask", None)
        if mask is None:
            return action
        action = np.asarray(action, dtype=np.float32).copy()
        # 每臂 7 维 = xyzrpy(6) + gripper(1)；只 mask 前 6 维
        if action.shape[0] == 14:
            action[0:6] = action[0:6] * mask
            action[7:13] = action[7:13] * mask
        elif action.shape[0] >= 6:
            n = min(6, action.shape[0])
            action[:n] = action[:n] * mask[:n]
        return action

    def _apply_mask_to_action_space(self) -> None:
        """Clamp frozen xyz/rpy dims' action_space bounds to 0; gripper untouched."""
        mask = getattr(self, "pose_action_mask", None)
        if mask is None or self.control_mode != "pose":
            return
        low = np.array(self.action_space.low, dtype=np.float32, copy=True)
        high = np.array(self.action_space.high, dtype=np.float32, copy=True)
        if low.shape[0] == 14:
            offsets = (0, 7)
        else:
            offsets = (0,)
        for offset in offsets:
            for i in range(6):
                if offset + i >= low.shape[0]:
                    break
                if mask[i] == 0:
                    low[offset + i] = 0.0
                    high[offset + i] = 0.0
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def compute_next_pose(self, curr_pose, delta_action) -> np.ndarray:
        '''
        input:
            curr_pose: np.ndarray, shape: (7,) x,y,z, quat_x, quat_y, quat_z, quat_w
            delta_action: np.ndarray, shape: (7,) x,y,z, r, p, y, gripper
        output:
            next_pose: np.ndarray, shape: (7,) x, y, z, r, p, y, gripper
        '''
        delta_action = self.apply_pose_action_mask(delta_action)
        curr_pose_t, curr_pose_quat = curr_pose[0:3], curr_pose[3:7]
        curr_pose_euler = Rotation.from_quat(curr_pose_quat).as_euler("xyz")
        curr_pose_new = np.hstack([curr_pose_t, curr_pose_euler])
        # print("curr_pose:", curr_pose_new)
        # print('delta action:', delta_action)
        curr_mat = np.eye(4)
        curr_mat[:3, :3] = Rotation.from_quat(curr_pose[3:]).as_matrix()
        curr_mat[:3, 3] = curr_pose[:3]
        
        delta_t = delta_action[0:3] * self.action_scale[0]
        # print('delta_t:', delta_t)
        delta_euler = delta_action[3:6] * self.action_scale[1]
        delta_mat = np.eye(4)  # 4x4 齐次变换矩阵
        delta_mat[:3, :3] = Rotation.from_euler("xyz", delta_euler).as_matrix()  # 旋转矩阵
        delta_mat[:3, 3] = delta_t  # 平移部分
        
        tar_mat = curr_mat @ delta_mat
        # tar_mat = delta_mat @ curr_mat
        
        tar_t = tar_mat[:3,3]
        tar_euler = Rotation.from_matrix(tar_mat[:3, :3]).as_euler("xyz")
        tar_gripper = self.action_scale[2] * delta_action[-1]
        
        next_pose = np.hstack([tar_t, tar_euler, tar_gripper])
        # print('next_pose:', next_pose)
        return next_pose
    
    def step(self, action: np.ndarray, xtele_joints: Dict[str, np.ndarray] = None, intervention_mode: str = "joint") -> Tuple[Dict[str, Any], int, bool, bool, Dict[str, Any]]:
        if self.fake_env:
            print("action:", action)
            return self._fake_step(action)
        # ========== 控制执行频率：按距上次发指令补齐 1/hz，再发本步指令 ==========
        now = time.perf_counter()
        
            # print(f"sleep_time: {sleep_time}s (period={1.0 / self.hz}s, since_last={now - self._last_cmd_time:.4f}s)")
        # else:
        #     print(f"sleep_time: 0s (first cmd, period={1.0 / self.hz}s)")

        start_time = time.perf_counter()

        # 夹爪动作需要一定时间完成，频繁控制可能导致问题
        # 如果距离上次夹爪动作超过 _gripper_sleep 秒，且未固定夹爪，则允许控制夹爪
        if (time.time() - self.last_gripper_act > self._gripper_sleep) and not self.fix_gripper:
            include_gripper = True  # 允许控制夹爪
        else:
            include_gripper = False  # 不控制夹爪，使用上次的值

        next_pose = None
        if self.control_mode == "joint" :
        # if self.control_mode == "joint" or (xtele_joints is not None and intervention_mode == "joint"):
            # 关节空间控制：直接发送关节角度命令
            if self.control_mode == "joint":
                obs = self._send_joint_command(action, include_gripper) 
            else:
                obs = self._send_joint_command(xtele_joints, include_gripper=True) 
            self.currpos = None

        elif self.control_mode == "pose":
            # 笛卡尔空间控制：需要将动作转换为目标位姿
            if self.absolute_action:
                # 绝对动作模式：动作直接表示目标位姿（不常用）
                if self.dual_arm:
                    raise NotImplementedError("Absolute action mode is not supported for dual arm robots")
                else:
                    next_pose = action
                    cur_euler = self.pose_quat2euler(self.currpos)
                    print("cur_euler:", cur_euler)
            else:
                action = action.clip(-1, 1)
                # if self.dual_arm:
                if "tienkung" in self.robot_type:
                    next_pose = {}
                    dual_arm_action = {
                        "left": action[0:7],
                        "right": action[7:14],
                    }
                    
                    # if self.trajectory_log:
                    #     if self.curr_path_length == 0:
                    #         with open(self.trajectory_log_file, "a", encoding="utf-8") as f:
                    #             f.write(f"========== Episode {self.episode_id} step ==========\n")
                    #     currpos = {name: self.pose_quat2euler(pose) for name, pose in self.currpos.items()}
                    #     self._log_trajectory_state("step-dual_arm_action: ", self.curr_path_length, send_pos_command=dual_arm_action)

                    for name, action in dual_arm_action.items():
                        curr_pose = self.currpos[name]
                        next_pose[name] = self.compute_next_pose(curr_pose, action)

                        # 裁剪到安全边界
                        #### todo debug 似乎查看这个安全边界是否能在tienyi上工作
                        # print("====================remove safety box please check it before traning==============")
                        next_pose[name] = self.clip_safety_box(next_pose[name], arm_name=name)  
                        
                    
                    # """
                    # 给定next_pose_test
                    # """
                    # next_pose = {
                    #     "left": [ 0.38585, 0.18   , 0.07821, 1.73977,-0.64275,-2.12937, 0.91681], 
                    #     "right": [ 0.36551,-0.20636, 0.09108,-1.81227,-0.45693, 1.72734, 0.80559]
                    # }

                    # for name, pose in next_pose.items():
                    #     # 在xy平面上添加随机偏移
                    #     pose[:2] += np.random.uniform(
                    #         -1, 1, (2,)
                    #     )
                    #     # 获取旋转角
                    #     axis_random = np.array(pose[3:6])
                    #     assert axis_random.shape == (3,)
                    #     # 在Z轴旋转角上添加随机扰动
                    #     axis_random[-1] += np.random.uniform(
                    #         -0.5, 0.5
                    #     )
                    #     pose[3:6] = axis_random

                    # print('next_pose:', next_pose)

                    if self.use_cmd_pose:
                        self.currpos = {name: self.pose_euler2quat(pose) for name, pose in next_pose.items()}
                    else:
                        self.currpos = None  # _get_obs 时从真机刷新
                elif "tienyi" in self.robot_type: 
                    next_pose = {}
                    if self.dual_arm:
                        dual_arm_action = {
                            "left": action[0:7],
                            "right": action[7:14],
                        }
                    else:
                        active = self.active_arm() if hasattr(self, "active_arm") else None
                        assert active is not None, (
                            "tienyi single-arm step requires free_left or free_right"
                        )
                        dual_arm_action = {active: action[0:7]}
                    # print("currpos: ", self.currpos)
                    for name, arm_action in dual_arm_action.items():
                        curr_pose = self.currpos[name]
                        next_pose[name] = self.compute_next_pose(curr_pose, arm_action)
                        # print(f"next_pose: {next_pose}")

                        # 裁剪到安全边界
                        # print("====================remove safety box please check it before traning==============")
                        # next_pose[name] = self.clip_safety_box(next_pose[name], arm_name=name)  
                        
                    if self.use_cmd_pose:
                        self.currpos = {name: self.pose_euler2quat(pose) for name, pose in next_pose.items()}
                    else:
                        self.currpos = None  # _get_obs 时从真机刷新
                else:
                    raise NotImplementedError("Unknown robot type") 
                # else:
                #     next_pose = self.compute_next_pose(self.currpos, action)
                #     # next_pose = self.clip_safety_box(next_pose)  # 裁剪到安全边界
            # input("Press Enter to Continue...")
            obs = self._send_pos_command(next_pose, include_gripper)
        else:
            raise NotImplementedError(f"Not valid control mode: {self.control_mode}")

        # 以发指令时刻为步周期锚点（含 hold）；sleep 后再取 obs，保证图像对应本步 action 到位后
        if hasattr(self, "_hold_free_robot_arms"):
            self._hold_free_robot_arms()
        self._last_cmd_time = time.perf_counter()
        # print(f"base env send_cmd time: {self._last_cmd_time - start_time}s")

        if include_gripper:
            self.last_gripper_act = time.time()  # 记录夹爪动作时间 
        
        self.curr_path_length += 1  # 增加当前 episode 的步数

        # 发令后补齐 1/hz，再取观测（锚点是 _last_cmd_time，不是 step 开头的 now）
        sleep_time = 1.0 / self.hz - (time.perf_counter() - self._last_cmd_time)
        if sleep_time > 0:
            # print(f"sleep_time: {sleep_time}s")
            time.sleep(sleep_time)
                
        # 获取完整的观察（图像 + 状态）
        obs = self._get_obs()

        curr_joints = self.curr_arm_joints
        self.joint_list.append(curr_joints)

        reward = 0.0  # 基础环境不计算奖励，奖励由包装器（如 reward_wrapper）计算
        terminated = False  # 基础环境不判断任务是否成功，由包装器判断
        truncated = self.curr_path_length >= self.max_episode_length  # 如果达到最大步数，则截断

        # 返回标准 Gymnasium 格式：(obs, reward, terminated, truncated, info)
        return obs, int(reward), terminated, truncated, {
            "succeed": terminated,  # 是否成功（基础环境始终为 False）
            "is_intervention": False,  # 是否有人工干预（基础环境始终为 False，由包装器设置）
            "include_gripper": include_gripper,
            # 与 action[-1] / 单臂 state 对齐：返回标量；双臂仍返回扁平向量
            "gripper_pose": self._info_gripper_pose(),
        }


    def reset(self, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.last_gripper_act = time.time()  # 重置夹爪动作时间
        self._last_cmd_time = None  # 新 episode 第一步不等待
        self.currpos = None
        self.joint_list = []
        self.debug_image = None
        # free_*：仅第一次且 free_arm_teleop=true 时允许同构控制 free 臂；
        # free_arm_teleop=false 时直接 hold / go_to_reset 到 reset 位姿
        allow_free_teleop = self.allow_free_arm_teleop_on_reset()
        has_free = bool(self.free_arm_names())
        if has_free and not allow_free_teleop:
            self._enable_free_arm_hold = True
            self._power_off_xtele_arms()
        else:
            self._enable_free_arm_hold = False

        if self.fake_env:
            self.curr_path_length = 0
            self._fake_step_idx = 0
            self._apply_fake_step_state(0)
            shared_state.terminate = False
            shared_state.emergency_terminate = False
            obs = self._get_obs()
            print("curr_action:", self.curr_action)
            return obs, {"success": False, "is_intervention": False}

        # if self.trajectory_log:
        #     with open(self.trajectory_log_file, mode="a", encoding="utf-8") as f:
        #         f.write(f"========== Episode {self.episode_id + 1} reset ==========\n")

        input("Press Enter to Continue...")
        # self.open_gripper()
        # self.go_to_reset(joint_reset=True, close_gripper=False)  # 移动到重置位置（关节空间）

        if self.ego_mode:
            # 自我模式用于需要手动重新摆放物体的任务（如抓取任务）
            # input("Press Enter to Continue...")
            # self.last_gripper_value = 1.0 if self.close_gripper else 0.0
            self.last_gripper_value = {
                "left": 1.0 if self.close_gripper else 0.0,
                "right": 1.0 if self.close_gripper else 0.0,
            }
            # 有 reset_hands 时：第一次可自由操作，之后锁定；无则不锁定 hand
            if not allow_free_teleop and getattr(self, "_reset_hands", None):
                if hasattr(self, "_apply_reset_hands_to_gripper"):
                    self._apply_reset_hands_to_gripper()
                elif isinstance(self.last_gripper_value, dict):
                    for name, val in self._reset_hands.items():
                        self.last_gripper_value[name] = val
            self._update_currpos()  # sync_xtele 需要 curr_arm_joints
            # 第一次 free_* reset：双臂同构；之后仅 active 臂同构，free 同构保持下电
            if has_free and not allow_free_teleop:
                sync_arms = self.active_arm_names()
                self.sync_xtele(arms=sync_arms)
                self._power_off_xtele_arms()
            else:
                self.sync_xtele()  # 同步遥操作设备位置
            shared_state.terminate = False  # 重置终止标志
            
            print("start to wait user to reposition the scene")
            input("Press Enter to Continue...")
            # 循环等待用户重新摆放场景
            while not shared_state.terminate:
                print("重新摆放场景, 按pause继续: ")
                obs = self.get_xtele()  # 获取遥操作设备的位置
                xtele_joints = obs['joints']  # 提取关节角度
                self._update_currpos()  # 更新当前位置
                if has_free and not allow_free_teleop:
                    # 仅用 active 同构臂控制对应机械臂；free 机械臂 hold reset
                    # （hand 仅在有 reset_hands 时由 _hold_free_robot_arms 锁定）
                    active = self.active_arm_names()
                    target_joint = {k: xtele_joints[k] for k in active}
                    self._send_joint_command(target_joint, include_gripper=True)
                    # print("send_joint_command: ", target_joint)
                    if hasattr(self, "_hold_free_robot_arms"):
                        self._hold_free_robot_arms()
                    if hasattr(self, "_power_off_xtele_arms"):
                        self._power_off_xtele_arms()
                else:
                    target_joint = xtele_joints.copy()  # 复制目标关节角度
                    # print("send_joint_command: ", target_joint)
                    self._send_joint_command(target_joint, include_gripper=True)  # 移动机器人到遥操作位置
                time.sleep(1 / self.hz)  # 控制执行频率
            shared_state.terminate = False  # 重置终止标志
        
        # input("Press Enter to Continue...")
        # self.go_to_reset(joint_reset=True, close_gripper=True)  # 移动到重置位置（关节空间）
        
        

        self.curr_path_length = 0  # 重置当前 episode 的步数


        self.last_gripper_act = time.time()  # 重置夹爪动作时间
        # 设置初始夹爪值；仅当有 reset_hands 时落到配置值
        if "tienyi" in self.robot_type or "tienkung" in self.robot_type:
            self.last_gripper_value = {
                "left": 1.0 if self.close_gripper else 0.0,
                "right": 1.0 if self.close_gripper else 0.0,
            }
            if getattr(self, "_reset_hands", None):
                if hasattr(self, "_apply_reset_hands_to_gripper"):
                    self._apply_reset_hands_to_gripper()
                else:
                    for name, val in self._reset_hands.items():
                        self.last_gripper_value[name] = val
        else:
            self.last_gripper_value = 1.0 if self.close_gripper else 0.0
            if getattr(self, "_reset_hands", None):
                self.last_gripper_value = next(iter(self._reset_hands.values()))

        
        self.go_to_reset(joint_reset=True)  # 移动到重置位置（关节空间）
        
        
        # if self.replay_trajectory:
        #     self._replay_trajectory()

        self.currpos = None 
        obs = self._get_obs(obs = None)  # 获取初始观察状态

        # 必须输入 Enter 才能继续
        while True:
            user_input = input("after go to reset, Press Enter to Continue...")
            if user_input == "":
                break
            print("Please press Enter (no other input) to continue.")


        # if self.trajectory_log:
        #     self._log_trajectory_state("reset done", self.curr_path_length)

        self.episode_id += 1
        # print('after base env reset ....')

        # free_*：第一次 reset（含同构控制）完成后标记，之后不再用同构控制 free 臂；
        # 若配置了 reset_hands，则同步锁定对应 hand
        if hasattr(self, "free_arm_names") and self.free_arm_names():
            self._free_arms_initialized = True
            self._enable_free_arm_hold = True

        return obs, {"success": False, "is_intervention": False}  # 初始状态：未成功，无干预

    def _replay_trajectory(self) -> None:
        with open(self.joint_trajectory_file, "rb") as f:
            joint_trajectory = np.load(f,allow_pickle=True)
        with open(self.action_trajectory_file, "rb") as f:
            action_trajectory = np.load(f,allow_pickle=True)
        
        for i, arm_joints in enumerate(joint_trajectory):
            arm_joints["left"] = np.append(arm_joints["left"], [1.0 if action_trajectory[i][6] > 0.5 else 0.0])
            arm_joints["right"] = np.append(arm_joints["right"], [1.0 if action_trajectory[i][13] > 0.5 else 0.0])
            print_green(f"------------------arm_joints: {arm_joints}")
            print_green(f"------------------action: {action_trajectory[i]}")
            self._send_joint_command(arm_joints, include_gripper=True)
            time.sleep(1 / self.hz)
            self._update_currpos()

            # if self.trajectory_log:
            #     self._log_trajectory_state("reset_replay", i)

        self.last_gripper_act = time.time()  # 记录夹爪动作时间 
        input("Press Enter to Continue...")

    def go_to_reset(
        self, joint_reset: bool = False, close_gripper: bool | None = None
    ) -> None:
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.

        Args:
            joint_reset: whether to use joint-space reset.
            close_gripper: whether to close the gripper during reset.
                None → use self.close_gripper from config.
        """
        if close_gripper is None:
            close_gripper = self.close_gripper

        grip_val = 1.0 if close_gripper else 0.0
        if "tienyi" in self.robot_type or "tienkung" in self.robot_type:
            self.last_gripper_value = {
                "left": grip_val,
                "right": grip_val,
            }
            if getattr(self, "_reset_hands", None):
                if hasattr(self, "_apply_reset_hands_to_gripper"):
                    self._apply_reset_hands_to_gripper()
                else:
                    for name, val in self._reset_hands.items():
                        self.last_gripper_value[name] = val
        else:
            self.last_gripper_value = grip_val
            if getattr(self, "_reset_hands", None):
                self.last_gripper_value = next(iter(self._reset_hands.values()))

        self._update_currpos()

        if self.dual_arm or (
            "tienyi" in self.robot_type and isinstance(self._reset_joint, dict)
        ):
            if self.dual_arm and "tienkung" in self.robot_type:
                waypoints = self._get_reset_waypoints()
                self._move_through_waypoints(waypoints, duration_per_waypoint=3.0)
            elif "tienyi" in self.robot_type and isinstance(self._reset_joint, dict):
                # 第一次 free_*（或 free_arm_teleop=false）：两臂都回到 reset_joint；
                # 之后 free 臂保持 reset，只移动 active 臂
                self._update_currpos()
                free_names = (
                    set(self.free_arm_names())
                    if hasattr(self, "free_arm_names")
                    else set()
                )
                hold_free = bool(free_names) and getattr(
                    self, "_free_arms_initialized", False
                )
                move_names = [
                    name
                    for name in self._reset_joint.keys()
                    if not (hold_free and name in free_names)
                ]
                # free_arm_teleop=false 且尚未 initialized：确保 free 臂也走到 reset
                # （move_names 已含 free，此处仅补 reset_hands）
                if (
                    free_names
                    and not hold_free
                    and not getattr(self, "free_arm_teleop", True)
                    and getattr(self, "_reset_hands", None)
                ):
                    if hasattr(self, "_apply_reset_hands_to_gripper"):
                        self._apply_reset_hands_to_gripper()
                    else:
                        for name, val in self._reset_hands.items():
                            if isinstance(self.last_gripper_value, dict):
                                self.last_gripper_value[name] = val
                paths = {}
                cnt = int(3 / (1 / self.hz))
                for name in move_names:
                    goal_j = np.asarray(
                        self._reset_joint[name], dtype=np.float64
                    ).flatten()[: self.joint_dim]
                    curr_j = np.asarray(
                        self.curr_arm_joints[name], dtype=np.float64
                    ).flatten()[: self.joint_dim]
                    paths[name] = np.linspace(curr_j, goal_j, cnt)
                if not paths:
                    # 仅 hold free 臂（无 active 需移动时）
                    for _ in range(cnt):
                        if hasattr(self, "_hold_free_robot_arms"):
                            self._hold_free_robot_arms()
                        time.sleep(1 / self.hz)
                else:
                    for i in range(cnt):
                        cmd = {name: paths[name][i] for name in paths}
                        self._send_joint_command(cmd, include_gripper=False)
                        if hold_free and hasattr(self, "_hold_free_robot_arms"):
                            self._hold_free_robot_arms()
                        time.sleep(1 / self.hz)
            else:
                raise NotImplementedError("Unknown robot type")
            # goal_joints = self._reset_joint.copy()
            # curr_joints = self.curr_arm_joints
            # goal_joints = goal_joints
            # cnt = int(3 / (1 / self.hz))
            # _goal_joints = {}
            # for name, _curr_joints in curr_joints.items():
            #     _goal_joints[name] = np.linspace(_curr_joints, goal_joints[name], cnt)

            # self._send_joint_command(_goal_joints, include_gripper=False)
        else:
            
            # reset_pose = self._reset_pose.copy()
            assert self._reset_joint.shape == (self.joint_dim,)
            if "ur" in self.robot_type:
                curr_pose = self.currpos.copy()
                curr_pose = self.pose_quat2euler(curr_pose)
                arm_joints = np.append(self.curr_arm_joints, self.last_gripper_value)
                for _ in range(5):
                    self._send_joint_command(arm_joints, include_gripper=False)
                    time.sleep(1 / self.hz)
                    
                for name, _robot in self.robot_station.get_robot_handle().items():
                    goal_joints = Joints(self._reset_joint, num_of_dofs=self.joint_dim)  # ✅ 使用 joint_dim
                    try:
                        return_val =_robot.reach_target_joint(goal_joints)
                    except Exception as e:
                        print(f"Error in reach_target_joint: {e}")
                for _ in range(5):
                    self._send_joint_command(self._reset_joint, include_gripper=False)
                    time.sleep(1 / self.hz)
            else:
                goal_joints = self._reset_joint.copy()
                # curr_joints = np.concatenate([self.curr_arm_joints, np.array([self.curr_gripper_joints])])
                # goal_joints = np.concatenate([goal_joints, np.array([self.last_gripper_value])])
                curr_joints = self.curr_arm_joints
                goal_joints = goal_joints
                cnt = int(3 / (1 / self.hz))
                path = np.linspace(curr_joints, goal_joints, cnt)
                for p in path:
                    self._send_joint_command(p, include_gripper=False)
                    time.sleep(1 / self.hz)

        self._update_currpos()
        # reset_pose = self.currpos.copy()
        # reset_pose = self.pose_quat2euler(reset_pose)
        # print("reset_pose:", reset_pose)
        
        if self._random_reset:  
            # 在xy平面上添加随机偏移
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
            reset_pose[3:] = axis_random  # ⚠️ 硬编码索引 [3:]
            self._send_pos_command(reset_pose, include_gripper=False)

    def _send_joint_command(self, joints: np.ndarray | Dict[str, np.ndarray], include_gripper: bool = False) -> Dict[str, Any]: 
        if self.dual_arm:
            '''
            joints = {
                "left": np.zeros(self.joint_dim),
                "right": np.zeros(self.joint_dim),
            }
            '''
            if "tienkung" in self.robot_type: # mingbo debug
                # robot_target = {
                #     "left": {
                #             "position": joints
                #         }
                # }
                # # curr_joints = np.concatenate([self.curr_arm_joints, self.curr_gripper_joints])
                # # goal_joints = np.concatenate([joints, np.array([self.last_gripper_value])])
                # curr_joints = self.curr_arm_joints
                # goal_joints = joints
                # # print("curr_joints:", curr_joints)
                # # print("goal_joints:", goal_joints)
                # # input("Press Enter to continue...")
                # obs = self.robot_station.step(robot_target)
                # input("Press Enter to continue...")
                for name in joints.keys():
                    if name == "left":
                        if include_gripper:
                            target_gripper = [joints[name][-1]]
                        else:
                            target_gripper = [0.0]
                        # print("exec_jointspace_arm_L_controller")
                        self.ros2controller.exec_jointspace_arm_L_controller(target=joints[name].tolist()[0:self.joint_dim])
                        self.left_gripper.sync_target_joint(target_gripper)
                    else:
                        if include_gripper:
                            target_gripper = [joints[name][-1]]
                        else:
                            target_gripper = [0.0]
                        # print("exec_jointspace_arm_R_controller")
                        self.ros2controller.exec_jointspace_arm_R_controller(target=joints[name].tolist()[0:self.joint_dim])
                        self.right_gripper.sync_target_joint(target_gripper)
                return None
            else:
                raise NotImplementedError("Unknown robot type")
        else:
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
            elif "franka2" in self.robot_type:
                robot_target = {
                    "arm_joints": {
                        "single": joints[0:self.joint_dim]
                    },
                    "hand_joints": {"single": [self.last_gripper_value]}
                }
                obs = self.robot_station.step(robot_target, robot_target)
                return obs
            elif "franka1" in self.robot_type:
                send_message = pickle.dumps({"arm_joints": joints[0:self.joint_dim+1]})
                recv = {"success": False}
                while not recv["success"]:
                    self._socket.send(send_message)
                    recv = self._socket.recv()
                    recv = pickle.loads(recv)
                return recv['obs']
            
            else:
                raise NotImplementedError("Unknown robot type")

    def _send_pos_command(self, pose: np.ndarray | Dict[str, np.ndarray], include_gripper: bool = False) -> Dict[str, Any]:
        if self.dual_arm:
            print("=============add tienyi debug=====================================================================")

            if "tienkung"or "tienyi" in self.robot_type:
                # step_dict = {
                #     "arm_pose": {
                #         "left": None,
                #         "right": None,
                #     },
                #     "hand_joints": {
                #         "left": None,
                #         "right": None,
                #     },
                # }

                # for name, _pose in pose.items():
                #     curr_pose = self.currpos[name]
                #     curr_pose = self.pose_quat2euler(curr_pose)
                #     step_dict["arm_pose"][name] = self.pose_euler2quat(_pose[0:6]) 
                #     step_dict["hand_joints"][name] = _pose[-1]
                #     print('in _send_pos_command, name:', name, 'curr_pose:', curr_pose, 'target_pose:', _pose)
                #     # pose[name] = self.compute_next_pose(curr_pose, _pose)
                # print('step_dict:', step_dict)
                # input("Press Enter to continue...")
                # # obs = self.robot_station.step_l_ee(pose["left"]) # 目前双臂机器人都是使用左臂的接口
                # obs = self.robot_station.step_ee(step_dict)
                # # print(f"obs is {obs}---")
                # print("pose:", pose)
                # input("Press Enter to continue...")


                # print("-----------------------pose:", pose)
                # print("-----------------------include_gripper:", include_gripper)

                for name in pose.keys():
                    target_pose = self.pose_euler2quat(pose[name])
                    if include_gripper:
                        target_gripper = [pose[name][-1]]

                        self.last_gripper_value[name] = 1.0 if target_gripper[0] >= 0.5 else 0.0
                       
                    else:   
                        target_gripper = [self.last_gripper_value[name]]
                    print("name: ", name)
                    print("target_pose: ", target_pose)
                    print("target_gripper: ", target_gripper)
                    if name == "left":
                        # print("exec_endpose_single_arm_qp_L_controller********************+++++++++++++++++++++++++")
                        self.ros2controller.exec_endpose_single_arm_qp_L_controller(from_frame="waist_yaw_link", to_frame="left_tcp_link", target=target_pose.tolist(), offset=[0.0, 0.0, 0.0])
                        self.left_gripper.sync_target_joint(target_gripper)
                    else:
                        self.ros2controller.exec_endpose_single_arm_qp_R_controller(from_frame="waist_yaw_link", to_frame="right_tcp_link", target=target_pose.tolist(), offset=[0.0, 0.0, 0.0])
                        self.right_gripper.sync_target_joint(target_gripper)

                return None

            else:
                raise NotImplementedError("Unknown robot type")
        else:
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
                arm_pose = np.append(pose[0:6], self.last_gripper_value)
                robot_target = {
                    "arm_pose": {
                        "single": arm_pose
                    },
                    "hand_joints": {}
                }
                if "franka2" in self.robot_type:
                    obs = self.robot_station.step_ee(robot_target)
                elif "franka1" in self.robot_type:
                    recv = {"success": False}
                    while not recv["success"]:
                        send_content = {"arm_pose": arm_pose}
                        self._socket.send(pickle.dumps(send_content))
                        recv = self._socket.recv()
                        recv = pickle.loads(recv)
                    obs = recv['obs']
                else:
                    raise NotImplementedError("Unknown robot type")
        return obs


    def _update_currpos(self, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if obs is None:
            obs = self._get_obs_from_robot()

        self.curr_gripper_joints = obs["hand_joints"]
        self.curr_arm_joints = obs["arm_joints"]
        print("curr_arm_joints:", self.curr_arm_joints)

        # 末端力：供 EndEffectorForceInterventionWrapper 等使用
        # self.curr_ee_force = obs["end_effector_force"]
        self.curr_ee_force = obs["end_effector_force"] if "end_effector_force" in obs.keys() else None

        # use_cmd_pose=False：始终用真机 arm_pose；True：仅在 currpos 为空时初始化
        if self.currpos is None or not getattr(self, "use_cmd_pose", True):
            self.currpos = obs["arm_pose"]
        return obs
    
    def _flatten(self, input_dict: Dict[str, Any]) -> np.ndarray:
        flattened_values = []
        for key in input_dict.keys():
            value = input_dict[key]
            flattened_values.append(np.array(value).flatten())
        return np.concatenate(flattened_values)

    def open_gripper(
        self,
        arm: str | list[str] | None = None,
        *,
        wait: float = 0.5,
    ) -> None:
        """单独打开夹爪（不改动手臂位姿）。约定：0.0=完全打开，1.0=完全关闭。

        Args:
            arm: 要打开的臂名 ``"left"`` / ``"right"``，或名称列表。
                ``None`` 时优先打开 ``active_arm()``；否则打开全部已知夹爪。
            wait: 发令后等待机械到位的秒数。
        """
        if self.fake_env:
            return

        open_val = 0.0

        if arm is None:
            if hasattr(self, "active_arm_names"):
                names = list(self.active_arm_names())
            elif hasattr(self, "active_arm") and self.active_arm() is not None:
                names = [self.active_arm()]
            elif isinstance(self.last_gripper_value, dict):
                names = list(self.last_gripper_value.keys())
            else:
                names = []
        elif isinstance(arm, str):
            names = [arm]
        else:
            names = list(arm)

        # tienyi / tienkung：直接 sync 夹爪
        if hasattr(self, "left_gripper") or hasattr(self, "right_gripper"):
            if not names:
                names = [
                    n
                    for n, h in (("left", "left_gripper"), ("right", "right_gripper"))
                    if hasattr(self, h)
                ]
            for name in names:
                if isinstance(self.last_gripper_value, dict):
                    self.last_gripper_value[name] = open_val
                else:
                    self.last_gripper_value = open_val
                target = [open_val]
                if name == "left" and hasattr(self, "left_gripper"):
                    self.left_gripper.sync_target_joint(target)
                elif name == "right" and hasattr(self, "right_gripper"):
                    self.right_gripper.sync_target_joint(target)
                else:
                    raise ValueError(f"Unknown gripper arm: {name}")
        else:
            # 单臂 UR / Franka：保持当前臂关节，只改 hand
            self.last_gripper_value = open_val
            self._update_currpos()
            joints = np.asarray(self.curr_arm_joints, dtype=np.float64).flatten()[
                : self.joint_dim
            ]
            cmd = np.append(joints, open_val)
            self._send_joint_command(cmd, include_gripper=True)

        self.last_gripper_act = time.time()
        if wait and wait > 0:
            time.sleep(wait)

    def _info_gripper_pose(self):
        """Scalar gripper for single-/active-arm control; flattened vector for dual-arm."""
        gripper = self.curr_gripper_joints
        if isinstance(gripper, dict):
            active = self.active_arm() if hasattr(self, "active_arm") else None
            if active is not None and active in gripper:
                gripper = gripper[active]
            else:
                gripper = self._flatten(gripper)
        gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
        if gripper.size == 1:
            return float(gripper[0])
        return gripper

    def _log_trajectory_state(self, tag: str, step_idx: int, send_pos_command: Optional[Dict[str, Any]] = None) -> None:
        """将当前 joint、ee_pos 追加写入 trajectory_log_file。"""
        if self.fake_env or self.currpos is None:
            return
        with open(self.trajectory_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{tag}] step={step_idx}\n")
            if self.dual_arm:
                for name in self.curr_arm_joints.keys():
                    if send_pos_command is not None and (name in send_pos_command.keys()):
                        send_pos_command_ = np.asarray(send_pos_command[name]).flatten()
                        f.write(f"  {name}_send_pos_command={np.array2string(send_pos_command_, separator=',')}\n")
            
                    joint = np.asarray(self.curr_arm_joints[name]).flatten()
                    ee_pose = np.asarray(self.currpos[name]).flatten()
               
                    f.write(f"  {name}_joint={np.array2string(joint, separator=',')}\n")
                    f.write(f"  {name}_ee_pos={np.array2string(ee_pose, separator=',')}\n")
                    
            else:
                if send_pos_command is not None and "single" in send_pos_command.keys():
                    send_pos_command_ = np.asarray(send_pos_command).flatten()
                    f.write(f"  send_pos_command={np.array2string(send_pos_command_, separator=',')}\n")

                joint = np.asarray(self.curr_arm_joints).flatten()
                ee_pose = np.asarray(self.currpos).flatten()
         
                f.write(f"  joint={np.array2string(joint, separator=',')}\n")
                f.write(f"  ee_pos={np.array2string(ee_pose, separator=',')}\n")
              
            f.write("\n")

    def _get_obs(self, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.fake_env:
            # images = self.get_fake_im()
            # state_observation = self.get_fake_pose()
            step_idx = self._fake_step_idx
            self._apply_fake_step_state(step_idx)
            if step_idx > self._fake_actions.shape[0] - 1:
                step_idx = self._fake_actions.shape[0] - 1
            self.curr_action = self._fake_actions[step_idx].astype(np.float64).copy()
            images = {
                key: self._decode_fake_image(step_idx, key) for key in self._image_keys
            }
            state_observation = {
                "tcp_pose": self.currpos,
                "gripper_pose": np.atleast_1d(self.curr_gripper_joints),
                "joints": self.curr_arm_joints,
            }

            # image_path = "/media/HIL-RL-Project/HIL-RL/experiments/pick_toy/origin_obs_dumps/episode_0000/step_0019"
            # # Match _decode_fake_image / decoder_image: cv2 loads BGR, optionally convert to RGB.
            # images = {}
            # for key in self._image_keys:
            #     bgr = cv2.imread(os.path.join(image_path, f"obs_origin_obs_{key}.png"))
            #     if bgr is None:
            #         raise RuntimeError(
            #             f"Failed to load fake image '{key}' from {image_path}"
            #         )
            #     if self._bgr2rgb:
            #         images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            #     else:
            #         images[key] = bgr
            # txt_path = os.path.join(image_path, "step_0019.txt")
            # with open(txt_path, "r", encoding="utf-8") as f:
            #     content = f.read()
            # proprio = None
            # for line in content.splitlines():
            #     if line.startswith("proprio="):
            #         vals = line.split("=", 1)[1].strip().strip("[]")
            #         proprio = np.array(
            #             [float(x) for x in vals.split(",") if x.strip()],
            #             dtype=np.float64,
            #         )
            #         break
            # if proprio is None:
            #     raise ValueError(f"proprio not found in {txt_path}")
            # # dump proprio = xyz(3) + quat(4) + gripper(1)
            # state_observation = {
            #     "tcp_pose": proprio[:7],
            #     "gripper_pose": np.atleast_1d(proprio[7]),
            #     "joints": np.zeros(self.joint_dim),
            # }
        else:
            # print('before self._update_currpos:', self.currpos)
            obs = self._update_currpos(obs)
            # print('after self._update_currpos:', self.currpos)
            
            images = {}
            if "tienkung" in self.robot_type:
                no_decode_keys = {"left", "right"}
            elif "tienyi" in self.robot_type:
                no_decode_keys = {}

            for key, cap in obs["images"].items():
                if key not in self._image_keys:
                    continue

                if key in no_decode_keys:
                    images[key] = cap
                else:
                    # tdecode_start_time = time.time()
                    rgb, _ = decode_image(cap, None, bgr2rgb=self._bgr2rgb)
                    images[key] = rgb
                    # tdecode_end_time = time.time()
                    # print(f"decode_image time: {tdecode_end_time - tdecode_start_time}s ... ")

                if not os.path.exists(f"online_image_{key}.png"):
                    bgr = cv2.cvtColor(images[key], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"online_image_{key}.png", bgr)

            if self.dual_arm:
                if "tienkung" in self.robot_type or "tienyi" in self.robot_type: # mingbo debug
                    state_observation = {
                        "tcp_pose": self._flatten(self.currpos),
                        "gripper_pose": self._flatten(self.curr_gripper_joints),
                        "joints": self._flatten(self.curr_arm_joints),
                    }

                else:
                    raise NotImplementedError("Unknown robot type") 
            else:
                active = self.active_arm() if hasattr(self, "active_arm") else None
                if active is not None and "tienyi" in self.robot_type:
                    # free_left / free_right：state 只暴露被控制的单臂
                    state_observation = {
                        "tcp_pose": np.asarray(self.currpos[active], dtype=np.float64).flatten(),
                        "gripper_pose": np.asarray(
                            self.curr_gripper_joints[active], dtype=np.float64
                        ).flatten(),
                        "joints": np.asarray(
                            self.curr_arm_joints[active], dtype=np.float64
                        ).flatten(),
                    }
                else:
                    state_observation = {
                        "tcp_pose": self.currpos,
                        "gripper_pose": self.curr_gripper_joints,
                        "joints": self.curr_arm_joints,
                    }
                if "franka1" in self.robot_type:
                    state_observation["ee_force"] = np.array(obs["ee_force"]['single'])
                    state_observation["arm_force"] = np.array(obs["arm_force"]['single'])
            
        return dict(images=images, state=state_observation)

    def _transform_hand_joints(self, hand_joint_dict):
        # sunny-gao debug:
        if self.robot_type == "tienkung":
            transform_joints_dict = {}
            for name, joints in hand_joint_dict.items():
                if np.array(joints).shape[0] == 1:
                    target_joint = [-0.5 * joints[0] + 1] * 6
                    target_joint[-1] = 1
                else:
                    # 6-dim joints -> 1-dim joints
                    # target_joint = [(1 - joints[0]) * 2]
                    target_joint = [(1 - joints[0])]
                transform_joints_dict[name] = target_joint

            return transform_joints_dict

        elif self.robot_type == "tienyi": # mingbo debug:完全复制tienkung代码
            transform_joints_dict = {}
            for name, joints in hand_joint_dict.items():
                if np.array(joints).shape[0] == 1:
                    target_joint = [-0.5 * joints[0] + 1] * 6
                    target_joint[-1] = 1
                else:
                    # 6-dim joints -> 1-dim joints
                    # target_joint = [(1 - joints[0]) * 2]
                    target_joint = [(1 - joints[0])]
                transform_joints_dict[name] = target_joint

            return transform_joints_dict          
 
        else:
            raise NotImplementedError("Unknown robot type")

    def pose_quat2euler(self, pose):
        pose_t, pose_quat = pose[0:3], pose[3:7]
        pose_euler = Rotation.from_quat(pose_quat).as_euler("xyz")
        pose = np.hstack([pose_t, pose_euler])
        return pose
    
    def pose_euler2quat(self, pose):
        pose_t, pose_euler = pose[0:3], pose[3:6]
        pose_quat = Rotation.from_euler("xyz", pose_euler).as_quat(canonical=True)
        pose = np.hstack([pose_t, pose_quat])
        return pose

    def get_ee_pose_from_joint(self, joint):
        import pinocchio as pin
        import numpy as np

        urdf_path = "/home/eai/Downloads/tiangong.urdf" 
        # q = np.array([0.10]*14) 
        # =========================================================

        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        pin.forwardKinematics(model, data, joint)
        pin.updateFramePlacements(model, data)

        ee_r_name = "right_tcp_link" 
        ee_l_name = "left_tcp_link"  

        id_r = model.getFrameId(ee_r_name)
        pose_r = data.oMf[id_r]
        pos_r = pose_r.translation  
        rot_r_mat = pose_r.rotation    
        rot_r_quat = Rotation.from_matrix(rot_r_mat).as_quat(canonical=True)

        id_l = model.getFrameId(ee_l_name)
        pose_l = data.oMf[id_l]
        pos_l = pose_l.translation  
        rot_l_mat = pose_l.rotation    
        rot_l_quat = Rotation.from_matrix(rot_l_mat).as_quat(canonical=True)

        pose_l = np.concatenate([pos_l, rot_l_quat], axis=-1)
        pose_r = np.concatenate([pos_r, rot_r_quat], axis=-1)
        # print("="*50)
        # print("右臂末端")
        # print("(xyz): ", np.round(pos_r, 4))
        # print("="*50)
        # print("左臂末端")
        # print("(xyz): ", np.round(pos_l, 4))
        # print('rotation:', rot_r)
        # exit(0)

        pose = {
            'left': pose_l,
            'right': pose_r
        }
        # print('=============> pin pose:', pose)
        # exit(0)
        return pose

    # 当前只适配天工双臂机器人或者其他单臂机器人
    # 后续适配多臂机器人
    def _get_obs_from_robot(self) -> Dict[str, Any]:
        # t0 = time.time()
        obs = self.robot_station.get_obs()
        print("obs keys:", obs.keys())
        # input("Press Enter to Continue...") # mingbo debug
        # print(f"get_obs_from_robot time: {time.time() - t0}s ... ")

        # ============ 每5秒打印一次末端xyz ============
        current_time = time.time()
        if not hasattr(self, '_last_print_time'):
            self._last_print_time = current_time
        
        if current_time - self._last_print_time >= 5.0:
            if self.dual_arm:
                if "tienkung" in self.robot_type:
                    left_xyz = obs["arm_pose"]["left"][0:3]
                    right_xyz = obs["arm_pose"]["right"][0:3]
                    print_green(f"[末端位置] Left XYZ: {np.round(left_xyz, 4)}, Right XYZ: {np.round(right_xyz, 4)}")
                elif "tienyi" in self.robot_type: # mingbo debug:完全复制tienkung代码
                    left_xyz = obs["arm_pose"]["left"][0:3]
                    right_xyz = obs["arm_pose"]["right"][0:3]
                    print_green(f"[末端位置] Left XYZ: {np.round(left_xyz, 4)}, Right XYZ: {np.round(right_xyz, 4)}")

                else:
                    raise NotImplementedError("Unknown robot type")
            else:
                # 单臂机器人
                xyz = obs["arm_pose"]["single"][0:3] if "single" in obs["arm_pose"] else obs["arm_pose"][0:3]
                print_green(f"[末端位置] XYZ: {np.round(xyz, 4)}")
            
            self._last_print_time = current_time
        # ============================================

        if self.dual_arm:
            if "tienkung" in self.robot_type:
                # ----------- arm pose standarization --------------
                arm_pose = obs["arm_pose"]
                arm_pose_t = {name: arm_pose[name][0:3] for name in arm_pose}
                arm_pose_quat = {name: arm_pose[name][3:] for name in arm_pose}
                arm_pose_quat = {name: Rotation.from_quat(arm_pose_quat[name]).as_quat(canonical=True) for name in arm_pose_quat}
                arm_pose = {name: np.hstack([arm_pose_t[name], arm_pose_quat[name]]) for name in arm_pose}
                obs["arm_pose"] = arm_pose
                obs["hand_joints"] = self._transform_hand_joints(obs["hand_joints"])
            elif "tienyi" in self.robot_type:
                #### todo debug 下面代码完全与上面的一样 但tienyi获取观测值的方法是一样的吗
                # ----------- arm pose standarization --------------
                arm_pose = obs["arm_pose"]
                arm_pose_t = {name: arm_pose[name][0:3] for name in arm_pose}
                arm_pose_quat = {name: arm_pose[name][3:] for name in arm_pose}
                arm_pose_quat = {name: Rotation.from_quat(arm_pose_quat[name]).as_quat(canonical=True) for name in arm_pose_quat}
                arm_pose = {name: np.hstack([arm_pose_t[name], arm_pose_quat[name]]) for name in arm_pose}
                obs["arm_pose"] = arm_pose
                obs["hand_joints"] = self._transform_hand_joints(obs["hand_joints"])
            else:
                raise NotImplementedError("Unknown robot type")
        else:
            raise NotImplementedError("Unknown robot type") 

        return obs

    def close(self) -> None:
        ## tienyi:ctrl.close()：actor断点捕捉到此处
        pass
    
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
