from __future__ import annotations

import numpy as np
import numpy.typing as npt
import rtde_control
import rtde_receive

from loguru import logger
from scipy.spatial.transform import Rotation as R


def _rotvec_to_rpy(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r = R.from_rotvec(rotvec)
    return np.asarray(r.as_euler(seq='xyz', degrees=False))

def _rpy_to_rotvec(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r = R.from_euler(rpy)
    return np.asarray(r.as_rotvec())

def _rotvec_to_quat_xyzw(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r = R.from_rotvec(rotvec)
    return np.asarray(r.as_quat())

def _rpy_to_rotvec(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r = R.from_euler(seq='xyz', angles=rpy, degrees=False)
    return np.asarray(r.as_rotvec())


class URRobot:

    def __init__(self, robot_ip: str) -> None:
        self._robot_ip = robot_ip
        self._controller = None
        self._receiver = None

        # Default parameters.
        self.tcp_speed = 0.25
        self.tcp_acceleration = 1.2
        self.joint_speed = 1.05
        self.joint_acceleration = 1.4

        # Servo parameters.
        self.velocity = 0.2  # NOT used in current version.
        self.acceleration = 0.2  # NOT used in current version.
        self.dt = 1.0 / 100
        self.lookahead_time = 0.2
        self.gain = 100

        self.servo_running = False

    def num_dofs(self) -> int:
        return 6

    def set_tcp_speed(self, tcp_speed: float) -> None:
        self.tcp_speed = tcp_speed

    def set_tcp_acceleration(self, tcp_acceleration: float) -> None:
        self.tcp_acceleration = tcp_acceleration

    def set_joint_speed(self, joint_speed: float) -> None:
        self.joint_speed = joint_speed

    def set_joint_acceleration(self, joint_acceleration: float) -> None:
        self.joint_acceleration = joint_acceleration

    def set_dt(self, dt: float) -> None:
        self.dt = dt

    def set_lookahead_time(self, lookahead_time: float) -> None:
        self.lookahead_time = lookahead_time

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    def connect(self) -> bool:
        try:
            self._controller = rtde_control.RTDEControlInterface(self._robot_ip)
            self._receiver = rtde_receive.RTDEReceiveInterface(self._robot_ip)
            logger.success(f"The UR5e robot {self._robot_ip} connected successfully.")
            return True
        except Exception:
            logger.exception(f"Failed to connect to the UR5e robot {self._robot_ip}.")
            raise ConnectionError("Failed to connect to the UR5e robot.")

    def disconnect(self) -> None:
        self._controller.disconnect()
        logger.success(f"The UR5e robot {self._robot_ip} disconnected successfully.")

    def get_current_joint(self):
        return np.asarray(self._receiver.getActualQ())

    def reach_target_joint(self, target_joint: np.ndarray, asynchronous: bool = False) -> bool:
        if self.servo_running:
            self._stop_servo()

        return self._controller.moveJ(
            target_joint,
            speed=self.joint_speed,
            acceleration=self.joint_acceleration,
            asynchronous=asynchronous,
        )

    def get_tool_cartesian_pose(self):
        rotvec_pose = np.asarray(self._receiver.getActualTCPPose())
        rotvec_quat_xyzw = _rotvec_to_quat_xyzw(rotvec_pose[3:])
        return np.concatenate((rotvec_pose[:3], rotvec_quat_xyzw))
    
    def get_tool_cartesian_pose_xyzrpy(self):
        rotvec_pose = np.asarray(self._receiver.getActualTCPPose())
        rpy = _rotvec_to_rpy(rotvec_pose[3:])
        return np.concatenate((rotvec_pose[:3], rpy))

    def reach_tool_cartesian_pose(self, pose, asynchronous: bool = False) -> bool:
        xyz_rpy_pose = pose
        xyz_rotvec_pose = np.concatenate((xyz_rpy_pose[:3], _rpy_to_rotvec(xyz_rpy_pose[3:])))

        if self.servo_running:
            self._stop_servo()

        validity_check = self._controller.isPoseWithinSafetyLimits(xyz_rotvec_pose)
        if validity_check:
            return self._controller.moveL(
                xyz_rotvec_pose,
                speed=self.tcp_speed,
                acceleration=self.tcp_acceleration,
                asynchronous=asynchronous,
            )
        else:
            logger.error("The target tool Cartesian pose is not within safety limits.")
            return False

    def get_forward_kinematics(self, joints_radian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        rotvec_pose = self._controller.getForwardKinematics(joints_radian, tcp_offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rotvec_quat_xyzw = _rotvec_to_quat_xyzw(rotvec_pose[3:])
        quat_xyzw_pose = np.concatenate((rotvec_pose[:3], rotvec_quat_xyzw))
        return quat_xyzw_pose

    def sync_target_joint(self, target_joint: npt.NDArray[np.float64]) -> None:
        self.servo_running = True
        t_start = self._controller.initPeriod()
        self._controller.servoJ(
            target_joint,
            self.velocity,  # NOT used in current version
            self.acceleration,  # NOT used in current version
            self.dt,
            self.lookahead_time,
            self.gain,
        )
        self._controller.waitPeriod(t_start)

    def sync_tool_cartesian_pose(self, tool_pose: npt.NDArray[np.float64]) -> None:
        assert len(tool_pose) == 6, "Please check the input value."

        rotvec = _rpy_to_rotvec(tool_pose[3:])
        xyz_rotvec_pose = np.concatenate((tool_pose[:3], rotvec))

        self.servo_running = True
        t_start = self._controller.initPeriod()
        self._controller.servoL(
            xyz_rotvec_pose,
            self.velocity,  # NOT used in current version
            self.acceleration,  # NOT used in current version
            self.dt,
            self.lookahead_time,
            self.gain,
        )
        self._controller.waitPeriod(t_start)

    def _stop_servo(self) -> None:
        self._controller.servoStop()
        self.servo_running = False
    
    def get_ee_pose_from_joint(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.get_forward_kinematics(joints)
