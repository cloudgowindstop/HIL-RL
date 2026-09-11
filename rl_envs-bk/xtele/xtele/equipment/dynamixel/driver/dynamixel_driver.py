#!/usr/bin/env python3
import enum
import os
import queue
import time
import warnings
from collections import deque
from threading import Event, Lock, Thread
from typing import Dict, Sequence, List
import subprocess

import numpy as np

from xtele.equipment.dynamixel.dynamixel_sdk.group_sync_read import GroupSyncRead
from xtele.equipment.dynamixel.dynamixel_sdk.group_sync_write import GroupSyncWrite
from xtele.equipment.dynamixel.dynamixel_sdk.packet_handler import PacketHandler
from xtele.equipment.dynamixel.dynamixel_sdk.port_handler import PortHandler
from xtele.equipment.dynamixel.dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_OPERATING_MODE = 11
LEN_OPERATING_MODE = 1

ADDR_TORQUE_ENABLE = 64
LEN_TORQUE_ENABLE = 1
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

ADDR_HARDWARE_ERROR_STATUS = 70
LEN_HARDWARE_ERROR_STATUS = 1

ADDR_GOAL_PWM = 100
LEN_GOAL_PWM = 2

ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2

ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4

ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4

HARDWARE_ERROR_MAP = {
    0: "Input Voltage",
    2: "Overheating",
    3: "Motor Encoder",
    4: "Electrical Shock",
    5: "Overload",
}


def find_ttyusb(full_path):
    base_path = os.path.dirname(full_path)
    port_name = os.path.basename(full_path)
    if not os.path.exists(full_path):
        raise RuntimeError(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        if os.path.islink(full_path):
            resolved_path = os.readlink(full_path)
            actual_device = os.path.basename(resolved_path)
        else:
            resolved_path = full_path
            actual_device = port_name

        if actual_device.startswith("ttyUSB"):
            return actual_device
        else:
            warnings.warn(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}.",
                RuntimeWarning,
                2,
            )
            return None
    except Exception as e:
        raise RuntimeError(
            f"Unable to resolve the symbolic link for '{port_name}'. {e}"
        )


class OperatingMode(enum.Enum):
    CURRENT = 0
    VELOCITY = 1
    POSITION = 3
    EXTENDED_POSITION = 4
    CURRENT_CONTROLLED_POSITION = 5
    PWM = 16
    UNKNOWN = -1


class DynamixelDriver:
    _lock = Lock()

    def __init__(
        self, ids: Sequence[int], port: str = "/dev/ttyUSB0", baudrate: int = 2000000
    ):
        """
        Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._time_len = 100  # Recording length
        self._pos_filter_len = 10  # 建议不大于10

        self._resolution = 4096
        self._max_velo = 0.5  # in rad
        self._max_acc = 0.7  # in rad

        self._ids = ids
        self._port = port
        self._baudrate = baudrate

        self._position = None
        self._velocity = None

        self.time_window = deque(maxlen=100)  # 近100次写线程耗时

        ttyUSBx = find_ttyusb(self._port)
        if ttyUSBx is not None:
            result = subprocess.run(
                f"cat /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            ttyUSB_latency_timer = int(result.stdout)
            if ttyUSB_latency_timer != 1:
                subprocess.run(
                    [
                        "pkexec",
                        "tee",
                        f"/sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer",
                    ],
                    input="1",
                    text=True,
                )

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncReadPosition = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncReadVelocity = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_VELOCITY,
        )
        self._groupSyncWriteOperatingMode = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_OPERATING_MODE,
            LEN_OPERATING_MODE,
        )
        self._groupSyncWriteTorqueEnable = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_TORQUE_ENABLE,
            LEN_TORQUE_ENABLE,
        )
        self._groupSyncWritePWM = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_PWM,
            LEN_GOAL_PWM,
        )
        self._groupSyncWritePosition = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self._groupSyncWriteCurrent = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncReadPosition.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        for dxl_id in self._ids:
            if not self._groupSyncReadVelocity.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = {dxl_id: False for dxl_id in self._ids}
        self._operating_mode = {dxl_id: OperatingMode.UNKNOWN for dxl_id in self._ids}

        self._damping_info = {}
        self._pos_update_timestamp = None
        self._vel_update_timestamp = None

        self._communication_task = queue.Queue(maxsize=16)
        self._stop_thread_flag = Event()
        self._communication_thread = Thread(
            target=self._communication_worker, daemon=True
        )
        self._communication_thread.start()

    def _communication_worker(self) -> None:
        pos_len_packet = len(self._ids) * (4 + LEN_PRESENT_POSITION) + 8
        vel_len_packet = len(self._ids) * (4 + LEN_PRESENT_VELOCITY) + 8
        sts_len_packet = len(self._ids) * (4 + LEN_HARDWARE_ERROR_STATUS) + 8
        error_counter = 0
        while not self._stop_thread_flag.is_set():
            loop_start = time.perf_counter()

            pos_instr_packet = self._constr_fast_instr(
                ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            self._portHandler.writePort(pos_instr_packet)
            pos_response = self._portHandler.readPort(pos_len_packet)

            if pos_response and len(pos_response) == pos_len_packet:
                if not self._read_position(pos_response):
                    error_counter += 1
                pass

            vel_instr_packet = self._constr_fast_instr(
                ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
            )
            self._portHandler.writePort(vel_instr_packet)
            vel_response = self._portHandler.readPort(vel_len_packet)
            if vel_response and len(vel_response) == vel_len_packet:
                if not self._read_velocity(vel_response):
                    error_counter += 1
                pass

            if error_counter > 50:
                sts_instr_packet = self._constr_fast_instr(
                    ADDR_HARDWARE_ERROR_STATUS, LEN_HARDWARE_ERROR_STATUS
                )
                self._portHandler.writePort(sts_instr_packet)
                # sts_response = self._portHandler.readPort(sts_len_packet)
                # if sts_response and len(sts_response) == sts_len_packet:
                #     self._read_status(sts_response)
                error_counter = 0

            if not self._communication_task.empty():
                self._communication_task.get()()

            loop_time = time.perf_counter() - loop_start
            self.time_window.append(loop_time)
            time.sleep(0.005)

    def _read_position(self, response: bytes) -> bool:
        position = self._decode_status_packet(response, LEN_PRESENT_POSITION)
        if position:
            self._position = position
            self._pos_update_timestamp = time.perf_counter()
            return True
        else:
            return False

    def _read_velocity(self, response: bytes) -> bool:
        velocity = self._decode_status_packet(response, LEN_PRESENT_VELOCITY)
        if velocity:
            self._velocity = velocity
            self._vel_update_timestamp = time.perf_counter()
            return True
        else:
            return False

    def _read_status(self, response: bytes) -> None:
        status = self._decode_status_packet(
            response, LEN_HARDWARE_ERROR_STATUS, ignore_error=True
        )
        if status and np.any(status != 0):
            print("\n\033[1;31m" + "=" * 60)
            print("⚠️  CRITICAL DYNAMIXEL SERVO ERRORS DETECTED".center(60))
            print("=" * 60 + "\033[0m")
            print("\033[33mPlease reboot the affected servo motors immediately:\033[0m")

            for dxl_id, byte_sts in zip(self._ids, status):
                active_errors = []
                for bit, err_name in HARDWARE_ERROR_MAP.items():
                    if byte_sts & (1 << bit):
                        active_errors.append(f"{err_name} Error")
                if active_errors:
                    for error in active_errors:
                        print(f"ID [{dxl_id}]: {error}")

            print("\033[1;31m" + "=" * 60 + "\033[0m\n")
            os._exit(-1)

    def _constr_fast_instr(self, addr, length):
        """
        :param addr: 寄存器地址
        :param length: 数据长度
        :return 编码的Instruction packet
        """
        # Construct the Fast Sync Read packet
        packet = [
            0xFF,  # 0
            0xFF,  # 1
            0xFD,  # 2
            0x00,  # 3
            0xFE,  # 4
            0x00,  # 5
            0x00,  # 6
            0x8A,  # 7
            0x00,  # 8
            0x00,  # 9
            0x00,  # 10
            0x00,  # 11
        ]
        packet[5] = len(self._ids) + 4 + 1 + 2  # Length of the packet
        packet[6] = 0  # Length checksum
        packet[8] = addr & 0xFF  # Starting address
        packet[9] = (addr >> 8) & 0xFF  # Starting address
        packet[10] = length & 0xFF
        packet[11] = (length >> 8) & 0xFF
        for dxl_id in self._ids:
            packet.append(dxl_id)

        crc = self._packetHandler.updateCRC(0, packet, len(packet))
        packet.append(crc & 0xFF)
        packet.append((crc >> 8) & 0xFF)
        return packet

    def _decode_status_packet(
        self, response: bytes, length: int, ignore_error=False
    ) -> List[int]:
        """
        Status packet解码
        """
        values = []
        unit_length = 4 + length
        index = 8
        while index < len(response):
            err = response[index]
            sts_data = response[index + 2 : index + 2 + length]
            crc_data = response[index + 2 + length] | (
                response[index + 3 + length] << 8
            )

            if not ignore_error and err != 0x00:
                return []

            # CRC校验, status_packet为当前舵机的Status Packet
            status_packet = response[0 : index + 2 + length]
            crc_calculated = self._packetHandler.updateCRC(
                0, status_packet, len(status_packet)
            )
            if crc_data != crc_calculated:
                return []

            decoded_value = int.from_bytes(sts_data, byteorder="little", signed=True)
            values.append(decoded_value)
            index += unit_length
        return values

    def set_operating_mode(self, dxl_ids: Sequence[int], mode: Sequence[OperatingMode]):
        assert len(mode) == len(dxl_ids), (
            "The length of mode must match the number of servos"
        )

        self._groupSyncWriteOperatingMode.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWriteOperatingMode.clearParam()

        for dxl_id, state in zip(dxl_ids, mode):
            param_operating_mode = [state.value]

            dxl_addparam_result = self._groupSyncWriteOperatingMode.addParam(
                dxl_id, param_operating_mode
            )
            if not dxl_addparam_result:
                raise RuntimeError(f"Failed to set mode for Dynamixel with ID {dxl_id}")

        self._groupSyncWriteOperatingMode.is_writable.clear()
        self._communication_task.put(
            lambda: (
                self._process_group_response(
                    self._groupSyncWriteOperatingMode.txPacket()
                ),
                self._groupSyncWriteOperatingMode.is_writable.set(),
            )
        )

        for dxl_id, state in zip(dxl_ids, mode):
            self._operating_mode[dxl_id] = state

    def set_torque_mode(self, dxl_ids: Sequence[int], enable: Sequence[bool]):
        assert len(enable) == len(dxl_ids), (
            "The length of enable must match the number of servos"
        )

        self._groupSyncWriteTorqueEnable.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWriteTorqueEnable.clearParam()

        for dxl_id, state in zip(dxl_ids, enable):
            param_torque_mode = [TORQUE_ENABLE if state else TORQUE_DISABLE]

            dxl_addparam_result = self._groupSyncWriteTorqueEnable.addParam(
                dxl_id, param_torque_mode
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set torque_enable for Dynamixel with ID {dxl_id}"
                )

        self._groupSyncWriteTorqueEnable.is_writable.clear()
        self._communication_task.put(
            lambda: (
                self._process_group_response(
                    self._groupSyncWriteTorqueEnable.txPacket()
                ),
                self._groupSyncWriteTorqueEnable.is_writable.set(),
            )
        )

        for dxl_id, state in zip(dxl_ids, enable):
            self._torque_enabled[dxl_id] = state

    def set_position(self, dxl_ids: Sequence[int], goal_positions: Sequence[int]):
        assert len(goal_positions) == len(dxl_ids), (
            "The length of goal_positions must match the number of servos"
        )

        self._groupSyncWritePosition.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWritePosition.clearParam()

        if checked_ids := [
            dxl_id
            for dxl_id in dxl_ids
            if self._operating_mode[dxl_id] != OperatingMode.EXTENDED_POSITION
        ]:
            self.set_operating_mode(
                checked_ids,
                [OperatingMode.EXTENDED_POSITION for _ in checked_ids],
            )

        if checked_ids := [
            dxl_id for dxl_id in dxl_ids if not self._torque_enabled[dxl_id]
        ]:
            self.set_torque_mode(checked_ids, [True for _ in checked_ids])

        for dxl_id, position in zip(dxl_ids, goal_positions):
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position)),
                DXL_HIBYTE(DXL_LOWORD(position)),
                DXL_LOBYTE(DXL_HIWORD(position)),
                DXL_HIBYTE(DXL_HIWORD(position)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._groupSyncWritePosition.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set position for Dynamixel with ID {dxl_id}"
                )

        self._groupSyncWritePosition.is_writable.clear()
        self._communication_task.put(
            lambda: (
                self._process_group_response(self._groupSyncWritePosition.txPacket()),
                self._groupSyncWritePosition.is_writable.set(),
            )
        )

    def set_current(self, dxl_ids: Sequence[int], goal_currents: Sequence[int]):
        assert len(goal_currents) == len(dxl_ids), (
            "The length of goal_currents must match the number of servos"
        )

        self._groupSyncWriteCurrent.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWriteCurrent.clearParam()

        if checked_ids := [
            dxl_id
            for dxl_id in dxl_ids
            if self._operating_mode[dxl_id] != OperatingMode.CURRENT
        ]:
            self.set_operating_mode(
                checked_ids, [OperatingMode.CURRENT for _ in checked_ids]
            )

        if checked_ids := [
            dxl_id for dxl_id in dxl_ids if not self._torque_enabled[dxl_id]
        ]:
            self.set_torque_mode(checked_ids, [True for _ in checked_ids])

        for dxl_id, goal_current in zip(dxl_ids, goal_currents):
            param_goal_current = [
                DXL_LOBYTE(goal_current),
                DXL_HIBYTE(goal_current),
            ]

            if not self._groupSyncWriteCurrent.addParam(dxl_id, param_goal_current):
                raise RuntimeError(
                    f"Failed to add goal current parameter for Dynamixel ID {dxl_id}"
                )

        self._groupSyncWriteCurrent.is_writable.clear()
        self._communication_task.put(
            lambda: (
                self._process_group_response(self._groupSyncWriteCurrent.txPacket()),
                self._groupSyncWriteCurrent.is_writable.set(),
            )
        )

    def set_position_current(
        self,
        dxl_ids: Sequence[int],
        goal_positions: Sequence[float],
        goal_currents: Sequence[int],
    ):
        assert len(goal_currents) == len(goal_positions) == len(dxl_ids), (
            "The length of goal_positions or goal_currents must match the number of servos"
        )

        self._groupSyncWritePosition.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWritePosition.clearParam()
        self._groupSyncWriteCurrent.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWriteCurrent.clearParam()

        if checked_ids := [
            dxl_id
            for dxl_id in dxl_ids
            if self._operating_mode[dxl_id] != OperatingMode.CURRENT_CONTROLLED_POSITION
        ]:
            self.set_operating_mode(
                checked_ids,
                [OperatingMode.CURRENT_CONTROLLED_POSITION for _ in checked_ids],
            )

        if checked_ids := [
            dxl_id for dxl_id in dxl_ids if not self._torque_enabled[dxl_id]
        ]:
            self.set_torque_mode(checked_ids, [True for _ in checked_ids])

        for dxl_id, goal_position, goal_current in zip(
            dxl_ids, goal_positions, goal_currents
        ):
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(goal_position)),
                DXL_HIBYTE(DXL_LOWORD(goal_position)),
                DXL_LOBYTE(DXL_HIWORD(goal_position)),
                DXL_HIBYTE(DXL_HIWORD(goal_position)),
            ]

            param_goal_current = [
                DXL_LOBYTE(goal_current),
                DXL_HIBYTE(goal_current),
            ]

            if not self._groupSyncWritePosition.addParam(dxl_id, param_goal_position):
                raise RuntimeError(
                    f"Failed to add goal position parameter for Dynamixel ID {dxl_id}"
                )
            if not self._groupSyncWriteCurrent.addParam(dxl_id, param_goal_current):
                raise RuntimeError(
                    f"Failed to add goal current parameter for Dynamixel ID {dxl_id}"
                )

        self._groupSyncWritePosition.is_writable.clear()
        self._groupSyncWriteCurrent.is_writable.clear()
        self._communication_task.put(
            lambda: (
                self._process_group_response(self._groupSyncWritePosition.txPacket()),
                self._groupSyncWritePosition.is_writable.set(),
                self._process_group_response(self._groupSyncWriteCurrent.txPacket()),
                self._groupSyncWriteCurrent.is_writable.set(),
            )
        )

    def set_pwm(self, dxl_ids: Sequence[int], goal_pwms: Sequence[int]):
        assert len(goal_pwms) == len(dxl_ids), (
            "The length of pwm_value must match the number of servos"
        )

        self._groupSyncWritePWM.is_writable.wait()  # wait for the last write to complete
        self._groupSyncWritePWM.clearParam()

        for dxl_id, goal_pwm in zip(dxl_ids, goal_pwms):
            param_goal_pwm = [
                DXL_LOBYTE(goal_pwm),
                DXL_HIBYTE(goal_pwm),
            ]

            dxl_addparam_result = self._groupSyncWritePWM.addParam(
                dxl_id, param_goal_pwm
            )
            if not dxl_addparam_result:
                raise RuntimeError(f"Failed to set pwm for Dynamixel with ID {dxl_id}")

        self._groupSyncWritePWM.is_writable.clear()
        self._communication_task.put(
            lambda: (
                self._process_group_response(self._groupSyncWritePWM.txPacket()),
                self._groupSyncWritePWM.is_writable.set(),
            )
        )

    def _process_response(self, dxl_id: int, dxl_comm_result: int, dxl_error: int):
        if dxl_comm_result != COMM_SUCCESS:
            raise ConnectionError(
                f"dxl_comm_result for motor {dxl_id}: {self._packetHandler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(f"dxl error {dxl_id}: {dxl_error}")
            raise ConnectionError(
                f"dynamixel error for motor {dxl_id}: {self._packetHandler.getTxRxResult(dxl_error)}"
            )

    def _process_group_response(self, dxl_comm_result: int):
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(
                f"Failed to group write/read: "
                f"{self._packetHandler.getTxRxResult(dxl_comm_result)}"
            )

    def get_position(self) -> Dict[int, int]:
        while self._position is None:
            time.sleep(0.001)
        _j = self._position.copy()
        return {dxl_id: pos for dxl_id, pos in zip(self._ids, _j)}

    def get_velocity(self) -> Dict[int, int]:
        while self._velocity is None:
            time.sleep(0.001)
        with self._lock:
            _v = self._velocity.copy()
        return {dxl_id: vel for dxl_id, vel in zip(self._ids, _v)}

    def get_frequency(self) -> Sequence[float]:
        return [
            len(self.time_window) / sum(self.time_window) if self.time_window else 0.0,
            np.around(self._pos_update_timestamp - time.perf_counter(), 3)
            if self._pos_update_timestamp
            else -999.0,
            np.around(self._vel_update_timestamp - time.perf_counter(), 3)
            if self._vel_update_timestamp
            else -999.0,
        ]

    def set_damping_pos(self, dxl_ids: Sequence[int], target_pos: Sequence[int]):
        with self._lock:
            for dxl_id, pos in zip(dxl_ids, target_pos):
                self._damping_info[dxl_id] = pos

    def _damping_lock(self):
        if not self._damping_info:
            return

        with self._lock:
            dxl_ids = list(self._damping_info.keys())
            target_pos = list(self._damping_info.values())

        c_pos = self.get_position()
        pwm_value, current_value = [], []

        for dxl_id, t_pos in zip(dxl_ids, target_pos):
            delta = abs(c_pos[dxl_id] - t_pos)
            pwm_value.append(20 + int(885 * (delta / 4096)))
            current_value.append(int(1193 * (delta / 2048)))

        self.set_position_current(dxl_ids, target_pos, current_value)
        self.set_pwm(dxl_ids, pwm_value)

    def close(self):
        self._stop_thread_flag.set()
        self._communication_thread.join()
        self._portHandler.closePort()
