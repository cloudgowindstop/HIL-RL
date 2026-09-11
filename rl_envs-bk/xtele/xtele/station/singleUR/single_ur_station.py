from typing import Sequence

from xtele.equipment.dynamixel.linker_agent import LinkerAgent, GripperAgent
from xtele.equipment.dynamixel.driver.dynamixel_driver import DynamixelDriver
from xtele.station.TeleStation import TeleStation


class TeleSingUR(TeleStation):
    """
    Teleoperation station for a single UR robot.
    """

    def __init__(self, config: dict):
        """
        Initialize the single UR teleoperation station.

        Args:
            config (dict): Configuration dictionary for the station.
        """
        super().__init__(config)
        driver = DynamixelDriver(
            ids=self.config["xlinker"]["single"]["robot"]["joint_ids"],
            port=self.config["xlinker"]["single"]["robot"]["port"],
            baudrate=2000000,
        )
        self._equips["dynamixel_arm"] = LinkerAgent(
            linker_config=self.config["xlinker"]["single"]["robot"],
            dynamic_config=self.dynamic_config["single"],
            identifier="single",
            driver=driver,
        )
        self._equips["dynamixel_gripper"] = GripperAgent(
            linker_config=self.config["xlinker"]["single"]["robot"],
            identifier="gripper",
            driver=driver,
        )

        self._is_support_reverse = True

        self.EQUIP_ORDER = [
            "dynamixel_arm",
            "dynamixel_gripper",
        ]

    def set_position_torque(
        self,
        dxl_ids: Sequence[int],
        goal_positions: Sequence[float],
        goal_torques: Sequence[float],
    ):
        self._equips["dynamixel_arm"].set_position_torque(
            dxl_ids, goal_positions, goal_torques
        )

    def sync_position(self, target_position):
        self._equips["dynamixel_arm"].set_position(target_position[:-1])
        self._equips["dynamixel_gripper"].set_position(target_position[-1])
