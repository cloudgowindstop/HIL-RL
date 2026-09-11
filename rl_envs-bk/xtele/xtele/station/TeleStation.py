import time
from abc import ABC
from typing import Dict

import numpy as np
import tomli

from xtele.common.common import DYNAMIC_MAP
from xtele.equipment.tele_base import TeleBase


class TeleStation(ABC):
    """Base class for teleoperation stations."""

    def __init__(self, config: dict):
        """
        Initialize the teleoperation station.

        Args:
            config (dict): Configuration dictionary for the station.
        """
        self.config = config
        self._equips: Dict[str, TeleBase] = {}
        self._is_support_reverse = False
        self.EQUIP_ORDER = []

        if "xlinker" in self.config.keys():
            if "single" in self.config["xlinker"].keys():
                arm_num, arm_keys = 1, ["single"]
            elif {"bimanual", "dual"} & self.config["xlinker"].keys():
                arm_num, arm_keys = 2, ["left", "right"]
            else:
                raise ValueError(f"wrong linker type: {self.config['xlinker']}!")

            if (
                "enable_dynamic" in self.config["basic"].keys()
                and self.config["basic"]["enable_dynamic"]
            ):
                dynamic_config_path = DYNAMIC_MAP[self.config["basic"]["station_type"]][
                    "config_path"
                ]
                with open(dynamic_config_path, "rb") as f:
                    base_config = tomli.load(f)
                urdf_paths = DYNAMIC_MAP[self.config["basic"]["station_type"]][
                    "urdf_path"
                ]
                assert len(urdf_paths) == arm_num, (
                    "The length of urdf_paths must match the number of arms"
                )
                if arm_num == 1:
                    self.dynamic_config = {
                        "single": {
                            "enable_fb": self.config["basic"]["enable_fb"],
                            "urdf_path": urdf_paths[0],
                            **base_config,
                        }
                    }
                else:
                    self.dynamic_config = {
                        key: {
                            "enable_fb": self.config["basic"]["enable_fb"],
                            "urdf_path": path,
                            **base_config,
                        }
                        for key, path in zip(arm_keys, urdf_paths)
                    }
                self._is_support_feedback = True
            else:
                self.dynamic_config = {key: None for key in arm_keys}
                self._is_support_feedback = False

    def act(self) -> np.ndarray:
        """
        Get the current values of all components.

        Returns:
            np.ndarray: Concatenated values of all components.
        """
        return np.concatenate([self._equips[key].act() for key in self.EQUIP_ORDER])

    def act_dict(self) -> dict:
        """
        Get the current values of all components as a dictionary.

        Returns:
            dict: Dictionary containing the values of all components.
        """
        out_dict = {"timestamp": np.around(time.perf_counter(), 3)}
        for value in self._equips.values():
            out_dict.update(value.act_dict())
        return out_dict

    def is_support_reverse(self) -> bool:
        """
        Check if reverse synchronization is supported.

        Returns:
            bool: True if reverse synchronization is supported, False otherwise.
        """
        return self._is_support_reverse

    def switch_reverse_mode(self) -> None:
        """
        Switch to reverse synchronization mode.

        Raises:
            AssertionError: If reverse synchronization is not supported.
        """
        assert self._is_support_reverse, (
            "Reverse func is unsupportable on this tele-station."
        )
        for name, value in self._equips.items():
            if "dynamixel" in name:
                self._equips[name].activate_torque()

    def switch_act_mode(self) -> None:
        """
        Switch to read value mode.
        """
        for name, value in self._equips.items():
            if "dynamixel" in name:
                self._equips[name].deactivate_torque()

    def sync_position(self, target_position: np.ndarray) -> None:
        """
        Perform reverse synchronization of actions.

        Args:
            target_position (np.ndarray): Target position for synchronization.
        """
        assert self._is_support_reverse, (
            "Reverse func is unsupportable on this tele-station."
        )

    def set_fb_params(self, target_position: np.ndarray, ee_force: np.ndarray) -> None:
        assert self._is_support_feedback, (
            "Force feedback func is unsupportable on this tele-station."
        )

    def close(self):
        """
        Close the teleoperation station and release resources.
        """
        for equipment in self._equips.values():
            equipment.close()
            time.sleep(0.01)
            
