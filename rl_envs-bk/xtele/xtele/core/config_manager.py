#!/usr/bin/env python3
"""
Creator: Jacob Ji
Developer
    - Shane Xie
First create: 2023-07-15
Last  modify: 2025-05-28

Version History:
v1.6.0 - Support for teleoperation product.

Requirement description:
    tomli==2.2.1
    tomli_w==1.2.0
"""

import os
import tomli
import tomli_w
from xtele.common.common import Path


class ConfigManager:
    """
    Manages the teleoperation configuration file.
    """

    def __init__(self, config_path=None):
        """
        Initialize the configuration manager instance and load the configuration
        from the default file path.

        Raises:
            RuntimeError: If the configuration file does not exist.
        """
        if config_path is None:
            self.config_path = Path.CONFIG_DIR
        else:
            self.config_path = config_path

        if os.path.exists(self.config_path):
            with open(self.config_path, "rb") as f:
                self.config = tomli.load(f)
        else:
            raise RuntimeError(
                "The configuration file does not exist, please create it."
            )

    def get_config(self) -> dict:
        """
        Retrieve the configuration dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return self.config

    def write_config(self) -> None:
        """
        Write the configuration dictionary back to the configuration file.
        """
        with open(self.config_path, "wb") as f:
            tomli_w.dump(self.config, f)


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.get_config()
    print(config)
