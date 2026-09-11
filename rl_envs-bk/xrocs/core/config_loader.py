"""Configuration loading module for the xROCS framework.

This module provides utilities for loading configuration settings from TOML files.
It includes a ConfigLoader class and pre-inints | Nostantiated configuration objects.
"""

import tomli as tomllib
# from xrocs.utils.path_manager import path_manager

__all__ = ["config", "config_loader"]


class ConfigLoader:
    """Load the configuration file."""

    def __init__(self, config_path=None) -> None:
        assert config_path is not None, "config_path is required"
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def get_config(self) -> dict:
        """Get the configuration.

        Returns:
            dict: The configuration.
        """
        return self.config


# config_loader = ConfigLoader()
# config = config_loader.config
