

class StationLoader:
    """Station loader."""

    def __init__(self, cfg_dict: dict = {}) -> None:
        """Initialize the station loader.

        Args:
            cfg_dict: The configuration dictionary.
        """
        self.cfg_dict = cfg_dict

    def generate_station_handle(self):
        """Generate the station handle.

        Returns:
            URStation: The station handle.
        """
        config = self.cfg_dict

        if "UR" == config["basic"]["station_type"]:
            from ..ur_station import URStation

            # Extract the config section from the TOML structure
            # The TOML has [config.robot.single], [config.hand.single], etc.
            # So we need to access config["config"] first
            config_section = config.get("config", {})
            
            print(f"config:{config}")
            robot_station = URStation(
                robot_dict=config_section.get("robot", {}),
                hand_dict=config_section.get("hand", {}),
                camera_dict=config_section.get("camera", {}),
                control_rate_hz=config_section.get("basic", {}).get("control_rate_hz", 100.0)
            )
        else:
            raise ValueError("Invalid Station Type")
        print("Work Station loaded...")
        return robot_station
