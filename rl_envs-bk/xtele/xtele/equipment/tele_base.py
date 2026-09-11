from abc import ABC, abstractmethod


class TeleBase(ABC):
    """
    Base class for teleoperation systems.
    """

    def __init__(self, identifier: str = ""):
        """
        Initialize the teleoperation base with a default status.
        """
        self.identifier = identifier
        self._crrt_status = {
            "code": 0,
            "info": "None",
        }

    def status(self) -> dict:
        """
        Retrieve the current status of the teleoperation system.

        Returns:
            dict: The current status with a code and information message.
        """
        return self._crrt_status

    @abstractmethod
    def act(self):
        """
        Retrieve the teleoperation system values, such as joint values or velocities.
        """
        pass

    @abstractmethod
    def act_dict(self):
        """
        Retrieve the teleoperation system values as a dictionary,
        mapping variable names to joint values or velocities.
        """
        pass

    def close(self):
        """
        关闭遥操系统
        """
        pass
