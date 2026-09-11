import numpy as np
import time

from xtele.common.common import TELE_TYPE_MAP
from xtele.core.config_manager import ConfigManager
from xtele.station.TeleStation import TeleStation

ACT_MODE = 0
SYNC_MODE = 1


class TeleCore:
    def __init__(self, config_path=None):
        """
        初始化遥操作核心
        """
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        tele_type = config["basic"]["station_type"]
        module = __import__(
            TELE_TYPE_MAP[tele_type]["module"],
            fromlist=[TELE_TYPE_MAP[tele_type]["class"]],
        )
        cls = getattr(module, TELE_TYPE_MAP[tele_type]["class"])

        self.tele_agent: TeleStation = cls(config)
        self.mode = ACT_MODE

    def act(self) -> np.ndarray:
        """
        返回遥操作变量值
        """
        return self.tele_agent.act()

    def act_dict(self) -> dict:
        """
        返回遥操作变量名及变量值字典
        """
        return self.tele_agent.act_dict()

    def switch_reverse(self):
        """
        切换到反向同步模式
        """
        assert self.tele_agent.is_support_reverse(), (
            "Reverse func is unsupportable on this tele-station."
        )
        self.tele_agent.switch_reverse_mode()
        self.mode = SYNC_MODE

    def switch_act(self) -> None:
        """
        切换到读值模式
        """
        self.tele_agent.switch_act_mode()
        self.mode = ACT_MODE

    def sync_position(self, target_position) -> None:
        assert self.mode, "Have to switch mode to reverse mode first."
        self.tele_agent.sync_position(target_position)

    def reach_position(self, target_position, freq) -> None:
        assert self.mode, "Have to switch mode to reverse mode first."
        crrt_pos = self.act()
        num_stp = int(100 / freq)
        stp_array = np.linspace(crrt_pos, target_position, num_stp)
        for stp in stp_array:
            self.sync_position(stp)
            time.sleep(1 / 100)
