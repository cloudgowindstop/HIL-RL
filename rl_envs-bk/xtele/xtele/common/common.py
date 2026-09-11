import os


class Path:
    CONFIG_DIR = os.path.expanduser("~/.config/xhumanoid/xtele/default.toml")


class SerialParams:
    BAUDRATE = 2000000
    PORT = "/dev/ttyUSB0"


TELE_TYPE_MAP = {
    "ur_sg": {
        "module": "xtele.station.singleUR.single_ur_station",
        "class": "TeleSingUR",
    },
}


DYNAMIC_MAP = {

}
