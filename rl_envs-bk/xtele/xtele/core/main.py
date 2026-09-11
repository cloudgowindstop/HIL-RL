import os
import argparse
import numpy as np
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    """
    Main function to handle different modes of operation.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    mode = args.mode
    config = os.path.expanduser(args.config)

    if args.openconfig:
        os.system(f"code {config}")
        return

    if mode == "cali":
        from xtele.core.calibration import Calibration

        print("Please place the tele-operator at its zero places.")
        m_cali = Calibration()
        m_cali.calibrate()
        print("Calibration has been finished.")

    elif mode == "getstates":
        from xtele.core.get_states import GetStates

        m_test = GetStates()
        m_test.run_test()

    elif mode == "systemtest":
        from xtele.core.system_selftest import SystemSelftest

        m_test = SystemSelftest()
        m_test.run_test()

    elif mode == "checksign":
        from xtele.quality.check_joint_sign import MujoCoBase

        proj_root_dir = Path(__file__).parent.parent.resolve()
        model_path = (
            proj_root_dir
            / "quality"
            / "model"
            / "ur_description"
            / "urdf"
            / "ur5_robot.xml"
        )
        m_quality = MujoCoBase(model_path=model_path)
        m_quality.run()

    elif mode == "remote":
        from xtele.core.remote_mode import TeleServer

        m_server = TeleServer()
        m_server.serve()
