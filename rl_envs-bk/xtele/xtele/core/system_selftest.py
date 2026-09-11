#!/usr/bin/env python3
"""
Description:
    Test serial connection.

Creator: Jacob Ji
Developer
    -
First create: 2025-05-29
Last  modify: 2025-07-11

Version History:
v1.6.0 - Support for teleoperation product.
"""

import os
import re
import subprocess
import sys

import tomlkit

from xtele.core.integrate_module import TeleCore


class SystemSelftest:
    def __init__(self):
        pass

    def run_test(self):
        self.check_serial_port()
        self.check_motor_power()

    def check_serial_port(self):
        """检查串口设备是否存在"""
        try:
            ports = self.list_serial_ports()
            serial_port = self.select_serial_port(ports)
            self.update_config_file(serial_port)
        except Exception as e:
            print(f"\n错误: {e}", file=sys.stderr)
            sys.exit(-1)
        return True

    def check_motor_power(self):
        """初始化TeleCore并执行动作测试"""
        try:
            print("\033[34m初始化同构臂控制器...\033[0m")
            m_tele = TeleCore()

            print("\033[36m执行硬件自检...\033[0m")
            result = m_tele.act_dict()
            print("Current states: ")
            print(result)

            print("\033[32m硬件自检成功完成\033[0m")
            return True
        except Exception as e:
            print(f"\033[31m初始化失败: {str(e)}\033[0m")
            print("可能原因：")
            print("1. 同构臂电源未连接")
            print("2. 硬件损坏")
            return False

    def list_serial_ports(self):
        """列出所有通过ID识别的串口设备"""
        try:
            result = subprocess.run(
                ["ls", "/dev/serial/by-id/"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            ports = result.stdout.strip().split("\n")
            if not ports or ports == [""]:
                raise ValueError("未检测到任何串口设备")
            print(f"\033[32m检测到串口设备: {ports}\033[0m")
            return ports
        except subprocess.CalledProcessError as e:
            print("\033[31m错误：串口设备不存在: \033[0m")
            print("请检查：")
            print("1. 串口线是否正确连接到电脑")
            print("2. 设备是否上电")
            raise RuntimeError(f"获取串口列表失败: {e.stderr}") from e

    def select_serial_port(self, ports):
        """让用户选择串口并返回选择的设备路径"""
        print("\n检测到以下串口设备:")
        for i, port in enumerate(ports, 1):
            print(f"{i}. {port}")

        while True:
            choice = input("请输入要使用的串口序号:")
            if bool(re.match(r"^(0 |[-+]?[1-9]\d*)$", choice)):
                choice = int(choice)
                if 1 <= int(choice) <= len(ports):
                    return ports[choice - 1]
            print("\033[31m错误: 请输入正确的串口序号！\033[0m")

    def update_config_file(self, target_port):
        """更新配置文件中的串口设置"""
        config_path = os.path.expanduser("~/.config/xhumanoid/xtele/default.toml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = tomlkit.parse(f.read())
            modify = False
            for arm_type in config["xlinker"].values():
                for arm_name in arm_type.values():
                    if os.path.basename(arm_name["port"]) != target_port:
                        arm_name["port"] = "/dev/serial/by-id/" + target_port
                        modify = True
        except Exception as e:
            raise RuntimeError(f"配置文件解析失败: {e}") from e

        try:
            if modify:
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(tomlkit.dumps(config))
        except Exception as e:
            raise RuntimeError(f"配置文件保存失败: {e}") from e
