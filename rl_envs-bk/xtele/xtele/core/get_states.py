#!/usr/bin/env python3
"""
Creator: Shane Xie
Developer
    -
First create: 2025-04-10
Last  modify: 2025-07-11

Version History:
v1.6.0 - Support for teleoperation product.
"""

import curses
import sys
import time

from xtele.core.integrate_module import TeleCore


class GetStates:
    def __init__(self):
        self.tele_agent = TeleCore()

    def display_joints(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(1)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

        log_buffer = []

        def log_hook(message):
            for line in message.strip().split("\n"):
                if line:
                    log_buffer.append(line)
                    if len(log_buffer) > 100:
                        log_buffer.pop(0)

        sys.stdout.write = lambda msg: log_hook(msg)
        sys.stderr.write = lambda msg: log_hook(msg)

        while True:
            try:
                height, width = stdscr.getmaxyx()

                stdscr.erase()
                stdscr.addstr(0, 0, "╭" + "─" * (width - 2) + "╮", curses.color_pair(1))
                title_content = time.strftime(
                    "xTele-Getstates Monitor [%Y-%m-%d %H:%M:%S]"
                )
                centered_title = title_content.center(width - 2)
                stdscr.addstr(1, 1, centered_title, curses.color_pair(1))
                stdscr.addstr(2, 0, "╰" + "─" * (width - 2) + "╯", curses.color_pair(1))

                try:
                    result = self.tele_agent.act_dict()

                    data_rows = 0
                    if result:
                        metrics = [
                            (len(f"{k}: "), len(f"{v:+.4f}")) for k, v in result.items()
                        ]

                        max_key = max(m[0] for m in metrics)
                        max_val = max(m[1] for m in metrics)
                        base_width = max_key + max_val
                        available_width = width - 2
                        cols = max(
                            1,
                            min(len(result), (available_width + 2) // (base_width + 2)),
                        )

                        data_rows = (len(result) + cols - 1) // cols
                        data_rows = min(data_rows, height - 6)

                        while cols > 1:
                            col_table = [{"key": 0, "val": 0} for _ in range(cols)]
                            for idx, (k_len, v_len) in enumerate(metrics):
                                col = idx % cols
                                col_table[col]["key"] = max(
                                    col_table[col]["key"], k_len
                                )
                                col_table[col]["val"] = max(
                                    col_table[col]["val"], v_len
                                )

                            total_width = sum(
                                ct["key"] + ct["val"] for ct in col_table
                            ) + 2 * (cols - 1)
                            if total_width <= available_width:
                                break
                            cols -= 1
                    else:
                        cols = 1

                    current_row = 3
                    if result:
                        col_table = [{"key": 0, "val": 0} for _ in range(cols)]
                        for idx, (key, value) in enumerate(result.items()):
                            col = idx % cols
                            key_len = len(f"{key}: ")
                            val_len = len(f"{value:+.4f}")
                            col_table[col]["key"] = max(col_table[col]["key"], key_len)
                            col_table[col]["val"] = max(col_table[col]["val"], val_len)

                        for idx, (key, value) in enumerate(result.items()):
                            col = idx % cols
                            x_pos = (
                                sum(ct["key"] + ct["val"] for ct in col_table[:col])
                                + 2 * col
                            )

                            if current_row + (idx // cols) >= height - 3:
                                break

                            stdscr.addstr(
                                current_row + (idx // cols),
                                x_pos,
                                f"{key}: ",
                                curses.color_pair(2),
                            )
                            stdscr.addstr(
                                current_row + (idx // cols),
                                x_pos + col_table[col]["key"],
                                f"{value:+.4f}",
                            )

                    log_start_row = 3 + data_rows + 2
                    available_log_rows = max(1, height - log_start_row - 1)
                    stdscr.addstr(log_start_row - 1, 0, "─" * width)
                    for i in range(min(available_log_rows, len(log_buffer))):
                        stdscr.addstr(
                            log_start_row + i,
                            0,
                            log_buffer[-(available_log_rows - i)].ljust(width)[
                                : width - 1
                            ],
                        )

                except curses.error:
                    continue

                stdscr.refresh()
                time.sleep(0.05)
            except curses.error:
                continue
            except KeyboardInterrupt:
                self.tele_agent.tele_agent.close()
                for log in log_buffer:
                    print(log)
                exit(0)
            finally:
                sys.stdout.write = sys.__stdout__.write
                sys.stderr.write = sys.__stderr__.write

    def run_test(self):
        curses.wrapper(self.display_joints)
        # while True:
        #     try:
        #         print(self.tele_agent.act_dict())
        #         time.sleep(0.1)
        #     except KeyboardInterrupt:
        #         self.tele_agent.tele_agent.close()
        #         exit(0)


if __name__ == "__main__":
    self_test = GetStates()
    self_test.run_test()
