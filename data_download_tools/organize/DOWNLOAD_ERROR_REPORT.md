# 原始数据下载错误报告

## 1. 检查范围

- Excel：`HIL-RL/四阶段流水线看板.xlsx`
- 下载计划：`/media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/download_plan.jsonl`
- 下载结果：`/media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/download_results.jsonl`
- 单条日志目录：`/media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/logs/`
- 下载目录：`/media/jushen/project-rl-dataset/raw_data_0831download`

检查时间：2026-08-31。

下载结果：

- 计划记录：76 条
- 下载成功：41 条
- 验证失败：35 条
- 成功下载 HDF5：4367 个
- 成功下载数据量：约 2.57 TiB
- 当前目录状态：`complete=41`、`partial=35`、`missing=0`、`conflict=0`

35 条失败记录均显示：

```text
failed_verification: no trajectory.hdf5 found
```

检查单条日志后确认，`bcecmd` 对不存在的 BOS 前缀返回退出码 0，并显示 `[0] success, [0] failure`。下载程序随后没有找到 `trajectory.hdf5`，因此正确地将记录标记为验证失败。

本报告中的建议路径已与 BOS 父目录核对。失败原因是 Excel 中的 BOS 路径拼写错误，不是 HDF5 损坏或磁盘空间不足。

## 2. 背部提手：机器人 368

| Excel 行 | 路径错误 | 建议修改 |
|---:|---|---|
| 13 | `installation 20260811_fail` 中间为空格 | `installation_20260811_fail` |
| 14 | 日期 `202650812` | `20260812` |
| 17 | `tiernyi`；`installatifon` | `tienyi`；`installation` |
| 19 | `installattion` 多一个 `t` | `installation` |
| 22 | `tienyi__prod2` 多一个 `_` | `tienyi_prod2` |
| 41 | `installattion` | `installation` |
| 42 | `tienyiprod2` 缺少 `_`；日期 `200260819` | `tienyi_prod2`；`20260819` |
| 47 | `instaallation` 多一个 `a` | `installation` |
| 54 | `instaallation` | `installation` |
| 55 | `instaallation` | `installation` |

Excel 第 13 行示例：

```text
错误：
tienyi_prod2_dualArm-gripper-3cameras_368_back-handle-installation 20260811_fail

正确：
tienyi_prod2_dualArm-gripper-3cameras_368_back-handle-installation_20260811_fail
```

## 3. 线缆连接：机器人 128

| Excel 行 | 路径错误 | 建议修改 |
|---:|---|---|
| 24 | `prod12` 多一个 `1` | `prod2` |
| 25 | `tieenyi` 多一个 `e` | `tienyi` |
| 26 | `usb20260811` 缺少 `_` | `usb_20260811` |
| 27 | `pprod2` 多一个 `p`；`faill` 多一个 `l` | `prod2`；`fail` |
| 29 | `uIsb` 多一个大写 `I` | `usb` |
| 30 | `tienyii` 多一个 `i` | `tienyi` |
| 31 | `etherinet` 多一个 `i` | `ethernet` |
| 32 | `tienyii` | `tienyi` |
| 33 | `type-cusb` 缺少 `_` | `type-c_usb` |
| 34 | `usb20260813` 缺少 `_` | `usb_20260813` |
| 35 | `type-cusb` | `type-c_usb` |
| 37 | `pprod2`；`usb__20260817` | `prod2`；`usb_20260817` |
| 38 | `usb20260817` | `usb_20260817` |
| 40 | `pprod2` | `prod2` |
| 49 | `uusb` 多一个 `u` | `usb` |
| 50 | `pprod2`；`usb__20260820` | `prod2`；`usb_20260820` |
| 53 | `tienyii`；`type-cusb` | `tienyi`；`type-c_usb` |
| 56 | `type-cusb 20260821`：缺少 `_` 且含空格 | `type-c_usb_20260821` |

Excel 第 27 行示例：

```text
错误：
tienyi_pprod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260811_am_faill

正确：
tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260811_am_fail
```

## 4. 电机插管：机器人 134

| Excel 行 | 路径错误 | 建议修改 |
|---:|---|---|
| 45 | `tifenyi` 多一个 `f` | `tienyi` |
| 46 | `tieenyi` 多一个 `e` | `tienyi` |
| 52 | `tieri`；`motpr` | `tienyi`；`motor` |
| 57 | 日期 `202260821` | `20260821` |

Excel 第 52 行示例：

```text
错误：
tieri_prod2_dualArm-gripper-3cameras_134_insert_hose_into_motpr_20260820_fail

正确：
tienyi_prod2_dualArm-gripper-3cameras_134_insert_hose_into_motor_20260820_fail
```

## 5. 小电池及盖板：机器人 515

| Excel 行 | 路径错误 | 建议修改 |
|---:|---|---|
| 59 | `tienyiprod2` 缺少 `_` | `tienyi_prod2` |
| 60 | `installattion` 多一个 `t` | `installation` |

## 6. 背部提手：机器人 394

| Excel 行 | 路径错误 | 建议修改 |
|---:|---|---|
| 61 | `tieenyi` 多一个 `e` | `tienyi` |

BOS 中确认存在：

```text
tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260821
```

Excel 第 61、62 行还存在元数据疑点：

```text
机器编号：天轶30
BOS 机器人目录：394
```

该问题不影响正确 BOS 路径下载，但需要人工确认这两条数据是否确实属于 `天轶30`。下载程序不应根据路径中的 `394` 自动修改工站。

## 7. 汇总

| 类型 | 数量 |
|---|---:|
| 背部提手，机器人 368 | 10 |
| 线缆连接，机器人 128 | 18 |
| 电机插管，机器人 134 | 4 |
| 小电池及盖板，机器人 515 | 2 |
| 背部提手，机器人 394 | 1 |
| 合计 | 35 |

检查结论：

- Excel 路径拼写错误：35 条
- 已确认存在对应 BOS 目录：35 条
- HDF5 损坏导致失败：0 条
- 磁盘空间导致失败：0 条

## 8. 建议处理方式

1. 将 35 条正确路径加入 `raw_data_mapping.json` 的 `source_overrides`。
2. 每条覆盖保存工作表、行号、原始单元格 SHA-256、正确 BOS 路径和明确的 `source_batch`。
3. 重新生成下载计划。
4. 检查计划记录、未解决记录和重复路径。
5. 使用 `download --resume` 重新执行。
6. 41 条已完成记录应显示 `skipped_complete`，只重新下载修正后的记录。
7. 全部成功后清理错误路径留下的 `.partial` 目录。
8. 最终状态应为：

```text
complete: 76
partial: 0
missing: 0
conflict: 0
```

## 9. 下载记录注意事项

`download_results.jsonl` 是追加日志，不是最终状态快照。再次运行下载后，新结果会追加到文件末尾。统计重试结果时，应按 `record_id` 读取最后一次记录，或增加 `run_id` 和 `attempt` 字段。
