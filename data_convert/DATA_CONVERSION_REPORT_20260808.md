# HIL-RL Cosmos 数据转换与 8 卡适配工作报告

报告日期：2026-08-08  
项目目录：`/media/jushen/linda-zhao/HIL-RL-Project/HIL-RL`  
数据目录：`/media/jushen/linda-zhao/HIL-RL-Project/raw_data`  
输出目录：`/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data`

## 1. 工作概述

本次工作的目标是梳理并完善 HIL-RL 项目的原始机器人轨迹到 Cosmos/LeRobot
训练数据的转换流程，使开发机上的 8 张 GPU 能够参与 VAE 编码，同时保持现有
数据语义、归一化方式和 LeRobot 输出格式不变。

主要完成内容如下：

1. 阅读并梳理 `data_convert`、`data_convert_refactored`、Cosmos policy 接口及相关
   LeRobot 写入代码。
2. 明确原转换流水线中 CPU 预处理、action/proprio 构造、Cosmos VAE 编码和
   episode 写入的边界。
3. 在不修改 `lerobot/src/lerobot/policies/cosmos/modeling_cosmos.py` 的前提下，
   增加转换侧的多 GPU VAE wrapper 池。
4. 支持通过 CLI 或环境变量选择编码卡数和设备编号。
5. 保留单 GPU 历史路径，只有显式请求多卡时才进入多 GPU 路径。
6. 增加输出完整性检查、预检、批量转换、内存监控和多卡测试工具。
7. 完成 4 组 back-handle 数据的正式批量转换，共生成 325 个 episode、436,360 帧。
8. 对输出 Parquet 与原始 HDF5 的 action/proprio 进行了逐帧数值核验。
9. 确认并排除损坏轨迹 `0804_014423`。

## 2. 数据转换流水线

当前转换流程可概括为：

```text
原始 trajectory.hdf5
        |
        v
递归发现并排序 episode
        |
        v
预检 HDF5 字段、长度、有限值、相机和统计文件
        |
        v
构造 Episode / ArmTrajectory
        |
        +--> action：puppet next-frame 相对位姿 + 绝对夹爪
        |
        +--> proprio：左右臂 EE pose + gripper
        |
        +--> 图像：头部相机 + 左右腕部相机裁剪/拼接
        |
        v
使用官方 dataset_statistics.json 做 min/max 归一化
        |
        v
按 micro-batch 将视频条件送入单卡或多卡 Cosmos VAE
        |
        v
注入 action / current proprio / future proprio / value latent
        |
        v
按 episode 串行写入 LeRobot Parquet 和 metadata
```

转换入口关系：

- `data_convert/convert.sh`：单次转换 shell 入口，负责环境、路径、参数和已有输出检查。
- `data_convert_refactored/convert.py`：重构版 Python CLI。
- `data_convert_refactored/pipeline.py`：预检和转换调度。
- `data_convert_refactored/preparation/episode.py`：读取并验证 HDF5 episode。
- `data_convert_refactored/preparation/actions.py`：确定 action 语义并编码。
- `data_convert_refactored/preparation/dataset_stats.py`：统计文件加载和范围检查。
- `data_convert_refactored/preparation/proprio.py`：proprio构造与归一化。
- `data_convert_refactored/cosmos_backend.py`：连接重构流水线与 Cosmos/LeRobot 后端。
- `data_convert/convert_raw_to_cosmos.py`：图像预处理、VAE 编码、latent 注入和写入。
- `data_convert/multi_gpu_vae.py`：本次新增的转换侧多 GPU VAE wrapper 池。
- `data_convert/batch_convert_back_handle.sh`：四组 back-handle 数据的批量转换入口。

## 3. 数据语义与转换配置

本批数据使用以下语义：

### 3.1 Action

- 来源：`puppet_next_frame`
- 编码：`legacy_euler`
- 维度：14
- 顺序：左臂 7 维 + 右臂 7 维
- 每臂顺序：`dx, dy, dz, rx, ry, rz, gripper`
- 位姿语义：`inverse(current_pose) @ next_pose`
- 位移缩放：`0.02 m`
- 欧拉角缩放：`0.06 rad`
- 夹爪缩放：`1.0`
- 位移和旋转在第一阶段裁剪到 `[-1, 1]`
- 夹爪目标裁剪到 `[0, 1]`
- 第二阶段使用官方 `actions_min/actions_max` 映射到 `[-1, 1]`
- 最后一帧不存在真实 next frame，因此重复最后一个有效 action，并在 metadata 中
  标记为 synthetic/padding。

双臂 action 顺序：

```text
left_dx, left_dy, left_dz,
left_rx, left_ry, left_rz, left_gripper,
right_dx, right_dy, right_dz,
right_rx, right_ry, right_rz, right_gripper
```

### 3.2 Proprio

- 维度：16
- 来源：同一帧的 puppet EE pose 和 gripper
- 四元数格式：`xyzw`
- 转换前执行归一化和符号连续化，避免相邻帧出现等价四元数符号跳变
- 使用官方 `proprio_min/proprio_max` 映射到 `[-1, 1]`
- 官方统计模式允许超出统计范围，不执行 clip
- 中间张量以 bfloat16 进入 Cosmos 状态，因此 Parquet 中可能出现约 `0.004`
  量级的正常量化误差

顺序为：

```text
left_x, left_y, left_z,
left_qx, left_qy, left_qz, left_qw, left_gripper,
right_x, right_y, right_z,
right_qx, right_qy, right_qz, right_qw, right_gripper
```

`future_proprio` 采用 action chunk 大小 16，即第 `i` 帧使用
`min(i + 16, episode_length - 1)` 对应帧的 proprio。

### 3.3 图像

- 相机状态：`normal`
- 腕部裁剪：`center_width`
- 腕部裁剪比例：`0.75`
- 左腕中心 X 偏移：64
- 右腕中心 X 偏移：64
- 头部图像顶部裁剪：100 像素
- VAE 输入在 CPU micro-batch 内完成 `uint8 -> float32 -> /127.5 - 1.0`
- 输出保存的是 Cosmos latent，不是 MP4 视频。

### 3.4 统计与语言条件

官方统计文件：

```text
/media/jushen/linda-zhao/HIL-RL-Project/raw_data/
tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/
dataset_statistics.json
```

T5 embedding：

```text
/media/jushen/linda-zhao/HIL-RL-Project/raw_data/
tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/
t5_embeddings.pkl
```

任务描述：

```text
Pick up the left black handle and attach it to the white back panel. Then pick
up the two black screws one by one and place them onto the handle.
```

## 4. 多 GPU 适配方案

### 4.1 原问题

Cosmos 使用的 `Wan2pt1VAEInterface` 不是标准 `torch.nn.Module`。它内部除了实际
VAE 网络，还持有 mean/std 等归一化张量。直接对外层 wrapper 调用通用的
`.to(device)`、`.parameters()` 或仅复制内部模型，会导致模型参数与归一化张量位于
不同 GPU，最终出现跨设备运算错误。

因此不能把它当作普通 PyTorch module 做浅复制或通用 DataParallel。

### 4.2 最终设计

最终采用：每张编码 GPU 完整初始化一个 `Wan2pt1VAEInterface`，并在整个转换进程
中缓存复用。

具体行为：

1. 主 GPU 直接复用 policy 已初始化的 tokenizer wrapper。
2. 其他 GPU 使用原始 Cosmos lazy config 分别初始化完整 wrapper。
3. 初始化后同时验证内部 VAE 参数和所有归一化张量确实位于目标 GPU。
4. 多卡 wrapper 顺序初始化，避免并发读取 checkpoint 和临时模型分配导致启动时
   CPU/GPU 内存峰值过大。
5. 编码阶段根据实际 batch 大小选择活跃 GPU。
6. CPU batch 按顺序切分到各卡，每张卡内部继续按 `encode_batch_size` 编码。
7. 各 GPU 通过线程池并发执行，latent 返回 CPU 后按原分片顺序拼接。
8. 检查拼接后的 latent batch 长度必须等于输入 batch 长度。
9. wrapper 池挂在 policy 上缓存；如果后续请求的设备集合或 batch size 与缓存签名
   不一致，则明确报错，不静默复用错误配置。

### 4.3 并行粒度与写入安全

当前不是多个进程同时转换多个 episode，而是：

```text
episode 之间：串行
episode 内 CPU 图像 micro-batch：串行准备
一个 micro-batch 的 VAE shard：多 GPU 并行
LeRobot episode 写入：单进程串行
```

这样设计的主要原因是 LeRobotDataset 会更新 Parquet、episode index、统计信息和
metadata。让多个 episode worker 直接写同一个输出目录，会引入文件名、索引和
metadata 更新冲突。当前方案只有一个 writer，不需要额外文件锁或结果合并阶段；
同时 VAE 这一最重的计算环节仍能使用 8 张 GPU。

多卡返回结果由 `ThreadPoolExecutor.map` 按输入分片顺序收集，再执行 `torch.cat`，
不会打乱原始帧顺序。

### 4.4 参数优先级

支持：

```text
--encode-world-size
--encode-device-ids
ENCODE_WORLD_SIZE
ENCODE_CUDA_DEVICES
CUDA_VISIBLE_DEVICES
```

CLI 参数优先于 `ENCODE_*` 环境变量。`--encode-device-ids` 是
`CUDA_VISIBLE_DEVICES` 映射后的进程内编号。例如：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
--encode-world-size 8
--encode-device-ids 0,1,2,3,4,5,6,7
```

## 5. 代码修改范围

### 5.1 新增文件

- `data_convert/multi_gpu_vae.py`
  - 完整 VAE wrapper 初始化、设备验证、缓存和多卡并发编码。
- `data_convert/test_multi_gpu_conversion.sh`
  - 单卡/多卡真实转换 smoke test 和输出对比入口。
- `data_convert_refactored/gpu_config.py`
  - 解析和验证 world size、设备编号及环境变量优先级。
- `data_convert_refactored/tools/compare_conversion_outputs.py`
  - 对比单卡与多卡 Parquet 输出。
- `data_convert_refactored/tests/test_gpu_config.py`
- `data_convert_refactored/tests/test_multi_gpu_encode_adapter.py`
- `data_convert_refactored/tests/test_multi_gpu_vae.py`
- `data_convert_refactored/tests/test_compare_conversion_outputs.py`

### 5.2 修改文件

- `data_convert/convert_raw_to_cosmos.py`
  - 增加多卡参数；
  - 保留历史单卡编码路径；
  - 多卡请求转入 `VAEReplicaPool`；
  - CPU micro-batch 归一化；
  - 内存事件记录；
  - 保存多卡配置到转换 metadata。
- `data_convert_refactored/config.py`
  - 在配置对象中增加 `encode_world_size` 和 `encode_device_ids`。
- `data_convert_refactored/convert.py`
  - 增加多 GPU CLI 参数和环境变量解析。
- `data_convert_refactored/cosmos_backend.py`
  - 将多卡配置传递给后端转换配置。
- `data_convert/convert.sh`、`data_convert/pipeline.sh`
  - 接收并转发多卡参数；
  - 增强已有输出完整性判断。
- `data_convert/batch_convert_back_handle.sh`
  - 批量发现四组输入；
  - 支持 preflight、fail-fast、多卡参数、内存监控；
  - 使用临时 symlink 视图排除损坏 episode，不移动或修改原始数据。
- `data_convert/README_PIPELINE.md`、`data_convert_refactored/README.md`
  - 补充多卡转换、测试和批处理说明。

本次明确未修改：

```text
lerobot/src/lerobot/policies/cosmos/modeling_cosmos.py
```

也没有从 `learner_copy_dist.py` 引入转换逻辑。对 Cosmos wrapper 的特殊处理全部限制
在数据转换模块中，降低对训练/推理代码的影响。

## 6. 输出完整性保护

`convert.sh` 过去可能只根据目录或配置 metadata 判断“已经完成”，从而误把中断后
留下的空目录或半成品当作完整结果。当前检查至少要求：

1. `meta/info.json` 存在；
2. `cosmos_dataset_metadata.json` 存在；
3. `total_episodes > 0`；
4. `episode_*.parquet` 数量大于 0；
5. Parquet 文件数等于 `total_episodes`；
6. 已有输出的重要任务、标签、统计文件 hash、action 配置和图像配置与当前请求一致。

如果输出目录存在但不完整，脚本会报错并要求使用新的输出目录，不会静默跳过。

批处理默认先执行 preflight。`--fail-fast` 会在第一组失败时停止，避免在输入配置
不正确时继续消耗数小时 GPU 时间。

## 7. 正式批量转换

正式命令：

```bash
cd /media/jushen/linda-zhao/HIL-RL-Project/HIL-RL
source /media/jushen/linda-zhao/HIL-RL-Project/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p /media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/logs
set -o pipefail

bash data_convert/batch_convert_back_handle.sh \
  --fail-fast \
  --encode-world-size 8 \
  --encode-device-ids 0,1,2,3,4,5,6,7 \
  --encode-batch-size 8 \
  --batch-size 1 \
  --monitor-memory \
  --memory-sample-interval 0.5 \
  2>&1 | tee /media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/logs/batch_multigpu8.log
```

批处理结果：

| 数据集 | Outcome | Episodes | Frames | Parquet |
|---|---|---:|---:|---:|
| 2026-08-03 PM | success | 159 | 246,052 | 159 |
| 2026-08-03 PM | failure | 8 | 7,159 | 8 |
| 2026-08-04 AM | success | 62 | 96,158 | 62 |
| 2026-08-04 AM | failure | 96 | 86,991 | 96 |
| **合计** |  | **325** | **436,360** | **325** |

最终批处理摘要：

```text
Batch complete
successful/already-complete: 4
skipped after preflight:     0
failed:                      0
```

四个输出目录：

```text
/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260803_pm_success
/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260803_pm_failure
/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260804_am_success
/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/back_handle_20260804_am_failure
```

总日志：

```text
/media/jushen/linda-zhao/HIL-RL-Project/cosmos_data/logs/batch_multigpu8.log
```

## 8. 转换后完整性验证

对全部四个输出执行了以下检查：

1. `meta/info.json` 的 `total_episodes`；
2. `meta/episodes.jsonl` 的有效行数；
3. `episode_*.parquet` 文件数；
4. 每个 Parquet footer 是否可以被 PyArrow 正常读取；
5. 所有 Parquet 行数之和是否等于 `total_frames`；
6. 日志是否包含 traceback、CUDA OOM、conversion failed 或 preflight failed；
7. 转换结束后是否仍存在残留转换进程。

验证结果：

| 输出 | info episodes | JSONL episodes | Parquet | info frames | Parquet rows | 坏文件 |
|---|---:|---:|---:|---:|---:|---:|
| 08-03 success | 159 | 159 | 159 | 246,052 | 246,052 | 0 |
| 08-03 failure | 8 | 8 | 8 | 7,159 | 7,159 | 0 |
| 08-04 success | 62 | 62 | 62 | 96,158 | 96,158 | 0 |
| 08-04 failure | 96 | 96 | 96 | 86,991 | 86,991 | 0 |

日志未发现 traceback、OOM 或转换失败，转换进程已经退出。

## 9. HDF5 与 Parquet 数值对比

抽取以下一一对应的 episode 做全帧比对：

```text
HDF5:
raw_data/..._20260803_pm/0803_173020/data/trajectory.hdf5

Parquet:
cosmos_data/back_handle_20260803_pm_success/
data/chunk-000/episode_000000.parquet
```

二者均为 1,432 帧。使用当前代码和正式统计文件，从 HDF5 重新计算全部 action、
proprio 和 future_proprio，再与 Parquet 比较：

| 字段 | Shape | 最大绝对误差 | 平均绝对误差 | 结论 |
|---|---|---:|---:|---|
| action | `(1432, 14)` | 0.0 | 0.0 | 全元素完全一致 |
| proprio | `(1432, 16)` | 0.003888607 | 0.000826186 | bfloat16 允许误差内一致 |
| future_proprio | `(1432, 16)` | 0.003888607 | 0.000826259 | bfloat16 允许误差内一致 |

action 最后一帧也与倒数第二帧完全相同，符合 `puppet_next_frame` 的 padding 设计。

proprio 的最大误差示例：

```text
重算 float32：-1.019513607
Parquet：      -1.015625
```

这是 bfloat16 舍入造成的正常差异。该值小于 -1 是因为官方统计模式明确采用
`allow_and_warn` 且不 clip，并非转换错误。

## 10. 损坏轨迹 `0804_014423`

问题文件：

```text
/media/jushen/linda-zhao/HIL-RL-Project/raw_data/
tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/
0804_014423/data/trajectory.hdf5
```

### 10.1 EE pose 后段归零

左右臂对齐 EE pose 均为 993 帧：

| 字段 | 最后有效索引 | 全零索引（0-based） | 人类帧编号 | 全零数 |
|---|---:|---|---|---:|
| left pose align | 926 | 927–992 | 第 928–993 帧 | 66 |
| right pose align | 926 | 927–992 | 第 928–993 帧 | 66 |

这 66 帧的 7 维 pose 全部为零，包括四元数：

```text
[x, y, z, qx, qy, qz, qw] = [0, 0, 0, 0, 0, 0, 0]
```

零四元数不是合法旋转，因此无法生成姿态矩阵或 next-frame action。

高频 raw pose 也存在同样问题：

| 字段 | 全零索引（0-based） | 全零数 |
|---|---|---:|
| left pose raw | 1856–1989 | 134 |
| right pose raw | 1856–1989 | 134 |

### 10.2 序列长度不一致

```text
EE pose align:          993
EE gripper align:       993
puppet arm joint align: 995
master arm joint align: 995
camera_head:            993
camera_left:            993
camera_right:           992
```

预检首先会报：

```text
ValueError: left.puppet_joints length 995 != 993
```

即使绕过该错误，后续仍会因为零四元数和右相机缺少一帧而失败。因此这不是一个只需
截掉最后几帧就能无风险修复的 episode。

### 10.3 排除方式

批处理脚本通过相对路径白名单明确排除：

```text
tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/
0804_014423/data/trajectory.hdf5
```

排除使用 `/tmp` 下的临时 symlink 输入视图，不删除、不移动、不修改原始 HDF5。
08-03 success 原始 160 条，排除该条后正确输出 159 条。

## 11. 测试工作

本次采用分层测试：

1. 静态和轻量单元测试
   - GPU 配置解析；
   - 重复、负数和非法设备编号；
   - CLI/环境变量优先级；
   - 多卡 adapter 路径选择；
   - wrapper 池缓存签名；
   - 分片顺序和输出长度；
   - 单卡/多卡输出比较工具。
2. 真实 wrapper smoke test
   - 2 GPU 完整 wrapper 初始化与编码通过；
   - 8 GPU 完整 wrapper 初始化与编码通过；
   - 每个 wrapper 的模型参数及 normalization tensor 均通过设备一致性验证。
3. 正式批处理验证
   - 四组数据 preflight 通过；
   - 正式 8 卡转换完成；
   - 325 个 Parquet footer 全部可读；
   - 436,360 行与 metadata 完全一致。
4. 语义数值回归
   - 选择真实 episode 全帧重算 action/proprio；
   - action 完全一致；
   - proprio/future_proprio 在 bfloat16 误差范围内一致。

## 12. 当前状态与注意事项

截至报告生成时：

- 当前 Git 分支为 `main`。
- 数据转换代码改动仍处于工作树中，尚未形成正式提交。
- `raw_data/` 和 `cosmos_data/` 是大型数据目录，不应提交到 Git。
- `cosmos-policy/wandb_logs/` 中存在运行产生的无关改动，应与本次代码提交分离。
- 正式提交前应只暂存本报告第 5 节列出的数据转换代码、测试和文档文件。
- 不应提交模型、原始 HDF5、Parquet、日志、临时 symlink 或 W&B 运行文件。

## 13. 后续建议

1. 将当前转换修改整理为独立 Git commit，便于回滚和 code review。
2. 在提交前于开发机再次运行轻量单元测试和 shell 语法检查。
3. 将 `0804_014423` 的排除原因保留在数据质量清单中，不只保留在代码注释里。
4. 后续新增数据批次时先执行 `--preflight-only`，确认字段、长度、四元数和相机帧数。
5. 输出上传前保留 `cosmos_dataset_metadata.json`、`meta/info.json`、批处理总日志以及
   官方统计文件 hash，形成可追溯交付记录。
6. 如果未来确实需要 episode 级并行，应采用“每个 worker 写独立临时数据集，完成后
   单线程重编号和合并”的方式，不能让多个进程直接写同一 LeRobotDataset。
7. 对所有新转换结果至少执行一次 Parquet footer/行数检查，并抽样做 HDF5 到 Parquet
   的 action/proprio 数值回归。

## 14. 最终结论

本次转换侧 8 GPU 适配已经完成，并在真实 8 卡环境及正式批量数据上通过验证。
多卡实现没有改变 action、proprio、图像、统计和 LeRobot 写入语义；通过每卡完整
VAE wrapper、进程内缓存、micro-batch 分片以及单 writer 串行写入，解决了原先
wrapper 内部模型和归一化张量跨设备的问题，同时避免了 episode 并行写入冲突。

四组 back-handle 数据已经全部完成转换，共 325 个有效 episode、436,360 帧；
metadata、JSONL、Parquet 数量和 Parquet 行数完全一致，日志中没有转换失败或 OOM。
抽样 episode 的 action 与 HDF5 重算结果逐元素完全一致，proprio 仅有符合 bfloat16
预期的量化误差。损坏轨迹 `0804_014423` 已确认存在 EE pose 归零、序列长度不一致
和相机缺帧等问题，当前排除策略正确。
