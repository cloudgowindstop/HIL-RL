# Cosmos Policy 数据管道

## 当前流程

```text
BOS / 本地 raw HDF5
        │
        ├── data_convert_refactored/scripts/pipeline.sh（下载 + 正式转换）
        └── data_convert_refactored/scripts/convert.sh（仅正式转换或单 episode action round-trip）
                         │
                         ▼
        data_convert_refactored.convert
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 legacy_euler                      cosmos_rotation_6d
 单臂 7D / 双臂 14D                单臂 10D / 双臂 20D
```

正式转换会进行图像预处理、VAE 编码和 latent 注入，需要 CUDA GPU。单 episode action
round-trip 只检查 HDF5、10D action、Parquet 和实机 7D 控制命令，不需要 GPU。

## 主要文件

```text
HIL-RL/
├── data_convert/
│   ├── pipeline.sh                   # 兼容入口，转发到重构目录
│   ├── convert.sh                    # 兼容入口，转发到重构目录
│   ├── convert_raw_to_cosmos.py      # HDF5 -> Cosmos LeRobot Parquet
│   ├── cosmos_rotation_6d.py         # rotation-6D/action 编解码
│   ├── debug_replay_cosmos_6d.py     # Parquet round-trip 与实机命令导出
│   └── README_PIPELINE.md
├── data_convert_refactored/
│   ├── convert.py                     # CLI入口
│   ├── config.py                      # 类型化配置
│   ├── pipeline.py                    # 数据准备流程编排
│   ├── preflight.py                   # 转换前检查
│   ├── cosmos_backend.py              # Cosmos阶段编排
│   ├── dataset_writer.py              # LeRobot写入
│   ├── preparation/                   # HDF5、action、proprio、stats、FK
│   ├── conditioning/                  # 相机、reward/done、transition
│   ├── encoding/                      # Policy、VAE、多GPU
│   ├── tools/                         # stats生成和结果对比
│   └── scripts/                       # 正式Shell入口
│       └── jobs/                      # 特定任务批处理入口
├── train_config_cosmos.json
├── dataset_raw/
├── dataset_cosmos/
└── 具身智能数据交付表.xlsx
```

## 前置条件

- Python 3.10+ 与项目虚拟环境 `/media/jushen/mingbo-ge/.venv`
- 正式转换需要可用的 CUDA GPU 和 Cosmos Policy/VAE 权重
- `lerobot`、`cosmos_policy`、`torch`、`torchvision`、`h5py`、`opencv-python`、
  `scipy`、`pyarrow`、`openpyxl`
- 使用 `pipeline.sh` 下载时需要 `bcecmd`
- 语言条件可在转换时跳过；如果不跳过，需要包含精确任务文本键的 T5 embedding pickle

使用 `--skip-t5` 时不加载 T5，也不会伪造零向量；LeRobot task 字符串仍会保存。
不传 `--skip-t5` 时，`--t5-embeddings` 仍为必填。

相机处理使用 `--camera-state normal`（默认值）。该状态对应当前正式裁剪方案：
左右腕相机完整保留高度、水平保留75%且裁剪中心向右偏移64像素，头部相机从
顶部裁掉100像素。当前传入其他状态会在加载VAE前报错；转换metadata同时保存
状态名和实际裁剪参数。

## Action 编码

### legacy_euler

单臂格式：

```text
[local_dx/0.02, local_dy/0.02, local_dz/0.02,
 local_rx/0.06, local_ry/0.06, local_rz/0.06,
 absolute_gripper/1.0]
```

单臂为 7D，双臂为 14D。

### cosmos_rotation_6d

先计算局部相对变换：

```text
delta_T = inverse(T_current) @ T_target
```

Rotation 6D 取 `delta_R` 的前两列：

```text
[R[:,0], R[:,1]]
```

单臂格式：

```text
[local_translation/0.02, rotation_6d(6), absolute_gripper/1.0]
```

单臂为 10D，双臂为 20D。6D 模式不使用 `rotation_scale=0.06`；该参数只在把
10D action 转成现有 BaseEnv 的 7D Euler 命令时使用。

静止旋转不是六个零，而是：

```text
[1, 0, 0, 0, 1, 0]
```

以上是第一层机器人控制尺度。正式转换强制加载预先由完整训练集合并计算的
`actions_min/actions_max`，并复用 CosmosPolicy 的 `rescale_action()` 做第二层映射：

```text
normalized = 2 * (first_stage_action - actions_min)
                 / (actions_max - actions_min) - 1
```

Parquet action 和 latent action chunk 都保存第二层结果。success/failure 分目录转换时必须
传入同一个共享统计文件。模型推理输出必须先撤销第二层，再交给机器人控制器。

完整训练集先生成一次共享统计；`--input` 可以重复：

```bash
PYTHONPATH=. python3 -m data_convert_refactored.tools.generate_dataset_stats \
  --input dataset_raw/pick_spoon/success \
  --input dataset_raw/pick_spoon/failure \
  --output dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --action-source puppet_next_frame \
  --action-encoding cosmos_rotation_6d \
  --translation-scale 0.02 \
  --gripper-scale 1.0
```

该命令同时生成 `shared_dataset_statistics.metadata.json`，绑定 action source、编码、scale、
维度和末帧策略。正式转换及 `--preflight-only` 都会强制校验，并检查输入是否超出范围。
`puppet_next_frame` 末帧复制最后一个有效 action，同时标记为 synthetic padding：复制是数值
填充策略，padding 表示它没有新的真实目标帧，两者并不矛盾。

官方统计使用独立模式，不需要也不会伪造 sidecar：

```bash
--stats-mode official \
--official-dataset-stats /path/to/official/dataset_statistics.json
```

该模式严格复用 `collect_data_cosmos.py` 使用的加载和归一化实现。数据超出官方 min/max
时仅警告，归一化结果允许超过 `[-1,1]`，不会裁剪；超界通道和数量写入转换 metadata。

## `convert.sh`：仅转换本地数据

当前 shell 默认 action 编码为 `cosmos_rotation_6d`。

先跳过 T5、正式转换一条 VAE 数据：

```bash
bash data_convert_refactored/scripts/convert.sh \
  --task pick_spoon \
  --episode-outcome success \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --max-episodes 1 \
  --action-encoding cosmos_rotation_6d \
  --skip-t5 \
  --encode-batch-size 8
```

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL

bash data_convert_refactored/scripts/convert.sh \
  --task pick_spoon \
  --episode-outcome success \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --max-episodes 1 \
  --action-encoding cosmos_rotation_6d \
  --t5-embeddings /path/to/pick_spoon_t5_embeddings.pkl \
  --encode-batch-size 8
```

输出目录会同时包含action来源和编码，避免不同语义相互覆盖：

```text
dataset_cosmos/pick_spoon_success_puppet_next_frame_cosmos_rotation_6d/
```

运行旧 Euler 分支：

```bash
bash data_convert_refactored/scripts/convert.sh \
  --task pick_spoon \
  --episode-outcome success \
  --dataset-stats /path/to/legacy_euler_shared_dataset_statistics.json \
  --action-encoding legacy_euler \
  --t5-embeddings /path/to/pick_spoon_t5_embeddings.pkl
```

旧格式输出到：

```text
dataset_cosmos/pick_spoon_cosmos/
```

这样可以避免 7D 和 10D action 写入同一个数据集。

### `convert.sh` 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--task` | 空 | 任务目录名；为空时读取 Excel 中全部已标注完成任务 |
| `--episode-outcome` | 必填 | 显式指定输入目录全部episode为`success`或`failure`；不根据目录名推断 |
| `--dataset-stats` | 必填 | 全部训练episode共享的统计JSON；旁边必须有`.metadata.json` |
| `--stats-mode` | `generated` | `generated`或`official` |
| `--official-dataset-stats` | 空 | official模式的原始统计JSON；不要求sidecar |
| `--action-encoding` | `cosmos_rotation_6d` | `legacy_euler` 或 `cosmos_rotation_6d` |
| `--translation-scale` | `0.02` | 局部平移动作 scale |
| `--action-source` | `puppet_next_frame` | action来源；天翼关节数据可显式使用 `master_joint_fk_same_frame` |
| `--kinematics-config` | 空 | `master_joint_fk_same_frame` 必填的官方URDF/FK配置 |
| `--rotation-scale` | `0.06` | legacy Euler/实机 BaseEnv Euler scale；6D 编码本身不使用 |
| `--gripper-scale` | `1.0` | 绝对夹爪目标 scale |
| `--t5-embeddings` | 空 | 不使用 `--skip-t5` 时必填 |
| `--skip-t5` | false | 转换时不加载 T5，task 字符串仍保存 |
| `--max-episodes` | `0` | 0 表示全部 |
| `--encode-batch-size` | `16` | VAE micro-batch，显存不足时减小 |
| `--encode-world-size` | `1` | 将当前 CPU micro-batch 分到多张可见 GPU 做 VAE encode |
| `--encode-device-ids` | 空 | `CUDA_VISIBLE_DEVICES` 映射后的进程内 GPU 编号，如 `0,2,4,6` |
| `--batch-size` | `1` | 训练 batch 对齐字段 |
| `--raw-dir` | `workspace/raw_data` | HDF5 根目录 |
| `--cosmos-dir` | `workspace/cosmos_data` | 正式输出根目录 |
| `--dry-run` | false | 只显示任务，不转换 |
| `--preflight-only` | false | 只检查HDF5/FK/action，不加载VAE或写数据集 |
| `--input` | 空 | 精确指定任务目录或单个`trajectory.hdf5`；不依赖Excel |
| `--output` | 自动生成 | 与`--input`配合，精确指定输出目录 |
| `--task-description` | 从任务名生成 | 精确指定写入LeRobot的任务文本 |
| `--cosmos-config` | `train_config_cosmos.json` | Cosmos配置文件 |
| `--wrist-crop-mode` | `center_width` | 双腕图裁剪方式：水平中心宽度、底部或不裁剪 |
| `--wrist-crop-fraction` | `0.75` | 保留75%宽度；旧参数`--wrist-crop-bottom`仅作为兼容别名 |
| `--wrist-left-center-offset-x` | `64` | 左腕相机裁剪中心向右偏移64个原图像素 |
| `--wrist-right-center-offset-x` | `64` | 右腕相机裁剪中心向右偏移64个原图像素 |
| `--head-crop-top-pixels` | `100` | 仅对名称包含`head`的主相机删除顶部100像素；腕部不做纵向裁剪 |
| `--fk-max-position-error-m` | `0.005` | FK最大位置误差阈值 |
| `--fk-max-rotation-error-deg` | `1.0` | FK最大旋转误差阈值 |

精确预检单条HDF5：

```bash
bash data_convert_refactored/scripts/convert.sh \
  --input dataset_raw/pick_spoon/0731_103412/trajectory.hdf5 \
  --output dataset_cosmos/pick_spoon_puppet_next_frame_cosmos_rotation_6d \
  --task-description "pick spoon" \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --action-source puppet_next_frame \
  --action-encoding cosmos_rotation_6d \
  --skip-t5 \
  --preflight-only
```

## 天翼双臂 master/puppet 官方 action 对齐

天翼 HDF5 保存的是同帧 master/puppet 七关节角，不包含 master 末端 pose。官方对齐分支使用同一套 FK 计算：

```text
delta_left  = inv(FK(puppet_left[i]))  @ FK(master_left[i])
delta_right = inv(FK(puppet_right[i])) @ FK(master_right[i])
```

先复制并填写运动学模板（禁止直接使用模板占位值）：

```text
data_convert/kinematics/tienyi_prod2_dual_arm.template.json
```

必须填写官方 URDF、左右关节顺序、base/end-effector link、关节符号/零偏和 TCP。先运行不加载 VAE 的 action/FK 检查：

```bash
PYTHONPATH=data_convert /media/jushen/mingbo-ge/.venv/bin/python3 \
  data_convert/debug_master_fk_actions.py \
  --hdf5 /path/to/trajectory.hdf5 \
  --kinematics-config /path/to/tienyi_prod2_dual_arm.json \
  --output debug_outputs/tienyi_master_fk_actions.npy
```

只有 FK 对 recorded puppet EE pose 的检查通过后，才能正式转换：

```bash
/media/jushen/mingbo-ge/.venv/bin/python3 \
  -m data_convert_refactored.convert \
  --input /path/to/task_or_episode_dir \
  --output dataset_cosmos/task_cosmos_6d_master_fk \
  --task "task instruction" \
  --episode-outcome success \
  --dataset-stats /path/to/shared_dataset_statistics.json \
  --action-encoding cosmos_rotation_6d \
  --action-source master_joint_fk_same_frame \
  --kinematics-config /path/to/tienyi_prod2_dual_arm.json \
  --translation-scale 0.02 \
  --gripper-scale 1.0 \
  --skip-t5
```

默认 FK 阈值为最大位置误差 `0.005 m`、最大旋转误差 `1 deg`，可通过
`--fk_max_position_error_m` 和 `--fk_max_rotation_error_deg` 调整。不要为了让错误的 URDF/关节顺序通过而直接放宽阈值。

## 单条 pick_spoon round-trip（不需要 GPU）

```bash
bash data_convert_refactored/scripts/convert.sh \
  --roundtrip-hdf5 dataset_raw/pick_spoon/0731_103412/trajectory.hdf5
```

该命令调用 `debug_replay_cosmos_6d.py` 并生成：

```text
debug_outputs/0731_103412_rotation6d.parquet
debug_outputs/0731_103412_rotation6d_report.csv
debug_outputs/0731_103412_rotation6d_report.base_env_actions.npy
```

这里的 Parquet 是 action-debug 数据，不含 VAE video latent，不能作为正式训练数据。它用于验证：

```text
HDF5 pose(t), pose(t+1)
  -> 10D rotation-6D action
  -> Parquet 保存与读取
  -> rotation-6D 解码
  -> 目标位姿重建
  -> 当前 BaseEnv 可接收的 7D Euler command
```

已有 `dataset_cosmos/pick_spoon_cosmos` 是 legacy 7D 数据。debug 脚本会拒绝将其误当成
10D rotation-6D Parquet。

## `pipeline.sh`：BOS 下载 + 正式转换

## Back-handle四目录串行转换

`batch_convert_back_handle.sh` 专门处理
当前 workspace 的 `raw_data/` 下四个 back-handle 一级目录。当前开发机使用：

```text
/media/jushen/linda-zhao/HIL-RL-Project/raw_data/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/dataset_statistics.json
/media/jushen/linda-zhao/HIL-RL-Project/raw_data/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/t5_embeddings.pkl
```
目录名包含 `fail` 时标为 failure，否则标为 success；四组共用 0803 PM 目录中的官方
stats 和 T5 cache，输出根目录固定为
当前 workspace 的 `cosmos_data/`。

先只检查四组数据：

```bash
bash data_convert_refactored/scripts/jobs/batch_convert_back_handle.sh --preflight-only
```

预检全部完成后，串行执行正式VAE转换：

```bash
bash data_convert_refactored/scripts/jobs/batch_convert_back_handle.sh
```

默认会先预检全部目录，再依次转换；一个目录失败时记录日志并继续下一个。完整且metadata
一致的已有输出由 `convert.sh` 自动跳过。日志保存在
当前 workspace 的 `cosmos_data/logs/`。可用
`--fail-fast` 改为首次失败即停止，或用
`--skip-preflight` 跳过批量预检阶段。

当前脚本的显式排除清单暂时跳过长度不一致的
`20260803_pm/0804_014423/data/trajectory.hdf5`，因此转换325条、跳过1条。过滤通过
`/tmp` 下的临时软链接视图完成，不移动或修改原始HDF5；脚本退出时自动删除临时视图。

```bash
bash data_convert_refactored/scripts/pipeline.sh \
  --task plug_in_ethernet \
  --episode-outcome success \
  --dataset-stats /path/to/shared_dataset_statistics.json \
  --max-episodes 1 \
  --action-encoding cosmos_rotation_6d \
  --skip-t5
```

处理 Excel 中所有已标注完成任务：

```bash
bash data_convert_refactored/scripts/pipeline.sh \
  --episode-outcome success \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --action-encoding cosmos_rotation_6d \
  --t5-embeddings /path/to/multi_task_t5_embeddings.pkl
```

仅预览：

```bash
bash data_convert_refactored/scripts/pipeline.sh --episode-outcome success --dry-run --action-encoding cosmos_rotation_6d
```

`pipeline.sh` 支持与 `convert.sh` 相同的 action/scale/T5 参数，同时支持：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--raw-dir` | `HIL-RL/dataset_raw` | BOS 下载目录 |
| `--cosmos-dir` | `HIL-RL/dataset_cosmos` | 转换输出目录 |
| `--dry-run` | false | 不下载、不转换 |

## 直接调用 Python

```bash
python3 -m data_convert_refactored.convert \
  --input dataset_raw/pick_spoon \
  --output dataset_cosmos/pick_spoon_puppet_next_frame_cosmos_rotation_6d \
  --task "pick spoon" \
  --episode-outcome success \
  --stats-mode generated \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --skip-t5 \
  --action-encoding cosmos_rotation_6d \
  --action-source puppet_next_frame \
  --translation-scale 0.02 \
  --gripper-scale 1.0 \
  --max-episodes 1 \
  --encode-batch-size 8
```

重构后的 shell 和 Python 模块入口统一使用连字符参数，例如
`--action-encoding`、`--action-source` 和 `--skip-t5`。

## 正式输出

```text
dataset_cosmos/{task}_cosmos_6d/
├── dataset_statistics.json
├── dataset_statistics.metadata.json
├── cosmos_dataset_metadata.json
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
└── data/chunk-000/
    └── episode_*.parquet
```

主要字段：

| 字段 | 单臂 | 双臂 | 说明 |
|---|---:|---:|---|
| `video` | `(16,9,28,28)` | `(16,9,28,28)` | VAE latent |
| `proprio` | `(8,)` | `(16,)` | 当前 proprio，训练集 min/max 到 `[-1,1]` |
| `future_proprio` | `(8,)` | `(16,)` | `t+chunk_size` proprio |
| `action` legacy | `(7,)` | `(14,)` | Euler action |
| `action` 6D | `(10,)` | `(20,)` | rotation-6D action |
| `value_function_return` | `(1,)` | `(1,)` | 当前实现的 MC return |
| `next.reward` / `next.done` | `(1,)` | `(1,)` | 当前规则保持不变 |

`cosmos_dataset_metadata.json` 记录 action encoding、通道顺序、scale、相机、proprio 顺序和
latent 布局。训练和推理前必须先检查该文件，不能仅凭目录名判断数据格式。

LeRobot metadata 的 FPS 保存 HDF5 检测到的物理采样频率，用于时间轴、时间戳和按秒采样；
Cosmos 内部条件窗口固定为16帧。二者用途不同，均记录在 `cosmos_dataset_metadata.json`；
同一批输入若物理FPS不一致，预检会拒绝转换。

## 实机验证

默认只生成命令，不发送机器人。先查看 CSV 中：

- `position_error_m`
- `rotation_error_deg`
- `translation_clipped`
- `base_env_rotation_clipped`
- 单步平移和旋转幅度

实机首次验证建议只执行一帧。直接调用 debug 脚本时，执行入口要求显式提供环境 factory：

```bash
PYTHONPATH=data_convert python3 data_convert/debug_replay_cosmos_6d.py \
  --hdf5 dataset_raw/pick_spoon/0731_103412/trajectory.hdf5 \
  --parquet debug_outputs/0731_103412_rotation6d.parquet \
  --report debug_outputs/0731_103412_rotation6d_report.csv \
  --execute \
  --env_factory your_module:create_robot_env \
  --max_execute_steps 1
```

程序还会要求在终端输入 `EXECUTE`。环境 factory 必须返回兼容当前 `BaseEnv` 的对象，至少提供
`reset()`、`step(action_7d)` 和可选的 `close()`。

在执行前必须确认机器人初始位姿、TCP/flange 定义、工作空间、夹爪方向和急停可用。

## 常见问题

### 提示缺少 T5 embeddings

如果本次只验证 VAE/action/proprio，传入：

```bash
--skip-t5
```

如果需要在转换 observation 中包含文本 embedding，传入有效缓存：

```bash
--t5-embeddings /path/to/task_t5_embeddings.pkl
```

缓存中的文本键必须与实际 `--task` 字符串完全一致。

### 已有输出被跳过

脚本在输出存在 `meta/info.json` 时会读取 `cosmos_dataset_metadata.json`，只有任务文本、
action source、action encoding、scale和T5设置一致时才跳过；不一致时会拒绝复用。
如果需要重新生成，请使用新的输出目录，或在确认目标后手动处理旧输出；脚本不会自动删除数据。

### VAE OOM

减小：

```bash
--encode-batch-size 4
```

### 旧 pick_spoon Parquet 是 7D

旧目录 `dataset_cosmos/pick_spoon_cosmos` 是 legacy Euler。新的 6D 正式输出应写入
`dataset_cosmos/pick_spoon_puppet_next_frame_cosmos_rotation_6d`，不要覆盖或混合。
