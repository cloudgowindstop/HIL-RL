# Cosmos 数据转换 Pipeline

## 数据来源

### BOS 存储结构

```
bos:/bd-dp-ten-6spt6-scjd/
├── lerobot_data_v3/          ← LeRobot v3 格式 (parquet + MP4)
│   └── {robot_type}/
│       └── {task_name}/
│           └── success/
│
└── raw_data/                 ← 原始采集数据 (trajectory.hdf5 × N)
    └── {robot_type}/
        └── {task_name}/
            └── success_episodes/
                ├── {timestamp_1}/data/trajectory.hdf5
                ├── {timestamp_2}/data/trajectory.hdf5
                └── ...
```

### Lerobot v3 格式 (parquet + MP4)

每个任务一个目录，包含全部 episode：

```
success/
├── meta/
│   ├── info.json           ← robot_type, fps, features, splits
│   ├── stats.json          ← 各列 min/max/std/q01/q50/q99
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/chunk-000/
│   └── file-000.parquet    ← 所有行 (frame级)
└── videos/
    ├── camera_observations.color_images.camera_head/chunk-000/file-*.mp4
    ├── camera_observations.color_images.camera_left/chunk-000/file-*.mp4
    └── camera_observations.color_images.camera_right/chunk-000/file-*.mp4
```

### Raw HDF5 格式

每个 episode 一个文件：

```
success_episodes/
├── {timestamp_1}/data/trajectory.hdf5    ← episode 0
├── {timestamp_2}/data/trajectory.hdf5    ← episode 1
└── ...
```

---

## Lerobot ↔ Raw data 对应关系

同一个任务的两种存储格式，路径映射规则：

```
lerobot:  .../lerobot_data_v3/{robot_type}/{task_name}/success/
raw:      .../raw_data/{robot_type}/{task_name}/success_episodes/
```

### 具体例子

```
任务: tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260626_am

lerobot:
  lerobot_data_v3/tienyi_prod2_dualArm-gripper-3cameras_128/
    tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260626_am/
      success/                    ← 40 episodes 合并为 parquet + MP4

raw:
  raw_data/tienyi_prod2_dualArm-gripper-3cameras_128/
    tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260626_am/
      success_episodes/
        0626_151539/data/trajectory.hdf5    ← episode 0
        0626_151718/data/trajectory.hdf5    ← episode 1
        ... (共 40 个)
```

---

## 任务命名规则

任务名即完整标识符，格式为：

```
{robot_type}_{task_name}_{date}_{period}
```

### 当前已知的 robot_type

| robot_type | 备注 |
|------|------|
| `tienyi_prod2_dualArm-gripper-3cameras_128` | 双臂 3 相机 128 配置 |
| `tienyi_prod2_dualArm-gripper-3cameras_134` | 双臂 3 相机 134 配置 |

### 当前已知的任务

从 Excel "中试工厂天轶移动采集" sheet：

| 任务类型 | camera 配置 | BOS 地址数 |
|------|:---:|:---:|
| plug_in_ethernet_type-c_usb | 128 | 8 |
| plug_in_ethernet_type-c_usb | 134 | — |
| assemble_bearings | 128 | — |
| assemble_bearings | 134 | 9 |

---

## 数据内容对比

### LeRobot v3 parquet 关键列

| 列 | 维度 | 含义 |
|------|:---:|------|
| `puppet.end_effector_left_pose_align.data` | 7 | 左末端绝对位姿 (xyz + qxyzw) |
| `puppet.end_effector_right_pose_align.data` | 7 | 右末端绝对位姿 |
| `puppet.end_effector_left_position_align.data` | 1 | 左夹爪开合度 |
| `puppet.end_effector_right_position_align.data` | 1 | 右夹爪开合度 |
| `puppet.arm_left_position_align.data` | 7 | 左臂关节角度 |
| `puppet.arm_right_position_align.data` | 7 | 右臂关节角度 |
| `camera_observations.color_images.camera_head` | video | 头部相机 (640×480) |
| `camera_observations.color_images.camera_left` | video | 左腕相机 (640×480) |
| `camera_observations.color_images.camera_right` | video | 右腕相机 (640×480) |

### Raw HDF5 关键字段

```
observations/rgb_images/camera_right      ← 主视图
observations/rgb_images/camera_wrist      ← 腕部
puppet/delta_end_effector                 ← 增量末端位姿 (动作)
puppet/end_effector                       ← 绝对末端位姿
puppet/hand_joint_position                ← 夹爪
language_instruction                      ← 任务文本
```

---

## 转换目标：Cosmos 训练格式

### Cosmos 数据流 (单臂单腕 9 帧)

```
SERLObsWrapper → CosmosWrapper → encode_episode_state → LeRobot

最终存储特征:
  video:               (16, 9, 28, 28) float32  ← VAE 潜视频
  proprio:             (8,)  float32            ← 末端位姿(7)+夹爪(1)，已归一化
  future_proprio:      (8,)
  value_function_return: (1,)
  action:              (7,)  float32            ← 增量末端位姿(6)+夹爪(1)
```

### 双臂扩展 (RoboMIND 双腕拼接)

```
  video:               (16, 9, 28, 28) float32  ← 和单臂相同布局
  proprio:             (16,) float32            ← 左8 + 右8
  action:              (14,) float32            ← 左7 + 右7
  两腕上下拼接 → resize 224×224 → 填充 wrist 帧
```

---

---

## convert_raw_to_lerobotv2.py 产出分析

### 输出格式：LeRobot v2.1

每个 episode 独立存一份 parquet + 3 个 MP4：

```
output_dir/{task_name}/
├── meta/
│   ├── info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── episode_000001.parquet ...
└── videos/chunk-000/
    ├── observation.image.image/  (or observation.images.image)
    ├── observation.image.left/
    └── observation.image.right/
```

### 产出的特征 (与 Cosmos 的差距)

| v2 特征 | 维度 | 数据来源 | Cosmos 是否需要 | 差距 |
|---------|:---:|---------|:---:|------|
| `observation.state.puppet_left_arm_position` | (7,) | 左臂关节角 | ✗ | Cosmos 需要末端位姿，不是关节角 |
| `observation.state.puppet_right_arm_position` | (7,) | 右臂关节角 | ✗ | 同上 |
| `observation.state.puppet_left_gripper_position` | (1,) | 左夹爪 | ✓ | 可用，需归一化 |
| `observation.state.puppet_right_gripper_position` | (1,) | 右夹爪 | ✓ | 可用，需归一化 |
| `observation.state.master_left_arm_position` | (7,) | 主手左关节角 | ✗ | 主手数据，不需要 |
| `observation.state.master_right_arm_position` | (7,) | 主手右关节角 | ✗ | 同上 |
| `observation.state.master_left_gripper_position` | (1,) | 主手左夹爪 | ✗ | 同上 |
| `observation.state.master_right_gripper_position` | (1,) | 主手右夹爪 | ✗ | 同上 |
| `action.left_arm_position` | (7,) | **绝对关节角** (抄自 master) | ✗ **严重不匹配** | Cosmos 需要**增量末端位姿** |
| `action.right_arm_position` | (7,) | **绝对关节角** | ✗ | 同上 |
| `action.left_gripper_position` | (1,) | 主手夹爪 | △ | 来源不对，应用 puppet 而非 master |
| `action.right_gripper_position` | (1,) | 主手夹爪 | △ | 同上 |
| `observation.image.image` | video (336×336) | 头部相机 | ✓ | 可用，需 resize 到 224×224 |
| `observation.image.left` | video (336×336) | 左腕相机 | ✓ | 可用 |
| `observation.image.right` | video (336×336) | 右腕相机 | ✓ | 可用 |
| `value_reward` | (1,) | 稀疏奖励 | △ | Cosmos 用 Monte Carlo return |
| `discounted_value_return` | (1,) | 折现累计回报 | ✓ | 可直接映射为 value_function_return |

### 核心差距

**动作不匹配** — LeRobot v2 的动作是：
```
action = 绝对关节角度 (master arm position)，16D
```

Cosmos 需要的是：
```
action = 增量末端位姿 (delta EE pose, 欧拉角)，14D
         左(dx,dy,dz,rx,ry,rz,grip) + 右(dx,dy,dz,rx,ry,rz,grip)
```

**缺少末端位姿** — v2 没有提取 `puppet/end_effector_*_pose_align` 等 EE pose 数据。

### 实际 raw HDF5 内容（已下载样本确认）

```
1520 帧，30 FPS，双臂移动操作机器人
```

| HDF5 字段 | 维度 | 用途 |
|------|:---:|------|
| `puppet/end_effector_left_pose_align/data` | (T, 7) | **左末端位姿** (xyz+qxyzw) ← Cosmos 需要 ✓ |
| `puppet/end_effector_right_pose_align/data` | (T, 7) | **右末端位姿** ← Cosmos 需要 ✓ |
| `puppet/end_effector_left_position_align/data` | (T, 1) | 左夹爪 ← Cosmos 需要 ✓ |
| `puppet/end_effector_right_position_align/data` | (T, 1) | 右夹爪 ← Cosmos 需要 ✓ |
| `puppet/arm_left_position_align/data` | (T, 7) | 左关节角（备用） |
| `puppet/arm_right_position_align/data` | (T, 7) | 右关节角（备用） |
| `camera_observations/color_images/camera_head` | (T,) JPEG bytes | 头部相机 (640×480) |
| `camera_observations/color_images/camera_left` | (T,) JPEG bytes | 左腕相机 |
| `camera_observations/color_images/camera_right` | (T,) JPEG bytes | 右腕相机 |
| `metadata/language_instruction` | str | 任务描述 |

**注意**：没有 `delta_end_effector` 字段——action 需要从绝对位姿计算增量。

### 决策

**Cosmos 转换脚本直接从 raw HDF5 读取，跳过 LeRobot v2 中间层。**

原因：
1. raw HDF5 已经包含 Cosmos 需要的所有字段（EE pose、gripper、三相机）
2. `convert_raw_to_lerobotv2.py` 不提取 EE pose，只提取关节角 — 补齐它和直接读 raw 的工作量差不多
3. 少一层中间格式，减少可能的转换误差

---

## 自动化 Pipeline 设计

### 三阶段（修正版）

```
Stage 1: 下载 raw_data
  Excel → 解析 BOS 地址
  下载 raw HDF5 → dataset_raw/{task_name}/

Stage 2: 环境
  source .venv
  export PYTHONPATH (参考 collect_data.sh)

Stage 3: 转换 (直接 raw HDF5 → Cosmos LeRobot)
  data_convert/convert_raw_to_cosmos.py
    --input  dataset_raw/{task_name}/success_episodes/
    --output dataset_cosmos/{task_name}_cosmos/

  自动检测 robot_type → 单臂/双臂 → action_dim/proprio_dim
  直接读 HDF5 的 EE pose 计算增量动作
```

### 与 collect_data_cosmos.py 的一致性

转换脚本产生的 LeRobot 数据集与 `collect_data_cosmos.py` 实机采集的格式**完全一致**：

| 特征 | 实机 (Franka 单臂) | 离线转换 (Tienyi 双臂) |
|------|:---:|:---:|
| `video` | (16, 9, 28, 28) float32 | (16, 9, 28, 28) float32 |
| `proprio` | (8,) | (16,) |
| `future_proprio` | (8,) | (16,) |
| `value_function_return` | (1,) | (1,) |
| `action` | (7,) | (14,) |
| 数据流 | encode_episode_state → add_frame | **同** |
| VAE 编码 | CosmosPolicy.encode() | **同** |
| 潜帧注入 | replace_latent_with_* | **同** |

### 需要新建的文件

```
HIL-RL/
  data_convert/convert_raw_to_cosmos.py  ← 主转换脚本
  pipeline_download_and_convert.sh  ← 一键下载+转换
```

---

## 下载命令

```bash
BCECMD=/media/linux-bcecmd-0.5.1/bcecmd
DATASET_DIR=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/dataset_lerobot

# 单个任务
$BCECMD bos sync \
  bos:/bd-dp-ten-6spt6-scjd/lerobot_data_v3/{robot_type}/{task_name}/success \
  $DATASET_DIR/{task_name}
```

## 环境变量 (参考 collect_data.sh)

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/
export PYTHONPATH=$PYTHONPATH:../../../cosmos-policy/cosmos_policy/
export PYTHONPATH=$PYTHONPATH:../../lerobot/src/
export PYTHONPATH=$PYTHONPATH:../../rl_envs/
export HF_HUB_CACHE=/path/to/Cosmos-Policy-LIBERO-Predict2-2B/
export HF_ENDPOINT=https://hf-mirror.com
```
