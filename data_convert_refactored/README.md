# Refactored Cosmos data conversion

该目录是正式数据转换实现。旧目录保留兼容Shell入口和尚未迁移的Python工具，方便旧命令继续运行。

## 模块边界

```text
convert.py         CLI入口
config.py          类型化用户配置和运行配置
pipeline.py        数据准备流程编排
preflight.py       数据布局、机器人类型和FK一致性检查
cosmos_backend.py  Cosmos转换阶段编排
dataset_writer.py  LeRobot schema、episode与metadata写入
memory_monitor.py  全流程内存监控

preparation/       HDF5、action、proprio、stats和运动学准备
conditioning/      相机处理、reward/done和transition构造
encoding/          Policy加载、CPU microbatch VAE和多GPU执行
tools/             stats生成和转换结果对比命令
scripts/           正式Shell入口
scripts/jobs/      特定任务批处理入口
tests/             无GPU单元测试和可选真实VAE测试
```

Shell脚本真实实现位于本目录。`data_convert/*.sh` 仅保留兼容转发，旧命令仍可用。

正式转换不再通过子进程重新运行旧转换脚本。`pipeline.py` 先由
`make_action_encoder()` 计算每条 episode 的完整 `(T,D)` action，再把
`EncodedEpisode` 交给 `cosmos_backend.py`。新流程不再导入
`convert_raw_to_cosmos.py`；后端使用重构目录内的图像预处理、VAE编码、
action/proprio潜帧注入和LeRobot写入模块，不负责选择action来源，也不执行FK。

Stats只通过 `cosmos_utils.load_dataset_stats()` 加载一次。Pipeline完成维度、语义和
范围检查后，把同一份有效stats对象传给Backend；action、proprio和metadata共用该对象。
运行阶段使用类型化 `CosmosRuntimeConfig`，不再通过 `SimpleNamespace` 兼容旧配置字段。

双臂腕部图像默认使用实验确认后的预处理：保留完整高度，裁剪中心向右偏移64个原图像素并保留75%宽度；左右图分别缩放为宽224×高112，再按左上、右下拼成224×224。单臂图像路径不应用该双臂融合裁剪。

头部主相机单独删除顶部100像素：原始`640×480`保留`x=0:640、y=100:480`，然后缩放为224×224。该纵向裁剪不会应用到左右腕部，也不会应用到名称不含`head`的主相机。

```text
--wrist-crop-mode center_width
--wrist-crop-fraction 0.75
--wrist-left-center-offset-x 64
--wrist-right-center-offset-x 64
--head-crop-top-pixels 100
```

数据边界如下：

```text
HDF5 -> Episode -> ActionEncoder -> EncodedEpisode(primary + Euler + rotation-6D)
     -> transition(precomputed_actions) -> Cosmos VAE/injection -> Parquet
```

其中 `action` 是CLI选中的训练表示，完成dataset min/max归一化并注入latent；
`action.euler_control` 和 `action.rotation_6d_control` 由同一个相对位姿同时计算，
仅用于审查和反解，不做第二层dataset归一化，也不注入latent。

Parquet同时保存 `action.latent_chunk`，形状为`(16, action_dim)`。它不是在
writer中重新计算，而是直接保存实际传给Cosmos
`replace_latent_with_action_chunk()`的dataset-normalized Tensor；末尾越界位置
重复episode最后一个action。该字段可与`video[:, 4, :, :]`逐元素核对latent注入。

当 `--action-encoding cosmos_rotation_6d` 时，Parquet还保存未归一化的当前观测、
目标动作和增量动作。它们与训练字段 `action` 并存，不改变现有20D训练action、
stats、16D proprio或latent注入。

| Parquet字段 | 单臂维度 | 时刻与定义 | HDF5来源 |
|---|---:|---|---|
| `action.latent_chunk` | `16×action_dim` | 实际注入latent索引4的`action[t:t+16]`，末尾重复补齐 | 归一化训练action |
| `observation.rotation_6d` | 6 | 当前`t`的绝对EE旋转 | 当前EE pose中的四元数 |
| `observation.ee_pose_xyzw` | 7 | 当前`t`的绝对EE位姿 | `puppet/end_effector_*_pose_align/data[t]` |
| `observation.joint_position` | 关节数 | 当前`t`的绝对关节位置 | `puppet/arm_*_position_align/data[t]` |
| `observation.gripper` | 1 | 当前`t`的夹爪位置 | `puppet/end_effector_*_position_align/data[t]` |
| `action.target_rotation_6d` | 6 | 目标`t+1`的绝对EE旋转 | 目标EE pose中的四元数 |
| `action.target_ee_pose_xyzw` | 7 | 目标`t+1`的绝对EE位姿 | `puppet/end_effector_*_pose_align/data[t+1]` |
| `action.target_joint_position` | 关节数 | 目标`t+1`的绝对关节位置 | `puppet/arm_*_position_align/data[t+1]` |
| `action.target_gripper` | 1 | 目标`t+1`的夹爪位置 | `puppet/end_effector_*_position_align/data[t+1]` |
| `action.delta_rotation_6d` | 6 | `inv(R_current) @ R_target`的矩阵前两列 | 当前、目标EE pose计算 |
| `action.delta_ee_xyz_euler` | 6 | `inv(T_current) @ T_target`的局部`[xyz,euler_xyz]`，单位m/rad | 当前、目标EE pose计算 |
| `action.delta_joint_position` | 关节数 | `joint_target - joint_current` | 当前、目标arm position计算 |

默认 `puppet_next_frame` 下，`current=t`、`target=t+1`。双臂按left、right拼接。
末帧`observation.*`保存真实末帧状态；没有真实`t+1`的`action.*`复制最后一个有效
动作并继续标记为padding。rotation-6D字段不是HDF5原字段，由EE四元数计算。

Parquet还保存action/proprio实际依赖的对齐低维源字段。默认
`puppet_next_frame` 保存 `source.puppet.pose_xyzw` 和
`source.puppet.gripper`；master分支按需额外保存master pose或
puppet/master joints与master gripper。双臂字段均按left、right顺序拼接。
不重复保存图像、timestamp或当前分支未使用的HDF5字段。

## 相机状态

转换器通过 `--camera-state` 选择相机预处理配置，目前仅支持且默认使用
`normal`。它对应已验证方案：双腕图像保持完整高度、水平保留75%、左右裁剪中心
均右移64像素；头部相机从顶部裁掉100像素。

省略具体crop参数时，由相机状态提供默认值；显式传入crop参数仍可用于实验覆盖。
状态和最终生效的裁剪参数都会写入转换metadata。

## Action两层归一化

正式转换与 `collect_data_cosmos.py` 对齐，按以下顺序处理action：

```text
相对SE(3)
-> 第一层机器人控制scale（translation/0.02，Euler/0.06，gripper/1.0；rotation-6D不除rotation scale）
-> 加载预先用全量训练集合并计算的actions_min/actions_max
-> policy.rescale_action映射到[-1,1]
-> Parquet单步action与latent action chunk
```

正式转换强制传入 `--dataset-stats`，不再按当前输入目录临时统计。这样 success、failure 和分批转换的数据可以共享同一套归一化。统计文件旁必须存在同名 `.metadata.json`，用于校验 action source、编码、scale、维度和末帧策略。机器人推理时必须使用同一统计文件撤销第二层归一化。

Stats 有两个显式模式：`--stats-mode generated --dataset-stats ...` 使用我们生成且带
sidecar 的共享统计；`--stats-mode official --official-dataset-stats ...` 直接使用
`collect_data_cosmos.py` 的原始统计，不要求 sidecar。official 模式通过同一个
`load_dataset_stats()` 加载，正式转换通过 `CosmosPolicy.rescale_action()` 与
`cosmos_utils.rescale_proprio()` 归一化，不裁剪超界值，只警告并写入 metadata。

先对完整训练集合并生成一次统计（`--input` 可重复）：

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

`puppet_next_frame` 的末帧没有真实 `t+1`，因此复制倒数第二帧的有效 action；metadata 同时标记它是 synthetic/padding，含义是“数值可用，但目标帧不是新观测”。单帧 episode 只能使用 self-delta。

为避免与官方函数相同的除零问题，如果全量训练数据存在
`actions_max == actions_min` 的常量action通道，转换会在加载VAE前明确失败并报告通道索引。

## 先做无GPU预检

从 `HIL-RL` 目录运行：

```bash
PYTHONPATH=. /media/jushen/mingbo-ge/.venv/bin/python3 \
  -m data_convert_refactored.convert \
  --input dataset_raw/pick_spoon/0731_103412 \
  --output dataset_cosmos/pick_spoon_refactored \
  --task "pick spoon" \
  --episode-outcome success \
  --stats-mode generated \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --action-source puppet_next_frame \
  --action-encoding cosmos_rotation_6d \
  --skip-t5 \
  --preflight-only
```

预检会打印机器人类型、episode数量、相机、action/proprio维度、第一条episode的action范围和末帧语义，不加载VAE，也不写数据集。

## 正式转换

确认预检输出后，去掉 `--preflight-only`：

```bash
PYTHONPATH=. /media/jushen/mingbo-ge/.venv/bin/python3 \
  -m data_convert_refactored.convert \
  --input dataset_raw/pick_spoon/0731_103412 \
  --output dataset_cosmos/pick_spoon_refactored \
  --task "pick spoon" \
  --episode-outcome success \
  --stats-mode generated \
  --dataset-stats dataset_raw/pick_spoon/shared_dataset_statistics.json \
  --action-source puppet_next_frame \
  --action-encoding cosmos_rotation_6d \
  --skip-t5
```

## 多 GPU VAE 编码

转换器保持 episode 串行和单一 LeRobot writer，只将每个 CPU video
micro-batch 通过转换侧 `data_convert_refactored.encoding.multi_gpu_vae.VAEReplicaPool` 按帧分到
多张 GPU。主卡复用已有 VAE wrapper，其余卡通过 Cosmos 原生 lazy config 分别
初始化一份完整 wrapper，并在整个转换进程内缓存。因此不会出现 episode index、
Parquet 或 metadata 并发写入冲突。默认仍为原单卡路径。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash data_convert_refactored/scripts/convert.sh \
  ... \
  --encode-world-size 8 \
  --encode-batch-size 16
```

也可以指定 `CUDA_VISIBLE_DEVICES` 映射后的进程内 GPU 编号：

```bash
--encode-device-ids 1,3,5,7
```

CLI 参数优先于 `ENCODE_WORLD_SIZE` / `ENCODE_CUDA_DEVICES` 环境变量。显式
device IDs 优先于 world size。为了让 8 张卡都有样本，建议
`--encode-batch-size >= 8`。

## 天翼master-joint官方分支

```bash
PYTHONPATH=. /media/jushen/mingbo-ge/.venv/bin/python3 \
  -m data_convert_refactored.convert \
  --input /path/to/tienyi/task \
  --output dataset_cosmos/tienyi_master_fk \
  --task "task instruction" \
  --episode-outcome success \
  --stats-mode generated \
  --dataset-stats /path/to/shared_dataset_statistics.json \
  --action-source master_joint_fk_same_frame \
  --action-encoding cosmos_rotation_6d \
  --kinematics-config /path/to/tienyi_prod2_dual_arm.json \
  --skip-t5 \
  --preflight-only
```

天翼分支必须先通过 FK 对 HDF5 puppet EE pose 的检查，失败时不会进入VAE阶段。

## 测试

```bash
PYTHONPATH=.:lerobot/src:../cosmos-policy:../cosmos-policy/cosmos_policy:rl_envs \
  ../.venv/bin/python \
  -m unittest discover -s data_convert_refactored/tests -v
```

当前重构版与旧版在单臂和双臂 `puppet_next_frame + cosmos_rotation_6d` action 上要求逐元素一致。

多卡正式回归使用同一个 episode、同一套 stats 和 task，分别写入两个全新的输出目录：

```bash
bash data_convert_refactored/scripts/convert.sh \
  --input /path/to/one_episode_or_task \
  --output /tmp/cosmos_single \
  --task-description "same task" \
  --episode-outcome success \
  --stats-mode official \
  --official-dataset-stats /path/to/dataset_statistics.json \
  --skip-t5 --max-episodes 1 \
  --encode-world-size 1

bash data_convert_refactored/scripts/convert.sh \
  --input /path/to/one_episode_or_task \
  --output /tmp/cosmos_multi \
  --task-description "same task" \
  --episode-outcome success \
  --stats-mode official \
  --official-dataset-stats /path/to/dataset_statistics.json \
  --skip-t5 --max-episodes 1 \
  --encode-world-size 8 --encode-device-ids 0,1,2,3,4,5,6,7

../.venv/bin/python -m data_convert_refactored.tools.compare_conversion_outputs \
  /tmp/cosmos_single /tmp/cosmos_multi
```

对比器要求 episode 文件集合、行数、列名和非浮点字段完全一致；浮点字段默认使用
`rtol=1e-5, atol=1e-6`。同时检查多卡日志包含 8 个 encode devices，且
`cosmos_dataset_metadata.json` 中 `episode_parallelism=false`、`writer_processes=1`，
用于确认只有 VAE micro-batch 被分片，episode 写入仍然串行且无冲突。

真实 VAE wrapper 的两卡/八卡 smoke test 默认跳过，显式运行：

```bash
RUN_REAL_COSMOS_VAE_TEST=1 REAL_COSMOS_VAE_WORLD_SIZE=8 \
  ../.venv/bin/python -m unittest \
  data_convert_refactored.tests.test_multi_gpu_vae.RealCosmosVAEReplicaTest -v
```

注意：单卡整批和多卡分片会采用不同的 bfloat16 kernel 形状，可能产生正常的低精度
舍入差异。严格验证 wrapper 权重一致性时，应使用相同的每卡 batch 形状；端到端对比
可以通过 `compare_conversion_outputs` 的 `--atol/--rtol` 按实际误差设置容差。
