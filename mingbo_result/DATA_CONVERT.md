# 数据转换：实现、对照采集脚本、改进

`collect_data_cosmos.py` 是**在线 HIL 采集**：环境逐步交互，回合结束再 VAE 编码并写入 LeRobot。  
`data_convert_refactored/` 是**离线工厂**：已有 HDF5 轨迹转成同一套 Cosmos 训练 Parquet。

二者共享训练合同（latent 布局、两层 action 归一化、官方 `rescale`），不共享入口职责。离线公平对比用的 14D / 20D 数据，走的是转换项目，不是采集脚本。代码根：`HIL-RL/data_convert_refactored/`。训练侧见 [CODE.md](CODE.md)。

## 成果摘要（截至 2026-09-11）

本阶段完成了两件不同但相连的工作：先把 BOS 原始数据下载、校验并按 HDF5 Schema 组织；再建立可审计、可续跑、支持 Euler/rotation-6D 与多 GPU VAE 的 Cosmos 离线转换链路。下列数字来自当前磁盘上的 manifest、LeRobot `meta/info.json` 和 Parquet 文件，不是估算值。

### 原始数据下载与 Schema 整理

| 指标 | 实际结果 |
| --- | ---: |
| 首批扫描 HDF5 | 7,947 |
| 9台设备表候选 BOS | 97 |
| 历史已下载 BOS | 74 |
| 增量补下载 BOS | 23（修复 1 + 新增 22） |
| 增量 HDF5 | 3,068 |
| 增量数据量 | 1,683,000,262,708 bytes（约 1.53 TiB） |
| 正式 Schema 数据集当前物理 HDF5 | 11,015 |
| 当前 Schema ID | 13 |
| 当前 quarantine | 8 |

增量 23/23 个批次均有完成标记，3,068/3,068 个 HDF5 可打开；移动计划 307/307 项写入 journal 且状态为 `verified`，源增量目录剩余 HDF5 为 0。之所以不是 23 次移动，是两个混合质量批次按 episode 拆分，正常数据和异常数据分别进入正式目录与 `_quarantine`。

当前正式数据按结构大类分布：

| 结构大类 | HDF5 |
| --- | ---: |
| A：Head 图像 + Puppet pose + 力传感器 | 5,885 |
| B：Head 图像 + Puppet pose、无力传感器 | 326 |
| C：Top 图像 + 只有关节角 | 4,557 |
| D：Head 图像 + 只有关节角 | 239 |
| quarantine（跨结构合计） | 8 |

正式入口：

```text
/media/jushen/project-rl-dataset/raw_data_0831download_by_schema
```

已知有 1 个物理重复：修复完整 `20260803_pm` 后，`0804_004915` 与之前单独下载的副本 SHA-256 都是 `6d33d03e...53e223fb`。因此 11,015 是物理文件数；按这个已知重复去重后至少为 11,014。生成训练 manifest 时必须排除其中一份，当前未自动删除原文件。

### 已生成的 Cosmos/LeRobot 数据

目前已经生成一套完整的 Euler/rotation-6D 成对数据。两种表示使用相同的 325 个 episode，逐批次 episode 数、帧数及每条 episode 的 length 完全一致，因此可以直接进行表示方法对照，差异不是由数据子集造成的。

| 表示 | 根目录 | Action | Episode/Parquet | 帧数 | Parquet 大小 |
| --- | --- | ---: | ---: | ---: | ---: |
| 3D Euler/RPY | `cosmos_data_3drpy` | 14D | 325 | 436,360 | 45.34 GiB |
| rotation-6D | `cosmos_data_6drotation_monitor_20260819` | 20D | 325 | 436,360 | 48.71 GiB |

绝对路径：

```text
/media/jushen/mingbo-ge/HIL-RL-Project/cosmos_data_3drpy
/media/jushen/mingbo-ge/HIL-RL-Project/cosmos_data_6drotation_monitor_20260819
```

四个源批次的成对统计如下：

| 批次 | Outcome | Episode | 帧数 |
| --- | --- | ---: | ---: |
| `back_handle_20260803_pm_failure` | failure | 8 | 7,159 |
| `back_handle_20260803_pm_success` | success | 159 | 246,052 |
| `back_handle_20260804_am_failure` | failure | 96 | 86,991 |
| `back_handle_20260804_am_success` | success | 62 | 96,158 |
| **合计** | success=221，failure=104 | **325** | **436,360** |

`20260803_pm_success` 的原始完整批次有 160 个 HDF5，其中 `0804_014423` 已在 Schema 校验中因 timestamp 停滞、缺少 `trajectory_length` 和多字段长度不一致进入 quarantine；因此两套可训练转换数据都使用其余 159 条。这是明确的数据质量过滤，不是转换中断。四个批次原始合计 326 条，最终有效转换 325 条，保留率 99.69%。

6D 数据的 `action` 是 20D，并额外保存 14D `action.euler_control`、20D `action.rotation_6d_control`、`source.puppet.pose_xyzw` 和 `source.puppet.gripper`，可用于从 6D 反解并核对原始控制来源。3D RPY 数据的 `action` 是 14D。两套 `video` 都是 `(16,9,28,28)` float32 latent。

这两套既有 Parquet 中尚无后来新增的 `action.latent_chunk` 字段；“最新版 writer 能保存 latent action chunk”和“这两套历史输出已经保存该字段”必须区分。如果训练或审计强依赖该列，应使用当前代码重转，或另行做不改变训练 action 的字段补充。

`cosmos_data_3drpy/install_handle` 是上述 325 条 Euler 数据的合并版本；`install_handle_split/train` 和 `eval` 是其 309/16 划分，仍然是同一批数据，不能再次累计为新增 episode。

较早的 `/media/jushen/project-rl-dataset/dataset_cosmos_back_handle` 仅有 58 episodes、73,314 帧、7.83 GiB，是 CPU OOM/续跑优化阶段留下的部分转换产物。它不再代表当前完整成果，也不能与上述 325 条相加。

仓库内 `HIL-RL/dataset_cosmos/` 还保留 11 组开发/回归产物，共 71 个 Parquet、82,148 帧、8.76 GiB。其中包含同一 raw episode 的旧版、新版、内存测试和 crop/shape trace 对照，不能与生产数据相加作为“唯一训练样本数”。其中较完整的独立实验包括：

- `insert_hose_20260626_pm`：29 episodes，28,992 帧；
- `pick_spoon_cosmos`：5 episodes，1,120 帧；
- CPU microbatch 内存回归：14 episodes，24,348 帧；
- 修改前内存基线：12 episodes，20,875 帧。

另有 `/media/jushen/project-rl-dataset/cosmos_dataset`：10 episodes、8,481 帧、0.97 GiB；因其来源与当前成对基线命名不一致，报告单列，不计入上述 325 条。

### 验证状态

2026-09-11 使用项目显式虚拟环境运行完整 CPU 单元测试：

```text
Ran 60 tests in 21.206s
OK (skipped=4)
```

即 56 项通过、0 项失败、4 项按条件跳过。跳过项分别依赖 CUDA tensor、两张 GPU 或 `RUN_REAL_COSMOS_VAE_TEST=1`；它们不是测试失败。已覆盖：HDF5 reader、三种 action source、Euler/6D、末帧 padding、两层 stats、crop、reward/done、latent slot、manifest、resume 事务、存储探测和多 GPU 路由。真实 VAE 的单/多卡等价测试仍需在满足 GPU 条件时单独执行。

## 实现了什么

按数据流。

```text
HDF5 → Episode → ActionEncoder → EncodedEpisode(primary + Euler + rotation-6D)
     → transition(预计算 action/proprio) → Cosmos VAE / latent 注入 → Parquet
```

**准备。** `pipeline.py` 发现 `trajectory.hdf5`，读成 `Episode`。`--preflight-only` 只核机器人类型、相机、action/proprio 维、FK（天翼）、第一条 episode 的 action 范围和末帧 padding，不加载 VAE，不写盘。

**Action。** `make_action_encoder()` 按 CLI 选来源和时间对齐，对每条 episode 算出完整 `(T,D)`：

- 来源：`puppet_next_frame`（默认，`current=t` / `target=t+1`）、`master_same_frame`、`master_joint_fk_same_frame`
- 表示：`legacy_euler`（单/双臂 7/14D）或 `cosmos_rotation_6d`（10/20D）

同一相对 SE(3) 同时出 Euler 和 6D。写入训练字段 `action` 的那条再做 dataset min/max；`action.euler_control` / `action.rotation_6d_control` 只审查、不二次归一化、不注入 latent。`puppet_next_frame` 末帧没有真实 `t+1`，复制倒数第二帧有效 action，并标 synthetic/padding。

两层归一化与采集脚本对齐：

```text
相对 SE(3)
→ 控制 scale（translation/0.02，Euler/0.06，gripper/1.0；6D 不除 rotation scale）
→ 全量训练集合并的 actions_min / actions_max
→ CosmosPolicy.rescale_action → [-1, 1]
→ 单步 action 与 latent action chunk
```

Stats 强制外置：`--stats-mode generated` 要带 sidecar 的共享统计；`--stats-mode official` 直接吃 `collect_data_cosmos.py` 那份，不要求 sidecar。常量通道 `max==min` 在加载 VAE 前失败。

**相机。** 状态默认 `normal`：双腕全高、水平留 75%、裁剪中心右移 64px，左右缩到 224×112 再左上/右下拼 224×224；头部 `640×480` 去掉顶部 100px 再缩到 224。单臂不做双腕融合。参数进 conversion metadata。

**Conditioning / VAE。** `transition_builder` 再打开 HDF5 只读图像。每步 video 模板 `(1,3,33,224,224)` uint8；编码后 `(1,16,9,28,28)`。注入槽位固定：action=4、当前 proprio=1、future proprio=5、value=8，调用官方 `replace_latent_with_action_chunk` / `replace_latent_with_proprio`。`action.latent_chunk` 形状 `(16, D)`，就是实际注入 slot 4 的那份 tensor，末尾越界重复最后一步。`--save-clean-restore-latent` 在注入前保存槽 1/4/5/8，给 World RGB 评测（见 [CODE.md](CODE.md)）。

**标签。** `--episode-outcome success|failure` 显式指定，不从图像推断。成功轨迹最后 5 帧 `(reward=10, done=True)`，此前 `-0.05`；失败全程 `-0.05, done=False`。演示数据 `is_intervention=True`，**整条 episode 写入**（采集脚本只写干预帧）。

**写入。** 单一 LeRobot writer，episode 串行。多卡只切 VAE micro-batch（`VAEReplicaPool`），不并行写 Parquet。`--resume` 校验已有输出后从下一 episode 续。FPS 来自源数据，`use_videos=False`（`video` 已是 float latent）。

**6D 时额外落盘（不改 20D 训练 action）。** 当前/目标 EE pose、关节、夹爪、`delta_rotation_6d`、`delta_ee_xyz_euler`，以及 action/proprio 依赖的对齐 source 字段。

入口：`python -m data_convert_refactored.convert`；Shell 在 `scripts/`，`data_convert/*.sh` 只做兼容转发。

## 和 `collect_data_cosmos.py` 的对照

| | 采集脚本 | 转换项目 |
| --- | --- | --- |
| 输入 | 真机/仿真 `env.step` | HDF5 目录 |
| 何时 encode | 每回合结束 | 预检通过后按 episode 批处理 |
| Action 从哪来 | 环境 / 遥操作 / 干预 | CLI 选定 source + encoding |
| 相机 | env wrapper | 显式 crop，写入 metadata |
| Reward / done | classifier / env | CLI outcome 规则 |
| 写哪些帧 | 仅 `is_intervention` | 全帧（标成干预演示） |
| Stats | 配置里一条路径 | generated sidecar 或 official |
| 多卡 | `policy.encode_episode_state` + 环境变量 | 转换侧 replica pool；writer 仍单进程 |
| 续跑 | 无 | `--resume` |
| Replay / 建 env | 有 | 无 |

采集脚本仍负责 HIL、干预检测、replay。转换不替代这些；它把已有遥操作数据做成可复现的 Cosmos 离线集，并和采集链路在 VAE 布局与两层归一化上对齐。

采集脚本里和训练合同无关、转换刻意去掉的部分：`action[2] = 1.0`、`input()` 暂停、Hydra+draccus 混配、整文件控制循环与编码揉在一起。

## 亮点

**同一合同，两条入口。** 在线采一回合、离线转一批 HDF5，下游 Cosmos 看到的 latent 槽位和 rescale 函数相同。official stats 模式用来核对「转出来的数」和采集脚本一致。

**Action 可审计，不是黑盒控制量。** 训练用一条归一化 `action`；Euler/6D、绝对 pose、增量、source 字段同盘可对。`action.latent_chunk` 可与 `video[:, 4, :, :]` 逐元素核对。

**公平对比的数据前提。** 同一 raw split 能分别转 14D 和 20D（`--episode-manifest` 冻结顺序）。这是 [CODE.md](CODE.md) 里「不是换一套数据再比表示」的上游。

**先失败、再烧 GPU。** 预检、FK、stats sidecar、shape trace、常量通道检查都在 `init_cosmos_policy` 之前。

**多卡只加速 encode。** metadata 写明 `episode_parallelism=false`、`writer_processes=1`。单卡/多卡输出用 `tools/compare_conversion_outputs.py` 对（浮点默认 `rtol=1e-5, atol=1e-6`）。

## 资源稳定性与多 GPU 优化

### OOM 定位：主要问题实际发生在 CPU/cgroup

转换早期曾出现处理若干 episode 后进程直接终止。结合容器 `memory.events`、RSS/cgroup 采样和关键节点日志，定位到主要风险不是 CUDA 显存，而是整条 episode 视频在 CPU 上一次性执行：

```python
episode_state_video.float() / 127.5 - 1.0
```

原始视频为 `uint8`，转成 `float32` 后单元素由 1 byte 变成 4 bytes；表达式计算期间还可能同时保留原张量和中间结果。以一次真实轨迹的日志为例：

```text
frames=1432
uint8 episode video≈7.1 GB（十进制估算）
完整 float32 副本≈28.4 GB
```

再叠加 transition 列表、未来帧复制、action/proprio chunk、VAE latent、Arrow/Parquet 缓冲及 Python 对象，容易触发容器 cgroup OOM。进程被系统直接杀死时往往没有 Python traceback，因此早期容易误判为 GPU OOM。

### CPU OOM 修复：只归一化当前 VAE microbatch

当前实现不再把整条 episode 一次性变成 float32，而是在 VAE 循环中按 `encode_batch_size` 执行：

```text
取 B 帧 uint8
  → 当前 microbatch 转 float32
  → 原地 div_(127.5).sub_(1.0)
  → 送入 VAE
  → latent 转回 CPU
  → 删除当前 batch 临时对象
  → 处理下一批
```

对于 `B=8`、输入 `(8,3,33,224,224)`，float32 输入约为 152 MiB；峰值由 `B` 控制，不再随 episode 总帧数线性膨胀。这个修改保持输入数值公式不变，也没有删除 latent statistics，因此不改变 Parquet 语义。

需要明确：转换仍会保留一条 episode 的 uint8 transition 视频、conditioning 和最终 latent；本次优化解决的是最危险的“整条视频 float32 膨胀”，不是把整条流水线改成完全流式写盘。

### GPU 显存控制

GPU侧采用相同的 microbatch 边界。单卡路径每次只传输当前归一化 batch，编码结束立即把 latent 搬回 CPU 并释放 `batch_gpu/latent_gpu`。因此：

- 单卡显存峰值主要由 VAE 权重、当前 microbatch 输入、VAE激活和当前 latent 决定；
- 减小 `--encode-batch-size` 可以降低单卡峰值，代价是吞吐下降；
- 增大 batch 可能提高吞吐，但必须通过真实任务监测显存；
- 当前代码没有依赖频繁 `torch.cuda.empty_cache()` 掩盖泄漏，而是依靠对象生命周期和固定 microbatch 控制峰值。

目前没有证据表明已发生的三次停止属于 GPU OOM，因此报告不把 CPU OOM 修复写成“已解决 CUDA OOM”。更准确的结论是：**CPU OOM 已针对根因优化，GPU OOM 通过可调 microbatch 进行预防和控制。**

### 多 GPU VAE 并行设计

Cosmos 的 `Wan2pt1VAEInterface` 不是可直接用 `DataParallel` 搬运的普通 `nn.Module`。当前实现新增 `VAEReplicaPool`：

1. 主卡复用 Policy 已加载的 VAE wrapper；
2. 其余GPU通过 Cosmos 原生 lazy config 各初始化一份完整 wrapper；
3. wrapper在进程内缓存并跨 episode 复用，避免重复加载权重；
4. CPU microbatch沿第0维按帧切分到多卡；
5. 各卡在线程中并发编码；
6. latent回到CPU后按原帧顺序拼接。

```text
CPU batch [0:N]
  ├─ shard 0 → cuda:0 → latent 0
  ├─ shard 1 → cuda:1 → latent 1
  └─ shard k → cuda:k → latent k
                         ↓
             按原始顺序 concat → CPU
```

这种方案刻意只并行最耗时的 VAE encode：

- episode发现、action/proprio计算仍保持确定顺序；
- Parquet与LeRobot metadata仍由一个writer串行写入；
- 不存在多个进程竞争 episode index 或同时修改 metadata；
- 失败恢复和 `--resume` 语义保持不变。

边界也很清楚：每张卡都保存完整 VAE 权重，所以多卡会增加**总GPU显存占用**，但每张卡的输入/激活由分片大小控制；它不是模型并行，也不加速 HDF5 读取、transition构造或Parquet写入。为了实际使用所有GPU，全局 `encode_batch_size` 至少应不小于GPU数量。

### 单卡/多卡正确性验证

项目提供 `scripts/test_multi_gpu_conversion.sh` 和 `tools/compare_conversion_outputs.py`。验证协议使用同一 episode、同一 stats、同一任务文本和两个全新输出目录，依次执行单卡和多卡转换，再比较：

- episode文件集合；
- Parquet行数与列名；
- 非浮点字段精确一致；
- 浮点字段在默认 `rtol=1e-5, atol=1e-6` 下等价；
- metadata明确记录 `episode_parallelism=false`、`writer_processes=1`。

不同分片形状可能调用不同 bfloat16 kernel，出现微小舍入误差是正常现象；不能把浮点逐bit不一致直接判断成多卡逻辑错误。当前 CPU 测试已经覆盖多GPU路由、设备参数校验、分片顺序和输出比较器；真实 VAE 多卡 smoke test需要显式设置 `RUN_REAL_COSMOS_VAE_TEST=1` 后运行。

### 资源监控能力

为避免再次只能看到“进程消失”，转换器加入可选监控：

```text
--monitor-memory --memory-sample-interval N
```

监控同时记录：

- 进程 RSS/VMS；
- cgroup current/limit/peak；
- anon、file、shmem；
- cgroup `oom` / `oom_kill` 计数；
- episode编号、帧数和业务阶段；
- microbatch拼接、float转换、VAE输出和latent拼接等关键事件。

输出为 `memory_samples.csv` 和 `memory_events.jsonl`。监控线程是可选且低侵入的；读取失败只产生一次 warning，不会中断正式转换。它使 CPU OOM、容器限制和GPU编码阶段能够被区分，而不是仅凭终端最后一行猜测。

### 当前性能结论的边界

目前可以确认：

- CPU float32 峰值已从“随整条episode长度增长”改为“受microbatch限制”；
- 多卡分片不会改变episode顺序或writer并发模型；
- 单/多卡等价性已有自动化比较工具和条件测试。

目前**不能**仅凭实现声称固定的多卡加速倍数。真实吞吐还受VAE计算占比、CPU拼接、PCIe传输、HDF5读取、Parquet写盘和batch大小影响。正式汇报加速比前，应对同一批数据记录单卡/2卡/4卡/8卡的总耗时、VAE耗时、每卡峰值显存及CPU峰值。

## 已经改进了什么

相对采集脚本和旧转换目录。

| 问题 | 改动 | 效果和边界 |
| --- | --- | --- |
| 采集脚本 800 行揉环境、编码、写盘 | 模块边界：preparation / conditioning / encoding / writer | 预检可无 GPU；旧 `convert_raw_to_cosmos.py` 不再被正式路径 import |
| 当前输入目录临时统计 min/max | 强制共享 stats；generated 要 sidecar | success/failure、分批转换共用归一化；推理必须同一文件反归一化 |
| Action 来源藏在 env | 三 source × 两 encoding，同一 delta 双出 | 天翼 FK 失败不进 VAE；末帧 padding 语义写进 metadata |
| 相机裁剪不可复述 | `normal` 状态 + 可覆盖 crop | 实验覆盖仍可用 CLI；默认是已验证方案 |
| 整条 episode 视频一次上 GPU | CPU microbatch 归一化再 encode | 降峰值显存；多卡是分片同一 microbatch，不是并行 episode |
| 中断要重跑 | `--resume` + episode 事务提交 | 续跑校验 feature/FPS；不解决源 HDF5 被改过的情况 |
| World RGB 评测会 decode 到被覆盖的槽 | `--save-clean-restore-latent` | 离线评测协议依赖此字段；默认不写以省盘 |
| 无回归 | unittest + 合成 HDF5 + 单/多卡对比器 | 真实 VAE 多卡 smoke 默认 skip，需 `RUN_REAL_COSMOS_VAE_TEST=1` |

### 结构与可维护性改进

- 原 `data_convert/convert_raw_to_cosmos.py` 曾约 1,880 行并同时承担读取、action、VAE、写盘；现在缩为 218 行兼容入口，正式实现不再反向 import 它。
- 正式代码按数据流组织为 `preparation/`、`conditioning/`、`encoding/`，共 26 个非测试 Python 文件；入口、配置、预检、恢复、存储监控与写入职责独立。
- HDF5 只由 preparation reader 建立结构化 `Episode`；action encoder 使用该对象，不再为 action 计算重复打开 HDF5。图像在 transition 阶段按需读取。
- `EncodedEpisode` 已并入 action 准备层；action normalization 与 action 表示放在同一模块语境，dataset stats 保持独立，避免“加载统计”和“变换 action”职责混合。
- Cosmos runtime 使用显式配置对象，不再通过 `_legacy_config`/`SimpleNamespace` 在生产路径传参；字段错误更早暴露。
- 官方 stats 只准备一次，同时保留 `source` 和适配当前 action 维度后的 `effective` 版本；归一化公式继续调用官方逻辑。
- writer、policy loader、transition builder 已从旧大文件迁出；`dataset_writer.py` 保持整体，不为追求小文件而继续拆分事务逻辑。
- Shell 入口移入 `data_convert_refactored/scripts/`，旧脚本保留兼容转发；环境固定使用 `/media/jushen/mingbo-ge/.venv`，减少不同机器误用系统 Python。
- 增加 shape trace：在 normalized、transition、conditioning、VAE input/output 关键节点打印一次 shape/dtype，可直接核查 `(T,16,D)` action chunk 和 `(T,16,9,28,28)` latent。
- CPU OOM 修复为 VAE microbatch 内转换 `uint8 → float32 → [-1,1]`，不再把整条 episode 的约 7–14 GiB 视频一次性复制成 float32；latent stats 保持原语义未删减。
- 多 GPU 只复制 VAE tokenizer 并分片 microbatch，保持 episode 顺序和单 writer；避免多进程同时改 LeRobot metadata。
- 增加冻结 `--episode-manifest`、输出 resume manifest 和 storage preflight，批量实验可复现 episode 子集与顺序，并在写盘前检查容量。

工程改进和「6D 是否更好学」分开：转换正确是公平对比的前提，不是表示实验的结果。

## 还可以改进

**不要为了对齐采集而去改的**

- latent 槽位、chunk_size=16、`rescale_action` / `rescale_proprio`
- 把转换改回「只写干预帧」再和现有离线集横比

**转换侧**

- 将 `cosmos_data_3drpy` 与 `cosmos_data_6drotation_monitor_20260819` 作为当前 325 条成对实验基线；旧 `dataset_cosmos_back_handle` 的 58 条部分输出只保留作历史调试证据，不与完整集混用
- 若训练/审计需要 `action.latent_chunk`，应对两套完整基线统一重转或统一补列，不能只更新 6D 一侧，否则会破坏公平对比
- 处理已知重复 `0804_004915`：先保留两份证据，生成训练 manifest 时排除旧的单 episode 副本；确认无引用后再做物理清理
- 对 8 个 quarantine 文件逐个审查。其中本次增量新增 2 个：一个相机 timestamp/HDF5 尾部读取异常，一个缺 `trajectory_length` 且多字段长度不一致
- 大量 `language_instruction='test01'` 或批次名与期望英文任务描述不一致，目前是 warning；正式 T5 训练依赖外部 task text/cache，不应把该 HDF5 字段直接当作已审核英文描述
- 失败轨迹与成功共用 stats 已支持；主训练是否并失败见 [RESULTS.md](RESULTS.md)，不是转换 bug
- 单卡整批 vs 多卡分片的 bfloat16 舍入差是预期；严核对齐应用相同每卡 batch 形状
- T5 仍是按任务文本查 cache；`--skip-t5` 适合预检和部分消融，正式训练集不要漏 embedding

**和采集脚本的缝**

- 在线采集仍无 sidecar、无预检、无 resume；若要两入口产出可混训，采集侧需要同一套 stats 文件和同一 crop
- 采集只写干预、转换写全帧：混用时要先统一过滤策略，再比样本数

测试入口：

```bash
PYTHONPATH=.:lerobot/src:../cosmos-policy:../cosmos-policy/cosmos_policy:rl_envs \
  ../.venv/bin/python -m unittest discover -s data_convert_refactored/tests -v
```
