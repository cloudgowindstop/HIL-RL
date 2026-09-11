# 原始机器人数据下载与 HDF5 整理报告

> 更新时间：2026-09-11  
> 代码目录：`HIL-RL/data_download_tools/`  
> 正式整理库：`/media/jushen/project-rl-dataset/raw_data_0831download_by_schema`

## 1. 工作目标与结论

`data_download_tools` 已形成两条互相独立的数据链路：

1. **按验收表下载并整理**：从 Excel 生成冻结下载计划，下载 BOS 数据，识别 HDF5 结构，按结构安全搬运并生成统计报告。
2. **工厂数据夜间增量下载**：每天扫描七台指定设备的 BOS 目录，只补本地不存在的新批次，并持续写入下载账本。

这套工具解决的不只是“把文件下载下来”，还覆盖了来源追踪、断点续传、结构分类、异常隔离、事务式搬运、结果核验和增量去重。

截至当前，正式整理库中有 **11,015 个物理 `trajectory.hdf5`**、**13 个 Schema ID**，正常结构分布为：

| 结构族 | HDF5 数量 | 含义 |
|---|---:|---|
| `A_head_pose_force` | 5,885 | Head 图像、Puppet 末端位姿、力传感器 |
| `B_head_pose_no_force` | 326 | Head 图像、Puppet 末端位姿、无力传感器 |
| `C_top_joint_only` | 4,557 | Top 图像、仅关节角 |
| `D_head_joint_only` | 239 | Head 图像、仅关节角 |
| `_quarantine` | 8 | 缺图像或存在严重结构/长度异常，隔离待查 |

已知有一组物理重复数据 `0804_004915`，两份文件 SHA-256 相同，因此当前已知去重后的文件数至少为 **11,014**。该数字不是全库重复扫描结论，只表示已经确认的一组重复。

## 2. 主要亮点、创新与优化

### 2.1 从“下载脚本”升级为可审计的数据工程流水线

普通下载脚本通常只负责调用 `bos sync`，无法回答数据从哪里来、是否完整、为何被分类到当前目录。本项目把流程扩展为：

```text
来源解析 → 冻结计划 → 安全下载 → 完整盘点 → Schema识别
         → 质量分级 → 事务搬运 → 一致性验证 → 统计报告
```

每个阶段都有 JSON/JSONL/CSV/Markdown 记录，因此任意 HDF5 都可以追溯到原始 Excel 行、BOS 路径、下载结果、Schema 和最终目录。

### 2.2 数据驱动的 HDF5 Schema 自动发现

没有预先假设数据只有固定几种格式，而是扫描所有 HDF5 的 Group、Dataset、shape、dtype、压缩和属性，自动生成稳定的 Schema 指纹。其改进点包括：

- 排除轨迹长度 `T`，避免不同 episode 长度被误判为不同结构；
- 同时提供逻辑 Schema 和存储 Schema，区分“模型输入差异”和“文件存储差异”；
- 使用稳定哈希作为机器标识，并增加 A～E 结构族作为人类可读分类；
- 自动生成 Schema Key 对比文件，能快速看出缺少末端位姿、力传感器或相机等关键差异。

这让后续 Cosmos Policy 转换可以按实际字段能力选数据，而不是为每批数据手工猜测格式。

### 2.3 分级质量管理，避免简单的成功/失败二分

早期统一使用 `validation_failed`，容易把轻微警告和真正不可用的数据混为一谈。现在改为：

```text
passed / warning / review / quarantine / unreadable
```

可用但存在非阻断差异的数据继续保留；缺图像、严重长度异常等数据进入 `_quarantine`。这种设计同时提高数据利用率和训练数据安全性。

### 2.4 可恢复、不可静默覆盖的事务式搬运

结构重组不是直接执行批量 `mv`，而是先生成计划，再通过 journal 记录 `moving → moved → verified`。它具有：

- 目标冲突拒绝覆盖；
- 中断后 `recover`；
- 操作撤销 `rollback`；
- 搬运后逐项验证；
- 同文件系统原子 rename，避免复制数 TiB 数据产生额外空间和时间消耗。

这是大规模数据整理中很关键的可靠性优化。

### 2.5 冻结计划、幂等下载和断点续传

下载任务先固化为计划，执行期间不再受 Excel 后续修改影响。每批数据采用：

```text
.partial 暂存 → HDF5可读性检查 → 原子改名 → 完成标记
```

配合逐条 JSONL、`fsync` 和 `--resume`，重复运行不会重复下载已完成批次，异常终止也不会把半成品当成成功数据。

### 2.6 跨来源精确增量去重

增量流程不靠相似文件夹名猜测是否下载过，而是使用规范化 BOS 根路径，与历史下载结果进行精确匹配。夜间流程还会联合扫描多个历史数据根和结果账本，从而做到：

- 已存在的数据不重复下载；
- 新增数据自动发现；
- 旧目录保持只读；
- 衍生目录不会被误认作新的原始数据。

在九台设备增量任务中，该机制从 97 个候选路径中准确识别出 74 个历史完成项和 23 个缺口。

### 2.7 异常元数据采用显式修复，不隐式猜测

对于 Excel 多路径单元格、任务名向下继承、错误 BOS 路径等问题，代码使用 `source_overrides` 显式记录修复规则。对于无法确认的工站或任务，保留 `unknown_station` / `unknown_task`，不会根据机器人编号自行填充。

这项设计保证整理结果中的元数据是可解释的事实，而不是难以追踪的程序猜测。

### 2.8 面向持续运行的夜间补数机制

除一次性整理外，还实现了七台产线设备的持续扫描：

- 按日期过滤无效或过旧目录；
- 文件锁防止重复进程；
- 低磁盘空间自动停止；
- 每批独立日志和每日摘要；
- 状态查询与任务停止接口；
- 新批次自动补齐，不修改历史库。

因此该工具既能完成历史数据治理，也能服务后续持续增长的数据源。

### 2.9 工程效果

上述优化已经在真实规模数据上完成验证：

- 管理 **11,015 个**物理 HDF5；
- 识别 **13种** Schema；
- 完成 **3,068 个**增量 HDF5 下载，约 **1.53 TiB**；
- 307 条增量搬运记录全部达到 `verified`；
- 5组核心自测全部通过；
- 异常数据被独立隔离，不混入正常结构目录。

## 3. 代码结构

```text
data_download_tools/
├── README.md
├── organize/
│   ├── download_raw_data.py
│   ├── inventory_bos.py
│   ├── organize_hdf5.py
│   ├── summarize_organized_data.py
│   ├── raw_data_mapping.json
│   └── 若干流程与问题说明文档
└── nightly/
    ├── nightly_download.py
    ├── start_nightly_download.sh
    ├── raw_data_mapping.json
    └── README.md
```

两套流程**不互相 import**，并各自维护一份 `raw_data_mapping.json`。如果新增任务同时需要被整理流程和夜间流程识别，必须同步修改两份映射，否则会产生任务目录命名不一致。

## 4. 按表下载与整理流程

完整调用链如下：

```text
Excel 验收表
  → 解析工站、任务和 BOS 路径
  → 生成冻结下载计划
  → 下载到 .partial 临时目录
  → 校验至少一个 trajectory.hdf5 可读取
  → 原子改名为正式暂存目录
  → 扫描全部 HDF5，生成 Schema 指纹
  → 生成搬运计划
  → 按 quality/family/schema/station/task 搬运
  → 校验目标文件
  → 生成数据集统计
```

### 4.1 `download_raw_data.py`

这是按表下载的主入口，支持：

- `plan`：根据四阶段流水线表生成下载计划。
- `plan-incremental`：根据九台设备表，与历史成功账本进行 BOS 根路径精确比较，只生成缺失项。
- `download`：按冻结计划逐批调用 `bcecmd bos sync`。
- `status`：查看计划中每个批次的下载状态。
- `inventory`：调用盘点逻辑，对照 Excel、BOS 与本地数据。
- `self-test` / `inventory-self-test`：运行无需联网的最小自测。

下载安全机制：

- 先写入 `.partial`，防止半成品伪装为正式目录。
- 下载后至少找到一个可被 HDF5 库打开的 `trajectory.hdf5`，才标记批次成功。
- 完成后以原子改名落盘，并写完成标记。
- 结果逐条追加到 JSONL，并执行 `fsync`，进程中断后可以通过 `--resume` 继续。
- 下载计划冻结后不随 Excel 内容变化，保证一次任务可重放、可审计。

需要注意：下载阶段的成功标准是“批次内至少一个 HDF5 可读”，不是“批次内每个 HDF5 都完全合格”。逐文件完整检查由后续 Schema 扫描承担。

### 4.2 `inventory_bos.py`

该工具只执行 BOS 列举和本地对照，不下载数据。它用于回答：

- Excel 中有哪些路径没有下载；
- BOS 中有哪些批次未出现在 Excel；
- Excel 路径是否确实存在于 BOS；
- 本地成功账本是否存在无法映射回来源的数据。

它通过 `BOS::raw_data/...` 形式的规范来源键去重，而不是根据目录名称模糊判断。

### 4.3 增量补下载成果

九台设备表增量处理结果：

- Excel 候选 BOS 根：**97**；
- 历史已成功下载：**74**；
- 需要补充：**23**，其中修复旧批次 **1**、新增批次 **22**；
- 新下载 HDF5：**3,068**；
- 下载大小：**1,683,000,262,708 bytes**，约 **1.53 TiB**；
- 23 个下载项均已完成，3,068 个 HDF5 均可读取；
- Schema 扫描后可用文件 3,066 个，隔离文件 2 个；
- 生成 307 条搬运记录，全部最终处于 `verified` 状态。

搬运记录多于下载根目录数量，是因为包含混合结构的批次需要拆到不同 Schema 目录。

## 5. HDF5 Schema 分析与分类

### 5.1 `organize_hdf5.py`

该文件负责逐个读取 `trajectory.hdf5`，记录：

- Group 与 Dataset 完整路径；
- `shape` 与 `dtype`；
- compression 与 chunks；
- HDF5 attributes；
- 可推断的轨迹长度、相机类型、机器人字段和质量问题。

它提供 `scan`、`list-schemas`、`show-schema`、`compare-schema`、`plan-move`、`move`、`verify`、`export-schema-metadata`、`recover`、`rollback` 和 `self-test`。

### 5.2 Schema 指纹

工具把结构信息规范化后计算哈希，得到稳定的 `schema_<hash>`：

- **逻辑 Schema**：关注 Key、除时间轴外的 shape、dtype 等会影响数据读取和转换的结构差异；轨迹长度 `T` 不参与分类，避免每种长度形成一个 Schema。
- **存储 Schema**：额外考虑 compression、chunks 等物理存储属性，用于排查同一逻辑数据的存储差异。

Schema ID 适合机器稳定识别，A～E 的结构族适合人快速理解。正式路径因此采用两层组织：

```text
结构族/schema_id/工站/任务/批次/episode/.../trajectory.hdf5
```

### 5.3 质量分类

扫描不会再把所有非完美数据笼统记为 `validation_failed`，而是分为：

- `passed`：没有发现问题；
- `warning`：存在非阻断性差异，通常仍可使用；
- `review`：需要人工确认；
- `quarantine`：缺关键图像、长度严重冲突等，不应直接进入转换；
- `unreadable`：HDF5 无法打开。

该分级避免把“可用但有差异”和“不可用数据”混在一起。

### 5.4 事务式搬运

`plan-move` 先生成 JSONL 搬运计划，不立刻改动数据。`move` 通过同一文件系统上的 `os.rename` 实现快速原子搬运，并记录：

```text
moving → moved → verified
```

若进程中断，可使用 `recover` 根据 journal 恢复；需要撤销时可使用 `rollback`。目标冲突会被拒绝，不会静默覆盖已有数据。由于依赖原子 rename，源目录和目标目录必须位于同一文件系统。

## 6. 数据统计

### 6.1 `summarize_organized_data.py`

统计工具联合 Schema manifest、catalog 和 move plan，将原始来源与当前物理路径对应起来，并输出：

- `summary.json`：总量和一致性摘要；
- `file_inventory.jsonl`：逐 HDF5 记录；
- `report.md`：人类可读报告；
- `by_family.csv`、`by_schema.csv`、`by_station.csv`、`by_task.csv` 等分类表；
- `schema_comparison.md/csv`：各 Schema Key 差异；
- `statistics_errors.jsonl`：统计阶段发现的问题。

统计过程不加载图像内容，只读取元数据、文件大小和轨迹长度，避免占用大量内存。

### 6.2 正式数据集与元数据位置

```text
/media/jushen/project-rl-dataset/raw_data_0831download_by_schema/
├── A_head_pose_force/
├── B_head_pose_no_force/
├── C_top_joint_only/
├── D_head_joint_only/
├── _quarantine/
└── _metadata/
    ├── download_state/   # 下载计划、结果、Excel/BOS 对照记录
    ├── schema_state/     # manifest、catalog、Schema Key 与扫描报告
    ├── move_state/       # 搬运计划、journal、校验记录
    └── statistics/       # 总结报告和各维度统计表
```

历史统计报告记录的首批数据为：7,947 个 HDF5、约 5.12 TiB、8,311,214 个已知帧、11 个逻辑 Schema。后续增量数据搬入后，物理文件数已经更新为 11,015；因此引用旧 `summary.json` 时必须注意生成时间，不能把它当作当前全量统计。

## 7. 夜间自动增量流程

### 7.1 `nightly_download.py`

夜间流程监控 BOS 中七个实际设备号：

```text
128, 134, 193, 368, 394, 468, 515
```

每轮流程：

```text
获取文件锁
  → 列出七台设备的 BOS 批次
  → 扫描历史数据根和下载账本
  → 跳过旧日期及已有批次
  → 逐批下载缺口
  → 校验 HDF5
  → 写逐批结果和当日摘要
```

默认只处理能解析出日期且日期不早于 `20260601` 的批次；`test`、`urdf_test` 等无日期目录不会被误下载。新数据只写入：

```text
/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/
└── {robot_type}/{task_id}/{batch}/
```

历史目录仅用于判断“是否已经存在”，不会被夜间脚本修改。脚本还会忽略 `_left_gripper_cut` 等加工后目录，防止把衍生数据误认为独立原始批次。

### 7.2 `start_nightly_download.sh`

Shell 脚本提供：

- `daily`：常驻等待，到指定时间运行；
- `run`：立即运行一轮；
- `status`：读取进程和本轮状态；
- `stop`：停止夜间下载进程。

它显式使用 `/media/jushen/mingbo-ge/.venv/bin/python`，并将标准输出写入 `cron.log`。Python 层还有 `fcntl` 文件锁，避免两个下载进程同时写同一批数据。

元数据位置：

```text
raw_data_0909_autoupdate/_metadata/
├── cron.log
├── current_run.json
├── nightly.lock
├── download_results.jsonl
├── runs/YYYY-MM-DD.json
└── logs/{batch}.log
```

### 7.3 当前运行状态

2026-09-11 的扫描结果为：

- BOS 批次：423；
- 本地已存在：384；
- 因日期过旧跳过：88；
- 当前待下载：0。

但 `status` 同时显示记录的 PID `704183` **已结束**、状态文件仍为 `phase=waiting`。这表示 `current_run.json` 留下了等待状态，但 daily 常驻进程实际上已经不在运行。若需要继续每日扫描，应重新执行：

```bash
bash HIL-RL/data_download_tools/nightly/start_nightly_download.sh daily
```

容器重启后常驻进程不会自动恢复，这不是系统级定时任务。

## 8. 映射规则

`raw_data_mapping.json` 主要包含：

- `stations`：中文工站名到规范工站 ID；
- `tasks`：中文任务名到英文短 ID 和英文描述；
- `path_task_patterns`：根据 BOS 批次名识别任务；
- `source_overrides`：对 Excel 异常单元格或特殊 BOS 路径做显式覆盖；
- `inventory`：盘点时应忽略的衍生目录规则。

代码不会用机器人编号猜测缺失工站。无法可靠解析时保留 `unknown_station` 或 `unknown_task`，避免把猜测写成事实。Excel 中一个单元格存在多个 BOS 路径等异常情况，通过显式 override 处理，而不是把“第几行”硬编码进程序。

## 9. 自测结果

2026-09-11 实际运行以下测试，全部通过：

```text
download_raw_data self-test: PASS
inventory_bos self-test: PASS
organize_hdf5 self-test: PASS
summarize_organized_data self-test: PASS
nightly_download self-test: PASS
```

这些测试覆盖了计划生成、BOS/本地盘点、Schema 扫描、搬运/校验/回滚、统计生成，以及夜间缺口识别和模拟下载。

## 10. 当前风险与改进建议

1. **磁盘空间风险很高**：2026-09-11 数据盘为 803T，总使用 793T，可用约 10T，使用率 99%。夜间程序虽有 200 GiB 的最低空间保护，但 10T 对多批大规模原始视频数据仍不宽裕，应持续监控并制定清理/扩容方案。
2. **夜间常驻进程已退出**：状态文件与真实 PID 不一致。建议重启 daily，并让 `status` 在 PID 不存在时明确标记 `stale_state`，避免把旧的 `waiting` 误认为正在守候。
3. **两份任务映射可能漂移**：`organize` 和 `nightly` 各有 mapping。建议增加一致性检查测试，但暂不强行合并，因为两条流程使用的字段范围不同。
4. **全量统计需要刷新**：当前正式库已从 7,947 增至 11,015 个 HDF5，而 `_metadata/statistics/summary.json` 仍是首批数据快照。应使用当前 manifest/move plan 重新生成一次全量统计，或者明确按批次保存带日期的快照。
5. **重复数据仍需全库审计**：目前只确认了一组相同哈希重复文件。若训练集要求严格去重，应单独执行按文件大小预分组、再计算哈希的全库重复检查。
6. **下载验证与数据可用性是两层标准**：下载成功只说明批次至少有一个可读 HDF5；是否满足 Cosmos Policy 的相机、action、proprio 字段要求，仍应以 Schema/质量扫描和转换预检为准。

## 11. 常用入口

查看整理流程：

```bash
sed -n '1,240p' HIL-RL/data_download_tools/organize/README.md
```

查看夜间流程状态：

```bash
bash HIL-RL/data_download_tools/nightly/start_nightly_download.sh status
```

查看正式库统计：

```bash
sed -n '1,240p' \
  /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/statistics/report.md
```

查看 Schema 差异：

```bash
sed -n '1,260p' \
  /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/statistics/schema_comparison.md
```

总体而言，当前工具已经具备可恢复、可审计的原始数据工程链路。下一步最重要的不是继续增加下载功能，而是刷新全量统计、解决夜间守护状态一致性，并在大规模转换前按 Schema 和质量等级选择真正满足 Cosmos Policy 输入要求的数据。
