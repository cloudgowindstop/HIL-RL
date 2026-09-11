# 工厂天轶夜间下载

扫 BOS 上 **7 台工厂产线机**，把本地还没有的批次下到：

```text
/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/{robot_type}/{task_id}/{batch}/
```

只做下载和记账，不整理 schema，也不往其它数据目录里写。本机没有 crontab，每天 22:00（上海时间）靠 `daily` 常驻进程等到点再跑。

| 文件 | 作用 |
|---|---|
| `start_nightly_download.sh` | 后台启动、看状态、停进程 |
| `nightly_download.py` | 列 BOS、跳过已有、下载、写账本 |
| `raw_data_mapping.json` | 本目录一份；只读其中的 `path_task_patterns` |

Python：`/media/jushen/mingbo-ge/.venv/bin/python`  
`bcecmd`：`/media/linux-bcecmd-0.5.1/bcecmd`

---

## 怎么用

先设一个短名，后面命令都用它：

```bash
SH=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/data_download_tools/nightly/start_nightly_download.sh
```

`sh` 把下载丢到后台。`run` 会先等扫描结束，再把 **pid、要下哪些批次** 打到终端，然后把下载留在后台。

| 命令 | 行为 |
|---|---|
| `$SH` 或 `$SH daily` | 挂着不退出，每天 22:00 扫一轮 |
| `$SH run` | 立刻扫一轮；终端先打出待下清单，下完进程退出 |
| `$SH status` | 看进程、阶段、待下清单、下到哪了 |
| `$SH stop` | 停掉所有 `nightly_download.py` |

总日志：`.../raw_data_0909_autoupdate/_metadata/cron.log`  
实时状态：`.../_metadata/current_run.json`（`$SH status` 读这份）

改 `daily` 的触发时间（仍是上海时区）：

```bash
NIGHTLY_AT=21:30 $SH daily
```

`run` / `daily` 后面可以再跟 Python 参数：

```bash
$SH run --dry-run              # 只列缺口，不真正 sync
$SH run --min-date 20260901    # 只下这个日期及之后
```

同一时刻只能有一个下载进程。`sh` 发现已有 `nightly_download.py` 会拒绝再开；Python 自己还有文件锁。`run` 和 `daily` 不要一起开。

这台机器是容器。重启后 `daily` 不会自动回来，要再执行一次 `$SH`。

不经过 `sh`、自己调 Python 也可以（记得 `PYTHONUNBUFFERED=1`，否则日志会攒很久才刷出来）：

```bash
PYTHONUNBUFFERED=1 /media/jushen/mingbo-ge/.venv/bin/python \
  /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/data_download_tools/nightly/nightly_download.py run
```

常用参数：`--dry-run`、`--min-date`（默认 `20260601`，空字符串表示不按日期滤）、`--output`、`--mapping`。`daily` 还有 `--at HH:MM`。

---

## 怎么判断结束、是否成功

`run` 和 `daily` 不一样：`run` 下完就退出；`daily` 一轮结束后继续活着等下一天。

**进程、待下清单、下到哪了**

```bash
$SH status
```

会打印 `pid`、运行中/已结束、`phase`（scanning / planned / downloading / done / waiting）、本轮要下的列表。`[x]` 已下完，`[>]` 正在下。`daily` 在等 22:00 时 `phase=waiting`，也算进程还在。

**看日志**

```bash
tail -f /media/jushen/project-rl-dataset/raw_data_0909_autoupdate/_metadata/cron.log
```

| 模式 | 一轮正常结束时日志里会出现 |
|---|---|
| `run` | `summary: .../_metadata/runs/YYYY-MM-DD.json`，然后进程退出 |
| `daily` | 同样有 `summary:`，接着一行 `daily run done exit=0`，然后 `next daily run at ...` |

停在 `download <任务>/<批次>` 中间、没有当天的 `summary:`，就是没跑完（崩了或被 `stop`）。

**看当天摘要**

```text
.../raw_data_0909_autoupdate/_metadata/runs/YYYY-MM-DD.json
```

看 `status_counts`：

| 内容 | 含义 |
|---|---|
| 只有 `downloaded` / `skipped_complete` | 这一轮正常（没有新批次时 `status_counts` 为空对象也算正常） |
| 有 `failed` / `failed_verification` | 进程结束了，但有批次没下好 |
| 顶层 `"status": "skipped_disk"` | 可用空间低于 200 GiB，整轮没下 |
| 只有 `dry_run` | 预演，没有真正下载 |

`downloaded`：这一轮新 sync 成功。  
`skipped_complete`：正式目录里已经有完成标记，没再下。

每条批次一行：`_metadata/download_results.jsonl`  
每个批次的 `bcecmd` 输出：`_metadata/logs/{batch}.log`

---

## 下哪些机器，为什么是这七台

设备号：`128, 134, 193, 368, 394, 468, 515`  
BOS / 本地第一层目录：`tienyi_prod2_dualArm-gripper-3cameras_{id}`

来源是验收清单 `HIL-RL/9台天轶设备.xlsx`。文件名写「9台」，表里真正出现的 BOS 设备号只有这 7 个。它们是中试工厂产线机，表里的分工大致是：

| 设备号 | 表里主要任务 |
|---:|---|
| 128 | 线缆连接 |
| 134 | 电机中间插管 |
| 193 | 轴承装配 |
| 368 / 394 / 468 | 背部提手安装 |
| 515 | 背部提手 + 小电池盖板 |

夜间下载跟的是 **BOS 设备号**，不是工站名。`raw_data_mapping.json` 里的「天轶7 / 天轶9 / 天轶30 / 天轶31」是整理 schema 时用的工站别名，桶上没有这个目录，不能按工站去 `ls`。

桶里还有大量其它 `tienyi_*`（测试、无日期、未验收任务）。脚本只盯这七台，避免把整桶拉下来。

路径对照：

| 写法 | 用在哪 |
|---|---|
| `bos:/bd-dp-ten-6spt6-scjd/raw_data/{robot}/{batch}` | `bcecmd` 真正访问的 URI |
| `BOS::raw_data/{robot}/{batch}` | 账本、去重用的内部键 |

批次名里要能读出日期，且 `>= 20260601`。`test`、`1`、`urdf_test` 这类没有 `20YYMMDD` 的目录会跳过。

---

## 任务名怎么变成目录

夜间下载只读 **本目录** `raw_data_mapping.json` 的 `path_task_patterns`：批次名里出现左边这段（大小写不敏感），本地第二层目录就用右边的短 id。`organize/` 里另有一份 mapping（含 `tasks` / `stations`），两边改了不会互相覆盖。

当前 5 条：

| 批次名里的片段 | 短 id（落盘目录） | 中文（`tasks`） |
|---|---|---|
| `plug_in_ethernet_type-c_usb` | `plug_cables` | 5-7线缆连接 |
| `insert_hose_into_motor` | `insert_hose` | 1电机中间插管 |
| `back-handle-installation` | `back_handle` | 15 背部提手安装 |
| `Back_small_battery_installation_and_small_battery_cover_installation` | `back_battery_and_cover` | 13-14背部小电池安装+小电池盖板安装 |
| `assemble_bearings` | `assemble_bearings` | 轴承装配 |

只命中一条才用短 id。对不上（例如 `press-fitting-of-head-plastic-bearing`）会从批次名去掉「机器前缀 + 日期后缀」，剩下那截当目录名；再抠不出来才是 `unknown_task`。

新任务在 `path_task_patterns` 加一行即可：

```json
"press-fitting-of-head-plastic-bearing": "press_fit_head_bearing"
```

---

## 落到哪，哪些目录只读

新数据只写输出根：

```text
/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/
└── tienyi_prod2_dualArm-gripper-3cameras_128/
    └── plug_cables/
        └── tienyi_prod2_dualArm-gripper-3cameras_128_plug_in_ethernet_type-c_usb_20260910_left/
```

账本在输出根的 `_metadata/`：

```text
_metadata/
├── cron.log                 # sh 重定向的总日志
├── current_run.json         # 当前进程 / 待下清单 / 进度
├── nightly.lock             # 互斥锁
├── download_results.jsonl   # 每批次一行
├── runs/YYYY-MM-DD.json     # 当天摘要
└── logs/{batch}.log         # 该批次的 bcecmd 输出
```

下面这些目录 **只用来判断「已经有了，别再下」**，脚本不会往里面写：

| 目录 | 说明 |
|---|---|
| `.../raw_data_0909_autoupdate` | 本输出根，已下过的也会跳过 |
| `.../raw_data_0804download` | 早期扁平下载 |
| `.../raw_data_0831download_by_schema` | 按 schema 整理后的库 |
| `.../cosmos_raw_data` | 中试交付那批；忽略名字以 `_left_gripper_cut` 结尾的加工目录 |

另外会读两份历史 jsonl（0831 主账本和 0903 增量账本），`status` 为 `downloaded` / `skipped_complete` 且至少 1 个可读 hdf5 的批次也算已有。

---

## 一轮在做什么

```text
加锁
  → 列七台机在 BOS 上的批次
  → 扫本地三处历史根 + 本输出根 + 历史 jsonl
  → 丢掉日期 < 20260601 和没日期的
  → 对缺口逐个 sync
  → 追加 jsonl，写当天 summary
```

`run`：上面走一遍就退出。  
`daily`：等到 `--at` → 走一遍 → 再等到第二天同一时刻。等待时最多睡 30 秒醒一次，所以 `stop` 比较及时。

---

## 实现要点

**互斥。** `_metadata/nightly.lock` 上 `fcntl` 排他锁。抢不到就打印 `another nightly_download run is in progress` 并返回 1。`sh` 还会用 `pgrep` 挡一层。

**列 BOS。** 对每台机执行 `bcecmd bos ls --all`，从 `PRE` 行取子目录名。某一台列失败只跳过那一台。

**何谓已有。** 目录名是「机器前缀 + 任务 + 日期」这种批次名，且里面有 `trajectory.hdf5` 或 `.download_complete.json`。机器目录本身也叫 `tienyi_..._{id}`，不会被当成批次，会继续往下走到 `task/batch`。

**下载。** 先 `bcecmd bos sync` 到 `{batch}.partial`。至少能用 h5py 打开 1 个 `trajectory.hdf5`，才写 `.download_complete.json` 并改名为正式目录。个别坏文件（例如 `bad heap free list`）记在 `skipped_bad_hdf5`，整批、整轮不会因此崩掉。一个都读不出来才是 `failed_verification`。默认 `--resume`，中断后接着 sync 同一个 `.partial`。

**磁盘。** 可用空间低于 200 GiB 则写 `skipped_disk`，整轮不下载。

**账本。** 每下一个批次就追加一行 jsonl 并 `fsync`。中途崩掉也不会丢已经成功的记录，下一轮会把它们当成已有。当天 summary 里有任意 `failed*` 则返回码 1（`daily` 仍会继续等下一天）。

---

## 自测

不连 BOS：

```bash
/media/jushen/mingbo-ge/.venv/bin/python \
  /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/data_download_tools/nightly/nightly_download.py self-test
```

按表下载和 schema 整理在 [`../organize/`](../organize/README.md)。
