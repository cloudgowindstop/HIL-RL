# 按表下载与 HDF5 重组

从验收 Excel 生成下载计划，把 BOS 上的 raw HDF5 拉下来，扫描 schema，再按结构搬进整理库。

不负责七台机的每日自动补洞，那是旁边的 [`../nightly/`](../nightly/README.md)。本目录有自己的 `raw_data_mapping.json`。

| 文件 | 作用 |
|---|---|
| `download_raw_data.py` | `plan` / `plan-incremental` / `download` / `status` / `inventory` |
| `inventory_bos.py` | 列 BOS 批次根，和本地已有对照（只 ls，不 sync） |
| `organize_hdf5.py` | 扫描 trajectory.hdf5 的 key，按 family/schema 搬家 |
| `summarize_organized_data.py` | 整理库统计 |
| `raw_data_mapping.json` | 工站、中文任务、路径片段、source_overrides |
| `INCREMENTAL_DOWNLOAD.md` | 9 台表增量下载的逐步命令 |
| `HDF5_SCHEMA_KEYS.md` | 已扫到的 schema key 清单 |
| `DOWNLOAD_ERROR_REPORT.md` | 历史路径错误记录 |

Python：`/media/jushen/mingbo-ge/.venv/bin/python`  
`bcecmd`：`/media/linux-bcecmd-0.5.1/bcecmd`  
默认 Excel：`HIL-RL/9台天轶设备.xlsx`

---

## 流程

```text
Excel 出计划  →  download 到暂存
              →  （可选）inventory 对 BOS
              →  organize_hdf5 扫描 / 搬家
              →  summarize 出统计
```

增量下载的完整命令和预期数量见 `INCREMENTAL_DOWNLOAD.md`。下载完成后先对暂存目录做 schema 扫描，不要直接把未校验数据写进 `raw_data_0831download_by_schema`。

常用入口：

```bash
PY=/media/jushen/mingbo-ge/.venv/bin/python
ORG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/data_download_tools/organize

$PY $ORG/download_raw_data.py plan-incremental --excel ... --mapping $ORG/raw_data_mapping.json ...
$PY $ORG/download_raw_data.py download --plan ... --state-dir ... --resume
$PY $ORG/download_raw_data.py inventory --mapping $ORG/raw_data_mapping.json ...
$PY $ORG/organize_hdf5.py scan --input ... --state-dir ... --mapping $ORG/raw_data_mapping.json
$PY $ORG/summarize_organized_data.py ...
```

自测：

```bash
$PY $ORG/download_raw_data.py self-test
$PY $ORG/download_raw_data.py inventory-self-test
$PY $ORG/organize_hdf5.py self-test
$PY $ORG/summarize_organized_data.py self-test
```

---

## mapping 这份读什么

| 字段 | 谁用 |
|---|---|
| `stations` | 中文工站 → `tienyi_7` 等；整理目录用。BOS 上没有「天轶7」这种目录 |
| `tasks` | 中文任务 → 短 id、英文描述 |
| `path_task_patterns` | 批次名片段 → 短 id，计划和盘点用 |
| `source_overrides` | 个别 BOS 路径改名 / 指定任务 |
| `inventory` | 盘点时跳过哪些本地目录后缀 |

和 `nightly/raw_data_mapping.json` 是两份独立文件。夜间下载只认那边的 `path_task_patterns`。

---

## 数据落在哪

| 路径 | 角色 |
|---|---|
| `.../raw_data_0903_incremental` | 按 9 台表增量下的暂存（`new/`、`repaired/`） |
| `.../raw_data_0831download_by_schema` | 按 schema 整理后的库 |
| `.../raw_data_0831download_by_schema/_metadata/` | 计划、账本、schema catalog |

`inventory` 对照本地时还会看 `raw_data_0804download`、`cosmos_raw_data`，只读不写。
