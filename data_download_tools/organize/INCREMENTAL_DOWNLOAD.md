# 9台设备数据增量下载

本流程只下载 `9台天轶设备.xlsx` 中尚未成功下载的 BOS 根目录，不重复下载历史成功批次。

## 存储位置

- 下载暂存：`/media/jushen/project-rl-dataset/raw_data_0903_incremental`
- 修复批次：`raw_data_0903_incremental/repaired/<task_id>/<batch>`
- 新增批次：`raw_data_0903_incremental/new/<task_id>/<batch>`
- 计划与日志：`raw_data_0831download_by_schema/_metadata/download_state/incremental_20260903`

## 1. 生成冻结计划

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project

/media/jushen/mingbo-ge/.venv/bin/python \
  HIL-RL/data_download_tools/organize/download_raw_data.py plan-incremental \
  --excel HIL-RL/9台天轶设备.xlsx \
  --mapping HIL-RL/data_download_tools/organize/raw_data_mapping.json \
  --history-results /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/download_results.jsonl \
  --output /media/jushen/project-rl-dataset/raw_data_0903_incremental \
  --state-dir /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/incremental_20260903 \
  --repair-bos-path BOS::raw_data/tienyi_prod2_dualArm-gripper-3cameras_394/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm
```

预期：97 个候选路径、74 个已下载、23 个增量记录，其中 repaired=1、new=22。

## 2. 人工检查计划

```bash
sed -n '1,240p' \
  /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/incremental_20260903/missing_bos.md
```

`unresolved_metadata.csv` 中的 `unknown_station` 是预期结果：9台表没有可靠工站字段，代码不会根据机器人编号猜测工站。

## 3. 执行下载

```bash
/media/jushen/mingbo-ge/.venv/bin/python \
  HIL-RL/data_download_tools/organize/download_raw_data.py download \
  --plan /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/incremental_20260903/download_plan.jsonl \
  --state-dir /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/incremental_20260903 \
  --resume
```

下载器先写入 `.partial` 目录；只有找到并成功打开至少一个 `trajectory.hdf5` 后，才重命名为正式批次目录并写入完成标记。

## 4. 查看状态

```bash
/media/jushen/mingbo-ge/.venv/bin/python \
  HIL-RL/data_download_tools/organize/download_raw_data.py status \
  --plan /media/jushen/project-rl-dataset/raw_data_0831download_by_schema/_metadata/download_state/incremental_20260903/download_plan.jsonl
```

下载完成后，再对 `raw_data_0903_incremental` 执行 Schema 扫描和追加移动。不要直接把未校验数据写进正式的 `raw_data_0831download_by_schema`。
