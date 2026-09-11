# 天轶 raw 数据工具

这里是两套互不 import 的流程，**各有一份** `raw_data_mapping.json`，改一边不会自动同步到另一边。

| 目录 | 做什么 | 说明 |
|---|---|---|
| [`organize/`](organize/README.md) | 按 Excel 计划下载，分析 HDF5 schema，按结构重新组织 | 读本目录 mapping 的 `tasks` / `stations` / `source_overrides` / `path_task_patterns` |
| [`nightly/`](nightly/README.md) | 七台工厂机每天扫 BOS，补本地没有的批次 | 只读本目录 mapping 的 `path_task_patterns` |

新任务如果两边都要认，需要分别改两份 mapping。
