# mingbo_result

Cosmos 离线公平对比（`action_representation_20260903`）的对外整理：数据转换、训练代码说明、2000-step 2×2 结果、对比图。不含 19G checkpoint，也不含原始 Parquet。

| 文件 | 内容 |
| --- | --- |
| [PRODUCTION_INTERNSHIP_REPORT.md](PRODUCTION_INTERNSHIP_REPORT.md) | 按课程模板整理的生产实习总结报告正文与精简附录 |
| [DATA_CONVERT.md](DATA_CONVERT.md) | HDF5→Cosmos 转换实现；对照 `collect_data_cosmos.py`；改进点 |
| [CODE.md](CODE.md) | 离线训练：实现了什么、亮点、已优化、还能优化 |
| [RESULTS.md](RESULTS.md) | 结果、分析、展望（优化目标 / 实验方向） |
| `figures/compare/` | 柱状图 / 曲线 fig1–fig11 |
| `figures/rgb_examples/` | 同一帧 World RGB 拼图（GT \| Pred \| Error） |
| `tables/` | csv + `compare_matrix.json` |

数据转换代码在 `HIL-RL/data_convert_refactored/`；在线采集仍是 `HIL-RL/collect_data_cosmos.py`。  
实验协议和启动命令仍以  
`HIL-RL/cosmos_offline_train/experiments/action_representation_20260903/README.md`  
为准。训练代码在 `HIL-RL/cosmos_offline_train/`。

**窄结论：** 左臂动作 E-P 更好，未来画面 6D-P 更好；这个预算下 joint 没赢（Value 排序除外）；旧 6D-P@1000 作废；H3 没做。
