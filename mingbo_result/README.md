# mingbo_result

Cosmos 离线公平对比（`action_representation_20260903`）的对外整理：训练代码说明、2000-step 2×2 结果、对比图。不含 19G checkpoint，也不含原始 Parquet。

| 文件 | 内容 |
| --- | --- |
| [CODE.md](CODE.md) | 实现了什么、亮点、已优化、还能优化 |
| [RESULTS.md](RESULTS.md) | 结果、分析、展望（优化目标 / 实验方向） |
| `figures/compare/` | 柱状图 / 曲线 fig1–fig11 |
| `figures/rgb_examples/` | 同一帧 World RGB 拼图（GT \| Pred \| Error） |
| `tables/` | csv + `compare_matrix.json` |

实验协议和启动命令仍以  
`HIL-RL/cosmos_offline_train/experiments/action_representation_20260903/README.md`  
为准。训练代码在 `HIL-RL/cosmos_offline_train/`。

**窄结论：** 左臂动作 E-P 更好，未来画面 6D-P 更好；这个预算下 joint 没赢（Value 排序除外）；旧 6D-P@1000 作废；H3 没做。
