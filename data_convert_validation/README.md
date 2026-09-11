# Cosmos转换结果比对验证

该目录用于比较两个由同一episode生成的Cosmos LeRobot Parquet，并验证低维信号是否正确注入video latent。

## 验证内容

`compare_cosmos_parquet.py` 会检查：

1. Parquet schema和帧数；
2. action、proprio、future_proprio、value、reward、done、timestamp及索引字段；
3. 根据每份Parquet自己的单步action重新构造16步action chunk；
4. 按官方`replace_latent_with_action_chunk()`的重复填充规则，逐元素检查`video[:, :, action_latent_idx, :, :]`；
5. 流式比较9个latent时间位置，避免同时加载两份完整video导致内存过高；
6. 如果candidate的`dataset_statistics.json`包含`actions_min/actions_max`，撤销第二层min/max归一化，再和reference action做闭环比较。

## 使用方法

从`HIL-RL`目录运行：

```bash
/media/jushen/mingbo-ge/.venv/bin/python3 \
  data_convert_validation/compare_cosmos_parquet.py \
  --candidate dataset_cosmos/pick_spoon_puppet_next_frame_cosmos_rotation_6d/data/chunk-000/episode_000000.parquet \
  --reference dataset_cosmos/pick_spoon_cosmos_6d/data/chunk-000/episode_000000.parquet \
  --output data_convert_validation/reports/pick_spoon_comparison.txt \
  --json-output data_convert_validation/reports/pick_spoon_comparison.json
```

脚本会在终端打印摘要，同时可保存TXT和完整JSON。

## Latent注入通过标准

以下结果表示action注入逐元素正确：

```text
candidate: passed=True max_abs=0.0 different=0
reference: passed=True max_abs=0.0 different=0
```

验证不是简单检查latent非零，而是：

```text
Parquet action
-> 末端用最后一帧padding形成action chunk
-> flatten
-> 重复填满(C,H,W)
-> 与action latent帧逐元素比较
```

## 相同episode但action不同

如果新版增加了第二层dataset min/max，而旧版保存第一层机器人控制action，则两份Parquet action及action latent不应直接相同。

应查看：

```text
Action normalization round-trip
allclose_atol_2e-6: True
```

该结果表示撤销candidate第二层归一化后，与reference第一层action一致。

## 图像latent差异

两次独立VAE编码可能因GPU、BFloat16舍入或micro-batch执行路径出现小量数值差异。判断时应区分：

```text
t=1 current proprio
t=4 action chunk
t=5 future proprio
t=8 value
```

和图像位置：

```text
t=2 current wrist
t=3 current primary
t=6 future wrist
t=7 future primary
```

action位置的明显差异可能来自归一化语义变化；图像位置约`1e-4`平均误差、`0.015625/0.03125`最大误差通常属于低精度VAE数值差异。

## 注意事项

- candidate和reference必须使用LeRobot目录结构：`dataset/data/chunk-NNN/episode.parquet`；
- 脚本会从数据集根目录自动读取`cosmos_dataset_metadata.json`和`dataset_statistics.json`；
- 两份数据action维度或video shape不一致时会直接报错；
- 若想证明来自同一episode，应同时检查proprio、timestamp、frame_index和episode长度，而不只比较task名称。
