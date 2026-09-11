# Cosmos 离线训练 Tiny 实验复用指南

## 目的

Tiny 实验用于在正式训练前验证完整链路：数据扫描、split、Parquet DataLoader、T5、VAE/tokenizer、模型加载、EDM loss、反向传播、验证、checkpoint 和曲线导出。

它不用于评价最终模型效果，也不写入已完成的 `back_handle_joint_6d` 20K 实验目录。

## 配置

配置文件：

```text
HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml
```

主要参数：

| 项目 | 设置 | 原因 |
|---|---:|---|
| 总 episode | 10 | 缩短索引和调试时间 |
| Train / Validation | 8 / 2 episodes | 同时覆盖训练和验证链路 |
| Train transitions | 9,950 | 可供 sampler 读取，但实际只训练100条 |
| Validation transitions | 3,877 | 正式tiny验证只抽取少量样本 |
| `max_steps` | 100 | 验证多次更新、日志和checkpoint |
| `batch_size_per_rank` | 1 | 与20K实验保持一致，降低显存需求 |
| `gradient_accumulation_steps` | 1 | 每个global step执行一次optimizer更新 |
| checkpoint间隔 | 50 steps | 验证中间和最终checkpoint保存 |
| validation上限 | 每个条件2 batches | 控制tiny实验耗时 |
| DataLoader workers | 0 | 降低CPU并发，便于定位异常 |

训练模式为 `base_joint`：

```text
Policy 50% / World 25% / Value 25% / Inverse Dynamics 0%
```

独立输出：

```text
train_outputs/back_handle_joint_6d_tiny
train_outputs/back_handle_joint_6d_tiny_smoke
```

## 首次准备

从项目根目录执行。`run.sh`固定使用 `/media/jushen/mingbo-ge/.venv/bin/python`，不依赖可能包含旧路径的 `activate` 文件。

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project

CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh env-check
```

检查数据、权重、T5和stats：

```bash
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh preflight
```

首次生成固定8/2 split；已有manifest时只读取，不重新随机划分：

```bash
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh prepare-splits
```

期望输出：

```text
[SPLIT] train=8 validation=2 test=0
```

## 10步 Smoke Test

```bash
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh smoke-single 0
```

Smoke test检查模型能否完成forward、backward和optimizer update。只有它通过后才运行100步tiny训练。

## 100步 Tiny 训练

```bash
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh train 0
```

预计耗时约2～5分钟，实际时间取决于模型加载和GPU状态。

需要后台运行时：

```bash
tmux new-session -s cosmos_tiny_train
```

在tmux中执行训练命令；按 `Ctrl+B`、`D` 退出但保留进程。重新进入：

```bash
tmux attach -t cosmos_tiny_train
```

## 结果检查

```bash
find /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/back_handle_joint_6d_tiny \
  -maxdepth 2 -type f | sort
```

应至少包含：

```text
metrics.jsonl
checkpoints/step_000000050.pt
checkpoints/step_000000100.pt
checkpoints/latest.json
splits/train.json
splits/val.json
```

检查最终训练记录：

```bash
tail -n 5 /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/back_handle_joint_6d_tiny/metrics.jsonl
```

生成训练曲线：

```bash
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh plot \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/back_handle_joint_6d_tiny/metrics.jsonl \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/back_handle_joint_6d_tiny/plots
```

## 中断后恢复

只在训练未达到100 steps时使用：

```bash
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/back_handle_joint_6d_tiny.yaml \
bash HIL-RL/cosmos_offline_train/run.sh resume 0 \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/back_handle_joint_6d_tiny/checkpoints/step_000000050.pt
```

如果checkpoint已经达到配置中的 `max_steps: 100`，resume会立即结束。

## 测试 Inverse Dynamics 混合训练

`base_joint`不训练Inverse Dynamics。若要测试四种objective混合，应复制配置到新文件，并修改：

```yaml
objectives:
  training_mode: joint_with_inverse
```

该模式比例：

```text
Policy 40% / World 20% / Value 20% / Inverse Dynamics 20%
```

同时为新配置更换 `output_dir`、`smoke_output_dir`、`train_manifest` 和 `val_manifest`，避免与base-joint tiny结果混写。

## 重跑原则

不要直接复用已有结果目录启动新的from-base实验。推荐为每次实验使用新后缀，例如：

```text
back_handle_joint_6d_tiny_v2
back_handle_joint_with_inverse_6d_tiny
```

这样保留旧checkpoint、日志、split和配置，结果可追踪、可比较。
