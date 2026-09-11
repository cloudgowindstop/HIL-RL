# Cosmos 离线训练绘图工具复用指南

## 工具与数据来源

绘图实现位于：

```text
HIL-RL/cosmos_offline_train/plot_metrics.py
```

输入是训练目录中的 `metrics.jsonl`。每行是一条 JSON 指标记录，通常由 rank 0 写入。工具不会读取 checkpoint，也不会重新执行模型推理；它只整理已有记录并生成图表、CSV 和 JSON 摘要。

训练正常结束时，`trainer.py` 会自动调用该工具。训练中断、旧结果补图或修改绘图代码后，可手动重新运行。

## 通用命令

从项目根目录执行：

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project

bash HIL-RL/cosmos_offline_train/run.sh plot \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/<experiment>/metrics.jsonl \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/<experiment>/plots
```

也可直接调用 Python 模块：

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL

/media/jushen/mingbo-ge/.venv/bin/python -m cosmos_offline_train.plot_metrics \
  --metrics /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/<experiment>/metrics.jsonl \
  --output-dir /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/<experiment>/plots
```

省略 `--output-dir` 时，默认写入 `metrics.jsonl` 同级的 `plots/`。

## 本次六卡 Policy 实验示例

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project

bash HIL-RL/cosmos_offline_train/run.sh plot \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/install_handle_euler_14d_policy_learner_eval/metrics.jsonl \
  /media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/install_handle_euler_14d_policy_learner_eval/plots
```

## 输出说明

| 文件 | 内容 | 主要用途 |
|---|---|---|
| `train_eval_edm_loss.png` | 原始 train actor EDM 与 validation actor EDM | 判断整体收敛和训练/验证差距 |
| `objective_edm_loss.png` | Policy、World、Value、Inverse Dynamics 目标 loss | 检查不同训练目标；未采样目标不会绘制 |
| `latent_region_edm_loss.png` | action、future proprio、wrist image、third-person image、value 区域 EDM | 定位哪个 latent 区域导致波动 |
| `validation_metrics.png` | validation EDM 与各区域 L1 | 查看固定验证协议下的 checkpoint 变化 |
| `optimization_health.png` | gradient norm、learning rate、loss scale、吞吐、step time、显存 | 排查优化和性能异常 |
| `checkpoint_comparison.png` | Base 与已有 checkpoint 的 validation 指标柱状对比 | 比较已保存权重；至少需要两个可匹配评测点 |
| `train_metrics.csv` | 训练记录展开为表格 | 自定义分析、Excel/Pandas 读取 |
| `eval_metrics.csv` | 评测记录展开为表格 | 绘制 validation 曲线和 checkpoint 对比 |
| `loss_summary.json` | 各训练指标的点数、首尾均值和最小值 | 快速检查趋势 |
| `validation_records.json` | `validation` 与 `eval` 原始记录汇总 | 保留验证协议和详细结果 |

## 重要行为

1. 所有训练曲线使用原始记录，不做平滑。尖峰不会被隐藏。
2. 相同 `(mode, protocol, global_step)` 重复出现时保留最后一条，避免重启或补评测造成重复点。
3. 图中横轴是 optimizer step，不是 micro-batch，也不是 episode。
4. `objective_edm_loss.png` 按该目标实际采样数过滤。纯 Policy 训练不会伪造 World、Value 或 ID 曲线。
5. `checkpoint_comparison.png` 只使用同时满足以下条件的 step：存在对应 `checkpoints/step_*.pt`，且 `metrics.jsonl` 中存在该 step 的 `eval` 记录。Base 使用 step 0。
6. 每次生成前会删除工具拥有的旧版和新版图片，再按当前日志重建；不会删除 `metrics.jsonl`、checkpoint 或其他文件。
7. `--smooth-window` 只为旧命令兼容而保留，当前不会执行平滑。

## 如何解读 loss

- `loss_actor_learner`：实际用于反向传播的 EDM loss，主训练曲线。
- `sample_*_edm_loss`：按 Policy/World/Value/ID mask 统计的诊断 loss。
- `*_edm_loss`：不同 latent 区域的加权 EDM 误差。
- `*_mse`、`*_l1_loss`：便于解释的辅助指标，不是本训练的反向传播目标。
- train 和 validation 只有在数据 split、objective、mask、noise protocol 一致时才适合直接比较。

单个尖峰不能仅凭图片判断原因。应同时检查 `latent_region_edm_loss.png`、`optimization_health.png` 和对应 step 的 `metrics.jsonl`，区分数据异常、噪声采样、梯度异常和验证协议变化。

## 快速检查

确认输入存在且含训练、评测记录：

```bash
test -f train_outputs/<experiment>/metrics.jsonl
grep -m 1 '"mode": "train"' train_outputs/<experiment>/metrics.jsonl
grep -m 1 '"mode": "eval"' train_outputs/<experiment>/metrics.jsonl
```

确认输出：

```bash
find train_outputs/<experiment>/plots -maxdepth 1 -type f | sort
```

若某张图缺失，通常表示日志中没有对应指标，而不是绘图失败。例如没有 `eval` 记录时，不会生成有效 validation 或 checkpoint 对比曲线。

