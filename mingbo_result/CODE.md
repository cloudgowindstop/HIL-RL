# Cosmos 离线训练系统：最终代码说明

本文说明 `HIL-RL/cosmos_offline_train/` 如何在 NVIDIA Cosmos-Predict2-2B 上完成预编码机器人数据的离线训练，以及本次 2×2 对比实验实际运行了什么。实验数字与结论见 [RESULTS.md](RESULTS.md)，HDF5 转换过程见 [DATA_CONVERT.md](DATA_CONVERT.md)。

## 1. 系统定位与技术边界

这套实现复用 Cosmos Policy 的模型、Wan2.1 VAE/tokenizer、T5 conditioning、latent-frame injection、混合噪声分布和优化器/学习率配置，但数据读取、DDP 循环、mask、checkpoint、验证与监控由 `cosmos_offline_train` 组织。

它不是把官方在线 learner 原样搬过来，也不是 Cosmos 3 Generator 的训练栈：

- 基座是 **Cosmos-Predict2-2B-Video2World**，本次四条 run 使用同一文件，SHA256 为 `fbc4f05d...e807a6f0`。文件目前放在名为 `Cosmos-Policy-LIBERO-Predict2-2B/` 的目录中，但哈希和配置确认加载的是 Predict2 Video2World 基础权重，不是本项目已经训练过的 checkpoint。
- 训练目标是 Cosmos Policy/Predict2 的 **EDM 去噪目标**，不是 Cosmos 3 的 rectified-flow velocity objective。
- Cosmos 3 mid-training 的 action loss 额外乘 `10×`；本次代码和四条 2000-step 配置**没有 action 10×**。所有目标位置按统一 EDM 元素权重进入 mask reduction。
- 本次是离线、全参数微调；没有 ReplayBuffer、环境 rollout、在线数据采集或闭环控制。

这一区分很重要：Cosmos 3 的 FD/ID/Policy 思想可用于设计我们的 conditioning mask，但不能把 Cosmos 3 的训练目标和规模直接写成本实验已经实现的内容。

## 2. 一条样本的数据合同

数据转换阶段已执行图像预处理和 VAE 编码。训练读到的主要字段是：

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| `video` | `(16, 9, 28, 28)` | 已注入机器人信息的完整 latent 序列 |
| `action` | `(14,)` 或 `(20,)` | 当前行动作，主要用于检查；模型目标实际位于 `video[:,4]` |
| `proprio` | 数据集状态维度 | 当前本体状态检查 |
| `future_proprio` | 数据集状态维度 | 未来本体状态检查 |
| `value_function_return` | scalar | Value GT 与排序评测 |
| `clean_restore_latent` | `(16,4,28,28)`，可选 | RGB 评测时恢复被 action/proprio/value 覆盖的原始视频槽 |

模型使用的 9 个 temporal latent 位置如下：

| 位置 | 内容 |
| ---: | --- |
| 0 | VAE temporal-compression 所需的 blank |
| 1 | 当前 proprio |
| 2 | 当前 wrist image |
| 3 | 当前 primary image |
| 4 | 16-step action chunk |
| 5 | future proprio |
| 6 | future wrist image |
| 7 | future primary image |
| 8 | future-state value |

`action` 列只是一帧的 D 维控制量；转换器会从当前行开始构造 `(16,D)` action chunk，再把它复制填充进位置 4 的 latent volume。训练直接读取已注入的 `video`，不会在 DataLoader 中重新注入 action。

## 3. 数据划分和加载

### 3.1 冻结划分

本次实验先冻结原始 episode 清单，再分别转换 Euler 和 rotation-6D，避免两个表示看到不同轨迹：

- 总计 221 个成功 episode；排除损坏的 `0804_014423`。
- task 内、完整 episode 粒度、seed 42，按 80/20 划分。
- train：177 episodes / 274,870 rows。
- validation：44 episodes / 67,340 rows。
- test：未设置。
- Euler/6D manifest 的文件哈希不同，但 `source_id`、episode 数和 row 数经过 `tools/audit_paired_datasets.py` 配对审计。

`train.json`、`val.json` 只记录划分和 Parquet 位置，不移动或修改原始数据。checkpoint 还保存 manifest、dataset、stats、T5、tokenizer 和基础权重哈希，加载时不一致会拒绝继续。

### 3.2 Lazy DataLoader + DDP

`CosmosParquetDataset` 先建立 episode/row 索引，`__getitem__` 被调用时才读取对应 Parquet row group，而不是由 rank 0 把完整数据集装进 ReplayBuffer：

```text
manifest
  -> 每个 rank 创建相同的 map-style Dataset
  -> DistributedSampler 为各 rank 分配不同 row index
  -> 每个 rank 的 4 个 worker 独立读取 Parquet row group
  -> pin-memory CPU batch
  -> non_blocking copy 到该 rank 对应 GPU
```

当前实现使用 row-group LRU，只读取训练需要的列。它降低了单次 IO 和 CPU 内存峰值；不过四条 2000-step 主实验是在这项优化合入前运行的，因此该优化不能用于解释模型效果差异。

## 4. Conditioning mask 与训练模式

`masks.py` 用 conditioning 位置决定同一网络在做哪种任务；没有额外的 Policy/World/Value head。

| 模式 | 给模型的干净条件 | 计算 loss 的目标 | 含义 |
| --- | --- | --- | --- |
| Policy | 0–3 | 4–8 | `p(a,s',V|s)` |
| World / FD | 0–4 | 5–8 | `p(s',V|s,a)` |
| Value | 0–7 | 8 | `p(V|s,a,s')` |
| Inverse Dynamics | 0–3、5–7 | 4 | `p(a|s,s')`；位置 8 不参与 |

配置支持：

- `policy_only`：概率 `(1,0,0,0)`；注意它仍预测 action、future state 和 value。
- `base_joint`：每个 sample 独立抽 `(0.50,0.25,0.25,0)`。
- `inverse_dynamics_only`：概率 `(0,0,0,1)`。
- `joint_with_inverse`：概率 `(0.40,0.20,0.20,0.20)`。
- `custom_mix`：由 YAML 显式设置四种概率。

这里的“每个 sample 独立抽取”意味着一个更新 batch 内可以同时包含多种任务。它不是每隔几步轮换任务，也不是将四个独立 loss 相加后各反向一次；所有样本经各自 mask 得到一个标量 loss，只执行一次反向传播。

ID 代码还支持 `current_only`、`future_only`、`future_shuffle` 消融。本次 ID 训练仍在进行，结果不进入 [RESULTS.md](RESULTS.md) 的完成实验结论。

## 5. 实际训练 loss

给定干净 latent `x0`、噪声 `epsilon` 和噪声强度 `sigma`，模型预测干净 latent `x_hat0`。代码先计算逐元素误差：

```math
e_{bcthw}=(\hat{x}_{0,bcthw}-x_{0,bcthw})^2
```

再乘 Cosmos EDM 的噪声权重 `w(sigma)` 和 temporal target mask `M`：

```math
L_{EDM}=\frac{\sum M_{bt}\,w(\sigma_{bt})\,e_{bcthw}}
{\sum M_{bt}\text{（扩展到所有 }c,h,w\text{）}}
```

四条 2000-step run 没有显式配置 `loss_reduction`，因此使用默认 `masked_mean`。`total_edm_loss` 是真正参与反向传播的标量。

训练时 `sigma` 由官方模型的 `HybridEDMSDE` 随机采样：70% 来自 log-normal，30% 来自 `[1,85]` uniform，从而增加高噪声训练。固定 `sigma=0.5` 只用于可复现验证；在 EDM 的实际尺度上它属于较低噪声的一步去噪诊断，不能代表从 `sigma≈80` 开始的完整推理。

日志中的以下量是诊断指标，不是额外相加的训练 loss：

- `sample_policy/world/value/inverse_dynamics_edm_loss`：按 sample objective 分组。
- `action/future_proprio/future_wrist_image/future_image/value_edm_loss`：按 latent 位置分组。
- 对应的 `*_mse`、`*_l1`：不带 EDM sigma 权重的误差。
- `gradient_norm`、learning rate、吞吐、GPU memory：优化稳定性。

由于图像 latent 包含大量元素，不同区域 L1/MSE 的量级不能直接解释成任务重要性。本次没有采用 Cosmos 3 的 modality-specific loss scale，也没有额外 action `10×`。

## 6. 一次 DDP 更新如何发生

本次四条主实验统一使用：

```text
4 ranks × 2 samples/rank/micro-batch × 16 accumulation
= 128 samples/optimizer step
```

每个 rank 持有一份完整模型副本，`DDP` 只包装 `model.net`。前 15 个 micro-batch 使用 `no_sync()` 累积本地梯度，第 16 个 micro-batch 触发 NCCL 梯度 all-reduce，然后依次执行 gradient clipping、optimizer step、scheduler step 和清梯度。因此：

- 一个 `global_step` 就是一次 optimizer update。
- 2000 steps 共处理 256,000 个训练 row-samples。
- 相对于 274,870 个 train rows，约为 0.93 个数据遍历；这里的 epoch 只是 sampler 重洗和遍历边界，实际停止条件是 `max_steps`。
- 使用 BF16 autocast；BF16 不启用 FP16 GradScaler。
- 优化器和 scheduler 通过官方 `model.init_optimizer_scheduler` 创建，基础学习率为 `1e-4`，使用官方 warm-up/decay 配置。

模型执行 full fine-tuning，没有冻结视觉层或单独添加 action head。

## 7. 验证协议必须区分三类

### 7.1 训练期间固定 probe

每 500 optimizer steps 验证一次：固定 `sigma=0.5`、固定 noise seed，只强制 Policy 条件，最多 40 batches/rank。4 卡、batch size 2 时约覆盖 320 个 validation row-samples。

它不是完整 67,340-row validation split。“full-suite validation”中的 full 指 Policy/World/Value 条件都运行，不代表全量验证数据。

### 7.2 step 2000 多目标 fixed suite

训练结束时在同一 capped probe 上分别强制 Policy、World、Value 条件，得到：

- 固定噪声的一步 EDM/MSE/L1；
- action chunk 反归一化后的控制尺度误差；
- Value scalar MAE。

用于主表的动作和 latent 数值来自这里。单独的 Value Spearman 复评使用 1 GPU、40 batches、batch size 2，即 `n=80`。

### 7.3 World RGB 多步生成

固定 validation 中 8 episodes × 2 rows，共 16 个样本。World 条件给定当前观测和 **GT action**，使用相同 noise seed、10-step sampler、guidance 1.0 生成未来 latent，再借助 `clean_restore_latent` 执行 VAE decode，计算未来 primary/wrist RGB 的 PSNR、SSIM，并输出 GT | Pred | Error 图和视频。

它衡量 `p(s'|s,a_gt)`，不包含“先预测 action 再预测未来”的策略误差。

## 8. 动作指标的尺度

评测首先撤销 `dataset_statistics.json` 的 min-max 映射，再计算左右臂、不同 horizon 的误差。此时结果回到了**第一层控制量尺度**：

- translation 在转换时是 `clip(delta_xyz / 0.02, -1, 1)`；当前评测没有再乘 `0.02 m`，所以表中的 translation MAE 不是米。换算成米需要乘 `0.02`，且被 clip 的样本无法恢复裁剪前位移。
- Euler 在测地角计算前乘回 `rotation_scale=0.06 rad`。
- rotation-6D 投影回 SO(3)，使用测地角误差。
- gripper 以 stats 中点作为二值阈值，报告 MAE 和 accuracy。

因此 Euler 14D 和 6D 20D 不应比较 raw vector MSE，应该比较同语义的平移、SO(3) 测地角和夹爪指标。当前若出现 `0.000°`，应理解为数值精度下不可分辨，不应写成完美旋转预测。

## 9. Checkpoint、日志与监控

每 500 steps 保存 checkpoint。单文件包含 model、optimizer、scheduler、scaler、step/epoch、RNG 和数据身份，约 19 GB，可精确恢复训练状态；但当前 resume 仍需从 epoch 开头逐 batch 跳过，尚不是 O(1) seek。

rank 0 同时写：

- `metrics.jsonl`：训练和验证原始指标。
- `resolved_config.yaml`：本次运行的最终配置与身份哈希。
- `plots/`：训练后曲线和导出表。
- WandB：远程训练曲线。
- Prometheus `/metrics`：百舸/CProm/Grafana 实时采集。

其他 rank 参与 DDP 计算和 metric reduction，但不重复写日志或 checkpoint。

## 10. 入口、测试与复现

主入口是 `run.sh`，支持 `preflight`、`prepare-splits`、`smoke-single`、`smoke-ddp`、`train`、`resume`、`validate`、`visualize`、`evaluate` 和 `summarize`。

四卡训练示例：

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project
source /media/jushen/mingbo-ge/.venv/bin/activate

CONFIG=HIL-RL/cosmos_offline_train/configs/compare/euler_14d_policy.yaml \
  bash HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3
```

正式训练前依次执行：配置/数据 preflight、CPU 单元测试、固定 batch overfit、单卡 smoke、DDP smoke。`tests/run_cpu_tests.py` 覆盖 split、manifest、mask、loss、action codec、checkpoint identity、评测统计等不依赖真实 2B GPU 前向的逻辑。

## 11. 已解决的问题与剩余代码债

| 已解决 | 作用 |
| --- | --- |
| 冻结 episode manifest + paired audit | 保证 14D/20D 公平对比 |
| Lazy row-group Parquet 读取 | 避免 rank 0 ReplayBuffer 和整 episode 常驻内存 |
| DDP + accumulation | 多卡独立读数、同步梯度，有效 batch 可控 |
| 固定噪声验证 + 独立 RGB 协议 | checkpoint 间可复现比较 |
| 数据身份哈希 | 防止 T5/stats/split/base 权重错配 |
| Prometheus/WandB/JSONL | 本地与集群实时监控 |
| Euler/6D action spec | 同一训练栈支持两种动作表示 |

仍需处理：

1. Resume 保存 sampler cursor，消除从 epoch 开头重扫。
2. 另存 model-only checkpoint，降低评测和交付成本。
3. 将 YAML 绝对路径环境变量化。
4. 合并根模块与早期 `pipeline/components` 兼容层，减少双重入口的阅读成本。
5. 将第一层 translation scale 也纳入指标反变换，明确输出米制 MAE。
6. 若目标是复现 Cosmos 3 mid-training，需要另行实现 rectified flow、per-modality sigma/time sampling、action `10×` 和相应训练数据混合；不能直接把当前 EDM 实验更名为 Cosmos 3。

## 12. 本次实验与当前代码的时间边界

四条 2000-step 实验结果由各 run 自带的 `resolved_config.yaml` 和 `metrics.jsonl` 固定。后来加入的 row-group IO、Value Spearman 汇总、RGB 工具、Prometheus/WandB 封装等属于当前代码能力；其中只有在原始输出中确有记录的部分才进入结果结论。ID 能力已经实现，但对应训练尚未完成，因此不进入完成实验矩阵。
