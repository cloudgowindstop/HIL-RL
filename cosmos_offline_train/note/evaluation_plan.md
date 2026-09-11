# Cosmos 离线中训练验证方案

## 1. 当前结论的准确含义

当前 `Base vs 20K` 对比只执行了 **Inverse Dynamics（ID）条件方式下的 action 验证**：

- 条件：当前观察 latent `0:4` 与未来观察 latent `5:8`；
- 预测目标：action latent `4`；
- 验证方式：在固定 `sigma={0.1,0.5,0.9}` 下做一次带噪去噪预测；
- 指标：action EDM、MSE、L1，以及反归一化后的双臂 translation MAE、rotation geodesic error、gripper MAE；
- 数据：相同 validation manifest，每个 sigma 取前 200 batches，噪声种子固定。

因此，它不是完整的模型能力验证，也不是闭环策略验证。它还有两个限制：

1. 20K checkpoint 使用 `base_joint` 训练，没有接受显式 ID 训练。当前结果表示 joint 中训练后的模型在 ID mask 下具有较强的 action 恢复能力，不能直接称为“ID 训练收益”。
2. 当前结果来自单步 denoising，不是从纯噪声开始的完整多步 diffusion sampling。它可用于快速 checkpoint 比较，但不能替代最终生成质量评估。

尤其需要增加 `future shuffle` 消融：把 future observation 在 batch 内随机打乱。如果 action 指标几乎不变，说明模型可能只在执行 current-observation policy，没有真正利用未来观察完成 ID。

## 2. 完整验证矩阵

### 2.1 训练与优化健康度

目的：确认训练过程稳定，不证明任务能力。

| 项目 | 指标 | 当前状态 |
|---|---|---|
| 总损失 | total EDM loss | 已有 |
| 训练目标 | policy/world/value/inverse EDM loss | 已有；20K 日志无 inverse |
| latent 区域 | action、future proprio、wrist image、third-person image、value 的 EDM/MSE/L1 | 已有 |
| 优化器 | learning rate、gradient norm、loss scale | 已有 |
| 性能 | step time、samples/s、GPU memory | 已有 |
| 异常检测 | NaN/Inf、梯度爆炸、空 mask | 部分已有 |

验收：loss 总体下降；梯度有限；不同模块没有长期为零、突然发散或被错误 mask。

### 2.2 Policy 验证

条件：当前观察 `0:4`。目标：action `4`，以及 Cosmos Policy 定义的 future/value 辅助目标。

离线指标：

- action latent EDM/MSE/L1；
- 双臂 translation MAE；
- 双臂 rotation geodesic error（度）；
- gripper MAE/二值准确率；
- action horizon 1/4/8/16 分项；
- 与“恒定动作”“训练集均值动作”“上一时刻动作”基线比较。

当前已有 action 连续误差。仍缺 gripper 二值准确率、简单基线和 episode 级统计。

注意：offline action error 只能表示行为克隆拟合程度，不能等价于任务成功率。

### 2.3 Inverse Dynamics 验证

条件：当前观察与真实未来观察。目标：中间 action。

除 Policy action 指标外，必须增加：

- **future shuffle**：打乱 future，验证模型是否真正使用未来信息；
- **current-only 对照**：屏蔽 future，比较完整 ID 条件的增益；
- **future-only 对照**：屏蔽 current，检查信息来源；
- 不同 future horizon 的 ID 误差；
- success/failure episode 分组；
- Base、Joint-20K、专用 ID checkpoint 的相同条件比较。

关键判据：完整 ID 条件优于 current-only；future shuffle 后明显变差。否则不能证明模型学会反向动力学。

### 2.4 Forward Dynamics / World Model 验证

条件：当前观察与真实 action。目标：future proprio、future wrist image、future third-person image，以及辅助 value。

分两级验证。

#### 快速 denoising 验证

- future proprio EDM/MSE/L1；
- future wrist/third-person latent EDM/MSE/L1；
- 固定多个 sigma；
- Base 与各 checkpoint 使用同一 batch、同一 noise。

当前训练代码已能计算这些 latent 指标，但尚未对 Base、5K、10K、15K、20K 做完整的确定性 checkpoint 对比。

#### 完整 rollout 验证

- 使用正式多步 sampler 从噪声生成 future latent；
- 恢复 latent injection 占用的位置，再用 tokenizer/VAE 解码 RGB；
- RGB 指标：PSNR、SSIM、LPIPS；
- latent 指标保留作为诊断；
- 按 wrist/third-person camera、预测 horizon 分开统计；
- 保存预测视频、真实视频和差异图。

PSNR 只适合严格对齐的短期预测。小位移和多模态合理未来会降低 PSNR，因此必须和 SSIM、LPIPS、可视化同时使用。

### 2.5 Value 验证

条件：当前观察、action 与 future。目标：value。

需要先从 latent slot `8` 解码真正的标量 value，再计算：

- value MAE/MSE；
- success/failure AUROC 与 AUPRC；
- Spearman/Kendall 排序相关性；
- calibration curve / Expected Calibration Error；
- 候选轨迹 top-k 排序准确率；
- success、failure 数据分别统计。

当前 `value_edm_loss/value_mse/value_l1` 是 latent 区域误差，尚未形成完整的标量价值评估。

### 2.6 Checkpoint 与样本效率验证

按照 Cosmos 3 的思路，在完全相同的下游设置下比较初始化和 checkpoint：

```text
Base
5K
10K
15K
20K
专用 ID 1K/3K/5K
```

对每个 checkpoint 固定：

- validation manifest；
- 样本顺序；
- noise seed；
- sigma 或 sampler 参数；
- T5、statistics、tokenizer；
- batch 数或完整 episode 集合。

报告每项指标随训练 step 的曲线。这样可同时判断最终效果和样本效率。

### 2.7 泛化与数据切片

需要按 episode 聚合，而不是把同一 episode 的帧当作独立样本：

- task；
- source dataset；
- success/failure；
- camera；
- action horizon；
- action 运动幅度；
- ID/OOD task 或采集条件。

每项报告 mean、median、标准差和 episode bootstrap 95% CI。当前 200 batches 只适合快速验证；最终报告应覆盖完整 validation episode。

### 2.8 闭环验证

不接实机时，最强验证仍是仿真闭环：

- task success rate；
- staged task score；
- 平均完成时间；
- failure 类型；
- ID/OOD 随机化条件；
- 每个 checkpoint 使用相同初始状态和随机种子。

如果当前任务没有数字孪生或 simulator，只能明确报告“离线代理指标”，不能宣称真实策略成功率提高。

## 3. 当前已经具备的能力

| 能力 | 代码位置 |
|---|---|
| policy/world/value/ID condition 与 loss mask | `masks.py` |
| 固定 objective、sigma、noise 的 validation | `trainer.py` |
| action latent 提取与 20D 反归一化 | `model.py` |
| 双臂 translation/6D rotation/gripper 指标 | `model.py`, `rotation_6d.py` |
| 分区域 EDM/MSE/L1 | `losses.py` |
| JSONL、loss 曲线、CSV/JSON 摘要 | `metrics.py`, `plot_metrics.py` |
| Base 与 offline checkpoint 验证入口 | `train.py`, `run.sh` |

## 4. 需要新增或修改的代码

### P0：先得到可信离线结论

#### 4.1 新增独立评估入口

建议新增：

```text
cosmos_offline_train/evaluate.py
cosmos_offline_train/evaluation.py
cosmos_offline_train/compare_evaluations.py
```

职责：

- `evaluate.py`：加载一次数据和模型，执行指定 objective、checkpoint 和消融；
- `evaluation.py`：按 sample/episode 聚合指标，保存逐样本结果；
- `compare_evaluations.py`：生成 Base/5K/10K/15K/20K 对比 CSV、JSON 和曲线。

原因：评估组合会快速增多，不应继续堆入训练主循环。

#### 4.2 修改 `trainer.py`

- 保留当前快速 validation；
- 增加 `condition_ablation`：`normal/current_only/future_only/future_shuffle`；
- validation 类型不再依赖训练模式；
- 输出 sample_id、episode_path、row_index 对应的逐样本指标；
- 支持完整 validation，不只 `max_validation_batches`。

#### 4.3 修改 `masks.py`

- 增加 ID 消融 mask 构造函数；
- 对未使用 latent slot 保持纯噪声，禁止目标泄漏；
- 为每种 objective 增加显式 mask 单元测试。

#### 4.4 修改 `model.py`

- 将 `preencoded_edm_forward` 拆成 condition 构造、加噪、denoise、metric decode；
- 增加 `extract_value()`，从 value latent 恢复标量；
- 增加 gripper 二值指标；
- action 指标返回逐样本/逐 horizon 数值，不只 batch mean；
- 增加正式多步 sampling 包装。

### P1：验证 World Model 生成能力

#### 4.5 新增 `sampling.py`

复用官方：

- `CosmosPolicyVideo2WorldModel.generate_samples_from_batch()`；
- `cosmos_utils.undo_latent_injection()`；
- `cosmos_utils.get_future_images_from_generated_samples()`；
- tokenizer/VAE `decode()`。

需要适配预编码输入、当前 9-slot latent 布局和 20D action injection。不能直接对含 action/proprio/value 注入的 latent 整体 VAE decode；必须先恢复占位图像 latent。

#### 4.6 新增 `video_metrics.py`

- PSNR；
- SSIM；
- LPIPS；
- camera/horizon 分组；
- 保存预测、GT、差异图和元数据。

需要确认 converted Parquet 是否保留可作为 GT 解码的原始 clean image latent。如果只保存注入后的 latent，则需在转换阶段额外保存 clean latent 或原始 RGB 路径。

### P1：完整 Value 验证

#### 4.7 新增 `value_metrics.py`

- 标量 value 解码；
- MAE/MSE；
- AUROC/AUPRC；
- Spearman/Kendall；
- calibration；
- candidate ranking。

同时检查当前数据是否包含足够 failure episode。success-only validation 无法计算分类 AUROC，也无法有效检验 value 排序能力。

### P2：统计与闭环

#### 4.8 修改 `metrics.py`

- episode-level accumulator；
- bootstrap 95% CI；
- 分 task/source/outcome/camera/horizon 报告；
- DDP 下按有效样本数加权，避免空 mask 的零值参与均值。

#### 4.9 新增 simulator adapter

```text
cosmos_offline_train/sim_eval/
```

统一 reset、seed、rollout、success/stage score 和录像接口。此项取决于任务是否存在可用仿真环境。

### P0 测试

修改 `tests/run_cpu_tests.py` 并新增 GPU integration test：

- 四种 objective mask；
- future shuffle 不改变 shape，但改变条件配对；
- inactive latent 不泄漏 GT；
- action/value encode-decode round trip；
- episode 聚合与 bootstrap；
- Base 与 checkpoint 固定 seed 可复现；
- 2-sample 多步 sampler + VAE decode smoke test。

## 5. 推荐实施顺序

1. P0：逐样本/episode 指标、ID condition 消融、checkpoint 批量比较。
2. 用 Base/5K/10K/15K/20K 完成 policy、ID、world、value 快速 denoising 对比。
3. P1：接入完整 sampler、VAE decode、PSNR/SSIM/LPIPS。
4. P1：完成标量 value 与 success/failure 排序验证。
5. 有仿真条件后增加闭环 success rate。

第一阶段完成后，才能回答“20K 中训练是否让模型真正学会动作与动力学”；第二阶段完成后，才能回答“未来视频生成是否改善”；闭环完成后，才能回答“策略任务能力是否改善”。
