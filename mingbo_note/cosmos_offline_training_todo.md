# HIL-RL 离线 Cosmos 训练待修改清单

更新时间：2026-08-11

目标：使用约 6000 个 episode 完成可复现、可验证、支持 DDP 的离线 Cosmos Policy 训练。

## 2026-08-12：新训练架构承接关系

本清单由独立目录 `HIL-RL/cosmos_offline_train/` 承接。旧
`learner_copy_dist.py` 保留给在线/混合流程，不再作为正式离线训练入口。

| 本清单内容 | 新模块 | 处理阶段 |
|---|---|---|
| Lazy Dataset、episode 索引 | `dataset.py` | 阶段 0 |
| train/val/test、DistributedSampler、workers | `dataloader.py` | 阶段 0、4 |
| 单一配置、metadata/action schema 校验 | `config.py` | 阶段 0 |
| Predict2 模型及显式预训练权重加载 | `model.py` | 阶段 1 |
| policy/world/value/failure/padding mask | `masks.py` | 阶段 1 后逐步上线 |
| EDM 及各 latent 区域 loss | `losses.py` | 阶段 1、3 |
| rotation-6D 投影和几何指标 | `rotation_6d.py`、`metrics.py` | 阶段 2 |
| train/validation、梯度累积、日志 | `trainer.py` | 阶段 1、3 |
| DDP 初始化和跨 rank 聚合 | `distributed.py` | 阶段 4 |
| 统一保存和完整 resume | `checkpoint.py` | 阶段 1、4 |
| smoke test 与验收 | `tests/`、`PLAN.md` | 阶段 1 至 5 |

状态说明：训练代码已完成；带“真实验证待完成”的项目必须在 converted 20D 数据和官方 2B checkpoint 就绪后再验收。

## P0：开始正式训练前必须完成

- [x] 建立 episode 索引式 Lazy Dataset，不在启动时把全部 episode 加载进 ReplayBuffer。原因：降低每个 rank 的 CPU 内存占用和启动时间，避免 rank 0 Python 序列化及 `scatter_object_list` 成为多卡瓶颈。已由 `cosmos_offline_train/dataset.py` 实现并通过 CPU 测试。
- [x] 使用 `DistributedSampler + DataLoader`，初始设置 `num_workers=4/rank`。原因：每个 rank 独立读取自己的样本分片，消除 rank 0 分发瓶颈，并提供清晰 epoch 语义。已实现；真实多卡吞吐待 converted 数据完成后验证。
- [x] 每个 epoch 调用 `sampler.set_epoch(epoch)`。原因：保证各 rank 在新 epoch 使用一致但不同的 shuffle 顺序，否则每个 epoch 可能重复相同顺序。已实现并通过 sampler CPU 测试。
- [x] 按 task 分层，在每个 task 内按完整 episode 划分 train/val/test，并保存固定 manifest。原因：目录、日期和 am/pm 不是可靠的连续采集边界；当前实验目标是测量已有 task 的同分布学习效果。`collection_group` 仅保留作分析 metadata，不参与默认划分。
- [ ] 只使用 train split 计算 action/proprio/value 归一化统计。原因：使用 val/test 统计量会产生验证和测试数据泄漏。
- [x] 增加独立 validation DataLoader 和 `torch.no_grad()` 验证循环。原因：当前只有训练 loss，无法发现过拟合，也无法可靠选择 checkpoint。代码已完成；真实 2B smoke test 待执行。
- [x] 实现官方 policy/world/value 三种 sample condition mask 和 loss mask。原因：当前固定联合预测不能等价复现官方 50% policy、25% world、25% value 训练目标。mask 单元测试已通过。
- [x] loss 使用有效 mask 元素归一化：`masked_loss.sum() / mask.sum().clamp_min(1)`。原因：若 masked zero 仍进入 `.mean()` 分母，loss 尺度会随 sample 类型和 batch 组成变化。单元测试已覆盖。
- [x] 校验 success/failure 标签逻辑，尤其不要默认 `returns == -1.0` 一定代表失败。原因：错误标签会清零成功样本 action loss，直接破坏 policy 学习。代码只接受 metadata 显式 `episode_labeling.outcome`；数据人工抽样仍待完成。
- [x] 失败样本只屏蔽 policy action loss，仍保留 world/value loss。原因：失败轨迹不应被 policy 模仿，但对动力学和价值学习很重要。新流程只接受 metadata 显式 outcome，不根据 return 猜测。

## P1：完善 loss 和实验记录

- [x] 返回统一 `loss, metrics`，不要只返回名为 `kendall_loss` 的 scalar。原因：当前变量实际是 EDM weighted MSE，命名错误且局部指标没有进入正式日志。
- [x] 将 `kendall_loss` 重命名为 `edm_loss`，或真正实现 Kendall uncertainty weighting 后再使用该名称。原因：避免实验记录和代码含义不一致。新路径不再使用错误命名。
- [x] 分别记录 train/val 的 `total_edm_loss`。原因：总训练 loss 不能反映泛化情况。
- [x] 分别记录 `policy_edm_loss`、`world_edm_loss`、`value_edm_loss`。原因：图像 latent 维度大，总 loss 可能掩盖 action 或 value 退化。
- [x] 分别记录 action、future proprio、external image、wrist image、value 区域的 weighted EDM loss。原因：当前区域指标只有普通 MSE/L1，无法解释各区域对实际优化目标的贡献。
- [x] 保留各区域 unweighted MSE/L1。原因：它们不受 sigma 权重影响，更适合跨实验比较。
- [x] 记录每种 sample 数量、success/failure 数量、各 mask 有效元素数量及 mask ratio。原因：验证采样比例、失败样本处理和 loss 分母是否正确。
- [ ] 按三个机器人任务分别记录 validation loss，并报告 macro average。原因：混合平均会被 episode 数量较大的任务主导。
- [ ] 按低、中、高 sigma 区间记录 loss。原因：判断模型是否只学会低噪声重建，而未改善高噪声生成。
- [x] 记录 learning rate、gradient norm、loss scale、step time、samples/s 和 GPU memory。原因：用于判断数值稳定性、性能瓶颈和不同实验是否公平。

## P2：增强可解释性和最终评估

- [ ] VAE decode 后记录 action MSE/L1，并按关节和 gripper 维度统计 MAE。原因：latent loss 低不一定代表物理控制误差小。
- [ ] VAE decode 后记录 value MAE/RMSE；若 value 是二分类，再记录 accuracy、precision、recall、AUROC。原因：latent value MSE不能直接说明成功预测质量。
- [ ] validation 固定少量样本解码 future image，记录 L1、PSNR、SSIM，可选 LPIPS。原因：latent image loss 无法完整反映可见预测质量；只解码固定样本可控制开销。
- [ ] 定期保存 action、future state、value 的预测与 GT 可视化。原因：数值 loss 难以发现时序错位、相机对应错误和动作维度错序。
- [ ] 最终记录离线 action 指标和真实 rollout success rate。原因：离线 loss 只能筛选 checkpoint，不能替代机器人闭环成功率。

## P3：性能和复现增强

- [x] DataLoader 支持 `persistent_workers=True`、适当 `prefetch_factor` 和 pinned memory。原因：减少 epoch 间 worker 重启和 CPU 到 GPU 数据传输等待；默认 prefetch=2，正式值仍需吞吐实测。
- [ ] 明确在线 VAE 与预编码 VAE 两种路径。原因：在线 VAE 支持每次随机图像增强；预编码 VAE 吞吐更高，但增强结果固定、随机性和数据多样性较低。
- [ ] 若采用预编码 VAE，为每个 episode 预生成多个增强版本或先做轻量像素增强再编码。原因：部分恢复官方在线 VAE 的增强随机性，同时保留较高训练吞吐。
- [ ] checkpoint 保存 model、optimizer、scheduler、scaler、epoch、global step、sampler seed 和 split manifest hash。原因：保证中断恢复后数据顺序和优化状态可复现。主体状态及 RNG 已保存；manifest hash、逐 rank RNG 仍待补。
- [x] 固定并记录 Python、NumPy、PyTorch、DataLoader worker 和 sampler seed。原因：仅设置单个主进程 seed 不能保证多 worker、多 rank 可复现。代码已固定；环境/配置由 resolved YAML 和 checkpoint 保存。
- [ ] 增加短 smoke test：单卡、双卡、少量 episode 各运行若干 step。原因：正式长训练前检查 shape、mask、DDP 梯度同步和 checkpoint 恢复。命令和自动验收已实现，等待真实数据/权重执行。

## 第一阶段最小验收条件

1. 6000+ episode 仅建立索引，启动时不加载完整数据。
2. 单卡和 DDP 均能跑通一个 epoch，无重复或遗漏样本。
3. train/val/test split 固定，归一化统计只来自 train。
4. policy/world/value sample 比例和 mask 经单元测试验证。
5. TensorBoard 或 W&B 能看到 total、三种 sample、五个输出区域的 train/val loss。
6. failure action mask 经人工抽样确认。
7. checkpoint 可恢复，恢复后 global step、scheduler 和 sampler 状态正确。
