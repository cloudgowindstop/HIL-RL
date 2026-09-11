# 实施计划

## 命令复用约定

- 所有人工执行流程通过 `run.sh` 子命令进入。
- 固定使用 `/media/jushen/mingbo-ge/.venv`，不猜测其他 Python 环境。
- 实验参数只放 YAML；shell 不维护第二份 batch size、学习率和 action schema。
- 正式运行时保存最终 YAML、环境摘要、git commit、命令和依赖版本。

## 阶段 0：数据与配置预检

- 建立按采集批次划分的 train/val/test manifest。
- 只用 train 生成归一化统计。
- 校验所有 episode、latent shape、相机顺序、chunk size、T5 key。
- 强制 metadata action 为双臂 20D `cosmos_rotation_6d`，包含两个 gripper。
- 校验 rotation 表示为 matrix first two columns，位姿语义为 `inv(current) @ target`。

验收：不加载 2B 模型即可完成全数据索引和 metadata 检查。

## 阶段 1：最小单卡训练

- 显式加载指定 Predict2 base checkpoint；输出路径、hash、加载参数数和 key 差异。
- 只取 2 个 episode、固定 1 个 batch、关闭随机 shuffle。
- 运行 10 次重复优化，验证同一 batch loss 明显下降。
- 验证参数发生变化、梯度有限、未冻结参数有梯度、冻结参数无梯度。
- 保存 checkpoint，重启恢复，再运行 1 step；确认 step、optimizer、scheduler 一致。

验收：无 NaN/Inf；loss 可下降；resume 后结果连续。

## 阶段 2：rotation-6D 测试

- 检查数据 action 维度 20，顺序为左右臂各 `xyz + rot6d + gripper`。
- identity rotation、随机合法 rotation、模型任意 6D 输出均能投影到 SO(3)。
- 编码、归一化、反归一化、解码 round-trip 旋转误差低于容差。
- 记录左右臂 geodesic mean/median/p95、translation MAE、gripper MAE。
- 确认 padding rotation 使用合法 identity 表示，不使用退化的六个零。

验收：训练、指标、解码全部使用相同 channel order 和统计文件。

## 阶段 3：单卡小数据集

- 取每个任务少量完整 episode，保留正式 train/val 划分。
- 训练 100 至 500 step，每 20 step 验证。
- 检查 train loss 下降，validation loss 有限，各输出区域均有有效 mask。
- 监控 GPU memory、samples/s、DataLoader wait time。

验收：三个任务均产生样本；validation 不参与 backward；日志字段完整。

## 阶段 4：双卡 DDP smoke test

- 运行 20 至 100 step，`workers_per_rank=0` 先验证正确性，再改为 4。
- 检查每个 rank 初始权重一致、样本分片无交叉、总样本数正确。
- 检查 DDP 后参数一致，梯度同步正常，`sampler.set_epoch()` 生效。
- 比较单卡与双卡相同 global batch 的首步 loss，允许浮点误差。
- 从双卡 checkpoint 恢复；再用不同 GPU 数量加载模型做兼容测试。

验收：无 hang、无 rank 0 scatter、无重复/漏样本、checkpoint 可恢复。

## 阶段 5：正式训练前试跑

- 使用约 1% 数据运行 1 至 3 epoch。
- 开启正式增强、bf16、gradient accumulation、4 workers/rank。
- 验证完整 train/val、保存、恢复、指标可视化和最佳 checkpoint 选择。
- 冻结配置、split manifest、dataset metadata 和初始 checkpoint hash。

验收后才能提交 6000+ episode 大规模任务。

## sample mask 上线顺序

1. 数据只有成功 demonstration：先用 100% policy sample，验证基础链路。
2. 有明确 failure/rollout 标签：加入 world/value sample。
3. 对齐官方基础比例：50% policy、25% world、25% value。
4. rollout refinement 实验再使用 10% policy、45% world、45% value。

不允许仅根据 return 数值猜 success/failure；必须读取显式标签。
