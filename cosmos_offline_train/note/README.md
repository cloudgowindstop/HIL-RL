# Cosmos Offline Train 文档索引

本目录保存离线训练、评测和实验复现说明。建议先从本页查找，不按文件创建时间阅读。

## 按需求查找

| 需求 | 阅读文档 | 内容 |
|---|---|---|
| 配置单机多卡 DDP/百舸任务 | [distributed_training_guide.md](distributed_training_guide.md) | 代码改造、4/6/8 卡资源、环境变量、启动命令、监控和故障排查 |
| 第一次运行小规模训练 | [tiny_training_guide.md](tiny_training_guide.md) | 环境检查、preflight、split、smoke、100-step tiny、resume |
| 修改 batch、步数、学习率、训练模式 | [training_parameters_guide.md](training_parameters_guide.md) | YAML 参数、有效全局 batch、Policy/World/Value/ID 配比 |
| 复现本次六卡 Policy 实验 | [policy_learner_eval_6gpu.md](policy_learner_eval_6gpu.md) | 数据、14D Euler、2,000 steps、完整验证、启动命令、输出位置 |
| 重新生成和解读训练曲线 | [plotting_guide.md](plotting_guide.md) | 绘图命令、六张图、CSV/JSON、注意事项 |
| 快速了解应报告哪些指标 | [evaluation_metrics_summary.md](evaluation_metrics_summary.md) | EDM、Policy、ID、World、视频、Value、泛化指标及公式 |
| 规划完整评测实验 | [evaluation_plan.md](evaluation_plan.md) | Base/checkpoint 对比、消融、完整 rollout、尚缺功能 |
| 理解训练/评测代码边界 | [refactor_and_evaluation_implementation.md](refactor_and_evaluation_implementation.md) | pipeline、components、evaluation 分层和迁移方案 |

## 推荐阅读顺序

### 启动新训练

1. [training_parameters_guide.md](training_parameters_guide.md)：确认数据、权重、训练模式和有效 batch。
2. [tiny_training_guide.md](tiny_training_guide.md)：先完成 preflight 与 smoke test。
3. 对应实验说明，例如 [policy_learner_eval_6gpu.md](policy_learner_eval_6gpu.md)：启动正式训练。
4. [plotting_guide.md](plotting_guide.md)：生成并检查结果图。

### 判断模型是否有效

1. [evaluation_metrics_summary.md](evaluation_metrics_summary.md)：确定指标含义和最低验收组合。
2. [evaluation_plan.md](evaluation_plan.md)：固定 validation split、noise seed 和 checkpoint 比较协议。
3. [plotting_guide.md](plotting_guide.md)：导出曲线、表格和 checkpoint 对比。

## 文档边界

- “指南”描述当前可直接复用的操作。
- “方案”包含已实现能力和未来工作；不能把计划项视为已完成。
- 具体实验参数以对应 YAML 为准；文档用于解释，不替代配置。
- `metrics.jsonl` 是原始训练记录，`plots/` 是可重新生成的派生结果。

## 新增文档约定

- 通用操作写入现有指南，避免为一次命令重复建文档。
- 独立实验使用 `<task>_<mode>_<hardware>.md` 命名，并记录配置、数据、有效 batch、步数、验证协议和输出目录。
- 新文档创建后必须加入本索引。
