# 训练与评测代码重构方案

## 目标结构

```text
cosmos_offline_train/
├── train.py                    # CLI，只解析参数并调用 pipeline
├── run.sh                      # shell 统一入口
├── pipeline/
│   ├── train_pipeline.py       # preflight、DDP、数据、模型、训练、保存编排
│   └── trainer.py              # epoch/step/forward/backward/validation 时机
├── components/
│   ├── config.py               # 配置读取与校验
│   ├── data.py                 # dataset、manifest、DataLoader
│   ├── objectives.py           # policy/world/value/ID mask 与 loss
│   ├── model_adapter.py        # 官方 Cosmos 加载、condition、denoise
│   ├── action_codec.py         # 20D action、6D rotation、反归一化
│   ├── optimization.py         # optimizer、scheduler、gradient
│   ├── checkpointing.py        # save/resume/warm-start/identity
│   ├── distributed.py          # DDP 与跨 rank 聚合
│   └── logging.py              # JSONL 与训练指标
├── evaluation/
│   ├── runner.py               # 独立评测入口
│   ├── protocols.py            # objective、sigma、seed、消融定义
│   ├── policy.py               # action 指标
│   ├── inverse_dynamics.py     # normal/current-only/future-only/shuffle
│   ├── forward_dynamics.py     # future latent/proprio 指标
│   ├── sampling.py             # 完整 diffusion sampling
│   ├── video.py                # VAE decode、PSNR/SSIM/LPIPS
│   ├── value.py                # value 回归、分类、排序、校准
│   ├── aggregation.py          # episode 聚合、bootstrap CI、切片
│   └── report.py               # CSV/JSON/图表/checkpoint 对比
├── tools/
│   └── plot_metrics.py         # 训练曲线工具
├── configs/
├── note/
└── tests/
```

## Pipeline 与组件边界

`pipeline` 只决定执行顺序和生命周期：何时初始化 DDP、加载数据、执行 forward/backward、验证、保存。它不实现指标公式、action 解码或数据格式。

`components` 提供九组可复用能力：配置、数据、objective、模型适配、action codec、优化器、checkpoint、分布式、日志。组件不读取 CLI，也不控制完整训练流程。

`evaluation` 独立于训练 pipeline。训练结束后可加载任意 Base/5K/10K/15K/20K checkpoint，使用相同 protocol 重复评测。训练中的轻量 validation 复用同一套指标函数，但不承担完整视频生成。

## 迁移策略

1. 先创建新 package 和测试，旧文件保留为兼容导入层。
2. 将纯函数先迁入 `components`，保持行为和 checkpoint 格式不变。
3. 将训练编排迁入 `pipeline`，`train.py` 缩减为 CLI。
4. 新评测全部写入 `evaluation`，不继续扩大 `trainer.py`。
5. CPU tests、10-step GPU smoke、旧 20K checkpoint validate 全部通过后，再删除兼容层。

这样避免一次性移动文件导致现有训练入口、resume 和用户本地修改失效。

## 评测实施阶段

### P0：不需要 VAE 的完整离线评测

- Policy action 指标；
- ID 四种条件消融；
- Forward Dynamics latent/proprio 指标；
- Value latent 与标量指标；
- Base/各 checkpoint 固定 seed 对比；
- sample/episode 指标、bootstrap 95% CI；
- CSV/JSON/曲线。

### P1：完整视频生成评测

- 接入官方多步 sampler；
- 恢复被 action/proprio/value injection 覆盖的 latent；
- VAE decode；
- PSNR、SSIM、LPIPS；
- camera/horizon 分组和预测视频。

### P2：闭环评测

- simulator adapter；
- success rate、staged score、完成时间和 failure 类型；
- ID/OOD 随机化。

没有 simulator 时保留接口并明确输出 unavailable，不能用离线 error 替代 success rate。

## 当前数据对 VAE decode 的限制

Parquet 只保存注入后的 `video[16,9,28,28]`。latent 位置 `1/4/5/8` 已被 proprio/action/future-proprio/value 覆盖；原 clean latent 没有保存。直接 VAE decode 会产生伪影。

future image 位于 `6/7`，但视频 VAE 在时间维存在耦合，仍应先恢复被覆盖位置。可选方案：

1. 新转换数据保存被覆盖位置的 `clean latent restore payload`，推荐 float16；
2. 从原始 RGB/HDF5 重新 VAE encode 得到 clean latent，仅用于评测；
3. 保存压缩 GT RGB，仅用于 RGB 指标；不能恢复生成 latent 的完整 decode 条件。

P1 开始前必须先选定数据 schema。现有数据可完成 P0，不能保证严格正确的 RGB decode 评测。

## VAE decode 性能

图像指标必须解码到 RGB；latent L1/MSE 不需要 VAE。

VAE decode 是一次网络前向，完整 diffusion sampling 通常需要 5–35 次 denoise，因此多数情况下 sampler 是主要耗时。decode 相对 encode 常为同量级或略快，但具体速度取决于 tokenizer、batch size、分辨率和显存，不能用固定倍数假设。

实施时新增 benchmark：分别测 encode/decode 的 images/s、峰值显存和 batch size。评测只 decode 选定 checkpoint、episode、camera，并缓存 RGB，避免重复解码。
