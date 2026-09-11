# Cosmos 离线训练

本目录提供独立离线 Cosmos Policy 训练路径，不复用在线 actor/learner 主循环。

## 边界

- 输入：已经完成质量检查、split 固化、统计固化、VAE/latent 转换的数据。
- 模型：官方 `CosmosPolicyVideo2WorldModel` 与 Predict2 配置。
- 分布式：每个 rank 使用 `DistributedSampler + DataLoader` 独立读取。
- 输出：统一 checkpoint、训练/验证指标、配置与数据版本记录。
- 不包含：actor server、在线 ReplayBuffer、SAC、expert training、机器人控制循环。

## 文件职责

```text
train.py             唯一入口，只编排流程
pipeline/            train lifecycle 与 trainer
components/          配置、数据、objective、模型、优化、checkpoint、DDP、日志
evaluation/          Policy/ID/World/Value/video/统计/报告
config.py            单一配置及 metadata 强校验
action_spec.py       14D Euler / 20D rotation-6D 动作规格
dataset.py           Lazy sample index 和按需读取
dataloader.py        split、DistributedSampler、workers
model.py             模型创建和显式预训练权重加载
masks.py             policy/world/value/inverse-dynamics/failure mask
losses.py            EDM 及分区域 loss
rotation_6d.py       torch SO(3) 投影和旋转几何指标
metrics.py           日志及跨 rank 聚合
plot_metrics.py      可复用 loss 曲线、CSV、摘要导出
trainer.py           train/validation loop
checkpoint.py        唯一保存和恢复格式
distributed.py       DDP 生命周期
configs/             实验配置
tests/               小规模及组件测试
tools/               旧数据校验与 metadata/manifest 侧文件生成
```

旧根模块暂时保留为兼容实现，避免破坏已有 checkpoint、resume 和脚本；新代码从
`pipeline/`、`components/`、`evaluation/` 入口使用。

## 参考来源

实现时复用并对齐以下已有流程，而非从零重新定义 Cosmos 训练：

1. 官方入口 `cosmos-policy/cosmos_policy/scripts/train.py`：Imaginaire 启动、配置与 trainer 生命周期。
2. 官方 trainer `cosmos-policy/cosmos_policy/trainer.py`：优化器、scheduler、GradScaler、checkpoint 语义。
3. 官方 experiment `cosmos_policy_experiment_configs.py`：Predict2-2B、Wan tokenizer、EDM SDE、官方 `DistributedSampler + DataLoader`。
4. 官方 `policy_video2world_model.py`：policy/world/value condition mask。
5. 当前 `learner_copy_dist.py`：已验证的 DDP 包装范围、Cosmos forward/backward 接入；不继承 rank 0 ReplayBuffer/scatter 架构。
6. 当前 `data_convert/cosmos_rotation_6d.py`：rotation-6D 唯一编码语义和 CPU round-trip。

详细实施顺序、6D 适配和 smoke test 见 `PLAN.md`。

## 当前实现状态

已实现：

- converted Cosmos Parquet 的 Lazy episode index、worker-local 单 episode cache；
- 按 task 分层、task 内按完整 episode 划分的 split manifest、`DistributedSampler + DataLoader`；
- 双臂 14D Euler 与 20D rotation-6D schema、SO(3) geodesic metric；
- policy/world/value/failure mask 与有效元素 EDM reduction；
- inverse dynamics：条件为 current + future，唯一训练目标为 action；
- 官方 Cosmos checkpoint 显式加载、预编码 latent EDM forward；
- bf16、gradient accumulation、DDP、train/validation、JSONL metrics；
- 每个 rank 独立加载本地 GPU 的 T5 cache，并在训练前检查全部 task key；
- model/optimizer/scheduler/scaler/RNG checkpoint；
- checkpoint 内数据集、manifest、statistics、T5、基础权重身份校验；
- `policy_only` 明确表示论文 Policy objective，保留 future state/value 辅助目标；
- 固定 8 train / 2 validation smoke manifest，存在时禁止覆盖；
- 固定 noise seed 与多 sigma validation、双臂 16-step horizon 物理指标；
- 独立 `validate` 模式，不创建 optimizer、不执行 backward；
- CPU correctness tests。

## 数据布局与动作格式

训练入口兼容两种布局：

- `unified`：一个未划分的数据根目录，由现有 split 逻辑生成 train/val manifest。
- `pre_split`：数据已位于独立 `train/`、`eval/`（或 `val/`）目录；代码只建立索引，不重新随机划分，也不改 Parquet。

旧 14D Euler 数据的关键配置如下：

```yaml
data:
  layout: pre_split
  root: /media/jushen/mingbo-ge/install_handle
  train_directory: train
  val_directory: eval
  action_encoding: legacy_euler
  action_dimension: 14
  rotation_scale: 0.06  # 必须与该数据转换时使用的值一致
```

首次使用旧数据时，通过工具校验 schema，并生成训练所需的 metadata 和 manifest：

```bash
python -m cosmos_offline_train.tools.prepare_legacy_dataset \
  --root /media/jushen/mingbo-ge/install_handle \
  --task "install handle" \
  --statistics-path /path/to/original_14d_dataset_statistics.json \
  --action-encoding legacy_euler \
  --rotation-scale 0.06
```

工具不会修改原始 Parquet。必须提供与该 14D 数据配套的统计文件和真实
`rotation_scale`；不能用 20D rotation-6D 统计代替，也不能从验证集重新拟合。
小规模兼容性训练可直接参考
`configs/install_handle_euler_14d_tiny.yaml`；其 `max_steps=10`，不会替代正式实验配置。

## Inverse dynamics 与曲线

`inverse_dynamics_only` 训练时，latent 位置 `0:4` 和 `5:8` 为条件，位置 `4`
为 action 目标，位置 `8` 不参与条件和 loss。失败 episode 保留 ID action loss。

```bash
CONFIG=cosmos_offline_train/configs/back_handle_inverse_dynamics_6d.yaml \
  bash cosmos_offline_train/run.sh smoke-single 0

bash cosmos_offline_train/run.sh plot \
  /path/to/train_outputs/experiment/metrics.jsonl

# 同一 validation 设置评估原始初始化，作为 checkpoint 对照。
CONFIG=cosmos_offline_train/configs/back_handle_inverse_dynamics_6d.yaml \
  bash cosmos_offline_train/run.sh validate-base 0
```

曲线工具导出 total/objective/region EDM、MSE、L1、6D rotation 几何误差、
translation/gripper、gradient norm、learning rate，以及 CSV/JSON 摘要。

## 独立全目标评测

```bash
CONFIG=cosmos_offline_train/configs/back_handle_inverse_dynamics_6d.yaml \
  bash cosmos_offline_train/run.sh evaluate 0 /path/to/checkpoint.pt

CONFIG=cosmos_offline_train/configs/back_handle_inverse_dynamics_6d.yaml \
  bash cosmos_offline_train/run.sh evaluate-base 0
```

评测包含 Policy、World、Value、ID，以及 ID 的 normal/current-only/future-only/
future-shuffle 消融。`evaluation/runner.py` 支持 `--max-batches` 和独立输出目录。

RGB 评测代码已提供 PSNR、SSIM、可选 LPIPS、多步 sampler 和安全 VAE decode。
现有 Parquet 没有 clean restore latent，故 decode 会显式拒绝运行，不会输出伪影指标。
完成 RGB 评测前需扩展转换 schema 或从原始 RGB 重新生成 restore latent。

当前 `cosmos_offline_6d.yaml` 已配置为 back-handle 08-03 success-only
rotation-6D smoke 实验：

```yaml
model:
  checkpoint_path: /media/jushen/linda-zhao/HIL-RL-Project/cosmos-policy/cosmos_policy/models/Cosmos-Policy-LIBERO-Predict2-2B/model-480p-16fps.pt
data:
  root: /media/jushen/linda-zhao/HIL-RL-Project/cosmos_data_6drotation_monitor_20260819/back_handle_20260803_pm_success
  t5_embeddings_path: /media/jushen/linda-zhao/HIL-RL-Project/raw_data/tienyi_prod2_dualArm-gripper-3cameras_394_back-handle-installation_20260803_pm/t5_embeddings.pkl
  statistics_path: /media/jushen/linda-zhao/HIL-RL-Project/cosmos_data_6drotation_monitor_20260819/back_handle_20260803_pm_success/dataset_statistics.json
runtime:
  output_dir: /media/jushen/linda-zhao/HIL-RL-Project/train_outputs/back_handle_0803_smoke
```

当前 smoke 数据集包含 159 个完整 episode、246052 帧；preflight 检查
metadata、20D action schema、Parquet 数量和统计文件一致性。
`prepare-splits` 从中固定选择 10 个完整 episode，划分为 8 train、2 validation；
该 smoke split 不创建 test 集。

## 统一 shell 入口

所有常用终端流程由 `run.sh` 表达：

```bash
bash cosmos_offline_train/run.sh env-check
bash cosmos_offline_train/run.sh preflight
bash cosmos_offline_train/run.sh prepare-splits
bash cosmos_offline_train/run.sh test-cpu
bash cosmos_offline_train/run.sh smoke-single 0
bash cosmos_offline_train/run.sh smoke-ddp 0,1
bash cosmos_offline_train/run.sh train 0,1
bash cosmos_offline_train/run.sh resume 0,1 /path/to/checkpoint
bash cosmos_offline_train/run.sh validate 0 /path/to/checkpoint
```

配置分层：

- `run.sh`：默认使用 workspace 下 `.venv`；可通过 `VENV_DIR` 和 `CONFIG` 环境变量覆盖。
- `configs/*.yaml`：数据版本、模型权重、action schema 和训练超参数，需要随实验保存。
- Python：业务校验和训练实现；shell 不负责解析数据 metadata。

`run.sh` 与 Python 训练链路均已实现。训练会先强制执行 schema/data preflight；
缺少转换数据、统计文件、T5 cache 或官方 checkpoint 时会主动拒绝，防止误训练。
