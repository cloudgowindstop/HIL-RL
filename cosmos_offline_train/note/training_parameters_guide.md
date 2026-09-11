# Cosmos 离线训练可调参数速查

> 适用目录：`HIL-RL/cosmos_offline_train/`
> 参数入口：`configs/*.yaml`（一个实验一个 yaml，改参数只改 yaml，不改 Python）
> 启动方式：`CONFIG=configs/xxx.yaml bash run.sh <mode> <gpus>`

---

## 一、配置文件总览

| yaml 文件 | 训练模式 | 用途 |
|-----------|----------|------|
| `cosmos_offline_6d.yaml` | `policy_only` | 纯策略 baseline |
| `cosmos_offline_6d_base_joint.yaml` | `base_joint` | 官方默认（policy 50% + world 25% + value 25%） |
| `back_handle_joint_6d.yaml` | `base_joint` | back-handle 任务 joint |
| `back_handle_inverse_dynamics_6d.yaml` | `inverse_dynamics_only` | 逆向动力学 |
| `cosmos_offline_6d_custom_mix.yaml` | `custom_mix` | 自定义目标混合 |

---

## 二、可调参数总表（按 yaml 段落分组）

### 1. `model` 段 —— 模型与权重

| 参数 | 默认值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `checkpoint_path` | Cosmos-Policy-LIBERO-Predict2-2B/model-480p-16fps.pt | 基础权重（禁止随机初始化 2B 模型） | `model.checkpoint_path` |
| `tokenizer_path` | .../tokenizer/tokenizer.pth | Wan2.1 VAE tokenizer 权重 | `model.tokenizer_path` |
| `experiment` | cosmos_predict2_2b_480p_our_dataset | 实验名（决定模型结构） | `model.experiment` |
| `initialization` | predict2_base | 初始化方式 | `model.initialization` |
| `precision` | bfloat16 | 训练精度 | `model.precision` |
| `strict_checkpoint` | false | 是否严格匹配 checkpoint key | `model.strict_checkpoint` |

### 2. `data` 段 —— 数据与加载

| 参数 | 默认值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `root` | cosmos_data_.../back_handle_..._success | 转换好的数据集根目录 | `data.root` |
| `train_manifest` / `val_manifest` | .../splits/xxx/train.json / val.json | 划分好的 manifest 路径 | `data.train_manifest` 等 |
| `t5_embeddings_path` | .../t5_embeddings.pkl | T5 文本编码缓存 | `data.t5_embeddings_path` |
| `statistics_path` | .../dataset_statistics.json | 动作统计（20 维 min/max） | `data.statistics_path` |
| `workers_per_rank` | 4 | DataLoader worker 数（加速读盘） | `data.workers_per_rank` |
| `prefetch_factor` | 2 | 每个 worker 预取批数 | `data.prefetch_factor` |
| `pin_memory` | true | 锁页内存加速 CPU→GPU | `data.pin_memory` |
| `persistent_workers` | true | worker 常驻（省重启开销） | `data.persistent_workers` |

> ⚠️ 以下字段是 **schema 强校验**，改错会 preflight 拒绝，不要乱改：
> `action_encoding=cosmos_rotation_6d`、`action_dimension=20`、`number_of_arms=2`、`chunk_size=16`、`action_includes_gripper=true`

### 3. `training` 段 —— 核心训练超参

| 参数 | 默认值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `seed` | 42 | 随机种子 | `training.seed` |
| `epochs` | 1 | 训练轮数 | `training.epochs` |
| `max_steps` | null（或 5000） | 最大步数；null=跑满 epochs | `training.max_steps` |
| **`batch_size_per_rank`** | **1** | 每张卡每步 batch（**2B 模型显存约束，别乱加大**） | `training.batch_size_per_rank` |
| **`gradient_accumulation_steps`** | **1** | 梯度累积步数（**提高有效 batch 的首选**） | `training.gradient_accumulation_steps` |
| `learning_rate` | 1.0e-4 | 学习率 | `training.learning_rate` |
| `validation_every_epochs` | 1 | 每几个 epoch 验证一次 | `training.validation_every_epochs` |
| `checkpoint_every_steps` | 1000 | 每多少 step 存 checkpoint | `training.checkpoint_every_steps` |
| `log_every_steps` | 10 | 每多少 step 打日志 | `training.log_every_steps` |
| `gradient_clip_norm` | 10.0 | 梯度裁剪阈值 | `training.gradient_clip_norm` |

### 4. `objectives` 段 —— 训练目标

| 参数 | 可选值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `training_mode` | policy_only / inverse_dynamics_only / base_joint / joint_with_inverse / custom_mix | 训练目标模式 | `objectives.training_mode` |
| `policy_auxiliary_targets` | true / false | policy_only 是否预测 future+value（必须 true） | `objectives.policy_auxiliary_targets` |
| `policy_probability` | 0~1 | custom_mix 时 policy 比例 | `objectives.policy_probability` |
| `world_probability` | 0~1 | custom_mix 时 world 比例 | `objectives.world_probability` |
| `value_probability` | 0~1 | custom_mix 时 value 比例 | `objectives.value_probability` |
| `inverse_dynamics_probability` | 0~1 | custom_mix 时 ID 比例 | `objectives.inverse_dynamics_probability` |

> 四种模式 → 概率元组 `(policy, world, value, inverse_dynamics)`：
> - `policy_only` = (1.0, 0, 0, 0)
> - `inverse_dynamics_only` = (0, 0, 0, 1.0)
> - `base_joint` = (0.5, 0.25, 0.25, 0)
> - `joint_with_inverse` = (0.4, 0.2, 0.2, 0.2)

### 5. `split` 段 —— 数据集划分

| 参数 | 默认值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `create_if_missing` | true | manifest 缺失时自动创建 | `split.create_if_missing` |
| `overwrite` | false | 已存在时是否覆盖（防数据漂移） | `split.overwrite` |
| `seed` | 42 | 划分随机种子 | `split.seed` |
| `max_episodes` | 10 | 最多用多少 episode（smoke 用小值） | `split.max_episodes` |
| `ratios` | [0.8, 0.2, 0.0] | train / val / test 比例 | `split.ratios` |

### 6. `evaluation` 段 —— 验证/评测

| 参数 | 默认值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `noise_seed` | 20260820 | 验证噪声种子（可复现） | `evaluation.noise_seed` |
| `sigma_values` | [0.1, 0.5, 0.9] | 验证用的噪声 sigma 列表 | `evaluation.sigma_values` |
| `max_validation_batches` | 2 或 200 | 验证批数上限（null=全量） | `evaluation.max_validation_batches` |
| `objectives` | null 或 [inverse_dynamics] | 评测哪些目标（null=全部） | `evaluation.objectives` |
| `inverse_condition_ablations` | [normal] | ID 消融（normal/current_only/future_only/future_shuffle） | `evaluation.inverse_condition_ablations` |
| `progress_every_batches` | 10 | 验证进度打印频率 | `evaluation.progress_every_batches` |
| `cache_episodes` | 2 | 验证时缓存 episode 数 | `evaluation.cache_episodes` |
| `bootstrap_samples` | 2000 | episode 聚合 bootstrap 采样数 | `evaluation.bootstrap_samples` |

### 7. `runtime` 段 —— 输出与冒烟

| 参数 | 默认值 | 作用 | 在哪改 |
|------|--------|------|--------|
| `output_dir` | .../train_outputs/xxx | 输出目录 | `runtime.output_dir` |
| `smoke_output_dir` | .../xxx_smoke | 冒烟测试输出目录 | `runtime.smoke_output_dir` |
| `smoke_test` | true / false | 是否冒烟测试 | `runtime.smoke_test` |
| `smoke_max_train_steps` | 10 | 冒烟训练步数 | `runtime.smoke_max_train_steps` |
| `smoke_max_validation_batches` | 2 | 冒烟验证批数 | `runtime.smoke_max_validation_batches` |

---

## 三、有效 batch 的计算（重要）

```python
有效全局 batch = batch_size_per_rank × world_size × gradient_accumulation_steps
```

- `batch_size_per_rank` 是 2B 模型显存硬约束，**通常固定 = 1**
- 想提高有效 batch，**优先调大 `gradient_accumulation_steps`**（时间换 batch）或**加卡**
- 直接调大 `batch_size_per_rank` 会 OOM（单卡 forward 峰值约 60GB，见 learner_copy_dist.py 日志）

---

## 四、常用调整场景速查

| 想做什么 | 改哪里 |
|----------|--------|
| 提高有效 batch（显存不够） | `training.gradient_accumulation_steps` ↑（或加卡） |
| 调学习率 | `training.learning_rate` |
| 跑多久 | `training.epochs` 或 `training.max_steps` |
| 换训练目标 | `objectives.training_mode` |
| 控制存点频率 | `training.checkpoint_every_steps` |
| 控制日志频率 | `training.log_every_steps` |
| 加速数据加载 | `data.workers_per_rank` / `data.prefetch_factor` |
| 用多少数据 | `split.max_episodes` |
| 调整验证强度 | `evaluation.sigma_values` / `evaluation.max_validation_batches` |
| 换数据集/权重 | `data.root` / `model.checkpoint_path` |
