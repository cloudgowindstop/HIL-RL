# Cosmos Policy 训练机制详解

## 一、数据流全景：从转换到训练

```
┌─────────────────────────────────────────────────────────────────────┐
│  转换阶段 (data_convert/convert_raw_to_cosmos.py)                     │
│                                                                       │
│  HDF5 轨迹                                                           │
│    │                                                                  │
│    ├─► 图像: JPEG解码 → prepare_images_for_model → 9帧序列             │
│    ├─► 动作: 绝对位姿 → 增量欧拉角 (7D/14D)                            │
│    ├─► Proprio: 末端位姿+夹爪 → min-max归一化                          │
│    │                                                                  │
│    ▼                                                                  │
│  VAE编码 → latent video (16, 9, 28, 28)                               │
│    │                                                                  │
│    ▼                                                                  │
│  写入 LeRobot 数据集:                                                  │
│  { video(latent), proprio, future_proprio, value, action, ... }       │
│    └─── 输入 ───┘  └────────── 目标 ──────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  训练阶段 (our_dataset.py → modeling_cosmos.py)                       │
│                                                                       │
│  LeRobot 数据集                                                       │
│    │                                                                  │
│    ├─► get_action_chunk_with_padding → action_chunk (16, 7)           │
│    ├─► 提取 proprio, future_proprio, value                            │
│    ├─► 提取 video latent (已编码，跳过VAE)                              │
│    │                                                                  │
│    ▼                                                                  │
│  构造 9 帧 latent 序列 (x0):                                           │
│    [blank, proprio, wrist, primary, action, future_*, value]          │
│    │                                                                  │
│    ▼                                                                  │
│  扩散训练:                                                             │
│    条件帧(0-3) → 保持干净, 作为输入                                     │
│    目标帧(4-8) → 加噪声, DiT 去噪预测                                   │
│    │                                                                  │
│    ▼                                                                  │
│  Loss = MSE(x0, model_pred) 逐位置计算                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、9 帧 Latent 序列布局

这是 Cosmos Policy 最核心的设计——**Latent Frame Injection（潜在帧注入）**。

### 2.1 序列结构

```
位置  内容              含义                    训练时               推理时
──────────────────────────────────────────────────────────────────────────
 0   [░░░░] blank        VAE 占位符             干净(条件)           干净(条件)
 1   [████] proprio      当前本体感知(位姿+夹爪)  干净(条件)           干净(条件)
 2   [████] wrist_img    当前腕部相机 latent      干净(条件)           干净(条件)
 3   [████] primary_img  当前主相机 latent        干净(条件)           干净(条件)
 4   [████] ACTION       动作 chunk latent        加噪 → 去噪(目标)   噪声 → 生成 ★
 5   [████] future_prop  未来本体感知 latent      加噪 → 去噪(目标)   噪声 → 生成
 6   [████] future_wrist 未来腕部相机 latent      加噪 → 去噪(目标)   噪声 → 生成
 7   [████] future_prim  未来主相机 latent        加噪 → 去噪(目标)   噪声 → 生成
 8   [████] value        价值函数 latent         加噪 → 去噪(目标)   噪声 → 生成
```

- 每个位置是一帧 latent，shape 为 `(16, 28, 28)`（16 通道 × 28 高 × 28 宽）
- 图像帧(2,3,6,7)来自 VAE 编码的真实图像
- 非图像帧(1,4,5,8)通过**广播/平铺**注入：将向量重复填充到 `(16, 28, 28)` 空间

### 2.2 时间压缩

```
原始视频: 33 帧 uint8 (33, 224, 224, 3)
    │  VAE 编码 (temporal compression factor = 4)
    ▼
Latent:   9 帧  (9, 28, 28, 16)

每 4 帧原始图像 → 1 帧 latent
9 × 4 = 36 ≈ 33 (第 0 帧只有 1 帧原始图像)
```

### 2.3 为什么用 Latent Frame Injection？

传统做法：
```
Video Encoder → Feature Vector → Action Head (MLP) → action
```

Cosmos Policy 做法：
```
直接把 action 当成"一帧 latent"塞进 DiT 的序列里
→ 零结构修改，完全复用预训练视频模型的时空推理能力
```

好处：
- 不需要额外的 action head、inverse dynamics model
- 预训练视频模型的物理先验（物体运动、碰撞、重力等）直接用于动作预测
- 动作预测和未来帧预测共享同一个去噪过程

---

## 三、扩散训练机制

### 3.1 核心思想

Cosmos Policy 基于 **EDM (Elucidating Diffusion Models)** 扩散框架。训练时模型学习从噪声中恢复完整的 9 帧 latent 序列。

### 3.2 训练步骤

```python
# Step 1: 构造 ground truth 序列
x0 = construct_9_frame_sequence(dataset_frame)  # (B, 16, 9, 28, 28)

# Step 2: 注入非图像数据
x0 = replace_latent_with_proprio(x0, proprio, position=1)
x0 = replace_latent_with_proprio(x0, future_proprio, position=5)
x0 = replace_latent_with_value(x0, value, position=8)
# action 本身已在数据集中，位于 position 4

# Step 3: 设定条件掩码
condition_mask = [1, 1, 1, 1, 0, 0, 0, 0, 0]  # 前4帧为条件
#                 ↑ 条件帧(输入)  ↑ 目标帧(要预测)

# Step 4: 加噪
sigma = random_noise_level()                    # 噪声强度
noise = random_normal(like=x0)                  # 随机噪声
xt = x0 + noise * sigma                         # 加噪后的序列

# Step 5: 去噪预测
model_pred = DiT(xt, sigma, condition)          # DiT 预测干净信号

# Step 6: 计算损失
loss = MSE(x0, model_pred.x0)                   # 逐像素 MSE
```

### 3.3 三种训练目标（按比例混合）

| 目标 | Batch 占比 | 条件帧 | 目标帧 | 学什么 |
|------|:---:|--------|--------|--------|
| **Policy** | 50% | 0-3 (s) | 4-8 (a, s', V) | p(a, s', V \| s) — 从当前状态预测动作和未来 |
| **World Model** | 25% | 0-4 (s, a) | 5-8 (s', V) | p(s', V \| s, a) — 从状态+动作预测未来 |
| **Value Function** | 25% | 0-7 (s, a, s') | 8 (V) | p(V \| s, a, s') — 从完整轨迹预测价值 |

```
Policy 目标:
  [✓] [✓] [✓] [✓] [✗] [✗] [✗] [✗] [✗]
   0    1    2    3    4    5    6    7    8
   条件=已知                 目标=要预测

World Model 目标:
  [✓] [✓] [✓] [✓] [✓] [✗] [✗] [✗] [✗]
   0    1    2    3    4    5    6    7    8
   条件=已知(含action)        目标=要预测

Value Function 目标:
  [✓] [✓] [✓] [✓] [✓] [✓] [✓] [✓] [✗]
   0    1    2    3    4    5    6    7    8
   条件=已知(全部)                 目标=要预测
```

### 3.4 条件帧数量随机化

训练时 `num_conditional_frames` 在 0~8 之间**随机采样**，而非固定为 4。这让模型学会在不同信息量下推理：

```python
# 可能随机到 2 个条件帧:
[✓] [✓] [✗] [✗] [✗] [✗] [✗] [✗] [✗]
 只有 blank + proprio 已知，连图像都没有 → 模型被迫从 proprio 推理

# 可能随机到 6 个条件帧:
[✓] [✓] [✓] [✓] [✓] [✓] [✗] [✗] [✗]
 全部当前帧 + action 已知 → 只需预测未来帧
```

推理时固定 `num_conditional_frames=4`（前 4 帧为条件），和在线采集一致。

### 3.5 逐位置损失

```python
# 每个 latent 位置单独计算和监督
action_loss         = MSE(位置4_真值, 位置4_预测)   # 动作预测质量
future_image_loss   = MSE(位置7_真值, 位置7_预测)   # 未来图像预测质量
future_wrist_loss   = MSE(位置6_真值, 位置6_预测)   # 未来腕部预测质量
future_proprio_loss  = MSE(位置5_真值, 位置5_预测)   # 未来状态预测质量
value_loss          = MSE(位置8_真值, 位置8_预测)   # 价值预测质量

total_loss = action_loss + future_image_loss + future_proprio_loss + value_loss
```

---

## 四、关键函数详解

### 4.1 `get_action_chunk_with_padding`

**用途**：训练时从 episode 的动作序列中切出固定长度的 action chunk。

```python
def get_action_chunk_with_padding(actions, relative_step_idx, chunk_size, num_steps):
    """
    actions:          (num_steps, action_dim) — 整个 episode 的动作序列
    relative_step_idx: 从第几步开始取
    chunk_size:       要取多长 (默认 16)
    num_steps:        episode 总步数
    """
```

**两种情况**：

```
情况1: step_idx=42, remaining=(100-42)=58 ≥ 16
  → actions[42:58] 直接切片  ✅

情况2: step_idx=90, remaining=(100-90)=10 < 16
  → actions[90:100] + 重复 actions[-1] 6 次  ✅
  → [a90, a91, ..., a99, a99, a99, a99, a99, a99]
                          ↑ 末尾帧重复填充
```

**为什么需要填充？** episode 末尾剩余步数不足 chunk_size，与其丢弃边界数据，不如用最后一帧重复填充——等价于"到达终点后保持不动"。

**在哪里使用？** 仅训练 Dataset 类中使用，转换脚本不需要（转换时逐帧独立计算 action）。

### 4.2 `duplicate_array`

```python
def duplicate_array(arr, total_num_copies=4):
    return np.stack([arr] * total_num_copies)

# arr: (224, 224, 3) → result: (4, 224, 224, 3)
```

**用途**：将一张图沿第 0 维复制 N 份，对应 VAE 的时间压缩因子 4。每个 latent 帧由 4 帧原始图像压缩而来，所以每张观测图要复制 4 次。

### 4.3 `rescale_proprio`

```python
def rescale_proprio(proprio, dataset_stats, non_negative_only=False):
    # [-1, +1] 归一化:
    arr = 2 * ((proprio - min) / (max - min)) - 1

    # [0, +1] 归一化:
    arr = (proprio - min) / (max - min)
```

将原始物理量（末端位姿 xyz 米级，四元数 [-1,1]，夹爪 [0,1]）统一到模型友好的 [-1, +1] 范围。

### 4.4 `prepare_images_for_model`

**用途**：图像预处理，必须在 VAE 编码前执行，和在线推理保持一致。

```
输入图像 (H, W, 3) uint8
    │
    ├─► JPEG 压缩往返 (Q=95):  模拟真实相机 JPEG 输出
    ├─► resize 224×224:        统一尺寸
    └─► 90% 中心裁剪+resize:   模拟训练时 RandomResizedCrop 的平均效应
    │
    ▼
输出图像 (224, 224, 3) uint8
```

**注意**：转换脚本中的是内联简化版（避免 cosmos_policy 重依赖），用 NumPy/OpenCV 替代了原版的 PyTorch 实现，语义等价但像素级有微小差异（`cv2.INTER_LINEAR` vs `F.resize(antialias=True)`），对训练影响可忽略。

### 4.5 `compute_delta_action_single`

```python
def compute_delta_action_single(pose_curr, pose_future, grip_curr, grip_future, action_scale):
    T_curr   = pose_7d_to_matrix(pose_curr)       # 当前末端 4×4 齐次矩阵
    T_future = pose_7d_to_matrix(pose_future)      # 未来末端 4×4 齐次矩阵
    T_delta  = inv(T_curr) @ T_future              # 相对变换矩阵

    dpos  = T_delta[:3, 3]                         # 平移增量 (3,)
    drot  = Rotation.from_matrix(T_delta[:3,:3]).as_euler("xyz")  # 旋转增量 (3,)
    dgrip = grip_future - grip_curr                # 夹爪增量 (1,)

    action = [dpos/scale_pos, drot/scale_rot, dgrip/scale_grip]  # 归一化 (7,)
    return clip(action, -1, 1)
```

将绝对位姿差转为增量欧拉角动作。chunk_size=16 表示从 t 到 t+16 的位姿差压缩为一个 7D 向量。

---

## 五、训练与转换的关系

| | 转换脚本 | 训练 Dataset |
|------|------|------|
| **职责** | 原料生产：逐帧计算并存储 | 成品组装：聚合为训练 batch |
| **action** | 计算单帧增量(7D/14D)并存储 | `get_action_chunk_with_padding` 聚合为 (16,7) chunk |
| **video** | VAE 编码为 latent 并存储 | 直接读取 latent（跳过 VAE） |
| **proprio** | 归一化并存储 | 直接读取 |
| **预处理** | 执行 `prepare_images_for_model` | 不需要（latent 已编码） |
| **产出** | LeRobot 数据集 | 训练梯度 |

**关键**：转换脚本产出的每一帧都同时包含输入和训练目标。训练时 Dataset 从多帧中聚合出 action chunk 作为 ground truth。

---

## 六、在线采集 vs 离线转换一致性

两条路径处理完全一致，确保训练-推理分布对齐：

| 步骤 | 在线采集 (CosmosWrapper) | 离线转换 (data_convert/convert_raw_to_cosmos.py) |
|------|------|------|
| 图像预处理 | `prepare_images_for_model` | `prepare_images_for_model` (内联版) |
| 9帧序列构造 | `build_nine_frame_sequence` | `build_nine_frame_sequence` |
| VAE 编码 | `policy.encode_episode_state` | `policy.encode_episode_state` |
| 潜帧注入 | `replace_latent_with_*` | 同（encode_episode_state 内部调用） |
| 写入格式 | LeRobot dataset | LeRobot dataset |

---

## 七、模型信息

- **模型**：Cosmos Policy Predict2-2B（非 Cosmos3）
- **基座**：Cosmos-Predict2-2B-Video2World
- **架构**：DiT (Diffusion Transformer) + Wan2.1 VAE
- **参数量**：2B
- **文本编码器**：T5-XXL (4096-dim)
- **动作**：chunk_size=16, action_dim=7 (单臂) / 14 (双臂)
- **推理去噪步数**：5 步
- **LIBERO 基准**：98.5% 平均成功率
