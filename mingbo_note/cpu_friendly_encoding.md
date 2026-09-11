# CPU 友好编码改造：避免所有帧同时占用 GPU

## 问题

原版 `policy.encode_episode_state` 在 VAE 编码前，会逐帧调用 `move_transition_to_device` 把所有 video tensor 移到 GPU，然后 `torch.cat` 再产生一份 GPU 副本。

以 1447 帧的 episode 为例：
```
video 输入数据:  ~7.2 GB GPU  (全部帧)
concat 副本:     ~7.2 GB GPU
VAE 激活:        ~2 GB
─────────────────────────────
GPU 峰值:        ~17 GB
```

`encode_batch_size` 只控制了 VAE 编码时每次处理几帧（计算层面的 micro-batch），但没有解决"所有帧从一开始就全部堆积在 GPU 上"的存储问题。

## 解决方案

新增 `encode_episode_cpu_friendly` 函数，与 `policy.encode_episode_state` 功能完全一致，但 video tensor 保持在 CPU，只在 VAE 编码时按 micro-batch 逐批搬到 GPU → 编码 → 立即移回 CPU。

```
改后:
  video 数据:      ~7.2 GB CPU (不占 GPU)
  concat 副本:     ~7.2 GB CPU (不占 GPU)
  VAE 激活:        ~4 GB GPU (16 帧, encode_batch_size=16)
  ─────────────────────────────
  GPU 峰值:        ~5 GB
```

### 核心改动：微批次编码循环

```python
# 原版：所有帧已在 GPU，切片不省显存
batch = gpu_tensor[start:start + 16]  # GPU tensor
latent = policy.encode(batch)          # GPU → GPU

# 新版：CPU 切片 → 上 GPU → 编码 → 立即移回
batch = cpu_tensor[start:start + 16]   # CPU tensor
batch = batch.to("cuda")               # 只有这 16 帧上 GPU
latent = policy.encode(batch).cpu()    # 编码 → 立即移回 CPU
del batch                               # 释放 GPU
```

### 其他保持不变

- 逐帧拷贝 future 图像（位置 2→6, 位置 3→7）
- `get_action_chunk_with_padding` 构造 action chunk
- 潜帧注入（action → 位置 4, proprio → 位置 1/5, value → 位置 8）
- 分配回 transition，写入 LeRobot 数据集

---

## 两个 batch_size 参数

### `--encode_batch_size`（有用）

控制 VAE 编码时每次搬多少帧到 GPU。

```python
for start in range(0, episode_length, encode_batch_size):
    batch = cpu_tensor[start:start + encode_batch_size].to("cuda")
    latent = policy.encode(batch).cpu()
    del batch
```

| 值 | GPU 峰值 | 速度 |
|:---:|:---:|:---:|
| 4 | ~1.5 GB | 慢（CPU↔GPU 搬运次数多） |
| 16 | ~4 GB | 适中 |
| 64 | ~10+ GB | 快（可能 OOM） |

建议：按显存大小调整，24 GB 显卡用 16~32。

### `--batch_size`（训练参数，转换阶段无用）

训练链路中 `batch_size` 的唯一作用——**训练时 DataLoader 每次取多少帧做一次梯度更新**：

```
LeRobot 数据集 → DataLoader(batch_size=1920) → DiT forward → loss → backward
                    ↑
              训练超参，一次取 1920 帧做一批梯度更新
```

和转换脚本的关系：**无关。** 转换阶段是一帧一帧编码存储，不存在批量概念。

原版 `policy.encode_episode_state` 中，这个参数用于构造 `episode_batch` 的 batch 维度，但构造出来的 tensor（`fps`、`padding_mask`）在编码路径中几乎不被读取——属于训练路径遗留下来的格式对齐参数。CPU 友好版本 (`encode_episode_cpu_friendly`) 直接绕过了 `episode_batch` 的构造，`batch_size` 传进来后全程未被引用。保留它仅为函数签名兼容。

### 三个易混参数辨析

| | `encode_batch_size` | `batch_size` | `chunk_size` |
|------|:---:|:---:|:---:|
| 含义 | VAE 编码 micro-batch | 训练 DataLoader batch | 动作预测跨度 |
| 控制什么 | GPU 显存 | 训练收敛速度 | 一次预测多少步 |
| 默认值 | 16 | 16（训练时实际用 1920） | 16 |
| 转换阶段有用吗 | ✅ | ❌ | ✅（注入 + 未来帧跨度） |

---

## 编码流程（6 阶段）

```
阶段 1: stack actions → (E, action_dim), MC returns → (E,)
阶段 2: 逐帧循环（全部 CPU）
        ├─ 拷贝 future wrist/primary 图像
        ├─ get_action_chunk_with_padding → (16, action_dim)
        ├─ 收集 proprio, future_proprio, value
阶段 3: torch.cat 全部 video → (E, 3, 33, 224, 224) CPU
阶段 4: uint8 → float [-1,1] 归一化
        micro-batch: CPU切片 → GPU → encode → CPU → 释放
阶段 5: 潜帧注入（CPU）
        ├─ replace_latent_with_action_chunk → 位置 4
        ├─ replace_latent_with_proprio → 位置 1
        ├─ replace_latent_with_proprio → 位置 5
        └─ value → 位置 8
阶段 6: 分配回 transition，逐帧写入 LeRobot 数据集
```

---

## 涉及的文件

| 文件 | 改动 |
|------|------|
| `data_convert/convert_raw_to_cosmos.py` | 新增 `encode_episode_cpu_friendly`、`LATENT_INDICES`、导入注入函数；修改 `encode_and_write_episode` 调用新函数；新增 `--batch-size` CLI；调整 `--encode_batch_size` 默认值为 16 |
| `pipeline.sh` | 新增 `--encode-batch-size` 和 `--batch-size` 参数传递 |
