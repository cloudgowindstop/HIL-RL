# 状态转换函数：从 data_batch 到 LeRobot frame

## 一句话概括

这两个函数负责**在不同数据格式之间搭桥**——把 Cosmos Policy 模型产出的 tensor 状态，转换成 LeRobot 数据集可以存储的 numpy frame。

```
data_batch (tensor dict) → Transition.state (tensor dict) → LeRobot frame (numpy dict)
        ↑                            ↑                            ↑
  build_nine_frame_sequence   cosmos_obs_to_transition_state   cosmos_encoded_state_to_frame
```

---

## 一、`cosmos_obs_to_transition_state` — 数据统一化

### 源码

```python
def cosmos_obs_to_transition_state(obs: dict) -> dict:
    state = {}
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            state[key] = val.detach().cpu()       # ① tensor → CPU tensor
        elif isinstance(val, np.ndarray):
            state[key] = torch.from_numpy(val)     # ② numpy → tensor
        elif isinstance(val, (int, float)):
            state[key] = torch.tensor(val)         # ③ 标量 → tensor
    return state
```

### 作用：把杂乱的输入统一为 `dict[str, torch.Tensor]`

| 输入类型 | 处理 | 示例 |
|----------|------|------|
| `torch.Tensor` | `.detach().cpu()` 切断梯度并移 CPU | `video: (1,3,33,224,224) uint8` |
| `np.ndarray` | `torch.from_numpy()` 转 tensor | `proprio: (8,) float32` |
| `int / float` | `torch.tensor()` 包装成 tensor | `fps: 16` |

### 在流水线中的位置

```
build_nine_frame_sequence 产出:
  data_batch = {
      "video":              torch.Tensor (uint8),
      "proprio":            torch.Tensor (bfloat16),
      "t5_text_embeddings": torch.Tensor (bfloat16),
      "fps":                torch.Tensor,
      "padding_mask":       torch.Tensor,
      "num_conditional_frames": 4,              ← int (非 tensor)
      "current_proprio_latent_idx":  tensor,    ← 各 latent 索引
      ...
  }

        │ cosmos_obs_to_transition_state(data_batch)
        ▼

  state = {
      "video":              torch.Tensor (CPU),
      "proprio":            torch.Tensor (CPU),
      "t5_text_embeddings": torch.Tensor (CPU),
      "fps":                torch.Tensor (CPU, 来自 int→tensor 转换),
      ...
  }
  所有值都是 torch.Tensor，都在 CPU 上，没有梯度
```

### 为什么需要这一步？

1. **统一类型**：下游代码（`encode_episode_state`、`Transition` 构造）只接受 tensor dict
2. **切断梯度**：`detach()` 确保推理产生的 tensor 不会带计算图（避免内存泄漏）
3. **移 CPU**：VAE 编码前在 CPU 上组织数据，编码时才搬 GPU

---

## 二、`cosmos_encoded_state_to_frame` — 写入前的拆包

### 源码

```python
def cosmos_encoded_state_to_frame(state: dict) -> dict:
    frame_state = {}
    for key, val in state.items():
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()
            if val.ndim > 0 and val.shape[0] == 1:
                val = val.squeeze(0)           # ① 去掉 batch 维度
            arr = val.float().numpy()          # ② tensor → numpy
        else:
            arr = np.asarray(val)
        if arr.ndim == 0:
            arr = np.array([arr.item()], dtype=arr.dtype)  # ③ 标量→1D
        frame_state[key] = arr
    return frame_state
```

### 作用：把 VAE 编码后的 tensor state 转为 LeRobot 兼容的 numpy frame

三步操作：

| 步骤 | 代码 | 作用 |
|------|------|------|
| ① 去 batch | `squeeze(0)` | `(1, 16, 9, 28, 28)` → `(16, 9, 28, 28)` |
| ② 转格式 | `.float().numpy()` | tensor bfloat16 → numpy float32 |
| ③ 保维度 | `array([scalar])` | 标量 `0.0` → 1D 数组 `[0.0]` |

### 逐维度示例

```
VAE 编码后的 state:
  video:         (1, 16, 9, 28, 28) bfloat16
  proprio:       (1, 8) bfloat16
  future_proprio:(1, 8) bfloat16
  value_function_return: (1,) bfloat16
  ...

        │ cosmos_encoded_state_to_frame
        ▼

LeRobot frame (numpy):
  video:         (16, 9, 28, 28) float32   ← 去掉 batch 维
  proprio:       (8,) float32              ← 去掉 batch 维
  value_function_return: (1,) float32      ← 保持 1D (LeRobot 要求)
```

### 为什么 LeRobot 要求标量也是 1D 数组？

LeRobot 的特征定义要求所有 feature 至少是 1 维的：

```python
"value_function_return": {
    "dtype": "float32",
    "shape": (1,),        # 必须是 (1,)，不是 ()
    "names": None,
},
```

---

## 三、完整数据变形链

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: 构造 data_batch (data_convert/convert_raw_to_cosmos.py)  │
│                                                                    │
│  data_batch = {                                                   │
│      "video": tensor(1, 3, 33, 224, 224) uint8,                   │
│      "proprio": tensor(1, 8) bfloat16,                            │
│      "t5_text_embeddings": tensor(1, 512, 4096) bfloat16,         │
│      "fps": tensor([16]),                                         │
│      "num_conditional_frames": 4,                                 │
│      ...                                                          │
│  }                                                                │
└──────────────────────────────────────────────────────────────────┘
        │
        │ cosmos_obs_to_transition_state(data_batch)
        │   tensor → CPU tensor; ndarray → tensor; scalar → tensor
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: Transition.state                                         │
│                                                                    │
│  transition = {                                                   │
│      "state": { 全 tensor, 全 CPU, 无梯度 },                       │
│      "action": tensor(7,) float32,                                │
│      "reward": 0.0,                                               │
│      "next_state": {},                                            │
│      "done": False,                                               │
│      ...                                                          │
│  }                                                                │
└──────────────────────────────────────────────────────────────────┘
        │
        │ policy.encode_episode_state(transition_list)
        │   VAE 编码 video → latent (16, 9, 28, 28)
        │   注入 proprio → position 1, value → position 8
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: encoded transition                                       │
│                                                                    │
│  transition["state"] = {                                          │
│      "video": tensor(1, 16, 9, 28, 28) bfloat16,   ← VAE latent  │
│      "proprio": tensor(1, 8) bfloat16,              ← 已注入      │
│      "future_proprio": tensor(1, 8) bfloat16,       ← 已注入      │
│      "value_function_return": tensor(1,) bfloat16,  ← 已注入      │
│  }                                                                │
└──────────────────────────────────────────────────────────────────┘
        │
        │ cosmos_encoded_state_to_frame(transition["state"])
        │   squeeze(0) 去 batch → .numpy() 转格式 → 保维度
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4: LeRobot frame (写入数据集)                                │
│                                                                    │
│  frame = {                                                        │
│      "video":         ndarray(16, 9, 28, 28) float32,             │
│      "proprio":       ndarray(8,) float32,                        │
│      "future_proprio":ndarray(8,) float32,                        │
│      "value_function_return": ndarray(1,) float32,                │
│      "action":        ndarray(7,) float32,                        │
│      "next.reward":   ndarray(1,) float32,                        │
│      "next.done":     ndarray(1,) bool,                           │
│  }                                                                │
│                                                                    │
│  dataset.add_frame(frame) → 写入磁盘                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 四、两个函数的关键差异

| | `cosmos_obs_to_transition_state` | `cosmos_encoded_state_to_frame` |
|------|------|------|
| **调用时机** | VAE 编码前 | VAE 编码后 |
| **输入** | data_batch（杂类型） | encoded state（纯 tensor） |
| **核心操作** | 统一为 CPU tensor | tensor → numpy，去 batch |
| **batch 维度** | 保留 `(1, ...)` | 去掉 `(...,)` |
| **video 状态** | uint8 `(1,3,33,224,224)` | latent `(16,9,28,28)` |
| **输出用途** | 构造 Transition | 写入 LeRobot 数据集 |
| **浮点精度** | 保持原精度 (bfloat16) | 统一转 float32 |

---

## 五、为什么不能合并？

两个函数的职责在流水线中处在 `encode_episode_state` 的**两侧**：

```
cosmos_obs_to_transition_state → [encode_episode_state] → cosmos_encoded_state_to_frame
         (编码前统一格式)              (VAE+注入)              (编码后拆包写入)
```

- 编码前：需要保持 tensor 格式和 batch 维度，因为 VAE 期望 `(B, C, T, H, W)` 输入
- 编码后：需要去掉 batch 维度并转 numpy，因为 LeRobot 存的是单帧数据

**如果合并**：VAE 编码前后的处理逻辑混在一起，代码会变得更难维护——尤其考虑到 `encode_episode_state` 内部还做了 proprio/value 注入（`replace_latent_with_*`），前后格式要求完全不同。

---

## 六、与在线采集的一致性

两个函数都来自 `collect_data_cosmos.py`（见注释），转换脚本通过内联复制而非 import 来避免 `pynput` 键盘监听器触发 X display 错误。逻辑完全一致：

| 在线采集 (collect_data_cosmos.py) | 离线转换 (data_convert/convert_raw_to_cosmos.py) |
|------|------|
| 实时从相机获取图像 | 从 HDF5 解码图像 |
| 其余流程完全相同 | 其余流程完全相同 |
