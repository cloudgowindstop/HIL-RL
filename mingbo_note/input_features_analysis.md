# input_features 辨析：需要包含哪些字段

## 结论

`input_features` 只需要三项，不需要 `future_proprio` 和 `value_function_return`。

```python
input_features={
    "video":              PolicyFeature(type=FeatureType.VISUAL, shape=(3, 33, 224, 224)),
    "proprio":            PolicyFeature(type=FeatureType.STATE,  shape=(proprio_dim,)),
    "t5_text_embeddings": PolicyFeature(type=FeatureType.ENV,    shape=(512, 1024)),
},
output_features={
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
},
```

---

## 为什么不需要？

### 1. 概念区分：LeRobot 存储字段 ≠ 模型 I/O

```
LeRobot 数据集 (磁盘存储)          模型 I/O (CosmosConfig)
┌──────────────────────────┐     ┌──────────────────────────┐
│ video           ← 存储   │     │ video           ← 输入   │
│ proprio         ← 存储   │     │ proprio         ← 输入   │
│ future_proprio  ← 存储   │     │ t5_text         ← 输入   │
│ value_return    ← 存储   │     │                          │
│ action          ← 存储   │     │ action           ← 输出  │
└──────────────────────────┘     └──────────────────────────┘
         ↑                                  ↑
  训练时加载 ground truth              模型对外的 I/O 边界
  注入到 latent 作为去噪目标
```

`future_proprio` 和 `value_function_return` 需要**存在数据集中**（训练 Dataset 加载它们当 ground truth），但**不是模型对外的输入**——它们通过 `replace_latent_with_*` 注入到 9 帧 latent 序列内部。

### 2. 9 帧 latent 序列结构

模型对外的 "video" 输入是一个完整的 9 帧 latent 序列，内部已包含所有信息：

```
位置 0: 空白占位
位置 1: proprio         ← 来自 input_features.proprio
位置 2: wrist_img       ← VAE 编码
位置 3: primary_img     ← VAE 编码
位置 4: action          ← output, 训练时作为目标
位置 5: future_proprio  ← 训练目标，不在 input_features 中
位置 6: future_wrist    ← VAE 编码
位置 7: future_primary  ← VAE 编码
位置 8: value           ← 训练目标，不在 input_features 中
```

`future_proprio` 和 `value` 是**扩散去噪的目标帧**，在 9 帧序列内部，不是从外部传入的独立输入。

### 3. 训练时的角色

| 字段 | 训练时的角色 | 在 input_features 中 |
|------|------|:---:|
| `video` | 完整 9 帧 latent，加噪→去噪 | ✅ |
| `proprio` | 当前位置，注入位置 1 | ✅ |
| `t5_text_embeddings` | 交叉注意力条件 | ✅ |
| `action` | 预测目标，注入位置 4 | ❌ (是 output) |
| `future_proprio` | 预测目标，注入位置 5 | ❌ |
| `value_function_return` | 预测目标，注入位置 8 | ❌ |

---

## 证据来源

### 在线采集（collect_data_cosmos.py）

```python
# 数据集 features（存储层面）— 包含 future_proprio 和 value
features["video"]                    = {...}   # 存
features["proprio"]                  = {...}   # 存
features["future_proprio"]           = {...}   # 存 ←
features["value_function_return"]    = {...}   # 存 ←

# 模型 input_features（模型 I/O 层面）— 不包含
policy_cfg.policy.input_features  # 只有 video, proprio, t5_text
```

### 架构原理（Cosmos Policy 论文 + 代码）

```
replace_latent_with_proprio()   → 注入 proprio 到 latent 位置 1/5
replace_latent_with_action_chunk() → 注入 action 到 latent 位置 4
value 直接赋值                 → 注入 value 到 latent 位置 8
```

所有非图像信息都通过 **latent frame injection** 嵌入到 video 序列内部，不经过模型的对外 `input_features` 接口。

---

## 和转换脚本的关系

转换脚本 `build_output_features` 定义的是 **LeRobot 存储格式**，需要包含全部字段以供训练时加载。`init_cosmos_policy` 中的 `input_features` 定义的是 **模型初始化接口**，不需要声明注入到 latent 内部的帧。两者是不同层面的概念，不矛盾。
