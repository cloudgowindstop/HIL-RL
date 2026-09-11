# Raw HDF5 → Cosmos 数据转换详解

## 一、整体流程

```
raw_data/{robot_type}/{task_name}/success_episodes/
    ├── {ts_1}/data/trajectory.hdf5    ← episode 0
    ├── {ts_2}/data/trajectory.hdf5    ← episode 1
    └── ...
        │
        ▼  data_convert/convert_raw_to_cosmos.py
        │
        ▼
dataset_cosmos/{task_name}_cosmos/
    ├── meta/info.json, stats.json, ...
    ├── data/chunk-000/
    └── LeRobot dataset (video 已 VAE 编码为 latent)
```

## 二、原始数据结构 (raw HDF5)

每个 `trajectory.hdf5` 包含一个 episode 的完整数据 (T 帧):

```
trajectory.hdf5
├── puppet/
│   ├── end_effector_left_pose_align/data      (T, 7)  float32  左末端位姿 xyz+qxyzw
│   ├── end_effector_right_pose_align/data     (T, 7)  float32  右末端位姿
│   ├── end_effector_left_position_align/data  (T, 1)  float32  左夹爪 [0,1]
│   ├── end_effector_right_position_align/data (T, 1)  float32  右夹爪 [0,1]
│   ├── arm_left_position_align/data           (T, 7)  float32  左关节角 (备用)
│   └── arm_right_position_align/data          (T, 7)  float32  右关节角 (备用)
├── camera_observations/color_images/
│   ├── camera_head    (T,) JPEG bytes  640×480  → primary
│   ├── camera_left    (T,) JPEG bytes  640×480  → wrist1 (左腕)
│   └── camera_right   (T,) JPEG bytes  640×480  → wrist2 (右腕)
├── metadata/
│   ├── language_instruction  str  任务描述
│   └── trajectory_length     int  帧数
└── camera_color_resolution/
    ├── camera_head:  [640, 480]
    ├── camera_left:  [640, 480]
    └── camera_right: [640, 480]
```

## 三、转换详程

### 3.1 图像处理

#### JPEG 解码

```python
jpeg_bytes = f["camera_observations/color_images/camera_head"][t]
img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (480, 640, 3) uint8
```

#### 预处理（复用 Cosmos 工具）

```python
from cosmos_policy.experiments.robot.cosmos_utils import prepare_images_for_model

all_images = prepare_images_for_model(
    [img_wrist, img_primary],  # 单臂: [wrist, primary]
                               # 双臂: [wrist_fused, primary]
    cfg
)
# 内部: resize(224,224), JPEG压缩模拟, 90%中心裁剪
```

#### 腕部处理规则

| 机器人 | 腕部相机 | 处理 |
|------|---------|------|
| 单臂 (UR, Franka) | camera_left 或 camera_right | **直接使用**，选一个即可 |
| 双臂 (Tienyi) | camera_left + camera_right | **上下拼接** → resize 224×224 |

```python
if is_dual_arm:
    # 双臂：上下拼接
    left  = cv2.resize(img_left,  (224, 224))
    right = cv2.resize(img_right, (224, 224))
    wrist = np.concatenate([left, right], axis=0)   # (448, 224, 3)
    wrist = cv2.resize(wrist, (224, 224))            # (224, 224, 3)
else:
    # 单臂：选一个腕部直接用
    wrist = cv2.resize(img_left, (224, 224))         # (224, 224, 3)
```

#### 9 帧序列构造（和 CosmosWrapper 完全一致）

```python
blank = np.zeros((224, 224, 3), dtype=np.uint8)
blank_dup = duplicate_array(blank, 4)        # (4, 224, 224, 3)
wrist_dup = duplicate_array(wrist, 4)        # (4, 224, 224, 3)
primary_dup = duplicate_array(primary, 4)    # (4, 224, 224, 3)

image_sequence = [
    np.expand_dims(blank, 0),    # f0:  blank (1帧, VAE placeholder)
    blank_dup,                   # f1-f4: proprio 占位
    wrist_dup,                   # f5-f8: wrist 真实图像 ★
    primary_dup,                 # f9-f12: primary 真实图像 ★
    blank_dup.copy(),            # f13-f16: action 占位
    blank_dup.copy(),            # f17-f20: future_proprio 占位
    wrist_dup.copy(),            # f21-f24: future_wrist 占位
    primary_dup.copy(),          # f25-f28: future_primary 占位
    blank_dup.copy(),            # f29-f32: value 占位
]

# 拼接: 33帧 → (1, 3, 33, 224, 224) uint8 tensor
raw = np.concatenate(image_sequence, axis=0)
raw = np.expand_dims(raw, 0)
raw = np.transpose(raw, (0, 4, 1, 2, 3))
video_tensor = torch.from_numpy(raw)   # uint8
```

### 3.2 动作转换：绝对位姿 → 增量

#### 输入

```
puppet/end_effector_left_pose_align/data[t]    → pose_curr  (7,) xyz+qxyzw
puppet/end_effector_left_pose_align/data[t+16] → pose_future (7,)
puppet/end_effector_left_position_align/data[t] → grip_curr  (1,)
puppet/end_effector_left_position_align/data[t+16] → grip_future (1,)
```

#### 计算

```python
from scipy.spatial.transform import Rotation

def pose_7d_to_matrix(pose):
    """xyz(3) + quat(4) → 4×4 齐次矩阵"""
    xyz, quat = pose[:3], pose[3:7]
    R = Rotation.from_quat(quat).as_matrix()    # 四元数 → 3×3 旋转矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

def compute_delta_action(pose_curr, pose_future, grip_curr, grip_future, action_scale):
    """绝对位姿 → 增量末端动作 (dx,dy,dz, rx,ry,rz, grip)"""
    T_curr = pose_7d_to_matrix(pose_curr)        # 当前末端 4×4
    T_future = pose_7d_to_matrix(pose_future)    # 未来末端 4×4
    T_delta = np.linalg.inv(T_curr) @ T_future  # 相对变换

    dpos = T_delta[:3, 3]                                      # (3,) 平移
    drot = Rotation.from_matrix(T_delta[:3, :3]).as_euler("xyz")  # (3,) 欧拉角
    dgrip = grip_future - grip_curr                            # (1,) 夹爪

    action = np.concatenate([
        dpos / action_scale[0],       # dx, dy, dz
        drot / action_scale[1],       # rx, ry, rz
        dgrip / action_scale[2],      # grip
    ])  # (7,) ∈ [-1, +1]
    return action
```

#### 双臂拼接

```python
action_left  = compute_delta_action(pose_left_curr, pose_left_future, grip_left, scale)   # (7,)
action_right = compute_delta_action(pose_right_curr, pose_right_future, grip_right, scale) # (7,)
action = np.concatenate([action_left, action_right])  # (14,)
```

#### action_scale 计算

从全部 episode 统计相邻帧差分，取绝对最大值：

```python
# 对第 0 帧到第 T-chunk_size 帧，统计 delta 分布
deltas = []
for t in range(T - chunk_size):
    dpos = pose[t+chunk_size][:3] - pose[t][:3]  # or via inv(T_curr) @ T_next
    deltas.append(dpos)
deltas = np.abs(np.array(deltas))

action_scale[0] = np.percentile(deltas[:, :3], 99)    # 平移
action_scale[1] = np.percentile(deltas[:, 3:6], 99)   # 旋转  
action_scale[2] = np.percentile(np.abs(grip_diffs), 99)  # 夹爪
```

### 3.3 Proprio 构造

proprio 是**当前帧的绝对末端位姿 + 夹爪**，不做差分。

```python
def build_proprio(row_t, dataset_stats):  # row_t = 第 t 帧的 HDF5 数据
    if is_dual_arm:
        raw = np.concatenate([
            row_t["puppet/end_effector_left_pose_align/data"],     # (7,)
            [row_t["puppet/end_effector_left_position_align/data"]], # (1,)
            row_t["puppet/end_effector_right_pose_align/data"],    # (7,)
            [row_t["puppet/end_effector_right_position_align/data"]],# (1,)
        ])  # (16,)
    else:
        raw = np.concatenate([
            row_t["puppet/end_effector_left_pose_align/data"],     # (7,)
            [row_t["puppet/end_effector_left_position_align/data"]], # (1,)
        ])  # (8,)

    # 归一化到 [-1, +1]
    proprio = rescale_proprio(raw, dataset_stats, non_negative_only=False)
    return proprio  # (16,) or (8,)
```

### 3.4 data_batch 组装

```python
data_batch = {
    "video":              video_tensor,            # (1, 3, 33, 224, 224) uint8
    "proprio":            proprio_tensor,          # (1, D) bfloat16
    "t5_text_embeddings": t5_embedding,            # (1, 512, 4096) bfloat16
    "fps":                torch.tensor([16]),
    "padding_mask":       torch.zeros(1, 1, 224, 224),
    "num_conditional_frames": 4,
    # Latent 索引 (9帧布局，和 CosmosWrapper 一致)
    "current_proprio_latent_idx":      torch.tensor([1]),
    "current_wrist_image_latent_idx":  torch.tensor([2]),
    "current_image_latent_idx":        torch.tensor([3]),
    "action_latent_idx":               torch.tensor([4]),
    "future_proprio_latent_idx":       torch.tensor([5]),
    "future_wrist_image_latent_idx":   torch.tensor([6]),
    "future_image_latent_idx":         torch.tensor([7]),
    "value_latent_idx":                torch.tensor([8]),
}
```

### 3.5 VAE 编码 + 潜帧注入（复用）

```python
# 和 collect_data_cosmos.py 完全一致
from collect_data_cosmos import cosmos_obs_to_transition_state

state = cosmos_obs_to_transition_state(data_batch)  # 只保留 tensor

transition_list.append(Transition(
    state=state,
    action=torch.tensor(action),     # (14,) or (7,)
    reward=0.0,
    done=(t == T - 1),
    complementary_info={"is_intervention": True},
))

# Episode 结束时:
transition_list = policy.encode_episode_state(
    transition_list, encode_batch_size, device, cosmos_cfg, batch_size
)
# 此时 transition.state["video"] → (1, 16, 9, 28, 28) latent
# action/proprio/value 已注入
```

### 3.6 写入 LeRobot 数据集（复用）

```python
from collect_data_cosmos import cosmos_encoded_state_to_frame

for transition in transition_list:
    state = cosmos_encoded_state_to_frame(transition["state"])
    dataset.add_frame({
        **state,                              # video, proprio, future_proprio, value
        "action": transition["action"].numpy(),
        "next.reward": np.float32(transition["reward"]),
        "next.done": transition["done"],
        "complementary_info.is_intervention": True,
    }, task=task_instruction)

dataset.save_episode()
```

## 四、单臂与双臂的差异总结

| | 单臂 (UR/Franka) | 双臂 (Tienyi) |
|------|:---:|:---:|
| EE pose 读取 | 左 or 右 (7D) | 左 + 右 (7D × 2) |
| 腕部图像 | 直接使用 (1 个) | 上下拼接后 resize (2 个) |
| action 维度 | 7D | 14D |
| proprio 维度 | 8D | 16D |
| 潜帧布局 | 9 帧 | 9 帧 (相同!) |
| action_scale | (3,) | (3,) 双臂共用 |
| dataset_stats | 按臂统计 | 按双臂统计 |

## 五、关键不变的部分

- **CosmosWrapper 逻辑不动** — data_batch 构造在脚本内实现
- **LATENT_INDICES 不动** — 9 帧布局不变
- **encode_episode_state 不动** — 直接调用
- **replace_latent_with_* 不动** — encode_episode_state 内部调用
- **LeRobot 输出格式不动** — 和实机采集完全一致
