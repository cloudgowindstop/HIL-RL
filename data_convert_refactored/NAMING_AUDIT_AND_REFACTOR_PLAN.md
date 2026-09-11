# 数据转换代码命名审查与重构计划

## 1. 审查范围

本报告审查当前活跃的数据转换实现，包括：

- `data_convert_refactored` 下的生产 Python 模块；
- 转换 CLI、Shell 脚本、工具、测试和 README；
- `data_convert` 对重构模块的兼容调用边界。

不建议修改 `lerobot` 和 `cosmos-policy` 中的上游公开 API 名称。对于 Parquet schema、Cosmos 上游函数和已经公开的 CLI 参数，应优先使用内部准确命名和兼容别名，避免直接破坏已有数据与调用命令。

本轮只记录问题和修改方案，不实施重命名。

## 2. 总体结论

当前代码的主要命名问题不是单纯的风格不统一，而是名称经常没有表达数据所处的处理阶段。例如，同一个对象可能先保存机器人控制尺度下的 action，随后又保存 dataset min/max 归一化后的 action；同一个 `video` 字段在 VAE 前表示 uint8 视频，在 VAE 后表示已注入低维条件的 latent。

建议优先明确以下状态边界：

1. 原始 HDF5 低维数据；
2. 机器人控制尺度下的 action；
3. dataset min/max 归一化后的 action/proprio；
4. VAE 编码前的 33 帧 conditioning video；
5. VAE 输出并注入 action/proprio/value 后的 conditioning latent；
6. 最终为兼容训练 schema 而写入 Parquet 的字段。

## 3. 优先级 0：数据阶段命名

### 3.1 `EncodedEpisode` 没有表达真实内容和 action 阶段

当前位置：`preparation/actions.py`

当前定义大致为：

```python
@dataclass(frozen=True)
class EncodedEpisode:
    episode: Episode
    actions: np.ndarray
    action_source: ActionSource
    action_encoding: ActionEncoding
    last_action_is_padding: bool
```

它实际保存“原始低维 Episode 加上已计算 action”，并不表示整个 episode 已完成编码。它不包含 normalized proprio、图像、VAE latent、future proprio 或 value。

此外，同一个类在不同阶段可能保存两种不同语义的 action：

- 第一层机器人控制尺度处理后的 action；
- 第二层 dataset min/max 归一化后的 action。

当前类型没有字段区分两种状态，存在重复归一化或错误反归一化风险。

建议修改为：

```python
class ActionNormalizationStage(str, Enum):
    ROBOT_SCALED = "robot_scaled"
    DATASET_NORMALIZED = "dataset_normalized"


@dataclass(frozen=True)
class EpisodeWithActions:
    episode: Episode
    actions: np.ndarray
    action_source: ActionSource
    action_encoding: ActionEncoding
    normalization_stage: ActionNormalizationStage
    last_action_is_padding: bool
```

同步重命名：

| 当前名称 | 建议名称 |
|---|---|
| `EncodedEpisode` | `EpisodeWithActions` |
| `encoded_episodes` | `episodes_with_actions` |
| `normalize_encoded_episodes` | `normalize_episode_actions` |

归一化函数应验证输入状态为 `ROBOT_SCALED`，输出状态为 `DATASET_NORMALIZED`。

### 3.2 `video` 在处理前后表示两种完全不同的数据

VAE 编码前：

```text
transition["state"]["video"]
shape = (1, 3, 33, 224, 224)
dtype = uint8
```

VAE 编码和低维条件注入后：

```text
transition["state"]["video"]
shape = (1, 16, 9, 28, 28)
dtype = float
```

建议内部使用两个明确名称：

```text
raw_conditioning_video
conditioning_latent
```

只在最终写入 LeRobot/Parquet 的边界执行：

```python
frame["video"] = conditioning_latent
```

Parquet 的 `video` 列属于训练兼容 schema，不应直接改名。

### 3.3 `build_nine_frame_sequence()` 实际生成 33 帧

当前函数返回：

```text
(1, 3, 33, 224, 224)
```

名称中的 `nine` 来自 VAE 时间压缩后的 9 个 latent 时间槽，不是该函数直接生成的帧数。

建议：

```text
build_nine_frame_sequence
-> build_33_frame_conditioning_video
```

文档明确记录：

```text
33 raw frames -> VAE temporal compression -> 9 latent slots
```

### 3.4 `value_function_return` 实际使用未来 chunk 位置的 return

当前实现为：

```python
future = min(t + chunk_size, episode_length - 1)
value_function_return = returns[future]
```

它不是 `returns[t]`。建议内部名称：

| 当前名称 | 建议名称 |
|---|---|
| `returns` | `success_conditioned_returns` |
| `transition_value_list` | `future_return_list` |
| `episode_state_value` | `episode_future_returns` |
| `terminal_reward` | `terminal_success_return` |

Parquet 的 `value_function_return` 字段需要保留，避免破坏训练代码；应在 metadata 和文档中明确其 future-chunk 语义。

## 4. 优先级 1：stats 命名

### 4.1 `PreparedDatasetStats`

该对象实际包含：

- 显式路径加载的原始 stats；
- 运行时可能经过 action 维度适配的 stats；
- 原始 stats 文件 SHA256。

建议：

| 当前名称 | 建议名称 |
|---|---|
| `PreparedDatasetStats` | `ResolvedDatasetStats` |
| `prepare_dataset_stats` | `resolve_dataset_stats` |
| `source` | `source_stats` |
| `effective` | `effective_stats` |
| `prepared_stats` | `resolved_stats` |
| 后端局部变量 `dataset_stats` | `effective_stats` |

### 4.2 `effective_official_stats()`

该函数的特殊职责是把兼容的双臂 14D Euler action stats 适配为 20D rotation-6D stats。当前名称过宽。

建议：

```text
effective_official_stats
-> adapt_official_stats_for_action_layout
```

### 4.3 `validate_external_stats()`

该函数只校验 generated stats 及其语义 sidecar，并不校验所有外部 stats。

建议：

```text
validate_external_stats
-> validate_generated_stats_and_metadata
```

## 5. 优先级 2：proprio 命名

### 5.1 `raw_proprio_for_episode()` 并不完全 raw

该函数已经执行：

- 四元数单位化；
- 相邻帧四元数连续化；
- pose 与绝对夹爪拼接；
- 单臂 8D 或双臂 16D 组合。

它只是尚未执行 dataset min/max 归一化。

建议：

| 当前名称 | 建议名称 |
|---|---|
| `raw_proprio_for_episode` | `build_pre_stats_proprio` |
| `normalized_proprio_for_episode` | `build_normalized_proprio` |

### 5.2 `_continuous_quaternions()` 返回完整 pose

该函数输入和返回完整 pose 数组，不是单独的 quaternion 数组。

建议：

```text
_continuous_quaternions
-> normalize_and_continuize_pose_quaternions
```

## 6. 优先级 3：VAE 和 Policy 命名

### 6.1 `init_cosmos_policy()`

该函数实际会：

- 创建 CosmosPolicy；
- 加载 VAE checkpoint；
- 删除 DiT 和 EMA；
- 把保留部分移动到主 GPU；
- 可选创建多 GPU VAE replica pool。

建议：

```text
init_cosmos_policy
-> load_vae_only_cosmos_policy
```

返回变量建议：

```text
cosmos_cfg
-> cosmos_world_config
```

### 6.2 `encode_episode_cpu_friendly()`

该函数不仅做 CPU 友好编码，还执行：

- action chunk 构造；
- future proprio 选择；
- future image 填充；
- future return 计算；
- VAE microbatch 编码；
- action/proprio/future proprio/value latent 注入；
- transition state 替换。

建议：

```text
encode_episode_cpu_friendly
-> encode_episode_with_vae_and_inject_conditions
```

如果希望名称更短，可使用：

```text
encode_episode_with_vae_microbatches
```

并在 docstring 中明确它还会注入低维条件。

### 6.3 `encode_world_size` 不是分布式 world size

当前实现是在一个进程中用线程驱动多张 GPU，并不是多进程分布式 world size。

建议内部改为：

```text
encode_world_size
-> vae_device_count
```

CLI 可新增：

```bash
--vae-device-count
```

旧 `--encode-world-size` 暂时保留为兼容别名。

## 7. 优先级 4：无效参数或错误语义

### 7.1 `batch_size` 当前没有实际作用

`batch_size` 从 CLI 进入 `ConversionConfig`、`CosmosRuntimeConfig`、writer 和 VAE 函数签名，但在 `encode_episode_cpu_friendly()`函数体中未使用。

建议：

1. 先用回归测试确认不同 `--batch-size` 不改变输出；
2. 删除内部参数和配置字段；
3. CLI 参数保留一个版本并打印 deprecated warning；
4. 文档明确真正控制 VAE 显存的是 `--encode-batch-size`。

### 7.2 `robot_type="single_absolute"`

该值实际描述单臂布局，但 `absolute` 与当前 action 语义无关。当前 action 可能是 `puppet_next_frame` 局部 delta。

建议内部使用：

```text
single
dual
```

为兼容旧 metadata，可同时写入：

```json
{
  "robot_layout": "single",
  "legacy_robot_type": "single_absolute"
}
```

### 7.3 `head_channel_only` 实际选择 current primary latent

工具当前选择：

```python
video[0, 3, :, :]
```

latent 时间索引 3 表示 current primary image。单臂数据中的 primary camera 不一定是 head camera。

建议：

```text
head_channel_only
-> current_primary_latent_channel_only
```

新增 CLI：

```bash
--current-primary-latent-channel-only
```

旧 `--head-channel-only` 保留为兼容别名。

## 8. 优先级 5：相机处理命名

### 8.1 `CameraState.NORMAL`

它实际选择一套图像预处理 profile，不只是描述硬件相机状态。

建议：

| 当前名称 | 建议名称 |
|---|---|
| `CameraState` | `CameraProcessingProfile` |
| `camera_state` | `camera_profile` |
| `NORMAL` | `VERIFIED_DEFAULT_V1` |
| `resolve_camera_processing` | `resolve_camera_profile` |

旧 CLI `--camera-state normal`可继续接受，metadata 新增更准确字段：

```json
"camera_processing_profile": "verified_default_v1"
```

### 8.2 `use_jpeg_compression`

该字段不控制 Parquet 或输出文件压缩，而是执行 JPEG quality=95 的重新编码模拟。

建议内部名称：

```text
simulate_jpeg_roundtrip
```

### 8.3 `trained_with_image_aug`

该字段实际控制当前转换是否执行确定性中心 crop 和 resize。

建议内部名称：

```text
apply_training_center_crop
```

### 8.4 `flip_images` 需要单独核查语义

当前代码对形状 `(N,H,W,3)` 的数组调用：

```python
np.flipud(images_array)
```

这会反转第一维的相机列表顺序，并不是上下翻转每张图。当前 transition 又硬编码传入 `flip_images=False`。

建议先确认官方配置原意：

- 如果目标是交换相机顺序，改名为 `reverse_camera_order`；
- 如果目标是上下翻转每张图，应改为空间轴翻转；
- 如果该功能永远不启用，应删除参数。

该项可能涉及行为变化，不能与纯重命名混在同一提交中。

## 9. 优先级 6：transition 和 writer 命名

建议映射：

| 当前名称 | 建议名称 |
|---|---|
| `cosmos_obs_to_transition_state` | `conditioning_batch_to_cpu_state` |
| `select_cameras` | `resolve_camera_roles` |
| `build_transition_list_for_episode` | `build_conditioning_transitions` |
| `data_batch` | `conditioning_batch` |
| `transition_video_list` | `raw_conditioning_videos` |
| `episode_state_action` | `episode_action_chunks` |
| `latent_state` | `episode_conditioning_latents` |
| `cosmos_encoded_state_to_frame` | `latent_state_to_lerobot_fields` |
| `build_output_features` | `build_lerobot_parquet_features` |
| `encoded` | `latent_transitions` |
| `save_conversion_metadata` | `save_dataset_stats_and_metadata` |

`dataset_writer.py` 当前还负责 feature schema、VAE调用、Parquet写入、stats和metadata。按照此前约束不拆分该文件；如需要进一步提高名称准确性，可只把模块改名为：

```text
lerobot_output.py
```

## 10. 优先级 7：Pipeline 命名

建议映射：

| 当前名称 | 建议名称 |
|---|---|
| `prepare()` | `load_and_preflight_episodes()` |
| `create_action_encoder()` | `build_action_encoder()` |
| `inspect_actions()` | `summarize_actions()` |
| `encode_actions()` | `build_episode_actions()` |
| `prepared_stats` | `resolved_stats` |
| `stats` | `effective_stats` |

`ConversionPipeline`名称准确，可保留。

`CosmosBackend`建议改为：

```text
CosmosConversionBackend
```

因为它是数据转换后端，不是通用 Cosmos runtime backend。

## 11. 优先级 8：目录和工具命名

### 11.1 `data_convert_refactored`

该目录已经是正式实现，`refactored`仍像迁移中的临时目录。

最终理想结构：

```text
data_convert/          正式实现
data_convert_legacy/   旧代码和兼容入口
```

迁移期间，原 `data_convert/convert_raw_to_cosmos.py`可保留为 deprecated wrapper，不应立即删除。

### 11.2 两个比较工具名称重叠

当前：

```text
compare_conversion_outputs.py
compare_parquet.py
```

建议：

```text
compare_lerobot_datasets.py
compare_parquet_files.py
```

数据集比较器应复用单文件按 batch 比较实现，避免整列加载大型 latent。

### 11.3 `decode_parquet_action_6d.py`

当前只支持单臂 10D rotation-6D action，但名称没有说明单臂限制。

两种处理方案：

- 改名为 `decode_single_arm_parquet_action_6d.py`；
- 或扩展支持双臂 20D 后保留当前名称。

## 12. 不应直接改名的兼容字段

以下名称受训练代码或上游 API 约束：

### 12.1 Parquet schema

```text
video
action
proprio
future_proprio
value_function_return
next.reward
next.done
```

### 12.2 Cosmos 上游函数

```text
rescale_action
rescale_proprio
get_action_chunk_with_padding
replace_latent_with_action_chunk
replace_latent_with_proprio
```

正确做法是内部使用准确名称，在最终边界映射到兼容字段。

## 13. 推荐实施顺序

### 阶段 1：纯内部重命名

优先修改：

```text
EncodedEpisode -> EpisodeWithActions
增加 normalization_stage
normalize_encoded_episodes -> normalize_episode_actions
PreparedDatasetStats -> ResolvedDatasetStats
prepare_dataset_stats -> resolve_dataset_stats
raw_proprio_for_episode -> build_pre_stats_proprio
```

同步修改类型标注、测试和 README。此阶段不得改变任何数值公式或输出 schema。

### 阶段 2：明确 raw video 和 latent 边界

```text
raw_conditioning_video
conditioning_latent
build_33_frame_conditioning_video
future_return相关内部变量
```

Parquet字段保持不变。

### 阶段 3：清理死参数和错误语义

```text
删除无效 batch_size
核查 flip_images
single_absolute 改为 single
camera state 改为 processing profile
```

该阶段可能涉及兼容逻辑，应单独提交。

### 阶段 4：公共名称兼容迁移

- 新增准确 CLI 名称；
- 旧 CLI 保留为 deprecated alias；
- metadata在过渡期同时保存新旧字段；
- 旧 Python import 保留薄别名一个版本。

### 阶段 5：目录最终迁移

```text
data_convert_refactored -> data_convert
旧实现 -> data_convert_legacy
```

在完成调用方、脚本和文档迁移前，不应执行该阶段。

## 14. 每阶段验证要求

每个阶段完成后至少执行：

1. 全部单元测试；
2. 真实 HDF5 `--preflight-only`；
3. 单 episode 正式 VAE 转换；
4. 修改前后 Parquet 逐列比较；
5. current primary VAE latent 严格比较；
6. action 反解 NPY 比较；
7. 单 GPU 与多 GPU 结果比较；
8. metadata 和 stats SHA256 检查。

对于纯命名阶段，验收标准应为：

```text
Parquet逐元素一致
action反解一致
VAE latent一致
reward/done/value一致
metadata中兼容字段不丢失
```

## 15. 首轮建议修改范围

第一轮只建议处理以下五项：

1. `EpisodeWithActions`及`normalization_stage`；
2. `normalize_episode_actions`；
3. `ResolvedDatasetStats`；
4. `build_pre_stats_proprio`；
5. `build_33_frame_conditioning_video`。

这五项能最大幅度降低误用风险，同时保持修改范围和回归风险可控。完成并验证后，再进入 raw video/latent 命名分离和无效参数清理。
