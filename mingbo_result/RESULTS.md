# Cosmos 离线中训练：最终实验结果与结论

本文汇总后把手双臂安装任务上四条已完成的 2000-step 实验，回答两个问题：

1. Euler 14D 与 rotation-6D 20D，哪种动作表示在当前数据和预算下更有效？
2. Policy-only 与 Policy/World/Value joint training，哪种训练模式更有效？

Inverse Dynamics（ID）训练仍在进行，本报告不使用未完成 checkpoint，不给出 ID 有效性结论。

## 项目亮点与工程贡献

本项目的贡献不是提出新的基础模型，而是将 Cosmos 视频生成模型系统化改造成可训练、可扩展、可评测的机器人离线中训练框架：

- **端到端离线闭环**：打通 HDF5 机器人轨迹、Cosmos latent 数据、DDP 训练、checkpoint 管理以及离线评测，避免依赖在线 ReplayBuffer 和实机 rollout。
- **统一多目标训练**：用 temporal conditioning/target mask 在同一个 2B 模型中支持 Policy、World、Value 和 Inverse Dynamics，无需为每种任务维护独立网络。
- **动作表示可审计对比**：同时支持 Euler 14D 与 rotation-6D 20D，并固定数据划分、基础权重、有效 batch 和训练步数，得到可解释的受控实验。
- **可扩展数据管线**：使用 Lazy Parquet、row-group LRU、`DistributedSampler` 和每 rank 独立读取，消除 rank 0 完整加载与 Python object scatter 的内存、启动和通信瓶颈。
- **分层效果验证**：除了训练 loss，还从 action、latent、Value 排序和解码 RGB 四层评测，包含旋转测地角、Spearman、PSNR/SSIM与样例可视化，避免仅凭单一 loss 判断中训练有效性。
- **可复现与可观测性**：记录 resolved config、数据/权重身份、固定噪声验证、分目标 loss、checkpoint 对比，并接入 Prometheus/WandB，便于复现和定位训练异常。

由此得到的核心实验发现是：**动作表示存在任务相关权衡，多目标训练也存在有限预算下的目标竞争**。Euler 在困难手臂的动作回归上更好，rotation-6D 在未来 RGB 预测上更好；Joint 改善 Value 排序，但在相同总步数下没有超过 Policy-only 的动作与视觉表现。

## 1. 执行摘要

最稳妥的结论是：

> **动作表示没有单一赢家。** Euler Policy-only 在困难的左臂平移预测上最好；rotation-6D Policy-only 在给定 GT action 的未来 RGB 预测上最好。相同 2000-step、不到一个 epoch 的预算下，Joint 没有超过同表示的 Policy-only，但 Joint 的 Value 进度排序更好。

关键数字：

- 左臂 translation MAE：E-P `0.0281`，6D-P `0.0494`；E-P 低 43.1%。
- 右臂 translation MAE：E-P `0.0162`，6D-P `0.0154`；基本持平。
- future-camera PSNR：E-P `19.42 dB`，6D-P `19.82 dB`；6D-P 最好。
- Predict2 base 的 future-camera PSNR 约 `7.4 dB`，四个微调模型均提升到 `18.8–19.8 dB`。
- Joint 相对 Policy-only：Euler 左臂误差高 25.0%，6D 左臂误差高 6.7%。
- Value Spearman：E-J `0.879`、6D-J `0.866`，均高于同表示 Policy-only；但 6D-J 的 Value MAE 最差。

这些结论只适用于当前任务、成功演示、seed 42、2000-step 和开环离线评测。

## 2. 实验矩阵与公平性

| Run | 动作表示 | 训练模式 | 条件任务概率 | 状态 |
| --- | --- | --- | --- | --- |
| E-P | Euler 14D | `policy_only` | P/W/V = 100/0/0 | 完成 2000 |
| E-J | Euler 14D | `base_joint` | P/W/V = 50/25/25 | 完成 2000 |
| 6D-P | rotation-6D 20D | `policy_only` | P/W/V = 100/0/0 | 完成 2000 |
| 6D-J | rotation-6D 20D | `base_joint` | P/W/V = 50/25/25 | 完成 2000 |

共同条件：

- 同一批 221 个成功 episode：177 train / 44 validation，无 test split。
- train 274,870 rows，validation 67,340 rows。
- 同一 raw episode split、任务文本/T5、seed 42 和基础权重。
- 基础权重 SHA256：`fbc4f05d...e807a6f0`，对应 Cosmos-Predict2-2B-Video2World；目录名中的 `LIBERO` 不代表本次从 LIBERO policy checkpoint 起步。
- 4 卡 DDP，micro-batch `2/rank`，梯度累积 16，有效 batch 128。
- 2000 optimizer steps，共 256,000 row-samples，约 0.93 epoch。
- BF16、基础学习率 `1e-4`、每 500 steps 保存和验证。
- 四条 run 均从基础权重重新开始，不是从旧 1000-step run resume。

唯一主动变化的是动作表示和训练模式。Euler/6D 使用各自匹配的 stats，但 source episode 与 row 数一致。

## 3. 训练目标：本报告中的 Policy、World、Value

三种模式共享同一个 2B diffusion transformer，只改变干净条件和计算 loss 的 latent 位置：

- Policy：当前状态 → action chunk + future state + value，即 `p(a,s',V|s)`。
- World/FD：当前状态 + GT action → future state + value，即 `p(s',V|s,a)`。
- Value：当前状态 + GT action + future state → value，即 `p(V|s,a,s')`。

因此 `policy_only` 并不是“只预测 action”：所有 Policy 样本也学习 future proprio、future images 和 value。Joint 的差别是只有约 50% 样本走 Policy conditioning，其余样本将 action 或 future state作为条件。

本实验使用 Cosmos Policy 的 EDM 去噪 loss。它不是普通 raw MSE，也不是 Cosmos 3 的 rectified-flow loss；本次没有采用 Cosmos 3 action `10×` 权重。详细公式见 [CODE.md](CODE.md#5-实际训练-loss)。

## 4. 评测协议与指标含义

### 4.1 固定噪声 latent probe

step 500/1000/1500/2000 使用固定 `sigma=0.5`、固定 noise seed。周期验证强制 Policy objective；step 2000 结束时另跑 Policy/World/Value fixed suite。

每个 condition 最多 40 batches/rank，即四卡约 320 row-samples。它是固定、可重复的 validation probe，**不是完整 67,340-row validation split**。`sigma=0.5` 是 EDM 尺度上的低噪声单步去噪，不等同于从纯噪声开始的完整推理。

### 4.2 Action 指标

预测 action chunk 和 GT 撤销 dataset min-max 后计算：

- translation MAE：`mean(|delta_xyz_pred-delta_xyz_gt|)`。
- rotation geodesic：`acos((tr(R_pred^T R_gt)-1)/2)`，单位度。
- gripper accuracy：按 stats 中点二值化后比较开/合。
- horizon-16：action chunk 第 16 个槽位，不是闭环执行 16 步。

translation 数值仍处于 `delta_xyz/0.02` 的控制尺度，没有乘回米；例如 `0.0281` 若仅作线性换算约为 `0.000562 m`，但存在转换时 clipping，所以主表保留控制尺度。14D/20D raw MSE 不可直接比较。

### 4.3 World 指标

- latent L1：固定噪声下一步去噪后的 future proprio、wrist/primary latent 误差。
- RGB PSNR：`10 log10(1/MSE)`，越高越好。
- RGB SSIM：比较局部亮度、对比度和结构，越高越好。

RGB 使用独立的 10-step generation：固定 8 validation episodes × 2 rows，共 16 样本，给定当前观测和 GT action。它是 open-loop forward dynamics，不是策略闭环成功率。

### 4.4 Value 指标

- scalar MAE：数值标定误差。
- Spearman：预测与 GT return 的秩相关，衡量进度排序；单独复评 `n=80`。

训练只有成功轨迹，return 随 episode 进度变化。因此 Spearman 可以评价“是否将更接近完成的状态排得更高”，但不能评价成功/失败分类，不能报告 AUROC。

## 5. H1：Euler 14D vs rotation-6D 20D

只比较相同 Policy-only 模式的 E-P 与 6D-P。

![E-P 与 6D-P 平移对比](figures/compare/fig1_translation_ep_vs_6dp.png)

| 指标 | E-P | 6D-P | 结论 |
| --- | ---: | ---: | --- |
| 左臂 translation MAE | **0.0281** | 0.0494 | E-P 低 43.1% |
| 右臂 translation MAE | 0.0162 | **0.0154** | 接近，6D-P 略低 |
| 左 gripper accuracy | 0.963 | **0.966** | 接近饱和 |
| 右 gripper accuracy | 1.000 | 1.000 | 无区分度 |
| 左 horizon-16 translation | **0.0353** | 0.0547 | E-P 更好 |
| 右 horizon-16 translation | **0.0161** | 0.0167 | 基本持平 |

后把手安装中左臂承担更精细的对准，差异主要集中在左臂；右臂和夹爪已接近饱和。当前证据支持“同预算下 Euler 更容易学好困难侧平移”，但不足以证明 6D 表示本身更差：6D 目标维度更高、曲线仍在下降，而且它在未来 RGB 上反而占优。

Euler 与多数 6D run 在 step 2000 的测地角显示为 `0.000°`。这表示当前计算精度下不可分辨，并受到小幅增量动作和数值投影影响，不能解释为旋转预测完美。

## 6. H2：Policy-only vs Joint

![左右臂平移](figures/compare/fig2_translation_2x2.png)

![horizon-16 平移](figures/compare/fig3_h16_translation_2x2.png)

![夹爪准确率](figures/compare/fig4_gripper_2x2.png)

| 指标 | E-P | E-J | 6D-P | 6D-J |
| --- | ---: | ---: | ---: | ---: |
| 左 translation MAE | **0.0281** | 0.0351 | 0.0494 | 0.0527 |
| 右 translation MAE | 0.0162 | 0.0168 | **0.0154** | 0.0170 |
| 左 gripper accuracy | 0.963 | **0.968** | 0.966 | 0.927 |
| 右 gripper accuracy | 1.000 | 1.000 | 1.000 | 1.000 |
| 左 horizon-16 translation | **0.0353** | 0.0456 | 0.0547 | 0.0573 |
| 右 horizon-16 translation | **0.0161** | 0.0162 | 0.0167 | 0.0180 |

在每种 encoding 内，Joint 均未改善主要动作指标：

- E-J 左臂误差比 E-P 高 25.0%。
- 6D-J 左臂误差比 6D-P 高 6.7%，左夹爪 accuracy 从 `0.966` 降至 `0.927`。

最直接的机制解释是监督预算稀释：P 模式的 256,000 个样本全部训练 `p(a,s',V|s)`；Joint 期望只有约 128,000 个样本使用 Policy conditioning。不到一个 epoch 时，Joint 尚未获得与 P 相同数量的动作条件监督。

这是合理解释，不是已经被单独验证的因果结论。要区分“混合任务有干扰”和“Policy 样本数少”，必须增加等 Policy-sample 控制实验。

## 7. World latent 与 Value

![World latent L1](figures/compare/fig5_world_l1_2x2.png)

![Value MAE](figures/compare/fig6_value_2x2.png)

![Value Spearman](figures/compare/fig9_value_spearman.png)

| 指标 | E-P | E-J | 6D-P | 6D-J | 越好 |
| --- | ---: | ---: | ---: | ---: | --- |
| future primary latent L1 | **0.0775** | 0.0873 | 0.0780 | 0.0790 | 低 |
| future wrist latent L1 | **0.1001** | 0.1122 | 0.1054 | 0.1010 | 低 |
| future proprio L1 | 0.0453 | 0.0437 | **0.0329** | 0.0393 | 低 |
| Value scalar MAE | 0.0041 | 0.0035 | **0.0028** | 0.0094 | 低 |
| Value Spearman (`n=80`) | 0.789 | **0.879** | 0.816 | 0.866 | 高 |

Joint 没有稳定改善 World latent。Value 则出现“排序”和“标定”分离：

- E-J/6D-J 的 Spearman 均高于对应 P，说明 Joint 更善于排列任务进度。
- 6D-J 的 scalar MAE 最差，说明排序正确不代表绝对值准确。
- Spearman 只有 80 个样本且无 bootstrap CI，只能作为方向性证据。

另外，表中 Value MAE 来自 step-2000 fixed suite；Spearman 表对应的 value-only 复评也有一套同分布 MAE。两次评测用途不同，不应把不同表中的 MAE 混为同一统计量。

## 8. World RGB：中训练带来的最明显收益

协议：当前观测 + GT action → 未来 wrist/primary RGB；固定 16 个 validation rows、noise seed `20260820`、10 denoising steps、guidance 1.0。

![未来相机 PSNR](figures/compare/fig8_future_cameras_psnr.png)

![分相机 PSNR](figures/compare/fig7_world_psnr_2x2.png)

| 指标 | Euler-base | 6D-base | E-P | E-J | 6D-P | 6D-J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| future cameras PSNR | 7.40 | 7.38 | 19.42 | 18.82 | **19.82** | 19.33 |
| primary PSNR | 6.93 | 6.86 | 20.62 | 20.07 | **21.37** | 20.80 |
| wrist PSNR | 7.88 | 7.91 | 18.22 | 17.57 | **18.27** | 17.86 |
| future cameras SSIM | 0.198 | 0.196 | 0.726 | 0.680 | **0.735** | 0.717 |

四个微调模型都显著超过基础模型，future-camera PSNR 绝对提升约 `11.4–12.4 dB`。排序为：

> **6D-P > E-P > 6D-J > E-J**

这说明机器人数据中训练确实让基础视频模型学会了当前场景中的动作条件未来预测。6D-P 的 primary PSNR `21.37 dB` 和综合 PSNR `19.82 dB` 均最高，是 rotation-6D 在本次实验中的主要优势。

latent L1 与 RGB PSNR 不完全同序：E-P 的 primary latent L1 略低于 6D-P，但 VAE decode 后 6D-P 的 RGB 更好。因此不能用 latent L1 替代像素指标。

同一 validation 样本的定性图中，左列 GT、中列 prediction、右列归一化 absolute error；上行为 wrist，下行为 primary：

| Base | E-P | 6D-P |
| --- | --- | --- |
| ![Euler base](figures/rgb_examples/euler_base_0804_am_success_0804_095314_row750.png) | ![Euler policy](figures/rgb_examples/euler_14d_policy_2000_0804_am_success_0804_095314_row750.png) | ![6D policy](figures/rgb_examples/rotation6d_20d_policy_2000_0804_am_success_0804_095314_row750.png) |

微调后已能恢复台面、机械臂和主要物体轮廓；边缘、接触区域和细小结构仍是主要误差来源。单帧只能用于解释，排名应以固定样本均值为准。

## 9. 训练过程与收敛状态

![左臂验证曲线](figures/compare/fig10_left_arm_curve.png)

| step | E-P | E-J | 6D-P | 6D-J |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 0.0375 | 0.0435 | 0.0694 | 0.0925 |
| 1000 | 0.0379 | 0.0416 | 0.0561 | 0.0580 |
| 1500 | 0.0354 | 0.0383 | 0.0540 | 0.0728 |
| 2000 | **0.0281** | 0.0351 | 0.0494 | 0.0527 |

E-P 和 6D-P 到 step 2000 仍在改善；6D-J 在 step 1500 有明显验证波动，随后恢复。所有主 run 的训练 EDM 在早期快速下降，之后在随机 sigma 和随机 batch 下波动，未出现 NaN、梯度爆炸或持续发散。

因此 2000-step 是有效的受限预算对比，不是收敛实验。H1 的差距可能随更长训练变化，H2 也可能因 Joint 获得更多 Policy 样本而变化。

### 作废的旧 6D-P@1000

![旧新 6D-P 对比](figures/compare/fig11_old_vs_new_left_trans.png)

旧 6D-P@1000 在训练末期异常，左臂 translation MAE 为 `0.1153`、RGB PSNR 为 `17.38`。新的从头 run 在 step 1000 已达到 `0.0561`，step 2000 为 `0.0494`。旧结果不能作为 6D baseline，也不能用于声称“Joint 将 6D 救活”。

## 10. 三个研究问题的当前回答

### H1：6D 是否优于 Euler？

不能给出整体“是/否”。当前任务中：

- Policy 动作：E-P 的困难侧平移更准。
- Forward Dynamics RGB：6D-P 更准。
- 右臂和夹爪：差异很小或已饱和。

所以应按使用目标选择指标，而不是比较 14D/20D raw loss。

### H2：Joint 是否优于 Policy-only？

当前 2000-step 配方下没有。Policy-only 在动作、World latent 和 RGB 上总体更好；Joint 只在 Value 排序上显示优势。由于 Joint 的 Policy 样本预算约减半，这不能外推成“多目标训练永远有害”。

### H3：Inverse Dynamics 是否有效？

尚未回答。代码和消融协议已经具备，但 ID run 尚未完成并做同预算终评，因此不进入本报告数字和排名。

## 11. 结果不能说明什么

- 不能说明真实机器人成功率：没有闭环 rollout 或实机实验。
- 不能说明跨任务泛化：只有后把手安装任务。
- 不能说明统计显著性：只有 seed 42，没有重复实验或置信区间。
- 不能说明最终收敛优劣：训练约 0.93 epoch，曲线仍下降。
- 不能说明失败识别能力：训练和主验证只有成功轨迹。
- 不能把 `sigma=0.5` probe 当成高噪声完整采样；只有 RGB 评测执行了多步生成。
- 不能把 PSNR 当作唯一世界模型质量。生成未来可能多模态，像素不完全一致不一定物理上错误。
- 不能将当前 EDM 训练结果称为 Cosmos 3 rectified-flow mid-training。

## 12. 下一步实验优先级

1. **完成 ID 同预算实验。** 到 2000 steps 后比较 normal 与 future-shuffle；若打乱 future 使 action error 增大，才说明模型使用了未来信息。
2. **控制 Joint 的 Policy 样本预算。** 将 Joint 训练到累计 Policy samples 与 P 相同，或增加“50% sample 但仍全为 Policy”的稀释对照。
3. **增加随机种子。** 至少 3 seeds，报告均值、标准差或 bootstrap CI。
4. **扩展固定 RGB 样本。** 当前只有 16 rows；增加 episode 覆盖并给出逐样本分布。
5. **加入失败轨迹的独立 Value 评测。** 在不污染当前成功-only主表的前提下报告 AUROC/AUPRC 与校准。
6. **更长预算。** 至少约 2 epochs，确认 Euler 左臂优势和 Joint 劣势是否持续。
7. **最终闭环评测。** 在仿真或实机报告 task success rate，验证 action MAE 与 RGB PSNR 哪个更能预测部署效果。

## 13. 最终项目总结

本项目完成了从 HDF5 双臂轨迹到 Cosmos latent 数据、可审计 action encoding、Lazy Parquet DDP、统一 Policy/World/Value conditioning、EDM 训练、checkpoint 身份校验以及 action/latent/RGB/Value 多层评测的完整离线链路。

实验表明，机器人数据中训练能将未来相机 PSNR 从约 `7.4 dB` 提升到 `19–20 dB`；Euler 与 rotation-6D 在动作精度和世界建模上表现出不同优势；在不到一个 epoch 的固定预算下，Joint 未超过 Policy-only，但改善了 Value 的进度排序。该结论清晰、可复现，同时保留了单 seed、成功-only、短预算和无闭环验证的边界。

数字底稿位于 `tables/`，完整原始日志位于 `train_outputs/action_representation_20260903/*_2000/`，WandB 项目为 `cosmos-offline-compare-2000`。checkpoint 约 19 GB/个，不随本目录归档。
