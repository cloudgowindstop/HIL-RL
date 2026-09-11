# 面向具身智能的机器人数据治理与 Cosmos Policy 离线中训练实践

**课程：**《工业工程生产实践》  
**姓名：**葛铭博  
**学号：**2023010345  
**实习单位：**北京人形机器人创新中心有限公司  
**实习部门：**具身智能部  
**企业导师：**Chris Ren（具身智能算法工程师）、Linda Zhao  
**实习时间：**2026 年 7 月 1 日—9 月 4 日，共 10 周  

## 摘要

本次生产实习围绕真实双臂机器人数据如何可靠进入具身智能模型训练展开。针对原始数据来源分散、HDF5 字段与结构不统一、异常轨迹混入、Cosmos 数据转换计算开销大，以及离线训练与效果评测链路不完整等问题，本人在团队已有机器人采集代码和 Cosmos Policy 工程基础上，完成了原始数据下载与 Schema 治理、HDF5 到 Cosmos latent 的转换重构、多 GPU VAE 编码、Euler 与 rotation-6D 动作表示、分布式离线训练和多层评测工具的开发。

数据工程方面，工具共管理 11,015 个物理 HDF5，识别 13 种 Schema，并对严重异常数据进行隔离；转换侧完成 4 组后把手任务共 325 个 episode、436,360 帧的处理，8 卡 VAE 有效吞吐约 29.4 帧/秒。模型实验方面，在相同数据划分、基础权重和 2,000-step 预算下，完成 Euler 14D/rotation-6D 20D 与 Policy-only/Policy-World-Value Joint 的 2×2 对照。结果显示：动作表示不存在单一最优解，Euler Policy-only 在困难左臂动作回归上更好，rotation-6D Policy-only 在未来图像预测上更好；Joint 训练改善了 Value 排序，但在有限总步数下未超过 Policy-only 的动作与视觉表现。本项目形成了从生产数据治理到离线训练和效果验证的完整、可复现闭环。

**关键词：**具身智能；机器人数据治理；Cosmos Policy；离线训练；分布式训练；动作表示

## 一、背景信息

### 1. 实习单位与部门

北京人形机器人创新中心有限公司聚焦人形机器人和具身智能关键技术研发，围绕机器人本体、控制系统、通用具身智能平台及场景应用开展技术攻关。本次实习所在的具身智能部主要承担具身智能系统开发以及 VLA、强化学习（RL）、WAM 等模型研发，处于连接机器人生产数据、算法模型和实际部署的重要位置。

本人在企业导师 Chris Ren、Linda Zhao 的指导下参与机器人数据处理、模型调研、代码调试、离线训练和实验评测。实习项目不存在额外保密要求，但报告仍以技术流程和汇总指标为主，不包含账号、密钥等运行环境信息。

### 2. 实习目标与工作范围

真实机器人模型研发并不是取得数据后直接训练。设备侧轨迹首先需要经过来源追踪、质量检查、动作语义确认、视觉编码和训练格式组织；模型训练后还要用统一协议判断改进是否真实有效。因此，本次实习的目标是打通以下链路：

```text
生产现场与云端原始数据
        ↓
下载、追踪、Schema 识别与质量分级
        ↓
动作/状态构造、图像预处理与 VAE 编码
        ↓
Cosmos/LeRobot 数据集与固定 train/validation 划分
        ↓
Cosmos Policy 离线分布式训练
        ↓
Action、World、Value 与 RGB 分层评测
```

具体工作包括三部分：第一，学习强化学习、世界模型和 Cosmos Policy 方法并熟悉团队代码；第二，建立稳定、可审计的数据下载与转换流程；第三，在原始视频模型上开展机器人数据离线中训练，并比较动作表示和训练目标的影响。

## 二、典型一线生产、运作实践与调研总结

### 1. 所参与的生产实践

实习期间接触了机器人生产现场的数据采集和算法研发流程，参与真实机器人轨迹检查、设备及项目代码调试、转换结果核验和模型训练。生产数据由双臂机器人执行装配等任务产生，单条 HDF5 通常包含左右臂末端位姿、关节位置、夹爪状态、头部或顶部相机、左右腕部相机及时间戳等信息。本人也将转换后的动作还原为控制量，对机械臂运动方向和姿态变化进行反向检查，以验证动作定义和坐标系是否符合控制接口。

前期理论学习从贝尔曼方程、TD、SARSA 和 Q-Learning 延伸到 DQN、Actor-Critic、DDPG、PPO，并补充学习 BCQ、CQL 等离线强化学习方法。与此同时，本人熟悉机器人项目代码并修复部分实际 bug，使 state、action、reward、future state 和 value 等理论概念能够与采集字段和模型接口对应。

### 2. 企业数据运作流程

现场采集数据先上传至对象存储，再由算法人员按任务下载。由于采集程序、设备型号和生产批次不断变化，不同 HDF5 的相机名称、机器人字段、轨迹长度甚至关键字段完整性可能不同。原始数据必须经过整理和转换，才能进入模型训练。

本项目将运作流程整理为两个相互衔接的阶段：

```text
阶段 A：数据治理
验收表/BOS → 冻结下载计划 → 断点续传 → HDF5 扫描
             → Schema 分类 → 质量分级 → 事务搬运 → 核验

阶段 B：模型数据生产
HDF5 → episode 检查 → action/proprio 构造 → 图像裁剪
     → 多 GPU VAE → latent 注入 → Parquet/metadata → preflight
```

训练阶段不修改原始 HDF5，而是读取转换生成的 Parquet 和独立 manifest。这样既能保护原始数据，也能固定 train/validation 划分，使不同实验使用完全相同的样本范围。

### 3. 现场问题与实践认识

一线数据主要存在以下问题：

- **字段和结构不统一。** 不同批次可能使用 Head 或 Top 相机，也可能只有 joint 而没有末端 pose；若用单一读取规则，会在转换中途失败或产生错误语义。
- **异常轨迹不能仅靠文件存在性发现。** 实际发现过末端位姿后段全零、机器人字段长度不一致和相机缺帧等问题。
- **动作语义容易混淆。** 同一文件中同时存在 master/puppet、joint/末端位姿、绝对量/相对量；选错字段后程序仍可能正常运行，但训练目标已经错误。
- **数据规模放大工程风险。** 一次任务可能涉及上万 HDF5 和数 TiB 数据，静默覆盖、重复下载、半成品误判和内存峰值都会造成明显成本。
- **训练 loss 不能单独代表机器人能力。** 不同动作维度、噪声水平和目标区域的数值尺度不同，还需要动作、未来状态和图像层面的验证。

实践使我认识到，生产级机器人学习首先是数据语义和流程可靠性问题。只有保证来源可追踪、字段可解释、异常可隔离、输出可复现，后续模型实验的差异才具有分析价值。

## 三、专题项目改善总结

### 1. 项目背景、范围与计划

Cosmos Policy 将当前机器人状态、动作、未来状态和价值信息组织到统一的视频 latent 序列中，通过生成式模型学习状态—动作—未来之间的关系。团队已有采集代码和 Cosmos Policy 基础能力，但缺少一套适配现有生产数据、支持批量转换、离线分布式训练和系统评测的完整流程。

项目范围覆盖原始数据进入本地后的全部环节，但不包括采集硬件改造和正式实机闭环部署。10 周工作大致分为：

| 阶段 | 主要内容 | 阶段产出 |
| --- | --- | --- |
| 第 1—2 周 | 强化学习与 Cosmos 调研、代码熟悉和 bug 修复 | 理论基础与接口认识 |
| 第 3—6 周 | 数据下载、HDF5 分析、转换重构、多 GPU VAE | 可批量执行的数据流水线 |
| 第 7—8 周 | 离线训练、DDP、训练模式和监控 | 可复现训练框架 |
| 第 9—10 周 | 2×2 对照、分层评测、绘图与总结 | 定量结果与改进结论 |

### 2. 问题的结构化描述

本专题同时包含工业工程改善、统计分析和算法实验三类问题。

**数据流程改善问题。** 决策内容是如何识别 Schema、处理异常、组织数据和分配 VAE 计算；评价指标包括可读取文件数、异常隔离率、转换完整性、内存峰值和编码吞吐；约束包括原始数据只读、存储空间有限、GPU 数量有限以及中断后必须能够恢复。

**模型实验问题。** 决策变量为动作表示（Euler 14D 或 rotation-6D 20D）和训练目标（Policy-only 或 Joint）；控制变量包括相同原始 episode、train/validation 划分、任务文本、基础权重、随机种子、有效 batch、优化器和训练步数。评价指标覆盖动作平移误差、旋转测地角、夹爪准确率、未来 latent 误差、Value MAE/Spearman 以及解码图像 PSNR/SSIM。

**主要约束。** Cosmos-Predict2-2B 参数规模较大，训练必须使用 GPU 和混合精度；机器人数据已经预编码，不能在每次训练中重新做随机图像增强；2,000 steps 约为 0.93 epoch，属于有限预算比较而不是充分收敛实验；当前不使用实机成功率，因此结论只能限定为开环离线性能。

### 3. 行业实践与文献调研

离线强化学习研究表明，模型容易受到数据分布外动作和价值高估影响，因此训练数据范围、行为分布和验证集固定尤为重要。Cosmos Policy 的核心启发是把机器人状态和动作注入视频 latent，使视频生成先验能够参与策略学习；Cosmos 3 则提出 Pre-training—Mid-training—Post-training 范式，并通过 Forward Dynamics、Inverse Dynamics、Policy 等任务建立动作与世界变化的联系。

基于文献和代码调研，本项目没有直接宣称复现 Cosmos 3 的完整训练目标，而是复用 Cosmos-Predict2-2B、Wan2.1 VAE、T5 条件和 Cosmos Policy 的 EDM 去噪机制，重新组织适合当前数据的离线训练流程。当前使用的是 EDM loss，不是 Cosmos 3 的 rectified-flow velocity loss，也没有使用其 action loss 额外 10 倍权重。明确方法边界能够避免把参考论文中的设计误写为已经完成的实验。

### 4. 解决方案及实施过程

#### 4.1 原始数据下载与治理

本人在团队数据源和已有接口基础上开发 `data_download_tools`，将普通下载扩展为可审计流水线：

- 从验收表生成冻结计划，使用 `.partial` 暂存、完成标记、JSONL 账本和 `fsync` 支持断点恢复；
- 扫描 HDF5 的 Group、Dataset、shape、dtype、压缩和属性，排除可变时间长度后生成稳定 Schema 指纹；
- 将质量分为 passed、warning、review、quarantine 和 unreadable，避免轻微警告与严重异常混为一类；
- 采用“生成搬运计划—journal 记录—原子 rename—结果验证”的事务式流程，拒绝静默覆盖，并支持 recover/rollback；
- 通过规范化 BOS 来源路径进行增量去重，避免根据相似目录名误判。

最终正式整理库管理 11,015 个物理 `trajectory.hdf5`，识别 13 种 Schema，其中 8 个严重异常文件进入隔离区；增量任务完成 3,068 个 HDF5、约 1.53 TiB 数据下载，307 条搬运记录全部达到 verified 状态。该结果将一次性脚本改善为可重复运行、可追踪和可恢复的数据生产工具。

#### 4.2 HDF5 到 Cosmos 数据转换

转换工作的首要难点不是文件写入，而是动作语义。结合采集和控制逻辑，动作采用 puppet 从当前时刻到下一时刻的相对末端运动，Euler 方案中左右臂各包含三维平移、三维旋转和夹爪，共 14 维。为改善欧拉角不连续和奇异性问题，又增加 rotation-6D 20D 表示，以旋转矩阵前两列表示每只机械臂姿态，同时保留平移和夹爪信息。

转换后执行数值 round-trip：撤销归一化并由当前位姿重建下一时刻位姿，再与原始 HDF5 比较。在一条 1,432 帧真实轨迹上，14D action 与转换结果一致；proprio 与 future proprio 最大差异约 0.0039，符合中间 BF16 表示带来的量化误差。

视觉侧将头部及左右腕部图像裁剪、缩放到模型输入，再由 Wan2.1 VAE 编码。原流程只有一张 GPU 工作，且 VAE wrapper 还包含 mean/std 等非网络状态，不能简单套用 `DataParallel`。本人为每张 GPU 初始化并缓存完整 wrapper，将 CPU micro-batch 分片并行编码，结果返回 CPU 后按原顺序拼接；episode 和数据写入仍保持串行，从而兼顾并行效率与文件一致性。同时采用 uint8 常驻 CPU、按批转 float、流式编码和及时释放中间张量，降低 CPU/GPU 峰值。

该流程完成 4 组数据、325 个 episode、436,360 帧的正式转换，8 卡 VAE 有效吞吐约 29.4 帧/秒，运行期间未发生 CUDA OOM。由于没有同机器、同数据条件的严格单卡基线，本报告只报告实际吞吐，不夸大固定加速倍数。

#### 4.3 Cosmos Policy 离线训练架构

本人将团队在线 learner 的模型调用方式抽取为独立离线框架 `cosmos_offline_train`。训练数据采用 Lazy Parquet DataLoader，每个 DDP rank 使用 `DistributedSampler` 独立读取自己的样本，避免 rank 0 加载完整 ReplayBuffer 后通过 Python 对象分发造成的内存和通信瓶颈。row-group LRU 进一步减少重复文件解析。

一个训练样本被组织为 9 个 temporal latent 位置，依次包含 blank、当前 proprio、当前腕部/主相机、16-step action chunk、future proprio、future 腕部/主相机和 value。通过 conditioning mask 与 target mask 的组合，同一网络可以切换：

- Policy：由当前状态预测 action、未来状态和 value；
- World：给定当前状态和真实 action，预测未来状态和 value；
- Value：给定当前状态、action 和未来状态，预测 value；
- Inverse Dynamics：给定当前和未来状态，预测中间 action。

本次主实验使用 Cosmos Policy/Predict2 的 EDM 加权去噪损失。设模型预测与目标误差为 (e)，噪声权重为 (w(\sigma))，目标 mask 为 (M)，则反向传播标量可写为：

\[
L_{EDM}=\frac{\sum M\,w(\sigma)e^2}{\sum M}.
\]

训练时噪声强度随机采样；验证时固定噪声和随机种子，使不同 checkpoint 可以公平比较。系统还支持梯度累积、BF16、checkpoint/断点恢复、配置归档、Prometheus 和 WandB 监控。四卡主实验中 micro-batch 为 2/rank、累积 16 次，因此每次参数更新的有效 batch 为 (2\times4\times16=128)。

#### 4.4 分层评测设计

仅观察总 loss 无法回答动作是否准确、世界模型是否生成合理未来。因此建立四层指标：

| 层次 | 主要指标 | 验证目的 |
| --- | --- | --- |
| Action | translation MAE、rotation geodesic、gripper accuracy | 判断控制量预测质量 |
| Latent/World | future proprio/image EDM、MSE、L1 | 判断未来状态建模能力 |
| Value | MAE、Spearman 等级相关 | 判断进度估计和排序能力 |
| RGB | PSNR、SSIM、预测图和误差图 | 判断完整去噪后的可见未来质量 |

旋转误差使用两个旋转矩阵间的 SO(3) 测地角；Spearman 关注预测 value 是否保持样本进度排序；PSNR 和 SSIM 分别衡量像素误差和结构相似性。固定噪声 latent probe 适合快速、可重复地比较 checkpoint；多步去噪并经 VAE 解码的 RGB 评测更接近实际生成流程。两者结合，避免以低噪声单步误差替代完整推理效果。

### 5. 实验结果与改善效果

#### 5.1 实验设计

2×2 实验包含 Euler Policy-only（E-P）、Euler Joint（E-J）、rotation-6D Policy-only（6D-P）和 rotation-6D Joint（6D-J）。四组实验使用相同的 221 个成功 episode，其中 train 177 个、validation 44 个；训练集 274,870 rows，验证集 67,340 rows。所有 run 从同一 Cosmos-Predict2-2B-Video2World 基础权重开始，采用 seed 42、有效 batch 128、学习率 (10^{-4}) 和 2,000 optimizer steps，共处理 256,000 row-samples，约 0.93 epoch。

#### 5.2 主要定量结果

| 指标 | E-P | E-J | 6D-P | 6D-J |
| --- | ---: | ---: | ---: | ---: |
| 左臂 translation MAE | **0.0281** | 0.0351 | 0.0494 | 0.0527 |
| 右臂 translation MAE | 0.0162 | 0.0168 | **0.0154** | 0.0170 |
| 左夹爪准确率 | 96.31% | **96.82%** | 96.60% | 92.66% |
| future-camera PSNR/dB | 19.42 | 18.82 | **19.82** | 19.33 |
| future-camera SSIM | 0.726 | 0.680 | **0.735** | 0.717 |
| Value Spearman | 0.789 | **0.879** | 0.816 | 0.866 |

注：translation MAE 保留数据转换后的控制尺度，不直接写成米；不同维度的 raw loss 也不直接用于比较 Euler 与 6D。

RGB 指标属于 World/Forward Dynamics 条件：模型获得当前状态和真实动作后预测未来画面，并非模型自主预测动作后的端到端控制结果。旋转测地角在多数 run 中低于当前数值分辨率，不能解释为姿态预测绝对无误。

结果表明：

1. **动作表示没有单一赢家。** E-P 左臂平移误差比 6D-P 低 43.1%，而两者右臂差异不足 0.001；第 16 个动作 horizon 上也呈现相同趋势，说明 Euler 的左臂优势并非只来自当前一步。另一方面，6D-P 的未来相机 PSNR 和 SSIM 均为最高，说明连续旋转表示可能更有利于视觉变化建模。由于 rotation-6D 需要投影回旋转矩阵，而 Euler 与 6D 维数也不同，不能用两者的 raw MSE 直接判断优劣。
2. **夹爪和右臂指标接近饱和。** 四组实验右夹爪准确率均为 100%，左夹爪除 6D-J 外约为 96%—97%，右臂平移误差也十分接近。因此当前模型间差异主要由更困难的左臂轨迹和未来图像贡献，后续应增加困难样本或按操作阶段分层评测，避免被简单样本掩盖。
3. **有限预算下存在多目标竞争。** Joint 中约 50% 样本使用 Policy conditioning，其余用于 World/Value。在相同总步数下，E-J 左臂误差比 E-P 高 25.0%，6D-J 比 6D-P 高 6.7%，RGB 指标也未超过对应 Policy-only。这更可能说明固定总预算下各目标分摊了更新机会，而不能直接推导出多目标训练本身无效。
4. **Joint 对 Value 排序有正面作用。** E-J 和 6D-J 的 Spearman 分别达到 0.879 和 0.866，高于相同动作表示的 Policy-only，说明多目标监督有助于保持任务进度排序。不过 6D-J 的同分布 Value MAE 为 0.0104，高于 6D-P 的 0.0063，表明“排序正确”和“数值标定准确”是两个不同目标，应同时报告。
5. **机器人中训练产生明显视觉改善。** 基础视频模型 future-camera PSNR 约为 7.4 dB、SSIM 约为 0.20；四个微调模型分别达到 18.8—19.8 dB 和 0.68—0.74。6D-P 的主相机 PSNR 达到 21.37 dB，综合 PSNR 达到 19.82 dB。定性图中，微调模型已经能够恢复台面、机械臂和主要物体轮廓，误差主要集中在物体边缘、接触区域和细小结构。

#### 5.3 训练过程与结论边界

固定验证协议下，四组模型的左臂 translation MAE 随 checkpoint 变化如下：

| step | E-P | E-J | 6D-P | 6D-J |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 0.0375 | 0.0435 | 0.0694 | 0.0925 |
| 1,000 | 0.0379 | 0.0416 | 0.0561 | 0.0580 |
| 1,500 | 0.0354 | 0.0383 | 0.0540 | 0.0728 |
| 2,000 | **0.0281** | 0.0351 | 0.0494 | 0.0527 |

E-P 与 6D-P 到 2,000 steps 仍在改善；6D-J 在 1,500 steps 出现验证波动，随后恢复。训练 EDM 在早期快速下降，之后随随机 batch 和随机噪声正常波动，未出现 NaN、梯度爆炸或持续发散。旧的异常 6D 1,000-step run 已通过从基础权重重新训练排除，不进入最终比较。

因此，现有结果可以支持“离线中训练有效”和“当前预算下存在表示及目标权衡”，但不能支持模型已经收敛、6D/Euler 存在普遍优劣或 Joint 永远无效。RGB 指标也只使用固定的 16 个验证样本，适合作为受控诊断，尚需扩大样本量。

### 6. 项目亮点与个人贡献

本项目没有提出新的基础模型，创新和改善主要体现在生产数据、训练系统与实验方法三个层面：

- 建立从 BOS/HDF5 到 Cosmos latent、DDP 训练和多层评测的端到端闭环；
- 通过自动 Schema 发现、质量分级和事务式搬运，提高大规模机器人数据治理的可靠性；
- 实现 Euler/rotation-6D 及 Policy/World/Value/ID 的统一、可扩展训练接口；
- 用固定数据划分和控制变量实验揭示动作表示权衡与多目标竞争，而不是只展示单条 loss 曲线；
- 建立 action、latent、Value、RGB 四层验证，以及配置、权重和数据身份记录，提高结论可复现性。

上述工作均以团队已有采集系统、机器人控制代码和 Cosmos Policy 基础代码为起点。本人承担了 BOS 下载与 HDF5 分类工具、11,015 个文件的整理和 Schema 分析、数据转换重构、8 卡 VAE 适配、两种动作编码、离线 DDP 框架、多目标 mask、监控、2×2 实验、评测绘图和结果总结。团队提供项目需求、原始数据、基础模型接口、计算平台及指导；本报告不把 Cosmos 官方模型或团队原有采集能力计为个人原创。

## 四、经验体会、不足与后续计划

### 1. 经验体会

第一，生产现场的数据质量问题往往比算法公式更先决定实验成败。字段名称、坐标系、当前/下一时刻和 master/puppet 的任何误解，都可能形成“程序正常、语义错误”的隐蔽问题。数值 round-trip 和真机方向检查比仅验证 shape 更有价值。

第二，工业工程中的标准化、可追踪和持续改善方法同样适用于算法研发。冻结计划、质量分级、事务记录、固定实验矩阵和统一指标，使数据处理和模型训练从个人脚本转变为可重复的生产流程。

第三，多 GPU 并不等于简单复制模型。VAE wrapper 状态、数据顺序、写入互斥、DDP sampler 和梯度同步都需要分别处理。性能优化也应报告可验证的吞吐和资源使用，不在缺少严格基线时宣称固定加速比。

第四，模型效果需要分层解释。训练 loss 下降只说明优化目标减小；动作误差、Value 排序和 RGB 质量回答的是不同问题。只有控制数据、权重和预算，并使用相同验证协议，才能把差异归因于动作表示或训练模式。

### 2. 当前不足

- 主实验只有单任务、成功演示和单随机种子，泛化性尚未验证；
- 2,000 steps 不到一个 epoch，Joint 可能因每种目标获得的样本预算不足而处于劣势；
- 当前是开环离线评测，尚不能替代机器人闭环成功率和安全性测试；
- 预编码图像减少了在线随机增强，可能降低视觉数据多样性；
- 当前是 Cosmos Policy/Predict2 的 EDM 微调，并非 Cosmos 3 rectified-flow mid-training；
- ID 训练仍在进行，本报告不使用未完成结果。

### 3. 后续计划

后续实验按照“先补齐公平性，再验证方法改进，最后进入闭环”的顺序开展：

| 优先级 | 实验方向 | 对照设计与判定依据 |
| --- | --- | --- |
| P0 | ID 有效性 | 完成同预算 ID；比较 normal、current-only、future-only、future-shuffle。若打乱 future 后 action error 显著增大，才能说明模型真正使用未来信息 |
| P0 | Joint 公平预算 | 保持 Policy 更新次数相同，而不是只保持总 step 相同；同时扫描 P/W/V 混合比例，判断当前劣势是否来自目标分摊 |
| P1 | 统计可靠性 | 每个配置至少运行 3 个随机种子，报告均值、标准差和置信区间；扩大 RGB 样本并按 episode 聚合 |
| P1 | 泛化能力 | 增加多个任务、不同日期和不同场景的留出集，并引入失败轨迹，验证跨任务与成败识别能力 |
| P1 | 损失与时间建模 | 比较 action loss 加权、EDM 与 rectified-flow；研究 Video Sparse/Action Dense，兼顾视觉冗余和高频动作监督 |
| P2 | 鲁棒性与部署 | 引入 Noisy History、图像扰动和状态噪声，随后用实机成功率、执行时间、越界率和动作平滑性验证闭环效果 |

其中最先应完成的是 ID 消融和 Joint 公平预算实验，因为它们直接决定当前多任务架构的解释。之后再研究 action 加权和时间采样，避免同时改变多个变量。实机阶段应固定控制频率、动作缩放和安全边界，并保留同一组初始状态，使离线指标与真实成功率能够建立对应关系。

## 五、总结

本次实习从真实生产数据出发，完成了机器人原始数据治理、Cosmos 数据转换、分布式离线中训练和多层效果验证。工作不仅解决了字段不统一、异常轨迹、多 GPU 编码和大规模读取等工程问题，也通过受控实验说明：Euler 与 rotation-6D 在动作回归和视觉预测上存在不同优势；有限预算下 Joint 会与 Policy 目标竞争，但能改善 Value 排序。

通过这一过程，我把强化学习和世界模型知识落实到机器人数据语义、训练系统和实验分析中，也更加理解了工业场景中“数据可靠—流程可复现—指标可解释”的重要性。项目形成的工具和评测协议能够复用于后续任务，为进一步开展 ID、联合训练和实机闭环验证奠定了基础。

## 参考文献

[1] NVIDIA. *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning*.

[2] NVIDIA. *Cosmos 3: Omnimodal World Models for Physical AI*.

[3] Ho J, Jain A, Abbeel P. Denoising Diffusion Probabilistic Models.

[4] Zhou Y, Barnes C, Lu J, Yang J, Li H. On the Continuity of Rotation Representations in Neural Networks.

[5] Fujimoto S, Meger D, Precup D. Off-Policy Deep Reinforcement Learning without Exploration.

[6] Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-Learning for Offline Reinforcement Learning.

## 附录：建议保留的精简图表

为控制正文篇幅，正式排版时建议正文只保留一张总流程图、一张动作/训练模式对比图和一张 RGB 预测示例，其余材料置于附录：

1. 动作表示与训练模式对比：[fig2_translation_2x2.png](figures/compare/fig2_translation_2x2.png)
2. 未来相机 PSNR：[fig8_future_cameras_psnr.png](figures/compare/fig8_future_cameras_psnr.png)
3. Value 排序对比：[fig9_value_spearman.png](figures/compare/fig9_value_spearman.png)
4. Euler Policy RGB 示例：[euler_14d_policy_2000](figures/rgb_examples/euler_14d_policy_2000_0804_am_success_0804_095314_row750.png)
5. rotation-6D Policy RGB 示例：[rotation6d_20d_policy_2000](figures/rgb_examples/rotation6d_20d_policy_2000_0804_am_success_0804_095314_row750.png)

详细代码、指标公式和数字底稿分别见同目录下的 [CODE.md](CODE.md)、[DATA_CONVERT.md](DATA_CONVERT.md)、[DATA_DOWNLOAD.md](DATA_DOWNLOAD.md)、[RESULTS.md](RESULTS.md) 与 `tables/`。这些技术说明不必全部并入 12 页正文。
