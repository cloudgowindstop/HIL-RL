# 《面向具身智能的机器人数据治理与 Cosmos Policy 离线中训练实践》PPT 大纲（待确认）

## Slide 1｜封面

- 标题：面向具身智能的机器人数据治理与 Cosmos Policy 离线中训练实践
- 副标题：《工业工程生产实践》总结汇报
- 姓名与学号：葛铭博，2023010345
- 实习单位：北京人形机器人创新中心有限公司；部门：具身智能部
- 企业导师：Chris Ren、Linda Zhao；实习时间：2026.07.01—09.04
- 视觉构想：浅色学术封面，以机器人视觉帧、数据流和神经网络网格的抽象组合为背景
- 页面角色：封面；建立“机器人生产数据 × 世界模型”的技术主题
- 必需源图：
  - 本轮上传图3（单臂机器人实拍）；严格输入素材；作为封面右侧主视觉，保持机器人设备真实外观，不生成替代机器人

## Slide 2｜实习背景、部门职能与项目目标

- 具身智能部主要开展具身智能系统、VLA、RL、WAM 等模型研发
- 个人工作位于生产数据、模型训练与效果验证的连接环节
- 10 周路径：理论与代码熟悉 → 数据工程 → 离线训练 → 对照实验与总结
- 项目目标：建立“可靠数据—可复现训练—可解释评测”的完整闭环
- 个人与团队边界：基于团队采集系统、控制代码和 Cosmos 基础能力开展个人开发
- 视觉构想：左侧组织与目标，右侧四阶段时间轴
- 页面角色：背景与时间线；快速交代实践环境和个人任务
- 必需源图：无

## Slide 3｜一线生产流程与核心痛点

- 现场数据包含双臂末端位姿、关节、夹爪、头部/腕部相机和时间戳
- 原始链路：机器人采集 → BOS → HDF5 → 模型数据 → 训练
- 痛点一：设备与批次导致字段、相机、Schema 不统一
- 痛点二：全零 pose、长度不一致、相机缺帧等异常不易被发现
- 痛点三：master/puppet、joint/EE、绝对/相对和时刻语义容易混淆
- 痛点四：上万文件、TiB 数据带来重复下载、中断恢复、内存和审计压力
- 视觉构想：横向生产流程，四个红色风险节点对应痛点
- 页面角色：问题定义；体现对一线生产运作的理解
- 必需源图：
  - 本轮上传图1（四宫格多相机作业现场）；严格输入素材；用于说明真实机器人多视角数据来源，保持四个视角内容不变

## Slide 4｜总体改善方案：从原始数据到可验证模型

- 数据治理：冻结下载计划、来源追踪、Schema 识别、质量分级、事务搬运
- 数据生产：动作/状态构造、图像预处理、多 GPU VAE、latent 注入
- 模型训练：Lazy Parquet、DDP、Policy/World/Value/ID mask、EDM loss
- 效果验证：Action、World latent、Value、RGB 四层指标
- 关键原则：原始数据只读、处理过程可恢复、实验变量受控、结论边界明确
- 视觉构想：一条从 BOS/HDF5 到评测报告的闭环架构图，突出四层模块
- 页面角色：总览流程；承上启下
- 必需源图：无

## Slide 5｜数据治理与转换工程改善

- 数据治理成果：管理 11,015 个物理 HDF5，自动识别 13 种 Schema，8 个严重异常文件隔离
- 增量下载：3,068 个 HDF5、约 1.53 TiB；307 条搬运记录全部 verified
- 转换成果：4 组任务、325 episodes、436,360 帧
- 动作编码：Euler 14D 与 rotation-6D 20D；通过数值 round-trip 检查物理语义
- 计算优化：8 卡完整 VAE wrapper 并行、micro-batch 流式编码，有效吞吐约 29.4 帧/秒
- 视觉构想：左侧治理漏斗/Schema 分类，右侧转换流水线与三张核心指标卡
- 页面角色：工程改善与量化成果
- 必需源图：无

## Slide 6｜Cosmos Policy 离线训练架构

- 基座：Cosmos-Predict2-2B-Video2World；机器人图像已预编码为 latent
- 9 个 temporal slots：blank、当前 proprio/双相机、action chunk、未来 proprio/双相机、value
- 统一 mask：Policy、World、Value、Inverse Dynamics 共享同一 2B 网络
- 训练目标：Cosmos Policy EDM 加权去噪；训练 sigma 随机，验证 sigma 和 noise seed 固定
- 分布式实现：每 rank 独立 DataLoader + DistributedSampler；DDP、BF16、梯度累积
- 主实验有效 batch：2/rank × 4 GPU × 16 accum = 128
- 视觉构想：中央9槽 latent 时间带，上方 conditioning mask，下方四种 objective 与 DDP 数据流
- 页面角色：核心技术架构解释
- 必需源图：无

## Slide 7｜受控实验设计与分层评测

- 2×2 自变量：Euler/rotation-6D × Policy-only/Policy-World-Value Joint
- 控制变量：同一221个成功 episode、177/44 train/validation、同一基座/seed/batch/学习率
- 训练预算：每组 2,000 optimizer steps，256,000 row-samples，约 0.93 epoch
- Action：平移 MAE、旋转测地角、夹爪准确率
- World/Value/RGB：latent L1、Value MAE/Spearman、PSNR/SSIM与误差图
- 评测边界：固定噪声 probe 不是全量验证；RGB 使用 GT action，不等同闭环控制
- 视觉构想：左侧2×2实验矩阵，右侧四层指标金字塔
- 页面角色：实验方法；证明比较公平且指标可解释
- 必需源图：无

## Slide 8｜动作结果：Euler 与 rotation-6D 各有优势

- 左臂 translation MAE：E-P 0.0281，6D-P 0.0494；Euler 低 43.1%
- 右臂 translation MAE：E-P 0.0162，6D-P 0.0154；差异很小
- 左夹爪约 96%—97%，右夹爪均为100%；简单指标接近饱和
- Policy-only 总体优于同表示 Joint：当前预算下存在目标竞争
- 验证曲线到 step 2,000 仍下降，因此属于阶段性公平比较而非充分收敛结论
- 视觉构想：主图为2×2平移误差柱状图，辅图为左臂checkpoint曲线；旁边放三条结论
- 页面角色：Action 数据证据与解释
- 必需源图：
  - 2×2平移误差对比；严格输入证据图；保留数据、坐标轴、图例、颜色和数值

    ![Translation 2x2](../figures/compare/fig2_translation_2x2.png)

  - 左臂验证曲线；严格输入证据图；保留数据、坐标轴、图例、颜色和数值

    ![Left arm curve](../figures/compare/fig10_left_arm_curve.png)

## Slide 9｜World 与 Value 结果：视觉改善和多目标权衡

- 基础模型 future-camera PSNR 约7.4 dB，微调后达到18.8—19.8 dB
- 6D-P 综合 PSNR 19.82 dB、SSIM 0.735，为四组最佳
- Joint 的 Value Spearman 更高：E-J 0.879、6D-J 0.866
- 但 Value MAE 未同步改善：排序能力与数值标定能力需要分别评价
- 综合排序：视觉预测 6D-P > E-P > 6D-J > E-J；Joint 当前主要优势体现在Value排序
- 视觉构想：PSNR和Spearman双图并列，底部以小型“GT｜Pred｜Error”示例说明视觉含义
- 页面角色：World/Value 数据证据与综合分析
- 必需源图：
  - future-camera PSNR；严格输入证据图；保留数据、坐标轴、图例、颜色和数值

    ![Future cameras PSNR](../figures/compare/fig8_future_cameras_psnr.png)

  - Value Spearman；严格输入证据图；保留数据、坐标轴、图例、颜色和数值

    ![Value Spearman](../figures/compare/fig9_value_spearman.png)

  - 6D-P同一验证样本的未来RGB；严格输入证据图；保留GT、Prediction、Error及上下相机关系

    ![6D Policy RGB](../figures/rgb_examples/rotation6d_20d_policy_2000_0804_am_success_0804_095314_row750.png)

  - 本轮上传图2（三联模型预测对比）；可选补充素材；主证据优先采用上方本地高分辨率预测图

## Slide 10｜项目贡献、结论与未来实验

- 完成BOS/HDF5治理、Cosmos转换、DDP训练和多层评测的端到端闭环
- 核心结论：动作表示存在任务相关权衡；有限预算下Joint存在目标竞争，但改善Value排序
- 方法边界：当前是Predict2 EDM微调、单任务/单seed/开环评测，不是Cosmos 3完整mid-training
- P0：完成ID normal/future-shuffle消融；按相同Policy更新次数重做Joint公平预算
- P1：3 seeds、更多RGB样本、多任务/失败轨迹、action loss权重和时间采样
- P2：Noisy History与实机闭环，评价成功率、执行时间、越界率和动作平滑性
- 结束语：从“能处理数据”推进到“结果可复现、指标可解释、流程可扩展”
- 视觉构想：左侧三项个人贡献，中央两条结论，右侧P0/P1/P2路线图
- 页面角色：总结与展望；突出个人价值并保持学术边界
- 必需源图：无

## 统一要求（待后续风格确认）

- 比例：16:9；总页数：10页
- 类型：学术报告/生产实习答辩；简约、克制、信息层次清晰
- 语言：中文为主，保留必要技术英文与公式
- 数据原则：所有数字以 `mingbo_result/tables/` 和最终报告为准；不得生成或修改实验数值
- 图表原则：Slide 8—9 的指定图为严格输入证据；必须保持轴、图例、数值和语义完整
- ID：只作为后续计划，不展示未完成训练结果
- 用户上传素材映射：图1→Slide 3、图3→Slide 1，均作为严格输入素材；图2可作为Slide 9补充，主证据使用本地高分辨率预测图
