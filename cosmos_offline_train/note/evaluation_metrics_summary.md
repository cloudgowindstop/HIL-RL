# Cosmos 离线中训练评测指标汇总

## 评测目标

判断中训练是否同时改善优化稳定性、动作建模、动力学、视频生成、价值判断和泛化能力。所有对比必须固定 validation split、样本顺序、noise seed、T5、statistics、tokenizer 和 sampler 参数，并按完整 episode 聚合。

## 1. 训练稳定性

### EDM loss

扩散模型在噪声强度 \(\sigma\) 下预测干净 latent \(x_0\)：

\[
L_{EDM}=\frac{1}{|M|}\sum_{i\in M}w(\sigma_i)\|\hat{x}_{0,i}-x_{0,i}\|_2^2
\]

其中 \(M\) 是有效 target mask，\(w(\sigma)\) 平衡不同噪声强度。它直接对应训练目标，用于确认模型是否学会去噪；不能单独证明任务能力。

同时记录 total、policy、world、value、inverse 以及 action/future/value 各 latent 区域 loss。辅助指标包括 gradient norm、learning rate、step time、GPU memory、NaN/Inf。

## 2. Policy：当前观察预测动作

### Action MAE / MSE

\[
MAE=\frac{1}{N}\sum_i|\hat a_i-a_i|,
\qquad
MSE=\frac{1}{N}\sum_i(\hat a_i-a_i)^2
\]

MAE 易解释、对异常值较稳健；MSE 更重罚大错误。translation、rotation、gripper 必须分开统计，不能只报告混合 20D MSE。

### Rotation geodesic error

先把 6D rotation 恢复为旋转矩阵，再计算 SO(3) 最短角距离：

\[
\theta=\cos^{-1}\left(\frac{\operatorname{tr}(R_p^TR_t)-1}{2}\right)
\]

单位为度。6D 向量欧氏误差不等于真实旋转误差，因此 geodesic error 是主要旋转指标。

### Gripper 与 horizon

- gripper MAE；
- 按开/关阈值计算 accuracy、precision、recall；
- action horizon 1/4/8/16 分项误差。

目的：判断 policy 是否能预测物理正确、长期不漂移的动作。离线 action error 不等于闭环成功率。

## 3. Inverse Dynamics：前后观察反推动作

使用与 Policy 相同的 action 指标，但条件为 current + future observation。

必须增加条件消融：

| 设置 | 条件 | 验证目的 |
|---|---|---|
| normal | current + future | 完整 ID |
| current-only | current | 排除模型仅执行 policy |
| future-only | future | 检查未来状态信息量 |
| future-shuffle | current + 错配 future | 检查模型是否真正使用 future |

定义 future 信息增益：

\[
\Delta_{future}=E_{shuffle}-E_{normal}
\]

其中 \(E\) 是 action error。\(\Delta_{future}>0\) 且足够大，才支持“模型学到反向动力学”。当前 Base/20K 结果只有 normal ID 单步 denoising，尚未完成此因果检查。

## 4. Forward Dynamics / World Model

条件为 current observation + ground-truth action，预测 future observation。

快速验证使用：

\[
L1=\frac{1}{N}\sum_i|\hat z_i-z_i|,
\qquad
MSE=\frac{1}{N}\sum_i(\hat z_i-z_i)^2
\]

分别统计 future proprio、wrist-image latent、third-person-image latent，并按预测 horizon 分组。

目的：验证模型是否理解“执行动作后状态如何变化”。latent error 适合快速 checkpoint 筛选，但不直接表示 RGB 视频质量。

## 5. 完整视频生成质量

必须运行多步 diffusion sampler，并用 VAE/tokenizer 解码预测和 GT latent。

### PSNR

\[
PSNR=10\log_{10}\frac{MAX_I^2}{MSE}
\]

单位 dB，越高越好。它衡量严格对齐的像素重建，适合短期确定性预测；对轻微位移和多模态合理未来敏感。

### SSIM

\[
SSIM(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
\]

衡量亮度、对比度和局部结构，越高越好。

### LPIPS

\[
LPIPS=\sum_l\|w_l\odot(\phi_l(\hat I)-\phi_l(I))\|_2^2
\]

使用深层视觉特征衡量感知差异，越低越好。比 PSNR 更接近人眼感知，但仍不能判断物理因果正确性。

三者必须联合报告，并按 camera、frame horizon、episode 分组，同时保存预测视频和差异图。

## 6. Value 能力

先从 value latent 恢复标量，再计算：

- MAE/MSE：数值回归精度；
- AUROC/AUPRC：区分 success/failure 的排序能力；
- Spearman 相关系数：

\[
\rho=1-\frac{6\sum_i d_i^2}{n(n^2-1)}
\]

其中 \(d_i\) 是预测与真实排名差；

- Expected Calibration Error：预测成功概率与真实成功频率的一致性；
- candidate top-k ranking accuracy：规划时能否选择更优轨迹。

目的：Value 用于比较候选未来，不只拟合一个数。success-only validation 不能计算可靠 AUROC，也不能充分验证排序能力。

## 7. 泛化与最终任务能力

### Episode 统计

- task/source/outcome/camera/horizon 分组；
- mean、median、标准差；
- episode bootstrap 95% confidence interval；
- ID/OOD gap：\(E_{OOD}-E_{ID}\)。

按 episode 聚合可避免同一轨迹大量相邻帧被错误视为独立样本。

### 闭环指标

若有 simulator，记录：

\[
SuccessRate=\frac{N_{success}}{N_{rollout}}
\]

同时记录 staged task score、完成时间和 failure 类型。闭环 success rate 是策略有效性的最强证据；没有仿真或实机时，只能报告离线代理指标。

## 最小验收组合

| 层级 | 必须指标 | 能证明什么 |
|---|---|---|
| 训练健康 | EDM loss、gradient norm、NaN | 训练流程正常 |
| 动作能力 | Policy action error、ID 消融 | 动作预测与反向动力学改善 |
| 世界建模 | future latent error、PSNR/SSIM/LPIPS | 状态和视频预测改善 |
| 价值能力 | value MAE、AUROC、ranking | 候选轨迹判断改善 |
| 泛化能力 | episode CI、OOD、sim success | 改善可泛化并影响任务结果 |

推荐固定比较：`Base / 5K / 10K / 15K / 20K / 专用 ID checkpoint`。训练 loss 下降是必要条件；离线能力指标改善提供中等证据；OOD 或闭环成功率改善才是强证据。
