# 离线训练代码：实现、亮点、优化

这条栈挂在官方 Cosmos Policy 2B / Predict2 上，**不改 mask / loss / action 布局**，用来公平对比 Euler 14D vs rotation-6D、Policy-only vs joint。对比实验走根目录模块 + `run.sh`，不走 `pipeline/` / `components/` 那套尚未接完的分层。

代码根：`HIL-RL/cosmos_offline_train/`。数字结论见 [RESULTS.md](RESULTS.md)。

## 实现了什么

按数据流，不是按文件清单。

**数据。** 冻结 221 条后把手成功 episode（排除 `0804_014423`），同一 raw split 分别转 14D Euler 和 20D rotation-6D。`tools/audit_paired_datasets.py` 核对 source_id、平移/夹爪统计；T5 和 `dataset_statistics.json` 按 encoding 分开，不能混用。split 以完整 episode 为单位，seed 42，train 177 / val 44。

**训练。** 输入已是 VAE latent，前向是预编码 EDM（`model.preencoded_edm_forward`）。4 卡 DDP，`batch_size_per_rank=2`，`accum=16`，有效 batch 128，lr `1e-4`，seed 42，`max_steps=2000`（约 0.93 epoch）。`policy_only` 已经预测 action + future + value。`base_joint` 每个 sample 独立 `torch.multinomial` 抽 policy / world / value，概率 0.50 / 0.25 / 0.25，不是按固定顺序轮转，也不是多几个 head。

```17:19:HIL-RL/cosmos_offline_train/masks.py
def sample_types(batch_size: int, probabilities: tuple[float, ...], device=None) -> torch.Tensor:
    probs = torch.tensor(probabilities, dtype=torch.float32, device=device)
    return torch.multinomial(probs, batch_size, replacement=True)
```

**动作规格。** `action_spec.py` + `rotation_6d.py`：14D Euler（增量 × `rotation_scale=0.06`）和 20D 6D、反归一化、SO(3) 测地角。主指标用物理空间，不用 14D/20D raw MSE。

**评测。**

- Policy：左右平移 MAE、测地角、夹爪 accuracy、chunk 第 16 槽（horizon-16，不是闭环）。
- World RGB：当前观测 + GT action → 未来相机。冻结 8 episode × 2 行，固定 noise seed，`clean_restore_latent` 后再 VAE decode，只评未来腕/主相机 PSNR/SSIM。
- Value：标量 MAE，以及预测第 8 槽均值 vs GT `value_function_return` 的 Spearman。全 val 拼一次再算，不是 batch 平均。成功数据不报 AUROC。

**入口。** `run.sh`：preflight、train、resume、validate、visualize、evaluate、summarize。Checkpoint 写入数据身份（T5 / stats / manifest / 基础权重），评测加载时核对。

**故意没改：** `masks.py`、`losses.py`、`action_spec.py`、`rotation_6d.py`、VAE 注入。公平对比能成立，靠的是这些不动。

## 亮点

**公平对比，不是换一套数据再比表示。** 同一批成功 episode、同一 T5、同一 seed 和预算。有冻结 split 和配对审计，避免「Euler 一个任务、6D 另一套数据」。

**主指标在物理空间。** 平移 MAE、测地角、夹爪。Euler 测地角接近 0° 是增量 × 0.06，不是旋转学完美。

**joint 和官方初始配方对齐。** 每样本 i.i.d. 抽 50/25/25。`policy_only` 本来就预测 future/value，差在条件任务混合。

**World RGB 是独立协议。** 不塞进训练 loop。latent L1 和 PSNR 可以不同序——2000-step 上就是如此。

**成功数据上的 Value 只报回归和排序。** Spearman 量的是「越接近完成预测越高」，不是失败检测。

## 已经优化了什么

| 问题 | 改动 | 效果和边界 |
| --- | --- | --- |
| `pq.read_table` 整集读入（单集约 181MB），行级 shuffle 后 cache 几乎不命中 | `ParquetFile.read_row_group` + 只读训练列 + `(path, rg)` LRU | 单次约 37MB。**2000-step 四条开跑时还没用上** |
| resume 的 CUDA RNG 在 GPU 上，restore 崩 | checkpoint 存 CPU `uint8` ByteTensor | resume 能加载；**不能**解决下面的 skip 慢 |
| resume 重扫 epoch-0，约 2s/batch，1000 step 要 skip ~9 小时 | 打了 skip 日志，对比实验改为从头训到 2000 | 诊断清楚了；没有做成 O(1) seek |
| `CONFIG=` 相对路径从仓库根启动找不到 | `run.sh` 从启动目录 / HIL-RL / 脚本目录解析 | 对比命令可复述 |
| 集群 `WANDB_PROJECT` 盖掉实验名 | YAML 的 project/name 优先 | 进了 `cosmos-offline-compare-2000` |
| 旧 6D-P@1000 末期异常被当成 6D 基线 | 从头 `*_2000`，主表标明作废 | 避免「joint 把 0.115 救到 0.057」 |

工程优化和科学结论分开：row-group 是 IO，不是「表示更好学」的证据。

## 还可以优化

**不要为了快去改的（会破可比性）**

- mask、采样概率、action 布局
- 把失败轨迹并进主训练再和现在的表横比

**下一次若再训**

- Resume：保存 sampler 游标，或按 `samples_seen` 构造 skip 索引；要么继续规定对比实验只从头训
- 采样 vs 缓存：按行 shuffle 时 row-group 命中仍低。若允许改顺序（先打乱 episode，再在集内顺序读），IO 会好很多，但不能和现有 2000-step 比
- ckpt 约 19G（含 optimizer）。评测/分享可另存 `model_only`
- 树里的 row-group 读取，新 run 会自动用上

**只评测、不动权重**

- Spearman 现在 n=80。加大 `max_validation_batches` 或 4 卡 gather
- World RGB 只有 16 帧均值；可做逐帧表
- 闭环 / 真机仍缺；horizon-16 不要写成闭环

**代码债**

- 根模块和 `pipeline/` `components/` 两套并存
- YAML 里是本机绝对路径
- `run.sh summarize` 仍指向 1000-step 目录
- Spearman 是后来单独 `evaluate --objectives value` 算的，不在训练 end-eval 的旧 jsonl 里

科学方向和优化目标的展开见 [RESULTS.md](RESULTS.md) 第 9 节，不在这里重复。
