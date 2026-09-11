# Cosmos 离线训练：分布式训练操作手册

本文针对当前 `cosmos_offline_train`，默认场景是百舸 **1 台 Worker、4/6/8 张
A100 80GB** 的单机多卡 DDP。当前 `run.sh` 使用 `torchrun --standalone`，不支持
直接跨多台 Worker；多机方式见文末。

## 1. 当前 DDP 架构

DDP 是一张 GPU 对应一个 Python 进程。每个 rank 保存完整的 Cosmos 2B 模型，
使用 `DistributedSampler` 独立读取不同 Parquet 样本，反向传播时通过 NCCL
AllReduce 同步梯度。DDP 提升数据并行吞吐，不会把一个放不进单卡的模型拆开。

```text
rank 0 -> GPU 0 -> Dataset/DataLoader shard 0 --\
rank 1 -> GPU 1 -> Dataset/DataLoader shard 1 ----> NCCL gradient AllReduce
...                                           --/  -> identical optimizer update
```

当前代码已经具备以下实现：

| 能力 | 代码位置 | 作用 |
|---|---|---|
| 初始化 NCCL、绑定 GPU | `distributed.py:28` | 读取 `RANK/LOCAL_RANK/WORLD_SIZE` |
| DDP 包装模型网络 | `train.py:278` | 每个进程包装 `model.net` |
| 每 rank 构建 DataLoader | `train.py:250-267` | 每张卡独立读取数据 |
| 数据分片 | `dataloader.py:149` | `DistributedSampler(dataset, rank, world_size)` |
| epoch 随机种子 | `trainer.py:240` | `sampler.set_epoch(epoch)` |
| 梯度累积 | `trainer.py:256-267` | 非更新 micro-batch 使用 `DDP.no_sync()` |
| 指标汇总 | `trainer.py:305-313` | `dist.all_reduce` 聚合所有 rank |
| 单点保存 | `trainer.py:343-344` | 只有 rank 0 写 checkpoint |

因此，当前训练不使用“rank 0 从 ReplayBuffer 采完整 batch，再 Python scatter”的
方式。所有 rank 都读取数据，DDP 只同步梯度和指标。

## 2. 如果从单卡代码改成 DDP，需要改什么

当前项目已经完成这些修改。迁移其他训练脚本时，必须保留以下五点。

### 2.1 初始化进程组并绑定设备

```python
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
device = torch.device("cuda", local_rank)
```

不要让所有进程默认使用 `cuda:0`。

### 2.2 使用 DistributedSampler

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed,
    drop_last=True,
)
loader = DataLoader(dataset, sampler=sampler, shuffle=False, ...)
```

`DataLoader` 设置 sampler 后不能再设置 `shuffle=True`。每个 epoch 开始前调用：

```python
sampler.set_epoch(epoch)
```

### 2.3 只包装参与训练的网络

```python
model.net = DDP(
    model.net,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False,
)
```

Tokenizer/VAE、数据集、日志器不需要放进 DDP。`find_unused_parameters=False` 性能
更好，但要求每次 forward 使用的参数图满足当前模型设计。

### 2.4 梯度累积时避免重复通信

非更新 micro-batch 使用 `model.net.no_sync()`，只在本次 optimizer update 的最后
一个 micro-batch 同步梯度。否则每个 micro-batch 都 AllReduce，会显著降低吞吐。

### 2.5 副作用只由 rank 0 执行

以下操作只允许 `rank == 0`：

- 写 `metrics.jsonl`；
- 启动 Prometheus HTTP server；
- 写 checkpoint、`latest.json`、曲线；
- 打印高频训练日志。

验证指标在所有 rank 上计算后 `all_reduce`，不能只汇报 rank 0 自己的数据。

## 3. YAML 必须确认的参数

正式训练前至少检查：

```yaml
data:
  root: /absolute/dataset/path
  train_manifest: /absolute/path/train.json
  val_manifest: /absolute/path/val.json
  t5_embeddings_path: /absolute/path/t5_embeddings.pkl
  statistics_path: /absolute/path/dataset_statistics.json
  workers_per_rank: 4
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true

training:
  epochs: 3
  max_steps: 2000
  batch_size_per_rank: 1
  gradient_accumulation_steps: 16
  validation_every_steps: 100
  checkpoint_every_steps: 1000

runtime:
  output_dir: /persistent/path/train_outputs/experiment_name
```

有效 batch 公式：

```text
effective_batch
= GPU 数量 × batch_size_per_rank × gradient_accumulation_steps
```

当每卡 batch 为 1、目标约 128 样本/update：

| GPU | accumulation | 有效 batch |
|---:|---:|---:|
| 4 | 32 | 128 |
| 6 | 21 | 126 |
| 8 | 16 | 128 |

从 6 卡切到 8 卡时，如果仍使用 accumulation=21，有效 batch 会从 126 变成 168，
实验不再等价。公平对比应将其改为 16。GPU 数量由启动命令决定，不写在 YAML。

`max_steps` 是 optimizer update 次数，不是 micro-batch 次数。8 卡、每卡 batch=1、
accumulation=16、2000 steps 时，共消费约：

```text
8 × 1 × 16 × 2000 = 256,000 samples
```

## 4. GPU、CPU、内存和共享内存

当前数据已经是预编码 Parquet latent，不进行训练时视频解码，因此 CPU 压力低于
在线 VAE，但每个 rank 仍有独立 Python 进程、Parquet cache 和 DataLoader worker。

推荐起始配置：

| GPU 配置 | CPU 核 | RAM | `/dev/shm` | workers/rank | 总 workers |
|---|---:|---:|---:|---:|---:|
| 4×A100 80GB | 32 | 128–256 GiB | ≥64 GiB | 4 | 16 |
| 6×A100 80GB | 48 | 192–384 GiB | ≥96 GiB | 4 | 24 |
| 8×A100 80GB | 64–96 | 256–512 GiB | ≥128 GiB | 4 | 32 |

百舸配置 `1 Worker × 8 A100 80GB + 96 CPU + 512 GiB RAM` 足够，是当前推荐的
8 卡配置。不要把 `workers_per_rank=4` 理解为整机 4 个 worker。

本机检查显示 116 个逻辑 CPU、约 962 GiB RAM、`/dev/shm` 约 482 GiB；但这不
代表百舸训练 Pod 的配额，必须在 Pod 内重新检查。当前 PFS 挂载利用率约 97%，
虽仍有约 28 TiB 可用，也应监控 checkpoint 和新数据集增长。

Pod 内检查：

```bash
nvidia-smi
lscpu | grep -E 'CPU\(s\)|Core|Socket|NUMA'
free -h
df -h /dev/shm /media
ulimit -n
```

若 GPU utilization 经常掉到 0 且 CPU、存储繁忙，可把 `workers_per_rank` 从 4
逐步调到 6 或 8。若出现 `Bus error`、worker 意外退出或 RAM 快速上涨，先降 worker
和 prefetch，再检查 `/dev/shm`，不要盲目继续增大。

## 5. 环境变量

百舸任务页面建议填写：

| 变量 | 值 | 说明 |
|---|---|---|
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | GPU 编号按 PCI 顺序稳定 |
| `NCCL_DEBUG` | `WARN` | 保留必要 NCCL 错误信息 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片 |
| `TOKENIZERS_PARALLELISM` | `false` | 避免 tokenizer 线程噪声 |
| `OMP_NUM_THREADS` | `1` | 避免 rank/worker 线程过量 |
| `MKL_NUM_THREADS` | `1` | 避免 CPU 过度订阅 |
| `COSMOS_METRICS_ENABLE` | `true` | 启用自定义 Prometheus 指标 |
| `COSMOS_METRICS_HOST` | `0.0.0.0` | 允许平台抓取 |
| `COSMOS_METRICS_PORT` | `8000` | Metrics 端口 |
| `COSMOS_METRICS_PATH` | `/metrics` | Metrics 路径 |

`run.sh` 会设置 `PYTHONPATH`、虚拟环境 PATH、allocator 和 tokenizer 变量。
单机模式不需要手工填写 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 或 `MASTER_ADDR`；
`torchrun --standalone` 自动生成。不要手工设置 `LOCAL_RANK`。

单机 NVLink/SXM 环境不要设置 `NCCL_P2P_DISABLE=1`，否则可能关闭高速卡间通信。

若使用百舸自定义监控，在任务页面额外添加 Metrics 端口：

```text
port: 8000
path: /metrics
```

只有 rank 0 监听该端口。日志出现下面内容表示服务已启动：

```text
[MONITOR] Prometheus listening on http://0.0.0.0:8000/metrics
```

## 6. 启动前检查顺序

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project
source /media/jushen/mingbo-ge/.venv/bin/activate

export CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/CONFIG_NAME.yaml

bash HIL-RL/cosmos_offline_train/run.sh env-check
bash HIL-RL/cosmos_offline_train/run.sh preflight
bash HIL-RL/cosmos_offline_train/run.sh test-cpu
bash HIL-RL/cosmos_offline_train/run.sh smoke-single 0
bash HIL-RL/cosmos_offline_train/run.sh smoke-ddp 0,1
```

验收标准：

- Python 来自 `/media/jushen/mingbo-ge/.venv/bin/python`；
- GPU 数量正确；
- checkpoint、tokenizer、T5、stats、train/val manifest 全部存在；
- smoke loss/gradient 有限且 checkpoint 能保存；
- 双卡 smoke 没有 hang、NCCL timeout 或 rank 退出。

## 7. 单机 4/6/8 卡启动命令

直接挂载为项目配置中的绝对路径时：

```bash
# 4卡
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/CONFIG_NAME.yaml \
bash /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3

# 6卡
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/CONFIG_NAME.yaml \
bash /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3,4,5

# 8卡
CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/CONFIG_NAME.yaml \
bash /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3,4,5,6,7
```

`run.sh` 根据逗号数量计算 `--nproc_per_node`。例如 8 个编号会启动 8 个 rank。

恢复训练：

```bash
CONFIG=/absolute/config.yaml \
bash HIL-RL/cosmos_offline_train/run.sh resume 0,1,2,3,4,5,6,7 \
  /persistent/output/checkpoints/step_000001000.pt
```

不要复用另一个实验的 `output_dir`，也不要用不同数据、stats 或基础权重强行恢复。
checkpoint 会校验数据和基础权重身份。

## 8. 百舸只允许填写一条命令时

### 8.1 推荐：挂载路径与 YAML 一致

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project && VENV_DIR=/media/jushen/mingbo-ge/.venv CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/CONFIG_NAME.yaml bash /media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3,4,5,6,7
```

### 8.2 平台实际挂载在 `/media/HIL-RL-Project` 时

先在任务中确认以下路径真实存在：

```text
/media/HIL-RL-Project
/media/install_handle
/media/.venv
```

然后使用兼容软链接。软链接只是让 YAML 中既有绝对路径能够解析，数据仍保存在
原来的 PFS 挂载中：

```bash
mkdir -p /media/jushen/mingbo-ge && ln -sfnT /media/HIL-RL-Project /media/jushen/mingbo-ge/HIL-RL-Project && ln -sfnT /media/install_handle /media/jushen/mingbo-ge/install_handle && cd /media/HIL-RL-Project && VENV_DIR=/media/.venv CONFIG=/media/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/CONFIG_NAME.yaml bash /media/HIL-RL-Project/HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3,4,5,6,7
```

不要在未确认挂载内容时创建链接。更稳妥的长期方案是让百舸挂载路径与 YAML 一致，
或为目标环境维护单独 YAML，而不是依赖软链接。

## 9. 训练状态与输出

rank 0 输出目录包含：

```text
output_dir/
  resolved_config.yaml
  metrics.jsonl
  checkpoints/
    step_000001000.pt
    latest.json
  plots/
```

终端重点观察：

```text
[TRAIN] step=... loss=...
[VAL] step=... loss_actor=...
[MONITOR] Prometheus listening ...
```

Pod 内验证 Prometheus：

```bash
curl -s http://127.0.0.1:8000/metrics | grep cosmos_
```

主要监控指标：

- `cosmos_global_step`
- `cosmos_train_total_edm_loss`
- `cosmos_eval_loss_actor`
- `cosmos_gradient_norm`
- `cosmos_learning_rate`
- `cosmos_samples_per_second`
- `cosmos_step_time_seconds`
- `cosmos_gpu_memory_allocated_gib`

完整 validation 可能包含上千 batch。若每 100 step 都跑完整验证，训练时间会被验证
显著放大。正式实验应根据目标选择频率；画验证曲线时保持所有实验的频率和 split
一致。

## 10. 常见故障

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `No such file or directory` | PFS 挂载路径与 YAML 不一致 | 修正挂载、YAML，或建立已验证软链接 |
| `Cosmos 2B training requires CUDA` | 驱动不可见或环境中的 PyTorch/CUDA 不兼容 | 在 Pod 内检查 `nvidia-smi` 和 `torch.cuda.is_available()` |
| `ChildFailedError` | 这是 torchrun 汇总错误，不是根因 | 向上查找第一个 rank 的原始 traceback |
| NCCL hang/timeout | 某 rank 提前异常、GPU 不健康或通信配置错误 | 查最早失败 rank、`NCCL_DEBUG=INFO` 重跑 smoke |
| CUDA OOM | 单卡 batch 太大、显存碎片或旧进程占用 | batch/rank 保持 1，清理进程，检查 allocator |
| DataLoader bus error | `/dev/shm` 太小 | 增大共享内存或降低 workers/prefetch |
| GPU 利用率低 | CPU/PFS 供数不足或验证过频 | 看吞吐、IO、step time，再调整 workers |
| 多卡 loss 不一致 | 数据、梯度或参数未正确同步 | 检查 DDP 包装、sampler、no_sync 和 all_reduce |

## 11. 多机多卡扩展

当前 `run.sh train` 使用 `--standalone`，只适用于一个 Worker。多机时不能让每个节点
各自执行一套 standalone 任务，需要把启动部分改为：

```bash
torchrun \
  --nnodes="$WORLD_SIZE" \
  --node_rank="$RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m cosmos_offline_train.train \
  --config "$CONFIG" \
  --mode train
```

这里百舸启动层的 `WORLD_SIZE` 是节点数，`RANK` 是节点编号；torchrun 启动 Python
进程后，程序看到的 `WORLD_SIZE` 会变成总 GPU 进程数。多机还需确认节点间 NCCL、
安全组、RDMA/IB 和共享存储；当前项目正式验证过的是单机多卡路径。

