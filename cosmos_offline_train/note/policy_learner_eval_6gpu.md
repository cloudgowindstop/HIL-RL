# Install Handle Policy 六卡训练

配置文件：`configs/install_handle_euler_14d_policy_learner_eval.yaml`

训练设定：

- 数据：`install_handle/train` 209 episodes，`install_handle/eval` 11 episodes。
- 动作：14D legacy Euler，`rotation_scale=0.06`。
- 目标：100% Policy；预测 action、future state、value。
- 有效 batch：6 GPUs × 1 sample/GPU × 21 次梯度累积 = 126 samples/update。
- 训练：最多 2,000 optimizer steps，最多 3 epochs。
- 验证：Base 与每 100 steps，在完整 eval split 上顺序评估；固定随机种子，使用训练噪声分布。
- 保存：step 1,000、step 2,000。
- 监控：rank 0 暴露 `http://0.0.0.0:8000/metrics`，训练指标每 5 steps 更新。

六卡 smoke test：

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project && CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/install_handle_euler_14d_policy_learner_eval.yaml bash HIL-RL/cosmos_offline_train/run.sh smoke-ddp 0,1,2,3,4,5
```

六卡正式训练：

```bash
cd /media/jushen/mingbo-ge/HIL-RL-Project && CONFIG=/media/jushen/mingbo-ge/HIL-RL-Project/HIL-RL/cosmos_offline_train/configs/install_handle_euler_14d_policy_learner_eval.yaml bash HIL-RL/cosmos_offline_train/run.sh train 0,1,2,3,4,5
```

`run.sh` 自动激活 `/media/jushen/mingbo-ge/.venv`，设置 `PYTHONPATH`、`PYTORCH_CUDA_ALLOC_CONF`、`TOKENIZERS_PARALLELISM`，并通过 `torchrun --standalone --nproc_per_node 6` 启动六个 DDP rank。

训练输出：

- 日志：`train_outputs/install_handle_euler_14d_policy_learner_eval/metrics.jsonl`
- 权重：`train_outputs/install_handle_euler_14d_policy_learner_eval/checkpoints/`
- 主曲线：`train_outputs/install_handle_euler_14d_policy_learner_eval/plots/train_eval_edm_loss.png`
- 验证表：`train_outputs/install_handle_euler_14d_policy_learner_eval/plots/eval_metrics.csv`

完整绘图命令和输出解释见 [`plotting_guide.md`](plotting_guide.md)。

百舸任务需增加 Metrics 端口：端口 `8000`，路径 `/metrics`。可用环境变量 `COSMOS_METRICS_ENABLE`、`COSMOS_METRICS_HOST`、`COSMOS_METRICS_PORT`、`COSMOS_METRICS_PATH` 覆盖 YAML。
