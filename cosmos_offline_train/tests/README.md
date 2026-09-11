# 大规模训练前测试

当前 CPU 测试入口：

```text
run_cpu_tests.py
    rotation-6D SO(3)、round-trip、geodesic error
    policy/world/value/failure mask 与有效分母
    task-stratified episode split、固定 8/2 smoke manifest、拒绝覆盖、两 rank sampler、set_epoch
    Lazy Dataset 按需读取
    model/optimizer/scheduler/scaler/RNG checkpoint 恢复及数据身份不匹配拒绝
    预编码 latent EDM forward 和 20D action extraction
```

运行：

```bash
bash cosmos_offline_train/run.sh test-cpu
```

GPU 测试由 trainer 内置：`smoke-single` 和 `smoke-ddp` 重复同一个 batch、固定
diffusion noise、检查 loss/gradient 有限及最终 loss 下降，并保存可恢复 checkpoint。
真实 2B smoke 仍需要 converted 20D 数据、T5、statistics 和 checkpoint。
