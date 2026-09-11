# 合成 HDF5 与真实 VAE 回归测试

本目录提供一条字段结构兼容当前读取器、所有低维数值都可独立计算的单臂轨迹。它用于检查生产转换流程，而不是另写一套转换实现。

## 合成轨迹

轨迹固定为 20 帧、30 FPS。每一步末端在自身坐标系沿 X 轴移动 0.01 m，同时绕 Z 轴旋转 0.03 rad。夹爪从 0 线性变化到 1。两路 480×640 RGB 图像分别使用不同的红色时间标记，绿色和蓝色通道保存水平、竖直坐标渐变。

测试参数固定为：

- `action_source=puppet_next_frame`
- `action_encoding=legacy_euler`
- `translation_scale=0.02`
- `rotation_scale=0.06`
- `gripper_scale=1.0`
- `chunk_size=16`
- `gamma=0.99`
- 成功 episode

因此除浮点舍入外，每个第一层有效 action 都应包含 `dx=0.5`、`rz=0.5`，最后一帧重复最后一个有效 action。测试使用每个通道不同的非恒等 min/max stats，第二层归一化后的各通道值不同，用于发现stats通道错位、漏归一化和gripper范围错误。

测试实现严格分成三层：

- `scenario_spec.py`只保存原始常量和stats字面值；
- `hdf5_generator.py`使用齐次矩阵递推生成HDF5，不导入期望值模块；
- `expected_values.py`使用闭式三角公式计算参考值，不读取HDF5，也不导入生成器。

## 快速 CPU 测试

在项目根目录执行：

```bash
export PYTHONPATH="$PWD/cosmos-policy:$PWD/cosmos-policy/cosmos_policy:$PWD/HIL-RL/lerobot/src:$PWD/HIL-RL:$PWD/HIL-RL/data_convert"
/media/jushen/mingbo-ge/.venv/bin/python -m unittest \
  data_convert_refactored.tests.synthetic.test_synthetic_conversion -v
```

该测试逐元素检查 HDF5 读取、末端 pose、gripper、action、proprio、reward、done、action chunk、future 索引、Monte Carlo return、图像时序标记和独立 latent 注入公式。它不加载 VAE。

## 真实 VAE 端到端测试

在可访问 CUDA GPU 的转换容器中执行：

```bash
bash HIL-RL/data_convert_refactored/scripts/test_synthetic_real_vae.sh \
  --work-dir /tmp/cosmos_synthetic_real_vae_check \
  --encode-batch-size 4
```

测试先直接调用正式 VAE 得到基准 latent，再执行完整 `ConversionPipeline` 并读取生成的 Parquet。最终逐元素比较：

- action；
- proprio；
- future proprio；
- value function return；
- reward；
- done；
- 注入后的完整 `(20,16,9,28,28)` VAE latent。

指定的 `--work-dir` 必须不存在，避免覆盖已有测试结果。未指定时会在 `/tmp` 下创建带进程号的新目录。

真实 VAE 数值无法手工推导。此测试将同一确定性视频直接送入真实 VAE，并用独立 NumPy 公式计算所有低维条件和注入后的最终 latent。它同时验证 VAE 调用、条件注入和 Parquet 保存路径。
