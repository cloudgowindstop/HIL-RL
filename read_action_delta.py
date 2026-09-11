#!/usr/bin/env python3
"""读取 action_delta.npy 并打印"""

import sys
import numpy as np

f = sys.argv[1] if len(sys.argv) > 1 else "pose_delta/pick_spoon/ep_000000/action_delta.npy"
data = np.load(f)

print(f"文件: {f}")
print(f"shape: {data.shape}  (frames, dims)")
print(f"dtype: {data.dtype}")
print(f"列:   dx(m)  dy(m)  dz(m)  rx(rad)  ry(rad)  rz(rad)  dgrip")
print()

n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"前 {n} 帧:")
for i in range(min(n, len(data))):
    vals = " ".join(f"{v:+.6f}" for v in data[i])
    print(f"  [{i:04d}] {vals}")

if len(data) > n * 2:
    print(f"  ... ({len(data) - n * 2} 帧省略)")

if len(data) > n:
    print(f"后 {n} 帧:")
    for i in range(max(n, len(data) - n), len(data)):
        vals = " ".join(f"{v:+.6f}" for v in data[i])
        print(f"  [{i:04d}] {vals}")
