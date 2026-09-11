import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(
    "/media/HIL-RL-Project/HIL-RL/experiments/0512_cosmos/demo_action_l1_summary.csv"
)

# 设置要绘制的损失列
loss_columns = [
    "max_demo_action_l1",
    "min_demo_action_l1",
    "mean_demo_action_l1",
    # "cosmos_training_loss",
]

# 横轴范围（optimization_step）；任一端为 None 则用该端对应的数据 min/max
X_MIN = None
X_MAX = None

fig, ax = plt.subplots(figsize=(12, 6))
x = df["optimization_step"]

# 绘制曲线并记录颜色与 CSV 末行收敛值
line_meta: list[tuple[str, str, float]] = []
for col in loss_columns:
    (line,) = ax.plot(x, df[col], label=col)
    s = df[col].dropna()
    last_val = float(s.iloc[-1]) if len(s) else float("nan")
    line_meta.append((col, line.get_color(), last_val))

_xmin = float(x.min()) if X_MIN is None else X_MIN
_xmax = float(x.max()) if X_MAX is None else X_MAX
ax.set_xlim(_xmin, _xmax)

# 灰色虚线横线：各指标在 CSV 最后一行的数值
for _col, _color, last_val in line_meta:
    if last_val != last_val:  # NaN
        continue
    ax.axhline(
        last_val,
        color="0.55",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        zorder=0,
    )

# 在图右侧用与曲线同色标注最终收敛值（横轴缩小后取可见右边界）
x_lo, x_hi = ax.get_xlim()
x_last = float(x.iloc[-1])
x_end = x_last if x_lo <= x_last <= x_hi else x_hi
for _col, color, last_val in line_meta:
    if last_val != last_val:
        continue
    ax.annotate(
        f"final: {last_val:.6g}",
        xy=(x_end, last_val),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9,
        color=color,
        va="center",
        clip_on=False,
    )

# 图例中写出各曲线最终值
legend_labels = [
    f"{col}\n(final={last_val:.6g})" if last_val == last_val else col
    for col, _c, last_val in line_meta
]
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, legend_labels, loc="upper right", fontsize=8)


ax.set_xlabel("Optimization Step")
ax.set_ylabel("L1 Loss")
ax.set_title("Demo Sample L1 Losses vs Optimization Step（灰色虚线 = CSV 末行收敛值）")
ax.grid(True, linestyle="--", alpha=0.6)
fig.tight_layout()

plt.savefig("loss_curve_0512.png", dpi=300, bbox_inches="tight")
plt.show()
