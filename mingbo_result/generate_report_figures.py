#!/usr/bin/env python3
"""Generate figures used only by the final Word report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures/compare/fig2_translation_2x2_report.png"


def main() -> None:
    labels = ["E-P", "E-J", "6D-P", "6D-J"]
    colors = ["#174F7A", "#5D91BC", "#BF5A34", "#D8A36D"]
    left = [0.0281, 0.0351, 0.0494, 0.0527]
    right = [0.0162, 0.0168, 0.0154, 0.0170]

    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=180)
    x = np.arange(2)
    width = 0.18
    for index, (label, color) in enumerate(zip(labels, colors)):
        values = [left[index], right[index]]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=2, fontsize=8)

    ax.set_ylabel("Translation MAE")
    ax.set_xticks(x, ["Left arm", "Right arm"])
    ax.set_ylim(0, 0.058)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.11, 0.015, "step 2000 · σ=0.5 · lower is better", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
