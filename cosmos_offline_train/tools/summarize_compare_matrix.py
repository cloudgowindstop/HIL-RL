"""Export the three action-representation compare tables from finished runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUNS = (
    ("E-P", "euler_14d_policy"),
    ("6D-P", "rotation6d_20d_policy"),
    ("6D-J", "rotation6d_20d_joint"),
    ("6D-J-ID", "rotation6d_20d_joint_id"),
)

POLICY_METRICS = (
    ("left_rotation_geodesic_deg", "左臂旋转 (°)"),
    ("right_rotation_geodesic_deg", "右臂旋转 (°)"),
    ("left_translation_mae", "左臂平移 MAE"),
    ("right_translation_mae", "右臂平移 MAE"),
    ("left_gripper_accuracy", "左夹爪 accuracy"),
    ("right_gripper_accuracy", "右夹爪 accuracy"),
    ("left_horizon_16_rotation_geodesic_deg", "左臂 horizon-16 旋转 (°)"),
    ("right_horizon_16_rotation_geodesic_deg", "右臂 horizon-16 旋转 (°)"),
    ("left_horizon_16_translation_mae", "左臂 horizon-16 平移"),
    ("right_horizon_16_translation_mae", "右臂 horizon-16 平移"),
)

AUX_METRICS = (
    ("sigma_0.5/world/future_image_l1", "World 主相机 L1"),
    ("sigma_0.5/world/future_wrist_image_l1", "World 腕部 L1"),
    ("sigma_0.5/world/future_proprio_l1", "World 本体 L1"),
    ("sigma_0.5/world/value_l1", "World value L1"),
    ("sigma_0.5/value/value_scalar_mae", "Value 标量 MAE"),
    ("sigma_0.5/value/value_spearman", "Value Spearman"),
    ("sigma_0.5/value/value_l1", "Value latent L1"),
)

ID_GAIN_METRICS = (
    ("action_physical_mae", "动作物理 MAE"),
    ("left_rotation_geodesic_deg", "左臂旋转 (°)"),
    ("right_rotation_geodesic_deg", "右臂旋转 (°)"),
    ("left_translation_mae", "左臂平移 MAE"),
    ("right_translation_mae", "右臂平移 MAE"),
    ("left_horizon_16_rotation_geodesic_deg", "左臂 horizon-16 旋转 (°)"),
    ("right_horizon_16_rotation_geodesic_deg", "右臂 horizon-16 旋转 (°)"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_end_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the last full-suite validation at the highest step."""
    candidates = [
        record
        for record in records
        if record.get("mode") in {"eval", "validation"}
        and any(str(key).startswith("sigma_") and "/policy/" in str(key) for key in record)
    ]
    if not candidates:
        raise ValueError("no policy validation records")
    return max(
        candidates,
        key=lambda record: (
            int(record.get("global_step") or 0),
            sum(1 for key in record if str(key).startswith("sigma_")),
        ),
    )


def overlay_summary(record: dict[str, Any], summary_path: Path | None) -> dict[str, Any]:
    """Fill missing inverse-dynamics keys from a later validate summary.

    Do not overwrite policy/world/value keys already logged in metrics.jsonl.
    """
    if summary_path is None or not summary_path.is_file():
        return record
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return record
    merged = dict(record)
    for key, value in payload.items():
        if "/inverse_dynamics/" not in str(key):
            continue
        if merged.get(key) is None:
            merged[key] = value
    return merged


def _number(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def policy_key(name: str) -> str:
    return f"sigma_0.5/policy/{name}"


def extract_policy_row(name: str, record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"run": name, "step": int(record.get("global_step") or 0)}
    for key, _label in POLICY_METRICS:
        row[key] = _number(record, policy_key(key))
    for key, _label in AUX_METRICS:
        row[key] = _number(record, key)
    return row


def extract_id_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label in ID_GAIN_METRICS:
        normal = _number(record, f"sigma_0.5/inverse_dynamics/normal/{key}")
        shuffled = _number(record, f"sigma_0.5/inverse_dynamics/future_shuffle/{key}")
        gain = _number(
            record, f"sigma_0.5/inverse_dynamics/future_information_gain/{key}"
        )
        if gain is None and normal is not None and shuffled is not None:
            gain = shuffled - normal
        rows.append(
            {
                "metric": key,
                "label": label,
                "normal": normal,
                "future_shuffle": shuffled,
                "delta_future": gain,
            }
        )
    return rows


def _fmt(value: float | None, *, accuracy: bool = False) -> str:
    if value is None:
        return "—"
    if accuracy:
        return f"{value:.3f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def markdown_policy_table(title: str, rows: list[dict[str, Any]]) -> str:
    headers = ["指标", *[row["run"] for row in rows]]
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key, label in POLICY_METRICS:
        accuracy = key.endswith("accuracy")
        cells = [_fmt(row.get(key), accuracy=accuracy) for row in rows]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def markdown_aux_table(rows: list[dict[str, Any]]) -> str:
    headers = ["指标", *[row["run"] for row in rows]]
    lines = [
        "### 表 2 补充：World / Value latent",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key, label in AUX_METRICS:
        cells = [_fmt(row.get(key)) for row in rows]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def markdown_id_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "### 表 3：6D-J-ID 的 Δ_future（shuffle − normal）",
        "",
        "| 指标 | ID normal | ID future_shuffle | Δ_future |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    _fmt(row["normal"]),
                    _fmt(row["future_shuffle"]),
                    _fmt(row["delta_future"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Δ_future > 0 表示打乱未来后误差变大，ID 用到了未来状态。",
            "horizon-16 是 chunk 第 16 槽，不是闭环 16 步。不要用 14D/20D raw MSE 做主结论。",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


VIZ_RUNS = (
    ("Euler-base", "visualize/euler_base"),
    ("6D-base", "visualize/rotation6d_base"),
    ("E-P", "visualize/euler_14d_policy"),
    ("6D-P", "visualize/rotation6d_20d_policy"),
    ("6D-J", "visualize/rotation6d_20d_joint"),
    ("6D-J-ID", "visualize/rotation6d_20d_joint_id"),
)

VIZ_METRICS = (
    ("future_cameras_psnr", "未来相机 PSNR"),
    ("future_primary_psnr", "主相机 PSNR"),
    ("future_wrist_psnr", "腕部 PSNR"),
    ("future_cameras_ssim", "未来相机 SSIM"),
    ("future_primary_ssim", "主相机 SSIM"),
    ("future_wrist_ssim", "腕部 SSIM"),
)


def collect_world_rgb(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, folder in VIZ_RUNS:
        summary_path = root / folder / "summary.json"
        if not summary_path.is_file():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if not isinstance(metrics, dict):
            continue
        row: dict[str, Any] = {
            "run": name,
            "folder": folder,
            "sample_count": payload.get("sample_count"),
        }
        for key, _label in VIZ_METRICS:
            value = metrics.get(key)
            row[key] = float(value) if isinstance(value, (int, float)) else None
        rows.append(row)
    return rows


def markdown_world_rgb(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "### 表 4：World RGB（尚未生成）"
    headers = ["指标", *[row["run"] for row in rows]]
    lines = [
        "### 表 4：World RGB（当前观测 + GT action → 未来相机）",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key, label in VIZ_METRICS:
        cells = []
        for row in rows:
            value = row.get(key)
            if value is None:
                cells.append("—")
            elif "psnr" in key:
                cells.append(f"{value:.2f}")
            else:
                cells.append(f"{value:.3f}")
        lines.append("| " + " | ".join([label, *cells]) + " |")
    lines.extend(
        [
            "",
            "冻结 8 episode × 2 行。PSNR/SSIM 越高越好。base 是未训练 Predict2 权重。",
        ]
    )
    return "\n".join(lines)


def summarize(root: Path, output_dir: Path) -> dict[str, Any]:
    policy_rows: list[dict[str, Any]] = []
    records_by_run: dict[str, dict[str, Any]] = {}
    for name, folder in RUNS:
        metrics_path = root / folder / "metrics.jsonl"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        record = select_end_validation(load_jsonl(metrics_path))
        record = overlay_summary(
            record, root / folder / "validation_summary_step_000001000.json"
        )
        records_by_run[name] = record
        policy_rows.append(extract_policy_row(name, record))
    id_rows = extract_id_table(records_by_run["6D-J-ID"])
    world_rgb_rows = collect_world_rgb(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_root": str(root),
        "table1_e_p_vs_6d_p": [row for row in policy_rows if row["run"] in {"E-P", "6D-P"}],
        "table2_6d_p_vs_6d_j": [row for row in policy_rows if row["run"] in {"6D-P", "6D-J", "6D-J-ID"}],
        "table3_id_delta_future": id_rows,
        "table4_world_rgb": world_rgb_rows,
        "all_policy": policy_rows,
    }
    (output_dir / "compare_matrix.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "table1_policy_physical.csv",
        payload["table1_e_p_vs_6d_p"],
        ["run", "step", *[key for key, _ in POLICY_METRICS]],
    )
    write_csv(
        output_dir / "table2_joint_physical.csv",
        payload["table2_6d_p_vs_6d_j"],
        ["run", "step", *[key for key, _ in POLICY_METRICS], *[key for key, _ in AUX_METRICS]],
    )
    write_csv(
        output_dir / "table3_id_delta_future.csv",
        id_rows,
        ["metric", "label", "normal", "future_shuffle", "delta_future"],
    )
    if world_rgb_rows:
        write_csv(
            output_dir / "table4_world_rgb.csv",
            world_rgb_rows,
            ["run", "folder", "sample_count", *[key for key, _ in VIZ_METRICS]],
        )
    markdown = "\n\n".join(
        [
            "# action_representation_20260903 主表",
            "来源：各 run `metrics.jsonl` 最后一次 full-suite validation（step 1000, σ=0.5）。"
            "6D-J-ID 的 ID 键可从 `validation_summary_step_000001000.json` 补齐。"
            "不要用 14D/20D raw MSE 做主结论。horizon-16 是 chunk 第 16 槽，不是闭环 16 步。",
            markdown_policy_table("表 1：E-P vs 6D-P（动作表示）", payload["table1_e_p_vs_6d_p"]),
            "E-P 旋转测地线接近 0°，因为增量 Euler × `rotation_scale=0.06` 本身很小；"
            "不要据此说 Euler 旋转已经学完美。表 1 主看平移 MAE 和夹爪 accuracy。",
            markdown_policy_table("表 2：6D-P vs 6D-J vs 6D-J-ID（训练目标）", payload["table2_6d_p_vs_6d_j"]),
            markdown_aux_table(payload["table2_6d_p_vs_6d_j"]),
            markdown_id_table(id_rows),
            markdown_world_rgb(world_rgb_rows),
        ]
    )
    (output_dir / "compare_matrix.md").write_text(markdown + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export E-P / 6D-P / 6D-J / 6D-J-ID compare tables")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/media/jushen/mingbo-ge/HIL-RL-Project/train_outputs/action_representation_20260903"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output or (args.root / "compare_matrix")
    payload = summarize(args.root.expanduser().resolve(), output.expanduser().resolve())
    print(f"[COMPARE] wrote {output}")
    print((output / "compare_matrix.md").read_text(encoding="utf-8"))
    _ = payload


if __name__ == "__main__":
    main()
