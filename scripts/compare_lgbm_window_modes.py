from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_ROLLING_DIR = REPO_ROOT / "outputs" / "analysis" / "lgbm_operational_multistep_retention"
DEFAULT_FIXED_DIR = REPO_ROOT / "outputs" / "analysis" / "lgbm_operational_multistep_retention_fixed_origin"
DEFAULT_BLOCKS_DIR = REPO_ROOT / "outputs" / "analysis" / "lgbm_operational_multistep_retention_fixed_blocks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "lgbm_operational_multistep_retention_compare_blocks"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare rolling-window, fixed-origin, and fixed-block LightGBM multistep runs.")
    parser.add_argument("--rolling-dir", type=Path, default=DEFAULT_ROLLING_DIR)
    parser.add_argument("--fixed-dir", type=Path, default=DEFAULT_FIXED_DIR)
    parser.add_argument("--blocks-dir", type=Path, default=DEFAULT_BLOCKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Dict[str, object]:
    """Load run_config.json from a completed training directory."""
    path = run_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path, mode_name: str) -> pd.DataFrame:
    """Load metrics CSV and attach a window_mode column."""
    path = run_dir / "train_valid_metrics_by_horizon.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    df.insert(0, "window_mode", mode_name)
    return df


def select_compare_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select the core metric rows used in the comparison report."""
    horizons = {"1", "10", "50", "all"}
    out = metrics_df.loc[
        (metrics_df["set_type"] == "valid")
        & (metrics_df["aggregation"].isin(["weighted", "group_macro"]))
        & (metrics_df["horizon"].astype(str).isin(horizons))
    ].copy()
    return out.sort_values(["window_mode", "aggregation", "horizon"], kind="mergesort")


def extract_window_stats(cfg: Mapping[str, object], mode_name: str) -> Dict[str, object]:
    """Extract effective sample count fields from run_config.json."""
    stats = cfg.get("window_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    args = cfg.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return {
        "window_mode": mode_name,
        "n_history": args.get("n_history", ""),
        "horizon_steps": args.get("horizon_steps", ""),
        "train_windows": stats.get("train_windows", ""),
        "valid_windows": stats.get("valid_windows", ""),
        "groups_train": stats.get("groups_train", ""),
        "groups_valid": stats.get("groups_valid", ""),
        "groups_with_windows_train": stats.get("groups_with_windows_train", ""),
        "groups_with_windows_valid": stats.get("groups_with_windows_valid", ""),
    }


def copy_scatter_image(src: Path, output_dir: Path, name: str) -> str:
    """Copy an image into the report directory and return a local markdown path."""
    if not src.exists():
        raise FileNotFoundError(f"Missing scatter image: {src}")
    dst = output_dir / name
    shutil.copy2(src, dst)
    return f"./{name}"


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R2 or NaN when it is not mathematically meaningful."""
    if y_true.size < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Format MAE/RMSE/R2 for plot annotations."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(float(mean_squared_error(y_true, y_pred))))
    r2 = safe_r2(y_true, y_pred)
    return f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR2={r2:.4f}"


def load_fixed_blocks_valid_predictions(pred_path: Path) -> pd.DataFrame:
    """Load fixed_blocks valid predictions and parse policy from group_key."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {pred_path}")
    pred = pd.read_csv(pred_path, encoding="utf-8")
    required = {"set_type", "group_key", "retention_true", "pred_retention"}
    missing = required.difference(pred.columns)
    if missing:
        raise RuntimeError(f"Prediction CSV missing columns for group scatter: {sorted(missing)}")
    valid = pred.loc[pred["set_type"] == "valid"].copy()
    if valid.empty:
        raise RuntimeError("No valid predictions found for group scatter.")
    valid["policy"] = valid["group_key"].astype(str).str.split("||", regex=False).str[0]
    return valid


def save_weighted_scatter(valid: pd.DataFrame, out_path: Path) -> None:
    """Create weighted scatter using all fixed_blocks valid points."""
    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    yt = valid["retention_true"].to_numpy(dtype=float)
    yp = valid["pred_retention"].to_numpy(dtype=float)
    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(yt, yp, s=8, alpha=0.25, color="#0ea5e9")
    ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", linewidth=1.2)
    ax.set_title("fixed_blocks weighted scatter: all valid horizon points")
    ax.text(0.03, 0.97, metric_text(yt, yp), transform=ax.transAxes, va="top")
    ax.set_xlabel("True retention")
    ax.set_ylabel("Pred retention")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="png", dpi=220)
    plt.close(fig)


def build_policy_group_metrics(valid: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics inside each policy using original prediction points."""
    rows: List[Dict[str, object]] = []
    for policy, part in valid.groupby("policy", sort=False):
        yt = part["retention_true"].to_numpy(dtype=float)
        yp = part["pred_retention"].to_numpy(dtype=float)
        rows.append(
            {
                "policy": str(policy),
                "n_points": int(len(part)),
                "n_cells": int(part["group_key"].nunique()),
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(math.sqrt(float(mean_squared_error(yt, yp)))),
                "r2": safe_r2(yt, yp),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["n_points", "policy"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def save_policy_group_scatter_grid(valid: pd.DataFrame, policy_metrics: pd.DataFrame, out_path: Path) -> None:
    """Create small-multiple scatter plots, one panel per policy."""
    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    n_policy = int(len(policy_metrics))
    n_cols = 4
    n_rows = int(math.ceil(n_policy / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows))
    flat_axes = np.asarray(axes).reshape(-1)
    all_true = valid["retention_true"].to_numpy(dtype=float)
    all_pred = valid["pred_retention"].to_numpy(dtype=float)
    lo = float(min(np.min(all_true), np.min(all_pred)))
    hi = float(max(np.max(all_true), np.max(all_pred)))

    for ax, row in zip(flat_axes, policy_metrics.itertuples(index=False)):
        part = valid.loc[valid["policy"] == row.policy]
        yt = part["retention_true"].to_numpy(dtype=float)
        yp = part["pred_retention"].to_numpy(dtype=float)
        ax.scatter(yt, yp, s=9, alpha=0.45, color="#f97316")
        ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", linewidth=0.9)
        ax.set_title(str(row.policy), fontsize=8)
        r2_text = "nan" if pd.isna(row.r2) else f"{float(row.r2):.3f}"
        ax.text(0.03, 0.97, f"n={int(row.n_points)}\nR2={r2_text}", transform=ax.transAxes, va="top", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in flat_axes[n_policy:]:
        ax.axis("off")
    fig.supxlabel("True retention")
    fig.supylabel("Pred retention")
    fig.suptitle("fixed_blocks valid predictions grouped by policy")
    fig.tight_layout()
    fig.savefig(out_path, format="png", dpi=220)
    plt.close(fig)


def metric_table(metrics_df: pd.DataFrame) -> List[str]:
    """Render selected metrics as a Markdown table."""
    lines = [
        "| window_mode | aggregation | horizon | n_windows | MAE | RMSE | R2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_df.itertuples(index=False):
        lines.append(
            f"| {row.window_mode} | {row.aggregation} | {row.horizon} | {int(row.n_windows)} | "
            f"{float(row.mae):.6f} | {float(row.rmse):.6f} | {float(row.r2):.6f} |"
        )
    return lines


def sample_table(sample_df: pd.DataFrame) -> List[str]:
    """Render effective sample counts as a Markdown table."""
    lines = [
        "| window_mode | N | M | train_windows | valid_windows | train_groups_available | valid_groups_available |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sample_df.itertuples(index=False):
        lines.append(
            f"| {row.window_mode} | {row.n_history} | {row.horizon_steps} | {row.train_windows} | "
            f"{row.valid_windows} | {row.groups_with_windows_train} | {row.groups_with_windows_valid} |"
        )
    return lines


def build_report(
    sample_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    rolling_dir: Path,
    fixed_dir: Path,
    blocks_dir: Path,
    output_dir: Path,
) -> str:
    """Build a Markdown comparison report."""
    rolling_img = copy_scatter_image(
        rolling_dir / "valid_retention_scatter_horizons.png",
        output_dir,
        "rolling_valid_retention_scatter_horizons.png",
    )
    fixed_img = copy_scatter_image(
        fixed_dir / "valid_retention_scatter_horizons.png",
        output_dir,
        "fixed_origin_valid_retention_scatter_horizons.png",
    )
    blocks_img = copy_scatter_image(
        blocks_dir / "valid_retention_scatter_horizons.png",
        output_dir,
        "fixed_blocks_valid_retention_scatter_horizons.png",
    )
    valid_pred = load_fixed_blocks_valid_predictions(blocks_dir / "train_valid_predictions_long.csv")
    policy_metrics = build_policy_group_metrics(valid_pred)
    policy_metrics.to_csv(output_dir / "fixed_blocks_policy_group_metrics.csv", index=False, encoding="utf-8")
    weighted_img = "./fixed_blocks_weighted_scatter.png"
    policy_group_img = "./fixed_blocks_policy_group_scatter_grid.png"
    save_weighted_scatter(valid_pred, output_dir / "fixed_blocks_weighted_scatter.png")
    save_policy_group_scatter_grid(
        valid=valid_pred,
        policy_metrics=policy_metrics,
        out_path=output_dir / "fixed_blocks_policy_group_scatter_grid.png",
    )
    policy_macro = {
        "mae": float(policy_metrics["mae"].mean()),
        "rmse": float(policy_metrics["rmse"].mean()),
        "r2": float(policy_metrics["r2"].mean()),
    }

    lines: List[str] = []
    lines.append("# LightGBM 滑窗 vs 固定起点 vs 分段固定起点对比报告")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 滑窗口径：每个电芯生成多个 `s:s+N-1 -> s+N:s+N+M-1` 样本。")
    lines.append("- 固定起点口径：每个电芯最多 1 个 `1:N -> N+1:N+M` 样本。")
    lines.append("- 分段固定起点口径：每个电芯最多 3 个样本，默认 block 起点为 `1,151,301`。")
    lines.append("- 固定起点和分段固定起点均使用 `min_child_samples=5` 适配小样本；滑窗正式结果复用已有产物。")
    lines.append("")
    lines.append("## 2. 有效样本数量")
    lines.extend(sample_table(sample_df))
    lines.append("")
    lines.append("## 3. 验证集预测精度")
    lines.extend(metric_table(metrics_df))
    lines.append("")
    lines.append("## 4. 散点图")
    lines.append("")
    lines.append("本节三张散点图分别对应三种样本构造口径，横轴均为真实 retention，纵轴均为预测 retention，红色虚线表示理想预测 `y=x`。三张图都展示验证集在 `h=1`、`h=10`、`h=50` 三个 horizon 下的预测效果，但每张图中的样本来源不同。")
    lines.append("")
    lines.append("- `rolling_valid_scatter`：滑窗口径。每个电芯会产生大量滚动窗口，例如 `s:s+N-1 -> s+N:s+N+M-1`。因此点数最多，覆盖的退化阶段也最连续；但相邻样本高度重叠，图上表现和 weighted 指标可能偏乐观。")
    lines.append("- `fixed_origin_valid_scatter`：单固定起点口径。每个电芯最多只有 1 个样本，即 `1:100 -> 101:150`。它最接近严格早期预测任务，点数最少，图上的离散程度更容易受少数电芯影响。")
    lines.append("- `fixed_blocks_valid_scatter`：分段固定起点口径。每个电芯最多有 3 个样本，即 `1:100 -> 101:150`、`151:250 -> 251:300`、`301:400 -> 401:450`。它在单固定起点和滑窗之间折中，样本量增加，但不会像滑窗那样引入大量相邻重叠窗口。")
    lines.append("")
    lines.append("读图时需要把三张图和第 3 章指标一起看：滑窗图点多且整体 R2 高，说明模型能学习滚动退化动态；固定起点图更稀疏，反映严格早期预测能力；分段固定起点图用于观察扩样后是否改善了固定起点的泛化。")
    lines.append("")
    lines.append(f"![rolling_valid_scatter]({rolling_img})")
    lines.append(f"![fixed_origin_valid_scatter]({fixed_img})")
    lines.append(f"![fixed_blocks_valid_scatter]({blocks_img})")
    lines.append("")
    lines.append("## 5. Weighted 与按 Policy 分组散点图")
    lines.append(f"![fixed_blocks_weighted_scatter]({weighted_img})")
    lines.append(f"![fixed_blocks_policy_group_scatter_grid]({policy_group_img})")
    lines.append("")
    lines.append("说明：第一张图是 weighted 口径，即 fixed_blocks 验证集所有原始 horizon 点直接参与散点和指标。第二张图是按 policy 分组的小图矩阵，每个子图仍使用该 policy 下的原始预测点计算 R2，不再对 `policy+cell_code` 做均值点替代。")
    lines.append(
        f"按 policy 分组后的 policy-macro 指标为：MAE={policy_macro['mae']:.6f}，RMSE={policy_macro['rmse']:.6f}，R2={policy_macro['r2']:.6f}。完整明细见 `fixed_blocks_policy_group_metrics.csv`。"
    )
    lines.append("")
    lines.append("## 6. 结论说明")
    lines.append("- 滑窗样本量远大于固定起点，适合学习滚动退化动态，但 window-weighted 指标可能偏乐观。")
    lines.append("- 固定起点更接近严格早期预测任务，但 train/valid 样本只有电芯组数量级，指标更容易受单个电芯影响。")
    lines.append("- 分段固定起点在固定历史窗口和低重叠之间折中，样本量高于单固定起点，低于滑窗。")
    lines.append("- 分段固定起点的 weighted R2 为正而按 policy 分组后的 R2 可能偏低或为负，主要因为分组后在每个 policy 内部单独计算 R2。部分 policy 内真实 retention 波动很小，SST 很小，轻微系统误差就会导致 SSE > SST，从而得到负 R2。")
    lines.append("- weighted R2 把所有点合并计算，包含不同电芯之间的 retention 差异，整体 SST 更大，因此更容易得到较高 R2。")
    lines.append("- 固定起点下每个电芯最多一个窗口，MAE/RMSE 的 weighted 与 group_macro 通常更接近。")
    lines.append("- 固定起点的单 horizon group-macro R2 不可定义，因为每个电芯组在该 horizon 只有 1 个点；`all` horizon 的 group-macro R2 仍可参考。")
    lines.append("- 本对比重点是样本构造口径差异，不是严格超参公平竞赛。")
    return "\n".join(lines)


def main() -> None:
    """Generate comparison CSV and Markdown report."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rolling_cfg = load_run_config(args.rolling_dir)
    fixed_cfg = load_run_config(args.fixed_dir)
    blocks_cfg = load_run_config(args.blocks_dir)
    sample_df = pd.DataFrame(
        [
            extract_window_stats(rolling_cfg, "rolling"),
            extract_window_stats(fixed_cfg, "fixed_origin"),
            extract_window_stats(blocks_cfg, "fixed_blocks"),
        ]
    )
    metrics_df = pd.concat(
        [
            load_metrics(args.rolling_dir, "rolling"),
            load_metrics(args.fixed_dir, "fixed_origin"),
            load_metrics(args.blocks_dir, "fixed_blocks"),
        ],
        axis=0,
        ignore_index=True,
    )
    compare_metrics_df = select_compare_metrics(metrics_df)

    sample_path = args.output_dir / "sample_count_comparison.csv"
    metric_path = args.output_dir / "valid_metric_comparison.csv"
    report_path = args.output_dir / "report.md"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    compare_metrics_df.to_csv(metric_path, index=False, encoding="utf-8")
    report = build_report(
        sample_df=sample_df,
        metrics_df=compare_metrics_df,
        rolling_dir=args.rolling_dir,
        fixed_dir=args.fixed_dir,
        blocks_dir=args.blocks_dir,
        output_dir=args.output_dir,
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved: {sample_path}")
    print(f"Saved: {metric_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
