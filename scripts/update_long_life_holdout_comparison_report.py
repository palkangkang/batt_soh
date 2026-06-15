"""Refresh the long-life holdout LightGBM/LSTM comparison report."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODING = "utf-8-sig"
REPORT_PATH = REPO_ROOT / "outputs" / "analysis" / "long_life_holdout_lgbm_lstm_blocks_h100_m50_comparison.md"
FIGURE_DIR = REPO_ROOT / "outputs" / "analysis" / "long_life_holdout_lgbm_lstm_blocks_h100_m50_figures"
ROUTE_DIR = FIGURE_DIR / "route_diagrams"
ROUTE_IMAGE_FILES = (
    "route_trend_baseline_gpt_image2.png",
    "route_lightgbm_gpt_image2.png",
    "route_lstm_pure_operational_gpt_image2.png",
    "route_lstm_history_retention_gpt_image2.png",
    "route_last_retention_only_ablation.png",
)

METHOD_SPECS: List[Dict[str, str]] = []


def configure_method_specs(history_len: int, include_last_only: bool = False) -> None:
    """Configure method labels that depend on history length."""

    global METHOD_SPECS
    h = int(history_len)
    METHOD_SPECS = [
        {
            "raw_method": "linear_last10",
            "label": "linear_last10",
            "plot_label": "linear_last10",
            "route": "trend baseline",
            "input": "历史最后10点 retention 线性外推",
            "uses_history_retention": "是",
            "claim": "低成本强基线",
        },
        {
            "raw_method": "persistence",
            "label": "persistence",
            "plot_label": "persistence",
            "route": "trend baseline",
            "input": "历史最后一个 retention",
            "uses_history_retention": "是",
            "claim": "低成本基线",
        },
        {
            "raw_method": "direct_retention",
            "label": "LightGBM direct",
            "plot_label": "LightGBM direct",
            "route": "LightGBM",
            "input": "55维工况 summary",
            "uses_history_retention": "否",
            "claim": "纯工况 tabular",
        },
        {
            "raw_method": "direct_retention_with_history_summary",
            "label": "LightGBM + history retention summary",
            "plot_label": "LightGBM + retention summary",
            "route": "LightGBM enhanced",
            "input": "55维工况 summary + 7维历史retention summary",
            "uses_history_retention": "是",
            "claim": "历史retention增强 tabular",
        },
        {
            "raw_method": "direct_retention_last_only",
            "label": "LightGBM last retention only",
            "plot_label": "LightGBM last-only",
            "route": "last retention only ablation",
            "input": "last retention标量",
            "uses_history_retention": "是，仅last",
            "claim": "仅last retention消融 tabular",
        },
        {
            "raw_method": "monotonic_lstm_delta_strict",
            "label": f"LSTM delta strict {h}x55",
            "plot_label": f"LSTM delta strict {h}x55",
            "route": "LSTM pure operational",
            "input": f"{h}x55工况序列 + last retention递推起点",
            "uses_history_retention": "否",
            "claim": "纯工况序列主对照",
        },
        {
            "raw_method": "monotonic_lstm_delta_with_history_retention",
            "label": f"LSTM delta {h}x56 history-retention-enhanced",
            "plot_label": f"LSTM delta {h}x56 + retention",
            "route": "LSTM enhanced",
            "input": f"{h}x55工况序列 + 历史retention通道",
            "uses_history_retention": "是",
            "claim": "历史retention增强序列",
        },
        {
            "raw_method": "monotonic_lstm_delta_last_retention_only",
            "label": "LSTM delta 1x1 last retention only",
            "plot_label": "LSTM last-only",
            "route": "last retention only ablation",
            "input": "1x1 last retention标量",
            "uses_history_retention": "是，仅last",
            "claim": "仅last retention消融 LSTM",
        },
    ]
    if not bool(include_last_only):
        METHOD_SPECS = [
            spec
            for spec in METHOD_SPECS
            if spec["raw_method"] not in {"direct_retention_last_only", "monotonic_lstm_delta_last_retention_only"}
        ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Refresh long_life_holdout LightGBM/LSTM comparison report.")
    parser.add_argument(
        "--lgbm-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "long_life_holdout_lgbm_blocks_h100_m50",
    )
    parser.add_argument(
        "--lgbm-history-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "long_life_holdout_lgbm_history_retention_blocks_h100_m50",
    )
    parser.add_argument(
        "--lstm-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "long_life_holdout_lstm_blocks_h100_m50",
    )
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--route-dir", type=Path, default=ROUTE_DIR)
    parser.add_argument("--lgbm-last-only-dir", type=Path, default=None)
    parser.add_argument("--lstm-last-only-dir", type=Path, default=None)
    return parser.parse_args()


def infer_run_meta(args: argparse.Namespace) -> Dict[str, object]:
    """Infer shared experiment metadata from the LightGBM-history run config."""

    config = read_json(args.lgbm_history_dir / "run_config.json")
    history_len = int(config.get("history_len", 100))
    horizon = int(config.get("horizon", 50))
    selected = ["H10", "H50", f"H{horizon}", "ALL"]
    selected = list(dict.fromkeys([item for item in selected if item == "ALL" or int(item.replace("H", "")) <= horizon]))
    return {
        "history_len": history_len,
        "horizon": horizon,
        "block_stride": int(config.get("block_stride", history_len + horizon)),
        "sample_mode": str(config.get("sample_mode", "non_overlapping_blocks")),
        "endpoint_horizon": f"H{horizon}",
        "selected_horizons": selected,
        "split_name": str(config.get("split_name", "long_life_holdout")),
        "train_split_path": str(config.get("train_split_path", "")),
        "valid_split_path": str(config.get("valid_split_path", "")),
        "baseline_source": str(read_json(args.lstm_dir / "run_config.json").get("baseline_source", "")),
    }


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with the project encoding and a clear error."""

    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return pd.read_csv(path, encoding=ENCODING)


def read_json(path: Path) -> Dict[str, object]:
    """Read a JSON file as a dictionary."""

    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(args: argparse.Namespace) -> pd.DataFrame:
    """Load and combine the selected valid-set retention metrics."""

    frames: List[pd.DataFrame] = []
    lgbm_metrics = read_csv(args.lgbm_dir / "retention_multistep_metrics.csv")
    frames.append(lgbm_metrics.loc[lgbm_metrics["method"].isin(["direct_retention", "persistence", "linear_last10"])].copy())

    lgbm_history_metrics = read_csv(args.lgbm_history_dir / "retention_multistep_metrics.csv")
    frames.append(
        lgbm_history_metrics.loc[
            lgbm_history_metrics["method"].isin(["direct_retention_with_history_summary"])
        ].copy()
    )
    if args.lgbm_last_only_dir is not None:
        lgbm_last_metrics = read_csv(args.lgbm_last_only_dir / "retention_multistep_metrics.csv")
        frames.append(
            lgbm_last_metrics.loc[lgbm_last_metrics["method"].isin(["direct_retention_last_only"])].copy()
        )

    lstm_metrics = read_csv(args.lstm_dir / "train_valid_metrics_by_horizon.csv")
    frames.append(
        lstm_metrics.loc[
            lstm_metrics["method"].isin(
                ["monotonic_lstm_delta_strict", "monotonic_lstm_delta_with_history_retention"]
            )
        ].copy()
    )
    if args.lstm_last_only_dir is not None:
        lstm_last_metrics = read_csv(args.lstm_last_only_dir / "train_valid_metrics_by_horizon.csv")
        frames.append(
            lstm_last_metrics.loc[
                lstm_last_metrics["method"].isin(["monotonic_lstm_delta_last_retention_only"])
            ].copy()
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[(combined["set_type"] == "valid") & (combined["target"] == "retention")].copy()
    combined["horizon"] = combined["horizon"].astype(str).str.upper()
    return combined


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    """Load and combine the selected valid-set retention predictions."""

    frames: List[pd.DataFrame] = []
    lgbm_pred = read_csv(args.lgbm_dir / "valid_retention_predictions_long.csv")
    frames.append(lgbm_pred.loc[lgbm_pred["method"].isin(["direct_retention", "persistence", "linear_last10"])].copy())

    lgbm_history_pred = read_csv(args.lgbm_history_dir / "valid_retention_predictions_long.csv")
    frames.append(
        lgbm_history_pred.loc[
            lgbm_history_pred["method"].isin(["direct_retention_with_history_summary"])
        ].copy()
    )
    if args.lgbm_last_only_dir is not None:
        lgbm_last_pred = read_csv(args.lgbm_last_only_dir / "valid_retention_predictions_long.csv")
        frames.append(lgbm_last_pred.loc[lgbm_last_pred["method"].isin(["direct_retention_last_only"])].copy())

    lstm_pred = read_csv(args.lstm_dir / "train_valid_predictions_long.csv")
    frames.append(
        lstm_pred.loc[
            (lstm_pred["set_type"] == "valid")
            & lstm_pred["method"].isin(
                ["monotonic_lstm_delta_strict", "monotonic_lstm_delta_with_history_retention"]
            )
        ].copy()
    )
    if args.lstm_last_only_dir is not None:
        lstm_last_pred = read_csv(args.lstm_last_only_dir / "train_valid_predictions_long.csv")
        frames.append(
            lstm_last_pred.loc[
                (lstm_last_pred["set_type"] == "valid")
                & lstm_last_pred["method"].isin(["monotonic_lstm_delta_last_retention_only"])
            ].copy()
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["residual_retention"] = combined["retention_true"] - combined["pred_retention"]
    return combined


def method_label_map() -> Dict[str, str]:
    """Return raw method to report label mapping."""

    return {spec["raw_method"]: spec["label"] for spec in METHOD_SPECS}


def method_plot_label_map() -> Dict[str, str]:
    """Return shorter raw method labels for plot titles."""

    return {spec["raw_method"]: spec["plot_label"] for spec in METHOD_SPECS}


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render selected dataframe columns as a Markdown table."""

    view = df.loc[:, list(columns)].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    lines = ["| " + " | ".join(view.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
    for _idx, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def metric_value(metrics: pd.DataFrame, method: str, horizon: str, metric: str) -> float:
    """Return one metric value for a method and horizon."""

    rows = metrics.loc[(metrics["method"] == method) & (metrics["horizon"] == horizon.upper())]
    if rows.empty:
        return float("nan")
    return float(rows[metric].iloc[0])


def build_summary_table(metrics: pd.DataFrame, meta: Mapping[str, object]) -> pd.DataFrame:
    """Build the high-level conclusion table."""

    rows: List[Dict[str, object]] = []
    selected_horizons = [str(item) for item in meta["selected_horizons"]]
    endpoint_horizon = str(meta["endpoint_horizon"])
    for spec in METHOD_SPECS:
        method = spec["raw_method"]
        row: Dict[str, object] = (
            {
                f"{endpoint_horizon}排名": 0,
                "路线": spec["route"],
                "方法": spec["label"],
                "输入信息": spec["input"],
                "是否使用历史retention": spec["uses_history_retention"],
                "可部署性/口径": spec["claim"],
            }
        )
        for horizon in selected_horizons:
            row[f"{horizon}_RMSE"] = metric_value(metrics, method, horizon, "rmse")
            row[f"{horizon}_R2"] = metric_value(metrics, method, horizon, "r2")
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(f"{endpoint_horizon}_RMSE", ascending=True).reset_index(drop=True)
    table[f"{endpoint_horizon}排名"] = np.arange(1, len(table) + 1)
    return table


def build_detail_table(metrics: pd.DataFrame, meta: Mapping[str, object]) -> pd.DataFrame:
    """Build the H10/H50/ALL metric detail table."""

    labels = method_label_map()
    order = {spec["raw_method"]: idx for idx, spec in enumerate(METHOD_SPECS)}
    selected_horizons = [str(item) for item in meta["selected_horizons"]]
    horizon_order = {horizon: idx for idx, horizon in enumerate(selected_horizons, start=1)}
    view = metrics.loc[metrics["method"].isin(labels.keys()) & metrics["horizon"].isin(selected_horizons)].copy()
    view["路线"] = view["method"].map({spec["raw_method"]: spec["route"] for spec in METHOD_SPECS})
    view["方法"] = view["method"].map(labels)
    view["horizon_sort"] = view["horizon"].map(horizon_order)
    view["method_sort"] = view["method"].map(order)
    view = view.sort_values(["method_sort", "horizon_sort"])
    return view.rename(
        columns={
            "horizon": "horizon",
            "n_rows": "n_rows",
            "mse": "MSE",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "R2",
        }
    )


def sample_for_plot(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a bounded deterministic sample for readable plots."""

    if len(frame) <= int(max_rows):
        return frame
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(frame), size=int(max_rows), replace=False))
    return frame.iloc[keep].copy()


def add_identity_line(ax: object, x_values: pd.Series, y_values: pd.Series) -> None:
    """Add a y=x reference line."""

    values = np.concatenate([x_values.to_numpy(dtype=float), y_values.to_numpy(dtype=float)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = (high - low) * 0.04 if high > low else 0.01
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)


def ensure_matplotlib() -> object:
    """Import matplotlib with the non-interactive backend."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_metric_plot(metrics: pd.DataFrame, out_path: Path, metric: str) -> None:
    """Save a cross-route metric-by-horizon line plot."""

    plt = ensure_matplotlib()
    labels = method_label_map()
    view = metrics.loc[(metrics["method"].isin(labels.keys())) & (metrics["horizon_step"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for spec in METHOD_SPECS:
        part = view.loc[view["method"] == spec["raw_method"]].sort_values("horizon_step")
        if part.empty:
            continue
        ax.plot(part["horizon_step"], part[metric], marker="o", markersize=2.2, linewidth=1.8, label=spec["label"])
    ax.set_xlabel("Future horizon step")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Valid retention {metric.upper()} by horizon")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_endpoint_bar(summary: pd.DataFrame, out_path: Path, endpoint_horizon: str) -> None:
    """Save a compact endpoint RMSE/R2 bar comparison."""

    plt = ensure_matplotlib()
    rmse_col = f"{endpoint_horizon}_RMSE"
    r2_col = f"{endpoint_horizon}_R2"
    view = summary.sort_values(rmse_col, ascending=True).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    axes[0].barh(view["方法"], view[rmse_col], color="#4C78A8")
    axes[0].invert_yaxis()
    axes[0].set_xlabel(f"{endpoint_horizon} RMSE (lower is better)")
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.25)
    axes[1].barh(view["方法"], view[r2_col], color="#59A14F")
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"{endpoint_horizon} R2 (higher is better)")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.suptitle(f"{endpoint_horizon} retention prediction comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_frame_for_horizon(predictions: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Return prediction rows for H10, H50, or ALL."""

    if horizon.upper() == "ALL":
        return predictions.copy()
    step = int(horizon.upper().replace("H", ""))
    return predictions.loc[predictions["horizon_step"] == step].copy()


def save_scatter_grid(predictions: pd.DataFrame, metrics: pd.DataFrame, horizon: str, out_path: Path) -> None:
    """Save a true-vs-predicted scatter grid for one horizon bucket."""

    plt = ensure_matplotlib()
    labels = method_plot_label_map()
    frame = plot_frame_for_horizon(predictions, horizon)
    n_cols = 2
    n_rows = int(math.ceil(len(METHOD_SPECS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.2, 5.0 * n_rows), squeeze=False)
    for idx, spec in enumerate(METHOD_SPECS):
        ax = axes[idx // n_cols][idx % n_cols]
        part = frame.loc[frame["method"] == spec["raw_method"]].copy()
        part = sample_for_plot(part, 4000, 20260512 + idx)
        ax.scatter(part["retention_true"], part["pred_retention"], s=10, alpha=0.32, edgecolors="none")
        add_identity_line(ax, part["retention_true"], part["pred_retention"])
        rmse = metric_value(metrics, spec["raw_method"], horizon, "rmse")
        r2 = metric_value(metrics, spec["raw_method"], horizon, "r2")
        ax.set_title(
            f"{labels[spec['raw_method']]}\n{horizon} | RMSE={rmse:.4f}, R2={r2:.3f}",
            fontsize=10.5,
            pad=9,
        )
        ax.set_xlabel("True retention")
        ax.set_ylabel("Predicted retention")
        ax.grid(True, linestyle="--", alpha=0.24)
    for idx in range(len(METHOD_SPECS), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Prediction scatter comparison ({horizon})", fontsize=13, y=0.995)
    fig.tight_layout(pad=2.0, h_pad=2.6, w_pad=2.2)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_residual_grid(predictions: pd.DataFrame, metrics: pd.DataFrame, horizon: str, out_path: Path) -> None:
    """Save a residual histogram grid for one horizon bucket."""

    plt = ensure_matplotlib()
    labels = method_plot_label_map()
    frame = plot_frame_for_horizon(predictions, horizon)
    n_cols = 2
    n_rows = int(math.ceil(len(METHOD_SPECS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.2, 4.8 * n_rows), squeeze=False)
    for idx, spec in enumerate(METHOD_SPECS):
        ax = axes[idx // n_cols][idx % n_cols]
        part = frame.loc[frame["method"] == spec["raw_method"]].copy()
        part = sample_for_plot(part, 8000, 20260512 + idx)
        ax.hist(part["residual_retention"].dropna(), bins=50, color="#4C78A8", alpha=0.82)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        rmse = metric_value(metrics, spec["raw_method"], horizon, "rmse")
        ax.set_title(
            f"{labels[spec['raw_method']]}\n{horizon} residual | RMSE={rmse:.4f}",
            fontsize=10.5,
            pad=9,
        )
        ax.set_xlabel("Residual = true - predicted")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.24)
    for idx in range(len(METHOD_SPECS), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle(f"Residual distribution comparison ({horizon})", fontsize=13, y=0.995)
    fig.tight_layout(pad=2.0, h_pad=2.6, w_pad=2.2)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def rel(path: Path, base_file: Path) -> str:
    """Return POSIX relative path from a Markdown file to an asset."""

    return path.resolve().relative_to(base_file.resolve().parent).as_posix()


def image(path: Path, alt: str, report_path: Path) -> str:
    """Return a Markdown image link relative to the report file."""

    return f"![{alt}]({rel(path, report_path)})"


def asset_row(path: Path, label: str, report_path: Path) -> Dict[str, object]:
    """Build one artifact index row."""

    return {
        "产物": label,
        "路径": rel(path, report_path),
        "存在": str(path.exists()).lower(),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
    }


def stage_route_images(source_dir: Path, figure_dir: Path) -> Path:
    """Copy route diagrams into the current report figure directory."""

    route_dir = figure_dir / "route_diagrams"
    route_dir.mkdir(parents=True, exist_ok=True)
    for filename in ROUTE_IMAGE_FILES:
        src = source_dir / filename
        dst = route_dir / filename
        if not src.exists():
            continue
        if src.resolve() == dst.resolve():
            continue
        if dst.exists():
            continue
        shutil.copy2(src, dst)
    return route_dir


def save_last_retention_ablation_route(out_path: Path, history_len: int, horizon: int) -> None:
    """Save a reproducible Chinese route diagram for the last-retention-only ablation."""

    plt = ensure_matplotlib()
    from matplotlib import patches

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.35, 5.25, "仅 last retention 消融实验：同一标量输入，不给工况统计和历史序列", fontsize=17, weight="bold")
    ax.text(
        0.35,
        4.88,
        f"固定样本口径：history_len={history_len}, horizon={horizon}, block_stride={history_len + horizon}, long_life_holdout",
        fontsize=11,
        color="#374151",
    )

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.2,
            edgecolor="#1f2937",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.18, y + h - 0.30, title, fontsize=12.2, weight="bold", va="top")
        ax.text(x + 0.18, y + h - 0.82, body, fontsize=9.6, va="top", linespacing=1.18)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#374151"})
        if label:
            ax.text((x0 + x1) / 2, y0 + 0.12, label, ha="center", fontsize=9.5, color="#374151")

    box(0.45, 3.05, 2.28, 1.38, "输入", "历史窗口只取最后一点\nx = retention_N", "#fef3c7")
    box(3.35, 3.05, 2.78, 1.38, "LightGBM last-only", "每个 horizon 单独训练\n输入维度 = 1", "#dbeafe")
    box(6.85, 3.05, 2.78, 1.38, "LSTM last-only", "1x1 标量序列\nmonotonic delta 递推", "#ede9fe")
    box(10.35, 3.05, 2.08, 1.38, "输出", "预测 N+1 到 N+M\nretention 曲线", "#dcfce7")
    arrow(2.75, 3.74, 3.32, 3.74)
    arrow(6.15, 3.74, 6.82, 3.74)
    arrow(9.65, 3.74, 10.32, 3.74)

    box(0.95, 1.05, 3.15, 1.58, "禁用信息", "不使用 55维工况summary\n不使用 100x55工况序列\n不使用历史retention全序列", "#fee2e2")
    box(4.75, 1.05, 3.10, 1.58, "公平比较点", "两条模型路线\n只看到同一个\nlast retention 标量", "#e0f2fe")
    box(8.55, 1.05, 3.25, 1.58, "结论解释", "若 LSTM 胜出\n才可归因于模型结构\n处理 last 标量的方式", "#f0fdf4")
    arrow(4.12, 1.84, 4.72, 1.84)
    arrow(7.88, 1.84, 8.52, 1.84)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def checks_summary(args: argparse.Namespace, meta: Mapping[str, object]) -> pd.DataFrame:
    """Build a concise evidence-chain check table."""

    lgbm_checks = read_csv(args.lgbm_history_dir / "dataset_checks.csv")
    lgbm_config = read_json(args.lgbm_history_dir / "run_config.json")
    lstm_config = read_json(args.lstm_dir / "run_config.json")

    def check_value(checks: pd.DataFrame, item: str) -> str:
        rows = checks.loc[checks["check_item"] == item]
        return "" if rows.empty else str(rows["value"].iloc[0])

    rows = [
        ("数据切分", str(lgbm_config.get("split_name", "")) == "long_life_holdout", "split_name=long_life_holdout"),
        ("样本块", check_value(lgbm_checks, "sample_mode") == "non_overlapping_blocks", "non_overlapping_blocks"),
        (
            "窗口",
            int(lgbm_config.get("history_len", 0)) == int(meta["history_len"])
            and int(lgbm_config.get("horizon", 0)) == int(meta["horizon"]),
            f"history_len={int(meta['history_len'])}, horizon={int(meta['horizon'])}",
        ),
        ("步长", int(lgbm_config.get("block_stride", 0)) == int(meta["block_stride"]), f"block_stride={int(meta['block_stride'])}"),
        ("工况特征", int(lgbm_config.get("feature_count", 0)) == 55, "feature_count=55"),
        (
            "历史retention增强",
            bool(lgbm_config.get("include_history_retention_summary", False))
            and int(lgbm_config.get("history_retention_summary_feature_count", 0)) == 7,
            "LightGBM + 7维历史retention summary",
        ),
        ("split重合", check_value(lgbm_checks, "split_overlap_zero") == "1", "train/valid policy-cell overlap=0"),
        ("LSTM baseline契约", str(lstm_config.get("baseline_source", "")).startswith("loaded:"), "LSTM加载long_life LightGBM baseline"),
    ]
    if args.lgbm_last_only_dir is not None:
        lgbm_last_config = read_json(args.lgbm_last_only_dir / "run_config.json")
        rows.append(
            (
                "LightGBM last-only",
                bool(lgbm_last_config.get("include_last_retention_only", False))
                and int(lgbm_last_config.get("last_retention_only_feature_count", 0)) == 1,
                "输入仅包含 last_retention_only_feature_count=1 的消融路线",
            )
        )
    if args.lstm_last_only_dir is not None:
        lstm_last_config = read_json(args.lstm_last_only_dir / "run_config.json")
        rows.append(
            (
                "LSTM last-only",
                str(lstm_last_config.get("lstm_route_set", "")) == "last_retention_only",
                "输入为 1x1 last retention 标量序列",
            )
        )
    return pd.DataFrame(
        [
            {"证据项": item, "状态": "PASS" if passed else "FAIL", "关键值": detail}
            for item, passed, detail in rows
        ]
    )


def conclusion_text(summary: pd.DataFrame, meta: Mapping[str, object]) -> List[str]:
    """Build direct conclusion lines for the report."""

    by_method = summary.set_index("方法")
    lgbm_hist = by_method.loc["LightGBM + history retention summary"]
    lstm_hist_label = f"LSTM delta {int(meta['history_len'])}x56 history-retention-enhanced"
    lstm_hist = by_method.loc[lstm_hist_label]
    endpoint_horizon = str(meta["endpoint_horizon"])
    delta_endpoint = float(lgbm_hist[f"{endpoint_horizon}_RMSE"]) - float(lstm_hist[f"{endpoint_horizon}_RMSE"])
    delta_all = float(lgbm_hist["ALL_RMSE"]) - float(lstm_hist["ALL_RMSE"])
    winner = "LSTM + 历史 retention" if delta_endpoint > 0 else "LightGBM + 历史 retention summary"
    return [
        f"直接回答：按 {endpoint_horizon} RMSE，`{winner}` 更好。",
        (
            f"{endpoint_horizon} 上 `{lstm_hist_label}` "
            f"RMSE=`{float(lstm_hist[f'{endpoint_horizon}_RMSE']):.6f}`、R2=`{float(lstm_hist[f'{endpoint_horizon}_R2']):.6f}`；"
            "`LightGBM + history retention summary` "
            f"RMSE=`{float(lgbm_hist[f'{endpoint_horizon}_RMSE']):.6f}`、R2=`{float(lgbm_hist[f'{endpoint_horizon}_R2']):.6f}`。"
        ),
        (
            f"{endpoint_horizon} RMSE 差值为 `{delta_endpoint:.6f}`，ALL RMSE 差值为 `{delta_all:.6f}`；"
            "正数表示 LSTM-history 误差更低。"
        ),
        f"但该胜利必须标注为 `{int(meta['history_len'])}x56 history-retention-enhanced`，不能写成“仅工况统计信息”的胜利。",
    ]


def last_retention_only_conclusion_text(summary: pd.DataFrame, meta: Mapping[str, object]) -> List[str]:
    """Build conclusion lines for the last-retention-only ablation."""

    by_method = summary.set_index("方法")
    lgbm_label = "LightGBM last retention only"
    lstm_label = "LSTM delta 1x1 last retention only"
    if lgbm_label not in by_method.index or lstm_label not in by_method.index:
        return ["本节尚未加载 last-retention-only 的 LightGBM 与 LSTM 同口径消融结果。"]
    lgbm = by_method.loc[lgbm_label]
    lstm = by_method.loc[lstm_label]
    endpoint_horizon = str(meta["endpoint_horizon"])
    delta_endpoint = float(lgbm[f"{endpoint_horizon}_RMSE"]) - float(lstm[f"{endpoint_horizon}_RMSE"])
    delta_all = float(lgbm["ALL_RMSE"]) - float(lstm["ALL_RMSE"])
    winner = "LSTM last-retention-only" if delta_endpoint > 0 else "LightGBM last-retention-only"
    return [
        f"直接回答：只给 last retention 标量时，按 {endpoint_horizon} RMSE，`{winner}` 更好。",
        (
            f"任务结论：在短历史 H{int(meta['history_len'])}、预测 M{int(meta['horizon'])} 的任务里，"
            "如果输入严格限制为 last retention，LSTM 的单调 delta 结构比 LightGBM 更会利用这个起点做未来曲线外推。"
        ),
        (
            f"{endpoint_horizon} 上 `{lstm_label}` RMSE=`{float(lstm[f'{endpoint_horizon}_RMSE']):.6f}`、"
            f"R2=`{float(lstm[f'{endpoint_horizon}_R2']):.6f}`；`{lgbm_label}` "
            f"RMSE=`{float(lgbm[f'{endpoint_horizon}_RMSE']):.6f}`、R2=`{float(lgbm[f'{endpoint_horizon}_R2']):.6f}`。"
        ),
        (
            f"{endpoint_horizon} RMSE 差值为 `{delta_endpoint:.6f}`，ALL RMSE 差值为 `{delta_all:.6f}`；"
            "正数表示 LSTM last-only 误差更低。"
        ),
        "该消融不包含 55维工况统计、不包含历史 retention 全序列，也不包含 7维 history summary，因此可用于回答“单纯 last retention”问题。",
    ]


def input_detail_table(meta: Mapping[str, object], has_last_only: bool) -> pd.DataFrame:
    """Build a route-level table that explains input data content and meaning."""

    history_len = int(meta["history_len"])
    horizon = int(meta["horizon"])
    rows: List[Dict[str, object]] = [
        {
            "路线": "trend baseline / linear_last10",
            "输入形态": "10个历史retention点",
            "输入内容": "历史窗口末端最后10个capacity retention观测值",
            "含义": "只利用容量保持率的局部平滑趋势，作为低成本强基线；不使用工况统计。",
        },
        {
            "路线": "trend baseline / persistence",
            "输入形态": "1个last retention标量",
            "输入内容": "历史窗口最后一个capacity retention观测值",
            "含义": "假设未来保持率等于当前状态，衡量模型是否超过最朴素起点基线。",
        },
        {
            "路线": "LightGBM direct",
            "输入形态": "385维tabular summary",
            "输入内容": f"{history_len}个历史cycle内的55个工况基础特征，逐列压缩为last/mean/std/min/max/delta/slope七类统计量。",
            "含义": "把工况时间序列压成表格摘要，不输入历史retention；用于检验工况统计本身的预测力。",
        },
        {
            "路线": "LightGBM + history retention summary",
            "输入形态": "392维tabular summary",
            "输入内容": "385维工况summary + 历史retention的last/mean/std/min/max/delta/slope七类summary。",
            "含义": "把历史retention作为7个统计特征加入LightGBM，但不保留完整retention时间序列。",
        },
        {
            "路线": "LSTM pure operational",
            "输入形态": f"{history_len}x55工况序列 + last retention递推起点",
            "输入内容": f"{history_len}个历史cycle的55个工况通道；last retention只用于monotonic delta递推起点，不作为输入通道。",
            "含义": "保留工况时序结构，检验序列模型是否能从工况变化中获得额外泛化收益。",
        },
        {
            "路线": "LSTM history-retention-enhanced",
            "输入形态": f"{history_len}x56序列",
            "输入内容": f"{history_len}x55工况序列 + 1个历史retention通道。",
            "含义": "显式输入历史retention全序列；若胜出，结论应标注为history-retention-enhanced，不属于纯工况胜利。",
        },
    ]
    if has_last_only:
        rows.extend(
            [
                {
                    "路线": "LightGBM last retention only",
                    "输入形态": "1维tabular",
                    "输入内容": "只输入历史窗口最后一个retention标量；禁用55维工况summary、历史retention全序列和7维history summary。",
                    "含义": f"同口径消融：只看last retention能否预测未来M{horizon}保持率曲线。",
                },
                {
                    "路线": "LSTM last retention only",
                    "输入形态": "1x1标量序列",
                    "输入内容": "只输入历史窗口最后一个retention标量，并通过单调delta结构从该起点向未来递推。",
                    "含义": f"在短历史H{history_len}、预测M{horizon}且输入严格限制为last retention时，检验LSTM结构是否比LightGBM更会利用这个起点做未来曲线外推。",
                },
            ]
        )
    return pd.DataFrame(rows)


def write_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    checks: pd.DataFrame,
    assets: Sequence[Dict[str, object]],
    meta: Mapping[str, object],
) -> None:
    """Write the Markdown comparison report."""

    report_path = args.report_path
    route_images = {
        "trend baseline": args.route_dir / "route_trend_baseline_gpt_image2.png",
        "LightGBM": args.route_dir / "route_lightgbm_gpt_image2.png",
        "纯工况 LSTM": args.route_dir / "route_lstm_pure_operational_gpt_image2.png",
        "历史 retention 增强 LSTM": args.route_dir / "route_lstm_history_retention_gpt_image2.png",
        "last retention only 消融": args.route_dir / "route_last_retention_only_ablation.png",
    }
    selected_horizons = [str(item) for item in meta["selected_horizons"]]
    endpoint_horizon = str(meta["endpoint_horizon"])
    summary_columns = [f"{endpoint_horizon}排名", "路线", "方法", "输入信息", "是否使用历史retention", "可部署性/口径"]
    for horizon in selected_horizons:
        summary_columns.extend([f"{horizon}_RMSE", f"{horizon}_R2"])
    has_last_only = summary["方法"].astype(str).isin(
        ["LightGBM last retention only", "LSTM delta 1x1 last retention only"]
    ).any()
    last_only_section = (
        [
            "### 2.5 last retention only 消融",
            "",
            image(route_images["last retention only 消融"], "last retention only ablation route diagram", report_path),
            "",
            "图 2-5 说明：该图为脚本生成的科研流程图；LightGBM 与 LSTM 都只接收同一个 last retention 标量，不接收工况统计和历史 retention 序列。",
            "",
        ]
        if has_last_only
        else []
    )
    lines = [
        f"# long_life_holdout H{int(meta['history_len'])}/M{int(meta['horizon'])} 工况统计 -> retention LightGBM/LSTM 评估汇总",
        "",
        "## 1. 直接执行",
        "",
        f"- split_name: `{meta['split_name']}`",
        f"- train_split_path: `{meta['train_split_path']}`",
        f"- valid_split_path: `{meta['valid_split_path']}`",
        f"- 样本口径：`history_len={int(meta['history_len'])}`，`horizon={int(meta['horizon'])}`，`block_stride={int(meta['block_stride'])}`，`sample_mode={meta['sample_mode']}`。",
        f"- LightGBM-history 输出目录：`{args.lgbm_history_dir.as_posix()}`",
        f"- LSTM baseline_source: `{meta['baseline_source']}`",
        "- 路线示意图：主路线沿用用户确认的科研论文中文流程图风格；last-only 消融图由脚本生成并写入同一图表目录。",
        "",
        "## 2. 路线总表与示意图",
        "",
        markdown_table(summary, summary_columns),
        "",
        "### 2.0 输入数据内容及含义",
        "",
        "表 2-0 用来区分“基础工况特征”“历史retention增强”和“last-retention-only消融”三种不同输入口径，避免把模型结构收益和输入信息量收益混写。",
        "",
        markdown_table(input_detail_table(meta, has_last_only), ["路线", "输入形态", "输入内容", "含义"]),
        "",
        "### 2.1 trend baseline",
        "",
        image(route_images["trend baseline"], "trend baseline route diagram", report_path),
        "",
        "图 2-1 说明：该路线图已按科研论文中文流程图风格刷新；该路线只使用历史 retention 的平滑趋势，代表最低成本强基线。",
        "",
        "### 2.2 LightGBM",
        "",
        image(route_images["LightGBM"], "LightGBM route diagram", report_path),
        "",
        "图 2-2 说明：该路线图已按科研论文中文流程图风格刷新；LightGBM 路线使用 tabular summary，其中增强版额外加入 7 个历史 retention summary 特征。",
        "",
        "### 2.3 纯工况 LSTM",
        "",
        image(route_images["纯工况 LSTM"], "pure operational LSTM route diagram", report_path),
        "",
        f"图 2-3 说明：该路线图已按科研论文中文流程图风格刷新；纯工况 LSTM 使用 `{int(meta['history_len'])}x55` 工况统计序列，并用 last retention 作为递推起点，不把历史 retention 作为输入通道。",
        "",
        "### 2.4 历史 retention 增强 LSTM",
        "",
        image(route_images["历史 retention 增强 LSTM"], "history retention enhanced LSTM route diagram", report_path),
        "",
        f"图 2-4 说明：该路线图已按科研论文中文流程图风格刷新；增强 LSTM 使用 `{int(meta['history_len'])}x56`，历史 retention 是显式输入通道，结论必须单独标注。",
        "",
        *last_only_section,
        "## 3. 图像证据",
        "",
        image(args.figure_dir / f"comparison_v2_{endpoint_horizon.lower()}_rmse_r2_bar.png", f"{endpoint_horizon} RMSE and R2 bar comparison", report_path),
        "",
        f"图 3-1 说明：左图是 {endpoint_horizon} RMSE，越低越好；右图是 {endpoint_horizon} R2，越高越好。",
        "",
        image(args.figure_dir / "comparison_v2_r2_by_horizon.png", "R2 by horizon comparison", report_path),
        "",
        "图 3-2 说明：X 轴是未来 horizon step，Y 轴是 valid R2，用于观察全预测窗口的泛化趋势。",
        "",
        image(args.figure_dir / "comparison_v2_rmse_by_horizon.png", "RMSE by horizon comparison", report_path),
        "",
        "图 3-3 说明：X 轴是未来 horizon step，Y 轴是 valid RMSE，越低表示误差越小。",
        "",
        f"## 4. {'/'.join(selected_horizons)} 指标与散点残差图",
        "",
        markdown_table(detail, ["路线", "方法", "horizon", "n_rows", "MSE", "RMSE", "MAE", "R2"]),
        "",
        *[
            item
            for horizon in selected_horizons
            for item in (
                image(args.figure_dir / f"comparison_scatter_{horizon.lower()}.png", f"{horizon} scatter comparison", report_path),
                "",
                image(args.figure_dir / f"comparison_residual_{horizon.lower()}.png", f"{horizon} residual comparison", report_path),
                "",
            )
        ],
        "",
        "## 5. 证据链检查",
        "",
        markdown_table(checks, ["证据项", "状态", "关键值"]),
        "",
        "## 6. 直接回答：LSTM + 历史 retention 是否优于 LightGBM + 历史 retention？",
        "",
    ]
    lines.extend([f"- {item}" for item in conclusion_text(summary, meta)])
    if has_last_only:
        lines.extend(
            [
                "",
                "## 7. last retention only 消融结论",
                "",
                *[f"- {item}" for item in last_retention_only_conclusion_text(summary, meta)],
            ]
        )
    lines.extend(
        [
            "",
            "## 8. 图表与产物索引",
            "",
            markdown_table(pd.DataFrame(assets), ["产物", "路径", "存在", "bytes"]),
            "",
            "## 9. 深度交互",
            "",
            "- 这次新增的 LightGBM-history 才是回答“LightGBM + 历史 retention”的同口径证据，不能继续用 `linear_last10` 或 pure LightGBM 代替。",
            f"- 若 LSTM-history 胜出，合理表述是“历史 retention 增强的序列模型胜出”；若要证明纯工况统计序列更强，应继续看 `{int(meta['history_len'])}x55` LSTM 与不含历史 retention 的 LightGBM。",
            "- last retention only 消融是回答“单纯 last retention”问题的同口径证据，不应与 `100x55` 工况序列或 `100x56` 历史序列增强结果混写。",
            "- `linear_last10` 仍需要保留，因为它代表短期 H50 retention 平滑趋势的最低成本解释。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Refresh comparison figures and the Markdown report."""

    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.route_dir = stage_route_images(args.route_dir, args.figure_dir)
    meta = infer_run_meta(args)
    include_last_only = args.lgbm_last_only_dir is not None or args.lstm_last_only_dir is not None
    configure_method_specs(int(meta["history_len"]), include_last_only=include_last_only)
    if include_last_only:
        save_last_retention_ablation_route(
            args.route_dir / "route_last_retention_only_ablation.png",
            int(meta["history_len"]),
            int(meta["horizon"]),
        )
    metrics = load_metrics(args)
    predictions = load_predictions(args)
    summary = build_summary_table(metrics, meta)
    detail = build_detail_table(metrics, meta)
    checks = checks_summary(args, meta)

    save_metric_plot(metrics, args.figure_dir / "comparison_v2_r2_by_horizon.png", "r2")
    save_metric_plot(metrics, args.figure_dir / "comparison_v2_rmse_by_horizon.png", "rmse")
    endpoint_horizon = str(meta["endpoint_horizon"])
    save_endpoint_bar(summary, args.figure_dir / f"comparison_v2_{endpoint_horizon.lower()}_rmse_r2_bar.png", endpoint_horizon)
    for horizon in [str(item) for item in meta["selected_horizons"]]:
        suffix = horizon.lower()
        save_scatter_grid(predictions, metrics, horizon, args.figure_dir / f"comparison_scatter_{suffix}.png")
        save_residual_grid(predictions, metrics, horizon, args.figure_dir / f"comparison_residual_{suffix}.png")

    assets = [
        asset_row(args.figure_dir / f"comparison_v2_{endpoint_horizon.lower()}_rmse_r2_bar.png", f"{endpoint_horizon} RMSE/R2柱状图", args.report_path),
        asset_row(args.figure_dir / "comparison_v2_r2_by_horizon.png", "跨路线R2曲线", args.report_path),
        asset_row(args.figure_dir / "comparison_v2_rmse_by_horizon.png", "跨路线RMSE曲线", args.report_path),
    ]
    for horizon in [str(item) for item in meta["selected_horizons"]]:
        suffix = horizon.lower()
        assets.extend(
            [
                asset_row(args.figure_dir / f"comparison_scatter_{suffix}.png", f"{horizon}散点图", args.report_path),
                asset_row(args.figure_dir / f"comparison_residual_{suffix}.png", f"{horizon}残差图", args.report_path),
            ]
        )
    assets.extend(
        [
        asset_row(args.route_dir / "route_trend_baseline_gpt_image2.png", "trend baseline路线示意图", args.report_path),
        asset_row(args.route_dir / "route_lightgbm_gpt_image2.png", "LightGBM路线示意图", args.report_path),
        asset_row(args.route_dir / "route_lstm_pure_operational_gpt_image2.png", "纯工况LSTM路线示意图", args.report_path),
        asset_row(args.route_dir / "route_lstm_history_retention_gpt_image2.png", "历史retention增强LSTM路线示意图", args.report_path),
        ]
    )
    if include_last_only:
        assets.append(
            asset_row(args.route_dir / "route_last_retention_only_ablation.png", "last retention only消融路线示意图", args.report_path)
        )
    write_report(args, summary, detail, checks, assets, meta)
    print(f"wrote_report {args.report_path}", flush=True)
    print(f"wrote_figures {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
