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
)

METHOD_SPECS: List[Dict[str, str]] = []


def configure_method_specs(history_len: int) -> None:
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

    lstm_metrics = read_csv(args.lstm_dir / "train_valid_metrics_by_horizon.csv")
    frames.append(
        lstm_metrics.loc[
            lstm_metrics["method"].isin(
                ["monotonic_lstm_delta_strict", "monotonic_lstm_delta_with_history_retention"]
            )
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

    lstm_pred = read_csv(args.lstm_dir / "train_valid_predictions_long.csv")
    frames.append(
        lstm_pred.loc[
            (lstm_pred["set_type"] == "valid")
            & lstm_pred["method"].isin(
                ["monotonic_lstm_delta_strict", "monotonic_lstm_delta_with_history_retention"]
            )
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
    }
    selected_horizons = [str(item) for item in meta["selected_horizons"]]
    endpoint_horizon = str(meta["endpoint_horizon"])
    summary_columns = [f"{endpoint_horizon}排名", "路线", "方法", "输入信息", "是否使用历史retention", "可部署性/口径"]
    for horizon in selected_horizons:
        summary_columns.extend([f"{horizon}_RMSE", f"{horizon}_R2"])
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
        "- 路线示意图：按用户确认的科研论文中文流程图风格，使用 Codex 内置图片生成工具生成，并复制到项目图表目录。",
        "",
        "## 2. 路线总表与示意图",
        "",
        markdown_table(summary, summary_columns),
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
    lines.extend(
        [
            "",
            "## 7. 图表与产物索引",
            "",
            markdown_table(pd.DataFrame(assets), ["产物", "路径", "存在", "bytes"]),
            "",
            "## 8. 深度交互",
            "",
            "- 这次新增的 LightGBM-history 才是回答“LightGBM + 历史 retention”的同口径证据，不能继续用 `linear_last10` 或 pure LightGBM 代替。",
            f"- 若 LSTM-history 胜出，合理表述是“历史 retention 增强的序列模型胜出”；若要证明纯工况统计序列更强，应继续看 `{int(meta['history_len'])}x55` LSTM 与不含历史 retention 的 LightGBM。",
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
    configure_method_specs(int(meta["history_len"]))
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
    write_report(args, summary, detail, checks, assets, meta)
    print(f"wrote_report {args.report_path}", flush=True)
    print(f"wrote_figures {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
