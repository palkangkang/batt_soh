from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Non-sequential baseline benchmark on charge delta_ah features."
    )
    parser.add_argument(
        "--charge-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_interval_features.csv",
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "life_performance.csv",
    )
    parser.add_argument(
        "--train-split-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv",
    )
    parser.add_argument(
        "--valid-split-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv",
    )
    parser.add_argument(
        "--lstm-metrics-path",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "analysis"
        / "lstm_charge_delta_ah_q_discharge_cpu"
        / "train_valid_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "delta_ah_nonseq_baseline",
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=20260407)
    return parser.parse_args()


def ensure_matplotlib_backend() -> None:
    """Set Agg backend for headless plots."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")


def load_cycle_level_dataset(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str]]:
    """Load cycle-level table with value+mask features."""

    import sys

    sys.path.append(str(REPO_ROOT))
    import scripts.train_lstm_charge_delta_ah as lstm_mod

    split_df = lstm_mod.load_split_map(args.train_split_path, args.valid_split_path)
    feature_df, _ = lstm_mod.build_cycle_feature_table(args.charge_path)
    value_cols, mask_cols = lstm_mod.get_value_mask_cols(feature_df)
    label_df = lstm_mod.load_life_labels(args.life_path, q_min=args.q_min, q_max=args.q_max)
    merged = lstm_mod.merge_dataset(feature_df=feature_df, label_df=label_df, split_df=split_df)
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    merged["seq_index"] = merged.groupby(["policy", "cell_code"]).cumcount()
    merged = merged.loc[merged["seq_index"] >= args.window_size - 1].copy()
    feat_cols = [*value_cols, *mask_cols]
    return merged, feat_cols


def fit_predict_ridge(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feat_cols: Sequence[str],
) -> np.ndarray:
    """Train ridge baseline and predict on valid."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    x_train = train_df[list(feat_cols)].to_numpy(dtype=float)
    y_train = train_df["q_discharge"].to_numpy(dtype=float)
    x_valid = valid_df[list(feat_cols)].to_numpy(dtype=float)
    model.fit(x_train, y_train)
    return model.predict(x_valid)


def fit_predict_rf(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feat_cols: Sequence[str],
    seed: int,
) -> np.ndarray:
    """Train RF baseline and predict on valid."""

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )
    x_train = train_df[list(feat_cols)].to_numpy(dtype=float)
    y_train = train_df["q_discharge"].to_numpy(dtype=float)
    x_valid = valid_df[list(feat_cols)].to_numpy(dtype=float)
    model.fit(x_train, y_train)
    return model.predict(x_valid)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def save_metric_plot(metric_df: pd.DataFrame, out_png: Path) -> None:
    """Save metric bar plot for valid set."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.6))
    x = np.arange(len(metric_df))
    labels = metric_df["model_name"].tolist()
    for ax, col, title in [
        (axes[0], "r2", "Valid R2"),
        (axes[1], "rmse", "Valid RMSE"),
        (axes[2], "mae", "Valid MAE"),
    ]:
        ax.bar(x, metric_df[col].to_numpy(dtype=float), color=["#0ea5e9", "#22c55e", "#f97316"][: len(x)])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("同口径非时序基线 vs LSTM（验证集）")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def build_report(
    args: argparse.Namespace,
    rows_after_window_filter: int,
    train_rows: int,
    valid_rows: int,
    feature_dim: int,
    result_df: pd.DataFrame,
) -> str:
    """Build Chinese markdown report."""

    lines: List[str] = []
    lines.append("# 非时序基线对照报告（delta_ah 同口径）")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 窗口口径对齐：仅保留 `seq_index >= {args.window_size - 1}` 的样本")
    lines.append(f"- 标签过滤：`{args.q_min} <= q_discharge <= {args.q_max}`")
    lines.append("")
    lines.append("## 2. 数据规模")
    lines.append(f"- 窗口对齐后总样本：**{rows_after_window_filter:,}**")
    lines.append(f"- 训练/验证样本：**{train_rows:,} / {valid_rows:,}**")
    lines.append(f"- 输入维度：**{feature_dim}**（`12 delta_ah + 12 mask`）")
    lines.append("")
    lines.append("## 3. 验证集指标")
    lines.append("| 模型 | MSE | RMSE | MAE | R2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in result_df.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {float(row.mse):.8f} | {float(row.rmse):.6f} | {float(row.mae):.6f} | {float(row.r2):.6f} |"
        )
    lines.append("")
    lines.append("## 4. 图表")
    lines.append("![baseline_vs_lstm](./baseline_vs_lstm_metrics.png)")
    lines.append("")
    lines.append("## 5. 结论")
    best = result_df.sort_values("r2", ascending=False, kind="mergesort").iloc[0]
    lines.append(
        f"- 在当前口径下，验证集 R2 最优模型为 **{best['model_name']}**，R2={float(best['r2']):.6f}。"
    )
    lines.append("- 该对照用于判断时序建模的真实收益，避免仅凭模型复杂度做判断。")
    return "\n".join(lines)


def main() -> None:
    """Run non-sequential baseline benchmark."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_matplotlib_backend()

    data_df, feat_cols = load_cycle_level_dataset(args=args)
    train_df = data_df.loc[data_df["set_type"] == "train"].copy()
    valid_df = data_df.loc[data_df["set_type"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train/valid data is empty after window-aligned filtering.")

    y_valid = valid_df["q_discharge"].to_numpy(dtype=float)
    pred_ridge = fit_predict_ridge(train_df=train_df, valid_df=valid_df, feat_cols=feat_cols)
    pred_rf = fit_predict_rf(train_df=train_df, valid_df=valid_df, feat_cols=feat_cols, seed=int(args.seed))

    rows: List[dict] = []
    for model_name, pred in [("ridge_nonseq", pred_ridge), ("rf_nonseq", pred_rf)]:
        m = calc_metrics(y_true=y_valid, y_pred=pred)
        rows.append({"model_name": model_name, **m})
    result_df = pd.DataFrame(rows)

    pred_df = valid_df[["policy", "cell_code", "cycles", "q_discharge"]].copy()
    pred_df["pred_ridge_nonseq"] = pred_ridge
    pred_df["pred_rf_nonseq"] = pred_rf

    if args.lstm_metrics_path.exists():
        lstm_metric = pd.read_csv(args.lstm_metrics_path)
        lstm_valid = lstm_metric.loc[lstm_metric["set_type"] == "valid"]
        if not lstm_valid.empty:
            row = lstm_valid.iloc[0]
            result_df = pd.concat(
                [
                    result_df,
                    pd.DataFrame(
                        [
                            {
                                "model_name": "lstm_seq",
                                "mse": float(row["mse"]),
                                "rmse": float(row["rmse"]),
                                "mae": float(row["mae"]),
                                "r2": float(row["r2"]),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    out_metrics = args.output_dir / "baseline_valid_metrics.csv"
    out_preds = args.output_dir / "baseline_valid_predictions.csv"
    out_plot = args.output_dir / "baseline_vs_lstm_metrics.png"
    out_report = args.output_dir / "baseline_delta_ah_report.md"

    result_df = result_df.sort_values("r2", ascending=False, kind="mergesort").reset_index(drop=True)
    result_df.to_csv(out_metrics, index=False, encoding="utf-8")
    pred_df.to_csv(out_preds, index=False, encoding="utf-8")
    save_metric_plot(metric_df=result_df, out_png=out_plot)
    out_report.write_text(
        build_report(
            args=args,
            rows_after_window_filter=int(len(data_df)),
            train_rows=int(len(train_df)),
            valid_rows=int(len(valid_df)),
            feature_dim=int(len(feat_cols)),
            result_df=result_df,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_preds}")
    print(f"Saved: {out_plot}")
    print(f"Saved: {out_report}")
    print("Best model on valid:", result_df.iloc[0]["model_name"], "R2=", f"{result_df.iloc[0]['r2']:.6f}")


if __name__ == "__main__":
    main()
