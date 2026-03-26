from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_rf_policy_discharge import (  # noqa: E402
    DEFAULT_USE_CYCLES_FEATURE,
    build_cycle_level_dataset,
    build_feature_columns,
    calc_metrics,
    drop_unusable_feature_columns,
    ensure_matplotlib_config,
    load_split_sample_tables,
)


RANDOM_SEED = 20260319

OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "model_benchmark_policy_discharge"
RF_METRICS_PATH = REPO_ROOT / "outputs" / "analysis" / "rf_policy_discharge" / "train_valid_metrics_comparison.csv"
XGB_METRICS_PATH = REPO_ROOT / "outputs" / "analysis" / "xgb_policy_discharge" / "train_valid_metrics_comparison.csv"


def build_fresh_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_alpha_1": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.0005,
                        l1_ratio=0.15,
                        max_iter=20000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=14,
                        min_samples_leaf=8,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=220,
                        max_depth=None,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    return models


def fit_eval_one(
    model_name: str,
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Tuple[dict, np.ndarray]:
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    train_metrics = calc_metrics(y_train, pred_train, "train")
    valid_metrics = calc_metrics(y_valid, pred_valid, "valid")

    row = {
        "model_name": model_name,
        "source_type": "fresh_train",
        "fit_seconds": fit_seconds,
        "train_mae": train_metrics.mae,
        "train_rmse": train_metrics.rmse,
        "train_r2": train_metrics.r2,
        "valid_mae": valid_metrics.mae,
        "valid_rmse": valid_metrics.rmse,
        "valid_r2": valid_metrics.r2,
    }
    return row, pred_valid


def load_existing_metrics(model_name: str, metrics_path: Path) -> dict:
    df = pd.read_csv(metrics_path)
    train = df.loc[df["set_type"] == "train"].iloc[0]
    valid = df.loc[df["set_type"] == "valid"].iloc[0]
    return {
        "model_name": model_name,
        "source_type": "existing_result",
        "fit_seconds": np.nan,
        "train_mae": float(train["mae"]),
        "train_rmse": float(train["rmse"]),
        "train_r2": float(train["r2"]),
        "valid_mae": float(valid["mae"]),
        "valid_rmse": float(valid["rmse"]),
        "valid_r2": float(valid["r2"]),
    }


def save_metric_plot(metrics_df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    rank_df = metrics_df.dropna(subset=["valid_r2"]).sort_values("valid_r2", ascending=False).reset_index(drop=True)
    labels = rank_df["model_name"].tolist()
    xx = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.0))
    axes[0].bar(xx, rank_df["valid_r2"], color="#0ea5e9")
    axes[0].set_title("验证集 R2（越大越好）")
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(xx, rank_df["valid_rmse"], color="#22c55e")
    axes[1].set_title("验证集 RMSE（越小越好）")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("小型模型基准对比（policy + 放电特征）")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def render_report(
    report_path: Path,
    python_exec: str,
    font_list: List[str],
    rows_train: int,
    rows_valid: int,
    n_features_before_drop: int,
    dropped_unusable_features: List[str],
    n_features_used: int,
    metrics_df: pd.DataFrame,
    metric_plot_png: Path,
) -> None:
    rank_df = metrics_df.sort_values("valid_r2", ascending=False).reset_index(drop=True)

    lines: List[str] = []
    lines.append("# 小型模型基准报告（policy + 放电特征）")
    lines.append("")
    lines.append("## 1. 评测设置")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{python_exec}`")
    lines.append(f"- 字体回退：`{', '.join(font_list)}`")
    lines.append(f"- 训练/验证样本：**{rows_train:,} / {rows_valid:,}**")
    lines.append(
        f"- 特征维度（剔除全空前/后）：**{n_features_before_drop} / {n_features_used}**"
    )
    lines.append(f"- 剔除训练集全空特征列：**{len(dropped_unusable_features)}**")
    lines.append("- 数据口径：`q_discharge<=1.5`、`range_count==1`、不使用 `cycles`")
    lines.append("")
    lines.append("## 2. 模型与来源")
    lines.append("- 新训练模型：`linear_regression`、`ridge_alpha_1`、`elastic_net`、`decision_tree`、`extra_trees`")
    lines.append("- 复用已有结果：`random_forest`、`xgboost`")
    lines.append("")
    lines.append("## 3. 指标对比（按验证集 R2 排序）")
    lines.append("| 排名 | 模型 | 来源 | valid_R2 | valid_RMSE | valid_MAE | train_R2 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for idx, row in rank_df.iterrows():
        lines.append(
            f"| {idx + 1} | {row['model_name']} | {row['source_type']} | "
            f"{row['valid_r2']:.6f} | {row['valid_rmse']:.6f} | "
            f"{row['valid_mae']:.6f} | {row['train_r2']:.6f} |"
        )
    lines.append("")
    lines.append("## 4. 可视化")
    lines.append(f"![benchmark_metrics](./{metric_plot_png.name})")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font_list = ensure_matplotlib_config()

    train_split, valid_split = load_split_sample_tables()
    dataset, _, feature_pack = build_cycle_level_dataset(train_split, valid_split)
    train_df = dataset.loc[dataset["set_type"] == "train"].copy()
    valid_df = dataset.loc[dataset["set_type"] == "valid"].copy()

    feature_cols = build_feature_columns(
        feature_pack=feature_pack,
        use_cycles_feature=DEFAULT_USE_CYCLES_FEATURE,
    )
    n_features_before_drop = len(feature_cols)
    feature_cols, dropped_unusable_features = drop_unusable_feature_columns(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
    )

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["q_discharge"].to_numpy(dtype=float)
    X_valid = valid_df[feature_cols].to_numpy(dtype=float)
    y_valid = valid_df["q_discharge"].to_numpy(dtype=float)

    results: List[dict] = []
    valid_pred_df = valid_df[["policy", "cell_code", "cycles", "q_discharge"]].copy()

    fresh_models = build_fresh_models()
    for model_name, model in fresh_models.items():
        try:
            row, pred_valid = fit_eval_one(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            results.append(row)
            valid_pred_df[f"pred_{model_name}"] = pred_valid
            print(
                f"[fresh] {model_name}: valid_r2={row['valid_r2']:.6f}, "
                f"valid_rmse={row['valid_rmse']:.6f}, fit_seconds={row['fit_seconds']:.2f}"
            )
        except Exception as exc:
            results.append(
                {
                    "model_name": model_name,
                    "source_type": "fresh_train_failed",
                    "fit_seconds": np.nan,
                    "train_mae": np.nan,
                    "train_rmse": np.nan,
                    "train_r2": np.nan,
                    "valid_mae": np.nan,
                    "valid_rmse": np.nan,
                    "valid_r2": np.nan,
                    "error": str(exc),
                }
            )
            print(f"[fresh] {model_name} failed: {exc}")

    if RF_METRICS_PATH.exists():
        results.append(load_existing_metrics("random_forest", RF_METRICS_PATH))
        print("[existing] random_forest loaded.")
    if XGB_METRICS_PATH.exists():
        results.append(load_existing_metrics("xgboost", XGB_METRICS_PATH))
        print("[existing] xgboost loaded.")

    metrics_df = pd.DataFrame(results).sort_values("valid_r2", ascending=False).reset_index(drop=True)
    out_metrics_csv = OUTPUT_DIR / "small_model_benchmark_metrics.csv"
    out_pred_csv = OUTPUT_DIR / "valid_predictions_fresh_models.csv"
    out_plot_png = OUTPUT_DIR / "small_model_benchmark_metrics.png"
    out_report_md = OUTPUT_DIR / "small_model_benchmark_report.md"

    metrics_df.to_csv(out_metrics_csv, index=False, encoding="utf-8")
    valid_pred_df.to_csv(out_pred_csv, index=False, encoding="utf-8")
    save_metric_plot(metrics_df, out_plot_png)
    render_report(
        report_path=out_report_md,
        python_exec=os.path.realpath(os.sys.executable),
        font_list=font_list,
        rows_train=len(train_df),
        rows_valid=len(valid_df),
        n_features_before_drop=n_features_before_drop,
        dropped_unusable_features=dropped_unusable_features,
        n_features_used=len(feature_cols),
        metrics_df=metrics_df,
        metric_plot_png=out_plot_png,
    )

    print(f"Saved: {out_metrics_csv}")
    print(f"Saved: {out_pred_csv}")
    print(f"Saved: {out_plot_png}")
    print(f"Saved: {out_report_md}")
    print(
        f"Rows train/valid: {len(train_df)}/{len(valid_df)} | "
        f"features(before/after)={n_features_before_drop}/{len(feature_cols)} | "
        f"dropped_all_nan={len(dropped_unusable_features)}"
    )


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
