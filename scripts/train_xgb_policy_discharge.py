from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_rf_policy_discharge import (
    DEFAULT_USE_CYCLES_FEATURE,
    FIRST_OCCURRENCE_RANGE_COUNT,
    Q_DISCHARGE_MAX_VALID,
    build_cycle_level_dataset,
    build_feature_columns,
    calc_metrics,
    drop_unusable_feature_columns,
    ensure_matplotlib_config,
    load_split_sample_tables,
)


OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "xgb_policy_discharge"

RANDOM_SEED = 20260319
APPLY_POLICY_WINSORIZATION = False

XGB_PARAMS: Dict[str, object] = {
    "objective": "reg:squarederror",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.2,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "early_stopping_rounds": 50,
}


def build_prediction_table(df: pd.DataFrame, pred: np.ndarray, set_type: str) -> pd.DataFrame:
    out = df[["policy", "cell_code", "cycles", "q_discharge"]].copy()
    out["pred_q_discharge"] = pred
    out["residual"] = out["q_discharge"] - out["pred_q_discharge"]
    out["set_type"] = set_type
    return out


def save_scatter_plot(
    pred_df: pd.DataFrame,
    out_png: Path,
    train_metrics: object,
    valid_metrics: object,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    metrics_map = {"train": train_metrics, "valid": valid_metrics}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for idx, set_type in enumerate(["train", "valid"]):
        ax = axes[idx]
        part = pred_df.loc[pred_df["set_type"] == set_type].copy()
        m = metrics_map[set_type]
        y_true = part["q_discharge"].to_numpy(dtype=float)
        y_pred = part["pred_q_discharge"].to_numpy(dtype=float)
        low = float(min(y_true.min(), y_pred.min()))
        high = float(max(y_true.max(), y_pred.max()))

        ax.scatter(y_true, y_pred, s=10, alpha=0.4, color="#0ea5e9")
        ax.plot([low, high], [low, high], linestyle="--", color="#ef4444", linewidth=1.4)
        ax.set_xlabel("True q_discharge (Ah)")
        ax.set_ylabel("Predicted q_discharge (Ah)")
        ax.set_title(
            f"{set_type.upper()} | R2={m.r2:.4f} | MAE={m.mae:.5f} | RMSE={m.rmse:.5f}"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("XGBoost Fit Scatter: Train vs Valid")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def save_rmse_curve_plot(curve_df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(curve_df["iter"], curve_df["train_rmse"], label="train_rmse", color="#2563eb")
    ax.plot(curve_df["iter"], curve_df["valid_rmse"], label="valid_rmse", color="#ea580c")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("XGBoost Learning Curve (RMSE)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def build_gain_importance_df(model: xgb.XGBRegressor, feature_cols: Sequence[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    weight_map = booster.get_score(importance_type="weight")
    cover_map = booster.get_score(importance_type="cover")

    rows: List[dict] = []
    for idx, col in enumerate(feature_cols):
        key = f"f{idx}"
        rows.append(
            {
                "feature": col,
                "gain": float(gain_map.get(key, 0.0)),
                "weight": float(weight_map.get(key, 0.0)),
                "cover": float(cover_map.get(key, 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("gain", ascending=False).reset_index(drop=True)


def render_report(
    report_path: Path,
    python_exec: str,
    font_list: List[str],
    dataset_stats: Dict[str, int],
    rows_train: int,
    rows_valid: int,
    use_cycles_feature: bool,
    dropped_unusable_features: Sequence[str],
    n_features: int,
    train_metrics: object,
    valid_metrics: object,
    best_iteration: int | None,
    learning_curve_png: Path,
    scatter_png: Path,
) -> None:
    lines: List[str] = []
    lines.append("# XGBoost：policy + 放电特征拟合放电容量")
    lines.append("")
    lines.append("## 1. 数据口径")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{python_exec}`")
    lines.append(f"- 字体回退：`{', '.join(font_list)}`")
    lines.append(
        f"- 异常样本剔除：`q_discharge > {Q_DISCHARGE_MAX_VALID}`（剔除 {dataset_stats['life_outlier_rows_removed_q_gt_1p5']} 行）"
    )
    lines.append(
        f"- 首次出现区间约束：`range_count == {FIRST_OCCURRENCE_RANGE_COUNT}`"
    )
    lines.append(
        f"- 训练/验证行数：**{rows_train:,} / {rows_valid:,}**"
    )
    lines.append(
        f"- 是否使用 `cycles` 特征：**{'是' if use_cycles_feature else '否'}**"
    )
    lines.append(
        f"- 训练集全空特征剔除数：**{len(dropped_unusable_features)}**"
    )
    lines.append(f"- 最终特征维度：**{n_features}**")
    lines.append("")
    lines.append("## 2. 模型参数（XGBoost）")
    for key, value in XGB_PARAMS.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 3. 训练结果")
    lines.append("| set | MAE | RMSE | R2 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| train | {train_metrics.mae:.6f} | {train_metrics.rmse:.6f} | {train_metrics.r2:.6f} |"
    )
    lines.append(
        f"| valid | {valid_metrics.mae:.6f} | {valid_metrics.rmse:.6f} | {valid_metrics.r2:.6f} |"
    )
    if best_iteration is not None:
        lines.append(f"- 最佳迭代轮次（早停）：**{best_iteration}**")
    lines.append("")
    lines.append("## 4. 图表")
    lines.append(f"![learning_curve](./{learning_curve_png.name})")
    lines.append("")
    lines.append(f"![fit_scatter](./{scatter_png.name})")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font_list = ensure_matplotlib_config()

    train_split, valid_split = load_split_sample_tables()
    dataset, dataset_stats, feature_pack = build_cycle_level_dataset(train_split, valid_split)

    train_df = dataset.loc[dataset["set_type"] == "train"].copy()
    valid_df = dataset.loc[dataset["set_type"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train or valid set is empty after split mapping.")

    if APPLY_POLICY_WINSORIZATION:
        # 当前默认关闭，与前一轮随机森林口径一致。
        pass

    feature_cols = build_feature_columns(
        feature_pack=feature_pack,
        use_cycles_feature=DEFAULT_USE_CYCLES_FEATURE,
    )
    feature_cols, dropped_unusable_features = drop_unusable_feature_columns(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
    )

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["q_discharge"].to_numpy(dtype=float)
    X_valid = valid_df[feature_cols].to_numpy(dtype=float)
    y_valid = valid_df["q_discharge"].to_numpy(dtype=float)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_metrics = calc_metrics(y_train, train_pred, "train")
    valid_metrics = calc_metrics(y_valid, valid_pred, "valid")

    pred_train = build_prediction_table(train_df, train_pred, "train")
    pred_valid = build_prediction_table(valid_df, valid_pred, "valid")
    pred_df = pd.concat([pred_train, pred_valid], axis=0, ignore_index=True)
    metrics_df = pd.DataFrame([train_metrics.__dict__, valid_metrics.__dict__])

    evals_result = model.evals_result()
    train_rmse = evals_result["validation_0"]["rmse"]
    valid_rmse = evals_result["validation_1"]["rmse"]
    rmse_curve_df = pd.DataFrame(
        {
            "iter": np.arange(len(train_rmse)),
            "train_rmse": train_rmse,
            "valid_rmse": valid_rmse,
        }
    )

    importance_df = build_gain_importance_df(model, feature_cols)
    best_iteration = getattr(model, "best_iteration", None)

    out_metrics_csv = OUTPUT_DIR / "train_valid_metrics_comparison.csv"
    out_pred_csv = OUTPUT_DIR / "train_valid_predictions.csv"
    out_importance_csv = OUTPUT_DIR / "feature_importance_gain.csv"
    out_curve_csv = OUTPUT_DIR / "learning_curve_rmse.csv"
    out_scatter_png = OUTPUT_DIR / "fit_scatter_train_valid.png"
    out_curve_png = OUTPUT_DIR / "learning_curve_rmse.png"
    out_report_md = OUTPUT_DIR / "xgb_policy_discharge_report.md"

    metrics_df.to_csv(out_metrics_csv, index=False, encoding="utf-8")
    pred_df.to_csv(out_pred_csv, index=False, encoding="utf-8")
    importance_df.to_csv(out_importance_csv, index=False, encoding="utf-8")
    rmse_curve_df.to_csv(out_curve_csv, index=False, encoding="utf-8")
    save_scatter_plot(pred_df, out_scatter_png, train_metrics, valid_metrics)
    save_rmse_curve_plot(rmse_curve_df, out_curve_png)
    render_report(
        report_path=out_report_md,
        python_exec=os.path.realpath(os.sys.executable),
        font_list=font_list,
        dataset_stats=dataset_stats,
        rows_train=len(train_df),
        rows_valid=len(valid_df),
        use_cycles_feature=DEFAULT_USE_CYCLES_FEATURE,
        dropped_unusable_features=dropped_unusable_features,
        n_features=len(feature_cols),
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        best_iteration=best_iteration,
        learning_curve_png=out_curve_png,
        scatter_png=out_scatter_png,
    )

    print(f"Saved: {out_metrics_csv}")
    print(f"Saved: {out_pred_csv}")
    print(f"Saved: {out_importance_csv}")
    print(f"Saved: {out_curve_csv}")
    print(f"Saved: {out_scatter_png}")
    print(f"Saved: {out_curve_png}")
    print(f"Saved: {out_report_md}")
    print(
        f"Rows train/valid: {len(train_df)}/{len(valid_df)} | "
        f"feature_count={len(feature_cols)} | dropped_all_nan={len(dropped_unusable_features)}"
    )
    print(
        f"Train metrics: MAE={train_metrics.mae:.6f}, RMSE={train_metrics.rmse:.6f}, R2={train_metrics.r2:.6f}"
    )
    print(
        f"Valid metrics: MAE={valid_metrics.mae:.6f}, RMSE={valid_metrics.rmse:.6f}, R2={valid_metrics.r2:.6f}"
    )
    print(f"Best iteration: {best_iteration}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
