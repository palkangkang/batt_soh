from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_rf_policy_discharge import (
    DEFAULT_APPLY_POLICY_WINSORIZATION,
    DEFAULT_USE_CYCLES_FEATURE,
    OUTPUT_DIR,
    RANDOM_SEED,
    apply_policy_winsorization,
    build_cycle_level_dataset,
    build_feature_columns,
    calc_metrics,
    drop_unusable_feature_columns,
    load_split_sample_tables,
    train_model,
)


N_TRIALS = 8
APPLY_POLICY_WINSORIZATION_IN_TUNING = DEFAULT_APPLY_POLICY_WINSORIZATION

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 150, 200, 250],
    "max_depth": [12, 14, 16, 18, 20, 24],
    "min_samples_leaf": [1, 2, 3, 4],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, "sqrt"],
    "bootstrap": [True],
    "max_samples": [0.7, 0.85, 1.0],
    "criterion": ["squared_error"],
}


def normalize_model_params(raw: Dict[str, object]) -> Dict[str, object]:
    params = dict(raw)
    if params.get("max_samples", None) == 1.0:
        params["max_samples"] = None
    if not params.get("bootstrap", True):
        params["max_samples"] = None
    params["random_state"] = RANDOM_SEED
    params["n_jobs"] = 1
    return params


def evaluate_one(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    model_params: Dict[str, object],
) -> Dict[str, object]:
    model = train_model(train_df, feature_cols, model_params=model_params)

    train_pred = model.predict(train_df[feature_cols].to_numpy(dtype=float))
    valid_pred = model.predict(valid_df[feature_cols].to_numpy(dtype=float))

    train_metrics = calc_metrics(
        train_df["q_discharge"].to_numpy(dtype=float),
        train_pred,
        set_type="train",
    )
    valid_metrics = calc_metrics(
        valid_df["q_discharge"].to_numpy(dtype=float),
        valid_pred,
        set_type="valid",
    )
    gap = train_metrics.r2 - valid_metrics.r2
    score = valid_metrics.r2

    row: Dict[str, object] = {
        "score": score,
        "train_r2": train_metrics.r2,
        "valid_r2": valid_metrics.r2,
        "train_mae": train_metrics.mae,
        "valid_mae": valid_metrics.mae,
        "train_rmse": train_metrics.rmse,
        "valid_rmse": valid_metrics.rmse,
        "r2_gap_train_minus_valid": gap,
    }
    row.update(model_params)
    return row


def render_report(
    report_path: Path,
    trial_count: int,
    rows_train: int,
    rows_valid: int,
    n_features: int,
    results_df: pd.DataFrame,
    best_row: pd.Series,
) -> None:
    lines: List[str] = []
    lines.append("# Quick tuning report (RandomForest, no cycles)")
    lines.append("")
    lines.append("## 1. Search setup")
    lines.append(f"- Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Trials: **{trial_count}**")
    lines.append(f"- Use cycles feature: **{DEFAULT_USE_CYCLES_FEATURE}**")
    lines.append(
        f"- Policy-wise winsorization in tuning: **{APPLY_POLICY_WINSORIZATION_IN_TUNING}**"
    )
    lines.append(f"- Rows train/valid: **{rows_train} / {rows_valid}**")
    lines.append(f"- Feature count: **{n_features}**")
    lines.append("- Objective: maximize `valid_r2`")
    lines.append("")
    lines.append("## 2. Best trial")
    lines.append(
        f"- Best valid R2: **{best_row['valid_r2']:.6f}**, "
        f"valid RMSE: **{best_row['valid_rmse']:.6f}**, "
        f"valid MAE: **{best_row['valid_mae']:.6f}**"
    )
    lines.append(
        f"- Train R2: **{best_row['train_r2']:.6f}** "
        f"(gap={best_row['r2_gap_train_minus_valid']:.6f})"
    )
    lines.append("- Best params:")
    lines.append(f"  - `n_estimators={best_row['n_estimators']}`")
    lines.append(f"  - `max_depth={best_row['max_depth']}`")
    lines.append(f"  - `min_samples_leaf={best_row['min_samples_leaf']}`")
    lines.append(f"  - `min_samples_split={best_row['min_samples_split']}`")
    lines.append(f"  - `max_features={best_row['max_features']}`")
    lines.append(f"  - `max_samples={best_row['max_samples']}`")
    lines.append(f"  - `criterion={best_row['criterion']}`")
    lines.append("")
    lines.append("## 3. Top 10 trials")
    lines.append("| rank | valid_r2 | valid_rmse | valid_mae | train_r2 | r2_gap |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    top10 = results_df.head(10).reset_index(drop=True)
    for idx, row in top10.iterrows():
        lines.append(
            f"| {idx + 1} | {row['valid_r2']:.6f} | {row['valid_rmse']:.6f} | "
            f"{row['valid_mae']:.6f} | {row['train_r2']:.6f} | {row['r2_gap_train_minus_valid']:.6f} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_split, valid_split = load_split_sample_tables()
    dataset, filter_stats, feature_pack = build_cycle_level_dataset(train_split, valid_split)
    if dataset.empty:
        raise RuntimeError("Dataset is empty after merge.")

    train_df = dataset.loc[dataset["set_type"] == "train"].copy()
    valid_df = dataset.loc[dataset["set_type"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train or valid set is empty after split mapping.")

    feature_cols = build_feature_columns(
        feature_pack=feature_pack,
        use_cycles_feature=DEFAULT_USE_CYCLES_FEATURE,
    )
    feature_cols, dropped_unusable_features = drop_unusable_feature_columns(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
    )
    clip_cols = [c for c in feature_cols if c not in feature_pack["policy_cols"]]
    if APPLY_POLICY_WINSORIZATION_IN_TUNING:
        train_df, valid_df = apply_policy_winsorization(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=clip_cols,
        )

    sampler = ParameterSampler(
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_TRIALS,
        random_state=RANDOM_SEED,
    )
    trial_rows: List[Dict[str, object]] = []

    for trial_idx, raw_params in enumerate(sampler, start=1):
        params = normalize_model_params(raw_params)
        row = evaluate_one(train_df, valid_df, feature_cols, params)
        row["trial"] = trial_idx
        trial_rows.append(row)
        print(
            f"[trial {trial_idx:02d}/{N_TRIALS}] valid_r2={row['valid_r2']:.6f} "
            f"valid_rmse={row['valid_rmse']:.6f} params="
            f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
            f"min_samples_leaf={params['min_samples_leaf']}, min_samples_split={params['min_samples_split']}, "
            f"max_features={params['max_features']}, max_samples={params['max_samples']}, "
            f"criterion={params['criterion']}"
        )

    results_df = pd.DataFrame(trial_rows)
    results_df = results_df.sort_values(["valid_r2", "valid_rmse"], ascending=[False, True]).reset_index(drop=True)
    best_row = results_df.iloc[0]

    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "max_features": best_row["max_features"],
        "bootstrap": bool(best_row["bootstrap"]),
        "max_samples": None if pd.isna(best_row["max_samples"]) else float(best_row["max_samples"]),
        "criterion": str(best_row["criterion"]),
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }

    out_results_csv = OUTPUT_DIR / "quick_tuning_results.csv"
    out_best_json = OUTPUT_DIR / "quick_tuning_best_params.json"
    out_report_md = OUTPUT_DIR / "quick_tuning_report.md"

    results_df.to_csv(out_results_csv, index=False, encoding="utf-8")
    out_best_json.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    render_report(
        report_path=out_report_md,
        trial_count=N_TRIALS,
        rows_train=len(train_df),
        rows_valid=len(valid_df),
        n_features=len(feature_cols),
        results_df=results_df,
        best_row=best_row,
    )

    print(f"Saved: {out_results_csv}")
    print(f"Saved: {out_best_json}")
    print(f"Saved: {out_report_md}")
    print(
        f"Rows train/valid: {len(train_df)}/{len(valid_df)} | "
        f"outlier_removed={filter_stats['life_outlier_rows_removed_q_gt_1p5']} | "
        f"features={len(feature_cols)} | dropped_all_nan={len(dropped_unusable_features)}"
    )
    print(f"Best valid R2={best_row['valid_r2']:.6f} | Best params={best_params}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
