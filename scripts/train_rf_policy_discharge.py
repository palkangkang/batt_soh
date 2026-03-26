from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DISCHARGE_FEATURE_PATH = REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv"
TRAIN_SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv"
VALID_SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "rf_policy_discharge"
MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"

ENCODING = "utf-8-sig"
FIRST_OCCURRENCE_RANGE_COUNT = 1
Q_DISCHARGE_MAX_VALID = 1.5
RANDOM_SEED = 20260318
DEFAULT_USE_CYCLES_FEATURE = False
DEFAULT_APPLY_POLICY_WINSORIZATION = False
DEFAULT_RF_ENSEMBLE_SEEDS = [
    20260318,
    20260319,
    20260320,
]
POLICY_CLIP_QUANTILE_LOW = 0.01
POLICY_CLIP_QUANTILE_HIGH = 0.99

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 24,
    "min_samples_leaf": 3,
    "min_samples_split": 10,
    "max_features": 0.3,
    "bootstrap": True,
    "max_samples": 0.85,
    "criterion": "squared_error",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}


def parse_range_low(range_label: str) -> float:
    try:
        body = range_label.strip()[1:]
        return float(body.split(",")[0])
    except Exception:
        return float("inf")


def sanitize_range_label(range_label: str) -> str:
    return (
        range_label.replace("[", "")
        .replace("]", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def ensure_matplotlib_config() -> List[str]:
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams  # noqa: WPS433

    candidates = ["fonts-noto-cjk"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected = [f for f in candidates if f in installed]
    if not selected:
        selected = ["DejaVu Sans"]

    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    return selected


def load_split_sample_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_SAMPLE_PATH, encoding=ENCODING)
    valid = pd.read_csv(VALID_SAMPLE_PATH, encoding=ENCODING)

    required = {
        "policy",
        "cell_code",
        "initial_c_rate",
        "switch_soc_percent",
        "post_switch_c_rate",
    }
    for name, df in [("train", train), ("valid", valid)]:
        missing = required.difference(set(df.columns))
        if missing:
            raise KeyError(f"Missing required columns in {name} split: {sorted(missing)}")

    keep_cols = [
        "policy",
        "cell_code",
        "initial_c_rate",
        "switch_soc_percent",
        "post_switch_c_rate",
    ]
    train = train[keep_cols].copy()
    valid = valid[keep_cols].copy()
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    return train, valid


def pivot_interval_feature(
    df: pd.DataFrame,
    value_col: str,
    feature_prefix: str,
    range_order: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    tmp = df[["policy", "cell_code", "cycles", "range", value_col]].copy()
    tmp["feature_name"] = tmp["range"].map(
        lambda x: f"{feature_prefix}_{sanitize_range_label(str(x))}"
    )
    wide = (
        tmp.pivot_table(
            index=["policy", "cell_code", "cycles"],
            columns="feature_name",
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
    )
    ordered_cols = [f"{feature_prefix}_{sanitize_range_label(str(x))}" for x in range_order]
    feature_cols = [x for x in ordered_cols if x in wide.columns]
    wide = wide[["policy", "cell_code", "cycles"] + feature_cols].copy()
    return wide, feature_cols


def add_group_stats(df: pd.DataFrame, cols: Sequence[str], prefix: str) -> List[str]:
    created: List[str] = []
    if not cols:
        return created

    block = df[list(cols)]
    feature_map = {
        f"{prefix}_nonnull_count": block.notna().sum(axis=1),
        f"{prefix}_sum": block.sum(axis=1, skipna=True),
        f"{prefix}_mean": block.mean(axis=1, skipna=True),
        f"{prefix}_std": block.std(axis=1, skipna=True),
    }
    for col, values in feature_map.items():
        df[col] = values
        created.append(col)
    return created


def build_cycle_level_dataset(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, List[str]]]:
    life = pd.read_csv(
        LIFE_PERFORMANCE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    discharge = pd.read_csv(
        DISCHARGE_FEATURE_PATH,
        encoding=ENCODING,
        usecols=[
            "policy",
            "cell_code",
            "cycles",
            "range",
            "delta_ah",
            "charge_duration_s",
            "avg_temper",
            "range_count",
        ],
    )

    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    for col in ["cycles", "delta_ah", "charge_duration_s", "avg_temper", "range_count"]:
        discharge[col] = pd.to_numeric(discharge[col], errors="coerce")

    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life["cycles"] = life["cycles"].astype(int)
    life = (
        life.sort_values(["policy", "cell_code", "cycles"])
        .drop_duplicates(subset=["policy", "cell_code", "cycles"], keep="first")
        .copy()
    )

    stats: Dict[str, int] = {
        "life_rows_before_outlier_filter": int(len(life)),
        "life_outlier_rows_removed_q_gt_1p5": int((life["q_discharge"] > Q_DISCHARGE_MAX_VALID).sum()),
    }
    life = life.loc[life["q_discharge"] <= Q_DISCHARGE_MAX_VALID].copy()
    stats["life_rows_after_outlier_filter"] = int(len(life))

    discharge = discharge.dropna(
        subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]
    ).copy()
    discharge["cycles"] = discharge["cycles"].astype(int)
    discharge["range_count"] = discharge["range_count"].astype(int)
    discharge = discharge.loc[discharge["range_count"] == FIRST_OCCURRENCE_RANGE_COUNT].copy()

    discharge_agg = (
        discharge.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(
            delta_ah=("delta_ah", "sum"),
            charge_duration_s=("charge_duration_s", "sum"),
            avg_temper=("avg_temper", "mean"),
        )
    )
    range_order = sorted(
        discharge_agg["range"].dropna().unique().tolist(),
        key=parse_range_low,
        reverse=True,
    )

    wide_delta, delta_cols = pivot_interval_feature(
        discharge_agg,
        value_col="delta_ah",
        feature_prefix="discharge_delta_ah",
        range_order=range_order,
    )
    wide_dur, dur_cols = pivot_interval_feature(
        discharge_agg,
        value_col="charge_duration_s",
        feature_prefix="discharge_duration_s",
        range_order=range_order,
    )
    wide_temp, temp_cols = pivot_interval_feature(
        discharge_agg,
        value_col="avg_temper",
        feature_prefix="discharge_avg_temper",
        range_order=range_order,
    )

    discharge_wide = (
        wide_delta.merge(wide_dur, on=["policy", "cell_code", "cycles"], how="left")
        .merge(wide_temp, on=["policy", "cell_code", "cycles"], how="left")
        .copy()
    )

    stats_cols_delta = add_group_stats(discharge_wide, delta_cols, "discharge_delta_ah_stats")
    stats_cols_dur = add_group_stats(discharge_wide, dur_cols, "discharge_duration_s_stats")
    stats_cols_temp = add_group_stats(discharge_wide, temp_cols, "discharge_avg_temper_stats")

    split_map = pd.concat([train_split, valid_split], axis=0, ignore_index=True)
    split_map = split_map.drop_duplicates(subset=["policy", "cell_code"], keep="first")
    for col in ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]:
        split_map[col] = pd.to_numeric(split_map[col], errors="coerce")

    dataset = (
        life.merge(discharge_wide, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(
            split_map[
                [
                    "policy",
                    "cell_code",
                    "initial_c_rate",
                    "switch_soc_percent",
                    "post_switch_c_rate",
                    "set_type",
                ]
            ],
            on=["policy", "cell_code"],
            how="inner",
            validate="many_to_one",
        )
        .copy()
    )

    feature_pack = {
        "policy_cols": ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"],
        "delta_cols": delta_cols,
        "dur_cols": dur_cols,
        "temp_cols": temp_cols,
        "stats_cols": stats_cols_delta + stats_cols_dur + stats_cols_temp,
    }
    return dataset, stats, feature_pack


def build_feature_columns(
    feature_pack: Dict[str, List[str]],
    use_cycles_feature: bool = DEFAULT_USE_CYCLES_FEATURE,
) -> List[str]:
    feature_cols = (
        feature_pack["policy_cols"]
        + feature_pack["delta_cols"]
        + feature_pack["dur_cols"]
        + feature_pack["temp_cols"]
        + feature_pack["stats_cols"]
    )
    if use_cycles_feature:
        feature_cols = feature_cols + ["cycles"]
    return feature_cols


def drop_unusable_feature_columns(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for col in feature_cols:
        if train_df[col].notna().sum() == 0:
            dropped.append(col)
        else:
            kept.append(col)
    _ = valid_df  # reserved for future rules
    return kept, dropped


def apply_policy_winsorization(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_col: str = "policy",
    low_q: float = POLICY_CLIP_QUANTILE_LOW,
    high_q: float = POLICY_CLIP_QUANTILE_HIGH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_cols:
        return train_df.copy(), valid_df.copy()

    train_out = train_df.copy()
    valid_out = valid_df.copy()
    features = list(feature_cols)
    train_out[features] = train_out[features].astype(float)
    valid_out[features] = valid_out[features].astype(float)

    global_low = train_out[features].quantile(low_q, interpolation="linear")
    global_high = train_out[features].quantile(high_q, interpolation="linear")
    policy_bounds: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    for policy, part in train_out.groupby(group_col, sort=False):
        low = part[features].quantile(low_q, interpolation="linear")
        high = part[features].quantile(high_q, interpolation="linear")
        policy_bounds[str(policy)] = (low, high)

    for policy, idx in train_out.groupby(group_col, sort=False).groups.items():
        low, high = policy_bounds[str(policy)]
        train_out.loc[idx, features] = train_out.loc[idx, features].clip(
            lower=low, upper=high, axis=1
        )

    for policy, idx in valid_out.groupby(group_col, sort=False).groups.items():
        low, high = policy_bounds.get(str(policy), (global_low, global_high))
        valid_out.loc[idx, features] = valid_out.loc[idx, features].clip(
            lower=low, upper=high, axis=1
        )

    return train_out, valid_out


@dataclass
class Metrics:
    set_type: str
    n_rows: int
    mae: float
    rmse: float
    r2: float


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, set_type: str) -> Metrics:
    return Metrics(
        set_type=set_type,
        n_rows=int(len(y_true)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def train_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: Dict[str, object] | None = None,
) -> Pipeline:
    params = MODEL_PARAMS if model_params is None else model_params
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(**params)),
        ]
    )
    X_train = train_df[list(feature_cols)].to_numpy(dtype=float)
    y_train = train_df["q_discharge"].to_numpy(dtype=float)
    model.fit(X_train, y_train)
    return model


def build_prediction_table(df: pd.DataFrame, pred: np.ndarray, set_type: str) -> pd.DataFrame:
    out = df[["policy", "cell_code", "cycles", "q_discharge"]].copy()
    out["pred_q_discharge"] = pred
    out["residual"] = out["q_discharge"] - out["pred_q_discharge"]
    out["set_type"] = set_type
    return out


def save_scatter_plot(pred_df: pd.DataFrame, out_png: Path, metrics_map: Dict[str, Metrics]) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for idx, set_type in enumerate(["train", "valid"]):
        ax = axes[idx]
        part = pred_df.loc[pred_df["set_type"] == set_type].copy()
        m = metrics_map[set_type]

        y_true = part["q_discharge"].to_numpy(dtype=float)
        y_pred = part["pred_q_discharge"].to_numpy(dtype=float)
        low = float(min(y_true.min(), y_pred.min()))
        high = float(max(y_true.max(), y_pred.max()))

        ax.scatter(y_true, y_pred, s=12, alpha=0.42, color="#0ea5e9")
        ax.plot([low, high], [low, high], linestyle="--", color="#ef4444", linewidth=1.4)
        ax.set_xlabel("True q_discharge (Ah)")
        ax.set_ylabel("Predicted q_discharge (Ah)")
        ax.set_title(
            f"{set_type.upper()} | R2={m.r2:.4f} | MAE={m.mae:.5f} | RMSE={m.rmse:.5f}"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("RandomForest Fit Scatter: Train vs Valid")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def render_report(
    report_path: Path,
    font_list: List[str],
    filter_stats: Dict[str, int],
    n_rows_all: int,
    feature_pack: Dict[str, List[str]],
    use_cycles_feature: bool,
    apply_policy_winsorization_flag: bool,
    ensemble_seeds: Sequence[int],
    dropped_unusable_features: Sequence[str],
    metrics_df: pd.DataFrame,
    out_scatter_png: Path,
) -> None:
    n_policy = len(feature_pack["policy_cols"])
    n_delta = len(feature_pack["delta_cols"])
    n_dur = len(feature_pack["dur_cols"])
    n_temp = len(feature_pack["temp_cols"])
    n_stats = len(feature_pack["stats_cols"])
    n_total = n_policy + n_delta + n_dur + n_temp + n_stats + (1 if use_cycles_feature else 0)

    lines: List[str] = []
    lines.append("# RandomForest: policy + first-occurrence discharge features")
    lines.append("")
    lines.append("## 1. Data and constraints")
    lines.append(f"- Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python executable: `{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- Font fallback: `{', '.join(font_list)}`")
    lines.append(
        f"- Outlier filter: remove samples with `q_discharge > {Q_DISCHARGE_MAX_VALID}`"
    )
    lines.append(
        f"- Life rows before/after outlier filter: "
        f"**{filter_stats['life_rows_before_outlier_filter']:,} / "
        f"{filter_stats['life_rows_after_outlier_filter']:,}**"
    )
    lines.append(
        f"- Outlier rows removed: **{filter_stats['life_outlier_rows_removed_q_gt_1p5']:,}**"
    )
    lines.append(f"- Total cycle-level rows used: **{n_rows_all:,}**")
    lines.append(
        f"- First-occurrence rule: only use rows with "
        f"`range_count == {FIRST_OCCURRENCE_RANGE_COUNT}`"
    )
    if dropped_unusable_features:
        lines.append(
            f"- Dropped unusable feature columns (all-NaN in train): "
            f"**{len(dropped_unusable_features)}**"
        )
    if ensemble_seeds:
        lines.append(
            f"- Model strategy: RandomForest ensemble average "
            f"({len(ensemble_seeds)} seeds)"
        )
        lines.append(f"- Ensemble seeds: `{list(ensemble_seeds)}`")
    else:
        lines.append("- Model strategy: single RandomForest")
    if apply_policy_winsorization_flag:
        lines.append(
            f"- Policy-wise winsorization on discharge-derived features: "
            f"`Q{int(POLICY_CLIP_QUANTILE_LOW * 100)}-Q{int(POLICY_CLIP_QUANTILE_HIGH * 100)}`"
        )
    else:
        lines.append("- Policy-wise winsorization on discharge-derived features: **Disabled**")
    lines.append("")
    lines.append("## 2. Feature design")
    lines.append(f"- Policy triad features: **{n_policy}**")
    lines.append(f"- Discharge interval `delta_ah` features: **{n_delta}**")
    lines.append(f"- Discharge interval duration features: **{n_dur}**")
    lines.append(f"- Discharge interval temperature features: **{n_temp}**")
    lines.append(f"- Discharge group-stat features: **{n_stats}**")
    lines.append(
        f"- Include `cycles` feature: **{'Yes' if use_cycles_feature else 'No'}**"
    )
    lines.append(f"- Total feature count: **{n_total}**")
    lines.append("")
    lines.append("## 3. Train vs valid metrics")
    lines.append("| set | n_rows | MAE | RMSE | R2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['set_type']} | {int(row['n_rows'])} | "
            f"{row['mae']:.6f} | {row['rmse']:.6f} | {row['r2']:.6f} |"
        )
    lines.append("")
    lines.append("## 4. Scatter plot")
    lines.append(f"![train_valid_scatter](./{out_scatter_png.name})")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font_list = ensure_matplotlib_config()

    train_split, valid_split = load_split_sample_tables()
    dataset, filter_stats, feature_pack = build_cycle_level_dataset(train_split, valid_split)
    if dataset.empty:
        raise RuntimeError("Dataset is empty after merge.")

    train_df = dataset.loc[dataset["set_type"] == "train"].copy()
    valid_df = dataset.loc[dataset["set_type"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train or valid set is empty after split mapping.")

    use_cycles_feature = DEFAULT_USE_CYCLES_FEATURE
    feature_cols = build_feature_columns(
        feature_pack=feature_pack,
        use_cycles_feature=use_cycles_feature,
    )
    feature_cols, dropped_unusable_features = drop_unusable_feature_columns(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
    )
    apply_policy_winsorization_flag = DEFAULT_APPLY_POLICY_WINSORIZATION
    if apply_policy_winsorization_flag:
        clip_feature_cols = [c for c in feature_cols if c not in feature_pack["policy_cols"]]
        train_df, valid_df = apply_policy_winsorization(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=clip_feature_cols,
        )

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_valid = valid_df[feature_cols].to_numpy(dtype=float)

    ensemble_seeds = list(DEFAULT_RF_ENSEMBLE_SEEDS)
    if not ensemble_seeds:
        ensemble_seeds = [RANDOM_SEED]

    train_pred_list: List[np.ndarray] = []
    valid_pred_list: List[np.ndarray] = []
    importance_list: List[np.ndarray] = []
    for seed in ensemble_seeds:
        model = train_model(
            train_df,
            feature_cols,
            model_params={**MODEL_PARAMS, "random_state": seed},
        )
        train_pred_list.append(model.predict(X_train))
        valid_pred_list.append(model.predict(X_valid))
        importance_list.append(model.named_steps["rf"].feature_importances_)

    train_pred = np.mean(np.vstack(train_pred_list), axis=0)
    valid_pred = np.mean(np.vstack(valid_pred_list), axis=0)
    mean_importance = np.mean(np.vstack(importance_list), axis=0)

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

    metrics_df = pd.DataFrame([train_metrics.__dict__, valid_metrics.__dict__])

    pred_train = build_prediction_table(train_df, train_pred, set_type="train")
    pred_valid = build_prediction_table(valid_df, valid_pred, set_type="valid")
    pred_df = pd.concat([pred_train, pred_valid], axis=0, ignore_index=True)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": mean_importance,
        }
    ).sort_values("importance", ascending=False)

    out_metrics_csv = OUTPUT_DIR / "train_valid_metrics_comparison.csv"
    out_pred_csv = OUTPUT_DIR / "train_valid_predictions.csv"
    out_importance_csv = OUTPUT_DIR / "feature_importance.csv"
    out_scatter_png = OUTPUT_DIR / "fit_scatter_train_valid.png"
    out_report_md = OUTPUT_DIR / "rf_policy_discharge_report.md"

    metrics_df.to_csv(out_metrics_csv, index=False, encoding="utf-8")
    pred_df.to_csv(out_pred_csv, index=False, encoding="utf-8")
    importance_df.to_csv(out_importance_csv, index=False, encoding="utf-8")
    save_scatter_plot(
        pred_df=pred_df,
        out_png=out_scatter_png,
        metrics_map={"train": train_metrics, "valid": valid_metrics},
    )
    render_report(
        report_path=out_report_md,
        font_list=font_list,
        filter_stats=filter_stats,
        n_rows_all=len(dataset),
        feature_pack=feature_pack,
        use_cycles_feature=use_cycles_feature,
        apply_policy_winsorization_flag=apply_policy_winsorization_flag,
        ensemble_seeds=ensemble_seeds,
        dropped_unusable_features=dropped_unusable_features,
        metrics_df=metrics_df,
        out_scatter_png=out_scatter_png,
    )

    print(f"Saved: {out_metrics_csv}")
    print(f"Saved: {out_pred_csv}")
    print(f"Saved: {out_importance_csv}")
    print(f"Saved: {out_scatter_png}")
    print(f"Saved: {out_report_md}")
    print(
        f"Rows train/valid: {len(train_df)}/{len(valid_df)} | "
        f"outlier removed rows: {filter_stats['life_outlier_rows_removed_q_gt_1p5']}"
    )
    print(
        f"Feature count policy/delta/dur/temp/stats/cycles/total: "
        f"{len(feature_pack['policy_cols'])}/"
        f"{len(feature_pack['delta_cols'])}/"
        f"{len(feature_pack['dur_cols'])}/"
        f"{len(feature_pack['temp_cols'])}/"
        f"{len(feature_pack['stats_cols'])}/"
        f"{1 if use_cycles_feature else 0}/{len(feature_cols)}"
    )
    print(
        f"Dropped all-NaN feature columns in train: {len(dropped_unusable_features)}"
    )
    print(f"Ensemble seeds used: {ensemble_seeds}")
    print(
        f"Train metrics: MAE={train_metrics.mae:.6f}, RMSE={train_metrics.rmse:.6f}, R2={train_metrics.r2:.6f}"
    )
    print(
        f"Valid metrics: MAE={valid_metrics.mae:.6f}, RMSE={valid_metrics.rmse:.6f}, R2={valid_metrics.r2:.6f}"
    )


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
