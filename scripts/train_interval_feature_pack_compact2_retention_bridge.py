"""Train a short recommended-feature -> compact2 dQdV -> retention bridge."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_interval_to_dqdv_retention_pipeline import (
    ENCODING,
    REPO_ROOT,
    build_allowed_key_tokens,
    calc_metrics,
    dedupe_keep_order,
    filter_allowed_keys,
    load_charge_feature_table,
    load_discharge_feature_table,
    load_dqdv_table,
    load_retention_labels,
    load_split,
    select_input_feature_columns,
    set_seed,
)

TARGET_COLS = ["main_peak_area", "main_peak_height_dqdv"]
FORBIDDEN_INPUT_COLS = {
    "cycles",
    "cycle_index_norm",
    "policy",
    "cell_code",
    "initial_c_rate",
    "switch_soc_percent",
    "post_switch_c_rate",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate recommended interval features through compact2 dQdV and retention bridge."
    )
    parser.add_argument(
        "--recommended-feature-path",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "analysis"
        / "interval_features_to_dqdv_correlation"
        / "recommended_feature_pack_union.csv",
    )
    parser.add_argument(
        "--charge-timeseries-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv",
    )
    parser.add_argument(
        "--discharge-interval-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv",
    )
    parser.add_argument(
        "--dqdv-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv",
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "interval_feature_pack_compact2_retention_bridge",
    )
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-train-cells", type=int, default=12)
    parser.add_argument("--smoke-valid-cells", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    return parser.parse_args()


def load_recommended_features(path: Path) -> List[str]:
    """Load and validate the recommended feature list."""

    df = pd.read_csv(path, encoding=ENCODING)
    if "feature" not in df.columns:
        raise ValueError(f"Missing feature column in {path}")
    features = dedupe_keep_order([str(item).strip() for item in df["feature"].dropna().tolist() if str(item).strip()])
    if not features:
        raise RuntimeError(f"No recommended features found in {path}")
    forbidden = sorted(set(features).intersection(FORBIDDEN_INPUT_COLS))
    if forbidden:
        raise RuntimeError(f"Forbidden input columns found in recommended feature pack: {forbidden}")
    return features


def build_cycle_table(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, int]]:
    """Build the cycle-level modeling table and return feature lists."""

    recommended_features = load_recommended_features(args.recommended_feature_path)
    _, _, split_map = load_split(args.train_split_path, args.valid_split_path)
    allowed_keys = (
        build_allowed_key_tokens(split_map, args.smoke_train_cells, args.smoke_valid_cells)
        if bool(args.smoke_test)
        else None
    )

    print("Loading charge features...", flush=True)
    charge_df, charge_stats, charge_cols = load_charge_feature_table(args.charge_timeseries_path, allowed_keys)
    print("Loading discharge features...", flush=True)
    discharge_df, discharge_stats, discharge_cols = load_discharge_feature_table(
        args.discharge_interval_path,
        allowed_keys,
    )
    interval_df = charge_df.merge(discharge_df, on=["policy", "cell_code", "cycles"], how="outer")
    interval_cols = dedupe_keep_order([*charge_cols, *discharge_cols])
    for col in interval_cols:
        interval_df[col] = pd.to_numeric(interval_df[col], errors="coerce").fillna(0.0)

    missing_recommended = sorted(set(recommended_features).difference(interval_cols))
    if missing_recommended:
        raise RuntimeError(f"Recommended features missing from constructed interval features: {missing_recommended}")
    full159_features = select_input_feature_columns(interval_cols, "charge_crossbin_discharge_capacity_stats")
    forbidden_full159 = sorted(set(full159_features).intersection(FORBIDDEN_INPUT_COLS))
    if forbidden_full159:
        raise RuntimeError(f"Forbidden input columns found in full159 feature pack: {forbidden_full159}")

    print("Loading compact2 dQdV targets and retention labels...", flush=True)
    dqdv_df = filter_allowed_keys(load_dqdv_table(args.dqdv_path, TARGET_COLS), allowed_keys)
    labels_df = filter_allowed_keys(
        load_retention_labels(
            args.life_path,
            q_min=float(args.q_min),
            q_max=float(args.q_max),
            q_ref_cycles=int(args.q_ref_cycles),
            retention_min=float(args.retention_min),
            retention_max=float(args.retention_max),
        ),
        allowed_keys,
    )
    split_use = filter_allowed_keys(split_map, allowed_keys)
    split_cols = ["policy", "cell_code", "set_type"]
    merged = (
        labels_df.merge(dqdv_df, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(split_use[split_cols], on=["policy", "cell_code"], how="inner")
        .merge(interval_df[["policy", "cell_code", "cycles", *interval_cols]], on=["policy", "cell_code", "cycles"], how="left")
    )
    for col in [*recommended_features, *full159_features]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(np.float32)
    for col in [*TARGET_COLS, "retention", "q_ref", "q_discharge"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=[*TARGET_COLS, "retention", "q_ref", "q_discharge"]).copy()
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    merged = apply_row_limits(merged, args.max_train_rows, args.max_valid_rows, int(args.random_seed))
    stats = {
        "charge_cross_bin_feature_dim": int(charge_stats.get("charge_cross_bin_feature_dim", 0)),
        "discharge_range_count": int(discharge_stats.get("discharge_range_count", 0)),
    }
    return merged, recommended_features, full159_features, stats


def apply_row_limits(df: pd.DataFrame, max_train_rows: int, max_valid_rows: int, seed: int) -> pd.DataFrame:
    """Apply optional per-split row limits without changing split membership."""

    parts: List[pd.DataFrame] = []
    for set_type, limit, offset in [("train", int(max_train_rows), 0), ("valid", int(max_valid_rows), 1)]:
        part = df.loc[df["set_type"] == set_type].copy()
        if limit > 0 and len(part) > limit:
            part = part.sample(n=limit, random_state=int(seed) + offset).sort_values(
                ["policy", "cell_code", "cycles"],
                kind="mergesort",
            )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["set_type", "policy", "cell_code", "cycles"], kind="mergesort")


def make_linear_multi_model(model_name: str, random_seed: int) -> Pipeline:
    """Create a linear multi-output model pipeline."""

    if model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=int(random_seed))
    elif model_name == "elasticnet":
        estimator = ElasticNet(alpha=1e-3, l1_ratio=0.15, max_iter=5000, random_state=int(random_seed))
    else:
        raise ValueError(f"Unsupported linear model: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(estimator)),
        ]
    )


def make_tree_multi_model(model_name: str, random_seed: int) -> Optional[Pipeline]:
    """Create a tree multi-output model pipeline."""

    if model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=int(random_seed),
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            return None
        estimator = LGBMRegressor(
            n_estimators=320,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=int(random_seed),
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unsupported tree model: {model_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(estimator)),
        ]
    )


def make_multioutput_models(random_seed: int) -> Dict[str, Pipeline]:
    """Create all compact2 prediction models available in the environment."""

    models: Dict[str, Pipeline] = {
        "ridge": make_linear_multi_model("ridge", random_seed),
        "elasticnet": make_linear_multi_model("elasticnet", random_seed),
        "hist_gradient_boosting": make_tree_multi_model("hist_gradient_boosting", random_seed),
    }
    lightgbm = make_tree_multi_model("lightgbm", random_seed)
    if lightgbm is not None:
        models["lightgbm"] = lightgbm
    return {key: value for key, value in models.items() if value is not None}


def make_single_model(model_name: str, random_seed: int) -> Pipeline:
    """Create a single-output model pipeline."""

    if model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=int(random_seed))
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    if model_name == "elasticnet":
        estimator = ElasticNet(alpha=1e-3, l1_ratio=0.15, max_iter=5000, random_state=int(random_seed))
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    if model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            random_state=int(random_seed),
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("LightGBM is not available.") from exc
        estimator = LGBMRegressor(
            n_estimators=320,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=int(random_seed),
            n_jobs=-1,
            verbosity=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    raise ValueError(f"Unsupported model: {model_name}")


def make_single_models(random_seed: int) -> Dict[str, Pipeline]:
    """Create all single-output regression models available in the environment."""

    names = ["ridge", "elasticnet", "hist_gradient_boosting"]
    models = {name: make_single_model(name, random_seed) for name in names}
    try:
        models["lightgbm"] = make_single_model("lightgbm", random_seed)
    except RuntimeError:
        pass
    return models


def split_arrays(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return train and valid feature arrays for selected columns."""

    train = df.loc[df["set_type"] == "train", list(feature_cols)].to_numpy(dtype=np.float32)
    valid = df.loc[df["set_type"] == "valid", list(feature_cols)].to_numpy(dtype=np.float32)
    return train, valid


def split_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/valid compact2 and retention targets."""

    train_mask = df["set_type"] == "train"
    valid_mask = df["set_type"] == "valid"
    y_train_dqdv = df.loc[train_mask, TARGET_COLS].to_numpy(dtype=np.float32)
    y_valid_dqdv = df.loc[valid_mask, TARGET_COLS].to_numpy(dtype=np.float32)
    y_train_ret = df.loc[train_mask, "retention"].to_numpy(dtype=np.float32)
    y_valid_ret = df.loc[valid_mask, "retention"].to_numpy(dtype=np.float32)
    return y_train_dqdv, y_valid_dqdv, y_train_ret, y_valid_ret


def metrics_to_frame(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Convert metric row mappings to a dataframe."""

    return pd.DataFrame(list(rows))


def build_target_metrics(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_valid_true: np.ndarray,
    y_valid_pred: np.ndarray,
    model_name: str,
    target_cols: Sequence[str],
) -> List[Dict[str, object]]:
    """Build per-target train and valid metrics."""

    rows: List[Dict[str, object]] = []
    for set_type, y_true, y_pred in [
        ("train", y_train_true, y_train_pred),
        ("valid", y_valid_true, y_valid_pred),
    ]:
        for idx, target in enumerate(target_cols):
            rows.append(asdict(calc_metrics(y_true[:, idx], y_pred[:, idx], model_name, set_type, target, 1)))
    return rows


def build_retention_metric_rows(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_valid_true: np.ndarray,
    y_valid_pred: np.ndarray,
    model_name: str,
    source_model: str,
) -> List[Dict[str, object]]:
    """Build train and valid retention metric rows."""

    rows: List[Dict[str, object]] = []
    for set_type, y_true, y_pred in [
        ("train", y_train_true, y_train_pred),
        ("valid", y_valid_true, y_valid_pred),
    ]:
        row = asdict(calc_metrics(y_true, y_pred, model_name, set_type, "retention", 1))
        row["source_model"] = source_model
        rows.append(row)
    return rows


def valid_mean_r2(metrics_df: pd.DataFrame, model_name: str) -> float:
    """Return mean valid R2 over compact2 targets for one model."""

    rows = metrics_df.loc[(metrics_df["model_name"] == model_name) & (metrics_df["set_type"] == "valid")]
    if rows.empty:
        return float("-inf")
    return float(rows["r2"].astype(float).mean())


def valid_retention_r2(metrics_df: pd.DataFrame, model_name: str) -> float:
    """Return valid retention R2 for one model."""

    rows = metrics_df.loc[(metrics_df["model_name"] == model_name) & (metrics_df["set_type"] == "valid")]
    if rows.empty:
        return float("-inf")
    return float(rows["r2"].astype(float).iloc[0])


def fit_compact2_models(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    random_seed: int,
    model_prefix: str,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Pipeline]]:
    """Train compact2 predictors and return metrics, predictions, and fitted models."""

    metric_rows: List[Dict[str, object]] = []
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    fitted: Dict[str, Pipeline] = {}
    for name, model in make_multioutput_models(random_seed).items():
        model_name = f"{model_prefix}_{name}"
        print(f"Training compact2 model: {model_name}", flush=True)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train).astype(np.float32)
        valid_pred = model.predict(x_valid).astype(np.float32)
        metric_rows.extend(build_target_metrics(y_train, train_pred, y_valid, valid_pred, model_name, TARGET_COLS))
        predictions[model_name] = (train_pred, valid_pred)
        fitted[model_name] = model
    return metrics_to_frame(metric_rows), predictions, fitted


def fit_retention_candidates(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    random_seed: int,
    model_prefix: str,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Pipeline]]:
    """Train single-output retention candidates and return metrics and predictions."""

    metric_rows: List[Dict[str, object]] = []
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    fitted: Dict[str, Pipeline] = {}
    for name, model in make_single_models(random_seed).items():
        model_name = f"{model_prefix}_{name}"
        print(f"Training retention model: {model_name}", flush=True)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train).astype(np.float32)
        valid_pred = model.predict(x_valid).astype(np.float32)
        metric_rows.extend(build_retention_metric_rows(y_train, train_pred, y_valid, valid_pred, model_name, name))
        predictions[model_name] = (train_pred, valid_pred)
        fitted[model_name] = model
    return metrics_to_frame(metric_rows), predictions, fitted


def best_model_name_by_valid_r2(metrics_df: pd.DataFrame) -> str:
    """Pick the model with the highest valid R2."""

    valid = metrics_df.loc[metrics_df["set_type"] == "valid"].copy()
    if valid.empty:
        raise RuntimeError("No valid metrics available for model selection.")
    grouped = valid.groupby("model_name", as_index=False)["r2"].mean()
    grouped = grouped.sort_values(["r2", "model_name"], ascending=[False, True], kind="mergesort")
    return str(grouped.iloc[0]["model_name"])


def fit_selected_bridge(
    x_train_true: np.ndarray,
    x_valid_true: np.ndarray,
    x_train_pred: np.ndarray,
    x_valid_pred: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Train compact2-to-retention bridge candidates and evaluate oracle/deployable paths."""

    candidate_metrics, _candidate_predictions, candidate_models = fit_retention_candidates(
        x_train_true,
        x_valid_true,
        y_train,
        y_valid,
        random_seed=random_seed,
        model_prefix="oracle_bridge_candidate",
    )
    best_candidate_name = best_model_name_by_valid_r2(candidate_metrics)
    bridge_model = candidate_models[best_candidate_name]
    bridge_algo = best_candidate_name.replace("oracle_bridge_candidate_", "")
    oracle_train_pred = bridge_model.predict(x_train_true).astype(np.float32)
    oracle_valid_pred = bridge_model.predict(x_valid_true).astype(np.float32)
    deploy_train_pred = bridge_model.predict(x_train_pred).astype(np.float32)
    deploy_valid_pred = bridge_model.predict(x_valid_pred).astype(np.float32)
    metric_rows = [
        *build_retention_metric_rows(y_train, oracle_train_pred, y_valid, oracle_valid_pred, "oracle_bridge", bridge_algo),
        *build_retention_metric_rows(y_train, deploy_train_pred, y_valid, deploy_valid_pred, "deployable_bridge", bridge_algo),
    ]
    pred_df = pd.concat(
        [
            build_retention_prediction_frame("oracle_bridge", y_train, y_valid, oracle_train_pred, oracle_valid_pred),
            build_retention_prediction_frame("deployable_bridge", y_train, y_valid, deploy_train_pred, deploy_valid_pred),
        ],
        ignore_index=True,
    )
    return pd.DataFrame(metric_rows), pred_df, best_candidate_name, bridge_algo


def build_retention_prediction_frame(
    model_name: str,
    y_train_true: np.ndarray,
    y_valid_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_valid_pred: np.ndarray,
) -> pd.DataFrame:
    """Build a simple long retention prediction table."""

    rows: List[Dict[str, object]] = []
    for set_type, y_true, y_pred in [
        ("train", y_train_true, y_train_pred),
        ("valid", y_valid_true, y_valid_pred),
    ]:
        for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
            rows.append(
                {
                    "row_id": int(idx),
                    "set_type": set_type,
                    "model_name": model_name,
                    "retention_true": float(truth),
                    "pred_retention": float(pred),
                    "residual_retention": float(truth) - float(pred),
                }
            )
    return pd.DataFrame(rows)


def attach_meta_to_prediction_frame(pred_df: pd.DataFrame, train_meta: pd.DataFrame, valid_meta: pd.DataFrame) -> pd.DataFrame:
    """Attach policy/cell/cycle metadata to retention predictions."""

    meta = pd.concat(
        [
            train_meta.reset_index(drop=True).assign(row_id=lambda frame: frame.index, set_type="train"),
            valid_meta.reset_index(drop=True).assign(row_id=lambda frame: frame.index, set_type="valid"),
        ],
        ignore_index=True,
    )
    return pred_df.merge(meta, on=["set_type", "row_id"], how="left")


def build_compact2_prediction_frame(
    train_meta: pd.DataFrame,
    valid_meta: pd.DataFrame,
    y_train_true: np.ndarray,
    y_valid_true: np.ndarray,
    predictions: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Build compact2 prediction rows for selected first-stage paths."""

    rows: List[pd.DataFrame] = []
    for model_name, (train_pred, valid_pred) in predictions.items():
        for set_type, meta, y_true, y_pred in [
            ("train", train_meta, y_train_true, train_pred),
            ("valid", valid_meta, y_valid_true, valid_pred),
        ]:
            part = meta.reset_index(drop=True).copy()
            part["set_type"] = set_type
            part["model_name"] = model_name
            for idx, target in enumerate(TARGET_COLS):
                part[f"true_{target}"] = y_true[:, idx].astype(float)
                part[f"pred_{target}"] = y_pred[:, idx].astype(float)
                part[f"residual_{target}"] = y_true[:, idx].astype(float) - y_pred[:, idx].astype(float)
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def sample_valid(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a bounded valid sample for plotting."""

    valid = df.loc[df["set_type"] == "valid"].copy()
    if len(valid) > int(max_rows):
        valid = valid.sample(n=int(max_rows), random_state=int(seed))
    return valid


def save_compact2_scatter(pred_df: pd.DataFrame, model_name: str, out_path: Path, seed: int) -> None:
    """Save true-vs-pred scatter for compact2 targets."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = sample_valid(pred_df.loc[pred_df["model_name"] == model_name], 30000, seed)
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(11, 4.5))
    if len(TARGET_COLS) == 1:
        axes = [axes]
    for ax, target in zip(axes, TARGET_COLS):
        true_col = f"true_{target}"
        pred_col = f"pred_{target}"
        ax.scatter(view[true_col], view[pred_col], s=6, alpha=0.25)
        low = float(min(view[true_col].min(), view[pred_col].min()))
        high = float(max(view[true_col].max(), view[pred_col].max()))
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(target)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_retention_scatter(pred_df: pd.DataFrame, out_path: Path, seed: int) -> None:
    """Save true-vs-pred scatter for retention paths."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = sample_valid(pred_df, 20000, seed)
    model_names = list(valid["model_name"].drop_duplicates())
    n_cols = 3
    n_rows = int(math.ceil(len(model_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    axes_arr = np.array(axes).reshape(-1)
    for ax, model_name in zip(axes_arr, model_names):
        part = valid.loc[valid["model_name"] == model_name]
        ax.scatter(part["retention_true"], part["pred_retention"], s=6, alpha=0.25)
        low = float(min(part["retention_true"].min(), part["pred_retention"].min()))
        high = float(max(part["retention_true"].max(), part["pred_retention"].max()))
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(str(model_name))
        ax.set_xlabel("True retention")
        ax.set_ylabel("Predicted retention")
        ax.grid(True, linestyle="--", alpha=0.3)
    for ax in axes_arr[len(model_names) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_r2_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Save valid R2 comparison bar plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = metrics_df.loc[metrics_df["set_type"] == "valid"].copy()
    valid = valid.sort_values("r2", ascending=False, kind="mergesort")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(valid["model_name"], valid["r2"].astype(float), color="#4C78A8")
    ax.set_ylabel("Valid R2")
    ax.set_title("Retention path valid R2 comparison")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_dataset_checks(
    merged: pd.DataFrame,
    recommended_features: Sequence[str],
    full159_features: Sequence[str],
    stats: Mapping[str, int],
) -> pd.DataFrame:
    """Build dataset checks with pass flags."""

    train_keys = set(merged.loc[merged["set_type"] == "train", ["policy", "cell_code"]].drop_duplicates().apply(tuple, axis=1))
    valid_keys = set(merged.loc[merged["set_type"] == "valid", ["policy", "cell_code"]].drop_duplicates().apply(tuple, axis=1))
    forbidden = sorted(set(recommended_features).intersection(FORBIDDEN_INPUT_COLS))
    split_overlap_count = int(len(train_keys.intersection(valid_keys)))
    split_is_disjoint = int(split_overlap_count == 0)
    checks = [
        ("recommended_feature_count", int(len(recommended_features)), int(len(recommended_features) == 55), "expected 55"),
        ("full159_feature_count", int(len(full159_features)), int(len(full159_features) == 159), "expected 159"),
        ("forbidden_input_columns_present", int(len(forbidden)), int(len(forbidden) == 0), ";".join(forbidden)),
        ("merged_cycle_rows", int(len(merged)), int(len(merged) > 0), ""),
        ("train_cycle_rows", int((merged["set_type"] == "train").sum()), int((merged["set_type"] == "train").sum() > 0), ""),
        ("valid_cycle_rows", int((merged["set_type"] == "valid").sum()), int((merged["set_type"] == "valid").sum() > 0), ""),
        ("train_policy_cell_count", int(len(train_keys)), int(len(train_keys) > 0), ""),
        ("valid_policy_cell_count", int(len(valid_keys)), int(len(valid_keys) > 0), ""),
        ("split_overlap_zero", split_is_disjoint, split_is_disjoint, f"overlap_count={split_overlap_count}"),
        ("compact2_target_dim", int(len(TARGET_COLS)), int(len(TARGET_COLS) == 2), ",".join(TARGET_COLS)),
        ("retention_range_check", int(((merged["retention"] >= 0.3) & (merged["retention"] <= 1.1)).all()), int(((merged["retention"] >= 0.3) & (merged["retention"] <= 1.1)).all()), ""),
        ("charge_cross_bin_feature_dim", int(stats.get("charge_cross_bin_feature_dim", 0)), int(stats.get("charge_cross_bin_feature_dim", 0) == 60), ""),
        ("discharge_range_count", int(stats.get("discharge_range_count", 0)), int(stats.get("discharge_range_count", 0) == 16), ""),
    ]
    return pd.DataFrame(checks, columns=["check_item", "value", "pass_flag", "details"])


def write_feature_columns(out_path: Path, recommended_features: Sequence[str], full159_features: Sequence[str]) -> None:
    """Write feature column lists."""

    rows: List[Dict[str, object]] = []
    for idx, feature in enumerate(recommended_features, start=1):
        rows.append({"feature_pack": "recommended55", "rank": idx, "feature": feature})
    for idx, feature in enumerate(full159_features, start=1):
        rows.append({"feature_pack": "full159", "rank": idx, "feature": feature})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding=ENCODING)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional dependencies."""

    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
        else:
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in work.columns]
    rows = work.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    checks_df: pd.DataFrame,
    compact_metrics: pd.DataFrame,
    retention_metrics: pd.DataFrame,
    selected: Mapping[str, str],
    output_dir: Optional[Path] = None,
) -> str:
    """Build the Chinese markdown report."""

    def valid_row(model_name: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """Return a valid metric row by model name."""

        rows = df.loc[(df["model_name"] == model_name) & (df["set_type"] == "valid")]
        return None if rows.empty else rows.iloc[0]

    compact_valid = compact_metrics.loc[compact_metrics["set_type"] == "valid"].copy()
    compact_summary = (
        compact_valid.groupby("model_name", as_index=False)["r2"]
        .mean()
        .rename(columns={"r2": "mean_valid_r2"})
        .sort_values("mean_valid_r2", ascending=False, kind="mergesort")
    )
    deploy = valid_row("deployable_bridge", retention_metrics)
    oracle = valid_row("oracle_bridge", retention_metrics)
    direct = valid_row("direct_retention_baseline", retention_metrics)
    full159_bridge = valid_row("full159_compact2_bridge", retention_metrics)
    full159_direct = valid_row("full159_direct_retention", retention_metrics)

    def image_link(image_name: str) -> str:
        """Return an absolute image link when output_dir is available."""

        image_path = Path(image_name)
        if output_dir is None:
            target = f"./{image_name}"
        else:
            target = (output_dir.resolve() / image_name).as_posix()
        return f"![{image_path.stem}]({target})"

    lines: List[str] = []
    lines.append("# recommended feature pack + compact2 dQdV retention 短链路验证报告")
    lines.append("")
    lines.append("## 1. 摘要")
    lines.append(f"- 入选 compact2 预测器：`{selected.get('best_compact2_model', '')}`")
    lines.append(f"- 入选 compact2->retention bridge：`{selected.get('best_bridge_candidate', '')}`")
    if not compact_summary.empty:
        lines.append(f"- recommended55 compact2 最佳 valid 平均 R2：`{float(compact_summary.iloc[0]['mean_valid_r2']):.6f}`")
    if oracle is not None:
        lines.append(f"- oracle bridge valid R2：`{float(oracle['r2']):.6f}`")
    if deploy is not None:
        lines.append(f"- deployable bridge valid R2：`{float(deploy['r2']):.6f}`")
    if direct is not None:
        lines.append(f"- direct recommended55 retention valid R2：`{float(direct['r2']):.6f}`")
    if full159_direct is not None:
        lines.append(f"- full159 direct retention valid R2：`{float(full159_direct['r2']):.6f}`")
    lines.append("")
    lines.append("## 2. 数据检查")
    lines.append(dataframe_to_markdown(checks_df))
    lines.append("")
    lines.append("## 3. compact2 可预测性")
    lines.append(dataframe_to_markdown(compact_summary))
    lines.append("")
    lines.append("## 4. retention 链路指标")
    display_cols = ["model_name", "set_type", "source_model", "mse", "rmse", "mae", "r2"]
    lines.append(dataframe_to_markdown(retention_metrics[display_cols]))
    lines.append("")
    lines.append("## 5. 必答结论")
    if oracle is None or deploy is None or direct is None:
        lines.append("- 部分必要指标缺失，暂不建议做训练路线判断。")
    else:
        oracle_r2 = float(oracle["r2"])
        deploy_r2 = float(deploy["r2"])
        direct_r2 = float(direct["r2"])
        gap_oracle_deploy = oracle_r2 - deploy_r2
        gap_direct_deploy = direct_r2 - deploy_r2
        if compact_summary.empty:
            lines.append("- 55维工况 -> compact2：缺少可预测性指标，不能判断。")
        else:
            lines.append(
                "- 55维工况 -> compact2："
                f"最佳模型 `{compact_summary.iloc[0]['model_name']}` 的 valid 平均 R2 为 `{float(compact_summary.iloc[0]['mean_valid_r2']):.6f}`，"
                "说明 compact2 在 cycle 级表格特征下已经足够可预测。"
            )
        lines.append(f"- 真实 compact2 -> retention 上限：oracle bridge valid R2 为 `{oracle_r2:.6f}`。")
        lines.append(f"- 预测 compact2 -> retention 部署链路：deployable bridge valid R2 为 `{deploy_r2:.6f}`。")
        lines.append(f"- oracle 到 deployable 的 R2 损失：`{gap_oracle_deploy:.6f}`。")
        lines.append(
            "- direct retention baseline 对比："
            f"direct valid R2 为 `{direct_r2:.6f}`，比 deployable bridge 高 `{gap_direct_deploy:.6f}`。"
        )
        if full159_bridge is not None:
            lines.append(f"- full159 compact2 bridge valid R2：`{float(full159_bridge['r2']):.6f}`。")
        if full159_direct is not None:
            lines.append(f"- full159 direct retention valid R2：`{float(full159_direct['r2']):.6f}`。")
        if deploy_r2 >= direct_r2 - 0.01:
            lines.append("- 建议：compact2 bridge 与 direct baseline 接近，可以继续做短历史序列训练，再考虑扩展到 compact3。")
        elif oracle_r2 > direct_r2 and gap_oracle_deploy > 0.05:
            lines.append("- 建议：compact2 含有 retention 信息，但工况 -> compact2 是瓶颈，应先提升 compact2 预测再做长窗口训练。")
        else:
            lines.append("- 建议：direct retention baseline 明显更强，compact2 暂时更适合作解释层，不建议优先投入 compact2 长窗口训练。")
    lines.append("")
    lines.append("## 6. 图表")
    lines.append(image_link("compact2_predicted_vs_true.png"))
    lines.append("")
    lines.append(image_link("retention_predicted_vs_true.png"))
    lines.append("")
    lines.append(image_link("bridge_r2_comparison.png"))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, str]:
    """Run the full short-chain experiment."""

    set_seed(int(args.random_seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged, recommended_features, full159_features, stats = build_cycle_table(args)
    checks_df = build_dataset_checks(merged, recommended_features, full159_features, stats)
    checks_df.to_csv(args.output_dir / "dataset_checks.csv", index=False, encoding=ENCODING)
    if int(checks_df["pass_flag"].min()) != 1:
        raise RuntimeError(f"Dataset checks failed. See {args.output_dir / 'dataset_checks.csv'}.")

    train_meta = merged.loc[merged["set_type"] == "train", ["policy", "cell_code", "cycles", "q_ref", "q_discharge"]].reset_index(drop=True)
    valid_meta = merged.loc[merged["set_type"] == "valid", ["policy", "cell_code", "cycles", "q_ref", "q_discharge"]].reset_index(drop=True)
    x_train_rec, x_valid_rec = split_arrays(merged, recommended_features)
    x_train_full, x_valid_full = split_arrays(merged, full159_features)
    y_train_dqdv, y_valid_dqdv, y_train_ret, y_valid_ret = split_targets(merged)

    compact_metrics_rec, compact_predictions_rec, _compact_models_rec = fit_compact2_models(
        x_train_rec,
        x_valid_rec,
        y_train_dqdv,
        y_valid_dqdv,
        random_seed=int(args.random_seed) + 100,
        model_prefix="recommended55",
    )
    best_compact_model = best_model_name_by_valid_r2(compact_metrics_rec)
    best_train_pred, best_valid_pred = compact_predictions_rec[best_compact_model]

    compact_metrics_full, compact_predictions_full, _compact_models_full = fit_compact2_models(
        x_train_full,
        x_valid_full,
        y_train_dqdv,
        y_valid_dqdv,
        random_seed=int(args.random_seed) + 200,
        model_prefix="full159",
    )
    full159_lgbm_name = "full159_lightgbm" if "full159_lightgbm" in compact_predictions_full else best_model_name_by_valid_r2(compact_metrics_full)
    full159_train_pred, full159_valid_pred = compact_predictions_full[full159_lgbm_name]
    compact_metrics = pd.concat([compact_metrics_rec, compact_metrics_full], ignore_index=True)

    bridge_metrics, bridge_pred_df, best_bridge_candidate, bridge_algo = fit_selected_bridge(
        x_train_true=y_train_dqdv,
        x_valid_true=y_valid_dqdv,
        x_train_pred=best_train_pred,
        x_valid_pred=best_valid_pred,
        y_train=y_train_ret,
        y_valid=y_valid_ret,
        random_seed=int(args.random_seed) + 300,
    )
    bridge_candidate_model = make_single_model(bridge_algo, int(args.random_seed) + 300)
    bridge_candidate_model.fit(y_train_dqdv, y_train_ret)
    full159_bridge_train_pred = bridge_candidate_model.predict(full159_train_pred).astype(np.float32)
    full159_bridge_valid_pred = bridge_candidate_model.predict(full159_valid_pred).astype(np.float32)
    bridge_metrics = pd.concat(
        [
            bridge_metrics,
            pd.DataFrame(
                build_retention_metric_rows(
                    y_train_ret,
                    full159_bridge_train_pred,
                    y_valid_ret,
                    full159_bridge_valid_pred,
                    "full159_compact2_bridge",
                    bridge_algo,
                )
            ),
        ],
        ignore_index=True,
    )
    full159_bridge_pred_df = build_retention_prediction_frame(
        "full159_compact2_bridge",
        y_train_ret,
        y_valid_ret,
        full159_bridge_train_pred,
        full159_bridge_valid_pred,
    )

    direct_metrics_rec, direct_predictions_rec, _direct_models_rec = fit_retention_candidates(
        x_train_rec,
        x_valid_rec,
        y_train_ret,
        y_valid_ret,
        random_seed=int(args.random_seed) + 400,
        model_prefix="direct55_candidate",
    )
    best_direct_rec = best_model_name_by_valid_r2(direct_metrics_rec)
    direct_train_pred, direct_valid_pred = direct_predictions_rec[best_direct_rec]
    bridge_metrics = pd.concat(
        [
            bridge_metrics,
            pd.DataFrame(
                build_retention_metric_rows(
                    y_train_ret,
                    direct_train_pred,
                    y_valid_ret,
                    direct_valid_pred,
                    "direct_retention_baseline",
                    best_direct_rec.replace("direct55_candidate_", ""),
                )
            ),
        ],
        ignore_index=True,
    )
    direct_pred_df = build_retention_prediction_frame(
        "direct_retention_baseline",
        y_train_ret,
        y_valid_ret,
        direct_train_pred,
        direct_valid_pred,
    )

    full159_direct_model = make_single_model("lightgbm", int(args.random_seed) + 500)
    full159_direct_model.fit(x_train_full, y_train_ret)
    full159_direct_train_pred = full159_direct_model.predict(x_train_full).astype(np.float32)
    full159_direct_valid_pred = full159_direct_model.predict(x_valid_full).astype(np.float32)
    bridge_metrics = pd.concat(
        [
            bridge_metrics,
            pd.DataFrame(
                build_retention_metric_rows(
                    y_train_ret,
                    full159_direct_train_pred,
                    y_valid_ret,
                    full159_direct_valid_pred,
                    "full159_direct_retention",
                    "lightgbm",
                )
            ),
        ],
        ignore_index=True,
    )
    full159_direct_pred_df = build_retention_prediction_frame(
        "full159_direct_retention",
        y_train_ret,
        y_valid_ret,
        full159_direct_train_pred,
        full159_direct_valid_pred,
    )

    compact_pred_df = build_compact2_prediction_frame(
        train_meta,
        valid_meta,
        y_train_dqdv,
        y_valid_dqdv,
        {
            best_compact_model: (best_train_pred, best_valid_pred),
            full159_lgbm_name: (full159_train_pred, full159_valid_pred),
        },
    )
    retention_pred_df = pd.concat(
        [bridge_pred_df, full159_bridge_pred_df, direct_pred_df, full159_direct_pred_df],
        ignore_index=True,
    )
    retention_pred_df = attach_meta_to_prediction_frame(retention_pred_df, train_meta, valid_meta)

    out = args.output_dir
    checks_df.to_csv(out / "dataset_checks.csv", index=False, encoding=ENCODING)
    write_feature_columns(out / "feature_columns.csv", recommended_features, full159_features)
    compact_metrics.to_csv(out / "compact2_prediction_metrics.csv", index=False, encoding=ENCODING)
    compact_pred_df.to_csv(out / "compact2_predictions.csv", index=False, encoding=ENCODING)
    bridge_metrics.to_csv(out / "retention_bridge_metrics.csv", index=False, encoding=ENCODING)
    retention_pred_df.to_csv(out / "retention_predictions.csv", index=False, encoding=ENCODING)
    save_compact2_scatter(compact_pred_df, best_compact_model, out / "compact2_predicted_vs_true.png", int(args.random_seed))
    save_retention_scatter(retention_pred_df, out / "retention_predicted_vs_true.png", int(args.random_seed))
    save_r2_bar(bridge_metrics, out / "bridge_r2_comparison.png")
    selected = {
        "best_compact2_model": best_compact_model,
        "full159_compact2_model": full159_lgbm_name,
        "best_bridge_candidate": best_bridge_candidate,
        "best_direct55_model": best_direct_rec,
    }
    report = build_report(checks_df, compact_metrics, bridge_metrics, selected, out)
    (out / "interval_feature_pack_compact2_retention_bridge_report.md").write_text(report, encoding="utf-8")
    config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "target_columns": TARGET_COLS,
        "recommended_feature_columns": list(recommended_features),
        "full159_feature_columns": list(full159_features),
        "selected_models": selected,
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved outputs to: {out}", flush=True)
    return {
        "output_dir": str(out),
        "report_path": str(out / "interval_feature_pack_compact2_retention_bridge_report.md"),
    }


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
