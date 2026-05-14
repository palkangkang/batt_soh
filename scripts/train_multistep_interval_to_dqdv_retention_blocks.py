"""Block-based multistep interval -> dQdV -> retention validation."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from train_interval_to_dqdv_retention_pipeline import (
    ENCODING,
    REPO_ROOT,
    build_allowed_key_tokens,
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


TARGET_PACKS: Dict[str, List[str]] = {
    "compact2": ["main_peak_area", "main_peak_height_dqdv"],
    "compact3": ["main_peak_area", "main_peak_height_dqdv", "main_peak_voltage_v"],
    "compact4": [
        "main_peak_area",
        "main_peak_height_dqdv",
        "main_peak_voltage_v",
        "main_peak_skewness",
    ],
}
FORBIDDEN_INPUT_COLS = {
    "cycles",
    "cycle_index_norm",
    "policy",
    "cell_code",
    "initial_c_rate",
    "switch_soc_percent",
    "post_switch_c_rate",
}
SUMMARY_STATS = ["last", "mean", "std", "min", "max", "delta", "slope"]
SELECTED_HORIZONS = [1, 5, 10, 20, 50]
warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)


@dataclass
class BlockSample:
    """Metadata and arrays for one non-overlapping future-prediction block."""

    block_id: int
    set_type: str
    policy: str
    cell_code: str
    block_index: int
    input_start_cycle: int
    input_end_cycle: int
    target_start_cycle: int
    target_end_cycle: int
    history_x: np.ndarray
    history_cycles: np.ndarray
    history_retention: np.ndarray
    future_cycles: np.ndarray
    future_dqdv: np.ndarray
    future_retention: np.ndarray
    future_q_ref: np.ndarray
    future_q_discharge: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Block-based future prediction: interval stats -> dQdV -> retention."
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
        default=REPO_ROOT
        / "outputs"
        / "analysis"
        / "multistep_interval_to_dqdv_retention_blocks_h100_m50",
    )
    parser.add_argument("--split-name", type=str, default="balanced")
    parser.add_argument("--history-len", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--block-stride", type=int, default=150)
    parser.add_argument("--sample-mode", choices=["non_overlapping_blocks"], default="non_overlapping_blocks")
    parser.add_argument("--block-stage-filter", choices=["none", "early_train_late_valid"], default="none")
    parser.add_argument("--train-max-relative-input-end", type=float, default=0.45)
    parser.add_argument("--valid-min-relative-input-start", type=float, default=0.55)
    parser.add_argument("--feature-pack", choices=["recommended55", "full159"], default="recommended55")
    parser.add_argument("--target-pack", choices=sorted(TARGET_PACKS), default="compact4")
    parser.add_argument("--history-representation", choices=["summary", "flatten"], default="summary")
    parser.add_argument(
        "--include-history-retention-summary",
        action="store_true",
        help="Train an additional direct_retention_with_history_summary route using seven historical retention summary features.",
    )
    parser.add_argument("--model-family", choices=["lightgbm"], default="lightgbm")
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-train-cells", type=int, default=12)
    parser.add_argument("--smoke-valid-cells", type=int, default=6)
    parser.add_argument("--max-train-blocks", type=int, default=0)
    parser.add_argument("--max-valid-blocks", type=int, default=0)
    return parser.parse_args()


def load_recommended_features(path: Path) -> List[str]:
    """Load recommended feature names and reject forbidden deployment columns."""

    df = pd.read_csv(path, encoding=ENCODING)
    if "feature" not in df.columns:
        raise ValueError(f"Missing feature column in {path}")
    features = dedupe_keep_order([str(item).strip() for item in df["feature"].dropna().tolist()])
    features = [feature for feature in features if feature]
    forbidden = sorted(set(features).intersection(FORBIDDEN_INPUT_COLS))
    if forbidden:
        raise RuntimeError(f"Forbidden input columns in recommended feature pack: {forbidden}")
    if not features:
        raise RuntimeError(f"No features found in {path}")
    return features


def make_lgbm(seed: int) -> Pipeline:
    """Create a LightGBM single-output regression pipeline."""

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for this script.") from exc
    estimator = LGBMRegressor(
        n_estimators=260,
        learning_rate=0.045,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=5,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics for one target vector."""

    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mse = float(mean_squared_error(true, pred))
    return {
        "n_rows": int(true.size),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)) if true.size > 1 else float("nan"),
    }


def resolved_path_text(path: Path) -> str:
    """Return a stable absolute path string for configs and reports."""

    return Path(path).expanduser().resolve().as_posix()


def build_cycle_table(args: argparse.Namespace, target_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Load and merge cycle-level interval features, dQdV targets, retention labels, and split labels."""

    recommended_features = load_recommended_features(args.recommended_feature_path)
    _train_split, _valid_split, split_map = load_split(args.train_split_path, args.valid_split_path)
    allowed_keys = (
        build_allowed_key_tokens(split_map, args.smoke_train_cells, args.smoke_valid_cells)
        if bool(args.smoke_test)
        else None
    )

    print("Loading charge features...", flush=True)
    charge_df, charge_stats, charge_cols = load_charge_feature_table(args.charge_timeseries_path, allowed_keys)
    print("Loading discharge features...", flush=True)
    discharge_df, discharge_stats, discharge_cols = load_discharge_feature_table(args.discharge_interval_path, allowed_keys)
    interval_df = charge_df.merge(discharge_df, on=["policy", "cell_code", "cycles"], how="outer")
    interval_cols = dedupe_keep_order([*charge_cols, *discharge_cols])
    for col in interval_cols:
        interval_df[col] = pd.to_numeric(interval_df[col], errors="coerce").fillna(0.0).astype(np.float32)

    full159_features = select_input_feature_columns(interval_cols, "charge_crossbin_discharge_capacity_stats")
    feature_cols = recommended_features if args.feature_pack == "recommended55" else full159_features
    missing = sorted(set(feature_cols).difference(interval_cols))
    if missing:
        raise RuntimeError(f"Requested feature columns are missing from interval table: {missing}")
    forbidden = sorted(set(feature_cols).intersection(FORBIDDEN_INPUT_COLS))
    if forbidden:
        raise RuntimeError(f"Forbidden input columns selected: {forbidden}")

    print("Loading dQdV targets and retention labels...", flush=True)
    dqdv_df = filter_allowed_keys(load_dqdv_table(args.dqdv_path, list(target_cols)), allowed_keys)
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
    merged = (
        labels_df.merge(dqdv_df, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(split_use[["policy", "cell_code", "set_type"]], on=["policy", "cell_code"], how="inner")
        .merge(interval_df[["policy", "cell_code", "cycles", *feature_cols]], on=["policy", "cell_code", "cycles"], how="left")
    )
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(np.float32)
    for col in [*target_cols, "retention", "q_ref", "q_discharge"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=[*target_cols, "retention", "q_ref", "q_discharge"]).copy()
    merged = merged.sort_values(["set_type", "policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    stats = {
        "recommended_feature_count": int(len(recommended_features)),
        "full159_feature_count": int(len(full159_features)),
        "feature_count": int(len(feature_cols)),
        "charge_cross_bin_feature_dim": int(charge_stats.get("charge_cross_bin_feature_dim", 0)),
        "discharge_range_count": int(discharge_stats.get("discharge_range_count", 0)),
        "forbidden_input_columns_present": int(len(set(feature_cols).intersection(FORBIDDEN_INPUT_COLS))),
    }
    return merged, list(feature_cols), stats


def is_consecutive(cycles: np.ndarray) -> bool:
    """Return true when cycle numbers are strictly consecutive by one."""

    if cycles.size <= 1:
        return True
    expected = int(cycles[0]) + np.arange(cycles.size, dtype=np.int64)
    return bool(np.array_equal(cycles.astype(np.int64), expected))


def build_block_samples(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    history_len: int,
    horizon: int,
    block_stride: int,
) -> List[BlockSample]:
    """Build non-overlapping 1:N -> N+1:N+M block samples per policy-cell."""

    samples: List[BlockSample] = []
    total_len = int(history_len) + int(horizon)
    block_id = 0
    for (policy, cell_code), group in merged.groupby(["policy", "cell_code"], sort=False):
        group = group.sort_values("cycles", kind="mergesort").reset_index(drop=True)
        n_rows = int(len(group))
        block_index = 0
        for start_idx in range(0, n_rows - total_len + 1, int(block_stride)):
            end_idx = start_idx + total_len
            span = group.iloc[start_idx:end_idx].copy()
            cycles = span["cycles"].to_numpy(dtype=np.int64)
            if not is_consecutive(cycles):
                continue
            history = span.iloc[:history_len]
            future = span.iloc[history_len:total_len]
            samples.append(
                BlockSample(
                    block_id=int(block_id),
                    set_type=str(span["set_type"].iloc[0]),
                    policy=str(policy),
                    cell_code=str(cell_code),
                    block_index=int(block_index),
                    input_start_cycle=int(history["cycles"].iloc[0]),
                    input_end_cycle=int(history["cycles"].iloc[-1]),
                    target_start_cycle=int(future["cycles"].iloc[0]),
                    target_end_cycle=int(future["cycles"].iloc[-1]),
                    history_x=history[list(feature_cols)].to_numpy(dtype=np.float32),
                    history_cycles=history["cycles"].to_numpy(dtype=np.float32),
                    history_retention=history["retention"].to_numpy(dtype=np.float32),
                    future_cycles=future["cycles"].to_numpy(dtype=np.float32),
                    future_dqdv=future[list(target_cols)].to_numpy(dtype=np.float32),
                    future_retention=future["retention"].to_numpy(dtype=np.float32),
                    future_q_ref=future["q_ref"].to_numpy(dtype=np.float32),
                    future_q_discharge=future["q_discharge"].to_numpy(dtype=np.float32),
                )
            )
            block_id += 1
            block_index += 1
    return samples


def downsample_blocks(samples: Sequence[BlockSample], max_train: int, max_valid: int, seed: int) -> List[BlockSample]:
    """Optionally downsample train and valid blocks while preserving split labels."""

    rng = np.random.default_rng(int(seed))
    result: List[BlockSample] = []
    for set_type, limit in [("train", int(max_train)), ("valid", int(max_valid))]:
        part = [sample for sample in samples if sample.set_type == set_type]
        if limit > 0 and len(part) > limit:
            keep = np.sort(rng.choice(len(part), size=limit, replace=False))
            part = [part[int(idx)] for idx in keep]
        result.extend(part)
    return sorted(result, key=lambda item: (item.set_type, item.policy, item.cell_code, item.input_start_cycle))


def max_cycles_by_key(merged: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Return max observed cycle for each policy-cell key."""

    result: Dict[Tuple[str, str], int] = {}
    grouped = merged.groupby(["policy", "cell_code"], sort=False)["cycles"].max()
    for (policy, cell_code), max_cycle in grouped.items():
        result[(str(policy), str(cell_code))] = int(max_cycle)
    return result


def relative_stage(sample: BlockSample, max_cycles: Mapping[Tuple[str, str], int], point: str) -> float:
    """Calculate a block's relative stage using input start or input end cycle."""

    denominator = int(max_cycles.get((sample.policy, sample.cell_code), 0))
    if denominator <= 0:
        return float("nan")
    numerator = sample.input_start_cycle if point == "input_start" else sample.input_end_cycle
    return float(numerator) / float(denominator)


def filter_blocks_by_stage(
    samples: Sequence[BlockSample],
    max_cycles: Mapping[Tuple[str, str], int],
    mode: str,
    train_max_relative_input_end: float,
    valid_min_relative_input_start: float,
) -> List[BlockSample]:
    """Apply optional early-train and late-valid block filtering."""

    if mode == "none":
        return list(samples)
    if mode != "early_train_late_valid":
        raise ValueError(f"Unsupported block stage filter: {mode}")
    kept: List[BlockSample] = []
    for sample in samples:
        if sample.set_type == "train":
            rel_end = relative_stage(sample, max_cycles, "input_end")
            if np.isfinite(rel_end) and rel_end <= float(train_max_relative_input_end):
                kept.append(sample)
        elif sample.set_type == "valid":
            rel_start = relative_stage(sample, max_cycles, "input_start")
            if np.isfinite(rel_start) and rel_start >= float(valid_min_relative_input_start):
                kept.append(sample)
    return kept


def summarize_history(history_x: np.ndarray) -> np.ndarray:
    """Summarize one fixed history block into last/mean/std/min/max/delta/slope features."""

    x = np.asarray(history_x, dtype=np.float32)
    positions = np.arange(x.shape[0], dtype=np.float32)
    centered = positions - positions.mean()
    denom = float(np.sum(centered**2))
    if denom <= 0.0:
        slope = np.zeros(x.shape[1], dtype=np.float32)
    else:
        slope = (centered[:, None] * (x - x.mean(axis=0, keepdims=True))).sum(axis=0) / denom
    return np.concatenate(
        [
            x[-1],
            x.mean(axis=0),
            x.std(axis=0),
            x.min(axis=0),
            x.max(axis=0),
            x[-1] - x[0],
            slope.astype(np.float32),
        ]
    ).astype(np.float32)


def build_history_matrix(samples: Sequence[BlockSample], feature_cols: Sequence[str], representation: str) -> Tuple[np.ndarray, List[str]]:
    """Create model input matrix from block histories."""

    if representation == "summary":
        rows = [summarize_history(sample.history_x) for sample in samples]
        columns = [f"{feature}__{stat}" for stat in SUMMARY_STATS for feature in feature_cols]
        return np.vstack(rows).astype(np.float32), columns
    if representation == "flatten":
        rows = [sample.history_x.reshape(-1) for sample in samples]
        columns = [f"{feature}__t{idx + 1:03d}" for idx in range(samples[0].history_x.shape[0]) for feature in feature_cols]
        return np.vstack(rows).astype(np.float32), columns
    raise ValueError(f"Unknown history representation: {representation}")


def summarize_retention_history(history_retention: np.ndarray) -> np.ndarray:
    """Summarize one fixed retention history as last/mean/std/min/max/delta/slope."""

    y = np.asarray(history_retention, dtype=np.float32)
    positions = np.arange(y.shape[0], dtype=np.float32)
    centered = positions - positions.mean()
    denom = float(np.sum(centered**2))
    if denom <= 0.0:
        slope = 0.0
    else:
        slope = float(np.sum(centered * (y - y.mean())) / denom)
    return np.asarray(
        [
            float(y[-1]),
            float(y.mean()),
            float(y.std()),
            float(y.min()),
            float(y.max()),
            float(y[-1] - y[0]),
            slope,
        ],
        dtype=np.float32,
    )


def build_history_retention_matrix(samples: Sequence[BlockSample]) -> Tuple[np.ndarray, List[str]]:
    """Create seven historical retention summary features for optional tabular baselines."""

    rows = [summarize_retention_history(sample.history_retention) for sample in samples]
    columns = [f"history_retention__{stat}" for stat in SUMMARY_STATS]
    return np.vstack(rows).astype(np.float32), columns


def future_arrays(samples: Sequence[BlockSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack future arrays for dQdV, retention, q_ref, q_discharge, and cycles."""

    dqdv = np.stack([sample.future_dqdv for sample in samples]).astype(np.float32)
    retention = np.stack([sample.future_retention for sample in samples]).astype(np.float32)
    q_ref = np.stack([sample.future_q_ref for sample in samples]).astype(np.float32)
    q_discharge = np.stack([sample.future_q_discharge for sample in samples]).astype(np.float32)
    cycles = np.stack([sample.future_cycles for sample in samples]).astype(np.float32)
    return dqdv, retention, q_ref, q_discharge, cycles


def train_dqdv_models(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    target_cols: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Train one LightGBM model per horizon and dQdV target."""

    horizon = int(y_train.shape[1])
    n_targets = int(y_train.shape[2])
    train_pred = np.zeros_like(y_train, dtype=np.float32)
    valid_pred = np.zeros_like(y_valid, dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for step in range(horizon):
        for target_idx, target in enumerate(target_cols):
            model = make_lgbm(seed + step * 31 + target_idx)
            model.fit(x_train, y_train[:, step, target_idx])
            train_pred[:, step, target_idx] = model.predict(x_train).astype(np.float32)
            valid_pred[:, step, target_idx] = model.predict(x_valid).astype(np.float32)
            for set_type, truth, pred in [
                ("train", y_train[:, step, target_idx], train_pred[:, step, target_idx]),
                ("valid", y_valid[:, step, target_idx], valid_pred[:, step, target_idx]),
            ]:
                row = regression_metrics(truth, pred)
                row.update(
                    {
                        "stage": "dqdv_prediction",
                        "method": "interval_to_dqdv_lightgbm",
                        "set_type": set_type,
                        "target": target,
                        "horizon": f"H{step + 1}",
                        "horizon_step": int(step + 1),
                    }
                )
                rows.append(row)
    for target_idx, target in enumerate(target_cols):
        for set_type, truth, pred in [
            ("train", y_train[:, :, target_idx], train_pred[:, :, target_idx]),
            ("valid", y_valid[:, :, target_idx], valid_pred[:, :, target_idx]),
        ]:
            row = regression_metrics(truth.reshape(-1), pred.reshape(-1))
            row.update(
                {
                    "stage": "dqdv_prediction",
                    "method": "interval_to_dqdv_lightgbm",
                    "set_type": set_type,
                    "target": target,
                    "horizon": "all",
                    "horizon_step": 0,
                }
            )
            rows.append(row)
    return train_pred, valid_pred, pd.DataFrame(rows)


def train_bridge_and_direct_models(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train_dqdv: np.ndarray,
    y_valid_dqdv: np.ndarray,
    pred_train_dqdv: np.ndarray,
    pred_valid_dqdv: np.ndarray,
    y_train_ret: np.ndarray,
    y_valid_ret: np.ndarray,
    last_train_ret: np.ndarray,
    last_valid_ret: np.ndarray,
    linear_train_ret: np.ndarray,
    linear_valid_ret: np.ndarray,
    seed: int,
    x_train_with_history_retention: Optional[np.ndarray] = None,
    x_valid_with_history_retention: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Train oracle bridge, deployable bridge, direct retention models, and evaluate baselines."""

    horizon = int(y_train_ret.shape[1])
    preds = {
        "oracle_bridge": np.zeros_like(y_valid_ret, dtype=np.float32),
        "deployable_bridge": np.zeros_like(y_valid_ret, dtype=np.float32),
        "direct_retention": np.zeros_like(y_valid_ret, dtype=np.float32),
        "persistence": last_valid_ret.astype(np.float32),
        "linear_last10": linear_valid_ret.astype(np.float32),
    }
    train_preds = {
        "oracle_bridge": np.zeros_like(y_train_ret, dtype=np.float32),
        "deployable_bridge": np.zeros_like(y_train_ret, dtype=np.float32),
        "direct_retention": np.zeros_like(y_train_ret, dtype=np.float32),
        "persistence": last_train_ret.astype(np.float32),
        "linear_last10": linear_train_ret.astype(np.float32),
    }
    use_history_retention = x_train_with_history_retention is not None and x_valid_with_history_retention is not None
    if use_history_retention:
        preds["direct_retention_with_history_summary"] = np.zeros_like(y_valid_ret, dtype=np.float32)
        train_preds["direct_retention_with_history_summary"] = np.zeros_like(y_train_ret, dtype=np.float32)
    for step in range(horizon):
        bridge = make_lgbm(seed + 1000 + step)
        bridge.fit(y_train_dqdv[:, step, :], y_train_ret[:, step])
        train_preds["oracle_bridge"][:, step] = bridge.predict(y_train_dqdv[:, step, :]).astype(np.float32)
        preds["oracle_bridge"][:, step] = bridge.predict(y_valid_dqdv[:, step, :]).astype(np.float32)
        train_preds["deployable_bridge"][:, step] = bridge.predict(pred_train_dqdv[:, step, :]).astype(np.float32)
        preds["deployable_bridge"][:, step] = bridge.predict(pred_valid_dqdv[:, step, :]).astype(np.float32)

        direct = make_lgbm(seed + 2000 + step)
        direct.fit(x_train, y_train_ret[:, step])
        train_preds["direct_retention"][:, step] = direct.predict(x_train).astype(np.float32)
        preds["direct_retention"][:, step] = direct.predict(x_valid).astype(np.float32)

        if use_history_retention:
            direct_with_history = make_lgbm(seed + 2500 + step)
            direct_with_history.fit(x_train_with_history_retention, y_train_ret[:, step])
            train_preds["direct_retention_with_history_summary"][:, step] = direct_with_history.predict(
                x_train_with_history_retention
            ).astype(np.float32)
            preds["direct_retention_with_history_summary"][:, step] = direct_with_history.predict(
                x_valid_with_history_retention
            ).astype(np.float32)

    rows: List[Dict[str, object]] = []
    for method, valid_pred in preds.items():
        train_pred = train_preds[method]
        for step in range(horizon):
            for set_type, truth, pred in [
                ("train", y_train_ret[:, step], train_pred[:, step]),
                ("valid", y_valid_ret[:, step], valid_pred[:, step]),
            ]:
                row = regression_metrics(truth, pred)
                row.update(
                    {
                        "stage": "retention_prediction",
                        "method": method,
                        "set_type": set_type,
                        "target": "retention",
                        "horizon": f"H{step + 1}",
                        "horizon_step": int(step + 1),
                    }
                )
                rows.append(row)
        for set_type, truth, pred in [
            ("train", y_train_ret, train_pred),
            ("valid", y_valid_ret, valid_pred),
        ]:
            row = regression_metrics(truth.reshape(-1), pred.reshape(-1))
            row.update(
                {
                    "stage": "retention_prediction",
                    "method": method,
                    "set_type": set_type,
                    "target": "retention",
                    "horizon": "all",
                    "horizon_step": 0,
                }
            )
            rows.append(row)
    return preds, pd.DataFrame(rows)


def build_linear_last10(samples: Sequence[BlockSample]) -> np.ndarray:
    """Predict future retention by extrapolating the last ten history points."""

    rows: List[np.ndarray] = []
    for sample in samples:
        x_hist = sample.history_cycles[-10:].astype(np.float64)
        y_hist = sample.history_retention[-10:].astype(np.float64)
        if x_hist.size < 2 or float(np.ptp(x_hist)) <= 0.0:
            rows.append(np.full(sample.future_retention.shape[0], float(sample.history_retention[-1]), dtype=np.float32))
            continue
        slope, intercept = np.polyfit(x_hist, y_hist, deg=1)
        rows.append((slope * sample.future_cycles.astype(np.float64) + intercept).astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def build_persistence(samples: Sequence[BlockSample]) -> np.ndarray:
    """Predict future retention as the last observed history retention."""

    return np.vstack(
        [
            np.full(sample.future_retention.shape[0], float(sample.history_retention[-1]), dtype=np.float32)
            for sample in samples
        ]
    )


def block_metadata_frame(samples: Sequence[BlockSample]) -> pd.DataFrame:
    """Build a metadata dataframe for block samples."""

    return pd.DataFrame(
        [
            {
                "block_id": sample.block_id,
                "set_type": sample.set_type,
                "policy": sample.policy,
                "cell_code": sample.cell_code,
                "block_index": sample.block_index,
                "input_start_cycle": sample.input_start_cycle,
                "input_end_cycle": sample.input_end_cycle,
                "target_start_cycle": sample.target_start_cycle,
                "target_end_cycle": sample.target_end_cycle,
                "history_len": int(sample.history_x.shape[0]),
                "horizon": int(sample.future_retention.shape[0]),
                "last_history_retention": float(sample.history_retention[-1]),
            }
            for sample in samples
        ]
    )


def dqdv_predictions_long(
    samples: Sequence[BlockSample],
    target_cols: Sequence[str],
    pred_dqdv: np.ndarray,
) -> pd.DataFrame:
    """Build valid dQdV long prediction table."""

    rows: List[Dict[str, object]] = []
    for block_idx, sample in enumerate(samples):
        for step in range(pred_dqdv.shape[1]):
            for target_idx, target in enumerate(target_cols):
                truth = float(sample.future_dqdv[step, target_idx])
                pred = float(pred_dqdv[block_idx, step, target_idx])
                rows.append(
                    {
                        "block_id": sample.block_id,
                        "policy": sample.policy,
                        "cell_code": sample.cell_code,
                        "input_start_cycle": sample.input_start_cycle,
                        "input_end_cycle": sample.input_end_cycle,
                        "target_cycle": int(sample.future_cycles[step]),
                        "horizon_step": int(step + 1),
                        "target": target,
                        "method": "interval_to_dqdv_lightgbm",
                        "true_dqdv": truth,
                        "pred_dqdv": pred,
                        "residual_dqdv": truth - pred,
                    }
                )
    return pd.DataFrame(rows)


def retention_predictions_long(
    samples: Sequence[BlockSample],
    preds: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Build valid retention long prediction table for all methods."""

    rows: List[Dict[str, object]] = []
    for method, pred_matrix in preds.items():
        for block_idx, sample in enumerate(samples):
            for step in range(pred_matrix.shape[1]):
                truth = float(sample.future_retention[step])
                pred = float(pred_matrix[block_idx, step])
                q_ref = float(sample.future_q_ref[step])
                rows.append(
                    {
                        "block_id": sample.block_id,
                        "policy": sample.policy,
                        "cell_code": sample.cell_code,
                        "input_start_cycle": sample.input_start_cycle,
                        "input_end_cycle": sample.input_end_cycle,
                        "target_cycle": int(sample.future_cycles[step]),
                        "horizon_step": int(step + 1),
                        "method": method,
                        "retention_true": truth,
                        "pred_retention": pred,
                        "residual_retention": truth - pred,
                        "q_ref": q_ref,
                        "true_q_discharge": float(sample.future_q_discharge[step]),
                        "pred_q_discharge": pred * q_ref,
                    }
                )
    return pd.DataFrame(rows)


def write_feature_columns(path: Path, columns: Sequence[str]) -> None:
    """Write model history feature columns."""

    pd.DataFrame({"rank": np.arange(1, len(columns) + 1), "feature": list(columns)}).to_csv(
        path,
        index=False,
        encoding=ENCODING,
    )


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


def selected_horizon_metrics(metrics: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
    """Return selected H1/H5/H10/H20/HM and all rows."""

    horizons = {f"H{h}" for h in SELECTED_HORIZONS if h <= int(max_horizon)}
    horizons.add(f"H{int(max_horizon)}")
    horizons.add("all")
    return metrics.loc[metrics["horizon"].isin(horizons)].copy()


def save_retention_metric_plot(metrics: pd.DataFrame, out_path: Path, metric: str) -> None:
    """Save retention metric lines by horizon."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = metrics.loc[
        (metrics["set_type"] == "valid")
        & (metrics["target"] == "retention")
        & (metrics["horizon_step"] > 0)
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for method, part in view.groupby("method", sort=False):
        part = part.sort_values("horizon_step")
        ax.plot(part["horizon_step"], part[metric], marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Future horizon step")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Valid retention {metric.upper()} by horizon")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_dqdv_r2_plot(metrics: pd.DataFrame, out_path: Path) -> None:
    """Save dQdV R2 lines by horizon and target."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = metrics.loc[(metrics["set_type"] == "valid") & (metrics["horizon_step"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for target, part in view.groupby("target", sort=False):
        part = part.sort_values("horizon_step")
        ax.plot(part["horizon_step"], part["r2"], marker="o", linewidth=1.8, label=target)
    ax.set_xlabel("Future horizon step")
    ax.set_ylabel("R2")
    ax.set_title("Valid dQdV R2 by horizon and target")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def metric_lookup(metrics: pd.DataFrame, method: str, horizon: int, metric: str, target: Optional[str] = None) -> float:
    """Look up one valid-set metric for plot titles."""

    view = metrics.loc[(metrics["set_type"] == "valid") & (metrics["horizon"] == f"H{int(horizon)}")].copy()
    if "method" in view.columns:
        view = view.loc[view["method"] == method]
    if target is not None:
        view = view.loc[view["target"] == target]
    if view.empty:
        return float("nan")
    return float(view[metric].iloc[0])


def add_identity_line(ax: object, x_values: pd.Series, y_values: pd.Series) -> None:
    """Add a y=x reference line and equal visual limits to a scatter axis."""

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


def sample_for_plot(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a bounded deterministic sample for readable plots."""

    if len(frame) <= int(max_rows):
        return frame
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(frame), size=int(max_rows), replace=False))
    return frame.iloc[keep].copy()


def save_retention_scatter(pred_long: pd.DataFrame, out_path: Path, max_horizon: int, seed: int) -> None:
    """Save selected-horizon retention true-vs-pred scatter plot with explicit axis semantics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [h for h in [1, 10, 20, int(max_horizon)] if h <= int(max_horizon)]
    methods = ["deployable_bridge", "direct_retention", "linear_last10"]
    fig, axes = plt.subplots(len(horizons), len(methods), figsize=(4.6 * len(methods), 3.9 * len(horizons)), squeeze=False)
    for row_idx, horizon in enumerate(horizons):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            part = pred_long.loc[(pred_long["horizon_step"] == horizon) & (pred_long["method"] == method)].copy()
            part = sample_for_plot(part, 5000, int(seed) + row_idx * 17 + col_idx)
            ax.scatter(part["retention_true"], part["pred_retention"], s=10, alpha=0.35, edgecolors="none")
            add_identity_line(ax, part["retention_true"], part["pred_retention"])
            ax.set_title(f"{method} H{horizon}")
            ax.set_xlabel("X: true retention")
            ax.set_ylabel("Y: predicted retention")
            ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_retention_residual_plot(pred_long: pd.DataFrame, metrics: pd.DataFrame, out_path: Path, max_horizon: int, seed: int) -> None:
    """Save selected-horizon retention residual distributions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [h for h in [1, 10, 20, int(max_horizon)] if h <= int(max_horizon)]
    methods = ["deployable_bridge", "direct_retention", "linear_last10"]
    fig, axes = plt.subplots(len(horizons), len(methods), figsize=(4.6 * len(methods), 3.9 * len(horizons)), squeeze=False)
    for row_idx, horizon in enumerate(horizons):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            part = pred_long.loc[(pred_long["horizon_step"] == horizon) & (pred_long["method"] == method)].copy()
            part = sample_for_plot(part, 5000, int(seed) + row_idx * 17 + col_idx)
            ax.hist(part["residual_retention"].dropna(), bins=55, color="#66AA55", alpha=0.8)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            rmse = metric_lookup(metrics, method=method, horizon=horizon, metric="rmse")
            ax.set_title(f"{method} H{horizon}, RMSE={rmse:.4f}")
            ax.set_xlabel("X: residual = true - predicted")
            ax.set_ylabel("Y: count")
            ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_dqdv_scatter_plot(pred_long: pd.DataFrame, metrics: pd.DataFrame, out_path: Path, max_horizon: int, seed: int) -> None:
    """Save dQdV true-vs-pred scatter plots for selected horizons."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [h for h in [1, 10, 20, int(max_horizon)] if h <= int(max_horizon)]
    targets = [target for target in TARGET_PACKS["compact4"] if target in set(pred_long["target"].astype(str))]
    fig, axes = plt.subplots(len(targets), len(horizons), figsize=(4.3 * len(horizons), 3.5 * len(targets)), squeeze=False)
    for row_idx, target in enumerate(targets):
        for col_idx, horizon in enumerate(horizons):
            ax = axes[row_idx][col_idx]
            part = pred_long.loc[(pred_long["target"] == target) & (pred_long["horizon_step"] == horizon)].copy()
            part = sample_for_plot(part, 5000, int(seed) + row_idx * 19 + col_idx)
            ax.scatter(part["true_dqdv"], part["pred_dqdv"], s=8, alpha=0.30, edgecolors="none")
            add_identity_line(ax, part["true_dqdv"], part["pred_dqdv"])
            r2 = metric_lookup(metrics, method="interval_to_dqdv_lightgbm", target=target, horizon=horizon, metric="r2")
            ax.set_title(f"{target} H{horizon}, R2={r2:.3f}")
            ax.set_xlabel("X: true dQdV feature")
            ax.set_ylabel("Y: predicted dQdV feature")
            ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_dqdv_residual_plot(pred_long: pd.DataFrame, metrics: pd.DataFrame, out_path: Path, max_horizon: int, seed: int) -> None:
    """Save dQdV residual distributions for selected horizons."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [h for h in [1, 10, 20, int(max_horizon)] if h <= int(max_horizon)]
    targets = [target for target in TARGET_PACKS["compact4"] if target in set(pred_long["target"].astype(str))]
    fig, axes = plt.subplots(len(targets), len(horizons), figsize=(4.3 * len(horizons), 3.5 * len(targets)), squeeze=False)
    for row_idx, target in enumerate(targets):
        for col_idx, horizon in enumerate(horizons):
            ax = axes[row_idx][col_idx]
            part = pred_long.loc[(pred_long["target"] == target) & (pred_long["horizon_step"] == horizon)].copy()
            part = sample_for_plot(part, 5000, int(seed) + row_idx * 19 + col_idx)
            ax.hist(part["residual_dqdv"].dropna(), bins=55, color="#4477AA", alpha=0.8)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            rmse = metric_lookup(metrics, method="interval_to_dqdv_lightgbm", target=target, horizon=horizon, metric="rmse")
            ax.set_title(f"{target} H{horizon}, RMSE={rmse:.4f}")
            ax.set_xlabel("X: residual = true - predicted")
            ax.set_ylabel("Y: count")
            ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_dataset_checks(
    args: argparse.Namespace,
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    samples: Sequence[BlockSample],
    stats: Mapping[str, int],
) -> pd.DataFrame:
    """Build dataset check rows."""

    train_keys = set(merged.loc[merged["set_type"] == "train", ["policy", "cell_code"]].drop_duplicates().apply(tuple, axis=1))
    valid_keys = set(merged.loc[merged["set_type"] == "valid", ["policy", "cell_code"]].drop_duplicates().apply(tuple, axis=1))
    train_blocks = sum(1 for sample in samples if sample.set_type == "train")
    valid_blocks = sum(1 for sample in samples if sample.set_type == "valid")
    max_cycle_map = max_cycles_by_key(merged)
    train_rel_ends = [
        relative_stage(sample, max_cycle_map, "input_end")
        for sample in samples
        if sample.set_type == "train"
    ]
    valid_rel_starts = [
        relative_stage(sample, max_cycle_map, "input_start")
        for sample in samples
        if sample.set_type == "valid"
    ]
    train_rel_end_max = float(np.nanmax(train_rel_ends)) if train_rel_ends else float("nan")
    valid_rel_start_min = float(np.nanmin(valid_rel_starts)) if valid_rel_starts else float("nan")
    stage_filter = str(getattr(args, "block_stage_filter", "none"))
    checks = [
        ("split_name", str(args.split_name), 1, "declared train/valid split"),
        ("train_split_path", resolved_path_text(args.train_split_path), 1, ""),
        ("valid_split_path", resolved_path_text(args.valid_split_path), 1, ""),
        ("sample_mode", str(args.sample_mode), int(str(args.sample_mode) == "non_overlapping_blocks"), "1:N -> N+1:N+M fixed block samples"),
        ("history_len", int(args.history_len), int(int(args.history_len) > 0), "positive history length"),
        ("horizon", int(args.horizon), int(int(args.horizon) > 0), "positive forecast horizon"),
        ("block_stride", int(args.block_stride), int(int(args.block_stride) == int(args.history_len) + int(args.horizon)), "expected history_len+horizon"),
        ("block_stage_filter", stage_filter, 1, "none or early_train_late_valid"),
        (
            "train_relative_input_end_max",
            train_rel_end_max,
            int(stage_filter == "none" or train_rel_end_max <= float(args.train_max_relative_input_end) + 1e-9),
            f"threshold={float(args.train_max_relative_input_end):.3f}",
        ),
        (
            "valid_relative_input_start_min",
            valid_rel_start_min,
            int(stage_filter == "none" or valid_rel_start_min >= float(args.valid_min_relative_input_start) - 1e-9),
            f"threshold={float(args.valid_min_relative_input_start):.3f}",
        ),
        ("feature_count", int(len(feature_cols)), int(len(feature_cols) == int(stats.get("feature_count", 0))), ""),
        ("include_history_retention_summary", int(bool(args.include_history_retention_summary)), 1, "optional LightGBM history-retention route flag"),
        (
            "history_retention_summary_feature_count",
            int(len(SUMMARY_STATS)) if bool(args.include_history_retention_summary) else 0,
            1,
            "last,mean,std,min,max,delta,slope when enabled",
        ),
        ("forbidden_input_columns_present", int(stats.get("forbidden_input_columns_present", 0)), int(stats.get("forbidden_input_columns_present", 0) == 0), ""),
        ("target_dim", int(len(target_cols)), int(len(target_cols) == 4 if args.target_pack == "compact4" else len(target_cols) > 0), ",".join(target_cols)),
        ("train_policy_cell_count", int(len(train_keys)), int(len(train_keys) > 0), ""),
        ("valid_policy_cell_count", int(len(valid_keys)), int(len(valid_keys) > 0), ""),
        ("split_overlap_zero", int(len(train_keys.intersection(valid_keys)) == 0), int(len(train_keys.intersection(valid_keys)) == 0), f"overlap_count={len(train_keys.intersection(valid_keys))}"),
        ("train_block_count", int(train_blocks), int(train_blocks > 0), ""),
        ("valid_block_count", int(valid_blocks), int(valid_blocks > 0), ""),
    ]
    return pd.DataFrame(checks, columns=["check_item", "value", "pass_flag", "details"])


def build_report(
    args: argparse.Namespace,
    checks: pd.DataFrame,
    dqdv_metrics: pd.DataFrame,
    retention_metrics: pd.DataFrame,
    out_dir: Path,
) -> str:
    """Build a Chinese Markdown report for the multistep block experiment."""

    selected_ret = selected_horizon_metrics(
        retention_metrics.loc[
            (retention_metrics["set_type"] == "valid") & (retention_metrics["target"] == "retention")
        ],
        int(args.horizon),
    ).sort_values(["method", "horizon_step"])
    selected_dqdv = selected_horizon_metrics(
        dqdv_metrics.loc[dqdv_metrics["set_type"] == "valid"],
        int(args.horizon),
    ).sort_values(["target", "horizon_step"])

    def image(name: str) -> str:
        """Return an absolute image Markdown link."""

        return f"![{Path(name).stem}]({(out_dir / name).resolve().as_posix()})"

    all_ret = selected_ret.loc[selected_ret["horizon"] == "all"].copy()
    deploy_all = all_ret.loc[all_ret["method"] == "deployable_bridge", "r2"]
    direct_all = all_ret.loc[all_ret["method"] == "direct_retention", "r2"]
    oracle_all = all_ret.loc[all_ret["method"] == "oracle_bridge", "r2"]
    persistence_all = all_ret.loc[all_ret["method"] == "persistence", "r2"]
    linear_all = all_ret.loc[all_ret["method"] == "linear_last10", "r2"]

    def metric_value(method: str, horizon: str, metric: str) -> float:
        """Return one valid retention metric value."""

        rows = selected_ret.loc[(selected_ret["method"] == method) & (selected_ret["horizon"] == horizon)]
        if rows.empty:
            return float("nan")
        return float(rows[metric].iloc[0])
    lines = [
        "# recommended55 + compact4 多步未来预测短闭环报告",
        "",
        "## 1. 摘要",
        f"- split_name: `{args.split_name}`",
        f"- train_split_path: `{resolved_path_text(args.train_split_path)}`",
        f"- valid_split_path: `{resolved_path_text(args.valid_split_path)}`",
        f"- history_len: `{int(args.history_len)}`",
        f"- horizon: `{int(args.horizon)}`",
        f"- block_stride: `{int(args.block_stride)}`",
        f"- block_stage_filter: `{args.block_stage_filter}`",
        f"- feature_pack: `{args.feature_pack}`",
        f"- target_pack: `{args.target_pack}`",
        f"- history_representation: `{args.history_representation}`",
    ]
    if not deploy_all.empty and not direct_all.empty and not oracle_all.empty:
        lines.extend(
            [
                f"- all-horizon oracle bridge R2: `{float(oracle_all.iloc[0]):.6f}`",
                f"- all-horizon deployable bridge R2: `{float(deploy_all.iloc[0]):.6f}`",
                f"- all-horizon direct retention R2: `{float(direct_all.iloc[0]):.6f}`",
            ]
        )
    if not linear_all.empty:
        lines.append(f"- all-horizon linear_last10 baseline R2: `{float(linear_all.iloc[0]):.6f}`")
    lines.extend(
        [
            "",
            "## 2. 术语与代称解释",
            "- `recommended55`：从相关性分析中筛选出的 55 个工况统计特征，不包含 `cycles`、`policy` 或 policy 三元参数。",
            "- `compact4`：4 个 dQdV 中介特征，包含 `main_peak_area`、`main_peak_height_dqdv`、`main_peak_voltage_v`、`main_peak_skewness`。",
            "- `dQdV`：放电容量-电压曲线的微分特征，用于描述电芯退化相关的峰形状态。",
            "- `retention`：容量保持率，定义为当前 `q_discharge / q_ref`，其中 `q_ref` 是同一电芯前若干有效循环的参考容量。",
            "- `history_len` 或 `N`：模型可见的历史 cycle 数，本报告为 100。",
            "- `horizon` 或 `M`：要预测的未来 cycle 数，本报告为 50。",
            "- `block_stride`：相邻样本块起点间隔，本报告为 `N+M=150`，用于构造非重叠未来预测样本。",
            "- `H1/H10/H20/H50`：未来第 1/10/20/50 个预测步；`all` 表示把 H1 到 H50 全部预测点合并计算指标。",
            f"- `train/valid`：训练集/验证集，来自 `{args.split_name}`，按 `policy + cell_code` 电芯组合划分，不按单个 cycle 随机混切。",
            "- `LightGBM`：本报告使用的表格树模型，用历史工况摘要预测未来 dQdV 或 retention。",
            "- `summary`：历史 100 个 cycle 的特征压缩方式，包括 last、mean、std、min、max、delta、slope。",
            "- `baseline`：不经过本任务中介模型的对照方法，用于判断复杂链路是否真的有增益。",
            "- `oracle_bridge`：使用真实未来 dQdV 预测未来 retention，是中介表征的上限参考，部署时不可直接获得。",
            "- `deployable_bridge`：使用工况预测出来的未来 dQdV 再预测未来 retention，是 dQdV 中介链路的可部署版本。",
            "- `direct_retention`：直接用历史工况摘要预测未来 retention，不经过 dQdV 中介。",
            "- `persistence`：朴素基线，假设未来 retention 等于历史最后一个 retention。",
            "- `linear_last10`：朴素趋势基线，用历史最后 10 个 retention 点线性外推未来 retention。",
            "- `R2/RMSE/MAE/MSE`：R2 越高越好；RMSE、MAE、MSE 越低越好。",
            "- `residual`：残差，统一定义为 `true - predicted`；残差接近 0 说明预测误差小。",
            "",
            "## 3. 数据检查",
            markdown_table(checks, ["check_item", "value", "pass_flag", "details"]),
            "",
            "## 4. dQdV 多步预测指标",
            markdown_table(selected_dqdv, ["method", "target", "horizon", "n_rows", "r2", "rmse", "mae", "mse"]),
            "",
            image("dqdv_r2_by_horizon_target.png"),
            "",
            "**图 dQdV R2 by horizon 说明**：X 轴是未来预测步 `horizon_step`，Y 轴是验证集 R2；每条线代表一个 dQdV 特征。关键结论：`main_peak_area` 和 `main_peak_height_dqdv` 最稳定，`main_peak_skewness` 最弱但仍有可预测性。",
            "",
            image("valid_dqdv_scatter_selected_horizons.png"),
            "",
            "**图 dQdV scatter 说明**：X 轴是真实 dQdV 特征值，Y 轴是预测 dQdV 特征值，黑色虚线是理想预测 `Y=X`。点越贴近虚线，预测越准。关键结论：面积和峰高散点更贴近虚线，skewness 离散更明显。",
            "",
            image("valid_dqdv_residual_selected_horizons.png"),
            "",
            "**图 dQdV residual 说明**：X 轴是残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。分布越集中在 0 附近，误差越小。关键结论：H50 的残差分布比短 horizon 更宽，说明远期 dQdV 预测不确定性增大。",
            "",
            "## 5. retention 多步预测链路指标",
            markdown_table(selected_ret, ["method", "horizon", "n_rows", "r2", "rmse", "mae", "mse"]),
            "",
            image("retention_r2_by_horizon.png"),
            "",
            "**图 retention R2 by horizon 说明**：X 轴是未来预测步，Y 轴是 retention 的验证集 R2；每条线代表一种预测路径或基线。关键结论：`linear_last10` 全程最强，`direct_retention` 明显强于 `deployable_bridge`。",
            "",
            image("retention_rmse_by_horizon.png"),
            "",
            "**图 retention RMSE by horizon 说明**：X 轴是未来预测步，Y 轴是 RMSE；越低表示误差越小。关键结论：`linear_last10` 误差最低，说明容量保持率在 50 cycle 内非常平滑，简单趋势外推已经很强。",
            "",
            image("valid_retention_scatter_selected_horizons.png"),
            "",
            "**图 retention scatter 说明**：X 轴是真实 retention，Y 轴是预测 retention，黑色虚线是理想预测 `Y=X`。关键结论：`linear_last10` 最贴近虚线；`deployable_bridge` 的散点更分散，说明 dQdV 中介链路传递到 retention 后仍有误差损失。",
            "",
            image("valid_retention_residual_selected_horizons.png"),
            "",
            "**图 retention residual 说明**：X 轴是 retention 残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。关键结论：`linear_last10` 残差最集中，`deployable_bridge` 残差更宽，尤其在 H50 仍落后于 direct 和趋势外推。",
            "",
            "## 6. 结论",
            "- 本实验采用非重叠 block，重点评估未来预测而不是相邻滑窗拟合。",
        ]
    )
    if not deploy_all.empty and not direct_all.empty and not oracle_all.empty:
        oracle_val = float(oracle_all.iloc[0])
        deploy_val = float(deploy_all.iloc[0])
        direct_val = float(direct_all.iloc[0])
        lines.extend(
            [
                f"- 预测 dQdV 传递到 retention 后的 all-horizon R2 损失为 `{oracle_val - deploy_val:.6f}`。",
                f"- direct retention 相比 deployable bridge 的 all-horizon R2 优势为 `{direct_val - deploy_val:.6f}`。",
                f"- H50 上 deployable bridge R2 为 `{metric_value('deployable_bridge', f'H{int(args.horizon)}', 'r2'):.6f}`，direct retention R2 为 `{metric_value('direct_retention', f'H{int(args.horizon)}', 'r2'):.6f}`。",
            ]
        )
    if not persistence_all.empty and not linear_all.empty:
        lines.extend(
            [
                f"- persistence all-horizon R2 为 `{float(persistence_all.iloc[0]):.6f}`，linear_last10 all-horizon R2 为 `{float(linear_all.iloc[0]):.6f}`；朴素外推基线非常强，说明 retention 在 50 cycle 预测窗口内非常平滑。",
                "- 当前不建议直接进入 LSTM/TCN/Transformer 长训练。更低成本的下一步是预测相对 `linear_last10` 的 residual/delta，或增加 forecast gap，再判断深度时序模型是否真正提供增益。",
                "- compact4 dQdV 仍有解释价值，但在当前多步未来预测口径下，不应作为主预测路径替代 direct retention 或朴素趋势外推。",
            ]
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, object]:
    """Run the complete block-based multistep validation pipeline."""

    if int(args.block_stride) != int(args.history_len) + int(args.horizon):
        print("Warning: block_stride is not history_len+horizon; samples may overlap.", flush=True)
    set_seed(int(args.random_seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_cols = TARGET_PACKS[str(args.target_pack)]
    merged, feature_cols, stats = build_cycle_table(args, target_cols)
    samples = build_block_samples(
        merged=merged,
        feature_cols=feature_cols,
        target_cols=target_cols,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        block_stride=int(args.block_stride),
    )
    samples = filter_blocks_by_stage(
        samples,
        max_cycles_by_key(merged),
        str(args.block_stage_filter),
        float(args.train_max_relative_input_end),
        float(args.valid_min_relative_input_start),
    )
    samples = downsample_blocks(samples, int(args.max_train_blocks), int(args.max_valid_blocks), int(args.random_seed))
    checks = build_dataset_checks(args, merged, feature_cols, target_cols, samples, stats)
    checks.to_csv(args.output_dir / "dataset_checks.csv", index=False, encoding=ENCODING)
    if int(checks["pass_flag"].min()) != 1:
        raise RuntimeError(f"Dataset checks failed. See {args.output_dir / 'dataset_checks.csv'}")

    train_samples = [sample for sample in samples if sample.set_type == "train"]
    valid_samples = [sample for sample in samples if sample.set_type == "valid"]
    x_train, history_columns = build_history_matrix(train_samples, feature_cols, str(args.history_representation))
    x_valid, _history_columns_valid = build_history_matrix(valid_samples, feature_cols, str(args.history_representation))
    history_retention_columns: List[str] = []
    x_train_with_history_retention: Optional[np.ndarray] = None
    x_valid_with_history_retention: Optional[np.ndarray] = None
    if bool(args.include_history_retention_summary):
        train_history_retention, history_retention_columns = build_history_retention_matrix(train_samples)
        valid_history_retention, _valid_history_retention_columns = build_history_retention_matrix(valid_samples)
        x_train_with_history_retention = np.hstack([x_train, train_history_retention]).astype(np.float32)
        x_valid_with_history_retention = np.hstack([x_valid, valid_history_retention]).astype(np.float32)
    y_train_dqdv, y_train_ret, _qref_train, _q_train, _future_cycles_train = future_arrays(train_samples)
    y_valid_dqdv, y_valid_ret, _qref_valid, _q_valid, _future_cycles_valid = future_arrays(valid_samples)
    train_persistence = build_persistence(train_samples)
    valid_persistence = build_persistence(valid_samples)
    train_linear = build_linear_last10(train_samples)
    valid_linear = build_linear_last10(valid_samples)

    print("Training multistep dQdV models...", flush=True)
    pred_train_dqdv, pred_valid_dqdv, dqdv_metrics = train_dqdv_models(
        x_train,
        x_valid,
        y_train_dqdv,
        y_valid_dqdv,
        target_cols,
        int(args.random_seed),
    )
    print("Training retention bridge/direct models...", flush=True)
    retention_preds, retention_metrics = train_bridge_and_direct_models(
        x_train=x_train,
        x_valid=x_valid,
        y_train_dqdv=y_train_dqdv,
        y_valid_dqdv=y_valid_dqdv,
        pred_train_dqdv=pred_train_dqdv,
        pred_valid_dqdv=pred_valid_dqdv,
        y_train_ret=y_train_ret,
        y_valid_ret=y_valid_ret,
        last_train_ret=train_persistence,
        last_valid_ret=valid_persistence,
        linear_train_ret=train_linear,
        linear_valid_ret=valid_linear,
        seed=int(args.random_seed),
        x_train_with_history_retention=x_train_with_history_retention,
        x_valid_with_history_retention=x_valid_with_history_retention,
    )

    block_metadata_frame(samples).to_csv(args.output_dir / "block_samples.csv", index=False, encoding=ENCODING)
    write_feature_columns(args.output_dir / "history_feature_columns.csv", history_columns)
    if history_retention_columns:
        write_feature_columns(args.output_dir / "history_retention_feature_columns.csv", history_retention_columns)
    dqdv_metrics.to_csv(args.output_dir / "dqdv_multistep_metrics.csv", index=False, encoding=ENCODING)
    retention_metrics.to_csv(args.output_dir / "retention_multistep_metrics.csv", index=False, encoding=ENCODING)
    dqdv_long = dqdv_predictions_long(valid_samples, target_cols, pred_valid_dqdv)
    dqdv_long.to_csv(
        args.output_dir / "valid_dqdv_predictions_long.csv",
        index=False,
        encoding=ENCODING,
    )
    ret_long = retention_predictions_long(valid_samples, retention_preds)
    ret_long.to_csv(args.output_dir / "valid_retention_predictions_long.csv", index=False, encoding=ENCODING)

    save_retention_metric_plot(retention_metrics, args.output_dir / "retention_r2_by_horizon.png", "r2")
    save_retention_metric_plot(retention_metrics, args.output_dir / "retention_rmse_by_horizon.png", "rmse")
    save_dqdv_r2_plot(dqdv_metrics, args.output_dir / "dqdv_r2_by_horizon_target.png")
    save_dqdv_scatter_plot(
        dqdv_long,
        dqdv_metrics,
        args.output_dir / "valid_dqdv_scatter_selected_horizons.png",
        int(args.horizon),
        int(args.random_seed),
    )
    save_dqdv_residual_plot(
        dqdv_long,
        dqdv_metrics,
        args.output_dir / "valid_dqdv_residual_selected_horizons.png",
        int(args.horizon),
        int(args.random_seed),
    )
    save_retention_scatter(ret_long, args.output_dir / "valid_retention_scatter_selected_horizons.png", int(args.horizon), int(args.random_seed))
    save_retention_residual_plot(
        ret_long,
        retention_metrics,
        args.output_dir / "valid_retention_residual_selected_horizons.png",
        int(args.horizon),
        int(args.random_seed),
    )

    run_config = {
        "split_name": str(args.split_name),
        "train_split_path": resolved_path_text(args.train_split_path),
        "valid_split_path": resolved_path_text(args.valid_split_path),
        "history_len": int(args.history_len),
        "horizon": int(args.horizon),
        "block_stride": int(args.block_stride),
        "sample_mode": str(args.sample_mode),
        "block_stage_filter": str(args.block_stage_filter),
        "train_max_relative_input_end": float(args.train_max_relative_input_end),
        "valid_min_relative_input_start": float(args.valid_min_relative_input_start),
        "feature_pack": str(args.feature_pack),
        "feature_count": int(len(feature_cols)),
        "target_pack": str(args.target_pack),
        "target_cols": list(target_cols),
        "history_representation": str(args.history_representation),
        "include_history_retention_summary": bool(args.include_history_retention_summary),
        "history_feature_count": int(len(history_columns)),
        "history_retention_summary_feature_count": int(len(history_retention_columns)),
        "model_family": str(args.model_family),
        "train_blocks": int(len(train_samples)),
        "valid_blocks": int(len(valid_samples)),
        "input_feature_columns": list(feature_cols),
        "history_feature_columns": list(history_columns),
        "history_retention_summary_columns": list(history_retention_columns),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(args, checks, dqdv_metrics, retention_metrics, args.output_dir)
    (args.output_dir / "multistep_interval_to_dqdv_retention_blocks_report.md").write_text(report, encoding="utf-8")
    print(f"Saved outputs to: {args.output_dir}", flush=True)
    return run_config


def main() -> None:
    """CLI entrypoint."""

    run(parse_args())


if __name__ == "__main__":
    main()
