from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
sys.path.append(str(REPO_ROOT))
ENCODING = "utf-8-sig"
FIRST_OCCURRENCE_RANGE_COUNT = 1
N_CROSS_BINS = 60

import scripts.train_lstm_charge_delta_ah as base_lstm
import scripts.train_lstm_dqdv_retention as dqdv_retention
import scripts.train_rf_charge_aging_q_discharge as charge_rf

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "rf_charge_crossbin_multistep_retention"
DEFAULT_TIMESERIES_PATH = REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv"
DEFAULT_DISCHARGE_PATH = REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv"
DEFAULT_LIFE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_TRAIN_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv"
DEFAULT_VALID_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv"
MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"


@dataclass
class FeaturePacks:
    charge_cum_cols: List[str]
    charge_inc_cols: List[str]
    discharge_inc_delta_cols: List[str]
    discharge_cum_delta_cols: List[str]
    discharge_stat7_cols: List[str]

    @property
    def raw_feature_cols(self) -> List[str]:
        return (
            self.charge_cum_cols
            + self.charge_inc_cols
            + self.discharge_inc_delta_cols
            + self.discharge_cum_delta_cols
            + self.discharge_stat7_cols
        )


@dataclass
class WindowPack:
    x: np.ndarray
    y: np.ndarray
    group_keys: np.ndarray
    input_cycles: np.ndarray
    target_cycles: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF multistep retention using 5 operational feature groups (no policy params)."
    )
    parser.add_argument("--timeseries-path", type=Path, default=DEFAULT_TIMESERIES_PATH)
    parser.add_argument("--discharge-path", type=Path, default=DEFAULT_DISCHARGE_PATH)
    parser.add_argument("--life-path", type=Path, default=DEFAULT_LIFE_PATH)
    parser.add_argument("--train-split-path", type=Path, default=DEFAULT_TRAIN_SPLIT_PATH)
    parser.add_argument("--valid-split-path", type=Path, default=DEFAULT_VALID_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--n-history", type=int, default=100)
    parser.add_argument("--horizon-steps", type=int, default=50)

    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--min-samples-split", type=int, default=6)
    parser.add_argument("--max-features", type=float, default=0.35)
    parser.add_argument("--bootstrap", action="store_true", default=True)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--criterion", type=str, default="squared_error")

    parser.add_argument("--random-seed", type=int, default=20260508)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    return parser.parse_args()


def ensure_matplotlib_config() -> List[str]:
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams  # noqa: WPS433

    candidates = ["Noto Sans CJK SC", "DejaVu Sans"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = [font for font in candidates if font in installed] or ["DejaVu Sans"]
    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    return selected


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_range_start(range_label: str) -> float:
    values = re.findall(r"-?\d+(?:\.\d+)?", str(range_label))
    if not values:
        return float("nan")
    return float(values[0])


def sanitize_range_label(range_label: str) -> str:
    values = re.findall(r"-?\d+(?:\.\d+)?", str(range_label))
    if len(values) >= 2:
        left = values[0].replace(".", "p")
        right = values[1].replace(".", "p")
        return f"{left}_to_{right}"
    return re.sub(r"[^0-9A-Za-z]+", "_", str(range_label)).strip("_").lower()


def load_charge_feature_table(timeseries_path: Path) -> Tuple[pd.DataFrame, Dict[str, int], List[str], List[str]]:
    feat, stats, cum_cols, inc_cols, _ = charge_rf.load_feature_table(timeseries_path)
    keep_cols = ["policy", "cell_code", "cycles", *cum_cols, *inc_cols]
    out = feat[keep_cols].copy()
    out["policy"] = out["policy"].astype(str)
    out["cell_code"] = out["cell_code"].astype(str)
    out["cycles"] = pd.to_numeric(out["cycles"], errors="coerce").astype(int)
    return out, stats, cum_cols, inc_cols


def load_discharge_feature_table(discharge_path: Path) -> Tuple[pd.DataFrame, Dict[str, int], List[str], List[str], List[str]]:
    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "range",
        "delta_ah",
        "charge_duration_s",
        "avg_temper",
        "range_count",
    ]
    df = pd.read_csv(discharge_path, encoding=ENCODING, usecols=usecols)
    df["policy"] = df["policy"].astype(str)
    df["cell_code"] = df["cell_code"].astype(str)
    for col in ["cycles", "delta_ah", "charge_duration_s", "avg_temper", "range_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]).copy()
    df["cycles"] = df["cycles"].astype(int)
    df["range"] = df["range"].astype(str)
    df["range_count"] = df["range_count"].astype(int)
    df = df.loc[df["range_count"] == FIRST_OCCURRENCE_RANGE_COUNT].copy()
    if df.empty:
        raise RuntimeError("No valid discharge rows after first-occurrence filtering.")

    range_order = sorted(df["range"].dropna().unique().tolist(), key=parse_range_start, reverse=True)
    if len(range_order) != 16:
        raise RuntimeError(f"Expected 16 discharge ranges, but got {len(range_order)}.")
    suffix_map = {rng: sanitize_range_label(str(rng)) for rng in range_order}

    grouped = (
        df.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(
            discharge_delta_ah=("delta_ah", "sum"),
            discharge_duration_s=("charge_duration_s", "sum"),
            discharge_avg_temper=("avg_temper", "mean"),
            discharge_range_count=("range_count", "sum"),
        )
    )

    idx_cols = ["policy", "cell_code", "cycles"]
    wide = grouped[idx_cols].drop_duplicates().copy()

    # Incremental discharge delta_ah (16)
    pivot_delta = (
        grouped.pivot_table(
            index=idx_cols,
            columns="range",
            values="discharge_delta_ah",
            aggfunc="sum",
            fill_value=np.nan,
        )
        .reindex(columns=range_order)
        .reset_index()
    )
    inc_delta_cols = [f"discharge_inc_delta_ah_{suffix_map[rng]}" for rng in range_order]
    pivot_delta = pivot_delta.rename(columns={rng: col for rng, col in zip(range_order, inc_delta_cols)})
    pivot_delta[inc_delta_cols] = pivot_delta[inc_delta_cols].fillna(0.0)
    wide = wide.merge(pivot_delta[[*idx_cols, *inc_delta_cols]], on=idx_cols, how="left")

    # Duration + temp support stats
    pivot_dur = (
        grouped.pivot_table(
            index=idx_cols,
            columns="range",
            values="discharge_duration_s",
            aggfunc="sum",
            fill_value=np.nan,
        )
        .reindex(columns=range_order)
        .reset_index()
    )
    inc_dur_cols = [f"discharge_inc_duration_s_{suffix_map[rng]}" for rng in range_order]
    pivot_dur = pivot_dur.rename(columns={rng: col for rng, col in zip(range_order, inc_dur_cols)})
    pivot_dur[inc_dur_cols] = pivot_dur[inc_dur_cols].fillna(0.0)
    wide = wide.merge(pivot_dur[[*idx_cols, *inc_dur_cols]], on=idx_cols, how="left")

    pivot_temp = (
        grouped.pivot_table(
            index=idx_cols,
            columns="range",
            values="discharge_avg_temper",
            aggfunc="mean",
            fill_value=np.nan,
        )
        .reindex(columns=range_order)
        .reset_index()
    )
    temp_cols = [f"discharge_avg_temp_{suffix_map[rng]}" for rng in range_order]
    mask_cols = [f"discharge_mask_{suffix_map[rng]}" for rng in range_order]
    pivot_temp = pivot_temp.rename(columns={rng: col for rng, col in zip(range_order, temp_cols)})
    for col, mask_col in zip(temp_cols, mask_cols):
        pivot_temp[mask_col] = (~pivot_temp[col].isna()).astype(np.float32)
    pivot_temp[temp_cols] = pivot_temp[temp_cols].fillna(0.0)
    wide = wide.merge(pivot_temp[[*idx_cols, *temp_cols, *mask_cols]], on=idx_cols, how="left")

    # Cumulative discharge delta_ah (16)
    cum_delta_cols = [col.replace("discharge_inc_", "discharge_cum_") for col in inc_delta_cols]
    wide = wide.sort_values(idx_cols, kind="mergesort").copy()
    wide[cum_delta_cols] = wide.groupby(["policy", "cell_code"], sort=False)[inc_delta_cols].cumsum()

    # 7 discharge summary stats
    stat7_cols = [
        "discharge_cycle_total_delta_ah",
        "discharge_cycle_total_duration_s",
        "discharge_cycle_active_range_count",
        "discharge_cycle_avg_temp_mean",
        "discharge_cum_total_delta_ah",
        "discharge_cum_total_duration_s",
        "discharge_cum_active_range_count",
    ]
    wide[stat7_cols[0]] = wide[inc_delta_cols].sum(axis=1)
    wide[stat7_cols[1]] = wide[inc_dur_cols].sum(axis=1)
    wide[stat7_cols[2]] = (wide[inc_delta_cols] > 0.0).sum(axis=1)
    temp_sum = (wide[temp_cols] * wide[mask_cols].to_numpy(dtype=float)).sum(axis=1)
    temp_count = wide[mask_cols].sum(axis=1).replace(0, np.nan)
    wide[stat7_cols[3]] = (temp_sum / temp_count).fillna(0.0)
    wide[stat7_cols[4]] = wide[cum_delta_cols].sum(axis=1)
    wide[stat7_cols[5]] = wide.groupby(["policy", "cell_code"], sort=False)[stat7_cols[1]].cumsum()
    wide[stat7_cols[6]] = (wide[cum_delta_cols] > 0.0).sum(axis=1)

    keep_cols = [*idx_cols, *inc_delta_cols, *cum_delta_cols, *stat7_cols]
    wide[keep_cols[3:]] = wide[keep_cols[3:]].fillna(0.0)
    stats = {
        "discharge_interval_rows_after_filter": int(len(df)),
        "discharge_cycle_rows": int(len(wide)),
        "discharge_range_count": int(len(range_order)),
        "first_occurrence_only": 1,
    }
    return wide[keep_cols].copy(), stats, inc_delta_cols, cum_delta_cols, stat7_cols


def merge_cycle_dataset(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    label_df: pd.DataFrame,
    split_df: pd.DataFrame,
    raw_feature_cols: Sequence[str],
) -> pd.DataFrame:
    merged = (
        label_df.merge(charge_df, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(discharge_df, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(split_df, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    )
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    for col in [*raw_feature_cols, "retention", "q_discharge", "q_ref"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=[*raw_feature_cols, "retention", "q_discharge", "q_ref"]).copy()
    return merged


def aggregate_feature_names(raw_feature_cols: Sequence[str]) -> List[str]:
    last_cols = [f"{c}__last" for c in raw_feature_cols]
    mean_cols = [f"{c}__mean" for c in raw_feature_cols]
    std_cols = [f"{c}__std" for c in raw_feature_cols]
    slope_cols = [f"{c}__slope" for c in raw_feature_cols]
    return [*last_cols, *mean_cols, *std_cols, *slope_cols]


def compute_group_window_pack(
    part: pd.DataFrame,
    raw_feature_cols: Sequence[str],
    n_history: int,
    horizon_steps: int,
) -> Optional[WindowPack]:
    cycles = part["cycles"].to_numpy(dtype=np.int64)
    x_raw = part[list(raw_feature_cols)].to_numpy(dtype=np.float32)
    y_ret = part["retention"].to_numpy(dtype=np.float32)
    if x_raw.shape[0] < (n_history + horizon_steps):
        return None

    total_windows = int(x_raw.shape[0] - (n_history + horizon_steps) + 1)
    if total_windows <= 0:
        return None

    # consecutive coverage check: segment [s, s + n_history + horizon_steps - 1]
    diffs = np.diff(cycles)
    bad = (diffs != 1).astype(np.int32)
    csum_bad = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(bad, dtype=np.int64)])
    starts = np.arange(total_windows, dtype=np.int64)
    bad_counts = csum_bad[starts + n_history + horizon_steps - 1] - csum_bad[starts]
    valid_mask = bad_counts == 0
    if not bool(np.any(valid_mask)):
        return None

    hist_view = sliding_window_view(x_raw, window_shape=n_history, axis=0)[:total_windows]
    # NumPy may return (n_win, n_feat, n_hist); normalize to (n_win, n_hist, n_feat).
    if hist_view.ndim != 3:
        raise RuntimeError(f"Unexpected history view ndim: {hist_view.ndim}")
    if hist_view.shape[1] != n_history and hist_view.shape[2] == n_history:
        hist_view = np.swapaxes(hist_view, 1, 2)
    if hist_view.shape[1] != n_history:
        raise RuntimeError(f"Unexpected history view shape: {hist_view.shape}, n_history={n_history}")
    last = hist_view[:, -1, :].astype(np.float32)
    mean = hist_view.mean(axis=1, dtype=np.float64).astype(np.float32)
    std = hist_view.std(axis=1, ddof=0, dtype=np.float64).astype(np.float32)

    x_axis = np.arange(n_history, dtype=np.float64)
    x_sum = float(x_axis.sum())
    denom = float(n_history * np.sum(x_axis * x_axis) - x_sum * x_sum)
    weighted_sum = np.tensordot(hist_view.astype(np.float64), x_axis, axes=([1], [0]))
    y_sum = hist_view.astype(np.float64).sum(axis=1)
    slope = ((n_history * weighted_sum - x_sum * y_sum) / denom).astype(np.float32)

    x_aggr_all = np.concatenate([last, mean, std, slope], axis=1)

    target_view = sliding_window_view(y_ret, window_shape=horizon_steps, axis=0)
    target_cycles_view = sliding_window_view(cycles, window_shape=horizon_steps, axis=0)
    y_all = target_view[n_history : n_history + total_windows].astype(np.float32)
    target_cycles_all = target_cycles_view[n_history : n_history + total_windows].astype(np.int64)
    input_cycles_all = cycles[n_history - 1 : n_history - 1 + total_windows].astype(np.int64)

    x_out = x_aggr_all[valid_mask]
    y_out = y_all[valid_mask]
    target_cycles_out = target_cycles_all[valid_mask]
    input_cycles_out = input_cycles_all[valid_mask]

    policy = str(part.iloc[0]["policy"])
    cell_code = str(part.iloc[0]["cell_code"])
    group_key = f"{policy}||{cell_code}"
    group_keys_out = np.full(len(input_cycles_out), group_key, dtype=object)
    return WindowPack(
        x=x_out,
        y=y_out,
        group_keys=group_keys_out,
        input_cycles=input_cycles_out,
        target_cycles=target_cycles_out,
    )


def concat_window_packs(packs: Sequence[WindowPack], n_features: int, horizon_steps: int) -> WindowPack:
    if not packs:
        return WindowPack(
            x=np.empty((0, n_features), dtype=np.float32),
            y=np.empty((0, horizon_steps), dtype=np.float32),
            group_keys=np.empty((0,), dtype=object),
            input_cycles=np.empty((0,), dtype=np.int64),
            target_cycles=np.empty((0, horizon_steps), dtype=np.int64),
        )
    return WindowPack(
        x=np.concatenate([p.x for p in packs], axis=0),
        y=np.concatenate([p.y for p in packs], axis=0),
        group_keys=np.concatenate([p.group_keys for p in packs], axis=0),
        input_cycles=np.concatenate([p.input_cycles for p in packs], axis=0),
        target_cycles=np.concatenate([p.target_cycles for p in packs], axis=0),
    )


def sample_window_pack(pack: WindowPack, max_rows: int, seed: int) -> WindowPack:
    if max_rows <= 0 or len(pack.x) <= max_rows:
        return pack
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(pack.x), size=int(max_rows), replace=False))
    return WindowPack(
        x=pack.x[keep],
        y=pack.y[keep],
        group_keys=pack.group_keys[keep],
        input_cycles=pack.input_cycles[keep],
        target_cycles=pack.target_cycles[keep],
    )


def build_train_valid_windows(
    merged: pd.DataFrame,
    raw_feature_cols: Sequence[str],
    n_history: int,
    horizon_steps: int,
    max_train_windows: int,
    max_valid_windows: int,
    seed: int,
) -> Tuple[WindowPack, WindowPack, Dict[str, int]]:
    train_packs: List[WindowPack] = []
    valid_packs: List[WindowPack] = []
    counters = {
        "groups_total": 0,
        "groups_train": 0,
        "groups_valid": 0,
        "groups_with_windows_train": 0,
        "groups_with_windows_valid": 0,
    }

    for (policy, cell_code), part in merged.groupby(["policy", "cell_code"], sort=False):
        _ = policy, cell_code
        counters["groups_total"] += 1
        set_types = part["set_type"].dropna().unique().tolist()
        if len(set_types) != 1:
            raise RuntimeError(f"Split leakage detected for {(policy, cell_code)}.")
        set_type = str(set_types[0])
        if set_type == "train":
            counters["groups_train"] += 1
        elif set_type == "valid":
            counters["groups_valid"] += 1
        else:
            raise RuntimeError(f"Unknown set_type: {set_type}")

        part = part.sort_values("cycles", kind="mergesort").copy()
        wp = compute_group_window_pack(
            part=part,
            raw_feature_cols=raw_feature_cols,
            n_history=n_history,
            horizon_steps=horizon_steps,
        )
        if wp is None or len(wp.x) == 0:
            continue
        if set_type == "train":
            counters["groups_with_windows_train"] += 1
            train_packs.append(wp)
        else:
            counters["groups_with_windows_valid"] += 1
            valid_packs.append(wp)

    n_aggr_features = int(len(raw_feature_cols) * 4)
    train_pack = concat_window_packs(train_packs, n_features=n_aggr_features, horizon_steps=horizon_steps)
    valid_pack = concat_window_packs(valid_packs, n_features=n_aggr_features, horizon_steps=horizon_steps)

    train_pack = sample_window_pack(train_pack, max_rows=max_train_windows, seed=seed + 11)
    valid_pack = sample_window_pack(valid_pack, max_rows=max_valid_windows, seed=seed + 29)

    counters["train_windows"] = int(len(train_pack.x))
    counters["valid_windows"] = int(len(valid_pack.x))
    return train_pack, valid_pack, counters


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def calc_metrics_1d(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = safe_r2(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def calc_group_macro_metrics_1d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_keys: np.ndarray,
) -> Tuple[int, Dict[str, float]]:
    group_df = pd.DataFrame(
        {
            "group_key": group_keys.astype(object),
            "y_true": np.asarray(y_true, dtype=float).reshape(-1),
            "y_pred": np.asarray(y_pred, dtype=float).reshape(-1),
        }
    )
    rows: List[Dict[str, float]] = []
    for _, part in group_df.groupby("group_key", sort=False):
        rows.append(calc_metrics_1d(part["y_true"].to_numpy(float), part["y_pred"].to_numpy(float)))
    if not rows:
        return 0, {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    mm = pd.DataFrame(rows)
    return int(len(rows)), {
        "mse": float(mm["mse"].mean()),
        "rmse": float(mm["rmse"].mean()),
        "mae": float(mm["mae"].mean()),
        "r2": float(mm["r2"].mean()),
    }


def evaluate_multistep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_keys: np.ndarray,
    set_type: str,
    horizon_steps: int,
) -> List[Dict[str, object]]:
    horizons = [h for h in [1, 10, horizon_steps] if h <= horizon_steps]
    horizons = sorted(set(horizons))
    rows: List[Dict[str, object]] = []

    for h in horizons:
        yt = y_true[:, h - 1]
        yp = y_pred[:, h - 1]

        weighted = calc_metrics_1d(yt, yp)
        rows.append(
            {
                "set_type": set_type,
                "aggregation": "weighted",
                "horizon": str(h),
                "n_windows": int(len(yt)),
                "n_points": int(len(yt)),
                "n_groups": int(len(np.unique(group_keys))),
                **weighted,
            }
        )

        n_groups, macro = calc_group_macro_metrics_1d(yt, yp, group_keys)
        rows.append(
            {
                "set_type": set_type,
                "aggregation": "group_macro",
                "horizon": str(h),
                "n_windows": int(len(yt)),
                "n_points": int(len(yt)),
                "n_groups": int(n_groups),
                **macro,
            }
        )

    # All-horizon aggregation
    yt_all = y_true.reshape(-1)
    yp_all = y_pred.reshape(-1)
    weighted_all = calc_metrics_1d(yt_all, yp_all)
    rows.append(
        {
            "set_type": set_type,
            "aggregation": "weighted",
            "horizon": "all",
            "n_windows": int(y_true.shape[0]),
            "n_points": int(yt_all.size),
            "n_groups": int(len(np.unique(group_keys))),
            **weighted_all,
        }
    )
    group_all = np.repeat(group_keys.astype(object), y_true.shape[1])
    n_groups_all, macro_all = calc_group_macro_metrics_1d(yt_all, yp_all, group_all)
    rows.append(
        {
            "set_type": set_type,
            "aggregation": "group_macro",
            "horizon": "all",
            "n_windows": int(y_true.shape[0]),
            "n_points": int(yt_all.size),
            "n_groups": int(n_groups_all),
            **macro_all,
        }
    )
    return rows


def feature_category(raw_feature: str) -> str:
    if raw_feature.startswith("cross_bin_cum_"):
        return "charge_cross_bin_cum"
    if raw_feature.startswith("cross_bin_inc_"):
        return "charge_cross_bin_inc"
    if raw_feature.startswith("discharge_inc_delta_ah_"):
        return "discharge_inc_delta_ah"
    if raw_feature.startswith("discharge_cum_delta_ah_"):
        return "discharge_cum_delta_ah"
    if raw_feature.startswith("discharge_"):
        return "discharge_stat7"
    return "unknown"


def build_feature_importance_summary(feature_names: Sequence[str], importances: np.ndarray) -> pd.DataFrame:
    detail_rows: List[dict] = []
    for feat, imp in zip(feature_names, importances):
        if "__" in feat:
            raw_feature, stat_type = feat.rsplit("__", 1)
        else:
            raw_feature, stat_type = feat, "unknown"
        detail_rows.append(
            {
                "summary_level": "feature",
                "item_name": feat,
                "raw_feature": raw_feature,
                "stat_type": stat_type,
                "raw_category": feature_category(raw_feature),
                "importance": float(imp),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values("importance", ascending=False, kind="mergesort")

    by_stat = (
        detail.groupby("stat_type", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    )
    by_stat = by_stat.assign(
        summary_level="stat_type_sum",
        item_name=by_stat["stat_type"],
        raw_feature="",
        raw_category="",
    )[
        ["summary_level", "item_name", "raw_feature", "stat_type", "raw_category", "importance"]
    ]

    by_cat = (
        detail.groupby("raw_category", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    )
    by_cat = by_cat.assign(
        summary_level="raw_category_sum",
        item_name=by_cat["raw_category"],
        raw_feature="",
        stat_type="",
    )[
        ["summary_level", "item_name", "raw_feature", "stat_type", "raw_category", "importance"]
    ]

    return pd.concat(
        [detail[["summary_level", "item_name", "raw_feature", "stat_type", "raw_category", "importance"]], by_stat, by_cat],
        axis=0,
        ignore_index=True,
    )


def build_dataset_checks(
    split_df: pd.DataFrame,
    raw_feature_dim: int,
    aggr_feature_dim: int,
    expected_raw_dim: int,
    expected_aggr_dim: int,
    train_pack: WindowPack,
    valid_pack: WindowPack,
) -> pd.DataFrame:
    train_keys = set(
        (split_df.loc[split_df["set_type"] == "train", "policy"].astype(str) + "||" + split_df.loc[split_df["set_type"] == "train", "cell_code"].astype(str)).tolist()
    )
    valid_keys = set(
        (split_df.loc[split_df["set_type"] == "valid", "policy"].astype(str) + "||" + split_df.loc[split_df["set_type"] == "valid", "cell_code"].astype(str)).tolist()
    )
    overlap = len(train_keys.intersection(valid_keys))

    all_input = np.concatenate([train_pack.input_cycles, valid_pack.input_cycles], axis=0)
    all_target = np.concatenate([train_pack.target_cycles, valid_pack.target_cycles], axis=0)
    check_target_after_input = bool(np.all(all_input < all_target[:, 0])) if len(all_input) > 0 else False
    check_target_consecutive = bool(np.all(np.diff(all_target, axis=1) == 1)) if len(all_target) > 0 else False
    x_all = np.concatenate([train_pack.x, valid_pack.x], axis=0)
    check_finite = bool(np.isfinite(x_all).all()) if x_all.size > 0 else False

    rows = [
        {"check_item": "check_split_overlap_zero", "pass_flag": int(overlap == 0), "value": int(overlap)},
        {
            "check_item": "check_target_after_input",
            "pass_flag": int(check_target_after_input),
            "value": int(check_target_after_input),
        },
        {
            "check_item": "check_consecutive_horizon",
            "pass_flag": int(check_target_consecutive),
            "value": int(check_target_consecutive),
        },
        {
            "check_item": "check_feature_dim_159_raw",
            "pass_flag": int(raw_feature_dim == expected_raw_dim),
            "value": int(raw_feature_dim),
        },
        {
            "check_item": "check_feature_dim_636_aggregated",
            "pass_flag": int(aggr_feature_dim == expected_aggr_dim),
            "value": int(aggr_feature_dim),
        },
        {
            "check_item": "check_no_nan_inf_features",
            "pass_flag": int(check_finite),
            "value": int(check_finite),
        },
    ]
    return pd.DataFrame(rows)


def write_predictions_long_csv(
    out_path: Path,
    train_pack: WindowPack,
    valid_pack: WindowPack,
    train_pred: np.ndarray,
    valid_pred: np.ndarray,
) -> None:
    def write_one(set_type: str, pack: WindowPack, pred: np.ndarray, mode: str, header: bool) -> None:
        if len(pack.y) == 0:
            return
        n, m = pack.y.shape
        horizon = np.tile(np.arange(1, m + 1, dtype=np.int16), n)
        target_cycle = pack.target_cycles.reshape(-1).astype(np.int64)
        input_cycle = np.repeat(pack.input_cycles, m).astype(np.int64)
        y_true = pack.y.reshape(-1).astype(np.float32)
        y_pred = pred.reshape(-1).astype(np.float32)
        out = pd.DataFrame(
            {
                "set_type": set_type,
                "input_cycle": input_cycle,
                "horizon": horizon,
                "target_cycle": target_cycle,
                "retention_true": y_true,
                "pred_retention": y_pred,
                "residual_retention": y_true - y_pred,
            }
        )
        out.to_csv(out_path, mode=mode, index=False, header=header, encoding="utf-8")

    write_one("train", train_pack, train_pred, mode="w", header=True)
    write_one("valid", valid_pack, valid_pred, mode="a", header=False)


def save_valid_scatter_horizons(
    valid_true: np.ndarray,
    valid_pred: np.ndarray,
    horizon_steps: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    horizons = [h for h in [1, 10, horizon_steps] if h <= horizon_steps]
    if not horizons:
        return
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.0 * len(horizons), 4.8))
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        yt = valid_true[:, h - 1].astype(float)
        yp = valid_pred[:, h - 1].astype(float)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        m = calc_metrics_1d(yt, yp)
        ax.scatter(yt, yp, s=8, alpha=0.3, color="#0ea5e9")
        ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", linewidth=1.2)
        ax.set_title(f"h={h} | R2={m['r2']:.4f}")
        ax.set_xlabel("True retention")
        ax.set_ylabel("Pred retention")
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Valid Retention Scatter by Horizon")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def build_report(
    args: argparse.Namespace,
    fonts: Sequence[str],
    merged_rows: int,
    raw_feature_dim: int,
    aggr_feature_dim: int,
    charge_stats: Mapping[str, int],
    discharge_stats: Mapping[str, int],
    window_stats: Mapping[str, int],
    metrics_df: pd.DataFrame,
    checks_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# RF 多步容量保持率预测报告（5类工况特征）")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 字体回退：`{', '.join(fonts)}`")
    lines.append(f"- 任务口径：`1:{int(args.n_history)} -> {int(args.n_history)+1}:{int(args.n_history)+int(args.horizon_steps)}`")
    lines.append(
        f"- retention口径：`q_ref=前{int(args.q_ref_cycles)}个有效循环中位数`，过滤 `q∈[{args.q_min},{args.q_max}]`，`retention∈[{args.retention_min},{args.retention_max}]`"
    )
    lines.append("")

    lines.append("## 2. 特征口径")
    lines.append(f"- 充电cross-bin累计：**{N_CROSS_BINS}** 列")
    lines.append(f"- 充电cross-bin当前增量：**{N_CROSS_BINS}** 列")
    lines.append("- 放电当前区间容量增量：**16** 列")
    lines.append("- 放电累计区间容量：**16** 列")
    lines.append("- 放电汇总统计：**7** 列")
    lines.append(f"- raw特征维度：**{raw_feature_dim}**")
    lines.append(f"- 聚合后特征维度（last/mean/std/slope）：**{aggr_feature_dim}**")
    lines.append(f"- 放电区间口径：`range_count == {FIRST_OCCURRENCE_RANGE_COUNT}`")
    lines.append("")

    lines.append("## 3. 数据规模")
    lines.append(f"- merged cycle级样本：**{merged_rows:,}**")
    lines.append(f"- 训练组/验证组：**{window_stats.get('groups_train', 0)} / {window_stats.get('groups_valid', 0)}**")
    lines.append(
        f"- 可构造窗口组（train/valid）：**{window_stats.get('groups_with_windows_train', 0)} / {window_stats.get('groups_with_windows_valid', 0)}**"
    )
    lines.append(f"- 训练窗口数：**{window_stats.get('train_windows', 0):,}**")
    lines.append(f"- 验证窗口数：**{window_stats.get('valid_windows', 0):,}**")
    lines.append(
        f"- charge/discharge cycle行数：**{charge_stats.get('charge_cycle_rows', -1):,} / {discharge_stats.get('discharge_cycle_rows', -1):,}**"
    )
    lines.append("")

    lines.append("## 4. 指标结果")
    lines.append("| set_type | aggregation | horizon | n_windows | n_points | n_groups | MAE | RMSE | R2 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in metrics_df.itertuples(index=False):
        lines.append(
            f"| {row.set_type} | {row.aggregation} | {row.horizon} | {int(row.n_windows)} | {int(row.n_points)} | {int(row.n_groups)} | "
            f"{float(row.mae):.6f} | {float(row.rmse):.6f} | {float(row.r2):.6f} |"
        )
    lines.append("")

    def _lookup(st: str, agg: str, h: str, col: str) -> float:
        part = metrics_df.loc[
            (metrics_df["set_type"] == st) & (metrics_df["aggregation"] == agg) & (metrics_df["horizon"] == h),
            col,
        ]
        if part.empty:
            return float("nan")
        return float(part.iloc[0])

    train_h1 = _lookup("train", "weighted", "1", "r2")
    valid_h1 = _lookup("valid", "weighted", "1", "r2")
    train_hm = _lookup("train", "weighted", str(int(args.horizon_steps)), "r2")
    valid_hm = _lookup("valid", "weighted", str(int(args.horizon_steps)), "r2")
    valid_all_macro = _lookup("valid", "group_macro", "all", "r2")
    valid_all_weighted = _lookup("valid", "weighted", "all", "r2")

    lines.append("## 5. 结论")
    lines.append(
        f"- 短期预测（h=1）R2：train={train_h1:.6f}，valid={valid_h1:.6f}，gap={train_h1 - valid_h1:.6f}。"
    )
    lines.append(
        f"- 长期预测（h={int(args.horizon_steps)}）R2：train={train_hm:.6f}，valid={valid_hm:.6f}，gap={train_hm - valid_hm:.6f}。"
    )
    lines.append(
        f"- 验证集 `all` 指标：weighted R2={valid_all_weighted:.6f}，group-macro R2={valid_all_macro:.6f}。"
    )
    lines.append("- 若 weighted 与 group-macro 差距较大，优先以 group-macro 结论为准。")
    lines.append("")

    lines.append("## 6. 数据一致性检查")
    lines.append("| check_item | pass_flag | value |")
    lines.append("|---|---:|---:|")
    for row in checks_df.itertuples(index=False):
        lines.append(f"| {row.check_item} | {int(row.pass_flag)} | {int(row.value)} |")
    lines.append("")
    lines.append("## 7. 散点图")
    lines.append("![valid_retention_scatter_horizons](./valid_retention_scatter_horizons.png)")
    return "\n".join(lines)


def append_session_log(prompt_text: str, response_text: str) -> None:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"session_{datetime.now().strftime('%Y-%m-%d')}.md"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp}\n\n")
        f.write("[USER PROMPT]\n")
        f.write(prompt_text.strip() + "\n\n")
        f.write("[ASSISTANT RESPONSE]\n")
        f.write(response_text.strip() + "\n\n---\n")


def main() -> None:
    args = parse_args()
    if args.n_history < 2:
        raise ValueError("--n-history must be >= 2")
    if args.horizon_steps < 1:
        raise ValueError("--horizon-steps must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts = ensure_matplotlib_config()
    np.random.seed(int(args.random_seed))

    split_df = base_lstm.load_split_map(args.train_split_path, args.valid_split_path)
    charge_df, charge_stats_raw, charge_cum_cols, charge_inc_cols = load_charge_feature_table(args.timeseries_path)
    discharge_df, discharge_stats, inc_delta_cols, cum_delta_cols, stat7_cols = load_discharge_feature_table(args.discharge_path)
    label_df = dqdv_retention.load_retention_labels(
        life_path=args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )

    feature_packs = FeaturePacks(
        charge_cum_cols=charge_cum_cols,
        charge_inc_cols=charge_inc_cols,
        discharge_inc_delta_cols=inc_delta_cols,
        discharge_cum_delta_cols=cum_delta_cols,
        discharge_stat7_cols=stat7_cols,
    )
    raw_feature_cols = dedupe_keep_order(feature_packs.raw_feature_cols)
    if len(raw_feature_cols) != 159:
        raise RuntimeError(f"Expected raw feature dim 159, got {len(raw_feature_cols)}")

    merged = merge_cycle_dataset(
        charge_df=charge_df,
        discharge_df=discharge_df,
        label_df=label_df,
        split_df=split_df,
        raw_feature_cols=raw_feature_cols,
    )
    if merged.empty:
        raise RuntimeError("Merged dataset is empty.")

    train_pack, valid_pack, window_stats = build_train_valid_windows(
        merged=merged,
        raw_feature_cols=raw_feature_cols,
        n_history=int(args.n_history),
        horizon_steps=int(args.horizon_steps),
        max_train_windows=int(args.max_train_windows),
        max_valid_windows=int(args.max_valid_windows),
        seed=int(args.random_seed),
    )
    if len(train_pack.x) == 0 or len(valid_pack.x) == 0:
        raise RuntimeError("Train or valid windows are empty. Check N/M settings.")

    aggr_feature_cols = aggregate_feature_names(raw_feature_cols)
    if len(aggr_feature_cols) != 636:
        raise RuntimeError(f"Expected aggregated feature dim 636, got {len(aggr_feature_cols)}")

    model_params: Dict[str, object] = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_samples_split": int(args.min_samples_split),
        "max_features": float(args.max_features),
        "bootstrap": bool(args.bootstrap),
        "max_samples": None if args.max_samples is None else float(args.max_samples),
        "criterion": str(args.criterion),
        "random_state": int(args.random_seed),
        "n_jobs": 1,
    }
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(**model_params)),
        ]
    )
    model.fit(train_pack.x.astype(float), train_pack.y.astype(float))
    train_pred = model.predict(train_pack.x.astype(float)).astype(np.float32)
    valid_pred = model.predict(valid_pack.x.astype(float)).astype(np.float32)

    metric_rows: List[Dict[str, object]] = []
    metric_rows.extend(
        evaluate_multistep(
            y_true=train_pack.y,
            y_pred=train_pred,
            group_keys=train_pack.group_keys,
            set_type="train",
            horizon_steps=int(args.horizon_steps),
        )
    )
    metric_rows.extend(
        evaluate_multistep(
            y_true=valid_pack.y,
            y_pred=valid_pred,
            group_keys=valid_pack.group_keys,
            set_type="valid",
            horizon_steps=int(args.horizon_steps),
        )
    )
    metrics_df = pd.DataFrame(metric_rows)

    fi = model.named_steps["rf"].feature_importances_.astype(float)
    fi_df = build_feature_importance_summary(aggr_feature_cols, fi)

    checks_df = build_dataset_checks(
        split_df=split_df,
        raw_feature_dim=len(raw_feature_cols),
        aggr_feature_dim=train_pack.x.shape[1],
        expected_raw_dim=159,
        expected_aggr_dim=636,
        train_pack=train_pack,
        valid_pack=valid_pack,
    )

    out_metrics = args.output_dir / "train_valid_metrics_by_horizon.csv"
    out_preds = args.output_dir / "train_valid_predictions_long.csv"
    out_checks = args.output_dir / "dataset_checks.csv"
    out_fi = args.output_dir / "feature_importance_summary.csv"
    out_scatter = args.output_dir / "valid_retention_scatter_horizons.png"
    out_report = args.output_dir / "rf_charge_crossbin_multistep_retention_report.md"
    out_config = args.output_dir / "run_config.json"

    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8")
    write_predictions_long_csv(
        out_path=out_preds,
        train_pack=train_pack,
        valid_pack=valid_pack,
        train_pred=train_pred,
        valid_pred=valid_pred,
    )
    checks_df.to_csv(out_checks, index=False, encoding="utf-8")
    fi_df.to_csv(out_fi, index=False, encoding="utf-8")
    save_valid_scatter_horizons(
        valid_true=valid_pack.y,
        valid_pred=valid_pred,
        horizon_steps=int(args.horizon_steps),
        out_path=out_scatter,
    )

    charge_stats = {
        "charge_cycle_rows": int(len(charge_df)),
        "charge_cross_bin_feature_dim": int(N_CROSS_BINS),
        "charge_rows_after_filter": int(charge_stats_raw.get("timeseries_rows_after_dedup", -1)),
    }
    report = build_report(
        args=args,
        fonts=fonts,
        merged_rows=int(len(merged)),
        raw_feature_dim=int(len(raw_feature_cols)),
        aggr_feature_dim=int(train_pack.x.shape[1]),
        charge_stats=charge_stats,
        discharge_stats=discharge_stats,
        window_stats=window_stats,
        metrics_df=metrics_df,
        checks_df=checks_df,
    )
    out_report.write_text(report, encoding="utf-8")

    run_cfg = {
        "script": str(SCRIPT_PATH),
        "python_executable": os.path.realpath(os.sys.executable),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "raw_feature_dim": int(len(raw_feature_cols)),
        "aggregated_feature_dim": int(train_pack.x.shape[1]),
        "window_stats": window_stats,
        "model_params": model_params,
        "check_pass_all": int(bool((checks_df["pass_flag"] == 1).all())),
    }
    out_config.write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_preds}")
    print(f"Saved: {out_checks}")
    print(f"Saved: {out_fi}")
    print(f"Saved: {out_scatter}")
    print(f"Saved: {out_report}")
    print(f"Saved: {out_config}")
    print(
        f"Train/Valid windows={len(train_pack.x):,}/{len(valid_pack.x):,} | "
        f"raw/aggr dim={len(raw_feature_cols)}/{train_pack.x.shape[1]}"
    )

    append_session_log(
        prompt_text="实现RF多步预测任务（5类工况特征，N/M可配置，无泄露）并执行冒烟+全量运行。",
        response_text=(
            f"新增脚本 {SCRIPT_PATH.name}，完成5类特征口径、滑窗多步RF训练与输出；"
            f"输出目录={args.output_dir}，train_windows={len(train_pack.x)}，valid_windows={len(valid_pack.x)}。"
        ),
    )


if __name__ == "__main__":
    main()
