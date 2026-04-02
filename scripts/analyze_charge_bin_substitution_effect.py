from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"
N_CROSS_BINS = 60
POLICY_COLS = ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]


@dataclass
class DMLFitResult:
    """Container for one-bin DML residualization result."""

    theta_raw: float
    var_treatment: float
    y_residual: np.ndarray
    t_residual: np.ndarray
    groups: np.ndarray
    r2_y: float
    r2_t: float
    n_rows: int
    n_groups: int
    skip_reason: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze 60-bin charging features with screening + substitution-effect causal estimation."
        )
    )
    parser.add_argument(
        "--timeseries-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv",
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "life_performance.csv",
    )
    parser.add_argument(
        "--bin-edges-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_aging_path_bin_edges.csv",
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
        default=REPO_ROOT / "outputs" / "analysis" / "charge_bin_substitution_causal",
    )
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260402)
    parser.add_argument("--screen-n-estimators", type=int, default=240)
    parser.add_argument("--screen-perm-repeats", type=int, default=10)
    parser.add_argument("--dml-splits", type=int, default=5)
    parser.add_argument("--nuisance-n-estimators", type=int, default=120)
    parser.add_argument("--nuisance-max-depth", type=int, default=12)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--placebo-perm-iters", type=int, default=100)
    parser.add_argument("--timeseries-chunksize", type=int, default=200000)
    return parser.parse_args()


def ensure_matplotlib_config() -> List[str]:
    """Configure matplotlib backend and font fallback list."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

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


def load_split(
    train_path: Path,
    valid_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/valid split and produce split map."""

    train = pd.read_csv(train_path, encoding=ENCODING)
    valid = pd.read_csv(valid_path, encoding=ENCODING)
    cols = ["policy", "cell_code", *POLICY_COLS]
    train = train[cols].copy()
    valid = valid[cols].copy()
    for df in [train, valid]:
        df["policy"] = df["policy"].astype(str)
        df["cell_code"] = df["cell_code"].astype(str)
        for col in POLICY_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    split_map = (
        pd.concat([train, valid], ignore_index=True)
        .drop_duplicates(["policy", "cell_code"], keep="first")
    )

    train_keys = set(train["policy"] + "||" + train["cell_code"])
    valid_keys = set(valid["policy"] + "||" + valid["cell_code"])
    overlap = len(train_keys.intersection(valid_keys))
    if overlap > 0:
        raise RuntimeError(f"split leakage detected: {overlap}")
    return train, valid, split_map


def load_timeseries_panel(
    timeseries_path: Path,
    chunksize: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Build cycle-level panel with 60 increment bins and share features."""

    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "cross_bin",
        "cycle_charge_time_h",
        "nonzero_cross_bin_count_cycle",
        "is_abnormal_cell",
    ]
    part_frames: List[pd.DataFrame] = []
    total_rows_raw = 0
    total_rows_after_clean = 0

    reader = pd.read_csv(
        timeseries_path,
        usecols=usecols,
        encoding="utf-8",
        chunksize=chunksize,
        engine="python",
    )
    for chunk in reader:
        total_rows_raw += int(len(chunk))
        chunk["policy"] = chunk["policy"].astype(str)
        chunk["cell_code"] = chunk["cell_code"].astype(str)
        for col in [
            "cycles",
            "cross_bin",
            "cycle_charge_time_h",
            "nonzero_cross_bin_count_cycle",
            "is_abnormal_cell",
        ]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=usecols).copy()
        chunk["cycles"] = chunk["cycles"].astype(int)
        chunk["cross_bin"] = chunk["cross_bin"].astype(int)
        chunk["nonzero_cross_bin_count_cycle"] = chunk[
            "nonzero_cross_bin_count_cycle"
        ].astype(int)
        chunk["is_abnormal_cell"] = chunk["is_abnormal_cell"].astype(int)
        chunk = chunk[(chunk["cross_bin"] >= 1) & (chunk["cross_bin"] <= N_CROSS_BINS)].copy()
        if chunk.empty:
            continue
        total_rows_after_clean += int(len(chunk))
        part = (
            chunk.groupby(["policy", "cell_code", "cycles", "cross_bin"], as_index=False)
            .agg(
                cycle_charge_time_h=("cycle_charge_time_h", "sum"),
                nonzero_cross_bin_count_cycle=("nonzero_cross_bin_count_cycle", "max"),
                is_abnormal_cell=("is_abnormal_cell", "max"),
            )
            .reset_index(drop=True)
        )
        part_frames.append(part)

    if not part_frames:
        raise RuntimeError("No valid rows found in timeseries file.")

    merged = pd.concat(part_frames, ignore_index=True)
    merged = (
        merged.groupby(["policy", "cell_code", "cycles", "cross_bin"], as_index=False)
        .agg(
            cycle_charge_time_h=("cycle_charge_time_h", "sum"),
            nonzero_cross_bin_count_cycle=("nonzero_cross_bin_count_cycle", "max"),
            is_abnormal_cell=("is_abnormal_cell", "max"),
        )
        .reset_index(drop=True)
    )
    cycle_keys = ["policy", "cell_code", "cycles"]
    meta = (
        merged.groupby(cycle_keys, as_index=False)
        .agg(
            nonzero_cross_bin_count_cycle=("nonzero_cross_bin_count_cycle", "max"),
            is_abnormal_cell=("is_abnormal_cell", "max"),
        )
        .reset_index(drop=True)
    )

    inc_cols = [f"cross_bin_inc_{idx:02d}_h" for idx in range(1, N_CROSS_BINS + 1)]
    share_cols = [f"share_{idx:02d}" for idx in range(1, N_CROSS_BINS + 1)]

    pivot = (
        merged.pivot_table(
            index=cycle_keys,
            columns="cross_bin",
            values="cycle_charge_time_h",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=list(range(1, N_CROSS_BINS + 1)), fill_value=0.0)
        .reset_index()
        .rename(columns={idx: inc_cols[idx - 1] for idx in range(1, N_CROSS_BINS + 1)})
    )
    panel = pivot.merge(meta, on=cycle_keys, how="inner", validate="one_to_one")
    panel["cycle_total_charge_h"] = panel[inc_cols].sum(axis=1)
    rows_before_positive = int(len(panel))
    panel = panel[panel["cycle_total_charge_h"] > 0].copy()
    rows_after_positive = int(len(panel))
    for idx, share_col in enumerate(share_cols, start=1):
        inc_col = f"cross_bin_inc_{idx:02d}_h"
        panel[share_col] = panel[inc_col] / panel["cycle_total_charge_h"]
    panel["share_sum"] = panel[share_cols].sum(axis=1)

    stats = {
        "timeseries_rows_raw": total_rows_raw,
        "timeseries_rows_after_clean": total_rows_after_clean,
        "panel_rows_before_positive_filter": rows_before_positive,
        "panel_rows_after_positive_filter": rows_after_positive,
    }
    return panel, share_cols, stats


def load_labels(
    life_path: Path,
    q_min: float,
    q_max: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and range-filter labels."""

    label = pd.read_csv(
        life_path,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
        encoding=ENCODING,
    )
    label["policy"] = label["policy"].astype(str)
    label["cell_code"] = label["cell_code"].astype(str)
    label["cycles"] = pd.to_numeric(label["cycles"], errors="coerce")
    label["q_discharge"] = pd.to_numeric(label["q_discharge"], errors="coerce")
    rows_before = int(len(label))
    label = label.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    rows_after_dropna = int(len(label))
    lt_qmin = int((label["q_discharge"] < q_min).sum())
    gt_qmax = int((label["q_discharge"] > q_max).sum())
    label = label[
        (label["q_discharge"] >= float(q_min))
        & (label["q_discharge"] <= float(q_max))
    ].copy()
    label["cycles"] = label["cycles"].astype(int)
    label = (
        label.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")
        .drop_duplicates(["policy", "cell_code", "cycles"], keep="last")
        .reset_index(drop=True)
    )
    stats = {
        "label_rows_before_dropna": rows_before,
        "label_rows_after_dropna": rows_after_dropna,
        "label_rows_lt_qmin_removed": lt_qmin,
        "label_rows_gt_qmax_removed": gt_qmax,
        "label_rows_after_range_filter": int(len(label)),
    }
    return label, stats


def build_cycle_dataset(
    panel_df: pd.DataFrame,
    split_map: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge panel features with split and next-cycle labels."""

    cycle_keys = ["policy", "cell_code", "cycles"]
    merged = panel_df.merge(
        split_map[["policy", "cell_code", *POLICY_COLS, "set_type"]],
        on=["policy", "cell_code"],
        how="inner",
        validate="many_to_one",
    )
    label_t = label_df.rename(columns={"q_discharge": "q_t"}).copy()
    next_label = label_df[["policy", "cell_code", "cycles", "q_discharge"]].copy()
    next_label["cycles"] = next_label["cycles"] - 1
    next_label = next_label.rename(columns={"q_discharge": "q_next"})

    merged = merged.merge(
        label_t[cycle_keys + ["q_t"]],
        on=cycle_keys,
        how="inner",
        validate="many_to_one",
    )
    merged = merged.merge(
        next_label[cycle_keys + ["q_next"]],
        on=cycle_keys,
        how="inner",
        validate="many_to_one",
    )
    merged["group_key"] = merged["policy"] + "||" + merged["cell_code"]
    merged = merged.sort_values(cycle_keys, kind="mergesort").reset_index(drop=True)
    return merged


def normalize_minmax(series: pd.Series) -> pd.Series:
    """Normalize one numeric series with min-max scaling."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + 1e-15:
        return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
    return (values - lo) / (hi - lo)


def build_spearman_screen(
    train_df: pd.DataFrame,
    share_cols: Sequence[str],
) -> pd.DataFrame:
    """Build Spearman screening table for 60 share features."""

    rows: List[dict] = []
    y = pd.to_numeric(train_df["q_next"], errors="coerce")
    for col in share_cols:
        x = pd.to_numeric(train_df[col], errors="coerce")
        corr = float(x.corr(y, method="spearman"))
        if not np.isfinite(corr):
            corr = 0.0
        rows.append(
            {
                "share_feature": col,
                "cross_bin": int(col.split("_")[-1]),
                "spearman_corr": corr,
                "spearman_abs": abs(corr),
            }
        )
    return pd.DataFrame(rows)


def build_permutation_screen(
    train_df: pd.DataFrame,
    share_cols: Sequence[str],
    control_cols: Sequence[str],
    n_estimators: int,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Build permutation-importance screening on train subset holdout."""

    feature_cols = [*share_cols, *control_cols]
    x = train_df[feature_cols].to_numpy(float)
    y = train_df["q_next"].to_numpy(float)
    groups = train_df["group_key"].astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(x, y, groups=groups))
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]
    x_va = x[va_idx]
    y_va = y[va_idx]

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=14,
        min_samples_leaf=4,
        min_samples_split=8,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_tr, y_tr)
    perm = permutation_importance(
        model,
        x_va,
        y_va,
        n_repeats=int(n_repeats),
        random_state=seed,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "rf_perm_importance": perm.importances_mean.astype(float),
            "rf_perm_importance_std": perm.importances_std.astype(float),
        }
    )
    out["rf_perm_importance"] = out["rf_perm_importance"].clip(lower=0.0)
    out["cross_bin"] = out["feature"].map(
        lambda name: int(name.split("_")[-1]) if str(name).startswith("share_") else -1
    )
    return out[out["cross_bin"] >= 1].rename(columns={"feature": "share_feature"}).reset_index(drop=True)


def estimate_theta(y_residual: np.ndarray, t_residual: np.ndarray) -> Tuple[float, float]:
    """Estimate theta and treatment residual variance."""

    y = np.asarray(y_residual, dtype=float)
    t = np.asarray(t_residual, dtype=float)
    y = y - np.mean(y)
    t = t - np.mean(t)
    var_t = float(np.mean(t * t))
    if not np.isfinite(var_t) or var_t <= 1e-12:
        return float("nan"), var_t
    cov_yt = float(np.mean(y * t))
    theta = cov_yt / var_t
    return theta, var_t


def fit_dml_residualized(
    df: pd.DataFrame,
    treatment_col: str,
    control_cols: Sequence[str],
    outcome_col: str,
    group_col: str,
    n_splits: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    seed: int,
) -> DMLFitResult:
    """Fit one treatment with residualized DML via grouped cross-fitting."""

    work = df[[*control_cols, treatment_col, outcome_col, group_col]].dropna().copy()
    if work.empty:
        return DMLFitResult(
            theta_raw=float("nan"),
            var_treatment=float("nan"),
            y_residual=np.array([]),
            t_residual=np.array([]),
            groups=np.array([]),
            r2_y=float("nan"),
            r2_t=float("nan"),
            n_rows=0,
            n_groups=0,
            skip_reason="empty_after_dropna",
        )
    y = work[outcome_col].to_numpy(float)
    t = work[treatment_col].to_numpy(float)
    x = work[list(control_cols)].to_numpy(float)
    groups = work[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return DMLFitResult(
            theta_raw=float("nan"),
            var_treatment=float("nan"),
            y_residual=np.array([]),
            t_residual=np.array([]),
            groups=groups,
            r2_y=float("nan"),
            r2_t=float("nan"),
            n_rows=int(len(work)),
            n_groups=int(len(unique_groups)),
            skip_reason="too_few_groups",
        )
    folds = int(min(n_splits, len(unique_groups)))
    if folds < 2:
        folds = 2
    gkf = GroupKFold(n_splits=folds)
    y_hat = np.full(len(work), np.nan, dtype=float)
    t_hat = np.full(len(work), np.nan, dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        m_y = RandomForestRegressor(
            n_estimators=int(rf_n_estimators),
            max_depth=int(rf_max_depth),
            min_samples_leaf=20,
            min_samples_split=40,
            max_features="sqrt",
            random_state=seed + fold_idx * 37,
            n_jobs=1,
        )
        m_t = RandomForestRegressor(
            n_estimators=int(rf_n_estimators),
            max_depth=int(rf_max_depth),
            min_samples_leaf=20,
            min_samples_split=40,
            max_features="sqrt",
            random_state=seed + fold_idx * 71,
            n_jobs=1,
        )
        m_y.fit(x[tr_idx], y[tr_idx])
        m_t.fit(x[tr_idx], t[tr_idx])
        y_hat[va_idx] = m_y.predict(x[va_idx])
        t_hat[va_idx] = m_t.predict(x[va_idx])

    if np.isnan(y_hat).any() or np.isnan(t_hat).any():
        return DMLFitResult(
            theta_raw=float("nan"),
            var_treatment=float("nan"),
            y_residual=np.array([]),
            t_residual=np.array([]),
            groups=groups,
            r2_y=float("nan"),
            r2_t=float("nan"),
            n_rows=int(len(work)),
            n_groups=int(len(unique_groups)),
            skip_reason="crossfit_predictions_nan",
        )
    y_res = y - y_hat
    t_res = t - t_hat
    theta_raw, var_t = estimate_theta(y_res, t_res)
    if not np.isfinite(theta_raw):
        return DMLFitResult(
            theta_raw=float("nan"),
            var_treatment=var_t,
            y_residual=y_res,
            t_residual=t_res,
            groups=groups,
            r2_y=float(r2_score(y, y_hat)),
            r2_t=float(r2_score(t, t_hat)),
            n_rows=int(len(work)),
            n_groups=int(len(unique_groups)),
            skip_reason="low_treatment_variance",
        )
    return DMLFitResult(
        theta_raw=float(theta_raw),
        var_treatment=float(var_t),
        y_residual=y_res,
        t_residual=t_res,
        groups=groups,
        r2_y=float(r2_score(y, y_hat)),
        r2_t=float(r2_score(t, t_hat)),
        n_rows=int(len(work)),
        n_groups=int(len(unique_groups)),
        skip_reason="",
    )


def cluster_bootstrap_theta(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    groups: np.ndarray,
    n_iters: int,
    seed: int,
) -> np.ndarray:
    """Cluster bootstrap theta distribution by group keys."""

    y = np.asarray(y_residual, dtype=float)
    t = np.asarray(t_residual, dtype=float)
    g = np.asarray(groups)
    unique_groups = np.unique(g)
    n_groups = len(unique_groups)
    if n_groups < 2 or len(y) == 0:
        return np.array([], dtype=float)

    idx_map: Dict[str, np.ndarray] = {}
    for key in unique_groups:
        idx_map[str(key)] = np.where(g == key)[0]

    rng = np.random.default_rng(seed)
    values: List[float] = []
    unique_group_list = [str(x) for x in unique_groups]
    for _ in range(int(n_iters)):
        sampled = rng.choice(unique_group_list, size=n_groups, replace=True)
        idx_parts = [idx_map[k] for k in sampled]
        boot_idx = np.concatenate(idx_parts)
        theta, _ = estimate_theta(y[boot_idx], t[boot_idx])
        if np.isfinite(theta):
            values.append(float(theta))
    return np.asarray(values, dtype=float)


def placebo_permutation_theta(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    n_iters: int,
    seed: int,
) -> np.ndarray:
    """Run placebo permutation test by shuffling treatment residuals."""

    y = np.asarray(y_residual, dtype=float)
    t = np.asarray(t_residual, dtype=float)
    if len(y) == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(int(n_iters)):
        perm_t = rng.permutation(t)
        theta, _ = estimate_theta(y, perm_t)
        if np.isfinite(theta):
            values.append(float(theta))
    return np.asarray(values, dtype=float)


def benjamini_hochberg_qvalues(p_values: pd.Series) -> pd.Series:
    """Adjust p-values by Benjamini-Hochberg FDR."""

    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index)
    order = valid.sort_values().index.to_list()
    n = len(order)
    q = np.full(n, np.nan, dtype=float)
    prev = 1.0
    for rev_rank, idx in enumerate(reversed(order), start=1):
        rank = n - rev_rank + 1
        val = float(valid.loc[idx]) * n / rank
        prev = min(prev, val)
        q[n - rev_rank] = prev
    q = np.clip(q, 0.0, 1.0)
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    for pos, idx in enumerate(order):
        out.loc[idx] = q[pos]
    return out


def run_causal_estimation(
    df: pd.DataFrame,
    top_bins: pd.DataFrame,
    control_cols: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run DML + bootstrap + placebo for selected bins."""

    rows: List[dict] = []
    for row in top_bins.itertuples(index=False):
        share_col = str(row.share_feature)
        cross_bin = int(row.cross_bin)
        fit = fit_dml_residualized(
            df=df,
            treatment_col=share_col,
            control_cols=control_cols,
            outcome_col="q_next",
            group_col="group_key",
            n_splits=int(args.dml_splits),
            rf_n_estimators=int(args.nuisance_n_estimators),
            rf_max_depth=int(args.nuisance_max_depth),
            seed=int(args.random_seed) + cross_bin * 101,
        )
        result = {
            "cross_bin": cross_bin,
            "share_feature": share_col,
            "soc_bin": int(row.soc_bin),
            "rate_bin": int(row.rate_bin),
            "temp_bin": int(row.temp_bin),
            "cross_label": str(row.cross_label),
            "n_rows": int(fit.n_rows),
            "n_groups": int(fit.n_groups),
            "var_treatment": float(fit.var_treatment),
            "r2_y_nuisance": float(fit.r2_y),
            "r2_t_nuisance": float(fit.r2_t),
            "skip_reason": str(fit.skip_reason),
        }
        if fit.skip_reason:
            result.update(
                {
                    "effect_per_1pp_ah": np.nan,
                    "effect_per_5pp_ah": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "placebo_mean_1pp_ah": np.nan,
                    "placebo_std_1pp_ah": np.nan,
                }
            )
            rows.append(result)
            continue

        boot = cluster_bootstrap_theta(
            y_residual=fit.y_residual,
            t_residual=fit.t_residual,
            groups=fit.groups,
            n_iters=int(args.bootstrap_iters),
            seed=int(args.random_seed) + cross_bin * 131,
        )
        if len(boot) == 0:
            result["skip_reason"] = "bootstrap_failed"
            result.update(
                {
                    "effect_per_1pp_ah": np.nan,
                    "effect_per_5pp_ah": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "placebo_mean_1pp_ah": np.nan,
                    "placebo_std_1pp_ah": np.nan,
                }
            )
            rows.append(result)
            continue

        placebo = placebo_permutation_theta(
            y_residual=fit.y_residual,
            t_residual=fit.t_residual,
            n_iters=int(args.placebo_perm_iters),
            seed=int(args.random_seed) + cross_bin * 151,
        )
        effect_1pp = float(fit.theta_raw * 0.01)
        effect_5pp = float(fit.theta_raw * 0.05)
        boot_1pp = boot * 0.01
        ci_low, ci_high = np.percentile(boot_1pp, [2.5, 97.5]).tolist()
        p_value = float(
            min(
                1.0,
                2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))),
            )
        )
        result.update(
            {
                "effect_per_1pp_ah": effect_1pp,
                "effect_per_5pp_ah": effect_5pp,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": p_value,
                "q_value": np.nan,
                "placebo_mean_1pp_ah": float(np.mean(placebo) * 0.01) if len(placebo) > 0 else np.nan,
                "placebo_std_1pp_ah": float(np.std(placebo, ddof=1) * 0.01) if len(placebo) > 1 else np.nan,
            }
        )
        rows.append(result)

    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg_qvalues(out["p_value"])
    out["abs_effect_per_1pp_ah"] = out["effect_per_1pp_ah"].abs()
    out = out.sort_values(
        ["q_value", "abs_effect_per_1pp_ah"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def build_screening_tables(
    train_df: pd.DataFrame,
    share_cols: Sequence[str],
    control_cols: Sequence[str],
    bin_edges_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build screening scores and top-k bins, and compute stability overlap."""

    sp = build_spearman_screen(train_df, share_cols=share_cols)
    pm = build_permutation_screen(
        train_df=train_df,
        share_cols=share_cols,
        control_cols=control_cols,
        n_estimators=int(args.screen_n_estimators),
        n_repeats=int(args.screen_perm_repeats),
        seed=int(args.random_seed),
    )
    score = sp.merge(pm, on=["share_feature", "cross_bin"], how="inner", validate="one_to_one")
    score["score_corr_norm"] = normalize_minmax(score["spearman_abs"])
    score["score_perm_norm"] = normalize_minmax(score["rf_perm_importance"])
    score["combined_score"] = (score["score_corr_norm"] + score["score_perm_norm"]) / 2.0
    score = score.merge(
        bin_edges_df[
            [
                "cross_bin",
                "soc_bin",
                "rate_bin",
                "temp_bin",
                "soc_label",
                "rate_label",
                "temp_label",
                "cross_label",
            ]
        ],
        on="cross_bin",
        how="left",
        validate="many_to_one",
    )
    score = score.sort_values(
        ["combined_score", "spearman_abs", "rf_perm_importance"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    score["rank_combined"] = np.arange(1, len(score) + 1)

    sp_rerun = build_spearman_screen(train_df, share_cols=share_cols)
    pm_rerun = build_permutation_screen(
        train_df=train_df,
        share_cols=share_cols,
        control_cols=control_cols,
        n_estimators=int(args.screen_n_estimators),
        n_repeats=int(args.screen_perm_repeats),
        seed=int(args.random_seed),
    )
    score_rerun = sp_rerun.merge(
        pm_rerun,
        on=["share_feature", "cross_bin"],
        how="inner",
        validate="one_to_one",
    )
    score_rerun["score_corr_norm"] = normalize_minmax(score_rerun["spearman_abs"])
    score_rerun["score_perm_norm"] = normalize_minmax(score_rerun["rf_perm_importance"])
    score_rerun["combined_score"] = (
        score_rerun["score_corr_norm"] + score_rerun["score_perm_norm"]
    ) / 2.0
    score_rerun = score_rerun.sort_values(
        ["combined_score", "spearman_abs", "rf_perm_importance"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    top_a = set(score.head(int(args.top_k))["cross_bin"].astype(int).tolist())
    top_b = set(score_rerun.head(int(args.top_k))["cross_bin"].astype(int).tolist())
    overlap_ratio = float(len(top_a.intersection(top_b)) / max(1, int(args.top_k)))

    top_bins = score.head(int(args.top_k)).copy().reset_index(drop=True)
    return score, top_bins, overlap_ratio


def save_effect_forest_plot(
    causal_df: pd.DataFrame,
    out_png: Path,
) -> None:
    """Save forest plot of per-1pp effects with confidence intervals."""

    import matplotlib.pyplot as plt  # noqa: WPS433

    plot_df = causal_df[causal_df["skip_reason"] == ""].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))
        ax.text(0.5, 0.5, "No valid causal effects.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, format="png")
        plt.close(fig)
        return

    plot_df = plot_df.sort_values("effect_per_1pp_ah", ascending=True, kind="mergesort")
    labels = [
        f"bin{int(r.cross_bin):02d} ({str(r.cross_label)})"
        for r in plot_df.itertuples(index=False)
    ]
    y_pos = np.arange(len(plot_df))
    effects = plot_df["effect_per_1pp_ah"].to_numpy(float)
    ci_low = plot_df["ci_low"].to_numpy(float)
    ci_high = plot_df["ci_high"].to_numpy(float)
    err_left = effects - ci_low
    err_right = ci_high - effects

    fig_h = max(4.8, 0.5 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(11.0, fig_h))
    ax.errorbar(
        effects,
        y_pos,
        xerr=[err_left, err_right],
        fmt="o",
        color="#0ea5e9",
        ecolor="#334155",
        elinewidth=1.4,
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect on q_discharge_next (Ah) per +1pp substitution")
    ax.set_ylabel("Top bins")
    ax.set_title("Substitution Effect Forest Plot (Top bins vs rest pool)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def build_controlled_protocol(
    main_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Generate controlled experiment protocol and return selected bins."""

    m = main_df.copy()
    s = sens_df[["cross_bin", "effect_per_1pp_ah", "p_value", "skip_reason"]].rename(
        columns={
            "effect_per_1pp_ah": "effect_per_1pp_ah_sens",
            "p_value": "p_value_sens",
            "skip_reason": "skip_reason_sens",
        }
    )
    merged = m.merge(s, on="cross_bin", how="left", validate="one_to_one")
    merged["sign_main"] = np.sign(pd.to_numeric(merged["effect_per_1pp_ah"], errors="coerce"))
    merged["sign_sens"] = np.sign(pd.to_numeric(merged["effect_per_1pp_ah_sens"], errors="coerce"))
    merged["direction_consistent"] = (
        (merged["sign_main"] != 0)
        & (merged["sign_sens"] != 0)
        & (merged["sign_main"] == merged["sign_sens"])
    )
    merged["robust_flag"] = (
        (merged["skip_reason"] == "")
        & (merged["skip_reason_sens"].fillna("") == "")
        & (pd.to_numeric(merged["q_value"], errors="coerce") <= 0.1)
        & (pd.to_numeric(merged["p_value_sens"], errors="coerce") <= 0.1)
        & merged["direction_consistent"]
    )
    robust = merged[merged["robust_flag"]].copy()
    robust = robust.sort_values("abs_effect_per_1pp_ah", ascending=False, kind="mergesort")
    selected = robust.head(3).copy()
    if len(selected) < 3:
        fallback = (
            merged[merged["skip_reason"] == ""]
            .sort_values(["q_value", "abs_effect_per_1pp_ah"], ascending=[True, False], kind="mergesort")
            .head(3)
            .copy()
        )
        selected = fallback

    lines: List[str] = []
    lines.append("# 受控实验方案（自动生成）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 目的：验证“将充电时间份额从其余59区间替代到目标区间”是否改善下一循环放电容量。")
    lines.append("- 替代幅度：每个试验臂固定 `+5pp`（即 +0.05 时间份额）。")
    lines.append("")
    lines.append("## 1. 分组设计")
    lines.append("- 对照组：维持当前充电策略，不做份额替代。")
    for idx, row in enumerate(selected.itertuples(index=False), start=1):
        direction = "提升" if float(row.effect_per_1pp_ah) > 0 else "降低"
        lines.append(
            f"- 试验组{idx}：将总充电时间中 `+5pp` 替代到 `cross_bin={int(row.cross_bin)}`（{str(row.cross_label)}），"
            f"时间来源为其余59区间总池；预计方向：{direction}下一循环容量。"
        )
    lines.append("")
    lines.append("## 2. 执行约束")
    lines.append("- 保持每个cycle总充电时间不变，仅做区间份额重分配。")
    lines.append("- 满足现有SOC窗口与安全约束（温度、电流、截止条件）。")
    lines.append("- 若执行中触发安全阈值，立即回退到对照策略。")
    lines.append("")
    lines.append("## 3. 观测与终点")
    lines.append("- 建议观测窗口：每组至少30个连续cycle。")
    lines.append("- 主要终点：`q_discharge_{t+1}` 相对对照组的均值差。")
    lines.append("- 次要终点：30-cycle容量下降斜率、异常跳变事件率。")
    lines.append("")
    lines.append("## 4. 停止条件")
    lines.append("- 任一试验组出现连续3个cycle容量显著下降且伴随安全告警。")
    lines.append("- 关键安全指标超限（温度或电流保护触发）。")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return selected


def build_report(
    args: argparse.Namespace,
    fonts: Sequence[str],
    panel_stats: Dict[str, int],
    label_stats: Dict[str, int],
    dataset_df: pd.DataFrame,
    top_df: pd.DataFrame,
    main_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    overlap_ratio: float,
    selected_protocol_df: pd.DataFrame,
) -> str:
    """Build Chinese markdown report for full causal workflow."""

    checks = {
        "share_sum_min": float(dataset_df["share_sum"].min()),
        "share_sum_max": float(dataset_df["share_sum"].max()),
        "train_rows": int((dataset_df["set_type"] == "train").sum()),
        "valid_rows": int((dataset_df["set_type"] == "valid").sum()),
        "train_groups": int(dataset_df.loc[dataset_df["set_type"] == "train", "group_key"].nunique()),
        "valid_groups": int(dataset_df.loc[dataset_df["set_type"] == "valid", "group_key"].nunique()),
    }
    main_valid = main_df[main_df["skip_reason"] == ""].copy()
    sens_valid = sens_df[sens_df["skip_reason"] == ""].copy()

    direction_consistency = np.nan
    if not main_valid.empty and not sens_valid.empty:
        joined = main_valid[["cross_bin", "effect_per_1pp_ah"]].merge(
            sens_valid[["cross_bin", "effect_per_1pp_ah"]],
            on="cross_bin",
            suffixes=("_main", "_sens"),
            how="inner",
        )
        if not joined.empty:
            direction_consistency = float(
                np.mean(
                    np.sign(joined["effect_per_1pp_ah_main"].to_numpy(float))
                    == np.sign(joined["effect_per_1pp_ah_sens"].to_numpy(float))
                )
            )

    placebo_summary = np.nan
    if not main_valid.empty:
        placebo_summary = float(
            np.nanmean(np.abs(main_valid["placebo_mean_1pp_ah"].to_numpy(float)))
        )

    lines: List[str] = []
    lines.append("# 60区间替代效应因果分析报告")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 字体回退：`{', '.join(fonts)}`")
    lines.append(
        f"- 标签过滤区间：`{args.q_min} <= q_discharge <= {args.q_max}`，目标为下一循环容量 `q_discharge_(t+1)`。"
    )
    lines.append("")
    lines.append("## 2. 数据与校验")
    lines.append(
        f"- 时序原始行数/清洗后行数：`{panel_stats['timeseries_rows_raw']:,}` / `{panel_stats['timeseries_rows_after_clean']:,}`。"
    )
    lines.append(
        f"- cycle面板样本数（过滤 `cycle_total_charge_h<=0` 前/后）：`{panel_stats['panel_rows_before_positive_filter']:,}` / `{panel_stats['panel_rows_after_positive_filter']:,}`。"
    )
    lines.append(
        f"- 标签过滤剔除：`<q_min`={label_stats['label_rows_lt_qmin_removed']:,}，`>q_max`={label_stats['label_rows_gt_qmax_removed']:,}。"
    )
    lines.append(
        f"- share求和范围：`[{checks['share_sum_min']:.6f}, {checks['share_sum_max']:.6f}]`（目标区间 `[0.999,1.001]`）。"
    )
    lines.append(
        f"- train/valid 行数：`{checks['train_rows']:,}` / `{checks['valid_rows']:,}`；group数：`{checks['train_groups']}` / `{checks['valid_groups']}`。"
    )
    lines.append(
        f"- Top10稳定性（固定种子重复筛选重合率）：`{overlap_ratio:.2%}`。"
    )
    lines.append("")
    lines.append("## 3. 阶段一筛选（相关性 + 重要性）")
    lines.append("- 指标A：`share_i` 与 `q_(t+1)` 的 Spearman 绝对相关。")
    lines.append("- 指标B：RF permutation importance（60个share + 控制变量）。")
    lines.append("- 综合分数：A/B min-max归一化后取平均。")
    lines.append("")
    lines.append("| rank | cross_bin | cross_label | spearman_abs | perm_importance | combined_score |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    for row in top_df.itertuples(index=False):
        lines.append(
            f"| {int(row.rank_combined)} | {int(row.cross_bin)} | {str(row.cross_label)} | "
            f"{float(row.spearman_abs):.6f} | {float(row.rf_perm_importance):.6f} | {float(row.combined_score):.6f} |"
        )
    lines.append("")
    lines.append("## 4. 阶段二替代效应因果估计")
    lines.append(
        "- 定义：对每个Top10区间，估计“从其余59区间总池替代 `+1pp` 时间份额到该区间”对 `q_(t+1)` 的影响。"
    )
    lines.append("- 方法：按 `policy+cell` 分组的 5折 cross-fitting 残差化 DML，500次聚类bootstrap，BH-FDR校正。")
    lines.append("")
    lines.append("| cross_bin | cross_label | effect_per_1pp_ah | 95%CI | p_value | q_value |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in main_df.itertuples(index=False):
        if str(row.skip_reason) != "":
            lines.append(
                f"| {int(row.cross_bin)} | {str(row.cross_label)} | NaN | NaN | NaN | NaN |"
            )
        else:
            ci_text = f"[{float(row.ci_low):.6f}, {float(row.ci_high):.6f}]"
            lines.append(
                f"| {int(row.cross_bin)} | {str(row.cross_label)} | {float(row.effect_per_1pp_ah):.6f} | "
                f"{ci_text} | {float(row.p_value):.6f} | {float(row.q_value):.6f} |"
            )
    lines.append("")
    lines.append("## 5. 稳健性与负对照")
    if np.isfinite(direction_consistency):
        lines.append(f"- 剔除异常电芯后的方向一致率：`{direction_consistency:.2%}`。")
    else:
        lines.append("- 剔除异常电芯后的方向一致率：样本不足，未计算。")
    if np.isfinite(placebo_summary):
        lines.append(
            f"- 置换负对照（打乱 treatment residual）|平均绝对效应|：`{placebo_summary:.6e} Ah/1pp`（应接近0）。"
        )
    else:
        lines.append("- 置换负对照：样本不足，未计算。")
    lines.append("")
    lines.append("## 6. 受控实验建议摘要")
    if selected_protocol_df.empty:
        lines.append("- 未选出可执行试验区间，请检查主分析与敏感性结果。")
    else:
        for idx, row in enumerate(selected_protocol_df.itertuples(index=False), start=1):
            lines.append(
                f"- 建议试验臂{idx}：`cross_bin={int(row.cross_bin)}`（{str(row.cross_label)}），执行 `+5pp` 替代干预。"
            )
    lines.append("- 详细试验步骤见：`controlled_experiment_protocol.md`。")
    lines.append("")
    lines.append("## 7. 结论边界")
    lines.append("- 本结论属于“可操作优先级建议”，并非机制层面的最终证明。")
    lines.append("- 关键风险仍是未观测混杂，策略落地前需按受控实验方案验证。")
    return "\n".join(lines)


def main() -> None:
    """Run full pipeline: screening -> causal estimation -> protocol/report."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(args.random_seed))
    fonts = ensure_matplotlib_config()

    train_split, valid_split, split_map = load_split(
        args.train_split_path, args.valid_split_path
    )
    _ = train_split, valid_split
    panel_df, share_cols, panel_stats = load_timeseries_panel(
        timeseries_path=args.timeseries_path,
        chunksize=int(args.timeseries_chunksize),
    )
    label_df, label_stats = load_labels(
        life_path=args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
    )
    dataset_df = build_cycle_dataset(panel_df=panel_df, split_map=split_map, label_df=label_df)

    control_cols = [
        "q_t",
        "cycles",
        *POLICY_COLS,
        "cycle_total_charge_h",
        "nonzero_cross_bin_count_cycle",
        "is_abnormal_cell",
    ]
    dataset_df = dataset_df.dropna(subset=[*control_cols, "q_next"]).copy()
    dataset_df["cycles"] = dataset_df["cycles"].astype(int)
    dataset_df["nonzero_cross_bin_count_cycle"] = dataset_df[
        "nonzero_cross_bin_count_cycle"
    ].astype(int)
    dataset_df["is_abnormal_cell"] = dataset_df["is_abnormal_cell"].astype(int)

    train_df = dataset_df[dataset_df["set_type"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("No train rows after merge for screening stage.")

    bin_edges_df = pd.read_csv(args.bin_edges_path, encoding="utf-8")
    score_df, top_df, overlap_ratio = build_screening_tables(
        train_df=train_df,
        share_cols=share_cols,
        control_cols=control_cols,
        bin_edges_df=bin_edges_df,
        args=args,
    )

    main_causal = run_causal_estimation(
        df=dataset_df,
        top_bins=top_df,
        control_cols=control_cols,
        args=args,
    )
    sensitivity_df = dataset_df[dataset_df["is_abnormal_cell"] == 0].copy()
    sens_causal = run_causal_estimation(
        df=sensitivity_df,
        top_bins=top_df,
        control_cols=control_cols,
        args=args,
    )

    out_screen = args.output_dir / "screening_scores.csv"
    out_top = args.output_dir / "top10_bins.csv"
    out_main = args.output_dir / "causal_substitution_effects.csv"
    out_sens = args.output_dir / "causal_sensitivity_abnormal_excluded.csv"
    out_plot = args.output_dir / "effect_forest_plot.png"
    out_protocol = args.output_dir / "controlled_experiment_protocol.md"
    out_report = args.output_dir / "causal_report.md"

    score_df.to_csv(out_screen, index=False, encoding="utf-8")
    top_df.to_csv(out_top, index=False, encoding="utf-8")
    main_causal.to_csv(out_main, index=False, encoding="utf-8")
    sens_causal.to_csv(out_sens, index=False, encoding="utf-8")
    save_effect_forest_plot(main_causal, out_plot)
    selected_protocol_df = build_controlled_protocol(main_causal, sens_causal, out_protocol)

    report_text = build_report(
        args=args,
        fonts=fonts,
        panel_stats=panel_stats,
        label_stats=label_stats,
        dataset_df=dataset_df,
        top_df=top_df,
        main_df=main_causal,
        sens_df=sens_causal,
        overlap_ratio=overlap_ratio,
        selected_protocol_df=selected_protocol_df,
    )
    out_report.write_text(report_text, encoding="utf-8")

    print(f"Saved: {out_screen}")
    print(f"Saved: {out_top}")
    print(f"Saved: {out_main}")
    print(f"Saved: {out_sens}")
    print(f"Saved: {out_plot}")
    print(f"Saved: {out_protocol}")
    print(f"Saved: {out_report}")
    print(
        "Rows total/train/valid: "
        f"{len(dataset_df)}/{int((dataset_df['set_type']=='train').sum())}/{int((dataset_df['set_type']=='valid').sum())}"
    )
    print(
        "Top10 overlap ratio (same-seed rerun): "
        f"{overlap_ratio:.2%}"
    )


if __name__ == "__main__":
    main()
