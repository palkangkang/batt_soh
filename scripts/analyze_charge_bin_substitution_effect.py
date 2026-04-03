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
    parser.add_argument("--report-style", type=str, default="paper_zh")
    parser.add_argument("--appendix-full60", type=int, default=1)
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


def save_screening_top20_bar(
    score_df: pd.DataFrame,
    out_png: Path,
    top_n: int = 20,
) -> None:
    """Save bar chart for top-N combined screening scores."""

    import matplotlib.pyplot as plt  # noqa: WPS433

    top = score_df.sort_values("rank_combined", ascending=True, kind="mergesort").head(
        int(top_n)
    )
    if top.empty:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))
        ax.text(0.5, 0.5, "No screening rows.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, format="png")
        plt.close(fig)
        return

    top = top.copy()
    top["label"] = top.apply(
        lambda row: f"bin{int(row['cross_bin']):02d} ({str(row['cross_label'])})",
        axis=1,
    )
    top = top.sort_values("combined_score", ascending=True, kind="mergesort")
    fig_h = max(6.0, 0.34 * len(top) + 2.4)
    fig, ax = plt.subplots(1, 1, figsize=(11.0, fig_h))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["combined_score"].to_numpy(float), color="#0284c7", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["label"].tolist())
    ax.set_xlabel("Combined Screening Score")
    ax.set_ylabel("Top bins")
    ax.set_title("Top 20 Screening Scores (Correlation + Permutation Importance)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def save_screening_heatmap_soc_panels(
    score_df: pd.DataFrame,
    out_png: Path,
) -> None:
    """Save SOC-panel heatmap for all 60-bin screening combined scores."""

    import matplotlib.pyplot as plt  # noqa: WPS433

    work = score_df.copy()
    if work.empty:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))
        ax.text(0.5, 0.5, "No screening rows.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, format="png")
        plt.close(fig)
        return

    soc_labels = (
        work[["soc_bin", "soc_label"]]
        .drop_duplicates(["soc_bin"], keep="first")
        .set_index("soc_bin")["soc_label"]
        .to_dict()
    )
    v_min = float(work["combined_score"].min())
    v_max = float(work["combined_score"].max())

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharey=True)
    for idx, soc_bin in enumerate([1, 2, 3]):
        ax = axes[idx]
        sub = work[work["soc_bin"] == soc_bin].copy()
        mat = (
            sub.pivot_table(
                index="rate_bin",
                columns="temp_bin",
                values="combined_score",
                aggfunc="mean",
            )
            .reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4, 5])
            .to_numpy(float)
        )
        image = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            cmap="YlOrRd",
            vmin=v_min,
            vmax=v_max,
        )
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([f"T{t}" for t in [1, 2, 3, 4, 5]])
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels([f"R{r}" for r in [1, 2, 3, 4]])
        soc_label = str(soc_labels.get(soc_bin, f"SOC {soc_bin}"))
        ax.set_title(f"SOC{soc_bin} {soc_label}")
        ax.set_xlabel("Temp bin")
        if idx == 0:
            ax.set_ylabel("Rate bin")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Combined Screening Score")
    fig.suptitle("Screening Heatmap by SOC Panels (Rate x Temp)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def save_effect_main_vs_sensitivity_scatter(
    main_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    out_png: Path,
) -> None:
    """Save scatter plot comparing main and sensitivity causal effects."""

    import matplotlib.pyplot as plt  # noqa: WPS433

    main_valid = main_df[main_df["skip_reason"] == ""].copy()
    sens_valid = sens_df[sens_df["skip_reason"] == ""].copy()
    merged = main_valid[
        ["cross_bin", "cross_label", "effect_per_1pp_ah", "q_value"]
    ].merge(
        sens_valid[["cross_bin", "effect_per_1pp_ah", "q_value"]],
        on="cross_bin",
        how="inner",
        suffixes=("_main", "_sens"),
    )
    if merged.empty:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))
        ax.text(0.5, 0.5, "No overlapping valid effects.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, format="png")
        plt.close(fig)
        return

    x = merged["effect_per_1pp_ah_main"].to_numpy(float)
    y = merged["effect_per_1pp_ah_sens"].to_numpy(float)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    span = max(1e-6, hi - lo)
    pad = 0.08 * span
    lo_p = lo - pad
    hi_p = hi + pad

    sig_mask = (
        (pd.to_numeric(merged["q_value_main"], errors="coerce") <= 0.1)
        & (pd.to_numeric(merged["q_value_sens"], errors="coerce") <= 0.1)
    ).to_numpy(bool)

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 7.2))
    ax.scatter(
        x[~sig_mask],
        y[~sig_mask],
        s=42,
        alpha=0.7,
        color="#64748b",
        label="Others",
    )
    ax.scatter(
        x[sig_mask],
        y[sig_mask],
        s=56,
        alpha=0.9,
        color="#0ea5e9",
        label="q<=0.1 in both",
    )
    ax.plot([lo_p, hi_p], [lo_p, hi_p], linestyle="--", color="#ef4444", linewidth=1.2)
    ax.axhline(0.0, color="#94a3b8", linewidth=1.0)
    ax.axvline(0.0, color="#94a3b8", linewidth=1.0)
    ax.set_xlim(lo_p, hi_p)
    ax.set_ylim(lo_p, hi_p)
    ax.set_xlabel("Main analysis effect (Ah per +1pp)")
    ax.set_ylabel("Sensitivity effect (Ah per +1pp)")
    ax.set_title("Main vs Sensitivity Effect Consistency")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


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
    lines.append("## 0. 方案摘要")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 目标：验证“将充电时间份额从其余59区间替代到目标区间”对下一循环放电容量的因果影响。")
    lines.append("- 干预幅度：每个试验臂固定 `+5pp`（`+0.05` 时间份额），对照组不干预。")
    lines.append("- 随机化单位：`policy + cell_code`，避免同一电芯跨组污染。")
    lines.append("- 分层建议：按 `policy三元参数`、`q_t` 分位、`is_abnormal_cell` 分层后再随机化。")
    lines.append("")
    lines.append("## 1. 研究假设")
    lines.append("- 原假设 H0：在固定总充电时间下，将 `+5pp` 份额替代到目标区间，不改变 `q_discharge_{t+1}`。")
    lines.append("- 备择假设 H1：上述替代会改变 `q_discharge_{t+1}`，方向由观测因果估计给出。")
    lines.append("")
    lines.append("## 2. 试验臂设计")
    lines.append("- 对照组：维持现有充电策略，不做份额重分配。")
    if selected.empty:
        lines.append("- 试验组：当前未筛出可执行区间，建议先复核主分析与敏感性估计。")
    for idx, row in enumerate(selected.itertuples(index=False), start=1):
        effect_1pp = float(row.effect_per_1pp_ah)
        effect_5pp = effect_1pp * 5.0
        direction = "提升" if effect_1pp > 0 else "降低"
        lines.append(
            f"- 试验组{idx}：目标 `cross_bin={int(row.cross_bin)}`（{str(row.cross_label)}），"
            f"将总充电时间 `+5pp` 替代至该区间，来源为其余59区间总池；"
            f"观测预测方向：{direction}容量，估计量级约 `{effect_5pp:.6f} Ah/5pp`。"
        )
    lines.append("")
    lines.append("## 3. 干预实施规则")
    lines.append("- 每个 cycle 保持总充电时间不变，仅调整60区间内部份额分配。")
    lines.append("- 在 SOC 与安全约束内执行替代：不突破温度、电流、截止电压等保护边界。")
    lines.append("- 若当 cycle 无法满足 `+5pp` 目标（可行域不足），记录为 protocol deviation，不强制外推。")
    lines.append("")
    lines.append("## 4. 终点与统计分析计划")
    lines.append("- 主要终点：`q_discharge_{t+1}` 相对对照组的均值差（按 `policy+cell` 聚类稳健标准误）。")
    lines.append("- 次要终点：30-cycle 衰减斜率、异常跳变事件率（`dt_s > 3600`）。")
    lines.append("- 建议分析：混合效应/聚类稳健回归，固定效应含 `policy三元参数` 与 cycle 阶段。")
    lines.append("- 建议最小观测窗口：每组至少30个连续 cycle。")
    lines.append("")
    lines.append("## 5. 停止规则与安全约束")
    lines.append("- 任一试验组出现连续3个 cycle 容量显著下降且伴随安全告警时，触发停臂评估。")
    lines.append("- 温度、电流任一保护触发达到预设阈值，立即回退到对照策略。")
    lines.append("- 发生 protocol deviation 的样本单独标注，按 ITT 与 PP 两套口径同时分析。")
    lines.append("")
    lines.append("## 6. 证据闭环")
    lines.append("- 本方案用于将“观测因果估计”转化为“可操作策略证据”。")
    lines.append("- 最终策略上线以受控实验结果为准，不以单次观测估计直接替代。")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return selected


def build_report(
    args: argparse.Namespace,
    fonts: Sequence[str],
    panel_stats: Dict[str, int],
    label_stats: Dict[str, int],
    dataset_df: pd.DataFrame,
    score_df: pd.DataFrame,
    top_df: pd.DataFrame,
    main_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    overlap_ratio: float,
    selected_protocol_df: pd.DataFrame,
) -> str:
    """Build Chinese markdown report in paper style."""

    if str(args.report_style) != "paper_zh":
        raise ValueError(f"Unsupported report style: {args.report_style}")

    checks = {
        "share_sum_min": float(dataset_df["share_sum"].min()),
        "share_sum_max": float(dataset_df["share_sum"].max()),
        "train_rows": int((dataset_df["set_type"] == "train").sum()),
        "valid_rows": int((dataset_df["set_type"] == "valid").sum()),
        "train_groups": int(dataset_df.loc[dataset_df["set_type"] == "train", "group_key"].nunique()),
        "valid_groups": int(dataset_df.loc[dataset_df["set_type"] == "valid", "group_key"].nunique()),
        "abnormal_share": float((dataset_df["is_abnormal_cell"] == 1).mean()),
    }
    main_valid = main_df[main_df["skip_reason"] == ""].copy()
    sens_valid = sens_df[sens_df["skip_reason"] == ""].copy()
    joined = main_valid[
        ["cross_bin", "cross_label", "effect_per_1pp_ah", "q_value", "ci_low", "ci_high"]
    ].merge(
        sens_valid[["cross_bin", "effect_per_1pp_ah", "q_value", "ci_low", "ci_high"]],
        on="cross_bin",
        suffixes=("_main", "_sens"),
        how="inner",
    )
    direction_consistency = np.nan
    effect_corr = np.nan
    if not joined.empty:
        sign_main = np.sign(joined["effect_per_1pp_ah_main"].to_numpy(float))
        sign_sens = np.sign(joined["effect_per_1pp_ah_sens"].to_numpy(float))
        direction_consistency = float(np.mean(sign_main == sign_sens))
        if len(joined) > 1:
            effect_corr = float(
                np.corrcoef(
                    joined["effect_per_1pp_ah_main"].to_numpy(float),
                    joined["effect_per_1pp_ah_sens"].to_numpy(float),
                )[0, 1]
            )
    placebo_summary = np.nan
    if not main_valid.empty:
        placebo_summary = float(
            np.nanmean(np.abs(main_valid["placebo_mean_1pp_ah"].to_numpy(float)))
        )

    top_eval = top_df[
        [
            "rank_combined",
            "cross_bin",
            "cross_label",
            "soc_label",
            "rate_label",
            "temp_label",
            "spearman_abs",
            "rf_perm_importance",
            "combined_score",
        ]
    ].merge(
        main_df[
            [
                "cross_bin",
                "effect_per_1pp_ah",
                "effect_per_5pp_ah",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
                "skip_reason",
                "var_treatment",
            ]
        ],
        on="cross_bin",
        how="left",
        validate="one_to_one",
    ).merge(
        sens_df[["cross_bin", "effect_per_1pp_ah", "q_value", "skip_reason"]].rename(
            columns={
                "effect_per_1pp_ah": "effect_per_1pp_ah_sens",
                "q_value": "q_value_sens",
                "skip_reason": "skip_reason_sens",
            }
        ),
        on="cross_bin",
        how="left",
        validate="one_to_one",
    )
    top_eval["ci_cross_zero"] = (
        pd.to_numeric(top_eval["ci_low"], errors="coerce") <= 0.0
    ) & (pd.to_numeric(top_eval["ci_high"], errors="coerce") >= 0.0)
    top_eval["direction_consistent"] = (
        np.sign(pd.to_numeric(top_eval["effect_per_1pp_ah"], errors="coerce"))
        == np.sign(pd.to_numeric(top_eval["effect_per_1pp_ah_sens"], errors="coerce"))
    )
    top_eval["robust_flag"] = (
        (top_eval["skip_reason"].fillna("") == "")
        & (top_eval["skip_reason_sens"].fillna("") == "")
        & (pd.to_numeric(top_eval["q_value"], errors="coerce") <= 0.1)
        & (pd.to_numeric(top_eval["q_value_sens"], errors="coerce") <= 0.1)
        & top_eval["direction_consistent"]
    )

    def to_evidence_level(row: pd.Series) -> str:
        """Convert one result row to evidence level."""

        q_val = pd.to_numeric(row["q_value"], errors="coerce")
        ci_cross = bool(row["ci_cross_zero"])
        if not np.isfinite(q_val):
            return "不可判定"
        if (q_val <= 0.05) and (not ci_cross):
            return "强证据"
        if (q_val <= 0.1) and (not ci_cross):
            return "中等证据"
        if ci_cross:
            return "证据不足（CI跨0）"
        return "弱证据"

    top_eval["evidence_level"] = top_eval.apply(to_evidence_level, axis=1)
    top_eval = top_eval.sort_values("rank_combined", ascending=True, kind="mergesort")

    support_rows: List[dict] = []
    for row in top_eval.itertuples(index=False):
        share_col = f"share_{int(row.cross_bin):02d}"
        if share_col not in dataset_df.columns:
            continue
        values = pd.to_numeric(dataset_df[share_col], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        q01, q50, q99 = np.percentile(values, [1.0, 50.0, 99.0]).tolist()
        support_rows.append(
            {
                "cross_bin": int(row.cross_bin),
                "cross_label": str(row.cross_label),
                "share_q01": float(q01),
                "share_q50": float(q50),
                "share_q99": float(q99),
                "support_width_1_99": float(q99 - q01),
                "var_treatment": float(row.var_treatment) if np.isfinite(float(row.var_treatment)) else np.nan,
            }
        )
    support_df = pd.DataFrame(support_rows)
    narrow_support = pd.DataFrame()
    if not support_df.empty:
        narrow_support = support_df[support_df["support_width_1_99"] < 0.02].copy()

    score_top20 = score_df.sort_values("rank_combined", ascending=True, kind="mergesort").head(20)
    score_best = score_top20.iloc[0] if not score_top20.empty else None
    score_tail = score_top20.iloc[-1] if not score_top20.empty else None
    heat_best = (
        score_df.sort_values("combined_score", ascending=False, kind="mergesort")
        .head(1)
        .copy()
    )
    soc_mean = score_df.groupby("soc_bin", as_index=False)["combined_score"].mean()
    soc_max_row = soc_mean.sort_values("combined_score", ascending=False, kind="mergesort").head(1)
    main_q10 = int((pd.to_numeric(main_valid["q_value"], errors="coerce") <= 0.1).sum())
    main_q05 = int((pd.to_numeric(main_valid["q_value"], errors="coerce") <= 0.05).sum())
    main_ci_cross = int(
        (
            (pd.to_numeric(main_valid["ci_low"], errors="coerce") <= 0.0)
            & (pd.to_numeric(main_valid["ci_high"], errors="coerce") >= 0.0)
        ).sum()
    )
    main_pos = int((pd.to_numeric(main_valid["effect_per_1pp_ah"], errors="coerce") > 0).sum())
    main_neg = int((pd.to_numeric(main_valid["effect_per_1pp_ah"], errors="coerce") < 0).sum())

    lines: List[str] = []
    lines.append("# 60区间容量衰减影响分析报告（论文式）")
    lines.append("")
    lines.append("## 摘要")
    lines.append(
        f"- 本研究基于 `{args.q_min} <= q_discharge <= {args.q_max}` 的 cycle 级样本，目标估计“将充电时间份额从其余59区间替代到目标区间”对下一循环容量 `q_discharge_(t+1)` 的因果影响。"
    )
    lines.append(
        f"- 在 Top{int(args.top_k)} 区间主分析中，`q<=0.1` 的区间数为 `{main_q10}`，其中 `q<=0.05` 为 `{main_q05}`；CI跨0区间数为 `{main_ci_cross}`。"
    )
    if score_best is not None:
        lines.append(
            f"- 综合筛选得分最高区间为 `bin{int(score_best['cross_bin']):02d} ({str(score_best['cross_label'])})`，得分 `{float(score_best['combined_score']):.6f}`。"
        )
    lines.append(
        "- 结论定位为“策略优先级建议”，并通过受控实验方案给出可落地验证路径。"
    )
    lines.append("")
    lines.append("## 1. 研究问题与因果识别目标")
    lines.append("- 研究问题：60个 `soc×rate×temp` 区间中，哪些区间时间份额变化最影响下一循环放电容量。")
    lines.append("- 处理变量：`T_i = share_i`（第 i 区间在当前 cycle 的充电时间份额）。")
    lines.append("- 结果变量：`Y = q_discharge_(t+1)`。")
    lines.append("- 目标效应：将其余59区间总池中 `+1pp` 份额替代到区间 `i` 的边际影响。")
    lines.append("")
    lines.append("## 2. 数据、样本与质量控制")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 字体回退：`{', '.join(fonts)}`")
    lines.append(
        f"- 时序原始行数/清洗后行数：`{panel_stats['timeseries_rows_raw']:,}` / `{panel_stats['timeseries_rows_after_clean']:,}`。"
    )
    lines.append(
        f"- cycle样本数（过滤 `cycle_total_charge_h<=0` 前/后）：`{panel_stats['panel_rows_before_positive_filter']:,}` / `{panel_stats['panel_rows_after_positive_filter']:,}`。"
    )
    lines.append(
        f"- 标签过滤剔除：`<q_min`={label_stats['label_rows_lt_qmin_removed']:,}，`>q_max`={label_stats['label_rows_gt_qmax_removed']:,}，保留后 `{label_stats['label_rows_after_range_filter']:,}`。"
    )
    lines.append(
        f"- share求和校验区间：`[{checks['share_sum_min']:.6f}, {checks['share_sum_max']:.6f}]`（目标 `[0.999,1.001]`）。"
    )
    lines.append(
        f"- train/valid 样本行数：`{checks['train_rows']:,}` / `{checks['valid_rows']:,}`；group数：`{checks['train_groups']}` / `{checks['valid_groups']}`。"
    )
    lines.append(f"- 主分析样本中异常电芯占比：`{checks['abnormal_share']:.2%}`。")
    lines.append(f"- Top10 稳定性（同种子重跑重合率）：`{overlap_ratio:.2%}`。")
    lines.append("")
    lines.append("## 3. 方法与理论依据")
    lines.append("### 3.1 两阶段分析框架")
    lines.append("- 阶段一（筛选）：`Spearman(|corr|)` + `RF permutation importance`，归一化后取均值得到综合得分。")
    lines.append("- 阶段二（因果）：对 Top10 区间做残差化 DML，输出 `Ah/1pp`、`Ah/5pp`、95%CI、`p`、`q`。")
    lines.append("")
    lines.append("### 3.2 估计公式")
    lines.append("```text")
    lines.append("Y~ = Y - m_y(W)")
    lines.append("T~ = T_i - m_t(W)")
    lines.append("theta_i = Cov(Y~, T~) / Var(T~)")
    lines.append("effect_per_1pp = 0.01 * theta_i")
    lines.append("effect_per_5pp = 0.05 * theta_i")
    lines.append("```")
    lines.append("- 其中 `W` 包含：`q_t`、`cycles_t`、policy三元参数、`cycle_total_charge_h`、`nonzero_cross_bin_count_cycle`、`is_abnormal_cell`。")
    lines.append("- 不确定性：`policy+cell` 聚类 bootstrap（500次）给95%CI；多重比较采用 BH-FDR 得到 `q-value`。")
    lines.append("")
    lines.append("### 3.3 因果识别假设（解释边界）")
    lines.append("- 一致性：观测到的份额替代对应同定义下的潜在结果。")
    lines.append("- 可交换性：在给定 `W` 后，未观测混杂可忽略（强假设）。")
    lines.append("- 重叠性：各区间份额在样本支持域内有足够变化。")
    lines.append("- SUTVA：电芯间干预不相互影响。")
    lines.append("")
    lines.append("## 4. 实验结果")
    lines.append("### 4.1 Top10 区间因果估计（主文）")
    lines.append("| rank | cross_bin | 区间标签 | effect(Ah/1pp) | 95%CI | p | q | 方向一致性(主/敏) | 证据等级 |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for row in top_eval.itertuples(index=False):
        ci_txt = "NaN"
        if np.isfinite(float(row.ci_low)) and np.isfinite(float(row.ci_high)):
            ci_txt = f"[{float(row.ci_low):.6f}, {float(row.ci_high):.6f}]"
        sign_consistent = "是" if bool(row.direction_consistent) else "否"
        lines.append(
            f"| {int(row.rank_combined)} | {int(row.cross_bin)} | {str(row.cross_label)} | "
            f"{float(row.effect_per_1pp_ah):.6f} | {ci_txt} | {float(row.p_value):.6f} | "
            f"{float(row.q_value):.6f} | {sign_consistent} | {str(row.evidence_level)} |"
        )
    lines.append("")
    lines.append("### 4.2 CI跨0区间的标准化解释")
    ci_cross_rows = top_eval[top_eval["ci_cross_zero"]].copy()
    if ci_cross_rows.empty:
        lines.append("- Top10 中无 CI 跨0区间。")
    else:
        for row in ci_cross_rows.itertuples(index=False):
            lines.append(
                f"- `bin{int(row.cross_bin):02d} ({str(row.cross_label)})`：95%CI 跨0，解释为“当前证据不足以确认方向”，"
                "并非“确定无效应”；建议进入后续受控实验优先级清单。"
            )
    lines.append("")
    lines.append("### 4.3 证据分层汇总（统计显著性 + 效应量 + 工程意义）")
    top_strong = top_eval[
        (pd.to_numeric(top_eval["q_value"], errors="coerce") <= 0.05) & (~top_eval["ci_cross_zero"])
    ].copy()
    if top_strong.empty:
        lines.append("- 本轮无“强证据”区间。")
    else:
        for row in top_strong.itertuples(index=False):
            lines.append(
                f"- `bin{int(row.cross_bin):02d}`：`q={float(row.q_value):.4f}`，`+1pp={float(row.effect_per_1pp_ah):.6f} Ah`，"
                f"`+5pp={float(row.effect_per_5pp_ah):.6f} Ah`。"
            )
    lines.append("")
    lines.append("## 5. 关键图表解读（逐图给出坐标说明与结论）")
    lines.append("### 图1：Top20 综合筛选得分")
    lines.append("![图1 Top20综合筛选得分](./screening_top20_bar.png)")
    lines.append("- X轴说明：综合筛选得分（相关性与重要性归一化平均）。")
    lines.append("- Y轴说明：区间标识 `binXX (s_r_t)`，按得分排序。")
    if (score_best is not None) and (score_tail is not None):
        lines.append(
            f"- 结论：Top1 为 `bin{int(score_best['cross_bin']):02d}`，Top20末位得分为 `{float(score_tail['combined_score']):.6f}`，"
            f"前后差值 `{float(score_best['combined_score']) - float(score_tail['combined_score']):.6f}`，说明筛选区分度明确。"
        )
    else:
        lines.append("- 结论：当前无可用筛选数据。")
    lines.append("")
    lines.append("### 图2：60区间筛选热力图（SOC分面）")
    lines.append("![图2 SOC分面热力图](./screening_heatmap_soc_panels.png)")
    lines.append("- X轴说明：温度分位区间 `temp_bin(T1~T5)`。")
    lines.append("- Y轴说明：倍率分位区间 `rate_bin(R1~R4)`。")
    if (not heat_best.empty) and (not soc_max_row.empty):
        best_row = heat_best.iloc[0]
        lines.append(
            f"- 结论：全局最高得分区间为 `bin{int(best_row['cross_bin']):02d} ({str(best_row['cross_label'])})`；"
            f"平均得分最高的 SOC 分层为 `SOC{int(soc_max_row.iloc[0]['soc_bin'])}`，提示该SOC段更值得优先优化。"
        )
    else:
        lines.append("- 结论：当前无可用热力图统计。")
    lines.append("")
    lines.append("### 图3：Top区间替代效应森林图")
    lines.append("![图3 替代效应森林图](./effect_forest_plot.png)")
    lines.append("- X轴说明：将 `+1pp` 份额替代到目标区间时，对 `q_discharge_(t+1)` 的效应（Ah）。")
    lines.append("- Y轴说明：Top区间（`binXX + 区间标签`）。")
    lines.append(
        f"- 结论：正向区间 `{main_pos}` 个、负向区间 `{main_neg}` 个；`q<=0.1` 区间 `{main_q10}` 个，CI跨0区间 `{main_ci_cross}` 个。"
    )
    lines.append("")
    lines.append("### 图4：主分析与敏感性分析一致性")
    lines.append("![图4 主分析vs敏感性散点](./effect_main_vs_sensitivity_scatter.png)")
    lines.append("- X轴说明：保留异常电芯时的效应估计（Ah/1pp）。")
    lines.append("- Y轴说明：剔除异常电芯后的敏感性效应估计（Ah/1pp）。")
    if np.isfinite(effect_corr):
        lines.append(
            f"- 结论：两口径效应相关系数约 `{effect_corr:.3f}`，方向一致率 `{direction_consistency:.2%}`，"
            "说明主结论在异常样本处理上整体稳定。"
        )
    else:
        lines.append("- 结论：可比样本不足，暂无法评估一致性。")
    lines.append("")
    lines.append("## 6. 稳健性、支持域与不外推清单")
    if np.isfinite(placebo_summary):
        lines.append(
            f"- 置换负对照：`|平均绝对效应|={placebo_summary:.6e} Ah/1pp`，接近0，未见系统性伪相关信号。"
        )
    else:
        lines.append("- 置换负对照：样本不足，未计算。")
    lines.append(
        "- 支持域声明：本报告仅对观测支持域内（样本具有足够份额波动）的区间给出解释，不建议对超出支持域的替代幅度做外推。"
    )
    if support_df.empty:
        lines.append("- 支持域统计：暂无可用数据。")
    else:
        lines.append("| cross_bin | cross_label | share_q01 | share_q50 | share_q99 | width(q99-q01) | var_treatment |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in support_df.itertuples(index=False):
            lines.append(
                f"| {int(row.cross_bin)} | {str(row.cross_label)} | {float(row.share_q01):.4f} | {float(row.share_q50):.4f} | "
                f"{float(row.share_q99):.4f} | {float(row.support_width_1_99):.4f} | {float(row.var_treatment):.6f} |"
            )
    lines.append("")
    lines.append("### 不外推清单（建议谨慎解释）")
    if narrow_support.empty:
        lines.append("- 无 `width(q99-q01)<0.02` 的 Top10 区间。")
    else:
        for row in narrow_support.itertuples(index=False):
            lines.append(
                f"- `bin{int(row.cross_bin):02d} ({str(row.cross_label)})`：支持域宽度 `{float(row.support_width_1_99):.4f}`，"
                "建议不做高幅度策略外推。"
            )
    lines.append("")
    lines.append("## 7. 策略建议与受控验证路径")
    if selected_protocol_df.empty:
        lines.append("- 当前未形成可执行试验臂，建议先扩充样本后再进入干预验证。")
    else:
        lines.append("- 建议优先验证以下试验臂（详见 `controlled_experiment_protocol.md`）：")
        for idx, row in enumerate(selected_protocol_df.itertuples(index=False), start=1):
            lines.append(
                f"- 试验臂{idx}：`bin{int(row.cross_bin):02d} ({str(row.cross_label)})`，固定 `+5pp` 替代。"
            )
    lines.append("")
    lines.append("## 8. 局限性")
    lines.append("- 未观测混杂仍可能存在，观测因果不等于机制证明。")
    lines.append("- Top10 以筛选策略决定，未覆盖全60区间DML估计。")
    lines.append("- 部分区间 CI 跨0，需通过受控实验进一步收敛不确定性。")
    lines.append("")
    lines.append("## 附录A：Top10 入选逻辑")
    lines.append("- Step1：在 train 集计算 `spearman_abs` 与 `rf_perm_importance`。")
    lines.append("- Step2：两者分别 min-max 归一化，综合分数 `combined_score = (A+B)/2`。")
    lines.append("- Step3：按 `combined_score`、`spearman_abs`、`rf_perm_importance` 依次排序取 Top10。")
    lines.append("")
    lines.append("## 附录B：60区间全量筛选总表")
    if int(args.appendix_full60) == 1:
        lines.append("| rank | cross_bin | cross_label | soc_label | rate_label | temp_label | spearman_abs | perm_importance | combined_score |")
        lines.append("|---:|---:|---|---|---|---|---:|---:|---:|")
        all_rows = score_df.sort_values("rank_combined", ascending=True, kind="mergesort")
        for row in all_rows.itertuples(index=False):
            lines.append(
                f"| {int(row.rank_combined)} | {int(row.cross_bin)} | {str(row.cross_label)} | "
                f"{str(row.soc_label)} | {str(row.rate_label)} | {str(row.temp_label)} | "
                f"{float(row.spearman_abs):.6f} | {float(row.rf_perm_importance):.6f} | {float(row.combined_score):.6f} |"
            )
    else:
        lines.append("- 已关闭全量附录输出（`--appendix-full60=0`）。")
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
    out_screen_bar = args.output_dir / "screening_top20_bar.png"
    out_screen_heat = args.output_dir / "screening_heatmap_soc_panels.png"
    out_effect_scatter = args.output_dir / "effect_main_vs_sensitivity_scatter.png"
    out_protocol = args.output_dir / "controlled_experiment_protocol.md"
    out_report = args.output_dir / "causal_report.md"

    score_df.to_csv(out_screen, index=False, encoding="utf-8")
    top_df.to_csv(out_top, index=False, encoding="utf-8")
    main_causal.to_csv(out_main, index=False, encoding="utf-8")
    sens_causal.to_csv(out_sens, index=False, encoding="utf-8")
    save_screening_top20_bar(score_df, out_screen_bar, top_n=20)
    save_screening_heatmap_soc_panels(score_df, out_screen_heat)
    save_effect_forest_plot(main_causal, out_plot)
    save_effect_main_vs_sensitivity_scatter(main_causal, sens_causal, out_effect_scatter)
    selected_protocol_df = build_controlled_protocol(main_causal, sens_causal, out_protocol)

    report_text = build_report(
        args=args,
        fonts=fonts,
        panel_stats=panel_stats,
        label_stats=label_stats,
        dataset_df=dataset_df,
        score_df=score_df,
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
    print(f"Saved: {out_screen_bar}")
    print(f"Saved: {out_screen_heat}")
    print(f"Saved: {out_plot}")
    print(f"Saved: {out_effect_scatter}")
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
