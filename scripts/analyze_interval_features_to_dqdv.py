"""Analyze interval operation features against compact dQdV targets."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from train_interval_to_dqdv_retention_pipeline import (  # noqa: E402
    DQDV_TARGET_PACKS,
    ENCODING,
    POLICY_COLS,
    build_merged_cycle_table,
    load_charge_feature_table,
    load_dqdv_table,
    load_discharge_feature_table,
    load_retention_labels,
    load_split,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "interval_features_to_dqdv_correlation"
DEFAULT_CHARGE_PATH = REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv"
DEFAULT_DISCHARGE_PATH = REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv"
DEFAULT_DQDV_PATH = REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv"
DEFAULT_LIFE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_TRAIN_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv"
DEFAULT_VALID_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv"

TARGET_PACK_ALIASES: Mapping[str, str] = {
    "compact2": "compact2_area_height",
    "compact3": "compact3_area_height_voltage",
    "compact4": "compact_peak_shape_height_no_width",
}
COMPARISON_TARGET_PACKS: Mapping[str, List[str]] = {
    "compact2": ["main_peak_area", "main_peak_height_dqdv"],
    "compact3": ["main_peak_area", "main_peak_height_dqdv", "main_peak_voltage_v"],
    "compact4": ["main_peak_area", "main_peak_height_dqdv", "main_peak_voltage_v", "main_peak_skewness"],
}
DEFAULT_TARGET_ORDER = COMPARISON_TARGET_PACKS["compact4"]
EXCLUDED_INPUT_COLUMNS = {"cycles", "cycle_index_norm", *POLICY_COLS, "policy"}
EXPECTED_INPUT_FEATURE_DIM = 159
REDUNDANCY_THRESHOLD = 0.92
PSEUDO_GLOBAL_THRESHOLD = 0.45
PSEUDO_LOCAL_THRESHOLD = 0.12


@dataclass(frozen=True)
class ModelSpec:
    """Small regression model specification."""

    name: str
    estimator: Any
    needs_scaling: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze charge/discharge interval features against compact dQdV targets."
    )
    parser.add_argument("--charge-timeseries-path", type=Path, default=DEFAULT_CHARGE_PATH)
    parser.add_argument("--discharge-interval-path", type=Path, default=DEFAULT_DISCHARGE_PATH)
    parser.add_argument("--dqdv-path", type=Path, default=DEFAULT_DQDV_PATH)
    parser.add_argument("--life-path", type=Path, default=DEFAULT_LIFE_PATH)
    parser.add_argument("--train-split-path", type=Path, default=DEFAULT_TRAIN_SPLIT_PATH)
    parser.add_argument("--valid-split-path", type=Path, default=DEFAULT_VALID_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--input-feature-pack",
        type=str,
        choices=["charge_crossbin_discharge_capacity_stats"],
        default="charge_crossbin_discharge_capacity_stats",
    )
    parser.add_argument(
        "--target-pack",
        type=str,
        choices=sorted([*DQDV_TARGET_PACKS.keys(), *TARGET_PACK_ALIASES.keys()]),
        default="compact_peak_shape_height_no_width",
    )
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--min-cell-samples", type=int, default=8)
    parser.add_argument("--top-n-small", type=int, default=20)
    parser.add_argument("--top-n-medium", type=int, default=40)
    parser.add_argument("--max-train-rows-for-models", type=int, default=60000)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Read existing CSV outputs and regenerate only the Markdown report and session log.",
    )
    return parser.parse_args()


def ensure_matplotlib() -> None:
    """Configure matplotlib for headless plot output."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import rcParams

    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220


def resolve_target_pack(target_pack: str) -> Tuple[str, List[str]]:
    """Return canonical target pack name and ordered target columns."""

    canonical = TARGET_PACK_ALIASES.get(str(target_pack), str(target_pack))
    if canonical not in DQDV_TARGET_PACKS:
        raise ValueError(f"Unsupported target pack: {target_pack}")
    cols = [col for col in DEFAULT_TARGET_ORDER if col in DQDV_TARGET_PACKS[canonical]]
    return canonical, cols or list(DQDV_TARGET_PACKS[canonical])


def ordered_unique(values: Iterable[str]) -> List[str]:
    """Return values in first-seen order without duplicates."""

    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def safe_corr(method: Callable[[np.ndarray, np.ndarray], Any], x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute a correlation statistic and p-value without failing on degenerate arrays."""

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3 or len(y_clean) < 3:
        return float("nan"), float("nan")
    if float(np.nanstd(x_clean)) <= 0.0 or float(np.nanstd(y_clean)) <= 0.0:
        return float("nan"), float("nan")
    try:
        result = method(x_clean, y_clean)
        r = float(result.statistic if hasattr(result, "statistic") else result[0])
        p = float(result.pvalue if hasattr(result, "pvalue") else result[1])
        return (r, p) if np.isfinite(r) else (float("nan"), float("nan"))
    except Exception:
        return float("nan"), float("nan")


def corr_rows_for_scope(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    scope: str,
) -> List[Dict[str, Any]]:
    """Build Pearson, Spearman, and Kendall correlation rows for one data scope."""

    rows: List[Dict[str, Any]] = []
    for target in target_cols:
        y_all = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        for feature in feature_cols:
            x_all = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            pearson_r, pearson_p = safe_corr(stats.pearsonr, x_all, y_all)
            spearman_r, spearman_p = safe_corr(stats.spearmanr, x_all, y_all)
            kendall_tau, kendall_p = safe_corr(stats.kendalltau, x_all, y_all)
            rows.append(
                {
                    "scope": scope,
                    "target": target,
                    "feature": feature,
                    "n_rows": int((np.isfinite(x_all) & np.isfinite(y_all)).sum()),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "kendall_tau": kendall_tau,
                    "kendall_p": kendall_p,
                    "abs_spearman": abs(spearman_r) if np.isfinite(spearman_r) else float("nan"),
                }
            )
    return rows


def compute_global_correlations(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute global feature-target correlations."""

    return pd.DataFrame(corr_rows_for_scope(merged, feature_cols, target_cols, "all")).sort_values(
        ["target", "abs_spearman"],
        ascending=[True, False],
        kind="mergesort",
    )


def compute_split_correlations(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute train and valid feature-target correlations."""

    rows: List[Dict[str, Any]] = []
    for split_name, part in merged.groupby("set_type", sort=False):
        rows.extend(corr_rows_for_scope(part, feature_cols, target_cols, str(split_name)))
    return pd.DataFrame(rows).sort_values(["target", "feature", "scope"], kind="mergesort")


def compute_within_cell_correlations(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    min_cell_samples: int,
) -> pd.DataFrame:
    """Compute per-cell Spearman correlations and aggregate stability summaries."""

    rows: List[Dict[str, Any]] = []
    grouped = list(merged.groupby(["policy", "cell_code"], sort=False))
    for target in target_cols:
        for feature in feature_cols:
            cell_values: List[float] = []
            for _, part in grouped:
                if len(part) < int(min_cell_samples):
                    continue
                rho, _ = safe_corr(
                    stats.spearmanr,
                    pd.to_numeric(part[feature], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(part[target], errors="coerce").to_numpy(dtype=float),
                )
                if np.isfinite(rho):
                    cell_values.append(float(rho))
            arr = np.asarray(cell_values, dtype=float)
            if arr.size:
                q75, q25 = np.nanpercentile(arr, [75, 25])
                positive_rate = float(np.mean(arr > 0))
                negative_rate = float(np.mean(arr < 0))
                zero_rate = float(np.mean(arr == 0))
                sign_consistency = float(max(positive_rate, negative_rate, zero_rate))
                median = float(np.nanmedian(arr))
                abs_median = float(abs(median))
                iqr = float(q75 - q25)
            else:
                positive_rate = negative_rate = zero_rate = sign_consistency = float("nan")
                median = abs_median = iqr = float("nan")
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "n_cells": int(arr.size),
                    "within_cell_spearman_median": median,
                    "within_cell_abs_spearman_median": abs_median,
                    "within_cell_spearman_iqr": iqr,
                    "within_cell_positive_rate": positive_rate,
                    "within_cell_negative_rate": negative_rate,
                    "within_cell_zero_rate": zero_rate,
                    "within_cell_sign_consistency_rate": sign_consistency,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "within_cell_abs_spearman_median"],
        ascending=[True, False],
        kind="mergesort",
    )


def build_diff_frame(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Build first-difference rows within policy-cell sequences."""

    diff_parts: List[pd.DataFrame] = []
    cols = ["policy", "cell_code", "cycles", "set_type", *feature_cols, *target_cols]
    for _, part in merged[cols].groupby(["policy", "cell_code"], sort=False):
        work = part.sort_values("cycles", kind="mergesort").copy()
        if len(work) < 2:
            continue
        diff = work[[*feature_cols, *target_cols]].diff()
        diff.insert(0, "set_type", work["set_type"].to_numpy())
        diff.insert(0, "cycles", work["cycles"].to_numpy())
        diff.insert(0, "cell_code", work["cell_code"].to_numpy())
        diff.insert(0, "policy", work["policy"].to_numpy())
        diff_parts.append(diff.iloc[1:].copy())
    if not diff_parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(diff_parts, ignore_index=True)


def compute_diff_correlations(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute feature-target correlations on first differences."""

    diff_df = build_diff_frame(merged, feature_cols, target_cols)
    rows = corr_rows_for_scope(diff_df, feature_cols, target_cols, "first_difference")
    return pd.DataFrame(rows).sort_values(["target", "abs_spearman"], ascending=[True, False], kind="mergesort")


def compute_mutual_information(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    random_seed: int,
) -> pd.DataFrame:
    """Compute mutual information for every target-feature pair."""

    x_df = merged[list(feature_cols)].replace([np.inf, -np.inf], np.nan)
    x = SimpleImputer(strategy="median").fit_transform(x_df)
    rows: List[Dict[str, Any]] = []
    for target in target_cols:
        y = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 4:
            values = np.full(len(feature_cols), np.nan)
        else:
            values = mutual_info_regression(x[mask], y[mask], random_state=int(random_seed))
        finite = values[np.isfinite(values)]
        max_value = float(np.max(finite)) if finite.size else 0.0
        for feature, value in zip(feature_cols, values):
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "mutual_information": float(value) if np.isfinite(value) else float("nan"),
                    "mutual_information_norm": float(value / max_value) if max_value > 0 and np.isfinite(value) else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "mutual_information"],
        ascending=[True, False],
        kind="mergesort",
    )


def sign_direction(value: float) -> int:
    """Return a compact sign direction for a numeric value."""

    if not np.isfinite(value) or abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def build_stability_scores(
    global_corr: pd.DataFrame,
    split_corr: pd.DataFrame,
    within_corr: pd.DataFrame,
    diff_corr: pd.DataFrame,
    mi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge correlation summaries into one stability score table."""

    base = global_corr[["target", "feature", "spearman_r", "abs_spearman"]].rename(
        columns={"spearman_r": "global_spearman", "abs_spearman": "global_abs_spearman"}
    )
    split_wide = split_corr.pivot_table(
        index=["target", "feature"],
        columns="scope",
        values="spearman_r",
        aggfunc="first",
    ).reset_index()
    split_wide.columns.name = None
    split_wide = split_wide.rename(columns={"train": "train_spearman", "valid": "valid_spearman"})
    out = base.merge(split_wide, on=["target", "feature"], how="left")
    out = out.merge(within_corr, on=["target", "feature"], how="left")
    out = out.merge(
        diff_corr[["target", "feature", "spearman_r", "abs_spearman"]].rename(
            columns={"spearman_r": "diff_spearman", "abs_spearman": "diff_abs_spearman"}
        ),
        on=["target", "feature"],
        how="left",
    )
    out = out.merge(
        mi_df[["target", "feature", "mutual_information", "mutual_information_norm"]],
        on=["target", "feature"],
        how="left",
    )
    out["train_valid_direction_consistent"] = [
        int(sign_direction(t) != 0 and sign_direction(t) == sign_direction(v))
        for t, v in zip(out["train_spearman"], out["valid_spearman"])
    ]
    out["train_valid_abs_delta"] = (out["train_spearman"] - out["valid_spearman"]).abs()
    out["split_min_abs_spearman"] = np.fmin(out["train_spearman"].abs(), out["valid_spearman"].abs()).fillna(0.0)
    out["stability_score"] = (
        0.22 * out["global_abs_spearman"].fillna(0.0)
        + 0.22 * out["split_min_abs_spearman"].fillna(0.0)
        + 0.16 * out["within_cell_abs_spearman_median"].fillna(0.0)
        + 0.14 * out["diff_abs_spearman"].fillna(0.0)
        + 0.12 * out["within_cell_sign_consistency_rate"].fillna(0.0)
        + 0.10 * out["mutual_information_norm"].fillna(0.0)
        + 0.04 * out["train_valid_direction_consistent"].fillna(0.0)
        - 0.10 * out["train_valid_abs_delta"].fillna(1.0).clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0)
    out["suspected_aging_progress_proxy"] = (
        (out["global_abs_spearman"] >= PSEUDO_GLOBAL_THRESHOLD)
        & (out["within_cell_abs_spearman_median"].fillna(0.0) < PSEUDO_LOCAL_THRESHOLD)
        & (out["diff_abs_spearman"].fillna(0.0) < PSEUDO_LOCAL_THRESHOLD)
    )
    return out.sort_values(["target", "stability_score"], ascending=[True, False], kind="mergesort")


def compute_feature_redundancy(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    stability_scores: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group highly redundant features by absolute Spearman correlation."""

    corr = merged[list(feature_cols)].corr(method="spearman").abs().fillna(0.0)
    parent: Dict[str, str] = {feature: feature for feature in feature_cols}

    def find(name: str) -> str:
        """Find the representative parent for a feature."""

        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        """Union two redundancy groups."""

        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(feature_cols):
        for right in feature_cols[i + 1 :]:
            if float(corr.loc[left, right]) >= float(threshold):
                union(left, right)

    score_map = dict(
        zip(
            stability_scores.groupby("feature", as_index=False)["stability_score"].mean()["feature"],
            stability_scores.groupby("feature", as_index=False)["stability_score"].mean()["stability_score"],
        )
    )
    groups: Dict[str, List[str]] = {}
    for feature in feature_cols:
        groups.setdefault(find(feature), []).append(feature)
    group_rows: List[Dict[str, Any]] = []
    for group_index, features in enumerate(sorted(groups.values(), key=lambda values: (len(values), values[0]), reverse=True), start=1):
        representative = max(features, key=lambda name: (float(score_map.get(name, 0.0)), -feature_cols.index(name)))
        for feature in features:
            max_peer_corr = max([float(corr.loc[feature, peer]) for peer in features if peer != feature] or [0.0])
            group_rows.append(
                {
                    "redundancy_group_id": group_index,
                    "feature": feature,
                    "group_size": int(len(features)),
                    "representative_feature": representative,
                    "is_representative": int(feature == representative),
                    "mean_stability_score": float(score_map.get(feature, 0.0)),
                    "max_abs_spearman_to_group_peer": max_peer_corr,
                    "redundancy_threshold": float(threshold),
                }
            )
    return pd.DataFrame(group_rows).sort_values(
        ["redundancy_group_id", "is_representative", "mean_stability_score"],
        ascending=[True, False, False],
        kind="mergesort",
    ), corr


def select_top_features(
    stability_scores: pd.DataFrame,
    feature_corr: pd.DataFrame,
    target: str,
    top_n: int,
    redundancy_threshold: Optional[float] = None,
) -> List[str]:
    """Select top stable features, optionally skipping redundant candidates."""

    ranked = stability_scores.loc[stability_scores["target"] == target].sort_values(
        "stability_score",
        ascending=False,
        kind="mergesort",
    )
    selected: List[str] = []
    for feature in ranked["feature"].astype(str):
        if redundancy_threshold is not None:
            too_redundant = any(float(feature_corr.loc[feature, prev]) >= float(redundancy_threshold) for prev in selected)
            if too_redundant:
                continue
        selected.append(feature)
        if len(selected) >= int(top_n):
            break
    return selected


def build_recommendations(
    stability_scores: pd.DataFrame,
    feature_corr: pd.DataFrame,
    target_cols: Sequence[str],
    top_n_small: int,
    top_n_medium: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], List[str]]]:
    """Build target-wise and union recommended feature packs."""

    selection_map: Dict[Tuple[str, str], List[str]] = {}
    rows: List[Dict[str, Any]] = []
    for target in target_cols:
        pack_specs = [
            (f"top{top_n_small}_stable", select_top_features(stability_scores, feature_corr, target, top_n_small)),
            (f"top{top_n_medium}_stable", select_top_features(stability_scores, feature_corr, target, top_n_medium)),
            (
                "redundancy_pruned_top",
                select_top_features(stability_scores, feature_corr, target, top_n_medium, REDUNDANCY_THRESHOLD),
            ),
        ]
        for pack_name, features in pack_specs:
            selection_map[(target, pack_name)] = features
            target_scores = stability_scores.loc[stability_scores["target"] == target].set_index("feature")
            for rank, feature in enumerate(features, start=1):
                score = float(target_scores.loc[feature, "stability_score"]) if feature in target_scores.index else float("nan")
                rows.append({"target": target, "feature_pack": pack_name, "rank": rank, "feature": feature, "stability_score": score})
    per_target = pd.DataFrame(rows)
    union_features = ordered_unique(
        per_target.loc[per_target["feature_pack"] == "redundancy_pruned_top", "feature"].astype(str).tolist()
    )
    union_rows = [{"rank": rank, "feature": feature} for rank, feature in enumerate(union_features, start=1)]
    return per_target, pd.DataFrame(union_rows), selection_map


def make_model_specs(random_seed: int) -> List[ModelSpec]:
    """Create the small model suite used for predictability checks."""

    return [
        ModelSpec("ridge", Ridge(alpha=1.0), True),
        ModelSpec("elasticnet", ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=8000, random_state=int(random_seed)), True),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=int(random_seed),
            ),
            False,
        ),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.06,
                l2_regularization=0.02,
                random_state=int(random_seed),
            ),
            False,
        ),
    ]


def make_pipeline(spec: ModelSpec) -> Pipeline:
    """Build a preprocessing and model pipeline."""

    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.estimator))
    return Pipeline(steps)


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    feature_pack: str,
    model_name: str,
    set_type: str,
    n_features: int,
) -> Dict[str, Any]:
    """Build one regression metric output row."""

    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "target": target,
        "feature_pack": feature_pack,
        "model_name": model_name,
        "set_type": set_type,
        "n_rows": int(len(y_true)),
        "n_features": int(n_features),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def sample_train_rows(train_df: pd.DataFrame, max_rows: int, random_seed: int) -> pd.DataFrame:
    """Return a reproducible model-fitting subset if the training table is large."""

    if int(max_rows) <= 0 or len(train_df) <= int(max_rows):
        return train_df
    return train_df.sample(n=int(max_rows), random_state=int(random_seed)).sort_index()


def run_predictability_models(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    selection_map: Mapping[Tuple[str, str], List[str]],
    random_seed: int,
    max_train_rows: int,
    top_n_small: int,
    top_n_medium: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train small models and score train/valid predictability."""

    train_df = merged.loc[merged["set_type"] == "train"].copy()
    valid_df = merged.loc[merged["set_type"] == "valid"].copy()
    fit_df = sample_train_rows(train_df, max_train_rows, random_seed)
    specs = make_model_specs(random_seed)
    metric_rows: List[Dict[str, Any]] = []
    top_feature_rows: List[Dict[str, Any]] = []
    for target in target_cols:
        feature_sets: Dict[str, List[str]] = {
            "full_feature_pack": list(feature_cols),
            f"top{top_n_small}_stable": list(selection_map[(target, f"top{top_n_small}_stable")]),
            f"top{top_n_medium}_stable": list(selection_map[(target, f"top{top_n_medium}_stable")]),
            "redundancy_pruned_top": list(selection_map[(target, "redundancy_pruned_top")]),
        }
        for pack_name, cols in feature_sets.items():
            if not cols:
                continue
            x_fit = fit_df[cols]
            y_fit = pd.to_numeric(fit_df[target], errors="coerce").to_numpy(dtype=float)
            x_train = train_df[cols]
            y_train = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
            x_valid = valid_df[cols]
            y_valid = pd.to_numeric(valid_df[target], errors="coerce").to_numpy(dtype=float)
            for spec in specs:
                model = make_pipeline(spec)
                model.fit(x_fit, y_fit)
                train_pred = model.predict(x_train)
                valid_pred = model.predict(x_valid)
                metric_rows.append(metric_row(y_train, train_pred, target, pack_name, spec.name, "train", len(cols)))
                metric_rows.append(metric_row(y_valid, valid_pred, target, pack_name, spec.name, "valid", len(cols)))
                estimator = model.named_steps["model"]
                importances = getattr(estimator, "feature_importances_", None)
                coefs = getattr(estimator, "coef_", None)
                if importances is not None:
                    values = np.asarray(importances, dtype=float)
                    kind = "feature_importance"
                elif coefs is not None:
                    values = np.abs(np.asarray(coefs, dtype=float)).reshape(-1)
                    kind = "abs_coefficient"
                else:
                    values = np.full(len(cols), np.nan)
                    kind = "not_available"
                order = np.argsort(np.nan_to_num(values, nan=-1.0))[::-1][:20]
                for rank, idx in enumerate(order, start=1):
                    top_feature_rows.append(
                        {
                            "target": target,
                            "feature_pack": pack_name,
                            "model_name": spec.name,
                            "rank": rank,
                            "feature": cols[int(idx)],
                            "importance_type": kind,
                            "importance_value": float(values[int(idx)]) if np.isfinite(values[int(idx)]) else float("nan"),
                        }
                    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target", "set_type", "feature_pack", "r2"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    return metrics, pd.DataFrame(top_feature_rows)


def build_dataset_checks(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    charge_stats: Mapping[str, int],
    discharge_stats: Mapping[str, int],
    input_feature_pack: str,
    target_pack: str,
) -> pd.DataFrame:
    """Build basic dataset contract checks for the analysis run."""

    forbidden_present = sorted(set(feature_cols).intersection(EXCLUDED_INPUT_COLUMNS))
    rows = [
        {
            "check_item": "input_feature_pack",
            "value": input_feature_pack,
            "pass": int(input_feature_pack == "charge_crossbin_discharge_capacity_stats"),
            "details": "must use charge_crossbin_discharge_capacity_stats",
        },
        {
            "check_item": "input_feature_dim",
            "value": int(len(feature_cols)),
            "pass": int(len(feature_cols) == EXPECTED_INPUT_FEATURE_DIM),
            "details": "expected 60 charge cumulative + 60 charge increment + 16 discharge inc + 16 discharge cum + 7 summary stats",
        },
        {
            "check_item": "excluded_input_columns_present",
            "value": ";".join(forbidden_present),
            "pass": int(not forbidden_present),
            "details": "cycles, cycle_index_norm, policy labels and policy numeric parameters must not be input features",
        },
        {"check_item": "target_pack", "value": target_pack, "pass": 1, "details": ",".join(target_cols)},
        {"check_item": "merged_cycle_rows", "value": int(len(merged)), "pass": int(len(merged) > 0), "details": ""},
        {
            "check_item": "train_cycle_rows",
            "value": int((merged["set_type"] == "train").sum()),
            "pass": int((merged["set_type"] == "train").sum() > 0),
            "details": "",
        },
        {
            "check_item": "valid_cycle_rows",
            "value": int((merged["set_type"] == "valid").sum()),
            "pass": int((merged["set_type"] == "valid").sum() > 0),
            "details": "",
        },
        {
            "check_item": "policy_cell_count",
            "value": int(merged[["policy", "cell_code"]].drop_duplicates().shape[0]),
            "pass": int(merged[["policy", "cell_code"]].drop_duplicates().shape[0] > 0),
            "details": "",
        },
        {
            "check_item": "charge_cross_bin_feature_dim",
            "value": int(charge_stats.get("charge_cross_bin_feature_dim", 0)),
            "pass": int(charge_stats.get("charge_cross_bin_feature_dim", 0) == 60),
            "details": "",
        },
        {
            "check_item": "discharge_range_count",
            "value": int(discharge_stats.get("discharge_range_count", 0)),
            "pass": int(discharge_stats.get("discharge_range_count", 0) == 16),
            "details": "",
        },
    ]
    return pd.DataFrame(rows)


def write_plots(
    output_dir: Path,
    stability_scores: pd.DataFrame,
    global_corr: pd.DataFrame,
    predictability_metrics: pd.DataFrame,
    target_cols: Sequence[str],
) -> None:
    """Write the requested summary plots."""

    ensure_matplotlib()
    import matplotlib.pyplot as plt

    top_features = ordered_unique(
        stability_scores.sort_values("stability_score", ascending=False, kind="mergesort")["feature"].head(40).astype(str)
    )
    heat = global_corr.loc[global_corr["feature"].isin(top_features)].pivot_table(
        index="feature",
        columns="target",
        values="spearman_r",
        aggfunc="first",
    )
    heat = heat.reindex(index=top_features, columns=list(target_cols)).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(heat))))
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=6)
    ax.set_title("Top stable feature Spearman correlation")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap_top_features.png")
    plt.close(fig)

    fig, axes = plt.subplots(len(target_cols), 1, figsize=(10, max(4, 2.8 * len(target_cols))), squeeze=False)
    for row_idx, target in enumerate(target_cols):
        ax = axes[row_idx][0]
        part = stability_scores.loc[stability_scores["target"] == target].head(15).iloc[::-1]
        ax.barh(part["feature"], part["stability_score"], color="#4c78a8")
        ax.set_title(f"{target} top stable features")
        ax.set_xlabel("stability score")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "target_feature_barplots.png")
    plt.close(fig)

    valid = predictability_metrics.loc[predictability_metrics["set_type"] == "valid"].copy()
    best = valid.sort_values("r2", ascending=False, kind="mergesort").groupby(["target", "feature_pack"], as_index=False).head(1)
    packs = ["full_feature_pack", "top20_stable", "top40_stable", "redundancy_pruned_top"]
    x = np.arange(len(target_cols))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, pack in enumerate(packs):
        vals = []
        for target in target_cols:
            row = best.loc[(best["target"] == target) & (best["feature_pack"] == pack)]
            vals.append(float(row["r2"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=pack)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(target_cols, rotation=20, ha="right")
    ax.set_ylabel("best valid R2")
    ax.set_title("Predictability comparison by feature pack")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "predictability_r2_comparison.png")
    plt.close(fig)


def best_valid_metrics(predictability_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return best valid metric rows per target and feature pack."""

    valid = predictability_metrics.loc[predictability_metrics["set_type"] == "valid"].copy()
    return (
        valid.sort_values("r2", ascending=False, kind="mergesort")
        .groupby(["target", "feature_pack"], as_index=False)
        .head(1)
        .sort_values(["target", "r2"], ascending=[True, False], kind="mergesort")
    )


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int) -> str:
    """Render a compact Markdown table."""

    if df.empty:
        return "\n无。\n"
    work = df.loc[:, list(columns)].head(int(max_rows)).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            work[col] = work[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in work.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in work.itertuples(index=False, name=None):
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n" + "\n".join(lines) + "\n"


def choose_target_pack_recommendation(best_valid: pd.DataFrame, stability_scores: pd.DataFrame) -> Tuple[str, str]:
    """Choose compact2/compact3/compact4 recommendation from predictability and stability evidence."""

    rows: List[Dict[str, Any]] = []
    for pack_name, targets in COMPARISON_TARGET_PACKS.items():
        valid_rows = best_valid.loc[best_valid["target"].isin(targets)]
        r2_median = float(valid_rows["r2"].median()) if not valid_rows.empty else float("nan")
        score_rows = stability_scores.loc[stability_scores["target"].isin(targets)]
        top_stability = (
            score_rows.sort_values("stability_score", ascending=False, kind="mergesort")
            .groupby("target", as_index=False)
            .head(10)["stability_score"]
        )
        stability_mean = float(top_stability.mean()) if not top_stability.empty else float("nan")
        complexity_penalty = {"compact2": 0.00, "compact3": 0.02, "compact4": 0.04}[pack_name]
        combined = np.nan_to_num(r2_median, nan=-1.0) + 0.25 * np.nan_to_num(stability_mean, nan=0.0) - complexity_penalty
        rows.append(
            {
                "pack_name": pack_name,
                "combined": float(combined),
                "r2_median": r2_median,
                "stability_mean": stability_mean,
            }
        )
    ranking = pd.DataFrame(rows).sort_values("combined", ascending=False, kind="mergesort")
    winner = str(ranking.iloc[0]["pack_name"])
    reason = (
        f"{winner} 在比较分数中最高；median valid R2={ranking.iloc[0]['r2_median']:.4f}，"
        f"Top稳定性均值={ranking.iloc[0]['stability_mean']:.4f}。"
    )
    return winner, reason


def write_report(
    output_dir: Path,
    dataset_checks: pd.DataFrame,
    stability_scores: pd.DataFrame,
    best_valid: pd.DataFrame,
    recommended_union: pd.DataFrame,
    target_cols: Sequence[str],
) -> None:
    """Write the Markdown analysis report."""

    winner, reason = choose_target_pack_recommendation(best_valid, stability_scores)
    lines: List[str] = [
        "# 159维工况特征 -> dQdV 相关性与可预测性分析报告",
        "",
        "## 结论摘要",
        "",
        f"- 推荐后续优先 target pack：`{winner}`。{reason}",
        f"- 推荐后续 feature pack：使用 `recommended_feature_pack_union.csv` 的去冗余 union 特征包，当前包含 {len(recommended_union)} 个特征。",
        "- 输入侧已排除 `cycles`、`cycle_index_norm`、policy 标签与 policy 三元参数。",
        "",
        "## 数据检查",
        markdown_table(dataset_checks, ["check_item", "value", "pass", "details"], 20),
    ]
    for target in target_cols:
        target_scores = stability_scores.loc[stability_scores["target"] == target].copy()
        lines.extend(
            [
                "",
                f"## {target}",
                "",
                "### 最相关且最稳定的工况特征",
                markdown_table(
                    target_scores,
                    [
                        "feature",
                        "stability_score",
                        "global_spearman",
                        "train_spearman",
                        "valid_spearman",
                        "within_cell_spearman_median",
                        "diff_spearman",
                        "mutual_information_norm",
                    ],
                    12,
                ),
                "### 疑似老化进度伪相关",
                markdown_table(
                    target_scores.loc[target_scores["suspected_aging_progress_proxy"]],
                    ["feature", "global_spearman", "within_cell_spearman_median", "diff_spearman", "stability_score"],
                    10,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Top特征包 vs 全量特征包",
            "",
            "下表展示每个 target 和 feature pack 的最佳 valid 指标，模型在 `ridge`、`elasticnet`、`random_forest`、`hist_gradient_boosting` 中择优。",
            markdown_table(best_valid, ["target", "feature_pack", "model_name", "n_features", "mse", "rmse", "mae", "r2"], 80),
            "",
            "## compact2/compact3/compact4 判断",
            "",
            f"- 当前推荐：`{winner}`。",
            "- 若目标是最小可解释闭环，`compact2` 成本最低，只覆盖面积与高度。",
            "- 若希望把主峰电压漂移纳入桥接，`compact3` 是复杂度和信息量的折中。",
            "- 若 `main_peak_skewness` 的 valid R2 与稳定性没有明显拖累，`compact4` 更适合保留峰形非对称信息；反之应先用 compact3。",
        ]
    )
    (output_dir / "interval_features_to_dqdv_correlation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_session_log(output_dir: Path, summary: str) -> None:
    """Append this run summary to the required daily session log."""

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "session_2026-05-09.md"
    text = (
        "\n\n## Prompt\n"
        "执行工况特征 -> dQdV 相关性与可预测性分析；用户补充要求文件名不显式包含维度数字。\n\n"
        "## Response\n"
        f"{summary}\n"
        f"输出目录：`{output_dir}`\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def run_report_only(args: argparse.Namespace) -> None:
    """Regenerate the report from existing output CSV files."""

    dataset_checks = pd.read_csv(args.output_dir / "dataset_checks.csv", encoding=ENCODING)
    stability_scores = pd.read_csv(args.output_dir / "feature_stability_score_by_target.csv", encoding=ENCODING)
    predictability_metrics = pd.read_csv(args.output_dir / "predictability_metrics_by_target.csv", encoding=ENCODING)
    recommended_union = pd.read_csv(args.output_dir / "recommended_feature_pack_union.csv", encoding=ENCODING)
    target_row = dataset_checks.loc[dataset_checks["check_item"] == "target_pack"]
    if target_row.empty:
        _, target_cols = resolve_target_pack(args.target_pack)
    else:
        details = str(target_row["details"].iloc[0])
        target_cols = [col for col in details.split(",") if col]
    best_valid = best_valid_metrics(predictability_metrics)
    write_report(args.output_dir, dataset_checks, stability_scores, best_valid, recommended_union, target_cols)
    write_session_log(
        args.output_dir,
        (
            "基于已生成 CSV 补写 Markdown 报告；"
            f"targets={','.join(target_cols)}，"
            f"recommended_feature_count={len(recommended_union)}。"
        ),
    )
    print(f"Report regenerated: {args.output_dir / 'interval_features_to_dqdv_correlation_report.md'}")


def main() -> None:
    """Run the full analysis workflow."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        run_report_only(args)
        return

    canonical_target_pack, target_cols = resolve_target_pack(args.target_pack)

    _, _, split_map = load_split(args.train_split_path, args.valid_split_path)
    charge_df, charge_stats, charge_cols = load_charge_feature_table(args.charge_timeseries_path, None)
    discharge_df, discharge_stats, discharge_cols = load_discharge_feature_table(args.discharge_interval_path, None)
    dqdv_df = load_dqdv_table(args.dqdv_path, target_cols)
    label_df = load_retention_labels(
        args.life_path,
        args.q_min,
        args.q_max,
        args.q_ref_cycles,
        args.retention_min,
        args.retention_max,
    )
    merged, feature_cols, _ = build_merged_cycle_table(
        split_map=split_map,
        charge_df=charge_df,
        charge_cols=charge_cols,
        discharge_df=discharge_df,
        discharge_cols=discharge_cols,
        dqdv_df=dqdv_df,
        label_df=label_df,
        target_cols=target_cols,
        input_feature_pack=args.input_feature_pack,
        input_scaling="none",
    )
    forbidden_present = sorted(set(feature_cols).intersection(EXCLUDED_INPUT_COLUMNS))
    if forbidden_present:
        raise RuntimeError(f"Forbidden input columns selected: {forbidden_present}")
    if len(feature_cols) != EXPECTED_INPUT_FEATURE_DIM:
        raise RuntimeError(f"Expected {EXPECTED_INPUT_FEATURE_DIM} input features, got {len(feature_cols)}.")

    dataset_checks = build_dataset_checks(
        merged,
        feature_cols,
        target_cols,
        charge_stats,
        discharge_stats,
        args.input_feature_pack,
        canonical_target_pack,
    )
    dataset_checks.to_csv(args.output_dir / "dataset_checks.csv", index=False, encoding=ENCODING)
    pd.DataFrame({"feature": feature_cols}).to_csv(args.output_dir / "feature_columns.csv", index=False, encoding=ENCODING)
    target_rows = []
    for pack_name, cols in COMPARISON_TARGET_PACKS.items():
        for rank, col in enumerate(cols, start=1):
            target_rows.append({"target_pack": pack_name, "rank": rank, "target": col})
    pd.DataFrame(target_rows).to_csv(args.output_dir / "target_columns.csv", index=False, encoding=ENCODING)
    merged[["policy", "cell_code", "cycles", "set_type", *feature_cols, *target_cols]].head(500).to_csv(
        args.output_dir / "merged_interval_features_dqdv_sample.csv",
        index=False,
        encoding=ENCODING,
    )

    global_corr = compute_global_correlations(merged, feature_cols, target_cols)
    split_corr = compute_split_correlations(merged, feature_cols, target_cols)
    within_corr = compute_within_cell_correlations(merged, feature_cols, target_cols, args.min_cell_samples)
    diff_corr = compute_diff_correlations(merged, feature_cols, target_cols)
    mi_df = compute_mutual_information(merged, feature_cols, target_cols, args.random_seed)
    stability_scores = build_stability_scores(global_corr, split_corr, within_corr, diff_corr, mi_df)
    redundancy_groups, feature_corr = compute_feature_redundancy(merged, feature_cols, stability_scores, REDUNDANCY_THRESHOLD)
    recommended_by_target, recommended_union, selection_map = build_recommendations(
        stability_scores,
        feature_corr,
        target_cols,
        args.top_n_small,
        args.top_n_medium,
    )
    predictability_metrics, predictability_top_features = run_predictability_models(
        merged,
        feature_cols,
        target_cols,
        selection_map,
        args.random_seed,
        args.max_train_rows_for_models,
        args.top_n_small,
        args.top_n_medium,
    )

    global_corr.to_csv(args.output_dir / "global_correlation_by_target.csv", index=False, encoding=ENCODING)
    split_corr.to_csv(args.output_dir / "split_correlation_by_target.csv", index=False, encoding=ENCODING)
    within_corr.to_csv(args.output_dir / "within_cell_correlation_by_target.csv", index=False, encoding=ENCODING)
    diff_corr.to_csv(args.output_dir / "diff_correlation_by_target.csv", index=False, encoding=ENCODING)
    mi_df.to_csv(args.output_dir / "mutual_information_by_target.csv", index=False, encoding=ENCODING)
    stability_scores.to_csv(args.output_dir / "feature_stability_score_by_target.csv", index=False, encoding=ENCODING)
    redundancy_groups.to_csv(args.output_dir / "feature_redundancy_groups.csv", index=False, encoding=ENCODING)
    recommended_by_target.to_csv(args.output_dir / "recommended_feature_pack_by_target.csv", index=False, encoding=ENCODING)
    recommended_union.to_csv(args.output_dir / "recommended_feature_pack_union.csv", index=False, encoding=ENCODING)
    predictability_metrics.to_csv(args.output_dir / "predictability_metrics_by_target.csv", index=False, encoding=ENCODING)
    predictability_top_features.to_csv(args.output_dir / "predictability_top_features.csv", index=False, encoding=ENCODING)

    best_valid = best_valid_metrics(predictability_metrics)
    write_plots(args.output_dir, stability_scores, global_corr, predictability_metrics, target_cols)
    write_report(args.output_dir, dataset_checks, stability_scores, best_valid, recommended_union, target_cols)
    write_session_log(
        args.output_dir,
        (
            f"完成分析脚本运行；input_feature_dim={len(feature_cols)}，"
            f"targets={','.join(target_cols)}，merged_rows={len(merged)}。"
        ),
    )
    print(f"Analysis complete: {args.output_dir}")
    print(f"input_feature_dim={len(feature_cols)}")
    print(f"target_cols={','.join(target_cols)}")
    print(f"merged_cycle_rows={len(merged)}")


if __name__ == "__main__":
    main()
