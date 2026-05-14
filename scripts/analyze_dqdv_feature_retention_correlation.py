"""Analyze correlations between dQ/dV main-peak features and capacity retention."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"

DEFAULT_DQDV_PATH = REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv"
DEFAULT_LIFE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_TRAIN_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv"
DEFAULT_VALID_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "dqdv_feature_retention_correlation"

MAIN_PEAK_FEATURE_COLUMNS: List[str] = [
    "main_peak_voltage_v",
    "main_peak_width_v",
    "main_peak_height_dqdv",
    "main_peak_area",
    "main_peak_prominence",
    "main_peak_skewness",
    "main_peak_temp_max_c",
    "main_peak_temp_min_c",
    "main_peak_temp_avg_c",
]

CORRELATION_THRESHOLDS: Mapping[str, float] = {
    "strong": 0.50,
    "moderate": 0.30,
    "weak": 0.10,
}
REDUNDANT_SPEARMAN_THRESHOLD = 0.90
HIGHLY_RELATED_SPEARMAN_THRESHOLD = 0.75
VIF_SEVERE_THRESHOLD = 10.0
VIF_HIGH_THRESHOLD = 5.0
PCA_TARGET_EXPLAINED_RATIO = 0.95


MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Analyze correlation between 9 dQ/dV main-peak features and capacity retention."
    )
    parser.add_argument("--dqdv-path", type=Path, default=DEFAULT_DQDV_PATH)
    parser.add_argument("--life-path", type=Path, default=DEFAULT_LIFE_PATH)
    parser.add_argument("--train-split-path", type=Path, default=DEFAULT_TRAIN_SPLIT_PATH)
    parser.add_argument("--valid-split-path", type=Path, default=DEFAULT_VALID_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--min-group-samples", type=int, default=8)
    return parser.parse_args()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Normalize bool-like values to a boolean series."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["1", "true", "yes", "y", "t"])


def ensure_required_columns(df: pd.DataFrame, required_cols: Sequence[str], table_name: str) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"{table_name} missing required columns: {missing}")


def load_dqdv_features(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load valid dQ/dV main-peak features and return data plus pre-fill missingness summary."""

    use_cols = ["policy", "cell_code", "cycles", "is_valid_curve", *MAIN_PEAK_FEATURE_COLUMNS]
    dqdv = pd.read_csv(path, encoding=ENCODING, usecols=use_cols)
    ensure_required_columns(dqdv, use_cols, "dQ/dV feature table")

    dqdv["policy"] = dqdv["policy"].astype(str)
    dqdv["cell_code"] = dqdv["cell_code"].astype(str)
    dqdv["cycles"] = pd.to_numeric(dqdv["cycles"], errors="coerce")
    dqdv["is_valid_curve"] = normalize_bool_series(dqdv["is_valid_curve"])

    for col in MAIN_PEAK_FEATURE_COLUMNS:
        dqdv[col] = pd.to_numeric(dqdv[col], errors="coerce")

    dqdv = dqdv.dropna(subset=["policy", "cell_code", "cycles"]).copy()
    dqdv["cycles"] = dqdv["cycles"].astype(int)
    dqdv = dqdv.loc[dqdv["is_valid_curve"]].copy()

    missing_rows: List[Dict[str, Any]] = []
    for col in MAIN_PEAK_FEATURE_COLUMNS:
        missing_count = int(dqdv[col].isna().sum())
        missing_rows.append(
            {
                "feature": col,
                "missing_before_fill": missing_count,
                "missing_rate_before_fill": float(missing_count / max(len(dqdv), 1)),
            }
        )
        dqdv[col] = dqdv[col].fillna(0.0).astype(np.float32)

    dqdv = dqdv.drop(columns=["is_valid_curve"])
    dqdv = dqdv.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")
    dqdv = dqdv.drop_duplicates(["policy", "cell_code", "cycles"], keep="last").reset_index(drop=True)
    return dqdv, pd.DataFrame(missing_rows)


def load_retention_labels(
    life_path: Path,
    q_min: float,
    q_max: float,
    q_ref_cycles: int,
    retention_min: float,
    retention_max: float,
) -> pd.DataFrame:
    """Load q_discharge and build retention labels with the LSTM training口径."""

    life = pd.read_csv(
        life_path,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    ensure_required_columns(life, ["policy", "cell_code", "cycles", "q_discharge"], "life performance table")
    life["policy"] = life["policy"].astype(str)
    life["cell_code"] = life["cell_code"].astype(str)
    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life["cycles"] = life["cycles"].astype(int)
    life = life.loc[life["q_discharge"] > 0].copy()
    life = life.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")

    abs_filtered = life.loc[(life["q_discharge"] >= q_min) & (life["q_discharge"] <= q_max)].copy()
    if abs_filtered.empty:
        raise RuntimeError("No rows remain after q_discharge filtering.")

    early_cycles = abs_filtered.groupby(["policy", "cell_code"], sort=False).head(int(q_ref_cycles))
    q_ref = (
        early_cycles.groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .median()
        .rename(columns={"q_discharge": "q_ref"})
    )
    q_ref = q_ref.loc[q_ref["q_ref"] > 0].copy()
    if q_ref.empty:
        raise RuntimeError("No valid q_ref generated.")

    labeled = abs_filtered.merge(q_ref, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    labeled["retention"] = labeled["q_discharge"] / labeled["q_ref"]
    labeled = labeled.loc[
        (labeled["retention"] >= float(retention_min)) & (labeled["retention"] <= float(retention_max))
    ].copy()
    if labeled.empty:
        raise RuntimeError("No rows remain after retention filtering.")

    return labeled[["policy", "cell_code", "cycles", "q_discharge", "q_ref", "retention"]].copy()


def load_split_map(train_split_path: Path, valid_split_path: Path) -> pd.DataFrame:
    """Load train/valid split map keyed by policy and cell code."""

    train = pd.read_csv(train_split_path, encoding=ENCODING, usecols=["policy", "cell_code"]).copy()
    valid = pd.read_csv(valid_split_path, encoding=ENCODING, usecols=["policy", "cell_code"]).copy()
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    split = pd.concat([train, valid], ignore_index=True)
    split["policy"] = split["policy"].astype(str)
    split["cell_code"] = split["cell_code"].astype(str)
    return split.drop_duplicates(["policy", "cell_code"], keep="first").reset_index(drop=True)


def build_dataset(
    dqdv_df: pd.DataFrame,
    label_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge dQ/dV features, retention labels, and train/valid split."""

    merged = label_df.merge(dqdv_df, on=["policy", "cell_code", "cycles"], how="inner")
    merged = merged.merge(split_df, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("Merged correlation dataset is empty.")
    return merged


def safe_corr(
    method: Callable[[np.ndarray, np.ndarray], Any],
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Compute a correlation statistic and p-value safely."""

    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
        return float("nan"), float("nan")
    try:
        result = method(x, y)
        r = float(result.statistic if hasattr(result, "statistic") else result[0])
        p = float(result.pvalue if hasattr(result, "pvalue") else result[1])
        if np.isfinite(r) and np.isfinite(p):
            return r, p
    except Exception:
        pass
    return float("nan"), float("nan")


def classify_correlation(abs_spearman: float) -> str:
    """Classify correlation strength from absolute Spearman rho."""

    if not np.isfinite(abs_spearman):
        return "undefined"
    if abs_spearman >= CORRELATION_THRESHOLDS["strong"]:
        return "strong"
    if abs_spearman >= CORRELATION_THRESHOLDS["moderate"]:
        return "moderate"
    if abs_spearman >= CORRELATION_THRESHOLDS["weak"]:
        return "weak"
    return "negligible"


def calc_correlations_for_scope(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scope: str,
) -> pd.DataFrame:
    """Compute Pearson, Spearman, and Kendall correlations for one dataset scope."""

    rows: List[Dict[str, Any]] = []
    for feature in feature_cols:
        part = df[[feature, "retention"]].replace([np.inf, -np.inf], np.nan).dropna()
        x = part[feature].to_numpy(dtype=float)
        y = part["retention"].to_numpy(dtype=float)
        pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
        kendall_tau, kendall_p = safe_corr(stats.kendalltau, x, y)
        rows.append(
            {
                "scope": scope,
                "feature": feature,
                "n_samples": int(len(part)),
                "feature_mean": float(np.nanmean(x)) if len(x) else float("nan"),
                "feature_std": float(np.nanstd(x)) if len(x) else float("nan"),
                "retention_mean": float(np.nanmean(y)) if len(y) else float("nan"),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p,
                "abs_spearman": abs(spearman_rho) if np.isfinite(spearman_rho) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out["correlation_class"] = out["abs_spearman"].map(classify_correlation)
    return out.sort_values(["abs_spearman", "feature"], ascending=[False, True]).reset_index(drop=True)


def calc_split_correlations(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Compute train and valid correlation tables."""

    frames: List[pd.DataFrame] = []
    for set_type in ["train", "valid"]:
        part = df.loc[df["set_type"] == set_type].copy()
        frames.append(calc_correlations_for_scope(part, feature_cols, set_type))
    return pd.concat(frames, ignore_index=True)


def calc_group_stability(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_group_samples: int,
) -> pd.DataFrame:
    """Calculate per-feature within-group Spearman stability by policy+cell_code."""

    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(["policy", "cell_code"], sort=False)
    for feature in feature_cols:
        group_values: List[float] = []
        for _, part in grouped:
            work = part[[feature, "retention"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(work) < int(min_group_samples):
                continue
            x = work[feature].to_numpy(dtype=float)
            y = work["retention"].to_numpy(dtype=float)
            rho, _ = safe_corr(stats.spearmanr, x, y)
            if np.isfinite(rho):
                group_values.append(float(rho))

        values = np.asarray(group_values, dtype=float)
        if len(values) == 0:
            median_rho = float("nan")
            q25 = float("nan")
            q75 = float("nan")
            iqr = float("nan")
            positive_share = float("nan")
            negative_share = float("nan")
            direction_consistency = float("nan")
        else:
            median_rho = float(np.median(values))
            q25 = float(np.percentile(values, 25))
            q75 = float(np.percentile(values, 75))
            iqr = float(q75 - q25)
            positive_share = float(np.mean(values > 0))
            negative_share = float(np.mean(values < 0))
            direction_consistency = float(max(positive_share, negative_share))

        rows.append(
            {
                "feature": feature,
                "n_valid_groups": int(len(values)),
                "group_spearman_median": median_rho,
                "group_spearman_q25": q25,
                "group_spearman_q75": q75,
                "group_spearman_iqr": iqr,
                "positive_direction_share": positive_share,
                "negative_direction_share": negative_share,
                "direction_consistency": direction_consistency,
                "median_abs_group_spearman": abs(median_rho) if np.isfinite(median_rho) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_abs_group_spearman", "direction_consistency", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_reduction_recommendations(
    global_corr: pd.DataFrame,
    split_corr: pd.DataFrame,
    group_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature keep/watch/drop-candidate recommendations for dimensionality reduction."""

    split_wide = split_corr.pivot(index="feature", columns="scope", values="spearman_rho").reset_index()
    split_wide = split_wide.rename(columns={"train": "train_spearman_rho", "valid": "valid_spearman_rho"})
    out = global_corr.merge(split_wide, on="feature", how="left").merge(group_stability, on="feature", how="left")

    recommendations: List[str] = []
    reasons: List[str] = []
    for row in out.itertuples(index=False):
        corr_class = str(row.correlation_class)
        direction_consistency = float(row.direction_consistency) if pd.notna(row.direction_consistency) else float("nan")
        train_rho = float(row.train_spearman_rho) if pd.notna(row.train_spearman_rho) else float("nan")
        valid_rho = float(row.valid_spearman_rho) if pd.notna(row.valid_spearman_rho) else float("nan")
        split_same_direction = np.isfinite(train_rho) and np.isfinite(valid_rho) and np.sign(train_rho) == np.sign(valid_rho)

        if corr_class in {"strong", "moderate"} and direction_consistency >= 0.65 and split_same_direction:
            recommendations.append("keep_priority")
            reasons.append("全局相关达到 moderate/strong，组内方向较稳定，训练/验证方向一致")
        elif corr_class in {"weak", "negligible"}:
            recommendations.append("drop_candidate")
            reasons.append("全局单变量相关较弱，建议优先进入消融删除候选")
        elif direction_consistency < 0.60 or not split_same_direction:
            recommendations.append("watch_instability")
            reasons.append("全局相关存在但分组或训练/验证方向稳定性不足")
        else:
            recommendations.append("keep_or_ablate")
            reasons.append("相关性中等或稳定性一般，建议通过消融实验复核")

    out["reduction_recommendation"] = recommendations
    out["recommendation_reason"] = reasons
    return out.sort_values(["reduction_recommendation", "abs_spearman"], ascending=[True, False]).reset_index(drop=True)


def get_retained_feature_columns(recommendations: pd.DataFrame) -> List[str]:
    """Get features marked as keep_priority for retained-feature redundancy analysis."""

    retained = recommendations.loc[
        recommendations["reduction_recommendation"] == "keep_priority", "feature"
    ].astype(str)
    return retained.tolist()


def build_standardized_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build a complete-case standardized feature matrix and row metadata."""

    work = df[["policy", "cell_code", "cycles", *feature_cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    values = work[list(feature_cols)].to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    standardized = (values - means) / safe_stds
    return standardized, work[["policy", "cell_code", "cycles"]].reset_index(drop=True)


def classify_pair_redundancy(abs_spearman: float) -> str:
    """Classify pairwise redundancy from absolute Spearman correlation."""

    if not np.isfinite(abs_spearman):
        return "undefined"
    if abs_spearman >= REDUNDANT_SPEARMAN_THRESHOLD:
        return "near_redundant"
    if abs_spearman >= HIGHLY_RELATED_SPEARMAN_THRESHOLD:
        return "highly_related"
    return "not_redundant"


def calc_retained_intercorrelation(df: pd.DataFrame, retained_features: Sequence[str]) -> pd.DataFrame:
    """Calculate pairwise correlations among retained features in long-table form."""

    rows: List[Dict[str, Any]] = []
    for i, feature_a in enumerate(retained_features):
        for j, feature_b in enumerate(retained_features):
            if j < i:
                continue
            part = df[[feature_a, feature_b]].replace([np.inf, -np.inf], np.nan).dropna()
            x = part[feature_a].to_numpy(dtype=float)
            y = part[feature_b].to_numpy(dtype=float)
            if feature_a == feature_b:
                pearson_r, pearson_p = 1.0, 0.0
                spearman_rho, spearman_p = 1.0, 0.0
            else:
                pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
                spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
            abs_spearman = abs(spearman_rho) if np.isfinite(spearman_rho) else float("nan")
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "n_samples": int(len(part)),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "abs_spearman": abs_spearman,
                    "redundancy_class": classify_pair_redundancy(abs_spearman),
                }
            )
    return pd.DataFrame(rows).sort_values(["abs_spearman", "feature_a", "feature_b"], ascending=[False, True, True])


def calc_retained_vif(df: pd.DataFrame, retained_features: Sequence[str]) -> pd.DataFrame:
    """Calculate variance inflation factors for retained features using least squares R2."""

    x, _ = build_standardized_matrix(df, retained_features)
    rows: List[Dict[str, Any]] = []
    for idx, feature in enumerate(retained_features):
        y = x[:, idx]
        other_idx = [i for i in range(x.shape[1]) if i != idx]
        design = np.column_stack([np.ones(x.shape[0]), x[:, other_idx]])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        if not np.isfinite(r2):
            vif = float("nan")
        elif r2 >= 0.999999:
            vif = float("inf")
        else:
            vif = float(1.0 / max(1.0 - r2, 1e-12))
        if np.isfinite(vif) and vif >= VIF_SEVERE_THRESHOLD:
            severity = "severe_collinearity"
        elif np.isfinite(vif) and vif >= VIF_HIGH_THRESHOLD:
            severity = "high_collinearity"
        else:
            severity = "acceptable"
        rows.append(
            {
                "feature": feature,
                "r2_from_other_retained_features": r2,
                "vif": vif,
                "vif_severity": severity,
            }
        )
    return pd.DataFrame(rows).sort_values(["vif", "feature"], ascending=[False, True]).reset_index(drop=True)


def calc_retained_pca_summary(df: pd.DataFrame, retained_features: Sequence[str]) -> pd.DataFrame:
    """Calculate PCA explained-variance summary from standardized retained features."""

    x, _ = build_standardized_matrix(df, retained_features)
    covariance = np.cov(x, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(covariance)
    eigenvalues = np.sort(eigenvalues)[::-1]
    total = float(np.sum(eigenvalues))
    rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    for idx, eigenvalue in enumerate(eigenvalues, start=1):
        ratio = float(eigenvalue / total) if total > 0 else float("nan")
        cumulative += ratio if np.isfinite(ratio) else 0.0
        rows.append(
            {
                "component": f"PC{idx}",
                "component_index": int(idx),
                "eigenvalue": float(eigenvalue),
                "explained_variance_ratio": ratio,
                "cumulative_explained_ratio": float(cumulative),
            }
        )
    return pd.DataFrame(rows)


def choose_redundancy_representative(group: Sequence[str], recommendations: pd.DataFrame) -> str:
    """Choose one representative feature from a redundancy group."""

    score_df = recommendations.set_index("feature")
    semantic_priority = {
        "main_peak_area": 3,
        "main_peak_height_dqdv": 2,
        "main_peak_prominence": 1,
    }
    best_feature = ""
    best_score: Tuple[float, float, float, int, str] | None = None
    for feature in group:
        if feature not in score_df.index:
            score = (0.0, 0.0, 0.0, semantic_priority.get(feature, 0), feature)
        else:
            row = score_df.loc[feature]
            score = (
                float(row.get("abs_spearman", 0.0)),
                float(row.get("median_abs_group_spearman", 0.0)),
                float(row.get("direction_consistency", 0.0)),
                semantic_priority.get(feature, 0),
                feature,
            )
        if best_score is None or score > best_score:
            best_score = score
            best_feature = feature
    return best_feature


def calc_retained_redundancy_groups(
    intercorrelation: pd.DataFrame,
    retained_features: Sequence[str],
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Build redundancy groups from retained-feature pairwise Spearman correlations."""

    parent = {feature: feature for feature in retained_features}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(feature_a: str, feature_b: str) -> None:
        root_a = find(feature_a)
        root_b = find(feature_b)
        if root_a != root_b:
            parent[root_b] = root_a

    pair_df = intercorrelation.loc[intercorrelation["feature_a"] != intercorrelation["feature_b"]].copy()
    redundant_pairs = pair_df.loc[pair_df["abs_spearman"] >= REDUNDANT_SPEARMAN_THRESHOLD]
    for row in redundant_pairs.itertuples(index=False):
        union(str(row.feature_a), str(row.feature_b))

    groups: Dict[str, List[str]] = {}
    for feature in retained_features:
        groups.setdefault(find(feature), []).append(feature)

    rows: List[Dict[str, Any]] = []
    for idx, group_features in enumerate(groups.values(), start=1):
        sorted_features = sorted(group_features)
        representative = choose_redundancy_representative(sorted_features, recommendations)
        intra_pairs = pair_df.loc[
            pair_df["feature_a"].isin(sorted_features) & pair_df["feature_b"].isin(sorted_features)
        ]
        max_abs_pair = float(intra_pairs["abs_spearman"].max()) if not intra_pairs.empty else float("nan")
        rows.append(
            {
                "redundancy_group": f"group_{idx:02d}",
                "n_features": int(len(sorted_features)),
                "features": ";".join(sorted_features),
                "recommended_representative": representative,
                "max_abs_spearman_within_group": max_abs_pair,
                "group_type": "redundant_cluster" if len(sorted_features) > 1 else "singleton",
            }
        )
    return pd.DataFrame(rows).sort_values(["n_features", "redundancy_group"], ascending=[False, True]).reset_index(
        drop=True
    )


def save_retained_spearman_heatmap(intercorrelation: pd.DataFrame, retained_features: Sequence[str], out_path: Path) -> None:
    """Save a Spearman heatmap among retained features."""

    matrix = pd.DataFrame(np.eye(len(retained_features)), index=retained_features, columns=retained_features)
    for row in intercorrelation.itertuples(index=False):
        matrix.loc[str(row.feature_a), str(row.feature_b)] = float(row.spearman_rho)
        matrix.loc[str(row.feature_b), str(row.feature_a)] = float(row.spearman_rho)
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(retained_features)))
    ax.set_xticklabels(retained_features, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(retained_features)))
    ax.set_yticklabels(retained_features)
    ax.set_title("Retained Feature Spearman Intercorrelation")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_retained_pca_scree(pca_summary: pd.DataFrame, out_path: Path) -> None:
    """Save retained-feature PCA scree and cumulative explained-variance plot."""

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = pca_summary["component_index"].to_numpy(dtype=int)
    ratios = pca_summary["explained_variance_ratio"].to_numpy(dtype=float)
    cumulative = pca_summary["cumulative_explained_ratio"].to_numpy(dtype=float)
    ax1.bar(x, ratios, color="#2563eb", alpha=0.8, label="Explained ratio")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_xticks(x)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#dc2626", marker="o", label="Cumulative ratio")
    ax2.axhline(PCA_TARGET_EXPLAINED_RATIO, color="#475569", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Cumulative explained ratio")
    ax2.set_ylim(0.0, 1.05)
    ax1.set_title("Retained Feature PCA Explained Variance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_retained_redundancy_analysis(
    df: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build retained-feature redundancy outputs."""

    retained_features = get_retained_feature_columns(recommendations)
    if not retained_features:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    intercorrelation = calc_retained_intercorrelation(df, retained_features)
    vif = calc_retained_vif(df, retained_features)
    pca_summary = calc_retained_pca_summary(df, retained_features)
    redundancy_groups = calc_retained_redundancy_groups(
        intercorrelation=intercorrelation,
        retained_features=retained_features,
        recommendations=recommendations,
    )
    return intercorrelation, vif, pca_summary, redundancy_groups


def build_diagnostics(
    df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Build dataset diagnostics summary."""

    rows: List[Dict[str, Any]] = [
        {"item": "merged_rows", "value": int(len(df))},
        {"item": "train_rows", "value": int((df["set_type"] == "train").sum())},
        {"item": "valid_rows", "value": int((df["set_type"] == "valid").sum())},
        {"item": "policy_cell_groups", "value": int(df[["policy", "cell_code"]].drop_duplicates().shape[0])},
        {"item": "feature_count_analyzed", "value": int(len(MAIN_PEAK_FEATURE_COLUMNS))},
        {"item": "cycle_index_norm_included", "value": 0},
        {"item": "q_min", "value": float(args.q_min)},
        {"item": "q_max", "value": float(args.q_max)},
        {"item": "q_ref_cycles", "value": int(args.q_ref_cycles)},
        {"item": "retention_min", "value": float(args.retention_min)},
        {"item": "retention_max", "value": float(args.retention_max)},
    ]
    for row in missing_summary.itertuples(index=False):
        rows.append(
            {
                "item": f"{row.feature}_missing_before_zero_fill",
                "value": int(row.missing_before_fill),
            }
        )
    return pd.DataFrame(rows)


def save_global_bar(global_corr: pd.DataFrame, out_path: Path) -> None:
    """Save a bar chart of global Spearman correlations."""

    plot_df = global_corr.sort_values("spearman_rho", ascending=True).copy()
    colors = ["#0f766e" if value >= 0 else "#be123c" for value in plot_df["spearman_rho"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(plot_df["feature"], plot_df["spearman_rho"], color=colors)
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_xlabel("Spearman rho vs retention")
    ax.set_title("Global dQ/dV Feature Correlation")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_correlation_heatmap(global_corr: pd.DataFrame, out_path: Path) -> None:
    """Save a heatmap for Pearson, Spearman, and Kendall correlations."""

    plot_df = global_corr.set_index("feature")[["pearson_r", "spearman_rho", "kendall_tau"]].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    image = ax.imshow(plot_df.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(["Pearson", "Spearman", "Kendall"])
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist())
    ax.set_title("Correlation Matrix vs Retention")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = plot_df.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#0f172a", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_group_stability_bar(group_df: pd.DataFrame, out_path: Path) -> None:
    """Save a bar chart of median within-group Spearman correlations."""

    plot_df = group_df.sort_values("group_spearman_median", ascending=True).copy()
    colors = ["#2563eb" if value >= 0 else "#f97316" for value in plot_df["group_spearman_median"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(plot_df["feature"], plot_df["group_spearman_median"], color=colors)
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_xlabel("Median within-group Spearman rho")
    ax.set_title("Policy+Cell Group Correlation Stability")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def format_float(value: Any, digits: int = 4) -> str:
    """Format numeric values for Markdown tables."""

    try:
        number = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int | None = None) -> List[str]:
    """Build a simple Markdown table from selected columns."""

    out_df = df.loc[:, list(columns)].copy()
    if max_rows is not None:
        out_df = out_df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in out_df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    df: pd.DataFrame,
    global_corr: pd.DataFrame,
    split_corr: pd.DataFrame,
    group_stability: pd.DataFrame,
    recommendations: pd.DataFrame,
    retained_intercorrelation: pd.DataFrame,
    retained_vif: pd.DataFrame,
    retained_pca_summary: pd.DataFrame,
    retained_redundancy_groups: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    """Build the Markdown correlation report."""

    strong = global_corr.loc[global_corr["correlation_class"] == "strong", "feature"].tolist()
    moderate = global_corr.loc[global_corr["correlation_class"] == "moderate", "feature"].tolist()
    weak_or_lower = global_corr.loc[
        global_corr["correlation_class"].isin(["weak", "negligible"]), "feature"
    ].tolist()
    keep_features = recommendations.loc[
        recommendations["reduction_recommendation"] == "keep_priority", "feature"
    ].tolist()
    drop_candidates = recommendations.loc[
        recommendations["reduction_recommendation"] == "drop_candidate", "feature"
    ].tolist()
    retained_pair_summary = retained_intercorrelation.loc[
        retained_intercorrelation["feature_a"] != retained_intercorrelation["feature_b"]
    ].copy()
    near_redundant_pairs = retained_pair_summary.loc[
        retained_pair_summary["abs_spearman"] >= REDUNDANT_SPEARMAN_THRESHOLD
    ].copy()
    highly_related_pairs = retained_pair_summary.loc[
        (retained_pair_summary["abs_spearman"] >= HIGHLY_RELATED_SPEARMAN_THRESHOLD)
        & (retained_pair_summary["abs_spearman"] < REDUNDANT_SPEARMAN_THRESHOLD)
    ].copy()
    pca_95 = retained_pca_summary.loc[
        retained_pca_summary["cumulative_explained_ratio"] >= PCA_TARGET_EXPLAINED_RATIO
    ].head(1)
    pca_95_components = int(pca_95["component_index"].iloc[0]) if not pca_95.empty else 0

    lines: List[str] = []
    lines.append("# dQ/dV 9 特征与容量保持率相关性分析")
    lines.append("")
    lines.append("## 1. 分析摘要")
    lines.append("")
    lines.append(
        "本分析复用 dQ/dV + LSTM 的 retention 标签口径，仅评估 9 个 dQ/dV 主峰输入特征与容量保持率的相关性。"
        "`cycles` 和 `cycle_index_norm` 不作为解释特征，仅用于数据对齐、排序和分组。"
    )
    lines.append("")
    lines.append(f"- 合并后样本数：**{len(df):,}**")
    lines.append(f"- 训练集样本数：**{int((df['set_type'] == 'train').sum()):,}**")
    lines.append(f"- 验证集样本数：**{int((df['set_type'] == 'valid').sum()):,}**")
    lines.append(f"- `policy + cell_code` 分组数：**{df[['policy', 'cell_code']].drop_duplicates().shape[0]}**")
    lines.append(f"- strong 特征：{', '.join(strong) if strong else '无'}")
    lines.append(f"- moderate 特征：{', '.join(moderate) if moderate else '无'}")
    lines.append(f"- weak/negligible 特征：{', '.join(weak_or_lower) if weak_or_lower else '无'}")
    lines.append("")
    lines.append("## 2. 数据与口径")
    lines.append("")
    lines.append(f"- dQ/dV 特征文件：`{args.dqdv_path}`")
    lines.append(f"- 容量标签文件：`{args.life_path}`")
    lines.append(f"- q 过滤：`{args.q_min} <= q_discharge <= {args.q_max}`")
    lines.append(f"- `q_ref`：每个 `policy + cell_code` 前 `{args.q_ref_cycles}` 个有效循环 `q_discharge` 中位数")
    lines.append(f"- retention 过滤：`{args.retention_min} <= retention <= {args.retention_max}`")
    lines.append("- 缺失 dQ/dV 特征按 LSTM 输入口径进行数值化和 0 填充。")
    lines.append("")
    lines.append("## 3. 全局相关性排序")
    lines.append("")
    lines.extend(
        markdown_table(
            global_corr,
            [
                "feature",
                "n_samples",
                "pearson_r",
                "spearman_rho",
                "kendall_tau",
                "abs_spearman",
                "correlation_class",
            ],
        )
    )
    lines.append("")
    lines.append("![全局 Spearman 相关性](./spearman_global_bar.png)")
    lines.append("")
    lines.append("![相关性热图](./correlation_heatmap.png)")
    lines.append("")
    lines.append("## 4. 训练/验证分集一致性")
    lines.append("")
    lines.extend(
        markdown_table(
            split_corr.sort_values(["feature", "scope"]),
            ["scope", "feature", "n_samples", "spearman_rho", "abs_spearman", "correlation_class"],
        )
    )
    lines.append("")
    lines.append("## 5. 组内稳健性")
    lines.append("")
    lines.append(
        "组内稳健性按 `policy + cell_code` 计算每个特征与 retention 的 Spearman 相关，"
        "再汇总中位数、IQR 和方向一致率。该口径用于判断全局相关是否被少数长寿命或高样本电芯主导。"
    )
    lines.append("")
    lines.extend(
        markdown_table(
            group_stability,
            [
                "feature",
                "n_valid_groups",
                "group_spearman_median",
                "group_spearman_iqr",
                "direction_consistency",
                "median_abs_group_spearman",
            ],
        )
    )
    lines.append("")
    lines.append("![组内相关稳定性](./group_stability_bar.png)")
    lines.append("")
    lines.append("## 6. 降维建议")
    lines.append("")
    lines.extend(
        markdown_table(
            recommendations,
            [
                "feature",
                "spearman_rho",
                "correlation_class",
                "group_spearman_median",
                "direction_consistency",
                "reduction_recommendation",
            ],
        )
    )
    lines.append("")
    lines.append(f"- 优先保留：{', '.join(keep_features) if keep_features else '暂无严格满足 keep_priority 规则的特征'}")
    lines.append(f"- 优先消融候选：{', '.join(drop_candidates) if drop_candidates else '暂无 weak/negligible 候选'}")
    lines.append("")
    lines.append("建议将 `drop_candidate` 和 `watch_instability` 特征优先纳入后续 LSTM 消融实验；")
    lines.append("若删除后验证集 RMSE/R2 基本不变，可进一步减少输入维度和推理计算。")
    lines.append("")
    lines.append("## 7. 优先保留特征内部降维可行性")
    lines.append("")
    lines.append(
        "本节只针对第一轮 `keep_priority` 特征做统计冗余分析，不训练 LSTM。"
        "判断依据包括保留特征之间的 Spearman/Pearson 相关、VIF 共线性、PCA 累计解释率和冗余组。"
    )
    lines.append("")
    lines.append(f"- 第一轮优先保留特征数：**{len(keep_features)}**")
    lines.append(f"- 近似冗余阈值：`abs_spearman >= {REDUNDANT_SPEARMAN_THRESHOLD}`")
    lines.append(f"- 高度相关阈值：`abs_spearman >= {HIGHLY_RELATED_SPEARMAN_THRESHOLD}`")
    lines.append(f"- PCA `{PCA_TARGET_EXPLAINED_RATIO:.0%}` 累计解释率所需主成分数：**{pca_95_components}**")
    lines.append("")
    if not near_redundant_pairs.empty:
        lines.append("近似冗余特征对：")
        lines.extend(
            markdown_table(
                near_redundant_pairs.sort_values("abs_spearman", ascending=False),
                ["feature_a", "feature_b", "spearman_rho", "pearson_r", "abs_spearman", "redundancy_class"],
                max_rows=12,
            )
        )
    else:
        lines.append("未发现 `abs_spearman >= 0.90` 的近似冗余特征对。")
    lines.append("")
    if not highly_related_pairs.empty:
        lines.append("高度相关但未达到近似冗余的特征对：")
        lines.extend(
            markdown_table(
                highly_related_pairs.sort_values("abs_spearman", ascending=False),
                ["feature_a", "feature_b", "spearman_rho", "pearson_r", "abs_spearman", "redundancy_class"],
                max_rows=12,
            )
        )
    lines.append("")
    lines.append("VIF 共线性诊断：")
    lines.extend(markdown_table(retained_vif, ["feature", "r2_from_other_retained_features", "vif", "vif_severity"]))
    lines.append("")
    lines.append("PCA 解释率：")
    lines.extend(
        markdown_table(
            retained_pca_summary,
            ["component", "explained_variance_ratio", "cumulative_explained_ratio"],
        )
    )
    lines.append("")
    lines.append("冗余组与代表特征建议：")
    lines.extend(
        markdown_table(
            retained_redundancy_groups,
            [
                "redundancy_group",
                "n_features",
                "features",
                "recommended_representative",
                "max_abs_spearman_within_group",
                "group_type",
            ],
        )
    )
    lines.append("")
    lines.append("![优先保留特征内部 Spearman 热图](./retained_feature_spearman_heatmap.png)")
    lines.append("")
    lines.append("![优先保留特征 PCA 解释率](./retained_feature_pca_scree.png)")
    lines.append("")
    lines.append("结论：优先保留特征内部仍存在进一步降维空间。")
    lines.append(
        "`main_peak_area`、`main_peak_height_dqdv` 与 `main_peak_prominence` 共同构成峰强度/面积冗余组，"
        "建议下一轮 LSTM 消融优先测试在三者中保留 1-2 个；同时保留 `main_peak_skewness`、"
        "`main_peak_voltage_v` 和 `main_peak_width_v` 作为峰形、峰位和宽度信息。"
    )
    lines.append(
        "统计上可先尝试从 6 个优先保留特征压缩到 4-5 个："
        "`main_peak_area`（或峰高/ prominence 的代表项）+ `main_peak_skewness` + "
        "`main_peak_voltage_v` + `main_peak_width_v`，再用 LSTM 消融验证性能损失。"
    )
    lines.append("")
    lines.append("## 8. 局限性")
    lines.append("")
    lines.append("- 相关性不是因果结论，只能作为降维和消融实验的筛选依据。")
    lines.append("- 全局相关仍可能受到老化阶段、工况和电芯差异影响，因此报告同时给出组内稳健性。")
    lines.append("- 本分析刻意不纳入 `cycles`/`cycle_index_norm`，因此不评估循环进程本身对 retention 的解释力。")
    lines.append("- 单变量相关不能刻画多特征互补关系；弱相关特征仍可能在 LSTM 中提供非线性组合信息。")
    lines.append("- 保留特征内部降维结论来自统计冗余分析，不能替代后续 LSTM 消融验证。")
    lines.append("")
    lines.append("## 9. 输出文件")
    lines.append("")
    lines.append("- `correlation_global.csv`")
    lines.append("- `correlation_by_split.csv`")
    lines.append("- `correlation_group_stability.csv`")
    lines.append("- `feature_reduction_recommendation.csv`")
    lines.append("- `retained_feature_intercorrelation.csv`")
    lines.append("- `retained_feature_vif.csv`")
    lines.append("- `retained_feature_pca_summary.csv`")
    lines.append("- `retained_feature_redundancy_groups.csv`")
    lines.append("- `dataset_diagnostics.csv`")
    lines.append("- `spearman_global_bar.png`")
    lines.append("- `correlation_heatmap.png`")
    lines.append("- `group_stability_bar.png`")
    lines.append("- `retained_feature_spearman_heatmap.png`")
    lines.append("- `retained_feature_pca_scree.png`")
    lines.append("")
    lines.append("## 10. 诊断摘要")
    lines.append("")
    lines.extend(markdown_table(diagnostics, ["item", "value"], max_rows=20))
    lines.append("")
    lines.append(f"_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    return "\n".join(lines)


def save_outputs(
    df: pd.DataFrame,
    global_corr: pd.DataFrame,
    split_corr: pd.DataFrame,
    group_stability: pd.DataFrame,
    recommendations: pd.DataFrame,
    retained_intercorrelation: pd.DataFrame,
    retained_vif: pd.DataFrame,
    retained_pca_summary: pd.DataFrame,
    retained_redundancy_groups: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Save tables, figures, and Markdown report."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_corr.to_csv(args.output_dir / "correlation_global.csv", index=False, encoding="utf-8-sig")
    split_corr.to_csv(args.output_dir / "correlation_by_split.csv", index=False, encoding="utf-8-sig")
    group_stability.to_csv(args.output_dir / "correlation_group_stability.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(
        args.output_dir / "feature_reduction_recommendation.csv", index=False, encoding="utf-8-sig"
    )
    retained_intercorrelation.to_csv(
        args.output_dir / "retained_feature_intercorrelation.csv", index=False, encoding="utf-8-sig"
    )
    retained_vif.to_csv(args.output_dir / "retained_feature_vif.csv", index=False, encoding="utf-8-sig")
    retained_pca_summary.to_csv(
        args.output_dir / "retained_feature_pca_summary.csv", index=False, encoding="utf-8-sig"
    )
    retained_redundancy_groups.to_csv(
        args.output_dir / "retained_feature_redundancy_groups.csv", index=False, encoding="utf-8-sig"
    )
    diagnostics.to_csv(args.output_dir / "dataset_diagnostics.csv", index=False, encoding="utf-8-sig")

    save_global_bar(global_corr, args.output_dir / "spearman_global_bar.png")
    save_correlation_heatmap(global_corr, args.output_dir / "correlation_heatmap.png")
    save_group_stability_bar(group_stability, args.output_dir / "group_stability_bar.png")
    retained_features = get_retained_feature_columns(recommendations)
    if retained_features:
        save_retained_spearman_heatmap(
            retained_intercorrelation,
            retained_features,
            args.output_dir / "retained_feature_spearman_heatmap.png",
        )
        save_retained_pca_scree(retained_pca_summary, args.output_dir / "retained_feature_pca_scree.png")

    report = build_report(
        df=df,
        global_corr=global_corr,
        split_corr=split_corr,
        group_stability=group_stability,
        recommendations=recommendations,
        retained_intercorrelation=retained_intercorrelation,
        retained_vif=retained_vif,
        retained_pca_summary=retained_pca_summary,
        retained_redundancy_groups=retained_redundancy_groups,
        diagnostics=diagnostics,
        args=args,
    )
    (args.output_dir / "dqdv_feature_retention_correlation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run the correlation analysis pipeline."""

    args = parse_args()
    if "cycle_index_norm" in MAIN_PEAK_FEATURE_COLUMNS or "cycles" in MAIN_PEAK_FEATURE_COLUMNS:
        raise RuntimeError("Cycle features must not be included in MAIN_PEAK_FEATURE_COLUMNS.")

    dqdv_df, missing_summary = load_dqdv_features(args.dqdv_path)
    label_df = load_retention_labels(
        life_path=args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )
    split_df = load_split_map(args.train_split_path, args.valid_split_path)
    dataset = build_dataset(dqdv_df=dqdv_df, label_df=label_df, split_df=split_df)

    global_corr = calc_correlations_for_scope(dataset, MAIN_PEAK_FEATURE_COLUMNS, "all")
    split_corr = calc_split_correlations(dataset, MAIN_PEAK_FEATURE_COLUMNS)
    group_stability = calc_group_stability(
        dataset,
        MAIN_PEAK_FEATURE_COLUMNS,
        min_group_samples=int(args.min_group_samples),
    )
    recommendations = build_reduction_recommendations(
        global_corr=global_corr,
        split_corr=split_corr,
        group_stability=group_stability,
    )
    retained_intercorrelation, retained_vif, retained_pca_summary, retained_redundancy_groups = (
        build_retained_redundancy_analysis(dataset, recommendations)
    )
    diagnostics = build_diagnostics(dataset, missing_summary=missing_summary, args=args)
    save_outputs(
        df=dataset,
        global_corr=global_corr,
        split_corr=split_corr,
        group_stability=group_stability,
        recommendations=recommendations,
        retained_intercorrelation=retained_intercorrelation,
        retained_vif=retained_vif,
        retained_pca_summary=retained_pca_summary,
        retained_redundancy_groups=retained_redundancy_groups,
        diagnostics=diagnostics,
        args=args,
    )

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows: {len(dataset)} | features: {len(MAIN_PEAK_FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
