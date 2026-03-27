from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_POLICY_MEANING_PATH = REPO_ROOT / "data" / "processed" / "policy_meaning.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "causal_initial_rate_effect"

MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402


@dataclass
class NuisanceArtifacts:
    """Container for fitted nuisance quantities used by DR estimation."""

    treatment_residual_std: float
    treatment_density: np.ndarray
    outcome_model: GradientBoostingRegressor
    outcome_mu_observed: np.ndarray


def setup_plot_fonts() -> bool:
    """Configure plotting fonts and return whether CJK-capable font is available."""
    candidates = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "fonts-noto-cjk",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected = [name for name in candidates if name in installed]
    rcParams["axes.unicode_minus"] = False
    if selected:
        rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
        return True
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False


HAS_CJK_FONT = setup_plot_fonts()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate +1C causal effect on future relative capacity drop via GPS+AIPW."
    )
    parser.add_argument(
        "--life-performance-path",
        type=Path,
        default=DEFAULT_LIFE_PERFORMANCE_PATH,
        help="Path to life_performance.csv",
    )
    parser.add_argument(
        "--policy-meaning-path",
        type=Path,
        default=DEFAULT_POLICY_MEANING_PATH,
        help="Path to policy_meaning.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory under outputs/analysis.",
    )
    parser.add_argument(
        "--horizon-cycles",
        type=int,
        default=200,
        help="Future cycle horizon used to build t -> t+h windows.",
    )
    parser.add_argument(
        "--treatment-mode",
        type=str,
        choices=["initial", "effective_mean"],
        default="initial",
        help="Treatment definition: initial C-rate or effective mean C-rate.",
    )
    parser.add_argument(
        "--exclude-policy-prefix",
        type=str,
        default="VARCHARGE",
        help="Exclude policies that start with this prefix.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="Step size of treatment grid for dose-response estimation.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.02,
        help="Treatment support trimming quantile for grid boundaries.",
    )
    parser.add_argument(
        "--weight-clip-quantile",
        type=float,
        default=0.995,
        help="Clip raw kernel/GPS weights at this quantile.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=400,
        help="Cluster bootstrap repetitions for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260327,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding.",
    )
    return parser.parse_args()


def gaussian_pdf(x: np.ndarray) -> np.ndarray:
    """Evaluate standard normal density on an array."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric with coercion."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_treatment_column(policy_df: pd.DataFrame, treatment_mode: str) -> pd.Series:
    """Construct treatment variable from policy-level fields."""
    if treatment_mode == "initial":
        return policy_df["initial_c_rate"]
    if treatment_mode == "effective_mean":
        switch_ratio = policy_df["switch_soc_percent"] / 100.0
        return (
            policy_df["initial_c_rate"] * switch_ratio
            + policy_df["post_switch_c_rate"] * (1.0 - switch_ratio)
        )
    raise ValueError(f"Unsupported treatment_mode: {treatment_mode}")


def build_analysis_dataset(
    life_path: Path,
    policy_path: Path,
    horizon_cycles: int,
    treatment_mode: str,
    exclude_policy_prefix: str,
    encoding: str,
) -> pd.DataFrame:
    """Build rolling-window analysis dataset for causal estimation."""
    if horizon_cycles <= 0:
        raise ValueError("horizon_cycles must be a positive integer.")

    life_df = pd.read_csv(
        life_path,
        encoding=encoding,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    policy_df = pd.read_csv(
        policy_path,
        encoding=encoding,
        usecols=[
            "policy",
            "initial_c_rate",
            "switch_soc_percent",
            "post_switch_c_rate",
        ],
    )

    life_df = _to_numeric(life_df, columns=["cycles", "q_discharge"])
    policy_df = _to_numeric(
        policy_df,
        columns=["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"],
    )

    life_df = life_df.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life_df["cycles"] = life_df["cycles"].astype(int)
    life_df = life_df.loc[life_df["q_discharge"] > 0].copy()

    if exclude_policy_prefix:
        keep_mask_life = ~life_df["policy"].astype(str).str.startswith(exclude_policy_prefix)
        life_df = life_df.loc[keep_mask_life].copy()

    policy_df = policy_df.dropna(subset=["policy"]).drop_duplicates(subset=["policy"]).copy()
    if exclude_policy_prefix:
        keep_mask_pol = ~policy_df["policy"].astype(str).str.startswith(exclude_policy_prefix)
        policy_df = policy_df.loc[keep_mask_pol].copy()

    policy_df["treatment_value"] = _build_treatment_column(policy_df, treatment_mode=treatment_mode)
    policy_df = policy_df.dropna(
        subset=["treatment_value", "switch_soc_percent", "post_switch_c_rate"]
    ).copy()

    merged = life_df.merge(
        policy_df[
            [
                "policy",
                "treatment_value",
                "switch_soc_percent",
                "post_switch_c_rate",
            ]
        ],
        on="policy",
        how="inner",
        validate="many_to_one",
    ).copy()

    left_df = merged.rename(
        columns={"cycles": "window_start_cycle", "q_discharge": "q_start"}
    )[
        [
            "policy",
            "cell_code",
            "window_start_cycle",
            "q_start",
            "treatment_value",
            "switch_soc_percent",
            "post_switch_c_rate",
        ]
    ].copy()
    left_df["window_end_cycle"] = left_df["window_start_cycle"] + horizon_cycles

    right_df = merged.rename(
        columns={"cycles": "window_end_cycle", "q_discharge": "q_end"}
    )[
        ["policy", "cell_code", "window_end_cycle", "q_end"]
    ].copy()

    window_df = left_df.merge(
        right_df,
        on=["policy", "cell_code", "window_end_cycle"],
        how="inner",
        validate="many_to_one",
    ).copy()

    window_df = window_df.loc[window_df["q_start"] > 0].copy()
    window_df["y_rel_drop"] = (window_df["q_start"] - window_df["q_end"]) / window_df["q_start"]
    window_df["cluster_id"] = (
        window_df["policy"].astype(str) + "|" + window_df["cell_code"].astype(str)
    )

    rank_series = window_df["window_start_cycle"].rank(method="first")
    window_df["life_stage"] = pd.qcut(
        rank_series,
        q=3,
        labels=["early", "mid", "late"],
    ).astype(str)

    window_df = window_df.sort_values(
        ["policy", "cell_code", "window_start_cycle"]
    ).reset_index(drop=True)
    if window_df.empty:
        raise ValueError("No valid rolling-window samples were built.")
    return window_df


def build_covariate_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """Build covariate matrix for nuisance models."""
    cycle_centered = df["window_start_cycle"].to_numpy(dtype=float)
    cycle_centered = cycle_centered - cycle_centered.mean()
    x = np.column_stack(
        [
            df["switch_soc_percent"].to_numpy(dtype=float),
            df["post_switch_c_rate"].to_numpy(dtype=float),
            cycle_centered,
            cycle_centered ** 2,
        ]
    )
    feature_names = [
        "switch_soc_percent",
        "post_switch_c_rate",
        "window_start_cycle_centered",
        "window_start_cycle_centered_sq",
    ]
    return x, feature_names


def fit_nuisance_models(
    treatment: np.ndarray,
    covariates: np.ndarray,
    outcome: np.ndarray,
    seed: int,
) -> NuisanceArtifacts:
    """Fit treatment and outcome nuisance models for DR estimation."""
    treatment_model = LinearRegression()
    treatment_model.fit(covariates, treatment)
    treatment_mu = treatment_model.predict(covariates)

    residual = treatment - treatment_mu
    residual_std = float(np.std(residual, ddof=1))
    residual_std = max(residual_std, 0.1)

    treatment_density = gaussian_pdf(residual / residual_std) / residual_std
    treatment_density = np.clip(treatment_density, 1e-6, None)

    outcome_model = GradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        n_estimators=320,
        max_depth=3,
        min_samples_leaf=120,
        subsample=0.85,
        random_state=seed,
    )
    outcome_model.fit(np.column_stack([treatment, covariates]), outcome)
    mu_obs = outcome_model.predict(np.column_stack([treatment, covariates]))

    return NuisanceArtifacts(
        treatment_residual_std=residual_std,
        treatment_density=treatment_density,
        outcome_model=outcome_model,
        outcome_mu_observed=mu_obs,
    )


def default_bandwidth(treatment: np.ndarray) -> float:
    """Compute Silverman bandwidth with a lower bound for stability."""
    std = float(np.std(treatment, ddof=1))
    n = treatment.shape[0]
    raw = 1.06 * std * (n ** (-1.0 / 5.0))
    return max(raw, 0.2)


def build_grids(
    treatment: np.ndarray,
    step: float,
    trim_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Build base grid (t) and evaluation grid (t and t+1)."""
    if not (0.0 <= trim_quantile < 0.5):
        raise ValueError("trim_quantile must be in [0, 0.5).")
    if step <= 0:
        raise ValueError("step must be positive.")

    low = float(np.quantile(treatment, trim_quantile))
    high = float(np.quantile(treatment, 1.0 - trim_quantile))
    base_high = high - 1.0
    if base_high <= low:
        low = float(treatment.min())
        base_high = float(treatment.max()) - 1.0
    if base_high <= low:
        raise ValueError("Treatment support is too narrow to estimate +1C effect.")

    base_grid = np.arange(low, base_high + 0.5 * step, step)
    if base_grid.size < 5:
        base_grid = np.linspace(low, base_high, num=5)
    eval_grid = np.unique(np.concatenate([base_grid, base_grid + 1.0]))
    return base_grid, eval_grid, low, high


def marginal_kernel_density(
    treatment: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Estimate marginal treatment density on grid points by kernel averaging."""
    density_values: list[float] = []
    for t0 in grid:
        u = (treatment - t0) / bandwidth
        density_values.append(float(np.mean(gaussian_pdf(u) / bandwidth)))
    return np.asarray(density_values, dtype=float)


def compute_dr_table(
    treatment: np.ndarray,
    covariates: np.ndarray,
    outcome: np.ndarray,
    nuisance: NuisanceArtifacts,
    eval_grid: np.ndarray,
    bandwidth: float,
    weight_clip_quantile: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute DR pseudo outcomes for each evaluation treatment value."""
    if not (0.5 <= weight_clip_quantile <= 1.0):
        raise ValueError("weight_clip_quantile must be in [0.5, 1.0].")

    n = treatment.shape[0]
    m = eval_grid.shape[0]
    dr_matrix = np.empty((n, m), dtype=float)
    diagnostics: list[dict] = []

    for idx, t0 in enumerate(eval_grid):
        x_t0 = np.column_stack([np.full(n, t0, dtype=float), covariates])
        mu_t0 = nuisance.outcome_model.predict(x_t0)

        kernel = gaussian_pdf((treatment - t0) / bandwidth) / bandwidth
        raw_w = kernel / nuisance.treatment_density
        clip_threshold = float(np.quantile(raw_w, weight_clip_quantile))
        weight = np.clip(raw_w, 0.0, clip_threshold)

        dr_matrix[:, idx] = mu_t0 + weight * (outcome - nuisance.outcome_mu_observed)

        sum_w = float(weight.sum())
        ess = (sum_w * sum_w) / max(float(np.square(weight).sum()), 1e-12)
        diagnostics.append(
            {
                "t_eval": float(t0),
                "weight_mean": float(np.mean(weight)),
                "weight_std": float(np.std(weight, ddof=1)),
                "weight_p95": float(np.quantile(weight, 0.95)),
                "weight_p99": float(np.quantile(weight, 0.99)),
                "weight_max": float(weight.max()),
                "effective_sample_size": ess,
                "clip_threshold": clip_threshold,
            }
        )

    return dr_matrix, pd.DataFrame(diagnostics)


def _safe_normalize(weights: np.ndarray) -> np.ndarray:
    """Normalize non-negative weights with safe fallback."""
    total = float(weights.sum())
    if total <= 0:
        return np.full_like(weights, 1.0 / max(len(weights), 1))
    return weights / total


def _cluster_index_maps(cluster_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert cluster ids into per-cluster sums/counts index helpers."""
    unique_ids, inv = np.unique(cluster_ids, return_inverse=True)
    return unique_ids, inv


def bootstrap_ci_from_scores(
    score: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, int]:
    """Compute cluster bootstrap confidence interval from per-row influence scores."""
    if n_bootstrap <= 0:
        return float("nan"), float("nan"), 0
    if score.size == 0:
        return float("nan"), float("nan"), 0

    unique_clusters, inv = _cluster_index_maps(cluster_ids)
    n_cluster = unique_clusters.size
    if n_cluster < 2:
        return float("nan"), float("nan"), 0

    cluster_sum = np.zeros(n_cluster, dtype=float)
    cluster_count = np.zeros(n_cluster, dtype=float)
    np.add.at(cluster_sum, inv, score)
    np.add.at(cluster_count, inv, 1.0)

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_cluster, size=n_cluster)
        total_sum = float(cluster_sum[sampled].sum())
        total_count = float(cluster_count[sampled].sum())
        if total_count <= 0:
            continue
        estimates.append(total_sum / total_count)

    if not estimates:
        return float("nan"), float("nan"), 0
    arr = np.asarray(estimates, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(arr.size)


def summarize_effects(
    df: pd.DataFrame,
    dr_matrix: np.ndarray,
    eval_grid: np.ndarray,
    base_grid: np.ndarray,
    treatment: np.ndarray,
    bandwidth: float,
    n_bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize global and stage-specific +1C effects and DR curves."""
    idx_base = np.searchsorted(eval_grid, base_grid)
    idx_plus = np.searchsorted(eval_grid, base_grid + 1.0)

    dose_rows: dict[str, np.ndarray] = {"t_eval": eval_grid, "m_global": dr_matrix.mean(axis=0)}
    for stage in ["early", "mid", "late"]:
        mask = (df["life_stage"].to_numpy() == stage)
        if mask.any():
            dose_rows[f"m_{stage}"] = dr_matrix[mask].mean(axis=0)
        else:
            dose_rows[f"m_{stage}"] = np.full(eval_grid.shape[0], np.nan)
    dose_curve_df = pd.DataFrame(dose_rows)

    delta_matrix = dr_matrix[:, idx_plus] - dr_matrix[:, idx_base]
    weight_global = _safe_normalize(marginal_kernel_density(treatment, base_grid, bandwidth))
    delta_global = delta_matrix.mean(axis=0)
    score_global = delta_matrix @ weight_global
    effect_global = float(np.sum(weight_global * delta_global))

    global_ci_low, global_ci_high, global_boot_n = bootstrap_ci_from_scores(
        score=score_global,
        cluster_ids=df["cluster_id"].to_numpy(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    effect_rows: list[dict] = [
        {
            "group": "global",
            "effect_plus_1c": effect_global,
            "ci_low": global_ci_low,
            "ci_high": global_ci_high,
            "bootstrap_success": global_boot_n,
            "n_rows": int(df.shape[0]),
            "n_clusters": int(df["cluster_id"].nunique()),
        }
    ]

    delta_curve_rows: dict[str, np.ndarray] = {
        "base_t": base_grid,
        "next_t": base_grid + 1.0,
        "delta_global": delta_global,
        "weight_global": weight_global,
    }

    for stage_idx, stage in enumerate(["early", "mid", "late"]):
        mask = (df["life_stage"].to_numpy() == stage)
        stage_t = treatment[mask]
        if mask.any() and np.unique(df.loc[mask, "cluster_id"]).size >= 2:
            delta_stage = delta_matrix[mask].mean(axis=0)
            weight_stage = _safe_normalize(marginal_kernel_density(stage_t, base_grid, bandwidth))
            score_stage = delta_matrix[mask] @ weight_stage
            effect_stage = float(np.sum(weight_stage * delta_stage))
            ci_low, ci_high, boot_n = bootstrap_ci_from_scores(
                score=score_stage,
                cluster_ids=df.loc[mask, "cluster_id"].to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed + 100 + stage_idx,
            )
            effect_rows.append(
                {
                    "group": stage,
                    "effect_plus_1c": effect_stage,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_success": boot_n,
                    "n_rows": int(mask.sum()),
                    "n_clusters": int(df.loc[mask, "cluster_id"].nunique()),
                }
            )
            delta_curve_rows[f"delta_{stage}"] = delta_stage
            delta_curve_rows[f"weight_{stage}"] = weight_stage
        else:
            effect_rows.append(
                {
                    "group": stage,
                    "effect_plus_1c": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "bootstrap_success": 0,
                    "n_rows": int(mask.sum()),
                    "n_clusters": int(df.loc[mask, "cluster_id"].nunique()),
                }
            )
            delta_curve_rows[f"delta_{stage}"] = np.full(base_grid.shape[0], np.nan)
            delta_curve_rows[f"weight_{stage}"] = np.full(base_grid.shape[0], np.nan)

    effect_df = pd.DataFrame(effect_rows)
    delta_curve_df = pd.DataFrame(delta_curve_rows)
    return effect_df, dose_curve_df, delta_curve_df


def build_overlap_diagnostics(
    df: pd.DataFrame,
    treatment: np.ndarray,
    base_grid: np.ndarray,
    eval_grid: np.ndarray,
    bandwidth: float,
    nuisance: NuisanceArtifacts,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Build overlap and setup diagnostics table."""
    rows = [
        {"metric": "run_time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "python_executable", "value": sys.executable},
        {"metric": "python_version", "value": sys.version.split()[0]},
        {"metric": "horizon_cycles", "value": args.horizon_cycles},
        {"metric": "treatment_mode", "value": args.treatment_mode},
        {"metric": "exclude_policy_prefix", "value": args.exclude_policy_prefix},
        {"metric": "n_rows", "value": int(df.shape[0])},
        {"metric": "n_policy", "value": int(df["policy"].nunique())},
        {"metric": "n_cell", "value": int(df["cell_code"].nunique())},
        {"metric": "n_policy_cell", "value": int(df["cluster_id"].nunique())},
        {"metric": "treatment_min", "value": float(np.min(treatment))},
        {"metric": "treatment_max", "value": float(np.max(treatment))},
        {"metric": "treatment_q01", "value": float(np.quantile(treatment, 0.01))},
        {"metric": "treatment_q99", "value": float(np.quantile(treatment, 0.99))},
        {"metric": "treatment_std", "value": float(np.std(treatment, ddof=1))},
        {"metric": "grid_base_min", "value": float(base_grid.min())},
        {"metric": "grid_base_max", "value": float(base_grid.max())},
        {"metric": "grid_base_points", "value": int(base_grid.shape[0])},
        {"metric": "grid_eval_points", "value": int(eval_grid.shape[0])},
        {"metric": "bandwidth", "value": float(bandwidth)},
        {
            "metric": "treatment_residual_std",
            "value": float(nuisance.treatment_residual_std),
        },
    ]
    return pd.DataFrame(rows)


def save_dose_response_plot(dose_curve_df: pd.DataFrame, output_path: Path) -> None:
    """Save dose-response curve figure."""
    if HAS_CJK_FONT:
        stage_label_map = {"global": "全局", "early": "早期", "mid": "中期", "late": "后期"}
        xlabel = "处理强度 t（充电倍率 C）"
        ylabel = "估计的 E[Y | do(T=t)]"
        title = "剂量-反应曲线"
    else:
        stage_label_map = {"global": "global", "early": "early", "mid": "mid", "late": "late"}
        xlabel = "Treatment level t (C-rate)"
        ylabel = "Estimated E[Y | do(T=t)]"
        title = "Dose-response curve"
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(
        dose_curve_df["t_eval"],
        dose_curve_df["m_global"],
        label=stage_label_map["global"],
        linewidth=2.0,
    )
    for stage in ["early", "mid", "late"]:
        col = f"m_{stage}"
        if col in dose_curve_df.columns:
            plt.plot(
                dose_curve_df["t_eval"],
                dose_curve_df[col],
                label=stage_label_map[stage],
                linewidth=1.4,
                alpha=0.9,
            )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_stage_effect_plot(effect_df: pd.DataFrame, output_path: Path) -> None:
    """Save stage-wise +1C effect bar chart with confidence intervals."""
    order = ["global", "early", "mid", "late"]
    if HAS_CJK_FONT:
        label_map = {"global": "全局", "early": "早期", "mid": "中期", "late": "后期"}
        ylabel = "+1C 对相对容量下降的估计影响"
        title = "全局与分阶段 +1C 效应"
    else:
        label_map = {"global": "global", "early": "early", "mid": "mid", "late": "late"}
        ylabel = "Estimated +1C effect on relative drop"
        title = "Global and stage-wise +1C effect"
    plot_df = effect_df.set_index("group").reindex(order).reset_index()

    y = plot_df["effect_plus_1c"].to_numpy(dtype=float)
    ci_low = plot_df["ci_low"].to_numpy(dtype=float)
    ci_high = plot_df["ci_high"].to_numpy(dtype=float)

    yerr_low = np.where(np.isfinite(ci_low), y - ci_low, np.nan)
    yerr_high = np.where(np.isfinite(ci_high), ci_high - y, np.nan)
    yerr = np.vstack([yerr_low, yerr_high])

    plt.figure(figsize=(7.6, 4.8))
    x = np.arange(plot_df.shape[0])
    plt.bar(x, y, width=0.62, alpha=0.88)
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1.1)
    plt.xticks(x, [label_map.get(v, v) for v in plot_df["group"].tolist()])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_delta_plus1c_curve_plot(delta_curve_df: pd.DataFrame, output_path: Path) -> None:
    """Save +1C local effect curve by baseline treatment level."""
    if HAS_CJK_FONT:
        stage_label_map = {"global": "全局", "early": "早期", "mid": "中期", "late": "后期"}
        xlabel = "基准处理强度 base_t（充电倍率 C）"
        ylabel = "局部 +1C 效应：m(t+1)-m(t)"
        title = "+1C 局部效应曲线"
    else:
        stage_label_map = {"global": "global", "early": "early", "mid": "mid", "late": "late"}
        xlabel = "Baseline treatment level base_t (C-rate)"
        ylabel = "Local +1C effect: m(t+1)-m(t)"
        title = "Local +1C effect curve"
    plt.figure(figsize=(8.8, 5.2))
    plt.plot(
        delta_curve_df["base_t"],
        delta_curve_df["delta_global"],
        label=stage_label_map["global"],
        linewidth=2.0,
    )
    for stage in ["early", "mid", "late"]:
        col = f"delta_{stage}"
        if col in delta_curve_df.columns:
            plt.plot(
                delta_curve_df["base_t"],
                delta_curve_df[col],
                label=stage_label_map[stage],
                linewidth=1.4,
                alpha=0.9,
            )
    plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_treatment_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Save treatment distribution histogram."""
    treatment = df["treatment_value"].to_numpy(dtype=float)
    q01 = float(np.quantile(treatment, 0.01))
    q99 = float(np.quantile(treatment, 0.99))
    if HAS_CJK_FONT:
        xlabel = "处理变量 treatment_value（充电倍率 C）"
        ylabel = "样本数"
        title = "处理变量分布与支持区间"
    else:
        xlabel = "Treatment value (C-rate)"
        ylabel = "Sample count"
        title = "Treatment distribution and support"

    plt.figure(figsize=(8.4, 5.0))
    plt.hist(treatment, bins=36, color="#3B82F6", alpha=0.78, edgecolor="white")
    plt.axvline(q01, color="#F97316", linestyle="--", linewidth=1.6, label="q01")
    plt.axvline(q99, color="#EF4444", linestyle="--", linewidth=1.6, label="q99")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_outcome_by_stage_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Save outcome distribution by life stage."""
    order = ["early", "mid", "late"]
    if HAS_CJK_FONT:
        label_map = {"early": "早期", "mid": "中期", "late": "后期"}
        ylabel = "y_rel_drop（未来窗口相对容量下降）"
        xlabel = "寿命阶段"
        title = "结果变量在不同寿命阶段的分布"
    else:
        label_map = {"early": "early", "mid": "mid", "late": "late"}
        ylabel = "y_rel_drop (future relative drop)"
        xlabel = "Life stage"
        title = "Outcome distribution by stage"
    values = [df.loc[df["life_stage"] == stage, "y_rel_drop"].to_numpy(dtype=float) for stage in order]

    plt.figure(figsize=(7.8, 5.0))
    plt.boxplot(values, tick_labels=[label_map[s] for s in order], showfliers=False)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_weight_diagnostics_plot(weights_diag_df: pd.DataFrame, output_path: Path) -> None:
    """Save overlap diagnostics plot for ESS and high-quantile weights."""
    x = weights_diag_df["t_eval"].to_numpy(dtype=float)
    ess = weights_diag_df["effective_sample_size"].to_numpy(dtype=float)
    p99 = weights_diag_df["weight_p99"].to_numpy(dtype=float)
    wmax = weights_diag_df["weight_max"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))
    if HAS_CJK_FONT:
        x_label = "评估处理强度 t_eval（充电倍率 C）"
        y1_label = "有效样本量 ESS"
        y2_label = "权重诊断值"
        title = "权重与有效样本量诊断"
    else:
        x_label = "Evaluation treatment level t_eval (C-rate)"
        y1_label = "Effective sample size (ESS)"
        y2_label = "Weight diagnostics"
        title = "Weight and ESS diagnostics"

    ax1.plot(x, ess, color="#2563EB", linewidth=2.0, label="ESS")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.grid(alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(x, p99, color="#F97316", linewidth=1.6, label="weight_p99")
    ax2.plot(x, wmax, color="#DC2626", linewidth=1.3, alpha=0.9, label="weight_max")
    ax2.set_ylabel(y2_label, color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _format_float(value: float, digits: int = 6) -> str:
    """Format float with safe NaN handling."""
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _md_escape(value: str) -> str:
    """Escape text for markdown table cells."""
    return value.replace("|", "\\|")


def _build_parameter_sources(args: argparse.Namespace) -> pd.DataFrame:
    """Build parameter provenance table for report traceability."""
    rows: list[dict] = [
        {
            "parameter_name": "run_time",
            "section": "分析设定",
            "source_file": "运行时系统时钟",
            "source_columns": "N/A",
            "formula_or_rule": "datetime.now()",
            "notes": "报告生成时间，不来自数据表。",
        },
        {
            "parameter_name": "python_executable",
            "section": "分析设定",
            "source_file": "运行时环境",
            "source_columns": "N/A",
            "formula_or_rule": "sys.executable",
            "notes": "用于复现解释器路径。",
        },
        {
            "parameter_name": "python_version",
            "section": "分析设定",
            "source_file": "运行时环境",
            "source_columns": "N/A",
            "formula_or_rule": "sys.version.split()[0]",
            "notes": "用于复现 Python 主版本。",
        },
        {
            "parameter_name": "horizon_cycles",
            "section": "分析设定",
            "source_file": "命令行参数",
            "source_columns": "--horizon-cycles",
            "formula_or_rule": "window_end_cycle = window_start_cycle + horizon_cycles",
            "notes": "本次默认 200，但输出命名不显式包含 200。",
        },
        {
            "parameter_name": "treatment_mode",
            "section": "分析设定",
            "source_file": "命令行参数",
            "source_columns": "--treatment-mode",
            "formula_or_rule": "initial 或 effective_mean",
            "notes": "本次默认 initial。",
        },
        {
            "parameter_name": "exclude_policy_prefix",
            "section": "分析设定",
            "source_file": "命令行参数",
            "source_columns": "--exclude-policy-prefix",
            "formula_or_rule": "policy.startswith(prefix) 的样本被排除",
            "notes": "本次默认排除 VARCHARGE。",
        },
        {
            "parameter_name": "treatment_value",
            "section": "处理变量",
            "source_file": "data/processed/policy_meaning.csv",
            "source_columns": "initial_c_rate, switch_soc_percent, post_switch_c_rate",
            "formula_or_rule": "initial 模式: treatment_value=initial_c_rate; effective_mean 模式: C1*SOC + C2*(1-SOC)",
            "notes": "按 policy 级定义后并入窗口样本。",
        },
        {
            "parameter_name": "y_rel_drop",
            "section": "结果变量",
            "source_file": "data/processed/life_performance.csv",
            "source_columns": "q_discharge, cycles",
            "formula_or_rule": "y_rel_drop=(Q_t - Q_{t+h}) / Q_t",
            "notes": "Q_t 与 Q_{t+h} 由窗口连接得到。",
        },
        {
            "parameter_name": "life_stage",
            "section": "分阶段定义",
            "source_file": "analysis_dataset_windows.csv(中间表)",
            "source_columns": "window_start_cycle",
            "formula_or_rule": "按 window_start_cycle 排名后 qcut 三分位: early/mid/late",
            "notes": "用于分阶段效应估计。",
        },
        {
            "parameter_name": "effect_plus_1c",
            "section": "主效应与分阶段效应",
            "source_file": "delta_plus_1c_curve.csv(中间结果)",
            "source_columns": "delta_global, delta_early, delta_mid, delta_late, weight_*",
            "formula_or_rule": "effect=Σ weight(t)*[m(t+1)-m(t)]",
            "notes": "全局与分阶段分别加权汇总。",
        },
        {
            "parameter_name": "ci_low / ci_high",
            "section": "主效应与分阶段效应",
            "source_file": "analysis_dataset_windows.csv(中间表)",
            "source_columns": "cluster_id",
            "formula_or_rule": "按 cluster_id 进行 bootstrap，取 2.5% 与 97.5% 分位",
            "notes": "bootstrap 次数由 --n-bootstrap 控制。",
        },
        {
            "parameter_name": "n_rows",
            "section": "样本规模",
            "source_file": "analysis_dataset_windows.csv",
            "source_columns": "全部行",
            "formula_or_rule": "n_rows=len(df)",
            "notes": "滚动窗口样本行数。",
        },
        {
            "parameter_name": "n_policy / n_cell / n_policy_cell",
            "section": "样本规模",
            "source_file": "analysis_dataset_windows.csv",
            "source_columns": "policy, cell_code, cluster_id",
            "formula_or_rule": "nunique 统计",
            "notes": "policy_cell 由 policy|cell_code 组成。",
        },
        {
            "parameter_name": "treatment_min / treatment_max / treatment_q01 / treatment_q99 / treatment_std",
            "section": "处理变量诊断",
            "source_file": "analysis_dataset_windows.csv",
            "source_columns": "treatment_value",
            "formula_or_rule": "min/max/quantile/std",
            "notes": "用于支持区间与分布诊断。",
        },
        {
            "parameter_name": "grid_base_min / grid_base_max / grid_base_points / grid_eval_points",
            "section": "估计网格",
            "source_file": "模型配置 + treatment_value",
            "source_columns": "--grid-step, --trim-quantile, treatment_value",
            "formula_or_rule": "build_grids() 生成 base_grid 和 eval_grid",
            "notes": "base_grid 用于 m(t+1)-m(t) 计算。",
        },
        {
            "parameter_name": "bandwidth",
            "section": "核平滑设置",
            "source_file": "analysis_dataset_windows.csv",
            "source_columns": "treatment_value",
            "formula_or_rule": "max(1.06*std*n^(-1/5),0.2)",
            "notes": "用于 kernel/GPS 权重。",
        },
        {
            "parameter_name": "treatment_residual_std",
            "section": "GPS 建模",
            "source_file": "analysis_dataset_windows.csv",
            "source_columns": "treatment_value, switch_soc_percent, post_switch_c_rate, window_start_cycle",
            "formula_or_rule": "LinearRegression 残差标准差",
            "notes": "GPS 密度按高斯残差近似。",
        },
        {
            "parameter_name": "weight_mean / weight_std / weight_p95 / weight_p99 / weight_max / effective_sample_size / clip_threshold",
            "section": "权重诊断",
            "source_file": "diagnostics_weights.csv",
            "source_columns": "各权重统计列",
            "formula_or_rule": "按每个 t_eval 对 raw_w=kernel/gps 统计并裁剪",
            "notes": "用于判断重叠性和稳定性。",
        },
        {
            "parameter_name": "n_bootstrap",
            "section": "复现命令参数",
            "source_file": "命令行参数",
            "source_columns": "--n-bootstrap",
            "formula_or_rule": "bootstrap 重采样次数",
            "notes": f"本次默认 {args.n_bootstrap}。",
        },
        {
            "parameter_name": "grid_step / trim_quantile / weight_clip_quantile / seed / encoding",
            "section": "复现命令参数",
            "source_file": "命令行参数",
            "source_columns": "--grid-step, --trim-quantile, --weight-clip-quantile, --seed, --encoding",
            "formula_or_rule": "直接控制网格、裁剪、随机种子与读写编码",
            "notes": "保证复现可控。",
        },
    ]
    return pd.DataFrame(rows)


def render_markdown_report(
    args: argparse.Namespace,
    effect_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    dose_curve_df: pd.DataFrame,
    delta_curve_df: pd.DataFrame,
    data_df: pd.DataFrame,
    weights_diag_df: pd.DataFrame,
    parameter_sources_df: pd.DataFrame,
) -> str:
    """Render markdown report content."""
    effect_map = {row["group"]: row for row in effect_df.to_dict(orient="records")}
    global_row = effect_map.get("global", {})
    stage_map = {"early": "早期", "mid": "中期", "late": "后期", "global": "全局"}

    y_by_stage = (
        data_df.groupby("life_stage", as_index=False)["y_rel_drop"]
        .mean()
        .set_index("life_stage")["y_rel_drop"]
        .to_dict()
    )
    early_mean = float(y_by_stage.get("early", np.nan))
    late_mean = float(y_by_stage.get("late", np.nan))
    stage_rise = late_mean - early_mean if np.isfinite(early_mean) and np.isfinite(late_mean) else np.nan

    min_ess_row = weights_diag_df.loc[weights_diag_df["effective_sample_size"].idxmin()]
    min_ess = float(min_ess_row["effective_sample_size"])
    min_ess_t = float(min_ess_row["t_eval"])
    max_p99 = float(weights_diag_df["weight_p99"].max())

    delta_global_mean = float(delta_curve_df["delta_global"].mean())
    delta_global_max = float(delta_curve_df["delta_global"].max())
    delta_global_min = float(delta_curve_df["delta_global"].min())

    support_q01_row = overlap_df.loc[overlap_df["metric"] == "treatment_q01", "value"]
    support_q99_row = overlap_df.loc[overlap_df["metric"] == "treatment_q99", "value"]
    support_q01 = float(support_q01_row.iloc[0]) if not support_q01_row.empty else float("nan")
    support_q99 = float(support_q99_row.iloc[0]) if not support_q99_row.empty else float("nan")

    m_min = float(dose_curve_df["m_global"].min())
    m_max = float(dose_curve_df["m_global"].max())

    lines: list[str] = []
    lines.append("# 因果效应报告：初段充电倍率对未来相对容量下降的影响")
    lines.append("")
    lines.append("## 1. 分析设定")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{sys.executable}`")
    lines.append(f"- 预测窗口长度（horizon_cycles）：`{args.horizon_cycles}`")
    lines.append(f"- 处理变量定义（treatment_mode）：`{args.treatment_mode}`")
    lines.append(f"- 排除策略前缀（exclude_policy_prefix）：`{args.exclude_policy_prefix}`")
    lines.append("")
    lines.append("## 2. 主要结论")
    lines.append(
        f"- 全局 +1C 效应：**{_format_float(float(global_row.get('effect_plus_1c', np.nan)))}** "
        f"（95%置信区间：{_format_float(float(global_row.get('ci_low', np.nan)))} ~ "
        f"{_format_float(float(global_row.get('ci_high', np.nan)))})"
    )
    lines.append("")
    lines.append("## 3. 分寿命阶段 +1C 效应")
    lines.append("| 分组 | effect_plus_1c | ci_low | ci_high | n_rows | n_clusters | bootstrap_success |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in effect_df.iterrows():
        lines.append(
            f"| {stage_map.get(row['group'], row['group'])} | {_format_float(float(row['effect_plus_1c']))} | "
            f"{_format_float(float(row['ci_low']))} | {_format_float(float(row['ci_high']))} | "
            f"{int(row['n_rows'])} | {int(row['n_clusters'])} | {int(row['bootstrap_success'])} |"
        )
    lines.append("")
    lines.append("## 4. 关键诊断指标")
    for metric in ["n_rows", "n_policy_cell", "treatment_min", "treatment_max", "bandwidth"]:
        row = overlap_df.loc[overlap_df["metric"] == metric]
        if not row.empty:
            value = row["value"].iloc[0]
            lines.append(f"- {metric}: `{value}`")
    lines.append("")
    lines.append("## 5. 图表解读")
    lines.append("### 图1：剂量-反应曲线")
    lines.append("![图1 剂量-反应曲线](./fig_dose_response.png)")
    lines.append("- X轴含义：处理强度 `t_eval`（充电倍率 C）。")
    lines.append("- Y轴含义：`E[Y | do(T=t)]`，即在干预到倍率 t 时的未来窗口相对容量下降期望。")
    lines.append(
        f"- 关键性结论：全局曲线范围约 `{_format_float(m_min, 5)} ~ {_format_float(m_max, 5)}`，说明不同倍率下损失水平存在系统差异。"
    )
    lines.append("- 业务解释：这张图回答“把倍率设为某个具体值时，预期衰减水平是多少”。")
    lines.append("")
    lines.append("### 图2：全局与分阶段 +1C 效应")
    lines.append("![图2 全局与分阶段+1C效应](./fig_plus1c_by_stage.png)")
    lines.append("- X轴含义：分组（全局、早期、中期、后期）。")
    lines.append("- Y轴含义：`+1C` 对未来相对容量下降的增量影响。")
    lines.append(
        f"- 关键性结论：后期效应最大（`{_format_float(float(effect_map.get('late', {}).get('effect_plus_1c', np.nan)), 6)}`），高于早期阶段。"
    )
    lines.append("- 业务解释：同样增加 1C，在寿命后段带来的额外损失更明显。")
    lines.append("")
    lines.append("### 图3：+1C 局部效应曲线")
    lines.append("![图3 +1C局部效应曲线](./fig_delta_plus1c_curve.png)")
    lines.append("- X轴含义：基准处理强度 `base_t`（当前倍率）。")
    lines.append("- Y轴含义：`m(t+1)-m(t)`，即在该基准倍率处再提高 1C 的局部影响。")
    lines.append(
        f"- 关键性结论：全局局部效应均值约 `{_format_float(delta_global_mean, 6)}`，区间约 `{_format_float(delta_global_min, 6)} ~ {_format_float(delta_global_max, 6)}`。"
    )
    lines.append("- 业务解释：这张图回答“在不同当前倍率下，再加 1C 的边际代价是否一致”。")
    lines.append("")
    lines.append("### 图4：处理变量分布与支持区间")
    lines.append("![图4 处理变量分布](./fig_treatment_distribution.png)")
    lines.append("- X轴含义：处理变量 `treatment_value`（充电倍率 C）。")
    lines.append("- Y轴含义：样本数（直方图频数）。")
    lines.append(
        f"- 关键性结论：主要样本支持区间集中在 `q01={_format_float(support_q01, 3)}` 到 `q99={_format_float(support_q99, 3)}`。"
    )
    lines.append("- 业务解释：结论应优先解释在该支持区间内，避免超出样本支撑范围外推。")
    lines.append("")
    lines.append("### 图5：结果变量分阶段分布")
    lines.append("![图5 结果变量分阶段分布](./fig_outcome_by_stage.png)")
    lines.append("- X轴含义：寿命阶段（早期、中期、后期）。")
    lines.append("- Y轴含义：`y_rel_drop`，即未来窗口相对容量下降。")
    lines.append(
        f"- 关键性结论：后期均值相对早期上升约 `{_format_float(stage_rise, 6)}`。"
    )
    lines.append("- 业务解释：样本本身在后段衰减更快，是解释阶段异质效应的重要背景。")
    lines.append("")
    lines.append("### 图6：权重与有效样本量诊断")
    lines.append("![图6 权重与有效样本量诊断](./fig_weight_diagnostics.png)")
    lines.append("- X轴含义：评估处理强度 `t_eval`（充电倍率 C）。")
    lines.append("- Y轴含义：左轴为 ESS（有效样本量），右轴为高分位权重诊断（`weight_p99`、`weight_max`）。")
    lines.append(
        f"- 关键性结论：最小 ESS 约 `{_format_float(min_ess, 2)}`（出现在 t≈{_format_float(min_ess_t, 2)}），最大 p99 权重约 `{_format_float(max_p99, 2)}`。"
    )
    lines.append("- 业务解释：ESS 过低或高分位权重过大时，局部估计的不确定性会增加。")
    lines.append("")
    lines.append("## 6. 参数来源详解")
    lines.append("- 摘要说明：下表给出报告中关键参数的来源、字段与计算规则。")
    lines.append("- 完整追溯文件：`report_parameter_sources.csv`。")
    lines.append("")
    lines.append("| parameter_name | section | source_file | source_columns | formula_or_rule | notes |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in parameter_sources_df.iterrows():
        lines.append(
            f"| {_md_escape(str(row['parameter_name']))} | {_md_escape(str(row['section']))} | "
            f"{_md_escape(str(row['source_file']))} | {_md_escape(str(row['source_columns']))} | "
            f"{_md_escape(str(row['formula_or_rule']))} | {_md_escape(str(row['notes']))} |"
        )
    lines.append("")
    lines.append("## 7. 复现命令")
    lines.append("```bash")
    lines.append(
        "pipenv run python scripts/estimate_causal_initial_rate_effect.py "
        "--treatment-mode initial --exclude-policy-prefix VARCHARGE"
    )
    lines.append("```")
    lines.append("")
    lines.append("## 8. 说明")
    lines.append("- 本报告估计的是在当前调整变量条件下的“总效应”。")
    lines.append("- 输出目录与文件命名未显式使用 200。")
    return "\n".join(lines)


def collect_library_versions() -> pd.DataFrame:
    """Collect core package versions used by this script."""
    import joblib  # local import to keep top-level minimal
    import matplotlib
    import scipy
    import sklearn

    rows = [
        {"package": "python", "version": sys.version.split()[0]},
        {"package": "numpy", "version": np.__version__},
        {"package": "pandas", "version": pd.__version__},
        {"package": "scipy", "version": scipy.__version__},
        {"package": "scikit-learn", "version": sklearn.__version__},
        {"package": "matplotlib", "version": matplotlib.__version__},
        {"package": "joblib", "version": joblib.__version__},
    ]
    return pd.DataFrame(rows)


def ensure_dir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Run full causal estimation workflow and save outputs."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    data_df = build_analysis_dataset(
        life_path=args.life_performance_path,
        policy_path=args.policy_meaning_path,
        horizon_cycles=args.horizon_cycles,
        treatment_mode=args.treatment_mode,
        exclude_policy_prefix=args.exclude_policy_prefix,
        encoding=args.encoding,
    )

    treatment = data_df["treatment_value"].to_numpy(dtype=float)
    outcome = data_df["y_rel_drop"].to_numpy(dtype=float)
    covariates, _ = build_covariate_matrix(data_df)

    nuisance = fit_nuisance_models(
        treatment=treatment,
        covariates=covariates,
        outcome=outcome,
        seed=args.seed,
    )

    bandwidth = default_bandwidth(treatment)
    base_grid, eval_grid, _, _ = build_grids(
        treatment=treatment,
        step=args.grid_step,
        trim_quantile=args.trim_quantile,
    )

    dr_matrix, weights_diag_df = compute_dr_table(
        treatment=treatment,
        covariates=covariates,
        outcome=outcome,
        nuisance=nuisance,
        eval_grid=eval_grid,
        bandwidth=bandwidth,
        weight_clip_quantile=args.weight_clip_quantile,
    )

    effect_df, dose_curve_df, delta_curve_df = summarize_effects(
        df=data_df,
        dr_matrix=dr_matrix,
        eval_grid=eval_grid,
        base_grid=base_grid,
        treatment=treatment,
        bandwidth=bandwidth,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    overlap_df = build_overlap_diagnostics(
        df=data_df,
        treatment=treatment,
        base_grid=base_grid,
        eval_grid=eval_grid,
        bandwidth=bandwidth,
        nuisance=nuisance,
        args=args,
    )
    version_df = collect_library_versions()
    parameter_sources_df = _build_parameter_sources(args=args)
    report_text = render_markdown_report(
        args=args,
        effect_df=effect_df,
        overlap_df=overlap_df,
        dose_curve_df=dose_curve_df,
        delta_curve_df=delta_curve_df,
        data_df=data_df,
        weights_diag_df=weights_diag_df,
        parameter_sources_df=parameter_sources_df,
    )

    global_effect_df = effect_df.loc[effect_df["group"] == "global"].copy()
    stage_effect_df = effect_df.loc[effect_df["group"] != "global"].copy()

    data_df.to_csv(output_dir / "analysis_dataset_windows.csv", index=False, encoding=args.encoding)
    global_effect_df.to_csv(output_dir / "causal_effect_global.csv", index=False, encoding=args.encoding)
    stage_effect_df.to_csv(output_dir / "causal_effect_by_stage.csv", index=False, encoding=args.encoding)
    dose_curve_df.to_csv(output_dir / "dose_response_curve.csv", index=False, encoding=args.encoding)
    delta_curve_df.to_csv(output_dir / "delta_plus_1c_curve.csv", index=False, encoding=args.encoding)
    overlap_df.to_csv(output_dir / "diagnostics_overlap.csv", index=False, encoding=args.encoding)
    weights_diag_df.to_csv(output_dir / "diagnostics_weights.csv", index=False, encoding=args.encoding)
    version_df.to_csv(output_dir / "runtime_library_versions.csv", index=False, encoding=args.encoding)
    parameter_sources_df.to_csv(
        output_dir / "report_parameter_sources.csv",
        index=False,
        encoding=args.encoding,
    )

    (output_dir / "causal_initial_rate_report.md").write_text(report_text, encoding="utf-8")
    save_dose_response_plot(dose_curve_df=dose_curve_df, output_path=output_dir / "fig_dose_response.png")
    save_stage_effect_plot(effect_df=effect_df, output_path=output_dir / "fig_plus1c_by_stage.png")
    save_delta_plus1c_curve_plot(
        delta_curve_df=delta_curve_df,
        output_path=output_dir / "fig_delta_plus1c_curve.png",
    )
    save_treatment_distribution_plot(
        df=data_df,
        output_path=output_dir / "fig_treatment_distribution.png",
    )
    save_outcome_by_stage_plot(
        df=data_df,
        output_path=output_dir / "fig_outcome_by_stage.png",
    )
    save_weight_diagnostics_plot(
        weights_diag_df=weights_diag_df,
        output_path=output_dir / "fig_weight_diagnostics.png",
    )

    print("Analysis completed.")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
