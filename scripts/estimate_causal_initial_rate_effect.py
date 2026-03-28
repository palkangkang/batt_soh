from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_POLICY_MEANING_PATH = REPO_ROOT / "data" / "processed" / "policy_meaning.csv"
DEFAULT_CHARGE_INTERVAL_FEATURES_PATH = (
    REPO_ROOT / "data" / "processed" / "charge_interval_features.csv"
)
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


STAGE_SCHEME_CONFIGS: dict[str, dict[str, object]] = {
    "fixed4": {
        "scheme_display": "固定分段4段",
        "stage_col": "life_stage_fixed4",
        "stage_order": ["cycle_0_500", "cycle_500_1000", "cycle_1000_1500", "cycle_1500_max"],
        "stage_label_map_cn": {
            "cycle_0_500": "0-500",
            "cycle_500_1000": "500-1000",
            "cycle_1000_1500": "1000-1500",
            "cycle_1500_max": "1500-max",
        },
        "stage_label_map_en": {
            "cycle_0_500": "0-500",
            "cycle_500_1000": "500-1000",
            "cycle_1000_1500": "1000-1500",
            "cycle_1500_max": "1500-max",
        },
    },
    "quantile4": {
        "scheme_display": "分位数分段4段",
        "stage_col": "life_stage_quantile4",
        "stage_order": ["q1", "q2", "q3", "q4"],
        "stage_label_map_cn": {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"},
        "stage_label_map_en": {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"},
    },
}


TREATMENT_MODE_DISPLAY: dict[str, dict[str, str]] = {
    "initial": {
        "long_label": "初始充电倍率",
        "short_label": "初始充电倍率",
        "report_title": "因果效应报告：初始充电倍率对未来相对容量下降的影响",
    },
    "effective_mean": {
        "long_label": "策略平均充电倍率",
        "short_label": "策略平均充电倍率",
        "report_title": "因果效应报告：策略平均充电倍率对未来相对容量下降的影响",
    },
    "window_mean": {
        "long_label": "窗口真实平均充电倍率",
        "short_label": "窗口真实平均充电倍率",
        "report_title": "因果效应报告：窗口真实平均充电倍率对未来相对容量下降的影响",
    },
}


def _get_treatment_mode_display(treatment_mode: str) -> dict[str, str]:
    """Return display metadata for treatment mode with safe fallback."""
    if treatment_mode in TREATMENT_MODE_DISPLAY:
        return TREATMENT_MODE_DISPLAY[treatment_mode]
    long_label = f"充电倍率（{treatment_mode}）"
    return {
        "long_label": long_label,
        "short_label": long_label,
        "report_title": f"因果效应报告：充电倍率对未来相对容量下降的影响（{treatment_mode}）",
    }


def _treatment_mode_label(treatment_mode: str, use_short: bool = False) -> str:
    """Return treatment mode label in Chinese."""
    display = _get_treatment_mode_display(treatment_mode=treatment_mode)
    return display["short_label"] if use_short else display["long_label"]


def _treatment_mode_label_with_code(treatment_mode: str) -> str:
    """Return Chinese label with code identifier."""
    return f"{_treatment_mode_label(treatment_mode=treatment_mode)}（{treatment_mode}）"


def get_stage_label_map(scheme: str) -> dict[str, str]:
    """Return stage label map according to current font capability."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    if HAS_CJK_FONT:
        return cfg["stage_label_map_cn"]  # type: ignore[return-value]
    return cfg["stage_label_map_en"]  # type: ignore[return-value]


def add_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add stage columns for both fixed4 and quantile4 schemes."""
    out = df.copy()
    fixed_cfg = STAGE_SCHEME_CONFIGS["fixed4"]
    fixed_labels = fixed_cfg["stage_order"]  # type: ignore[assignment]
    out["life_stage_fixed4"] = pd.cut(
        out["window_start_cycle"],
        bins=[0, 500, 1000, 1500, float("inf")],
        labels=fixed_labels,
        right=False,
        include_lowest=True,
    ).astype(str)

    q_cfg = STAGE_SCHEME_CONFIGS["quantile4"]
    q_labels = q_cfg["stage_order"]  # type: ignore[assignment]
    rank_series = out["window_start_cycle"].rank(method="first")
    out["life_stage_quantile4"] = pd.qcut(
        rank_series,
        q=4,
        labels=q_labels,
    ).astype(str)
    return out


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
        "--charge-interval-features-path",
        type=Path,
        default=DEFAULT_CHARGE_INTERVAL_FEATURES_PATH,
        help="Path to charge_interval_features.csv, used by treatment_mode=window_mean.",
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
        choices=["initial", "effective_mean", "window_mean"],
        default="initial",
        help="Treatment definition: initial C-rate, policy effective mean C-rate, or window mean C-rate.",
    )
    parser.add_argument(
        "--compare-initial-vs-window-mean",
        action="store_true",
        help="Run initial and window_mean in one execution, then output side-by-side comparison.",
    )
    parser.add_argument(
        "--window-mean-qref-cycles",
        type=int,
        default=20,
        help="For window_mean treatment, use median q_discharge of the first N valid cycles as q_ref.",
    )
    parser.add_argument(
        "--window-mean-clip-quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile to clip window_mean treatment before modeling.",
    )
    parser.add_argument(
        "--window-mean-clip-quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile to clip window_mean treatment before modeling.",
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


def _validate_window_mean_clip(clip_low: float, clip_high: float) -> None:
    """Validate clip quantiles for window_mean treatment."""
    if not (0.0 <= clip_low < 1.0):
        raise ValueError("window_mean_clip_quantile_low must be in [0, 1).")
    if not (0.0 < clip_high <= 1.0):
        raise ValueError("window_mean_clip_quantile_high must be in (0, 1].")
    if clip_low >= clip_high:
        raise ValueError("window_mean clip quantiles must satisfy low < high.")


def _build_window_mean_treatment_cycle(
    life_df: pd.DataFrame,
    charge_interval_path: Path,
    qref_cycles: int,
    clip_low: float,
    clip_high: float,
    encoding: str,
    exclude_policy_prefix: str,
) -> Tuple[pd.DataFrame, dict[str, float]]:
    """Build cycle-level window_mean treatment and diagnostics."""
    if qref_cycles <= 0:
        raise ValueError("window_mean_qref_cycles must be a positive integer.")
    _validate_window_mean_clip(clip_low=clip_low, clip_high=clip_high)

    charge_df = pd.read_csv(
        charge_interval_path,
        encoding=encoding,
        usecols=["policy", "cell_code", "cycles", "delta_ah", "charge_duration_s"],
    )
    charge_df = _to_numeric(charge_df, columns=["cycles", "delta_ah", "charge_duration_s"])
    charge_df = charge_df.dropna(
        subset=["policy", "cell_code", "cycles", "delta_ah", "charge_duration_s"]
    ).copy()
    charge_df["cycles"] = charge_df["cycles"].astype(int)

    if exclude_policy_prefix:
        keep_mask_charge = ~charge_df["policy"].astype(str).str.startswith(exclude_policy_prefix)
        charge_df = charge_df.loc[keep_mask_charge].copy()

    charge_agg = (
        charge_df.groupby(["policy", "cell_code", "cycles"], as_index=False)
        .agg(
            delta_ah_sum=("delta_ah", "sum"),
            charge_duration_s_sum=("charge_duration_s", "sum"),
        )
        .copy()
    )
    charge_agg = charge_agg.loc[
        (charge_agg["delta_ah_sum"] > 0) & (charge_agg["charge_duration_s_sum"] > 0)
    ].copy()
    charge_agg["avg_current_a"] = charge_agg["delta_ah_sum"] / (
        charge_agg["charge_duration_s_sum"] / 3600.0
    )

    life_cells = life_df[["policy", "cell_code"]].drop_duplicates().copy()
    q_ref_df = (
        life_df.sort_values(["policy", "cell_code", "cycles"])
        .groupby(["policy", "cell_code"], as_index=False)
        .head(qref_cycles)
        .groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .median()
        .rename(columns={"q_discharge": "q_ref"})
    )
    q_ref_df = q_ref_df.loc[q_ref_df["q_ref"] > 0].copy()

    treatment_cycle = charge_agg.merge(
        q_ref_df,
        on=["policy", "cell_code"],
        how="inner",
        validate="many_to_one",
    ).copy()
    treatment_cycle["treatment_value_raw"] = treatment_cycle["avg_current_a"] / treatment_cycle["q_ref"]
    treatment_cycle = treatment_cycle.loc[np.isfinite(treatment_cycle["treatment_value_raw"])].copy()
    treatment_cycle = treatment_cycle.loc[treatment_cycle["treatment_value_raw"] > 0].copy()

    if treatment_cycle.empty:
        raise ValueError("No valid cycle-level treatment values for window_mean.")

    raw_low = float(np.quantile(treatment_cycle["treatment_value_raw"], clip_low))
    raw_high = float(np.quantile(treatment_cycle["treatment_value_raw"], clip_high))
    treatment_cycle["treatment_value"] = treatment_cycle["treatment_value_raw"].clip(
        lower=raw_low,
        upper=raw_high,
    )

    life_with_treat = life_df.merge(
        treatment_cycle[["policy", "cell_code", "cycles", "treatment_value_raw"]],
        on=["policy", "cell_code", "cycles"],
        how="left",
    )
    if not life_cells.empty:
        q_ref_match = life_cells.merge(
            q_ref_df[["policy", "cell_code"]].assign(_has_q_ref=1),
            on=["policy", "cell_code"],
            how="left",
        )
        q_ref_coverage = float(q_ref_match["_has_q_ref"].fillna(0).mean())
    else:
        q_ref_coverage = float("nan")
    window_mean_coverage = (
        float(life_with_treat["treatment_value_raw"].notna().mean()) if not life_with_treat.empty else float("nan")
    )
    clip_low_share = float((treatment_cycle["treatment_value_raw"] < raw_low).mean())
    clip_high_share = float((treatment_cycle["treatment_value_raw"] > raw_high).mean())

    diagnostics = {
        "q_ref_coverage": q_ref_coverage,
        "window_mean_coverage": window_mean_coverage,
        "window_mean_raw_min": float(treatment_cycle["treatment_value_raw"].min()),
        "window_mean_raw_max": float(treatment_cycle["treatment_value_raw"].max()),
        "window_mean_raw_q01": float(np.quantile(treatment_cycle["treatment_value_raw"], 0.01)),
        "window_mean_raw_q99": float(np.quantile(treatment_cycle["treatment_value_raw"], 0.99)),
        "window_mean_clip_low": raw_low,
        "window_mean_clip_high": raw_high,
        "window_mean_clip_low_share": clip_low_share,
        "window_mean_clip_high_share": clip_high_share,
        "window_mean_qref_cycles": float(qref_cycles),
        "window_mean_clip_quantile_low": float(clip_low),
        "window_mean_clip_quantile_high": float(clip_high),
    }

    cols = [
        "policy",
        "cell_code",
        "cycles",
        "treatment_value",
        "treatment_value_raw",
        "avg_current_a",
        "q_ref",
    ]
    return treatment_cycle[cols].copy(), diagnostics


def build_analysis_dataset_with_diagnostics(
    life_path: Path,
    policy_path: Path,
    horizon_cycles: int,
    treatment_mode: str,
    exclude_policy_prefix: str,
    encoding: str,
    charge_interval_features_path: Path | None = None,
    window_mean_qref_cycles: int = 20,
    window_mean_clip_quantile_low: float = 0.01,
    window_mean_clip_quantile_high: float = 0.99,
) -> Tuple[pd.DataFrame, dict[str, float]]:
    """Build rolling-window analysis dataset for causal estimation."""
    if horizon_cycles <= 0:
        raise ValueError("horizon_cycles must be a positive integer.")

    diagnostics: dict[str, float] = {}
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

    policy_cov_df = policy_df[
        ["policy", "switch_soc_percent", "post_switch_c_rate", "initial_c_rate"]
    ].copy()

    if treatment_mode in ("initial", "effective_mean"):
        policy_treat_df = policy_df.copy()
        policy_treat_df["treatment_value"] = _build_treatment_column(
            policy_treat_df, treatment_mode=treatment_mode
        )
        policy_treat_df = policy_treat_df.dropna(
            subset=["treatment_value", "switch_soc_percent", "post_switch_c_rate"]
        ).copy()
        merged = life_df.merge(
            policy_treat_df[
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
    elif treatment_mode == "window_mean":
        if charge_interval_features_path is None:
            raise ValueError("charge_interval_features_path is required for treatment_mode=window_mean.")
        treatment_cycle_df, wm_diag = _build_window_mean_treatment_cycle(
            life_df=life_df,
            charge_interval_path=charge_interval_features_path,
            qref_cycles=window_mean_qref_cycles,
            clip_low=window_mean_clip_quantile_low,
            clip_high=window_mean_clip_quantile_high,
            encoding=encoding,
            exclude_policy_prefix=exclude_policy_prefix,
        )
        diagnostics.update(wm_diag)
        merged = life_df.merge(
            treatment_cycle_df[
                [
                    "policy",
                    "cell_code",
                    "cycles",
                    "treatment_value",
                    "treatment_value_raw",
                    "avg_current_a",
                    "q_ref",
                ]
            ],
            on=["policy", "cell_code", "cycles"],
            how="inner",
            validate="one_to_one",
        ).copy()
        merged = merged.merge(
            policy_cov_df[["policy", "switch_soc_percent", "post_switch_c_rate"]],
            on="policy",
            how="inner",
            validate="many_to_one",
        ).copy()
    else:
        raise ValueError(f"Unsupported treatment_mode: {treatment_mode}")

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

    window_df = add_stage_columns(window_df)

    window_df = window_df.sort_values(
        ["policy", "cell_code", "window_start_cycle"]
    ).reset_index(drop=True)
    if window_df.empty:
        raise ValueError("No valid rolling-window samples were built.")
    return window_df, diagnostics


def build_analysis_dataset(
    life_path: Path,
    policy_path: Path,
    horizon_cycles: int,
    treatment_mode: str,
    exclude_policy_prefix: str,
    encoding: str,
    charge_interval_features_path: Path | None = None,
    window_mean_qref_cycles: int = 20,
    window_mean_clip_quantile_low: float = 0.01,
    window_mean_clip_quantile_high: float = 0.99,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning dataset only."""
    window_df, _ = build_analysis_dataset_with_diagnostics(
        life_path=life_path,
        policy_path=policy_path,
        horizon_cycles=horizon_cycles,
        treatment_mode=treatment_mode,
        exclude_policy_prefix=exclude_policy_prefix,
        encoding=encoding,
        charge_interval_features_path=charge_interval_features_path,
        window_mean_qref_cycles=window_mean_qref_cycles,
        window_mean_clip_quantile_low=window_mean_clip_quantile_low,
        window_mean_clip_quantile_high=window_mean_clip_quantile_high,
    )
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
    """Backward-compatible wrapper using fixed4 stage scheme."""
    return summarize_effects_by_scheme(
        df=df,
        dr_matrix=dr_matrix,
        eval_grid=eval_grid,
        base_grid=base_grid,
        treatment=treatment,
        bandwidth=bandwidth,
        n_bootstrap=n_bootstrap,
        seed=seed,
        scheme="fixed4",
    )


def summarize_effects_by_scheme(
    df: pd.DataFrame,
    dr_matrix: np.ndarray,
    eval_grid: np.ndarray,
    base_grid: np.ndarray,
    treatment: np.ndarray,
    bandwidth: float,
    n_bootstrap: int,
    seed: int,
    scheme: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize global and stage effects for a specified stage scheme."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    stage_col = str(cfg["stage_col"])
    stage_order = list(cfg["stage_order"])
    stage_label_map = get_stage_label_map(scheme)

    idx_base = np.searchsorted(eval_grid, base_grid)
    idx_plus = np.searchsorted(eval_grid, base_grid + 1.0)

    dose_rows: dict[str, np.ndarray] = {"t_eval": eval_grid, "m_global": dr_matrix.mean(axis=0)}
    for stage in stage_order:
        mask = (df[stage_col].to_numpy() == stage)
        if mask.any():
            dose_rows[f"m_{stage}"] = dr_matrix[mask].mean(axis=0)
        else:
            dose_rows[f"m_{stage}"] = np.full(eval_grid.shape[0], np.nan)
    dose_curve_df = pd.DataFrame(dose_rows)
    dose_curve_df["scheme"] = scheme

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
            "scheme": scheme,
            "group": "global",
            "group_label": "全局" if HAS_CJK_FONT else "global",
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

    for stage_idx, stage in enumerate(stage_order):
        mask = (df[stage_col].to_numpy() == stage)
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
                    "scheme": scheme,
                    "group": stage,
                    "group_label": stage_label_map.get(stage, stage),
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
                    "scheme": scheme,
                    "group": stage,
                    "group_label": stage_label_map.get(stage, stage),
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
    delta_curve_df["scheme"] = scheme
    return effect_df, dose_curve_df, delta_curve_df


def build_stage_sample_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Build sample distribution table for both stage schemes."""
    rows: list[dict] = []
    for scheme, cfg in STAGE_SCHEME_CONFIGS.items():
        stage_col = str(cfg["stage_col"])
        stage_order = list(cfg["stage_order"])
        label_map = get_stage_label_map(scheme)
        for stage in stage_order:
            part = df.loc[df[stage_col] == stage].copy()
            rows.append(
                {
                    "scheme": scheme,
                    "scheme_display": cfg["scheme_display"],
                    "group": stage,
                    "group_label": label_map.get(stage, stage),
                    "n_rows": int(part.shape[0]),
                    "n_clusters": int(part["cluster_id"].nunique()),
                    "window_start_cycle_min": float(part["window_start_cycle"].min()) if not part.empty else float("nan"),
                    "window_start_cycle_max": float(part["window_start_cycle"].max()) if not part.empty else float("nan"),
                    "window_start_cycle_mean": float(part["window_start_cycle"].mean()) if not part.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_overlap_diagnostics(
    df: pd.DataFrame,
    treatment: np.ndarray,
    base_grid: np.ndarray,
    eval_grid: np.ndarray,
    bandwidth: float,
    nuisance: NuisanceArtifacts,
    args: argparse.Namespace,
    extra_metrics: dict[str, float] | None = None,
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
    if extra_metrics:
        for key, value in extra_metrics.items():
            rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)


def save_dose_response_plot(
    dose_curve_df: pd.DataFrame,
    output_path: Path,
    scheme: str,
) -> None:
    """Save dose-response curve figure."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    stage_order = list(cfg["stage_order"])
    stage_label_map = get_stage_label_map(scheme)
    if HAS_CJK_FONT:
        xlabel = "处理强度 t（充电倍率 C）"
        ylabel = "估计的 E[Y | do(T=t)]"
        title = f"剂量-反应曲线（{cfg['scheme_display']}）"
    else:
        xlabel = "Treatment level t (C-rate)"
        ylabel = "Estimated E[Y | do(T=t)]"
        title = f"Dose-response curve ({cfg['scheme_display']})"
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(
        dose_curve_df["t_eval"],
        dose_curve_df["m_global"],
        label="全局" if HAS_CJK_FONT else "global",
        linewidth=2.0,
    )
    for stage in stage_order:
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


def save_stage_effect_plot(
    effect_df: pd.DataFrame,
    output_path: Path,
    scheme: str,
) -> None:
    """Save stage-wise +1C effect bar chart with confidence intervals."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    stage_order = list(cfg["stage_order"])
    order = ["global"] + stage_order
    if HAS_CJK_FONT:
        label_map = {"global": "全局", **get_stage_label_map(scheme)}
        ylabel = "+1C 对相对容量下降的估计影响"
        title = f"全局与分阶段 +1C 效应（{cfg['scheme_display']}）"
    else:
        label_map = {"global": "global", **get_stage_label_map(scheme)}
        ylabel = "Estimated +1C effect on relative drop"
        title = f"Global and stage-wise +1C effect ({cfg['scheme_display']})"
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


def save_delta_plus1c_curve_plot(
    delta_curve_df: pd.DataFrame,
    output_path: Path,
    scheme: str,
) -> None:
    """Save +1C local effect curve by baseline treatment level."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    stage_order = list(cfg["stage_order"])
    stage_label_map = get_stage_label_map(scheme)
    if HAS_CJK_FONT:
        xlabel = "基准处理强度 base_t（充电倍率 C）"
        ylabel = "局部 +1C 效应：m(t+1)-m(t)"
        title = f"+1C 局部效应曲线（{cfg['scheme_display']}）"
    else:
        xlabel = "Baseline treatment level base_t (C-rate)"
        ylabel = "Local +1C effect: m(t+1)-m(t)"
        title = f"Local +1C effect curve ({cfg['scheme_display']})"
    plt.figure(figsize=(8.8, 5.2))
    plt.plot(
        delta_curve_df["base_t"],
        delta_curve_df["delta_global"],
        label="全局" if HAS_CJK_FONT else "global",
        linewidth=2.0,
    )
    for stage in stage_order:
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


def save_outcome_by_stage_plot(
    df: pd.DataFrame,
    output_path: Path,
    scheme: str,
) -> None:
    """Save outcome distribution by life stage."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    stage_col = str(cfg["stage_col"])
    order = list(cfg["stage_order"])
    if HAS_CJK_FONT:
        label_map = get_stage_label_map(scheme)
        ylabel = "y_rel_drop（未来窗口相对容量下降）"
        xlabel = "寿命阶段"
        title = f"结果变量分阶段分布（{cfg['scheme_display']}）"
    else:
        label_map = get_stage_label_map(scheme)
        ylabel = "y_rel_drop (future relative drop)"
        xlabel = "Life stage"
        title = f"Outcome distribution by stage ({cfg['scheme_display']})"
    values = [df.loc[df[stage_col] == stage, "y_rel_drop"].to_numpy(dtype=float) for stage in order]

    plt.figure(figsize=(7.8, 5.0))
    plt.boxplot(values, tick_labels=[label_map[s] for s in order], showfliers=False)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_stage_effect_compare_plot(effect_stage_df: pd.DataFrame, output_path: Path) -> None:
    """Save comparison figure for scheme × stage +1C effects."""
    plot_df = effect_stage_df.copy().reset_index(drop=True)
    if HAS_CJK_FONT:
        x_labels = [
            f"{('固定4段' if s == 'fixed4' else '分位4段')}|{g}"
            for s, g in zip(plot_df["scheme"], plot_df["group_label"])
        ]
        ylabel = "+1C 对相对容量下降的估计影响"
        title = "两种分段方案的阶段效应对比"
    else:
        x_labels = [f"{s}|{g}" for s, g in zip(plot_df["scheme"], plot_df["group_label"])]
        ylabel = "Estimated +1C effect on relative drop"
        title = "Stage effect comparison across two schemes"

    y = plot_df["effect_plus_1c"].to_numpy(dtype=float)
    ci_low = plot_df["ci_low"].to_numpy(dtype=float)
    ci_high = plot_df["ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([np.where(np.isfinite(ci_low), y - ci_low, np.nan), np.where(np.isfinite(ci_high), ci_high - y, np.nan)])

    plt.figure(figsize=(11.0, 5.2))
    x = np.arange(plot_df.shape[0])
    colors = ["#2563EB" if s == "fixed4" else "#F97316" for s in plot_df["scheme"]]
    plt.bar(x, y, color=colors, alpha=0.88, width=0.72)
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1.05)
    plt.xticks(x, x_labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
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


def _report_title_by_treatment_mode(treatment_mode: str) -> str:
    """Return report title according to treatment mode."""
    return _get_treatment_mode_display(treatment_mode=treatment_mode)["report_title"]


def _build_parameter_sources(args: argparse.Namespace) -> pd.DataFrame:
    """Build parameter provenance table for report traceability."""
    window_mean_label = _treatment_mode_label("window_mean")
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
            "formula_or_rule": "initial / effective_mean / window_mean",
            "notes": f"window_mean 为{window_mean_label}口径。",
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
            "source_file": "data/processed/policy_meaning.csv + data/processed/charge_interval_features.csv + data/processed/life_performance.csv",
            "source_columns": "initial_c_rate, switch_soc_percent, post_switch_c_rate, delta_ah, charge_duration_s, q_discharge",
            "formula_or_rule": "initial: initial_c_rate; effective_mean: C1*SOC + C2*(1-SOC); window_mean: (Σdelta_ah/Σduration_h)/q_ref",
            "notes": "window_mean 中 q_ref 取每 policy+cell 前 N 个有效循环 q_discharge 中位数，并按分位数裁剪后用于建模。",
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
            "formula_or_rule": "并行双方案：fixed4=[0,500)/[500,1000)/[1000,1500)/[1500,+∞)；quantile4=rank(window_start_cycle)后qcut(4)",
            "notes": "固定4段与分位数4段并行估计，用于敏感性对比。",
        },
        {
            "parameter_name": "effect_plus_1c",
            "section": "主效应与分阶段效应",
            "source_file": "delta_plus_1c_curve.csv(中间结果)",
            "source_columns": "delta_global, delta_<stage>, weight_global, weight_<stage>",
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
    global_effect_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    effect_fixed_df: pd.DataFrame,
    effect_quantile_df: pd.DataFrame,
    dose_curve_fixed_df: pd.DataFrame,
    dose_curve_quantile_df: pd.DataFrame,
    delta_curve_fixed_df: pd.DataFrame,
    delta_curve_quantile_df: pd.DataFrame,
    data_df: pd.DataFrame,
    weights_diag_df: pd.DataFrame,
    effect_stage_compare_df: pd.DataFrame,
    stage_sample_compare_df: pd.DataFrame,
    parameter_sources_df: pd.DataFrame,
    figure_suffix: str = "",
) -> str:
    """Render markdown report content."""
    global_row = global_effect_df.iloc[0].to_dict() if not global_effect_df.empty else {}

    fixed_cfg = STAGE_SCHEME_CONFIGS["fixed4"]
    quant_cfg = STAGE_SCHEME_CONFIGS["quantile4"]
    fixed_order = list(fixed_cfg["stage_order"])
    quant_order = list(quant_cfg["stage_order"])

    fixed_label_map = get_stage_label_map("fixed4")
    quant_label_map = get_stage_label_map("quantile4")

    fixed_stage_col = str(fixed_cfg["stage_col"])
    quant_stage_col = str(quant_cfg["stage_col"])

    fixed_stage_means = (
        data_df.groupby(fixed_stage_col, as_index=False)["y_rel_drop"]
        .mean()
        .set_index(fixed_stage_col)["y_rel_drop"]
        .to_dict()
    )
    stage_rise = float(fixed_stage_means.get("cycle_1500_max", np.nan)) - float(
        fixed_stage_means.get("cycle_0_500", np.nan)
    )
    quant_stage_means = (
        data_df.groupby(quant_stage_col, as_index=False)["y_rel_drop"]
        .mean()
        .set_index(quant_stage_col)["y_rel_drop"]
        .to_dict()
    )
    stage_rise_quantile = float(quant_stage_means.get("q4", np.nan)) - float(
        quant_stage_means.get("q1", np.nan)
    )

    min_ess_row = weights_diag_df.loc[weights_diag_df["effective_sample_size"].idxmin()]
    min_ess = float(min_ess_row["effective_sample_size"])
    min_ess_t = float(min_ess_row["t_eval"])
    max_p99 = float(weights_diag_df["weight_p99"].max())

    delta_global_mean_fixed = float(delta_curve_fixed_df["delta_global"].mean())
    delta_global_max_fixed = float(delta_curve_fixed_df["delta_global"].max())
    delta_global_min_fixed = float(delta_curve_fixed_df["delta_global"].min())
    delta_global_mean_quantile = float(delta_curve_quantile_df["delta_global"].mean())
    delta_global_max_quantile = float(delta_curve_quantile_df["delta_global"].max())
    delta_global_min_quantile = float(delta_curve_quantile_df["delta_global"].min())

    support_q01_row = overlap_df.loc[overlap_df["metric"] == "treatment_q01", "value"]
    support_q99_row = overlap_df.loc[overlap_df["metric"] == "treatment_q99", "value"]
    support_q01 = float(support_q01_row.iloc[0]) if not support_q01_row.empty else float("nan")
    support_q99 = float(support_q99_row.iloc[0]) if not support_q99_row.empty else float("nan")

    m_min_fixed = float(dose_curve_fixed_df["m_global"].min())
    m_max_fixed = float(dose_curve_fixed_df["m_global"].max())
    m_min_quantile = float(dose_curve_quantile_df["m_global"].min())
    m_max_quantile = float(dose_curve_quantile_df["m_global"].max())

    fixed_compare = effect_stage_compare_df.loc[effect_stage_compare_df["scheme"] == "fixed4"].copy()
    quant_compare = effect_stage_compare_df.loc[effect_stage_compare_df["scheme"] == "quantile4"].copy()
    fixed_compare = fixed_compare.set_index("group").reindex(fixed_order).reset_index()
    quant_compare = quant_compare.set_index("group").reindex(quant_order).reset_index()

    fixed_effect = fixed_compare["effect_plus_1c"].to_numpy(dtype=float)
    quant_effect = quant_compare["effect_plus_1c"].to_numpy(dtype=float)

    fixed_pos = int(np.sum(fixed_effect > 0))
    quant_pos = int(np.sum(quant_effect > 0))
    all_effect = np.concatenate([fixed_effect, quant_effect])
    all_effect = all_effect[np.isfinite(all_effect)]
    if all_effect.size > 0:
        sign_values = np.sign(all_effect)
        sign_values = sign_values[sign_values != 0]
        if sign_values.size > 0 and np.unique(sign_values).size == 1:
            direction_consistency_ratio = 1.0
            direction_consistency_text = (
                "全部正向" if float(sign_values[0]) > 0 else "全部负向"
            )
        elif sign_values.size > 0:
            dominant_sign = 1.0 if np.sum(sign_values > 0) >= np.sum(sign_values < 0) else -1.0
            direction_consistency_ratio = float(np.mean(sign_values == dominant_sign))
            direction_consistency_text = (
                "以正向为主" if dominant_sign > 0 else "以负向为主"
            )
        else:
            direction_consistency_ratio = float("nan")
            direction_consistency_text = "零效应占主导"
    else:
        direction_consistency_ratio = float("nan")
        direction_consistency_text = "无可用阶段效应"

    fixed_first_last_diff = float(
        fixed_compare.loc[fixed_compare["group"] == "cycle_1500_max", "effect_plus_1c"].iloc[0]
    ) - float(fixed_compare.loc[fixed_compare["group"] == "cycle_0_500", "effect_plus_1c"].iloc[0])
    quant_first_last_diff = float(
        quant_compare.loc[quant_compare["group"] == "q4", "effect_plus_1c"].iloc[0]
    ) - float(quant_compare.loc[quant_compare["group"] == "q1", "effect_plus_1c"].iloc[0])

    fixed_ci_width = (
        fixed_compare["ci_high"].to_numpy(dtype=float) - fixed_compare["ci_low"].to_numpy(dtype=float)
    )
    quant_ci_width = (
        quant_compare["ci_high"].to_numpy(dtype=float) - quant_compare["ci_low"].to_numpy(dtype=float)
    )
    fixed_ci_width = fixed_ci_width[np.isfinite(fixed_ci_width)]
    quant_ci_width = quant_ci_width[np.isfinite(quant_ci_width)]
    fixed_ci_mean = float(np.mean(fixed_ci_width)) if fixed_ci_width.size > 0 else float("nan")
    fixed_ci_max = float(np.max(fixed_ci_width)) if fixed_ci_width.size > 0 else float("nan")
    quant_ci_mean = float(np.mean(quant_ci_width)) if quant_ci_width.size > 0 else float("nan")
    quant_ci_max = float(np.max(quant_ci_width)) if quant_ci_width.size > 0 else float("nan")

    fixed_n_rows = stage_sample_compare_df.loc[
        stage_sample_compare_df["scheme"] == "fixed4", "n_rows"
    ].to_numpy(dtype=float)
    quant_n_rows = stage_sample_compare_df.loc[
        stage_sample_compare_df["scheme"] == "quantile4", "n_rows"
    ].to_numpy(dtype=float)
    fixed_n_mean = float(np.mean(fixed_n_rows)) if fixed_n_rows.size > 0 else float("nan")
    quant_n_mean = float(np.mean(quant_n_rows)) if quant_n_rows.size > 0 else float("nan")
    fixed_n_std = float(np.std(fixed_n_rows, ddof=0)) if fixed_n_rows.size > 0 else float("nan")
    quant_n_std = float(np.std(quant_n_rows, ddof=0)) if quant_n_rows.size > 0 else float("nan")
    fixed_n_cv = fixed_n_std / fixed_n_mean if fixed_n_mean > 0 else float("nan")
    quant_n_cv = quant_n_std / quant_n_mean if quant_n_mean > 0 else float("nan")
    fixed_n_ratio = (
        float(np.max(fixed_n_rows) / np.min(fixed_n_rows))
        if fixed_n_rows.size > 0 and np.min(fixed_n_rows) > 0
        else float("nan")
    )
    quant_n_ratio = (
        float(np.max(quant_n_rows) / np.min(quant_n_rows))
        if quant_n_rows.size > 0 and np.min(quant_n_rows) > 0
        else float("nan")
    )
    suffix = f"_{figure_suffix}" if figure_suffix else ""

    def fig_path(base_name: str) -> str:
        return f"./{base_name}{suffix}.png"

    lines: list[str] = []
    lines.append(f"# {_report_title_by_treatment_mode(str(args.treatment_mode))}")
    lines.append("")
    lines.append("## 1. 分析设定")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{sys.executable}`")
    lines.append(f"- 预测窗口长度（horizon_cycles）：`{args.horizon_cycles}`")
    lines.append(
        f"- 处理变量定义（treatment_mode）：{_treatment_mode_label_with_code(str(args.treatment_mode))}"
    )
    lines.append(f"- 排除策略前缀（exclude_policy_prefix）：`{args.exclude_policy_prefix}`")
    lines.append("")
    lines.append("## 2. 主要结论")
    lines.append(
        f"- 全局 +1C 效应：**{_format_float(float(global_row.get('effect_plus_1c', np.nan)))}** "
        f"（95%置信区间：{_format_float(float(global_row.get('ci_low', np.nan)))} ~ "
        f"{_format_float(float(global_row.get('ci_high', np.nan)))})"
    )
    lines.append("")
    lines.append("## 3. 分段方案并行结果")
    lines.append("### 3.1 固定分段4段（0-500 / 500-1000 / 1000-1500 / 1500-max）")
    lines.append("| 分组 | effect_plus_1c | ci_low | ci_high | n_rows | n_clusters | bootstrap_success |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in fixed_compare.iterrows():
        lines.append(
            f"| {fixed_label_map.get(str(row['group']), str(row['group']))} | {_format_float(float(row['effect_plus_1c']))} | "
            f"{_format_float(float(row['ci_low']))} | {_format_float(float(row['ci_high']))} | "
            f"{int(row['n_rows'])} | {int(row['n_clusters'])} | {int(row['bootstrap_success'])} |"
        )
    lines.append("")
    lines.append("### 3.2 分位数分段4段（Q1 / Q2 / Q3 / Q4）")
    lines.append("| 分组 | effect_plus_1c | ci_low | ci_high | n_rows | n_clusters | bootstrap_success |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in quant_compare.iterrows():
        lines.append(
            f"| {quant_label_map.get(str(row['group']), str(row['group']))} | {_format_float(float(row['effect_plus_1c']))} | "
            f"{_format_float(float(row['ci_low']))} | {_format_float(float(row['ci_high']))} | "
            f"{int(row['n_rows'])} | {int(row['n_clusters'])} | {int(row['bootstrap_success'])} |"
        )
    lines.append("")
    lines.append("### 3.3 分段方案对比摘要（详见图11）")
    lines.append("- 固定分段强调工程阈值可解释性，分位数分段强调统计稳健性。")
    lines.append("- 对比可视化与图表解读见“图11：固定分段与分位数分段阶段效应对比”。")
    lines.append("")
    lines.append("### 3.4 两方案样本量对比")
    lines.append("| scheme | 分组 | n_rows | n_clusters | start_cycle_min | start_cycle_max |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in stage_sample_compare_df.iterrows():
        lines.append(
            f"| {row['scheme_display']} | {row['group_label']} | {int(row['n_rows'])} | {int(row['n_clusters'])} | "
            f"{_format_float(float(row['window_start_cycle_min']), 0)} | {_format_float(float(row['window_start_cycle_max']), 0)} |"
        )
    lines.append("")
    lines.append("### 3.5 两方案量化对比结论")
    lines.append(
        f"- 阶段效应方向一致性：固定分段正向阶段 `{fixed_pos}/4`，分位数分段正向阶段 `{quant_pos}/4`，"
        f"跨方案一致性约 `{_format_float(direction_consistency_ratio * 100.0, 2)}%`（{direction_consistency_text}）。"
    )
    lines.append(
        f"- 首末阶段效应差：固定分段 `1500-max - 0-500 = {_format_float(fixed_first_last_diff, 6)}`；"
        f"分位数分段 `Q4 - Q1 = {_format_float(quant_first_last_diff, 6)}`。"
    )
    lines.append(
        f"- 不确定性对比（CI宽度）：固定分段均值 `{_format_float(fixed_ci_mean, 6)}`、最大 `{_format_float(fixed_ci_max, 6)}`；"
        f"分位数分段均值 `{_format_float(quant_ci_mean, 6)}`、最大 `{_format_float(quant_ci_max, 6)}`。"
    )
    lines.append(
        f"- 样本均衡性（n_rows离散度）：固定分段 CV=`{_format_float(fixed_n_cv, 4)}`、max/min=`{_format_float(fixed_n_ratio, 2)}`；"
        f"分位数分段 CV=`{_format_float(quant_n_cv, 4)}`、max/min=`{_format_float(quant_n_ratio, 2)}`。"
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
    lines.append("### 5.1 固定分段4段（4张）")
    lines.append("#### 图1：剂量-反应曲线（固定分段4段）")
    lines.append(f"![图1 剂量-反应曲线]({fig_path('fig_dose_response')})")
    lines.append("- X轴含义：处理强度 `t_eval`（充电倍率 C）。")
    lines.append("- Y轴含义：`E[Y | do(T=t)]`，即在干预到倍率 t 时的未来窗口相对容量下降期望。")
    lines.append(
        f"- 关键性结论：固定分段全局曲线范围约 `{_format_float(m_min_fixed, 5)} ~ {_format_float(m_max_fixed, 5)}`，说明不同倍率下损失水平存在系统差异。"
    )
    lines.append("- 业务解释：这张图回答“把倍率设为某个具体值时，预期衰减水平是多少”。")
    lines.append("")
    lines.append("#### 图2：全局与分阶段 +1C 效应（固定分段4段）")
    lines.append(f"![图2 全局与分阶段+1C效应]({fig_path('fig_plus1c_by_stage')})")
    lines.append("- X轴含义：分组（全局、0-500、500-1000、1000-1500、1500-max）。")
    lines.append("- Y轴含义：`+1C` 对未来相对容量下降的增量影响。")
    fixed_last = fixed_compare.iloc[-1]["effect_plus_1c"] if not fixed_compare.empty else float("nan")
    lines.append(
        f"- 关键性结论：固定分段后段效应（1500-max）约 `{_format_float(float(fixed_last), 6)}`。"
    )
    lines.append("- 业务解释：同样增加 1C，在寿命后段带来的额外损失更明显。")
    lines.append("")
    lines.append("#### 图3：+1C 局部效应曲线（固定分段4段）")
    lines.append(f"![图3 +1C局部效应曲线]({fig_path('fig_delta_plus1c_curve')})")
    lines.append("- X轴含义：基准处理强度 `base_t`（当前倍率）。")
    lines.append("- Y轴含义：`m(t+1)-m(t)`，即在该基准倍率处再提高 1C 的局部影响。")
    lines.append(
        f"- 关键性结论：固定分段下，全局局部效应均值约 `{_format_float(delta_global_mean_fixed, 6)}`，区间约 `{_format_float(delta_global_min_fixed, 6)} ~ {_format_float(delta_global_max_fixed, 6)}`。"
    )
    lines.append("- 业务解释：这张图回答“在不同当前倍率下，再加 1C 的边际代价是否一致”。")
    lines.append("")
    lines.append("#### 图4：结果变量分阶段分布（固定分段4段）")
    lines.append(f"![图4 结果变量分阶段分布（固定分段）]({fig_path('fig_outcome_by_stage')})")
    lines.append("- X轴含义：寿命阶段（固定分段4段）。")
    lines.append("- Y轴含义：`y_rel_drop`，即未来窗口相对容量下降。")
    lines.append(
        f"- 关键性结论：固定分段下，1500-max 阶段均值相对 0-500 阶段变化 `{_format_float(stage_rise, 6)}`。"
    )
    lines.append("- 业务解释：该图给出阶段基线差异，是解释异质效应的背景信息。")
    lines.append("")
    lines.append("### 5.2 分位数分段4段（4张）")
    lines.append("#### 图5：剂量-反应曲线（分位数分段4段）")
    lines.append(f"![图5 剂量-反应曲线（分位数）]({fig_path('fig_dose_response_quantile4')})")
    lines.append("- X轴含义：处理强度 `t_eval`（充电倍率 C）。")
    lines.append("- Y轴含义：`E[Y | do(T=t)]`，即在干预到倍率 t 时的未来窗口相对容量下降期望。")
    lines.append(
        f"- 关键性结论：分位数分段全局曲线范围约 `{_format_float(m_min_quantile, 5)} ~ {_format_float(m_max_quantile, 5)}`。"
    )
    lines.append("- 业务解释：用于在样本更均衡分段下复核倍率干预的整体趋势。")
    lines.append("")
    lines.append("#### 图6：全局与分阶段 +1C 效应（分位数分段4段）")
    lines.append(f"![图6 全局与分阶段+1C效应（分位数）]({fig_path('fig_plus1c_by_stage_quantile4')})")
    lines.append("- X轴含义：分组（全局、Q1、Q2、Q3、Q4）。")
    lines.append("- Y轴含义：`+1C` 对未来相对容量下降的增量影响。")
    quant_last = quant_compare.iloc[-1]["effect_plus_1c"] if not quant_compare.empty else float("nan")
    lines.append(
        f"- 关键性结论：分位数分段后段效应（Q4）约 `{_format_float(float(quant_last), 6)}`。"
    )
    lines.append("- 业务解释：在样本量更均衡的阶段定义下，后段仍表现出更高边际损失。")
    lines.append("")
    lines.append("#### 图7：+1C 局部效应曲线（分位数分段4段）")
    lines.append(f"![图7 +1C局部效应曲线（分位数）]({fig_path('fig_delta_plus1c_curve_quantile4')})")
    lines.append("- X轴含义：基准处理强度 `base_t`（当前倍率）。")
    lines.append("- Y轴含义：`m(t+1)-m(t)`，即在该基准倍率处再提高 1C 的局部影响。")
    lines.append(
        f"- 关键性结论：分位数分段下，全局局部效应均值约 `{_format_float(delta_global_mean_quantile, 6)}`，区间约 `{_format_float(delta_global_min_quantile, 6)} ~ {_format_float(delta_global_max_quantile, 6)}`。"
    )
    lines.append("- 业务解释：用于检验在分位数分段口径下，边际代价曲线是否与固定分段一致。")
    lines.append("")
    lines.append("#### 图8：结果变量分阶段分布（分位数分段4段）")
    lines.append(f"![图8 结果变量分阶段分布（分位数）]({fig_path('fig_outcome_by_stage_quantile4')})")
    lines.append("- X轴含义：寿命阶段（分位数分段4段，Q1~Q4）。")
    lines.append("- Y轴含义：`y_rel_drop`，即未来窗口相对容量下降。")
    lines.append(
        f"- 关键性结论：分位数分段下，Q4 阶段均值相对 Q1 阶段变化 `{_format_float(stage_rise_quantile, 6)}`。"
    )
    lines.append("- 业务解释：在等样本量切分下比较衰减分布，减少样本不均衡干扰。")
    lines.append("")
    lines.append("### 5.3 通用诊断图（2张）")
    lines.append("#### 图9：处理变量分布与支持区间")
    lines.append(f"![图9 处理变量分布]({fig_path('fig_treatment_distribution')})")
    lines.append("- X轴含义：处理变量 `treatment_value`（充电倍率 C）。")
    lines.append("- Y轴含义：样本数（直方图频数）。")
    lines.append(
        f"- 关键性结论：主要样本支持区间集中在 `q01={_format_float(support_q01, 3)}` 到 `q99={_format_float(support_q99, 3)}`。"
    )
    lines.append("- 业务解释：结论应优先解释在该支持区间内，避免超出样本支撑范围外推。")
    lines.append("")
    lines.append("#### 图10：权重与有效样本量诊断")
    lines.append(f"![图10 权重与有效样本量诊断]({fig_path('fig_weight_diagnostics')})")
    lines.append("- X轴含义：评估处理强度 `t_eval`（充电倍率 C）。")
    lines.append("- Y轴含义：左轴为 ESS（有效样本量），右轴为高分位权重诊断（`weight_p99`、`weight_max`）。")
    lines.append(
        f"- 关键性结论：最小 ESS 约 `{_format_float(min_ess, 2)}`（出现在 t≈{_format_float(min_ess_t, 2)}），最大 p99 权重约 `{_format_float(max_p99, 2)}`。"
    )
    lines.append("- 业务解释：ESS 过低或高分位权重过大时，局部估计的不确定性会增加。")
    lines.append("")
    lines.append("### 5.4 方案对比图（1张）")
    lines.append("#### 图11：固定分段与分位数分段阶段效应对比")
    lines.append(f"![图11 两种分段方案阶段效应对比]({fig_path('fig_plus1c_by_stage_compare')})")
    lines.append("- X轴含义：`scheme × stage` 组合分组（固定4段与分位数4段）。")
    lines.append("- Y轴含义：各阶段 `+1C` 对未来相对容量下降的估计影响（含置信区间）。")
    lines.append(
        f"- 关键性结论：固定分段首末差 `{_format_float(fixed_first_last_diff, 6)}`、分位数分段首末差 `{_format_float(quant_first_last_diff, 6)}`，且分位数分段 CI 宽度均值更小（`{_format_float(quant_ci_mean, 6)}` vs `{_format_float(fixed_ci_mean, 6)}`）。"
    )
    lines.append("- 业务解释：两方案方向结论一致时，固定分段强调工程阈值可解释性，分位数分段强调统计稳健性。")
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
        f"--treatment-mode {args.treatment_mode} "
        f"--exclude-policy-prefix {args.exclude_policy_prefix}"
    )
    lines.append("```")
    lines.append("")
    lines.append("## 8. 说明")
    lines.append("- 本报告估计的是在当前调整变量条件下的“总效应”。")
    lines.append("- 分段方案并行输出用于敏感性分析：固定分段偏工程可解释，分位数分段偏统计稳健。")
    lines.append("- 输出目录与文件命名未显式使用 200。")
    return "\n".join(lines)


def intersect_window_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only common windows in two datasets for apples-to-apples comparison."""
    key_cols = ["policy", "cell_code", "window_start_cycle", "window_end_cycle"]
    common_keys = (
        left_df[key_cols]
        .merge(right_df[key_cols], on=key_cols, how="inner")
        .drop_duplicates()
        .copy()
    )
    out_left = left_df.merge(common_keys, on=key_cols, how="inner").copy()
    out_right = right_df.merge(common_keys, on=key_cols, how="inner").copy()
    out_left = out_left.sort_values(["policy", "cell_code", "window_start_cycle"]).reset_index(drop=True)
    out_right = out_right.sort_values(["policy", "cell_code", "window_start_cycle"]).reset_index(drop=True)
    return out_left, out_right


def run_single_mode_analysis(
    args: argparse.Namespace,
    data_df: pd.DataFrame,
    treatment_mode_label: str,
    extra_metrics: dict[str, float] | None = None,
    seed_offset: int = 0,
    figure_suffix: str = "",
) -> dict[str, Any]:
    """Run full causal estimation for one prepared dataset and one treatment definition."""
    treatment = data_df["treatment_value"].to_numpy(dtype=float)
    outcome = data_df["y_rel_drop"].to_numpy(dtype=float)
    covariates, _ = build_covariate_matrix(data_df)

    seed = args.seed + seed_offset
    nuisance = fit_nuisance_models(
        treatment=treatment,
        covariates=covariates,
        outcome=outcome,
        seed=seed,
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

    effect_fixed_df, dose_fixed_df, delta_fixed_df = summarize_effects_by_scheme(
        df=data_df,
        dr_matrix=dr_matrix,
        eval_grid=eval_grid,
        base_grid=base_grid,
        treatment=treatment,
        bandwidth=bandwidth,
        n_bootstrap=args.n_bootstrap,
        seed=seed,
        scheme="fixed4",
    )
    effect_quantile_df, dose_quantile_df, delta_quantile_df = summarize_effects_by_scheme(
        df=data_df,
        dr_matrix=dr_matrix,
        eval_grid=eval_grid,
        base_grid=base_grid,
        treatment=treatment,
        bandwidth=bandwidth,
        n_bootstrap=args.n_bootstrap,
        seed=seed + 1000,
        scheme="quantile4",
    )

    report_args = argparse.Namespace(**vars(args))
    report_args.treatment_mode = treatment_mode_label
    overlap_df = build_overlap_diagnostics(
        df=data_df,
        treatment=treatment,
        base_grid=base_grid,
        eval_grid=eval_grid,
        bandwidth=bandwidth,
        nuisance=nuisance,
        args=report_args,
        extra_metrics=extra_metrics,
    )

    stage_sample_compare_df = build_stage_sample_distribution(data_df)
    stage_effect_compare_df = pd.concat(
        [
            effect_fixed_df.loc[effect_fixed_df["group"] != "global"].copy(),
            effect_quantile_df.loc[effect_quantile_df["group"] != "global"].copy(),
        ],
        ignore_index=True,
    )
    global_effect_df = effect_fixed_df.loc[effect_fixed_df["group"] == "global"].copy()
    fixed_stage_df = effect_fixed_df.loc[effect_fixed_df["group"] != "global"].copy()
    quantile_stage_df = effect_quantile_df.loc[effect_quantile_df["group"] != "global"].copy()
    fixed_stage_df["scheme_display"] = STAGE_SCHEME_CONFIGS["fixed4"]["scheme_display"]
    quantile_stage_df["scheme_display"] = STAGE_SCHEME_CONFIGS["quantile4"]["scheme_display"]
    stage_effect_compare_df["scheme_display"] = stage_effect_compare_df["scheme"].map(
        {
            "fixed4": STAGE_SCHEME_CONFIGS["fixed4"]["scheme_display"],
            "quantile4": STAGE_SCHEME_CONFIGS["quantile4"]["scheme_display"],
        }
    )

    version_df = collect_library_versions()
    parameter_sources_df = _build_parameter_sources(args=report_args)
    report_text = render_markdown_report(
        args=report_args,
        global_effect_df=global_effect_df,
        overlap_df=overlap_df,
        effect_fixed_df=effect_fixed_df,
        effect_quantile_df=effect_quantile_df,
        dose_curve_fixed_df=dose_fixed_df,
        dose_curve_quantile_df=dose_quantile_df,
        delta_curve_fixed_df=delta_fixed_df,
        delta_curve_quantile_df=delta_quantile_df,
        data_df=data_df,
        weights_diag_df=weights_diag_df,
        effect_stage_compare_df=stage_effect_compare_df,
        stage_sample_compare_df=stage_sample_compare_df,
        parameter_sources_df=parameter_sources_df,
        figure_suffix=figure_suffix,
    )

    return {
        "data_df": data_df,
        "global_effect_df": global_effect_df,
        "stage_effect_compare_df": stage_effect_compare_df,
        "fixed_stage_df": fixed_stage_df,
        "quantile_stage_df": quantile_stage_df,
        "stage_sample_compare_df": stage_sample_compare_df,
        "dose_fixed_df": dose_fixed_df,
        "delta_fixed_df": delta_fixed_df,
        "dose_quantile_df": dose_quantile_df,
        "delta_quantile_df": delta_quantile_df,
        "overlap_df": overlap_df,
        "weights_diag_df": weights_diag_df,
        "version_df": version_df,
        "parameter_sources_df": parameter_sources_df,
        "report_text": report_text,
        "effect_fixed_df": effect_fixed_df,
        "effect_quantile_df": effect_quantile_df,
    }


def _with_suffix(stem: str, suffix: str) -> str:
    """Append suffix to file stem when suffix is non-empty."""
    return f"{stem}_{suffix}" if suffix else stem


def save_mode_outputs(
    output_dir: Path,
    bundle: dict[str, Any],
    encoding: str,
    suffix: str,
    write_legacy: bool = False,
) -> None:
    """Save CSVs, plots, and markdown report for one treatment mode."""
    data_df = bundle["data_df"]
    global_effect_df = bundle["global_effect_df"]
    stage_effect_compare_df = bundle["stage_effect_compare_df"]
    fixed_stage_df = bundle["fixed_stage_df"]
    quantile_stage_df = bundle["quantile_stage_df"]
    stage_sample_compare_df = bundle["stage_sample_compare_df"]
    dose_fixed_df = bundle["dose_fixed_df"]
    delta_fixed_df = bundle["delta_fixed_df"]
    dose_quantile_df = bundle["dose_quantile_df"]
    delta_quantile_df = bundle["delta_quantile_df"]
    overlap_df = bundle["overlap_df"]
    weights_diag_df = bundle["weights_diag_df"]
    version_df = bundle["version_df"]
    parameter_sources_df = bundle["parameter_sources_df"]
    report_text = bundle["report_text"]
    effect_fixed_df = bundle["effect_fixed_df"]
    effect_quantile_df = bundle["effect_quantile_df"]

    data_df.to_csv(
        output_dir / f"{_with_suffix('analysis_dataset_windows', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    global_effect_df.to_csv(
        output_dir / f"{_with_suffix('causal_effect_global', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    stage_effect_compare_df.to_csv(
        output_dir / f"{_with_suffix('causal_effect_by_stage', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    fixed_stage_df.to_csv(
        output_dir / f"{_with_suffix('causal_effect_by_stage_fixed4', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    quantile_stage_df.to_csv(
        output_dir / f"{_with_suffix('causal_effect_by_stage_quantile4', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    stage_effect_compare_df.to_csv(
        output_dir / f"{_with_suffix('causal_effect_by_stage_compare', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    stage_sample_compare_df.to_csv(
        output_dir / f"{_with_suffix('stage_sample_distribution_compare', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    dose_fixed_df.to_csv(
        output_dir / f"{_with_suffix('dose_response_curve', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    delta_fixed_df.to_csv(
        output_dir / f"{_with_suffix('delta_plus_1c_curve', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    dose_quantile_df.to_csv(
        output_dir / f"{_with_suffix('dose_response_curve_quantile4', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    delta_quantile_df.to_csv(
        output_dir / f"{_with_suffix('delta_plus_1c_curve_quantile4', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    overlap_df.to_csv(
        output_dir / f"{_with_suffix('diagnostics_overlap', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    weights_diag_df.to_csv(
        output_dir / f"{_with_suffix('diagnostics_weights', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    version_df.to_csv(
        output_dir / f"{_with_suffix('runtime_library_versions', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    parameter_sources_df.to_csv(
        output_dir / f"{_with_suffix('report_parameter_sources', suffix)}.csv",
        index=False,
        encoding=encoding,
    )
    report_name = (
        "causal_initial_rate_report.md"
        if not suffix
        else f"causal_initial_rate_report_{suffix}.md"
    )
    (output_dir / report_name).write_text(report_text, encoding="utf-8")

    save_dose_response_plot(
        dose_curve_df=dose_fixed_df,
        output_path=output_dir / f"{_with_suffix('fig_dose_response', suffix)}.png",
        scheme="fixed4",
    )
    save_dose_response_plot(
        dose_curve_df=dose_quantile_df,
        output_path=output_dir / f"{_with_suffix('fig_dose_response_quantile4', suffix)}.png",
        scheme="quantile4",
    )
    save_stage_effect_plot(
        effect_df=effect_fixed_df,
        output_path=output_dir / f"{_with_suffix('fig_plus1c_by_stage', suffix)}.png",
        scheme="fixed4",
    )
    save_stage_effect_plot(
        effect_df=effect_quantile_df,
        output_path=output_dir / f"{_with_suffix('fig_plus1c_by_stage_quantile4', suffix)}.png",
        scheme="quantile4",
    )
    save_delta_plus1c_curve_plot(
        delta_curve_df=delta_fixed_df,
        output_path=output_dir / f"{_with_suffix('fig_delta_plus1c_curve', suffix)}.png",
        scheme="fixed4",
    )
    save_delta_plus1c_curve_plot(
        delta_curve_df=delta_quantile_df,
        output_path=output_dir / f"{_with_suffix('fig_delta_plus1c_curve_quantile4', suffix)}.png",
        scheme="quantile4",
    )
    save_treatment_distribution_plot(
        df=data_df,
        output_path=output_dir / f"{_with_suffix('fig_treatment_distribution', suffix)}.png",
    )
    save_outcome_by_stage_plot(
        df=data_df,
        output_path=output_dir / f"{_with_suffix('fig_outcome_by_stage', suffix)}.png",
        scheme="fixed4",
    )
    save_outcome_by_stage_plot(
        df=data_df,
        output_path=output_dir / f"{_with_suffix('fig_outcome_by_stage_quantile4', suffix)}.png",
        scheme="quantile4",
    )
    save_weight_diagnostics_plot(
        weights_diag_df=weights_diag_df,
        output_path=output_dir / f"{_with_suffix('fig_weight_diagnostics', suffix)}.png",
    )
    save_stage_effect_compare_plot(
        effect_stage_df=stage_effect_compare_df,
        output_path=output_dir / f"{_with_suffix('fig_plus1c_by_stage_compare', suffix)}.png",
    )

    if write_legacy and suffix:
        # For explicit compare mode, avoid touching legacy unsuffixed files.
        pass


def save_treatment_mode_compare_plot(compare_df: pd.DataFrame, output_path: Path) -> None:
    """Save grouped comparison plot across treatment definitions."""
    mode_order = ["initial", "window_mean"]
    mode_label = {
        mode: _treatment_mode_label(mode, use_short=False)
        for mode in mode_order
    }
    fixed_order = list(STAGE_SCHEME_CONFIGS["fixed4"]["stage_order"])
    quant_order = list(STAGE_SCHEME_CONFIGS["quantile4"]["stage_order"])
    fixed_label_map = get_stage_label_map("fixed4")
    quant_label_map = get_stage_label_map("quantile4")

    ordered_categories: list[tuple[str, str, str]] = [("global", "global", "全局")]
    ordered_categories.extend(
        [("fixed4", g, f"固定|{fixed_label_map.get(g, g)}") for g in fixed_order]
    )
    ordered_categories.extend(
        [("quantile4", g, f"分位|{quant_label_map.get(g, g)}") for g in quant_order]
    )
    category_keys = [f"{s}:{g}" for s, g, _ in ordered_categories]
    category_labels = [lbl for _, _, lbl in ordered_categories]

    plot_df = compare_df.copy()
    plot_df["category_key"] = plot_df["scheme"] + ":" + plot_df["group"]

    x = np.arange(len(category_keys))
    width = 0.36
    plt.figure(figsize=(13.0, 5.6))
    for i, mode in enumerate(mode_order):
        part = plot_df.loc[plot_df["treatment_mode"] == mode].copy()
        part = part.set_index("category_key").reindex(category_keys).reset_index()
        y = part["effect_plus_1c"].to_numpy(dtype=float)
        ci_low = part["ci_low"].to_numpy(dtype=float)
        ci_high = part["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.where(np.isfinite(ci_low), y - ci_low, np.nan),
                np.where(np.isfinite(ci_high), ci_high - y, np.nan),
            ]
        )
        pos = x + (i - 0.5) * width
        plt.bar(pos, y, width=width, alpha=0.86, label=mode_label.get(mode, mode))
        plt.errorbar(pos, y, yerr=yerr, fmt="none", ecolor="black", capsize=2.5, linewidth=1.0)

    plt.xticks(x, category_labels, rotation=25, ha="right")
    plt.ylabel("+1C 对未来相对容量下降的估计影响")
    plt.title(
        f"{_treatment_mode_label('initial')} vs {_treatment_mode_label('window_mean')}：阶段效应对比"
    )
    plt.grid(axis="y", alpha=0.24)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def render_treatment_mode_compare_report(
    global_compare_df: pd.DataFrame,
    stage_compare_df: pd.DataFrame,
    sample_compare_df: pd.DataFrame,
) -> str:
    """Render markdown report for treatment-mode comparison."""
    mode_label = {
        mode: _treatment_mode_label(mode, use_short=False)
        for mode in ["initial", "window_mean"]
    }

    global_wide = global_compare_df.set_index("treatment_mode")
    init_g = global_wide.loc["initial"] if "initial" in global_wide.index else None
    wm_g = global_wide.loc["window_mean"] if "window_mean" in global_wide.index else None
    delta_global = (
        float(wm_g["effect_plus_1c"]) - float(init_g["effect_plus_1c"])
        if init_g is not None and wm_g is not None
        else float("nan")
    )
    rel_global = (
        delta_global / abs(float(init_g["effect_plus_1c"]))
        if init_g is not None and np.isfinite(float(init_g["effect_plus_1c"])) and float(init_g["effect_plus_1c"]) != 0
        else float("nan")
    )

    lines: list[str] = []
    lines.append(
        f"# treatment 口径对比报告：{_treatment_mode_label('initial')} vs {_treatment_mode_label('window_mean')}"
    )
    lines.append("")
    lines.append("## 1. 全局 +1C 效应对比")
    lines.append("| treatment_mode | effect_plus_1c | ci_low | ci_high | n_rows | n_clusters |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in global_compare_df.iterrows():
        lines.append(
            f"| {mode_label.get(str(row['treatment_mode']), str(row['treatment_mode']))} | "
            f"{_format_float(float(row['effect_plus_1c']))} | {_format_float(float(row['ci_low']))} | "
            f"{_format_float(float(row['ci_high']))} | {int(row['n_rows'])} | {int(row['n_clusters'])} |"
        )
    lines.append(
        f"- 关键差异：{_treatment_mode_label('window_mean')} - {_treatment_mode_label('initial')} 的全局效应差为 `{_format_float(delta_global, 6)}`，相对变化约 `{_format_float(rel_global * 100.0, 2)}%`。"
    )
    lines.append("")
    lines.append("## 2. 分阶段效应对比（固定4段 + 分位4段）")
    lines.append("| treatment_mode | scheme | group | effect_plus_1c | ci_low | ci_high |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, row in stage_compare_df.iterrows():
        lines.append(
            f"| {mode_label.get(str(row['treatment_mode']), str(row['treatment_mode']))} | "
            f"{_md_escape(str(row['scheme']))} | {_md_escape(str(row['group_label']))} | "
            f"{_format_float(float(row['effect_plus_1c']))} | {_format_float(float(row['ci_low']))} | "
            f"{_format_float(float(row['ci_high']))} |"
        )
    lines.append("")
    lines.append("## 3. 可比性诊断（交集样本口径）")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for _, row in sample_compare_df.iterrows():
        metric = _md_escape(str(row["metric"]))
        value = row["value"]
        if isinstance(value, (int, np.integer)):
            text = str(int(value))
        else:
            try:
                text = _format_float(float(value), 6)
            except Exception:
                text = _md_escape(str(value))
        lines.append(f"| {metric} | {text} |")
    lines.append("")
    lines.append("## 4. 图表解读")
    lines.append("![treatment口径对比图](./fig_plus1c_treatment_mode_compare.png)")
    lines.append("- X轴含义：全局与各寿命阶段（固定4段、分位4段）的分组。")
    lines.append("- Y轴含义：`+1C` 对未来相对容量下降的估计影响（含95%CI）。")
    lines.append("- 关键性结论：该图直接比较两种 treatment 定义下的效应幅值与不确定性差异。")
    lines.append(
        f"- 业务解释：{_treatment_mode_label_with_code('initial')} 更接近策略设定，{_treatment_mode_label_with_code('window_mean')} 更接近实际执行；两者差异可视为口径敏感性。"
    )
    lines.append("")
    lines.append("## 5. 说明")
    lines.append("- 本对比严格使用两口径交集样本，避免样本构成差异主导结果。")
    lines.append("- window_mean 定义：按 cycle 聚合充电电流均值后，以每电芯早期 q_ref 归一化。")
    lines.append("- window_mean 建模前按 q01~q99 裁剪，仅影响 treatment 输入，不改变原始诊断统计。")
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
    if args.compare_initial_vs_window_mean:
        initial_all_df, initial_diag = build_analysis_dataset_with_diagnostics(
            life_path=args.life_performance_path,
            policy_path=args.policy_meaning_path,
            horizon_cycles=args.horizon_cycles,
            treatment_mode="initial",
            exclude_policy_prefix=args.exclude_policy_prefix,
            encoding=args.encoding,
            charge_interval_features_path=args.charge_interval_features_path,
            window_mean_qref_cycles=args.window_mean_qref_cycles,
            window_mean_clip_quantile_low=args.window_mean_clip_quantile_low,
            window_mean_clip_quantile_high=args.window_mean_clip_quantile_high,
        )
        window_all_df, window_diag = build_analysis_dataset_with_diagnostics(
            life_path=args.life_performance_path,
            policy_path=args.policy_meaning_path,
            horizon_cycles=args.horizon_cycles,
            treatment_mode="window_mean",
            exclude_policy_prefix=args.exclude_policy_prefix,
            encoding=args.encoding,
            charge_interval_features_path=args.charge_interval_features_path,
            window_mean_qref_cycles=args.window_mean_qref_cycles,
            window_mean_clip_quantile_low=args.window_mean_clip_quantile_low,
            window_mean_clip_quantile_high=args.window_mean_clip_quantile_high,
        )
        initial_df, window_df = intersect_window_datasets(initial_all_df, window_all_df)
        if initial_df.empty or window_df.empty:
            raise ValueError("No intersection samples found between initial and window_mean datasets.")

        common_metrics = {
            "comparison_intersection_rows": float(initial_df.shape[0]),
            "comparison_intersection_clusters": float(initial_df["cluster_id"].nunique()),
            "comparison_initial_rows_before_intersection": float(initial_all_df.shape[0]),
            "comparison_window_mean_rows_before_intersection": float(window_all_df.shape[0]),
        }
        initial_metrics = dict(initial_diag)
        initial_metrics.update(common_metrics)
        window_metrics = dict(window_diag)
        window_metrics.update(common_metrics)

        initial_bundle = run_single_mode_analysis(
            args=args,
            data_df=initial_df,
            treatment_mode_label="initial",
            extra_metrics=initial_metrics,
            seed_offset=0,
            figure_suffix="initial",
        )
        window_bundle = run_single_mode_analysis(
            args=args,
            data_df=window_df,
            treatment_mode_label="window_mean",
            extra_metrics=window_metrics,
            seed_offset=10000,
            figure_suffix="window_mean",
        )
        save_mode_outputs(
            output_dir=output_dir,
            bundle=initial_bundle,
            encoding=args.encoding,
            suffix="initial",
        )
        save_mode_outputs(
            output_dir=output_dir,
            bundle=window_bundle,
            encoding=args.encoding,
            suffix="window_mean",
        )

        initial_global = initial_bundle["global_effect_df"].copy()
        initial_global["treatment_mode"] = "initial"
        window_global = window_bundle["global_effect_df"].copy()
        window_global["treatment_mode"] = "window_mean"
        global_compare_df = pd.concat([initial_global, window_global], ignore_index=True)

        initial_stage = initial_bundle["stage_effect_compare_df"].copy()
        initial_stage["treatment_mode"] = "initial"
        window_stage = window_bundle["stage_effect_compare_df"].copy()
        window_stage["treatment_mode"] = "window_mean"
        stage_compare_df = pd.concat([initial_stage, window_stage], ignore_index=True)

        sample_rows = [
            {"metric": "initial_n_rows_before_intersection", "value": float(initial_all_df.shape[0])},
            {"metric": "window_mean_n_rows_before_intersection", "value": float(window_all_df.shape[0])},
            {"metric": "intersection_n_rows", "value": float(initial_df.shape[0])},
            {"metric": "initial_n_clusters_before_intersection", "value": float(initial_all_df["cluster_id"].nunique())},
            {"metric": "window_mean_n_clusters_before_intersection", "value": float(window_all_df["cluster_id"].nunique())},
            {"metric": "intersection_n_clusters", "value": float(initial_df["cluster_id"].nunique())},
            {"metric": "q_ref_coverage", "value": float(window_diag.get("q_ref_coverage", np.nan))},
            {"metric": "window_mean_coverage", "value": float(window_diag.get("window_mean_coverage", np.nan))},
            {"metric": "window_mean_clip_low", "value": float(window_diag.get("window_mean_clip_low", np.nan))},
            {"metric": "window_mean_clip_high", "value": float(window_diag.get("window_mean_clip_high", np.nan))},
            {"metric": "window_mean_clip_low_share", "value": float(window_diag.get("window_mean_clip_low_share", np.nan))},
            {"metric": "window_mean_clip_high_share", "value": float(window_diag.get("window_mean_clip_high_share", np.nan))},
        ]
        sample_compare_df = pd.DataFrame(sample_rows)

        plot_global = global_compare_df.copy()
        plot_global["scheme"] = "global"
        plot_global["group"] = "global"
        plot_global["group_label"] = "全局"
        compare_plot_df = pd.concat(
            [plot_global, stage_compare_df],
            ignore_index=True,
            sort=False,
        )
        save_treatment_mode_compare_plot(
            compare_df=compare_plot_df,
            output_path=output_dir / "fig_plus1c_treatment_mode_compare.png",
        )
        compare_report_text = render_treatment_mode_compare_report(
            global_compare_df=global_compare_df,
            stage_compare_df=stage_compare_df,
            sample_compare_df=sample_compare_df,
        )

        global_compare_df.to_csv(
            output_dir / "causal_effect_global_treatment_compare.csv",
            index=False,
            encoding=args.encoding,
        )
        stage_compare_df.to_csv(
            output_dir / "causal_effect_by_stage_treatment_compare.csv",
            index=False,
            encoding=args.encoding,
        )
        sample_compare_df.to_csv(
            output_dir / "treatment_mode_sample_compare.csv",
            index=False,
            encoding=args.encoding,
        )
        (output_dir / "treatment_mode_compare_report.md").write_text(
            compare_report_text,
            encoding="utf-8",
        )
    else:
        data_df, extra_metrics = build_analysis_dataset_with_diagnostics(
            life_path=args.life_performance_path,
            policy_path=args.policy_meaning_path,
            horizon_cycles=args.horizon_cycles,
            treatment_mode=args.treatment_mode,
            exclude_policy_prefix=args.exclude_policy_prefix,
            encoding=args.encoding,
            charge_interval_features_path=args.charge_interval_features_path,
            window_mean_qref_cycles=args.window_mean_qref_cycles,
            window_mean_clip_quantile_low=args.window_mean_clip_quantile_low,
            window_mean_clip_quantile_high=args.window_mean_clip_quantile_high,
        )
        bundle = run_single_mode_analysis(
            args=args,
            data_df=data_df,
            treatment_mode_label=args.treatment_mode,
            extra_metrics=extra_metrics,
            seed_offset=0,
            figure_suffix="",
        )
        save_mode_outputs(
            output_dir=output_dir,
            bundle=bundle,
            encoding=args.encoding,
            suffix="",
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
