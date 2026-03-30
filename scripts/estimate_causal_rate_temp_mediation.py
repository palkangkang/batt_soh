from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_POLICY_MEANING_PATH = REPO_ROOT / "data" / "processed" / "policy_meaning.csv"
DEFAULT_CHARGE_INTERVAL_FEATURES_PATH = (
    REPO_ROOT / "data" / "processed" / "charge_interval_features.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "causal_rate_temp_mediation"

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


MODE_ORDER = ["initial", "effective_mean", "window_mean"]
SCHEME_ORDER = ["fixed4", "quantile4"]
BASELINE_SCENARIO = "baseline_tplus1_temp0_70"
SENSI_SCENARIOS = ["sensitivity_t_as_mediator", "sensitivity_temp_5_60"]

STAGE_SCHEME_CONFIGS: dict[str, dict[str, object]] = {
    "fixed4": {
        "scheme_display": "固定分段4段",
        "stage_col": "life_stage_fixed4",
        "stage_order": ["cycle_0_500", "cycle_500_1000", "cycle_1000_1500", "cycle_1500_max"],
        "stage_label_map": {
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
        "stage_label_map": {"q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4"},
    },
}

MODE_DISPLAY = {
    "initial": "初始充电倍率",
    "effective_mean": "策略平均充电倍率",
    "window_mean": "窗口真实平均充电倍率",
}
PRIMARY_TREATMENT_MODE = "window_mean"
RATE_BIN_FIXED_A_EDGES = [1.8, 2.6, 3.5]
RATE_BIN_FIXED_A_ORDER = ["rate_bin_a_q1", "rate_bin_a_q2", "rate_bin_a_q3", "rate_bin_a_q4"]
RATE_BIN_FIXED_A_LABEL_MAP = {
    "rate_bin_a_q1": "<1.8C",
    "rate_bin_a_q2": "1.8-2.6C",
    "rate_bin_a_q3": "2.6-3.5C",
    "rate_bin_a_q4": ">=3.5C",
}


@dataclass
class FittedModels:
    """Container for fitted mediator/outcome models."""

    mediator_model: LinearRegression
    outcome_model: LinearRegression


def setup_plot_fonts() -> bool:
    """Configure plotting font for Chinese rendering."""
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


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert specific columns to numeric with coercion."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_treatment_column(policy_df: pd.DataFrame, treatment_mode: str) -> pd.Series:
    """Build policy-level treatment according to selected mode."""
    if treatment_mode == "initial":
        return policy_df["initial_c_rate"]
    if treatment_mode == "effective_mean":
        switch_ratio = policy_df["switch_soc_percent"] / 100.0
        return (
            policy_df["initial_c_rate"] * switch_ratio
            + policy_df["post_switch_c_rate"] * (1.0 - switch_ratio)
        )
    raise ValueError(f"Unsupported treatment_mode: {treatment_mode}")


def _validate_quantiles(low: float, high: float, name: str) -> None:
    """Validate pair of lower and upper quantiles."""
    if not (0.0 <= low < 1.0):
        raise ValueError(f"{name}_low must be in [0, 1).")
    if not (0.0 < high <= 1.0):
        raise ValueError(f"{name}_high must be in (0, 1].")
    if low >= high:
        raise ValueError(f"{name}_low must be < {name}_high.")


def _parse_float_list(text: str) -> list[float]:
    """Parse comma-separated float values."""
    out: list[float] = []
    for item in text.split(","):
        raw = item.strip()
        if not raw:
            continue
        out.append(float(raw))
    if not out:
        raise ValueError("At least one float value is required.")
    return out


def get_stage_label_map(scheme: str) -> dict[str, str]:
    """Return stage display map for one scheme."""
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    return cfg["stage_label_map"]  # type: ignore[return-value]


def add_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed4 and quantile4 stage columns."""
    out = df.copy()
    fixed_cfg = STAGE_SCHEME_CONFIGS["fixed4"]
    fixed_labels = fixed_cfg["stage_order"]  # type: ignore[assignment]
    out["life_stage_fixed4"] = pd.cut(
        out["cycle_t"],
        bins=[0, 500, 1000, 1500, float("inf")],
        labels=fixed_labels,
        right=False,
        include_lowest=True,
    ).astype(str)

    quant_cfg = STAGE_SCHEME_CONFIGS["quantile4"]
    quant_labels = quant_cfg["stage_order"]  # type: ignore[assignment]
    rank_series = out["cycle_t"].rank(method="first")
    out["life_stage_quantile4"] = pd.qcut(rank_series, q=4, labels=quant_labels).astype(str)
    return out


def assign_rate_bin_fixed_a(treatment: pd.Series) -> pd.Series:
    """Assign fixed-rate bins for scheme A using treatment_value."""
    bins = [-float("inf"), *RATE_BIN_FIXED_A_EDGES, float("inf")]
    out = pd.cut(
        pd.to_numeric(treatment, errors="coerce"),
        bins=bins,
        labels=RATE_BIN_FIXED_A_ORDER,
        right=False,
        include_lowest=True,
    )
    return out.astype(str)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rate-temperature-capacity mediation decomposition using g-computation."
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
        help="Path to charge_interval_features.csv (for window_mean mode).",
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
        help="Outcome horizon H in Y_t+H.",
    )
    parser.add_argument(
        "--exclude-policy-prefix",
        type=str,
        default="VARCHARGE",
        help="Exclude policies with this prefix.",
    )
    parser.add_argument(
        "--treatment-modes",
        type=str,
        default="initial,effective_mean,window_mean",
        help="Comma-separated treatment modes.",
    )
    parser.add_argument(
        "--window-mean-qref-cycles",
        type=int,
        default=20,
        help="Number of early cycles for q_ref median in window_mean.",
    )
    parser.add_argument(
        "--window-mean-clip-quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile for window_mean clipping.",
    )
    parser.add_argument(
        "--window-mean-clip-quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile for window_mean clipping.",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=0.0,
        help="Physical minimum temperature filter for baseline.",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=70.0,
        help="Physical maximum temperature filter for baseline.",
    )
    parser.add_argument(
        "--temperature-clip-quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile for mediator clipping (modeling only).",
    )
    parser.add_argument(
        "--temperature-clip-quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile for mediator clipping (modeling only).",
    )
    parser.add_argument(
        "--sensitivity-temperature-min",
        type=float,
        default=5.0,
        help="Alternative minimum temperature for sensitivity scenario.",
    )
    parser.add_argument(
        "--sensitivity-temperature-max",
        type=float,
        default=60.0,
        help="Alternative maximum temperature for sensitivity scenario.",
    )
    parser.add_argument(
        "--cde-deltas",
        type=str,
        default="1,5",
        help="Comma-separated temperature increments for CDE.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=400,
        help="Cluster bootstrap repetitions (baseline only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260330,
        help="Random seed.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding.",
    )
    return parser.parse_args()
def _build_window_mean_treatment_cycle(
    life_df: pd.DataFrame,
    charge_interval_path: Path,
    qref_cycles: int,
    clip_low: float,
    clip_high: float,
    encoding: str,
    exclude_policy_prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build cycle-level treatment for window_mean mode."""
    _validate_quantiles(clip_low, clip_high, "window_mean_clip_quantile")
    if qref_cycles <= 0:
        raise ValueError("window_mean_qref_cycles must be positive.")

    charge_df = pd.read_csv(
        charge_interval_path,
        encoding=encoding,
        usecols=["policy", "cell_code", "cycles", "delta_ah", "charge_duration_s"],
    )
    charge_df = _to_numeric(charge_df, ["cycles", "delta_ah", "charge_duration_s"])
    charge_df = charge_df.dropna(subset=["policy", "cell_code", "cycles", "delta_ah", "charge_duration_s"]).copy()
    charge_df["cycles"] = charge_df["cycles"].astype(int)
    if exclude_policy_prefix:
        charge_df = charge_df.loc[
            ~charge_df["policy"].astype(str).str.startswith(exclude_policy_prefix)
        ].copy()

    charge_agg = (
        charge_df.groupby(["policy", "cell_code", "cycles"], as_index=False)
        .agg(delta_ah_sum=("delta_ah", "sum"), charge_duration_s_sum=("charge_duration_s", "sum"))
        .copy()
    )
    charge_agg = charge_agg.loc[
        (charge_agg["delta_ah_sum"] > 0) & (charge_agg["charge_duration_s_sum"] > 0)
    ].copy()
    charge_agg["avg_current_a"] = charge_agg["delta_ah_sum"] / (charge_agg["charge_duration_s_sum"] / 3600.0)

    q_ref_df = (
        life_df.sort_values(["policy", "cell_code", "cycles"])
        .groupby(["policy", "cell_code"], as_index=False)
        .head(qref_cycles)
        .groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .median()
        .rename(columns={"q_discharge": "q_ref"})
    )
    q_ref_df = q_ref_df.loc[q_ref_df["q_ref"] > 0].copy()

    merged = charge_agg.merge(q_ref_df, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    merged["treatment_value_raw"] = merged["avg_current_a"] / merged["q_ref"]
    merged = merged.loc[np.isfinite(merged["treatment_value_raw"])].copy()
    merged = merged.loc[merged["treatment_value_raw"] > 0].copy()
    if merged.empty:
        raise ValueError("No valid treatment rows for window_mean.")

    low_value = float(np.quantile(merged["treatment_value_raw"], clip_low))
    high_value = float(np.quantile(merged["treatment_value_raw"], clip_high))
    merged["treatment_value"] = merged["treatment_value_raw"].clip(lower=low_value, upper=high_value)

    diagnostics = {
        "window_mean_raw_min": float(merged["treatment_value_raw"].min()),
        "window_mean_raw_max": float(merged["treatment_value_raw"].max()),
        "window_mean_raw_q01": float(np.quantile(merged["treatment_value_raw"], 0.01)),
        "window_mean_raw_q99": float(np.quantile(merged["treatment_value_raw"], 0.99)),
        "window_mean_clip_low": low_value,
        "window_mean_clip_high": high_value,
        "window_mean_clip_low_share": float((merged["treatment_value_raw"] < low_value).mean()),
        "window_mean_clip_high_share": float((merged["treatment_value_raw"] > high_value).mean()),
        "window_mean_qref_cycles": float(qref_cycles),
    }
    out_cols = [
        "policy",
        "cell_code",
        "cycles",
        "treatment_value",
        "treatment_value_raw",
        "avg_current_a",
        "q_ref",
    ]
    return merged[out_cols].copy(), diagnostics


def _build_charge_cycle_average_temperature(
    charge_interval_path: Path,
    encoding: str,
    exclude_policy_prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build cycle-level charge average temperature weighted by charge duration."""
    charge_df = pd.read_csv(
        charge_interval_path,
        encoding=encoding,
        usecols=["policy", "cell_code", "cycles", "avg_temper", "charge_duration_s"],
    )
    charge_df = _to_numeric(charge_df, ["cycles", "avg_temper", "charge_duration_s"])
    charge_df = charge_df.dropna(
        subset=["policy", "cell_code", "cycles", "avg_temper", "charge_duration_s"]
    ).copy()
    charge_df["cycles"] = charge_df["cycles"].astype(int)
    if exclude_policy_prefix:
        charge_df = charge_df.loc[
            ~charge_df["policy"].astype(str).str.startswith(exclude_policy_prefix)
        ].copy()

    segment_rows_clean = float(charge_df.shape[0])
    charge_df = charge_df.loc[charge_df["charge_duration_s"] > 0].copy()
    if charge_df.empty:
        raise ValueError("No valid rows in charge_interval_features after duration filtering.")
    segment_rows_positive_duration = float(charge_df.shape[0])

    charge_df["temp_x_time"] = charge_df["avg_temper"] * charge_df["charge_duration_s"]
    temp_cycle = (
        charge_df.groupby(["policy", "cell_code", "cycles"], as_index=False)
        .agg(
            temp_x_time_sum=("temp_x_time", "sum"),
            charge_duration_s_sum=("charge_duration_s", "sum"),
            n_segments=("avg_temper", "size"),
        )
        .copy()
    )
    temp_cycle = temp_cycle.loc[temp_cycle["charge_duration_s_sum"] > 0].copy()
    if temp_cycle.empty:
        raise ValueError("No valid cycle-level average temperature could be built.")
    temp_cycle["charge_temp_cycle_avg"] = (
        temp_cycle["temp_x_time_sum"] / temp_cycle["charge_duration_s_sum"]
    )
    temp_cycle = temp_cycle.loc[np.isfinite(temp_cycle["charge_temp_cycle_avg"])].copy()

    diagnostics = {
        "charge_temp_segment_rows_clean": segment_rows_clean,
        "charge_temp_segment_rows_positive_duration": segment_rows_positive_duration,
        "charge_temp_segment_positive_duration_share": (
            segment_rows_positive_duration / segment_rows_clean if segment_rows_clean > 0 else float("nan")
        ),
        "charge_temp_cycle_rows": float(temp_cycle.shape[0]),
        "charge_temp_cycle_min": float(temp_cycle["charge_temp_cycle_avg"].min()),
        "charge_temp_cycle_max": float(temp_cycle["charge_temp_cycle_avg"].max()),
        "charge_temp_cycle_q01": float(np.quantile(temp_cycle["charge_temp_cycle_avg"], 0.01)),
        "charge_temp_cycle_q99": float(np.quantile(temp_cycle["charge_temp_cycle_avg"], 0.99)),
        "charge_temp_duration_sum_min": float(temp_cycle["charge_duration_s_sum"].min()),
        "charge_temp_duration_sum_q50": float(np.quantile(temp_cycle["charge_duration_s_sum"], 0.50)),
        "charge_temp_duration_sum_q99": float(np.quantile(temp_cycle["charge_duration_s_sum"], 0.99)),
    }
    return temp_cycle[["policy", "cell_code", "cycles", "charge_temp_cycle_avg"]].copy(), diagnostics


def build_mediation_dataset(
    life_path: Path,
    policy_path: Path,
    charge_interval_features_path: Path,
    treatment_mode: str,
    horizon_cycles: int,
    mediator_lag: int,
    temperature_min: float,
    temperature_max: float,
    temperature_clip_low: float,
    temperature_clip_high: float,
    window_mean_qref_cycles: int,
    window_mean_clip_low: float,
    window_mean_clip_high: float,
    exclude_policy_prefix: str,
    encoding: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build cycle-level dataset for mediation decomposition."""
    if horizon_cycles <= 0:
        raise ValueError("horizon_cycles must be positive.")
    if mediator_lag < 0:
        raise ValueError("mediator_lag must be >= 0.")
    _validate_quantiles(temperature_clip_low, temperature_clip_high, "temperature_clip_quantile")
    if temperature_min >= temperature_max:
        raise ValueError("temperature_min must be < temperature_max.")

    life_df = pd.read_csv(
        life_path,
        encoding=encoding,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    policy_df = pd.read_csv(
        policy_path,
        encoding=encoding,
        usecols=["policy", "initial_c_rate", "switch_soc_percent", "post_switch_c_rate"],
    )
    life_df = _to_numeric(life_df, ["cycles", "q_discharge"])
    policy_df = _to_numeric(policy_df, ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"])

    life_df = life_df.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life_df = life_df.loc[life_df["q_discharge"] > 0].copy()
    life_df["cycles"] = life_df["cycles"].astype(int)
    if exclude_policy_prefix:
        life_df = life_df.loc[~life_df["policy"].astype(str).str.startswith(exclude_policy_prefix)].copy()
        policy_df = policy_df.loc[~policy_df["policy"].astype(str).str.startswith(exclude_policy_prefix)].copy()

    policy_df = policy_df.dropna(subset=["policy"]).drop_duplicates(subset=["policy"]).copy()

    if treatment_mode in ("initial", "effective_mean"):
        policy_treat = policy_df.copy()
        policy_treat["treatment_value"] = _build_treatment_column(policy_treat, treatment_mode)
        cycle_df = life_df.merge(
            policy_treat[
                ["policy", "treatment_value", "switch_soc_percent", "post_switch_c_rate", "initial_c_rate"]
            ],
            on="policy",
            how="inner",
            validate="many_to_one",
        ).copy()
    elif treatment_mode == "window_mean":
        treat_cycle_df, wm_diag = _build_window_mean_treatment_cycle(
            life_df=life_df,
            charge_interval_path=charge_interval_features_path,
            qref_cycles=window_mean_qref_cycles,
            clip_low=window_mean_clip_low,
            clip_high=window_mean_clip_high,
            encoding=encoding,
            exclude_policy_prefix=exclude_policy_prefix,
        )
        cycle_df = life_df.merge(
            treat_cycle_df[
                ["policy", "cell_code", "cycles", "treatment_value", "treatment_value_raw", "avg_current_a", "q_ref"]
            ],
            on=["policy", "cell_code", "cycles"],
            how="inner",
            validate="one_to_one",
        ).copy()
        cycle_df = cycle_df.merge(
            policy_df[["policy", "switch_soc_percent", "post_switch_c_rate", "initial_c_rate"]],
            on="policy",
            how="inner",
            validate="many_to_one",
        ).copy()
    else:
        raise ValueError(f"Unsupported treatment_mode: {treatment_mode}")

    base_df = cycle_df.rename(
        columns={"cycles": "cycle_t", "q_discharge": "q_t"}
    )[
        [
            "policy",
            "cell_code",
            "cycle_t",
            "q_t",
            "treatment_value",
            "switch_soc_percent",
            "post_switch_c_rate",
            "initial_c_rate",
        ]
    ].copy()

    q_future = life_df[["policy", "cell_code", "cycles", "q_discharge"]].rename(
        columns={"cycles": "cycle_t", "q_discharge": "q_tph"}
    )
    q_future["cycle_t"] = q_future["cycle_t"] - horizon_cycles

    charge_temp_cycle_df, charge_temp_diag = _build_charge_cycle_average_temperature(
        charge_interval_path=charge_interval_features_path,
        encoding=encoding,
        exclude_policy_prefix=exclude_policy_prefix,
    )
    temp_t_df = charge_temp_cycle_df.rename(
        columns={"cycles": "cycle_t", "charge_temp_cycle_avg": "temp_t"}
    )
    mediator_df = charge_temp_cycle_df.rename(
        columns={"cycles": "cycle_t", "charge_temp_cycle_avg": "temp_mediator_raw"}
    )
    mediator_df["cycle_t"] = mediator_df["cycle_t"] - mediator_lag

    merged = base_df.merge(
        q_future[["policy", "cell_code", "cycle_t", "q_tph"]],
        on=["policy", "cell_code", "cycle_t"],
        how="left",
    ).merge(
        temp_t_df[["policy", "cell_code", "cycle_t", "temp_t"]],
        on=["policy", "cell_code", "cycle_t"],
        how="left",
    ).merge(
        mediator_df[["policy", "cell_code", "cycle_t", "temp_mediator_raw"]],
        on=["policy", "cell_code", "cycle_t"],
        how="left",
    )

    diag: dict[str, float] = {
        "rows_before_dropna": float(merged.shape[0]),
        "q_future_availability": float(merged["q_tph"].notna().mean()),
        "temp_t_availability": float(merged["temp_t"].notna().mean()),
        "mediator_availability": float(merged["temp_mediator_raw"].notna().mean()),
        "charge_temp_t_matchable_share": float(merged["temp_t"].notna().mean()),
        "charge_temp_tpluslag_matchable_share": float(merged["temp_mediator_raw"].notna().mean()),
        "charge_temp_joint_matchable_share": float(
            (merged["temp_t"].notna() & merged["temp_mediator_raw"].notna()).mean()
        ),
    }
    diag.update({k: float(v) for k, v in charge_temp_diag.items()})

    merged = merged.dropna(subset=["q_t", "q_tph", "temp_t", "temp_mediator_raw", "treatment_value"]).copy()
    merged = merged.loc[merged["q_t"] > 0].copy()
    merged["y_rel_drop"] = (merged["q_t"] - merged["q_tph"]) / merged["q_t"]
    merged = merged.loc[np.isfinite(merged["y_rel_drop"])].copy()
    diag["rows_after_basic_clean"] = float(merged.shape[0])
    rows_before_dropna = max(diag["rows_before_dropna"], 1.0)
    diag["charge_temp_effective_share_after_basic_clean"] = float(merged.shape[0] / rows_before_dropna)

    mask_temp = (
        merged["temp_t"].between(temperature_min, temperature_max)
        & merged["temp_mediator_raw"].between(temperature_min, temperature_max)
    )
    diag["temp_physical_keep_share"] = float(mask_temp.mean()) if merged.shape[0] > 0 else float("nan")
    merged = merged.loc[mask_temp].copy()
    if merged.empty:
        raise ValueError("No rows left after temperature physical filtering.")
    diag["charge_temp_effective_share_after_physical_filter"] = float(merged.shape[0] / rows_before_dropna)

    clip_low_v = float(np.quantile(merged["temp_mediator_raw"], temperature_clip_low))
    clip_high_v = float(np.quantile(merged["temp_mediator_raw"], temperature_clip_high))
    merged["temp_mediator_model"] = merged["temp_mediator_raw"].clip(lower=clip_low_v, upper=clip_high_v)
    merged["cluster_id"] = merged["policy"].astype(str) + "|" + merged["cell_code"].astype(str)
    merged = add_stage_columns(merged)
    merged = merged.sort_values(["policy", "cell_code", "cycle_t"]).reset_index(drop=True)

    diag.update(
        {
            "rows_final": float(merged.shape[0]),
            "n_clusters": float(merged["cluster_id"].nunique()),
            "temp_model_clip_low": clip_low_v,
            "temp_model_clip_high": clip_high_v,
            "temp_model_clip_low_share": float((merged["temp_mediator_raw"] < clip_low_v).mean()),
            "temp_model_clip_high_share": float((merged["temp_mediator_raw"] > clip_high_v).mean()),
            "temp_mediator_min": float(merged["temp_mediator_raw"].min()),
            "temp_mediator_max": float(merged["temp_mediator_raw"].max()),
            "temp_t_min": float(merged["temp_t"].min()),
            "temp_t_max": float(merged["temp_t"].max()),
            "cycle_t_min": float(merged["cycle_t"].min()),
            "cycle_t_max": float(merged["cycle_t"].max()),
        }
    )
    if treatment_mode == "window_mean":
        diag.update({k: float(v) for k, v in wm_diag.items()})
    return merged, diag
def build_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Build default time-varying confounder set C_t."""
    out = pd.DataFrame(index=df.index)
    cycle_centered = df["cycle_t"].to_numpy(dtype=float)
    cycle_centered = cycle_centered - np.mean(cycle_centered)
    out["cycle_t_c"] = cycle_centered
    out["cycle_t_c_sq"] = cycle_centered ** 2
    out["q_t"] = df["q_t"].to_numpy(dtype=float)
    out["temp_t"] = df["temp_t"].to_numpy(dtype=float)
    out["switch_soc_percent"] = df["switch_soc_percent"].to_numpy(dtype=float)
    out["post_switch_c_rate"] = df["post_switch_c_rate"].to_numpy(dtype=float)
    return out


def build_mediator_features(treatment: np.ndarray, cov_df: pd.DataFrame) -> np.ndarray:
    """Build mediator model features m(R,C)."""
    return np.column_stack([treatment, cov_df.to_numpy(dtype=float)])


def build_outcome_features(
    treatment: np.ndarray,
    mediator: np.ndarray,
    cov_df: pd.DataFrame,
) -> np.ndarray:
    """Build outcome model features g(R,T,C)."""
    return np.column_stack(
        [
            treatment,
            mediator,
            treatment * mediator,
            treatment ** 2,
            mediator ** 2,
            cov_df.to_numpy(dtype=float),
        ]
    )


def fit_models(df: pd.DataFrame) -> tuple[FittedModels, pd.DataFrame]:
    """Fit mediator and outcome models."""
    cov_df = build_covariates(df)
    treatment = df["treatment_value"].to_numpy(dtype=float)
    mediator = df["temp_mediator_model"].to_numpy(dtype=float)
    outcome = df["y_rel_drop"].to_numpy(dtype=float)

    mediator_model = LinearRegression()
    mediator_model.fit(build_mediator_features(treatment=treatment, cov_df=cov_df), mediator)

    outcome_model = LinearRegression()
    outcome_model.fit(
        build_outcome_features(treatment=treatment, mediator=mediator, cov_df=cov_df),
        outcome,
    )
    return FittedModels(mediator_model=mediator_model, outcome_model=outcome_model), cov_df


def compute_weight_diagnostics(treatment: np.ndarray, cov_df: pd.DataFrame) -> dict[str, float]:
    """Compute overlap/weight diagnostics from Gaussian treatment model."""
    tr_model = LinearRegression()
    x = cov_df.to_numpy(dtype=float)
    tr_model.fit(x, treatment)
    mu = tr_model.predict(x)
    resid = treatment - mu
    sigma = float(np.std(resid, ddof=1))
    sigma = max(sigma, 1e-3)
    density = stats.norm.pdf((treatment - mu) / sigma) / sigma
    density = np.clip(density, 1e-12, None)
    weight_raw = 1.0 / density
    clip_thr = float(np.quantile(weight_raw, 0.995))
    weight = np.minimum(weight_raw, clip_thr)
    ess = float((weight.sum() ** 2) / np.sum(weight ** 2))
    return {
        "treatment_sigma": sigma,
        "weight_mean": float(np.mean(weight)),
        "weight_std": float(np.std(weight)),
        "weight_p95": float(np.quantile(weight, 0.95)),
        "weight_p99": float(np.quantile(weight, 0.99)),
        "weight_max": float(np.max(weight)),
        "weight_clip_threshold": clip_thr,
        "effective_sample_size": ess,
    }


def predict_components(
    df: pd.DataFrame,
    models: FittedModels,
    cov_df: pd.DataFrame,
    cde_deltas: Sequence[float],
) -> pd.DataFrame:
    """Predict row-level components for TE/NDE/NIE/CDE decomposition."""
    treatment = df["treatment_value"].to_numpy(dtype=float)
    mediator_obs = df["temp_mediator_model"].to_numpy(dtype=float)
    n = treatment.shape[0]

    mediator_r = models.mediator_model.predict(build_mediator_features(treatment=treatment, cov_df=cov_df))
    mediator_r1 = models.mediator_model.predict(
        build_mediator_features(treatment=treatment + 1.0, cov_df=cov_df)
    )

    y_r = models.outcome_model.predict(
        build_outcome_features(treatment=treatment, mediator=mediator_r, cov_df=cov_df)
    )
    y_r1 = models.outcome_model.predict(
        build_outcome_features(treatment=treatment + 1.0, mediator=mediator_r1, cov_df=cov_df)
    )
    y_r1_t_r = models.outcome_model.predict(
        build_outcome_features(treatment=treatment + 1.0, mediator=mediator_r, cov_df=cov_df)
    )

    r_star = float(np.mean(treatment))
    r_star_vec = np.full(n, r_star, dtype=float)
    y_cde_base = models.outcome_model.predict(
        build_outcome_features(treatment=r_star_vec, mediator=mediator_obs, cov_df=cov_df)
    )

    comp_df = df[
        ["cluster_id", "life_stage_fixed4", "life_stage_quantile4", "cycle_t", "treatment_value"]
    ].copy()
    comp_df["rate_bin_fixed_a"] = assign_rate_bin_fixed_a(comp_df["treatment_value"])
    comp_df["te_comp"] = y_r1 - y_r
    comp_df["nde_comp"] = y_r1_t_r - y_r
    comp_df["nie_comp"] = y_r1 - y_r1_t_r
    comp_df["closure_comp"] = comp_df["te_comp"] - comp_df["nde_comp"] - comp_df["nie_comp"]
    for delta in cde_deltas:
        y_cde_up = models.outcome_model.predict(
            build_outcome_features(
                treatment=r_star_vec,
                mediator=mediator_obs + float(delta),
                cov_df=cov_df,
            )
        )
        col = f"cde_temp_plus_{str(delta).replace('.', 'p')}_comp"
        comp_df[col] = y_cde_up - y_cde_base
    return comp_df


def summarize_components(comp_df: pd.DataFrame, cde_deltas: Sequence[float]) -> dict[str, float]:
    """Summarize row-level decomposition components."""
    te = float(comp_df["te_comp"].mean())
    nde = float(comp_df["nde_comp"].mean())
    nie = float(comp_df["nie_comp"].mean())
    out = {
        "te_r": te,
        "nde_r": nde,
        "nie_r": nie,
        "nie_share": nie / te if np.isfinite(te) and abs(te) > 1e-12 else float("nan"),
        "closure_error": te - nde - nie,
        "n_rows": float(comp_df.shape[0]),
        "n_clusters": float(comp_df["cluster_id"].nunique()),
        "cycle_t_min": float(comp_df["cycle_t"].min()),
        "cycle_t_max": float(comp_df["cycle_t"].max()),
    }
    for delta in cde_deltas:
        col = f"cde_temp_plus_{str(delta).replace('.', 'p')}_comp"
        out[f"cde_temp_plus_{str(delta).replace('.', 'p')}"] = float(comp_df[col].mean())
    return out


def estimate_point_effects(
    df: pd.DataFrame,
    cde_deltas: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate global and stage effects for one dataset."""
    models, cov_df = fit_models(df)
    comp_df = predict_components(df=df, models=models, cov_df=cov_df, cde_deltas=cde_deltas)

    global_row = summarize_components(comp_df=comp_df, cde_deltas=cde_deltas)
    treatment = df["treatment_value"].to_numpy(dtype=float)
    mediator_raw = df["temp_mediator_raw"].to_numpy(dtype=float)
    overlap_diag = compute_weight_diagnostics(treatment=treatment, cov_df=cov_df)
    global_row.update(
        {
            "treatment_min": float(np.min(treatment)),
            "treatment_max": float(np.max(treatment)),
            "treatment_q01": float(np.quantile(treatment, 0.01)),
            "treatment_q99": float(np.quantile(treatment, 0.99)),
            "support_plus1_share": float(np.mean((treatment + 1.0 >= np.min(treatment)) & (treatment + 1.0 <= np.max(treatment)))),
            "mediator_min": float(np.min(mediator_raw)),
            "mediator_max": float(np.max(mediator_raw)),
            "mediator_q01": float(np.quantile(mediator_raw, 0.01)),
            "mediator_q99": float(np.quantile(mediator_raw, 0.99)),
        }
    )
    global_row.update(overlap_diag)
    global_df = pd.DataFrame([global_row])

    stage_rows: list[dict[str, float | str]] = []
    for scheme in SCHEME_ORDER:
        cfg = STAGE_SCHEME_CONFIGS[scheme]
        stage_col = str(cfg["stage_col"])
        stage_order = list(cfg["stage_order"])  # type: ignore[assignment]
        label_map = get_stage_label_map(scheme)
        for group in stage_order:
            part = comp_df.loc[comp_df[stage_col] == group].copy()
            if part.empty:
                continue
            row = summarize_components(part, cde_deltas=cde_deltas)
            row["scheme"] = scheme
            row["group"] = group
            row["group_label"] = label_map.get(group, group)
            stage_rows.append(row)
    stage_df = pd.DataFrame(stage_rows)
    stage_fixed_df = stage_df.loc[stage_df["scheme"] == "fixed4"].copy()
    stage_quant_df = stage_df.loc[stage_df["scheme"] == "quantile4"].copy()

    rate_rows: list[dict[str, float | str]] = []
    for group in RATE_BIN_FIXED_A_ORDER:
        part = comp_df.loc[comp_df["rate_bin_fixed_a"] == group].copy()
        if part.empty:
            continue
        row = summarize_components(part, cde_deltas=cde_deltas)
        row["rate_bin"] = group
        row["rate_bin_label"] = RATE_BIN_FIXED_A_LABEL_MAP.get(group, group)
        rate_rows.append(row)
    rate_df = pd.DataFrame(rate_rows)
    return global_df, stage_fixed_df, stage_quant_df, stage_df, rate_df


def _sample_cluster_indices(
    cluster_to_indices: dict[str, np.ndarray],
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample row indices by cluster bootstrap."""
    sampled_clusters = rng.choice(cluster_ids, size=cluster_ids.shape[0], replace=True)
    sampled_indices = [cluster_to_indices[cid] for cid in sampled_clusters]
    return np.concatenate(sampled_indices, axis=0)


def bootstrap_effects(
    df: pd.DataFrame,
    cde_deltas: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run cluster bootstrap and return global/stage bootstrap tables."""
    if n_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cluster_map = {
        cid: np.asarray(idx, dtype=int)
        for cid, idx in df.groupby("cluster_id").indices.items()
    }
    cluster_ids = np.asarray(list(cluster_map.keys()), dtype=object)
    rng = np.random.default_rng(seed)

    global_rows: list[pd.DataFrame] = []
    stage_rows: list[pd.DataFrame] = []
    rate_rows: list[pd.DataFrame] = []
    for b in range(n_bootstrap):
        sample_idx = _sample_cluster_indices(cluster_to_indices=cluster_map, cluster_ids=cluster_ids, rng=rng)
        boot_df = df.iloc[sample_idx].copy()
        try:
            g_df, _, _, s_df, r_df = estimate_point_effects(df=boot_df, cde_deltas=cde_deltas)
            g_df = g_df.assign(bootstrap_id=b)
            s_df = s_df.assign(bootstrap_id=b)
            r_df = r_df.assign(bootstrap_id=b)
            global_rows.append(g_df)
            stage_rows.append(s_df)
            rate_rows.append(r_df)
        except Exception:
            continue
    out_global = pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame()
    out_stage = pd.concat(stage_rows, ignore_index=True) if stage_rows else pd.DataFrame()
    out_rate = pd.concat(rate_rows, ignore_index=True) if rate_rows else pd.DataFrame()
    return out_global, out_stage, out_rate


def attach_ci(
    point_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    key_cols: Sequence[str],
    metric_cols: Sequence[str],
    n_bootstrap: int,
) -> pd.DataFrame:
    """Attach 95% bootstrap CI for selected metrics."""
    out = point_df.copy()
    out["bootstrap_success"] = 0
    if boot_df.empty:
        for metric in metric_cols:
            out[f"{metric}_ci_low"] = np.nan
            out[f"{metric}_ci_high"] = np.nan
        out["bootstrap_n"] = n_bootstrap
        return out

    for metric in metric_cols:
        out[f"{metric}_ci_low"] = np.nan
        out[f"{metric}_ci_high"] = np.nan

    for idx, row in out.iterrows():
        mask = np.ones(boot_df.shape[0], dtype=bool)
        for col in key_cols:
            mask = mask & (boot_df[col].astype(str).to_numpy() == str(row[col]))
        part = boot_df.loc[mask].copy()
        out.at[idx, "bootstrap_success"] = int(part.shape[0])
        if part.empty:
            continue
        for metric in metric_cols:
            values = part[metric].to_numpy(dtype=float)
            out.at[idx, f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
            out.at[idx, f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    out["bootstrap_n"] = n_bootstrap
    return out


def _format_float(value: float, digits: int = 6) -> str:
    """Format float safely for report rendering."""
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render dataframe as markdown table without extra dependency."""
    if df.empty:
        return "_无数据_"
    part = df.copy()
    if max_rows is not None and part.shape[0] > max_rows:
        part = part.head(max_rows).copy()

    headers = list(part.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in part.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(_format_float(float(val)))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _build_nonnegative_yerr(
    y: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> np.ndarray:
    """Build non-negative yerr for matplotlib error bars."""
    y_arr = np.asarray(y, dtype=float)
    low_arr = np.asarray(ci_low, dtype=float)
    high_arr = np.asarray(ci_high, dtype=float)
    lower = np.maximum(y_arr - low_arr, 0.0)
    upper = np.maximum(high_arr - y_arr, 0.0)
    lower[~np.isfinite(lower)] = 0.0
    upper[~np.isfinite(upper)] = 0.0
    return np.vstack([lower, upper])


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return robust ratio with finite/near-zero guard."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _is_ci_significant(ci_low: float, ci_high: float) -> bool:
    """Check whether CI does not cross zero."""
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return False
    return bool((ci_low > 0.0) or (ci_high < 0.0))


def build_contribution_summary(global_df: pd.DataFrame, cde_deltas: Sequence[float]) -> pd.DataFrame:
    """Build contribution summary from baseline global output."""
    baseline = global_df.loc[global_df["scenario"] == BASELINE_SCENARIO].copy()
    if baseline.empty:
        return pd.DataFrame()
    baseline = baseline.sort_values("mode_order").reset_index(drop=True)

    cde_keys = [f"cde_temp_plus_{str(float(d)).replace('.', 'p')}" for d in cde_deltas]

    rows: list[dict[str, object]] = []
    for _, row in baseline.iterrows():
        te = float(row["te_r"])
        nde = float(row["nde_r"])
        nie = float(row["nie_r"])
        abs_total = abs(nde) + abs(nie)
        abs_nde_share = _safe_ratio(abs(nde), abs_total)
        abs_nie_share = _safe_ratio(abs(nie), abs_total)

        if np.isfinite(abs_nie_share) and abs_nie_share >= 0.6:
            path_dominance = "温度路径主导"
        elif np.isfinite(abs_nde_share) and abs_nde_share >= 0.6:
            path_dominance = "直接路径主导"
        else:
            path_dominance = "混合"

        row_out: dict[str, object] = {
            "treatment_mode": row["treatment_mode"],
            "treatment_mode_display": MODE_DISPLAY.get(str(row["treatment_mode"]), str(row["treatment_mode"])),
            "te_r": te,
            "te_r_ci_low": float(row["te_r_ci_low"]),
            "te_r_ci_high": float(row["te_r_ci_high"]),
            "nde_r": nde,
            "nde_r_ci_low": float(row["nde_r_ci_low"]),
            "nde_r_ci_high": float(row["nde_r_ci_high"]),
            "nie_r": nie,
            "nie_r_ci_low": float(row["nie_r_ci_low"]),
            "nie_r_ci_high": float(row["nie_r_ci_high"]),
            "nie_share": float(row["nie_share"]),
            "nde_share": _safe_ratio(nde, te),
            "nie_share_signed": _safe_ratio(nie, te),
            "abs_nde_share": abs_nde_share,
            "abs_nie_share": abs_nie_share,
            "te_significant": _is_ci_significant(float(row["te_r_ci_low"]), float(row["te_r_ci_high"])),
            "nde_significant": _is_ci_significant(float(row["nde_r_ci_low"]), float(row["nde_r_ci_high"])),
            "nie_significant": _is_ci_significant(float(row["nie_r_ci_low"]), float(row["nie_r_ci_high"])),
            "path_dominance": path_dominance,
            "mode_order": int(row["mode_order"]) if np.isfinite(row["mode_order"]) else 99,
        }
        for key in cde_keys:
            key_low = f"{key}_ci_low"
            key_high = f"{key}_ci_high"
            val = float(row[key]) if key in row.index else float("nan")
            low = float(row[key_low]) if key_low in row.index else float("nan")
            high = float(row[key_high]) if key_high in row.index else float("nan")
            row_out[key] = val
            row_out[key_low] = low
            row_out[key_high] = high
            row_out[f"{key}_significant"] = _is_ci_significant(low, high)
        rows.append(row_out)
    return pd.DataFrame(rows).sort_values("mode_order").reset_index(drop=True)


def _fmt_ci_pair(row: pd.Series, metric: str, digits: int = 6) -> str:
    """Format point and CI for report sentence."""
    point = _format_float(float(row[metric]), digits)
    low = _format_float(float(row[f"{metric}_ci_low"]), digits)
    high = _format_float(float(row[f"{metric}_ci_high"]), digits)
    return f"{point}（95%CI: {low}, {high}）"


def plot_contribution_decomposition_global(contrib_df: pd.DataFrame, output_path: Path) -> None:
    """Plot global TE/NDE/NIE decomposition from contribution summary."""
    if contrib_df.empty:
        return
    plot_df = contrib_df.sort_values("mode_order").copy()
    labels = plot_df["treatment_mode_display"].tolist()
    x = np.arange(len(labels))
    width = 0.24
    metrics = ["te_r", "nde_r", "nie_r"]
    colors = ["#1d4ed8", "#16a34a", "#f59e0b"]

    plt.figure(figsize=(10.2, 5.4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        y = plot_df[metric].to_numpy(dtype=float)
        low = plot_df[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = plot_df[f"{metric}_ci_high"].to_numpy(dtype=float)
        yerr = _build_nonnegative_yerr(y=y, ci_low=low, ci_high=high)
        pos = x + (i - 1) * width
        plt.bar(pos, y, width=width, color=color, alpha=0.9, label=metric.upper())
        plt.errorbar(pos, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    plt.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel("对未来相对容量下降的边际影响")
    plt.title("全局贡献分解：TE / NDE / NIE（+1C）")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_contribution_share_global(contrib_df: pd.DataFrame, output_path: Path) -> None:
    """Plot signed and absolute contribution shares by treatment mode."""
    if contrib_df.empty:
        return
    plot_df = contrib_df.sort_values("mode_order").copy()
    labels = plot_df["treatment_mode_display"].tolist()
    x = np.arange(len(labels))
    width = 0.18

    series = [
        ("nde_share", "NDE/TE(签名)", "#2563eb"),
        ("nie_share_signed", "NIE/TE(签名)", "#f59e0b"),
        ("abs_nde_share", "|NDE|路径占比", "#1e40af"),
        ("abs_nie_share", "|NIE|路径占比", "#b45309"),
    ]

    plt.figure(figsize=(10.2, 5.4))
    for i, (col, label, color) in enumerate(series):
        y = plot_df[col].to_numpy(dtype=float)
        pos = x + (i - 1.5) * width
        plt.bar(pos, y, width=width, label=label, color=color, alpha=0.88)
    plt.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel("贡献占比")
    plt.title("全局贡献占比：签名占比与绝对占比")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_primary_window_mean_stage_decomposition(
    stage_df: pd.DataFrame,
    scheme: str,
    output_path: Path,
) -> None:
    """Plot stage decomposition for primary mode window_mean."""
    stage_part = stage_df.loc[
        (stage_df["scenario"] == BASELINE_SCENARIO)
        & (stage_df["scheme"] == scheme)
        & (stage_df["treatment_mode"] == PRIMARY_TREATMENT_MODE)
    ].copy()
    if stage_part.empty:
        return
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    order = list(cfg["stage_order"])  # type: ignore[assignment]
    label_map = get_stage_label_map(scheme)
    stage_part["group"] = pd.Categorical(stage_part["group"], categories=order, ordered=True)
    stage_part = stage_part.sort_values("group")
    x = np.arange(stage_part.shape[0])

    plt.figure(figsize=(10.2, 5.4))
    metrics = [
        ("te_r", "TE", "#1d4ed8"),
        ("nde_r", "NDE", "#16a34a"),
        ("nie_r", "NIE", "#f59e0b"),
    ]
    for metric, label, color in metrics:
        y = stage_part[metric].to_numpy(dtype=float)
        low = stage_part[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = stage_part[f"{metric}_ci_high"].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.8, label=label, color=color)
        plt.fill_between(x, low, high, alpha=0.15, color=color)
    plt.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    plt.xticks(x, [label_map.get(str(g), str(g)) for g in stage_part["group"].astype(str)])
    plt.ylabel("对未来相对容量下降的边际影响")
    plt.title(f"主口径 window_mean 分阶段贡献分解：{cfg['scheme_display']}")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_window_mean_rate_bin_fixed_a_decomposition(rate_bin_df: pd.DataFrame, output_path: Path) -> None:
    """Plot TE/NDE/NIE by fixed-rate bins for primary window_mean mode."""
    part = rate_bin_df.loc[
        (rate_bin_df["scenario"] == BASELINE_SCENARIO)
        & (rate_bin_df["treatment_mode"] == PRIMARY_TREATMENT_MODE)
    ].copy()
    if part.empty:
        return
    part["rate_bin"] = pd.Categorical(part["rate_bin"], categories=RATE_BIN_FIXED_A_ORDER, ordered=True)
    part = part.sort_values("rate_bin")
    x = np.arange(part.shape[0], dtype=float)
    width = 0.24
    metrics = ["te_r", "nde_r", "nie_r"]
    colors = ["#1d4ed8", "#16a34a", "#f59e0b"]
    labels = [RATE_BIN_FIXED_A_LABEL_MAP.get(str(v), str(v)) for v in part["rate_bin"].astype(str)]

    plt.figure(figsize=(10.4, 5.4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        y = part[metric].to_numpy(dtype=float)
        low = part[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = part[f"{metric}_ci_high"].to_numpy(dtype=float)
        yerr = _build_nonnegative_yerr(y=y, ci_low=low, ci_high=high)
        pos = x + (i - 1) * width
        plt.bar(pos, y, width=width, color=color, alpha=0.9, label=metric.upper())
        plt.errorbar(pos, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    plt.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel("对未来相对容量下降的边际影响")
    plt.title("方案A固定阈值倍率分段：TE / NDE / NIE（window_mean）")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_window_mean_rate_bin_fixed_a_nie_share(rate_bin_df: pd.DataFrame, output_path: Path) -> None:
    """Plot NIE share by fixed-rate bins for primary window_mean mode."""
    part = rate_bin_df.loc[
        (rate_bin_df["scenario"] == BASELINE_SCENARIO)
        & (rate_bin_df["treatment_mode"] == PRIMARY_TREATMENT_MODE)
    ].copy()
    if part.empty:
        return
    part["rate_bin"] = pd.Categorical(part["rate_bin"], categories=RATE_BIN_FIXED_A_ORDER, ordered=True)
    part = part.sort_values("rate_bin")
    x = np.arange(part.shape[0], dtype=float)
    labels = [RATE_BIN_FIXED_A_LABEL_MAP.get(str(v), str(v)) for v in part["rate_bin"].astype(str)]
    y = part["nie_share"].to_numpy(dtype=float)
    low = part["nie_share_ci_low"].to_numpy(dtype=float)
    high = part["nie_share_ci_high"].to_numpy(dtype=float)

    plt.figure(figsize=(10.4, 5.2))
    plt.plot(x, y, marker="o", linewidth=1.8, color="#f59e0b")
    plt.fill_between(x, low, high, alpha=0.2, color="#f59e0b")
    plt.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel("NIE / TE（温度路径贡献占比）")
    plt.title("方案A固定阈值倍率分段：NIE占比（window_mean）")
    plt.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_global_decomposition(global_df: pd.DataFrame, output_path: Path) -> None:
    """Plot global TE/NDE/NIE comparison by treatment mode."""
    baseline = global_df.loc[global_df["scenario"] == BASELINE_SCENARIO].copy()
    baseline = baseline.sort_values("mode_order")
    if baseline.empty:
        return
    labels = [MODE_DISPLAY.get(str(x), str(x)) for x in baseline["treatment_mode"]]
    x = np.arange(len(labels))
    width = 0.24
    metrics = ["te_r", "nde_r", "nie_r"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    plt.figure(figsize=(9.8, 5.2))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        y = baseline[metric].to_numpy(dtype=float)
        low = baseline[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = baseline[f"{metric}_ci_high"].to_numpy(dtype=float)
        yerr = _build_nonnegative_yerr(y=y, ci_low=low, ci_high=high)
        pos = x + (i - 1) * width
        plt.bar(pos, y, width=width, label=metric.upper(), color=color, alpha=0.88)
        plt.errorbar(pos, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("对未来相对容量下降的影响")
    plt.title("全局路径分解：TE / NDE / NIE（+1C）")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_cde(global_df: pd.DataFrame, output_path: Path, deltas: Sequence[float]) -> None:
    """Plot global temperature CDE by treatment mode."""
    baseline = global_df.loc[global_df["scenario"] == BASELINE_SCENARIO].copy()
    baseline = baseline.sort_values("mode_order")
    if baseline.empty:
        return
    labels = [MODE_DISPLAY.get(str(x), str(x)) for x in baseline["treatment_mode"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(9.8, 5.2))
    for i, delta in enumerate(deltas):
        key = f"cde_temp_plus_{str(delta).replace('.', 'p')}"
        y = baseline[key].to_numpy(dtype=float)
        low = baseline[f"{key}_ci_low"].to_numpy(dtype=float)
        high = baseline[f"{key}_ci_high"].to_numpy(dtype=float)
        yerr = _build_nonnegative_yerr(y=y, ci_low=low, ci_high=high)
        pos = x + (i - (len(deltas) - 1) / 2.0) * width
        plt.bar(pos, y, width=width, label=f"+{delta}°C", alpha=0.88)
        plt.errorbar(pos, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    plt.xticks(x, labels)
    plt.ylabel("固定倍率条件下的温度直接影响")
    plt.title("温度 CDE（固定倍率 r*）")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_stage_nie_share(stage_df: pd.DataFrame, scheme: str, output_path: Path) -> None:
    """Plot NIE share by stage for one stage scheme."""
    baseline = stage_df.loc[(stage_df["scenario"] == BASELINE_SCENARIO) & (stage_df["scheme"] == scheme)].copy()
    if baseline.empty:
        return
    cfg = STAGE_SCHEME_CONFIGS[scheme]
    order = list(cfg["stage_order"])  # type: ignore[assignment]
    labels = get_stage_label_map(scheme)
    plt.figure(figsize=(9.8, 5.2))
    for mode in MODE_ORDER:
        part = baseline.loc[baseline["treatment_mode"] == mode].copy()
        part["group"] = pd.Categorical(part["group"], categories=order, ordered=True)
        part = part.sort_values("group")
        x = np.arange(part.shape[0])
        y = part["nie_share"].to_numpy(dtype=float)
        low = part["nie_share_ci_low"].to_numpy(dtype=float)
        high = part["nie_share_ci_high"].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.6, label=MODE_DISPLAY.get(mode, mode))
        plt.fill_between(x, low, high, alpha=0.15)
    plt.xticks(np.arange(len(order)), [labels.get(g, g) for g in order])
    plt.ylabel("NIE / TE（温度路径贡献占比）")
    plt.title(f"分阶段温度路径贡献占比：{cfg['scheme_display']}")
    plt.grid(axis="y", alpha=0.24)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_overlap_diagnostics(global_df: pd.DataFrame, output_path: Path) -> None:
    """Plot support-plus1 and ESS diagnostics for baseline rows."""
    baseline = global_df.loc[global_df["scenario"] == BASELINE_SCENARIO].copy()
    baseline = baseline.sort_values("mode_order")
    if baseline.empty:
        return
    labels = [MODE_DISPLAY.get(str(x), str(x)) for x in baseline["treatment_mode"]]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(9.8, 5.2))
    support = baseline["support_plus1_share"].to_numpy(dtype=float)
    ess = baseline["effective_sample_size"].to_numpy(dtype=float)
    ax1.bar(x, support, width=0.45, color="#2563EB", alpha=0.75, label="support_plus1_share")
    ax1.set_ylim(0.0, 1.02)
    ax1.set_ylabel("R+1C 支持区间覆盖率")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(x, ess, color="#DC2626", marker="o", linewidth=1.8, label="ESS")
    ax2.set_ylabel("有效样本量 ESS")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    plt.title("支持区间与权重稳定性诊断")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def render_markdown_report(
    args: argparse.Namespace,
    cde_deltas: Sequence[float],
    global_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    data_diag_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    rate_bin_df: pd.DataFrame,
) -> str:
    """Render Chinese markdown report with fixed rate-bin scheme A as primary view."""
    baseline_global = global_df.loc[global_df["scenario"] == BASELINE_SCENARIO].copy().sort_values("mode_order")
    baseline_stage = stage_df.loc[stage_df["scenario"] == BASELINE_SCENARIO].copy()
    fixed_stage = baseline_stage.loc[baseline_stage["scheme"] == "fixed4"].copy().sort_values(
        ["mode_order", "stage_order"]
    )
    quant_stage = baseline_stage.loc[baseline_stage["scheme"] == "quantile4"].copy().sort_values(
        ["mode_order", "stage_order"]
    )
    rate_part = rate_bin_df.loc[
        (rate_bin_df["scenario"] == BASELINE_SCENARIO)
        & (rate_bin_df["treatment_mode"] == PRIMARY_TREATMENT_MODE)
    ].copy()
    if not rate_part.empty:
        rate_part["rate_bin"] = pd.Categorical(rate_part["rate_bin"], categories=RATE_BIN_FIXED_A_ORDER, ordered=True)
        rate_part = rate_part.sort_values("rate_bin")
    data_diag_df = data_diag_df.sort_values(["scenario_order", "mode_order"])
    sensitivity_df = sensitivity_df.sort_values(["scenario_order", "mode_order"])
    contribution_df = contribution_df.sort_values("mode_order") if not contribution_df.empty else contribution_df

    cde_keys = [f"cde_temp_plus_{str(float(d)).replace('.', 'p')}" for d in cde_deltas]
    cde_display = [f"+{_format_float(float(d), 1)}°C" for d in cde_deltas]

    lines: list[str] = []
    lines.append("# 因果路径分解报告：固定阈值倍率分段（方案A）")
    lines.append("")
    lines.append("## 1. 分析设定")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{sys.executable}`")
    lines.append(f"- 结果定义：`Y=(Q_t-Q_(t+H))/Q_t`，其中 `H={args.horizon_cycles}`")
    lines.append("- 主口径：`window_mean`（窗口真实平均充电倍率）。")
    lines.append("- 路径分解：`TE = NDE + NIE`。")
    lines.append("- 中介时序：`R_t -> T_(t+1) -> Y_(t+H)`。")
    lines.append(
        "- 温度定义：单次充电循环平均温度，按 `charge_duration_s` 对 `avg_temper` 加权聚合。"
    )
    lines.append(
        f"- 温度过滤：`[{_format_float(args.temperature_min, 1)}, {_format_float(args.temperature_max, 1)}]`；"
        f"建模裁剪：`q{int(args.temperature_clip_quantile_low*100):02d}~q{int(args.temperature_clip_quantile_high*100):02d}`。"
    )
    lines.append("- 方案A固定倍率分段：`<1.8C`、`1.8-2.6C`、`2.6-3.5C`、`>=3.5C`。")
    lines.append("- 温度直接效应 CDE：" + "、".join([f"`{x}`" for x in cde_display]) + "。")
    lines.append("")

    lines.append("## 2. 执行摘要（window_mean + 方案A）")
    primary_row: pd.Series | None = None
    if not contribution_df.empty:
        part = contribution_df.loc[contribution_df["treatment_mode"] == PRIMARY_TREATMENT_MODE].copy()
        if not part.empty:
            primary_row = part.iloc[0]
    if primary_row is not None:
        lines.append(f"- `TE`: {_fmt_ci_pair(primary_row, 'te_r')}")
        lines.append(f"- `NDE`: {_fmt_ci_pair(primary_row, 'nde_r')}")
        lines.append(f"- `NIE`: {_fmt_ci_pair(primary_row, 'nie_r')}")
        lines.append(
            f"- 温度中介贡献占比：`NIE/TE={_format_float(float(primary_row['nie_share_signed']), 4)}`；"
            f"`|NIE|路径占比={_format_float(float(primary_row['abs_nie_share']), 4)}`。"
        )
        cde_parts: list[str] = []
        for d, key in zip(cde_deltas, cde_keys):
            if key in primary_row.index:
                cde_parts.append(f"`+{_format_float(float(d), 1)}°C` 为 {_fmt_ci_pair(primary_row, key)}")
        if cde_parts:
            lines.append("- 温度直接效应（固定倍率）：" + "；".join(cde_parts) + "。")
        lines.append(f"- 路径主导类型：**{primary_row['path_dominance']}**。")
    lines.append("- 判读规则：95%CI 跨 0 表示方向不稳定，不跨 0 表示方向更稳定。")
    lines.append("")

    lines.append("## 3. 方案A固定倍率分段主结果")
    rate_cols = [
        "rate_bin_label",
        "te_r",
        "te_r_ci_low",
        "te_r_ci_high",
        "nde_r",
        "nde_r_ci_low",
        "nde_r_ci_high",
        "nie_r",
        "nie_r_ci_low",
        "nie_r_ci_high",
        "nie_share",
        "closure_error",
        "n_rows",
        "n_clusters",
    ]
    lines.append(df_to_markdown(rate_part[rate_cols] if not rate_part.empty else pd.DataFrame()))
    lines.append("")
    lines.append("![方案A分段贡献分解](./fig_window_mean_rate_bin_fixed_a_decomposition.png)")
    lines.append("- X轴含义：固定阈值倍率分段（方案A）。")
    lines.append("- Y轴含义：`TE/NDE/NIE` 对未来相对容量下降的边际影响。")
    lines.append("- 关键性结论：直接展示不同倍率区间的总效应、直接效应与温度中介效应。")
    lines.append("- 业务解释：用于设定分倍率区间的差异化控温与控倍率策略。")
    lines.append("")
    lines.append("![方案A分段NIE占比](./fig_window_mean_rate_bin_fixed_a_nie_share.png)")
    lines.append("- X轴含义：固定阈值倍率分段（方案A）。")
    lines.append("- Y轴含义：`NIE/TE`（温度路径贡献占比）。")
    lines.append("- 关键性结论：定位温度路径在不同倍率区间中的相对重要性。")
    lines.append("- 业务解释：若高倍率段 `NIE/TE` 更高，说明高倍率区更需优先强化热管理。")
    lines.append("")

    lines.append("## 4. 全局对照与贡献摘要")
    global_cols = [
        "treatment_mode",
        "te_r",
        "te_r_ci_low",
        "te_r_ci_high",
        "nde_r",
        "nde_r_ci_low",
        "nde_r_ci_high",
        "nie_r",
        "nie_r_ci_low",
        "nie_r_ci_high",
        "nie_share",
        "closure_error",
    ]
    for key in cde_keys:
        global_cols.extend([key, f"{key}_ci_low", f"{key}_ci_high"])
    lines.append(df_to_markdown(baseline_global[global_cols] if not baseline_global.empty else pd.DataFrame()))
    lines.append("")
    contrib_cols = [
        "treatment_mode",
        "te_r",
        "nde_r",
        "nie_r",
        "nie_share_signed",
        "abs_nde_share",
        "abs_nie_share",
        "te_significant",
        "nde_significant",
        "nie_significant",
    ]
    for key in cde_keys:
        contrib_cols.append(f"{key}_significant")
    contrib_cols.append("path_dominance")
    lines.append(df_to_markdown(contribution_df[contrib_cols] if not contribution_df.empty else pd.DataFrame()))
    lines.append("")
    lines.append("![温度直接效应CDE](./fig_cde_temperature_global.png)")
    lines.append("- X轴含义：倍率口径。")
    lines.append("- Y轴含义：固定倍率条件下温度上调 `ΔT` 的容量衰减增量。")
    lines.append("- 关键性结论：给出温度本身对衰减的直接边际贡献。")
    lines.append("- 业务解释：可直接用于温控收益评估。")
    lines.append("")

    lines.append("## 5. 附录：生命周期分段稳定性对照")
    stage_cols = [
        "treatment_mode",
        "group_label",
        "te_r",
        "nde_r",
        "nie_r",
        "nie_share",
        "n_rows",
        "n_clusters",
    ]
    lines.append("固定分段4段：")
    lines.append(df_to_markdown(fixed_stage[stage_cols] if not fixed_stage.empty else pd.DataFrame()))
    lines.append("")
    lines.append("分位数分段4段：")
    lines.append(df_to_markdown(quant_stage[stage_cols] if not quant_stage.empty else pd.DataFrame()))
    lines.append("")
    lines.append("![固定分段NIE占比](./fig_nie_share_fixed4.png)")
    lines.append("- X轴含义：固定寿命阶段。")
    lines.append("- Y轴含义：`NIE/TE`。")
    lines.append("- 关键性结论：寿命阶段视角下温度路径贡献变化。")
    lines.append("- 业务解释：用于验证主结论是否受分段口径影响。")
    lines.append("")
    lines.append("![分位分段NIE占比](./fig_nie_share_quantile4.png)")
    lines.append("- X轴含义：分位阶段（Q1~Q4）。")
    lines.append("- Y轴含义：`NIE/TE`。")
    lines.append("- 关键性结论：样本均衡切分下方向是否一致。")
    lines.append("- 业务解释：与固定分段交叉验证稳健性。")
    lines.append("")

    lines.append("## 6. 识别诊断与数据附录")
    diag_cols = [
        "treatment_mode",
        "support_plus1_share",
        "effective_sample_size",
        "weight_p99",
        "weight_max",
        "treatment_q01",
        "treatment_q99",
        "mediator_q01",
        "mediator_q99",
    ]
    lines.append(df_to_markdown(baseline_global[diag_cols] if not baseline_global.empty else pd.DataFrame()))
    lines.append("")
    lines.append("![支持与权重诊断](./fig_overlap_weight_diagnostics.png)")
    lines.append("- X轴含义：倍率口径。")
    lines.append("- Y轴含义：左轴 `support_plus1_share`，右轴 ESS。")
    lines.append("- 关键性结论：判断 +1C 推断是否依赖外推。")
    lines.append("- 业务解释：覆盖与ESS越稳，结论可信度越高。")
    lines.append("")
    lines.append("样本与预处理诊断：")
    lines.append(df_to_markdown(data_diag_df))
    lines.append("")
    lines.append("敏感性结果：")
    lines.append(df_to_markdown(sensitivity_df))
    lines.append("")

    lines.append("## 7. 自动结论")
    if primary_row is not None:
        lines.append(
            f"- 主口径 `{MODE_DISPLAY.get(PRIMARY_TREATMENT_MODE, PRIMARY_TREATMENT_MODE)}` 下，"
            f"`TE={_format_float(float(primary_row['te_r']))}`，`NIE/TE={_format_float(float(primary_row['nie_share_signed']), 4)}`。"
        )
        lines.append(f"- 路径主导：**{primary_row['path_dominance']}**。")
    if not rate_part.empty:
        top_bin = rate_part.iloc[int(np.nanargmax(rate_part["te_r"].to_numpy(dtype=float)))]
        lines.append(
            f"- 方案A中 `TE` 最高倍率段为 **{top_bin['rate_bin_label']}**，"
            f"点估计 `{_format_float(float(top_bin['te_r']))}`。"
        )
    closure_all = []
    if not baseline_global.empty:
        closure_all.extend(np.abs(baseline_global["closure_error"].to_numpy(dtype=float)).tolist())
    if not rate_part.empty:
        closure_all.extend(np.abs(rate_part["closure_error"].to_numpy(dtype=float)).tolist())
    if closure_all:
        lines.append(
            f"- 闭合误差最大值 `|TE-(NDE+NIE)|` 为 `{_format_float(float(np.nanmax(closure_all)), 6)}`。"
        )
    lines.append("- 解释口径：主结论优先看方向与CI，幅值结论结合样本量与诊断联合判断。")
    return "\n".join(lines)
def ensure_dir(path: Path) -> None:
    """Create output directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def scenario_order(name: str) -> int:
    """Provide fixed order for scenario display."""
    mapping = {
        BASELINE_SCENARIO: 0,
        "sensitivity_t_as_mediator": 1,
        "sensitivity_temp_5_60": 2,
    }
    return mapping.get(name, 99)


def mode_order(name: str) -> int:
    """Provide fixed order for treatment modes."""
    try:
        return MODE_ORDER.index(name)
    except ValueError:
        return 99


def main() -> int:
    """Run full mediation decomposition workflow."""
    args = parse_args()
    ensure_dir(args.output_dir.resolve())

    mode_list = [m.strip() for m in str(args.treatment_modes).split(",") if m.strip()]
    if not mode_list:
        raise ValueError("No treatment modes provided.")
    for m in mode_list:
        if m not in MODE_ORDER:
            raise ValueError(f"Unsupported treatment mode: {m}")

    cde_deltas = _parse_float_list(str(args.cde_deltas))
    for d in cde_deltas:
        if d <= 0:
            raise ValueError("All cde_deltas must be > 0.")

    effect_metrics = ["te_r", "nde_r", "nie_r", "nie_share", "closure_error"] + [
        f"cde_temp_plus_{str(d).replace('.', 'p')}" for d in cde_deltas
    ]

    global_rows: list[pd.DataFrame] = []
    stage_rows: list[pd.DataFrame] = []
    rate_rows: list[pd.DataFrame] = []
    data_diag_rows: list[dict[str, float | str]] = []
    sensitivity_rows: list[pd.DataFrame] = []

    for treatment_mode in mode_list:
        base_df, base_diag = build_mediation_dataset(
            life_path=args.life_performance_path,
            policy_path=args.policy_meaning_path,
            charge_interval_features_path=args.charge_interval_features_path,
            treatment_mode=treatment_mode,
            horizon_cycles=args.horizon_cycles,
            mediator_lag=1,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            temperature_clip_low=args.temperature_clip_quantile_low,
            temperature_clip_high=args.temperature_clip_quantile_high,
            window_mean_qref_cycles=args.window_mean_qref_cycles,
            window_mean_clip_low=args.window_mean_clip_quantile_low,
            window_mean_clip_high=args.window_mean_clip_quantile_high,
            exclude_policy_prefix=args.exclude_policy_prefix,
            encoding=args.encoding,
        )
        g_df, _, _, s_df, r_df = estimate_point_effects(df=base_df, cde_deltas=cde_deltas)
        g_boot, s_boot, r_boot = bootstrap_effects(
            df=base_df,
            cde_deltas=cde_deltas,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + mode_order(treatment_mode) * 1000,
        )
        g_df["treatment_mode"] = treatment_mode
        g_df["scenario"] = BASELINE_SCENARIO
        s_df["treatment_mode"] = treatment_mode
        s_df["scenario"] = BASELINE_SCENARIO
        r_df["treatment_mode"] = treatment_mode
        r_df["scenario"] = BASELINE_SCENARIO

        if not g_boot.empty:
            g_boot["treatment_mode"] = treatment_mode
            g_boot["scenario"] = BASELINE_SCENARIO
        if not s_boot.empty:
            s_boot["treatment_mode"] = treatment_mode
            s_boot["scenario"] = BASELINE_SCENARIO
        if not r_boot.empty:
            r_boot["treatment_mode"] = treatment_mode
            r_boot["scenario"] = BASELINE_SCENARIO

        g_df = attach_ci(
            point_df=g_df,
            boot_df=g_boot,
            key_cols=["treatment_mode", "scenario"],
            metric_cols=effect_metrics,
            n_bootstrap=args.n_bootstrap,
        )
        s_df = attach_ci(
            point_df=s_df,
            boot_df=s_boot,
            key_cols=["treatment_mode", "scenario", "scheme", "group"],
            metric_cols=effect_metrics,
            n_bootstrap=args.n_bootstrap,
        )
        r_df = attach_ci(
            point_df=r_df,
            boot_df=r_boot,
            key_cols=["treatment_mode", "scenario", "rate_bin"],
            metric_cols=effect_metrics,
            n_bootstrap=args.n_bootstrap,
        )

        global_rows.append(g_df)
        stage_rows.append(s_df)
        rate_rows.append(r_df)
        diag_row = dict(base_diag)
        diag_row["treatment_mode"] = treatment_mode
        diag_row["scenario"] = BASELINE_SCENARIO
        data_diag_rows.append(diag_row)

        sensi_t_df, sensi_t_diag = build_mediation_dataset(
            life_path=args.life_performance_path,
            policy_path=args.policy_meaning_path,
            charge_interval_features_path=args.charge_interval_features_path,
            treatment_mode=treatment_mode,
            horizon_cycles=args.horizon_cycles,
            mediator_lag=0,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            temperature_clip_low=args.temperature_clip_quantile_low,
            temperature_clip_high=args.temperature_clip_quantile_high,
            window_mean_qref_cycles=args.window_mean_qref_cycles,
            window_mean_clip_low=args.window_mean_clip_quantile_low,
            window_mean_clip_high=args.window_mean_clip_quantile_high,
            exclude_policy_prefix=args.exclude_policy_prefix,
            encoding=args.encoding,
        )
        sensi_t_global, _, _, _, _ = estimate_point_effects(df=sensi_t_df, cde_deltas=cde_deltas)
        sensi_t_global["treatment_mode"] = treatment_mode
        sensi_t_global["scenario"] = "sensitivity_t_as_mediator"
        sensitivity_rows.append(sensi_t_global)
        sensi_t_row = dict(sensi_t_diag)
        sensi_t_row["treatment_mode"] = treatment_mode
        sensi_t_row["scenario"] = "sensitivity_t_as_mediator"
        data_diag_rows.append(sensi_t_row)

        sensi_temp_df, sensi_temp_diag = build_mediation_dataset(
            life_path=args.life_performance_path,
            policy_path=args.policy_meaning_path,
            charge_interval_features_path=args.charge_interval_features_path,
            treatment_mode=treatment_mode,
            horizon_cycles=args.horizon_cycles,
            mediator_lag=1,
            temperature_min=args.sensitivity_temperature_min,
            temperature_max=args.sensitivity_temperature_max,
            temperature_clip_low=args.temperature_clip_quantile_low,
            temperature_clip_high=args.temperature_clip_quantile_high,
            window_mean_qref_cycles=args.window_mean_qref_cycles,
            window_mean_clip_low=args.window_mean_clip_quantile_low,
            window_mean_clip_high=args.window_mean_clip_quantile_high,
            exclude_policy_prefix=args.exclude_policy_prefix,
            encoding=args.encoding,
        )
        sensi_temp_global, _, _, _, _ = estimate_point_effects(df=sensi_temp_df, cde_deltas=cde_deltas)
        sensi_temp_global["treatment_mode"] = treatment_mode
        sensi_temp_global["scenario"] = "sensitivity_temp_5_60"
        sensitivity_rows.append(sensi_temp_global)
        sensi_temp_row = dict(sensi_temp_diag)
        sensi_temp_row["treatment_mode"] = treatment_mode
        sensi_temp_row["scenario"] = "sensitivity_temp_5_60"
        data_diag_rows.append(sensi_temp_row)

    global_df = pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame()
    stage_df = pd.concat(stage_rows, ignore_index=True) if stage_rows else pd.DataFrame()
    rate_df = pd.concat(rate_rows, ignore_index=True) if rate_rows else pd.DataFrame()
    sensitivity_df = pd.concat(sensitivity_rows, ignore_index=True) if sensitivity_rows else pd.DataFrame()
    data_diag_df = pd.DataFrame(data_diag_rows)

    for df in [global_df, stage_df, rate_df, sensitivity_df, data_diag_df]:
        if df.empty:
            continue
        df["mode_order"] = df["treatment_mode"].map(mode_order)
        df["scenario_order"] = df["scenario"].map(scenario_order)
        if "scheme" in df.columns:
            df["scheme_order"] = df["scheme"].map(lambda x: SCHEME_ORDER.index(x) if x in SCHEME_ORDER else 99)
        if "group" in df.columns and "scheme" in df.columns:
            stage_order_map: dict[tuple[str, str], int] = {}
            for scheme in SCHEME_ORDER:
                order = STAGE_SCHEME_CONFIGS[scheme]["stage_order"]  # type: ignore[index]
                for idx, g in enumerate(order):  # type: ignore[arg-type]
                    stage_order_map[(scheme, g)] = int(idx)
            df["stage_order"] = df.apply(
                lambda row: stage_order_map.get((str(row.get("scheme", "")), str(row.get("group", ""))), 99),
                axis=1,
            )
        if "rate_bin" in df.columns:
            rate_order_map = {name: idx for idx, name in enumerate(RATE_BIN_FIXED_A_ORDER)}
            df["rate_bin_order"] = df["rate_bin"].map(lambda x: rate_order_map.get(str(x), 99))
        df["treatment_mode_display"] = df["treatment_mode"].map(lambda x: MODE_DISPLAY.get(str(x), str(x)))

    contribution_df = build_contribution_summary(global_df=global_df, cde_deltas=cde_deltas)

    output_dir = args.output_dir.resolve()
    global_df.to_csv(output_dir / "mediation_effect_global.csv", index=False, encoding=args.encoding)
    stage_df.to_csv(output_dir / "mediation_effect_by_stage.csv", index=False, encoding=args.encoding)
    stage_df.loc[stage_df["scheme"] == "fixed4"].to_csv(
        output_dir / "mediation_effect_by_stage_fixed4.csv",
        index=False,
        encoding=args.encoding,
    )
    stage_df.loc[stage_df["scheme"] == "quantile4"].to_csv(
        output_dir / "mediation_effect_by_stage_quantile4.csv",
        index=False,
        encoding=args.encoding,
    )
    rate_df.to_csv(
        output_dir / "mediation_effect_by_rate_bin_fixed_a_window_mean.csv",
        index=False,
        encoding=args.encoding,
    )
    sensitivity_df.to_csv(output_dir / "mediation_sensitivity_global.csv", index=False, encoding=args.encoding)
    data_diag_df.to_csv(output_dir / "mediation_dataset_diagnostics.csv", index=False, encoding=args.encoding)
    contribution_df.to_csv(output_dir / "mediation_contribution_summary.csv", index=False, encoding=args.encoding)

    rate_sample_df = pd.DataFrame()
    rate_sample_part = rate_df.loc[
        (rate_df["scenario"] == BASELINE_SCENARIO)
        & (rate_df["treatment_mode"] == PRIMARY_TREATMENT_MODE)
    ].copy()
    if not rate_sample_part.empty:
        rate_sample_part["rate_bin"] = pd.Categorical(
            rate_sample_part["rate_bin"],
            categories=RATE_BIN_FIXED_A_ORDER,
            ordered=True,
        )
        rate_sample_part = rate_sample_part.sort_values("rate_bin")
        total_rows = float(rate_sample_part["n_rows"].sum())
        rate_sample_df = rate_sample_part[
            ["rate_bin", "rate_bin_label", "n_rows", "n_clusters"]
        ].copy()
        rate_sample_df["row_share"] = (
            rate_sample_df["n_rows"] / total_rows if total_rows > 0 else float("nan")
        )
    rate_sample_df.to_csv(
        output_dir / "rate_bin_fixed_a_sample_distribution.csv",
        index=False,
        encoding=args.encoding,
    )

    plot_global_decomposition(global_df=global_df, output_path=output_dir / "fig_path_decomposition_global.png")
    plot_cde(global_df=global_df, output_path=output_dir / "fig_cde_temperature_global.png", deltas=cde_deltas)
    plot_contribution_decomposition_global(
        contrib_df=contribution_df,
        output_path=output_dir / "fig_contribution_decomposition_global.png",
    )
    plot_contribution_share_global(
        contrib_df=contribution_df,
        output_path=output_dir / "fig_contribution_share_global.png",
    )
    plot_primary_window_mean_stage_decomposition(
        stage_df=stage_df,
        scheme="fixed4",
        output_path=output_dir / "fig_primary_window_mean_stage_decomposition_fixed4.png",
    )
    plot_primary_window_mean_stage_decomposition(
        stage_df=stage_df,
        scheme="quantile4",
        output_path=output_dir / "fig_primary_window_mean_stage_decomposition_quantile4.png",
    )
    plot_window_mean_rate_bin_fixed_a_decomposition(
        rate_bin_df=rate_df,
        output_path=output_dir / "fig_window_mean_rate_bin_fixed_a_decomposition.png",
    )
    plot_window_mean_rate_bin_fixed_a_nie_share(
        rate_bin_df=rate_df,
        output_path=output_dir / "fig_window_mean_rate_bin_fixed_a_nie_share.png",
    )
    plot_stage_nie_share(
        stage_df=stage_df,
        scheme="fixed4",
        output_path=output_dir / "fig_nie_share_fixed4.png",
    )
    plot_stage_nie_share(
        stage_df=stage_df,
        scheme="quantile4",
        output_path=output_dir / "fig_nie_share_quantile4.png",
    )
    plot_overlap_diagnostics(
        global_df=global_df,
        output_path=output_dir / "fig_overlap_weight_diagnostics.png",
    )

    report_text = render_markdown_report(
        args=args,
        cde_deltas=cde_deltas,
        global_df=global_df,
        stage_df=stage_df,
        data_diag_df=data_diag_df,
        sensitivity_df=sensitivity_df,
        contribution_df=contribution_df,
        rate_bin_df=rate_df,
    )
    (output_dir / "mediation_report.md").write_text(report_text, encoding="utf-8-sig")

    print("Mediation analysis completed.")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
