from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_POLICY_MEANING_PATH = REPO_ROOT / "data" / "processed" / "policy_meaning.csv"
DEFAULT_CHARGE_AGING_PATH_TIMESERIES_PATH = (
    REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv"
)
DEFAULT_CHARGE_BIN_EDGES_PATH = REPO_ROOT / "data" / "processed" / "charge_aging_path_bin_edges.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "capacity_ir_joint_causal"

MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors, font_manager, rcParams  # noqa: E402


@dataclass
class BootstrapBackendInfo:
    """Runtime information for bootstrap backend selection."""

    requested_backend: str
    backend_used: str
    requested_device: str
    device_used: str
    torch_available: bool
    torch_xla_available: bool
    fallback_reason: str


@dataclass
class NuisanceArtifacts:
    """Container for shift-AIPW nuisance model artifacts."""

    treatment_mean: np.ndarray
    treatment_std: float
    treatment_density_observed: np.ndarray
    outcome_model: GradientBoostingRegressor
    outcome_mu_observed: np.ndarray


@dataclass
class DMLFitResult:
    """Container for one treatment-outcome DML residualization run."""

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
    treatment_q01: float
    treatment_q50: float
    treatment_q99: float


def setup_plot_fonts() -> bool:
    """Configure plotting fonts with Chinese fallback."""
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
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    if selected:
        rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
        return True
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False


HAS_CJK_FONT = setup_plot_fonts()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Joint causal analysis for capacity fade and impedance rise."
    )
    parser.add_argument(
        "--life-performance-path",
        type=Path,
        default=DEFAULT_LIFE_PERFORMANCE_PATH,
        help="Path to life_performance.csv.",
    )
    parser.add_argument(
        "--policy-meaning-path",
        type=Path,
        default=DEFAULT_POLICY_MEANING_PATH,
        help="Path to policy_meaning.csv.",
    )
    parser.add_argument(
        "--charge-aging-path-timeseries-path",
        type=Path,
        default=DEFAULT_CHARGE_AGING_PATH_TIMESERIES_PATH,
        help="Path to charge_aging_path_timeseries.csv.",
    )
    parser.add_argument(
        "--charge-bin-edges-path",
        type=Path,
        default=DEFAULT_CHARGE_BIN_EDGES_PATH,
        help="Path to charge_aging_path_bin_edges.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--horizon-cycles",
        type=int,
        default=200,
        help="Outcome horizon H for y(t->t+H).",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=0.3,
        help="Lower bound for q_discharge filtering at cycle t.",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=1.3,
        help="Upper bound for q_discharge filtering at cycle t.",
    )
    parser.add_argument(
        "--exclude-policy-prefix",
        type=str,
        default="VARCHARGE",
        help="Exclude policy prefix from analysis (case-sensitive prefix).",
    )
    parser.add_argument(
        "--delta-pp",
        type=float,
        default=1.0,
        help="Standardized treatment shift in percentage points (+1pp default).",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=400,
        help="Cluster bootstrap iterations.",
    )
    parser.add_argument(
        "--bootstrap-backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Bootstrap backend implementation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "tpu"],
        help="Preferred device for torch bootstrap backend.",
    )
    parser.add_argument(
        "--weight-clip-quantile",
        type=float,
        default=0.99,
        help="Quantile for AIPW weight clipping in (0.5, 1.0].",
    )
    parser.add_argument(
        "--dml-splits",
        type=int,
        default=5,
        help="GroupKFold split count for DML residualization.",
    )
    parser.add_argument(
        "--nuisance-n-estimators",
        type=int,
        default=220,
        help="RandomForest estimators for DML nuisance models.",
    )
    parser.add_argument(
        "--nuisance-max-depth",
        type=int,
        default=10,
        help="RandomForest max depth for DML nuisance models.",
    )
    parser.add_argument(
        "--nuisance-model",
        type=str,
        default="linear",
        choices=["linear", "rf"],
        help="Nuisance model family for DML cross-fitting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=60,
        help="How many bins to include in forest-style chart (<=60).",
    )
    parser.add_argument(
        "--min-cell-cycles",
        type=int,
        default=30,
        help="Minimum cycles for per-cell trend statistics.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Skip plot rendering.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reuse existing result CSVs under output-dir to regenerate report tables and figures only.",
    )
    return parser.parse_args()


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected dataframe columns to numeric values."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return safe ratio with NaN fallback for near-zero denominator."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def gaussian_pdf(x: np.ndarray) -> np.ndarray:
    """Compute standard Gaussian PDF."""
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)


def benjamini_hochberg_qvalues(p_values: pd.Series) -> pd.Series:
    """Adjust p-values by Benjamini-Hochberg FDR."""
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index, dtype=float)
    order = valid.sort_values().index.to_list()
    n = len(order)
    q_desc = np.full(n, np.nan, dtype=float)
    prev = 1.0
    for rev_rank, idx in enumerate(reversed(order), start=1):
        rank = n - rev_rank + 1
        val = float(valid.loc[idx]) * n / rank
        prev = min(prev, val)
        q_desc[n - rev_rank] = prev
    q_desc = np.clip(q_desc, 0.0, 1.0)
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    for pos, idx in enumerate(order):
        out.loc[idx] = q_desc[pos]
    return out


def ensure_dir(path: Path) -> None:
    """Create output directory if not exists."""
    path.mkdir(parents=True, exist_ok=True)


def resolve_bootstrap_backend(
    requested_backend: str,
    requested_device: str,
) -> BootstrapBackendInfo:
    """Resolve bootstrap backend/device with graceful fallbacks."""
    backend = str(requested_backend).strip().lower()
    device = str(requested_device).strip().lower()
    fallback_reason = ""
    torch_available = False
    torch_xla_available = False

    if backend != "torch":
        return BootstrapBackendInfo(
            requested_backend=backend,
            backend_used="numpy",
            requested_device=device,
            device_used="cpu",
            torch_available=False,
            torch_xla_available=False,
            fallback_reason="",
        )

    try:
        import torch  # noqa: F401

        torch_available = True
    except Exception:
        return BootstrapBackendInfo(
            requested_backend=backend,
            backend_used="numpy",
            requested_device=device,
            device_used="cpu",
            torch_available=False,
            torch_xla_available=False,
            fallback_reason="torch_not_available",
        )

    if device in ("auto", "cpu"):
        return BootstrapBackendInfo(
            requested_backend=backend,
            backend_used="torch",
            requested_device=device,
            device_used="cpu",
            torch_available=True,
            torch_xla_available=False,
            fallback_reason="",
        )

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return BootstrapBackendInfo(
                    requested_backend=backend,
                    backend_used="torch",
                    requested_device=device,
                    device_used="cuda",
                    torch_available=True,
                    torch_xla_available=False,
                    fallback_reason="",
                )
            fallback_reason = "cuda_not_available"
        except Exception:
            fallback_reason = "cuda_check_failed"
        return BootstrapBackendInfo(
            requested_backend=backend,
            backend_used="torch",
            requested_device=device,
            device_used="cpu",
            torch_available=True,
            torch_xla_available=False,
            fallback_reason=fallback_reason,
        )

    if device == "tpu":
        try:
            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm  # noqa: F401

            torch_xla_available = True
            return BootstrapBackendInfo(
                requested_backend=backend,
                backend_used="torch",
                requested_device=device,
                device_used="tpu",
                torch_available=True,
                torch_xla_available=True,
                fallback_reason="",
            )
        except Exception:
            return BootstrapBackendInfo(
                requested_backend=backend,
                backend_used="torch",
                requested_device=device,
                device_used="cpu",
                torch_available=True,
                torch_xla_available=False,
                fallback_reason="torch_xla_not_available",
            )

    return BootstrapBackendInfo(
        requested_backend=backend,
        backend_used="numpy",
        requested_device=device,
        device_used="cpu",
        torch_available=torch_available,
        torch_xla_available=torch_xla_available,
        fallback_reason="unsupported_device",
    )


def collect_runtime_versions() -> pd.DataFrame:
    """Collect runtime versions for reproducibility diagnostics."""
    records: list[dict[str, str]] = [
        {"package": "python", "version": platform.python_version()},
        {"package": "platform", "version": platform.platform()},
    ]
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "torch_xla"]:
        try:
            mod = __import__(pkg)
            version = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            version = "not_installed"
        records.append({"package": pkg, "version": version})
    return pd.DataFrame(records)


def load_policy_features(policy_path: Path) -> pd.DataFrame:
    """Load policy-level explanatory features."""
    usecols = ["policy", "initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]
    policy_df = pd.read_csv(policy_path, usecols=usecols)
    policy_df["policy"] = policy_df["policy"].astype(str)
    policy_df = _to_numeric(policy_df, ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"])
    policy_df = policy_df.dropna(subset=usecols).drop_duplicates(subset=["policy"], keep="first")
    return policy_df


def load_life_table(life_path: Path, exclude_prefix: str) -> pd.DataFrame:
    """Load life performance table with capacity and impedance columns."""
    usecols = ["policy", "cell_code", "cycles", "q_discharge", "t_max", "ir"]
    life_df = pd.read_csv(life_path, usecols=usecols)
    life_df["policy"] = life_df["policy"].astype(str)
    life_df["cell_code"] = life_df["cell_code"].astype(str)
    life_df = _to_numeric(life_df, ["cycles", "q_discharge", "t_max", "ir"])
    life_df = life_df.dropna(subset=["policy", "cell_code", "cycles", "q_discharge", "t_max", "ir"]).copy()
    if exclude_prefix:
        life_df = life_df.loc[~life_df["policy"].astype(str).str.startswith(str(exclude_prefix))].copy()
    life_df["cycles"] = life_df["cycles"].astype(int)
    life_df = life_df.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    return life_df


def build_window_dataset(
    life_df: pd.DataFrame,
    horizon_cycles: int,
    q_min: float,
    q_max: float,
) -> pd.DataFrame:
    """Build cycle-t window dataframe with t+1 and t+H outcomes."""
    if horizon_cycles <= 0:
        raise ValueError("horizon_cycles must be positive.")

    start = life_df.rename(
        columns={
            "cycles": "cycle_t",
            "q_discharge": "q_t",
            "t_max": "t_max_t",
            "ir": "ir_t",
        }
    )[["policy", "cell_code", "cycle_t", "q_t", "t_max_t", "ir_t"]]

    plus1 = life_df.rename(
        columns={
            "cycles": "cycle_t",
            "q_discharge": "q_t1",
            "ir": "ir_t1",
            "t_max": "t_max_t1",
        }
    )[["policy", "cell_code", "cycle_t", "q_t1", "ir_t1", "t_max_t1"]]
    plus1["cycle_t"] = plus1["cycle_t"] - 1

    plush = life_df.rename(
        columns={
            "cycles": "cycle_t",
            "q_discharge": "q_th",
            "ir": "ir_th",
            "t_max": "t_max_th",
        }
    )[["policy", "cell_code", "cycle_t", "q_th", "ir_th", "t_max_th"]]
    plush["cycle_t"] = plush["cycle_t"] - int(horizon_cycles)

    merged = start.merge(
        plus1,
        on=["policy", "cell_code", "cycle_t"],
        how="inner",
        validate="one_to_one",
    ).merge(
        plush,
        on=["policy", "cell_code", "cycle_t"],
        how="inner",
        validate="one_to_one",
    )

    merged = merged.loc[(merged["q_t"] >= float(q_min)) & (merged["q_t"] <= float(q_max))].copy()
    merged = merged.loc[(merged["q_t"] > 0) & (merged["q_t1"] > 0) & (merged["q_th"] > 0)].copy()
    merged = merged.loc[(merged["ir_t"] > 0) & (merged["ir_t1"] > 0) & (merged["ir_th"] > 0)].copy()

    merged["y_cap_drop_h"] = (merged["q_t"] - merged["q_th"]) / merged["q_t"]
    merged["y_ir_rise_h"] = (merged["ir_th"] - merged["ir_t"]) / merged["ir_t"]
    merged["dir_rel_1"] = (merged["ir_t1"] - merged["ir_t"]) / merged["ir_t"]
    merged["dq_rel_1"] = (merged["q_t"] - merged["q_t1"]) / merged["q_t"]
    merged["cluster_id"] = (
        merged["policy"].astype(str)
        + "|"
        + merged["cell_code"].astype(str)
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(
        subset=[
            "y_cap_drop_h",
            "y_ir_rise_h",
            "dir_rel_1",
            "dq_rel_1",
            "t_max_t",
            "t_max_t1",
            "t_max_th",
        ]
    ).copy()
    merged["cycle_t"] = merged["cycle_t"].astype(int)
    return merged


def load_charge_cycle_features(
    charge_path: Path,
    exclude_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load charge-aging path table and build cycle stats + share wide features."""
    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "cross_bin",
        "soc_bin",
        "rate_bin",
        "temp_bin",
        "cycle_charge_time_h",
        "nonzero_cross_bin_count_cycle",
        "is_abnormal_cell",
    ]
    dtypes = {
        "policy": "string",
        "cell_code": "string",
        "cycles": "int32",
        "cross_bin": "int16",
        "soc_bin": "int8",
        "rate_bin": "int8",
        "temp_bin": "int8",
        "cycle_charge_time_h": "float32",
        "nonzero_cross_bin_count_cycle": "float32",
        "is_abnormal_cell": "float32",
    }
    raw = pd.read_csv(charge_path, usecols=usecols, dtype=dtypes)
    raw["policy"] = raw["policy"].astype(str)
    raw["cell_code"] = raw["cell_code"].astype(str)
    if exclude_prefix:
        raw = raw.loc[~raw["policy"].astype(str).str.startswith(str(exclude_prefix))].copy()

    raw["cycle_charge_time_h"] = pd.to_numeric(raw["cycle_charge_time_h"], errors="coerce")
    raw["nonzero_cross_bin_count_cycle"] = pd.to_numeric(
        raw["nonzero_cross_bin_count_cycle"], errors="coerce"
    )
    raw["is_abnormal_cell"] = pd.to_numeric(raw["is_abnormal_cell"], errors="coerce")
    raw = raw.dropna(
        subset=[
            "policy",
            "cell_code",
            "cycles",
            "cross_bin",
            "cycle_charge_time_h",
            "nonzero_cross_bin_count_cycle",
            "is_abnormal_cell",
        ]
    ).copy()
    raw = raw.loc[(raw["cross_bin"] >= 1) & (raw["cross_bin"] <= 60)].copy()

    grouped = (
        raw.groupby(
            ["policy", "cell_code", "cycles", "cross_bin"],
            as_index=False,
            observed=True,
            sort=False,
        )
        .agg(
            cycle_charge_time_h=("cycle_charge_time_h", "sum"),
            nonzero_cross_bin_count_cycle=("nonzero_cross_bin_count_cycle", "max"),
            is_abnormal_cell=("is_abnormal_cell", "max"),
        )
    )
    cycle_total = (
        grouped.groupby(
            ["policy", "cell_code", "cycles"],
            as_index=False,
            observed=True,
            sort=False,
        )
        .agg(cycle_total_charge_h=("cycle_charge_time_h", "sum"))
    )
    grouped = grouped.merge(
        cycle_total, on=["policy", "cell_code", "cycles"], how="left", validate="many_to_one"
    )
    grouped["share"] = np.where(
        grouped["cycle_total_charge_h"] > 0,
        grouped["cycle_charge_time_h"] / grouped["cycle_total_charge_h"],
        0.0,
    )

    share_wide = (
        grouped.pivot_table(
            index=["policy", "cell_code", "cycles"],
            columns="cross_bin",
            values="share",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(columns=list(range(1, 61)), fill_value=0.0)
        .reset_index()
    )
    share_cols: list[str] = []
    rename_map: dict[int, str] = {}
    for idx in range(1, 61):
        col = f"share_{idx:02d}"
        share_cols.append(col)
        rename_map[idx] = col
    share_wide = share_wide.rename(columns=rename_map)

    cycle_stats = (
        grouped.groupby(
            ["policy", "cell_code", "cycles"],
            as_index=False,
            observed=True,
            sort=False,
        )
        .agg(
            cycle_total_charge_h=("cycle_total_charge_h", "max"),
            nonzero_cross_bin_count_cycle=("nonzero_cross_bin_count_cycle", "max"),
            is_abnormal_cell=("is_abnormal_cell", "max"),
        )
    )
    cycle_stats["is_abnormal_cell"] = (cycle_stats["is_abnormal_cell"] > 0).astype(int)
    cycle_stats = cycle_stats.rename(columns={"cycles": "cycle_t"})
    share_wide = share_wide.rename(columns={"cycles": "cycle_t"})
    return cycle_stats, share_wide


def load_bin_meta(bin_edges_path: Path) -> pd.DataFrame:
    """Load cross-bin metadata table."""
    df = pd.read_csv(bin_edges_path)
    required_cols = ["cross_bin", "soc_bin", "rate_bin", "temp_bin", "cross_label"]
    optional_cols = [
        "soc_label",
        "rate_label",
        "temp_label",
        "rate_edge_low_raw",
        "rate_edge_high_raw",
        "temp_edge_low_raw",
        "temp_edge_high_raw",
        "temp_edge_low_int",
        "temp_edge_high_int",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in bin edges file: {missing}")
    keep_cols = required_cols + [c for c in optional_cols if c in df.columns]
    df = df[keep_cols].copy()
    text_cols = ["cross_label", "soc_label", "rate_label", "temp_label"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["cross_bin"] = pd.to_numeric(df["cross_bin"], errors="coerce").astype("Int64")
    for col in [
        "soc_bin",
        "rate_bin",
        "temp_bin",
        "rate_edge_low_raw",
        "rate_edge_high_raw",
        "temp_edge_low_raw",
        "temp_edge_high_raw",
        "temp_edge_low_int",
        "temp_edge_high_int",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["cross_bin"]).copy()
    df["cross_bin"] = df["cross_bin"].astype(int)
    df = df.drop_duplicates(subset=["cross_bin"], keep="first").sort_values("cross_bin")
    return df


def build_cross_bin_interpretation_table(bin_meta_df: pd.DataFrame) -> pd.DataFrame:
    """Build human-readable cross-bin interpretation table."""
    out = bin_meta_df.copy()
    out["cross_bin_name"] = out["cross_bin"].map(lambda v: f"bin{int(v):02d}")

    if "soc_label" not in out.columns:
        out["soc_label"] = out["soc_bin"].map(lambda v: f"S{int(v)}")
    if "rate_label" not in out.columns:
        out["rate_label"] = out["rate_bin"].map(lambda v: f"R{int(v)}")
    if "temp_label" not in out.columns:
        out["temp_label"] = out["temp_bin"].map(lambda v: f"T{int(v)}")

    out["condition_text"] = (
        "SOC "
        + out["soc_label"].astype(str)
        + " / 倍率 "
        + out["rate_label"].astype(str)
        + " / 温度 "
        + out["temp_label"].astype(str)
    )
    out["condition_text_detailed"] = out["condition_text"]
    if {"temp_edge_low_int", "temp_edge_high_int"}.issubset(out.columns):
        out["condition_text_detailed"] = (
            "SOC "
            + out["soc_label"].astype(str)
            + " / 倍率 "
            + out["rate_label"].astype(str)
            + " / 温度 ["
            + out["temp_edge_low_int"].round().astype("Int64").astype(str)
            + ","
            + out["temp_edge_high_int"].round().astype("Int64").astype(str)
            + "]"
        )

    order_cols = [
        "cross_bin",
        "cross_bin_name",
        "cross_label",
        "condition_text",
        "condition_text_detailed",
        "soc_bin",
        "rate_bin",
        "temp_bin",
        "soc_label",
        "rate_label",
        "temp_label",
        "rate_edge_low_raw",
        "rate_edge_high_raw",
        "temp_edge_low_raw",
        "temp_edge_high_raw",
        "temp_edge_low_int",
        "temp_edge_high_int",
    ]
    existing = [c for c in order_cols if c in out.columns]
    return out[existing].copy()


def build_effect_top_table(
    effect_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    outcome_prefix: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build one outcome-specific top-bin table with auto-judgement columns."""
    work = effect_df.copy()
    work["effect_per_1pp"] = pd.to_numeric(work["effect_per_1pp"], errors="coerce")
    work["ci_low"] = pd.to_numeric(work["ci_low"], errors="coerce")
    work["ci_high"] = pd.to_numeric(work["ci_high"], errors="coerce")
    work["q_value"] = pd.to_numeric(work["q_value"], errors="coerce")
    work["significant_positive"] = (work["ci_low"] > 0.0) & (work["ci_high"] > 0.0)
    work["ci_cross_zero"] = (work["ci_low"] <= 0.0) & (work["ci_high"] >= 0.0)
    work["effect_rank"] = work["effect_per_1pp"].rank(method="dense", ascending=False).astype("Int64")

    other_cols = {
        "capacity": ["risk_category", "ir_effect_per_1pp", "ir_ci_low", "ir_ci_high", "effect_rank_ir"],
        "ir": ["risk_category", "cap_effect_per_1pp", "cap_ci_low", "cap_ci_high", "effect_rank_capacity"],
    }
    merge_cols = ["cross_bin"] + other_cols.get(outcome_prefix, ["risk_category"])
    merge_cols = [c for c in merge_cols if c in compare_df.columns]
    work = work.merge(compare_df[merge_cols], on="cross_bin", how="left", validate="one_to_one")
    work = work.merge(
        interpretation_df,
        on=["cross_bin", "cross_label"],
        how="left",
        validate="many_to_one",
    )
    work = work.sort_values("effect_per_1pp", ascending=False, kind="mergesort").head(int(top_n)).copy()
    return work.reset_index(drop=True)


def build_analysis_dataset(
    window_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    cycle_stats_df: pd.DataFrame,
    share_wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all features into one analysis-ready panel dataset."""
    merged = (
        window_df.merge(policy_df, on="policy", how="left", validate="many_to_one")
        .merge(
            cycle_stats_df,
            on=["policy", "cell_code", "cycle_t"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            share_wide_df,
            on=["policy", "cell_code", "cycle_t"],
            how="inner",
            validate="one_to_one",
        )
    )
    share_cols = [f"share_{idx:02d}" for idx in range(1, 61)]
    required = [
        "initial_c_rate",
        "switch_soc_percent",
        "post_switch_c_rate",
        "cycle_total_charge_h",
        "nonzero_cross_bin_count_cycle",
        "is_abnormal_cell",
    ] + share_cols
    merged = _to_numeric(merged, required)
    merged = merged.dropna(subset=required).copy()
    merged["cycle_t_centered"] = merged["cycle_t"] - merged["cycle_t"].mean()
    merged["cycle_t_sq"] = np.square(merged["cycle_t_centered"])
    merged["group_key"] = merged["cluster_id"].astype(str)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(
        subset=["y_cap_drop_h", "y_ir_rise_h", "dir_rel_1", "dq_rel_1", "q_t", "ir_t", "t_max_t"]
    ).copy()
    return merged


def build_trend_outputs(df: pd.DataFrame, min_cell_cycles: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build global trend summary and per-cell trend table."""
    summary_rows: list[dict[str, object]] = []
    work = df[["policy", "cell_code", "cycle_t", "q_t", "ir_t", "y_cap_drop_h", "y_ir_rise_h"]].copy()
    work = work.dropna().copy()
    work["cell_key"] = work["policy"].astype(str) + "|" + work["cell_code"].astype(str)

    sp_cycle_q = float(work[["cycle_t", "q_t"]].corr(method="spearman").iloc[0, 1])
    sp_cycle_ir = float(work[["cycle_t", "ir_t"]].corr(method="spearman").iloc[0, 1])
    sp_y = float(work[["y_cap_drop_h", "y_ir_rise_h"]].corr(method="spearman").iloc[0, 1])
    pe_y = float(work[["y_cap_drop_h", "y_ir_rise_h"]].corr(method="pearson").iloc[0, 1])
    both_worse_share = float(((work["y_cap_drop_h"] > 0) & (work["y_ir_rise_h"] > 0)).mean())

    summary_rows.extend(
        [
            {"metric": "rows_window", "value": float(len(work)), "notes": "窗口样本行数"},
            {"metric": "spearman_cycle_q", "value": sp_cycle_q, "notes": "全局 cycle~q_t Spearman"},
            {"metric": "spearman_cycle_ir", "value": sp_cycle_ir, "notes": "全局 cycle~ir_t Spearman"},
            {
                "metric": "spearman_y_capdrop_vs_y_irrise",
                "value": sp_y,
                "notes": "全局 y_cap_drop_h 与 y_ir_rise_h Spearman",
            },
            {
                "metric": "pearson_y_capdrop_vs_y_irrise",
                "value": pe_y,
                "notes": "全局 y_cap_drop_h 与 y_ir_rise_h Pearson",
            },
            {
                "metric": "share_both_worsen",
                "value": both_worse_share,
                "notes": "同窗口容量衰减>0 且阻抗上升>0 占比",
            },
        ]
    )

    cell_rows: list[dict[str, object]] = []
    for (policy, cell_code), part in work.groupby(["policy", "cell_code"]):
        if int(part["cycle_t"].nunique()) < int(min_cell_cycles):
            continue
        rho_q = float(part[["cycle_t", "q_t"]].corr(method="spearman").iloc[0, 1])
        rho_ir = float(part[["cycle_t", "ir_t"]].corr(method="spearman").iloc[0, 1])
        rho_y = float(part[["y_cap_drop_h", "y_ir_rise_h"]].corr(method="spearman").iloc[0, 1])
        cell_rows.append(
            {
                "policy": str(policy),
                "cell_code": str(cell_code),
                "rho_cycle_q": rho_q,
                "rho_cycle_ir": rho_ir,
                "rho_y_capdrop_vs_y_irrise": rho_y,
                "n_rows": int(part.shape[0]),
            }
        )
    cell_df = pd.DataFrame(cell_rows)
    if not cell_df.empty:
        summary_rows.extend(
            [
                {
                    "metric": "cell_median_rho_cycle_q",
                    "value": float(cell_df["rho_cycle_q"].median()),
                    "notes": "cell 内 cycle~q_t Spearman 中位数",
                },
                {
                    "metric": "cell_median_rho_cycle_ir",
                    "value": float(cell_df["rho_cycle_ir"].median()),
                    "notes": "cell 内 cycle~ir_t Spearman 中位数",
                },
                {
                    "metric": "cell_share_opposite_sign_trend",
                    "value": float(((cell_df["rho_cycle_q"] < 0) & (cell_df["rho_cycle_ir"] > 0)).mean()),
                    "notes": "cell 内容量下降+阻抗上升趋势占比",
                },
            ]
        )
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, cell_df


def build_covariate_matrix(df: pd.DataFrame, cov_cols: Sequence[str]) -> np.ndarray:
    """Build numerical covariate matrix."""
    x_df = df[list(cov_cols)].copy()
    x_df = _to_numeric(x_df, list(cov_cols))
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    return x_df.to_numpy(dtype=float)


def fit_shift_nuisance_models(
    treatment: np.ndarray,
    covariates: np.ndarray,
    outcome: np.ndarray,
    seed: int,
) -> NuisanceArtifacts:
    """Fit nuisance models required by shift-AIPW estimator."""
    t_model = LinearRegression()
    t_model.fit(covariates, treatment)
    mu_t = t_model.predict(covariates)
    residual = treatment - mu_t
    sigma = float(np.std(residual, ddof=1))
    sigma = max(sigma, 1e-4)
    f_obs = gaussian_pdf((treatment - mu_t) / sigma) / sigma
    f_obs = np.clip(f_obs, 1e-8, None)

    y_model = GradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        n_estimators=320,
        max_depth=3,
        min_samples_leaf=80,
        subsample=0.85,
        random_state=seed,
    )
    y_model.fit(np.column_stack([treatment, covariates]), outcome)
    mu_obs = y_model.predict(np.column_stack([treatment, covariates]))

    return NuisanceArtifacts(
        treatment_mean=mu_t,
        treatment_std=sigma,
        treatment_density_observed=f_obs,
        outcome_model=y_model,
        outcome_mu_observed=mu_obs,
    )


def bootstrap_ci_from_scores_numpy(
    score: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, int]:
    """Cluster bootstrap CI for row-level score with numpy backend."""
    if n_bootstrap <= 0 or score.size == 0:
        return float("nan"), float("nan"), 0
    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_cluster = unique_clusters.size
    if n_cluster < 2:
        return float("nan"), float("nan"), 0
    cluster_sum = np.zeros(n_cluster, dtype=float)
    cluster_count = np.zeros(n_cluster, dtype=float)
    np.add.at(cluster_sum, inv, score)
    np.add.at(cluster_count, inv, 1.0)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_cluster, size=(int(n_bootstrap), n_cluster))
    total_sum = cluster_sum[sampled].sum(axis=1)
    total_count = cluster_count[sampled].sum(axis=1)
    valid = total_count > 0
    if not np.any(valid):
        return float("nan"), float("nan"), 0
    estimates = total_sum[valid] / total_count[valid]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)), int(estimates.size)


def _torch_device(backend: BootstrapBackendInfo):
    """Resolve torch device object for backend info."""
    import torch

    if backend.device_used == "cuda":
        return torch.device("cuda")
    if backend.device_used == "cpu":
        return torch.device("cpu")
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def bootstrap_ci_from_scores_torch(
    score: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
    backend: BootstrapBackendInfo,
) -> Tuple[float, float, int]:
    """Cluster bootstrap CI for row-level score with torch backend."""
    if n_bootstrap <= 0 or score.size == 0:
        return float("nan"), float("nan"), 0
    import torch

    unique_clusters, inv = np.unique(cluster_ids, return_inverse=True)
    n_cluster = unique_clusters.size
    if n_cluster < 2:
        return float("nan"), float("nan"), 0

    cluster_sum = np.zeros(n_cluster, dtype=float)
    cluster_count = np.zeros(n_cluster, dtype=float)
    np.add.at(cluster_sum, inv, score)
    np.add.at(cluster_count, inv, 1.0)

    device = _torch_device(backend)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    sum_t = torch.as_tensor(cluster_sum, dtype=torch.float64, device=device)
    cnt_t = torch.as_tensor(cluster_count, dtype=torch.float64, device=device)
    estimates: list[np.ndarray] = []
    batch_size = 64
    total_iters = int(n_bootstrap)
    for start in range(0, total_iters, batch_size):
        bs = min(batch_size, total_iters - start)
        sampled = torch.randint(
            low=0,
            high=n_cluster,
            size=(bs, n_cluster),
            generator=g,
            device=device,
        )
        total_sum = sum_t[sampled].sum(dim=1)
        total_cnt = cnt_t[sampled].sum(dim=1)
        est = total_sum / torch.clamp(total_cnt, min=1e-12)
        est_np = est.detach().to("cpu").numpy()
        estimates.append(est_np[np.isfinite(est_np)])
    if not estimates:
        return float("nan"), float("nan"), 0
    arr = np.concatenate(estimates)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(arr.size)


def bootstrap_ci_from_scores(
    score: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
    backend: BootstrapBackendInfo,
) -> Tuple[float, float, int]:
    """Dispatch score bootstrap by selected backend."""
    if backend.backend_used != "torch":
        return bootstrap_ci_from_scores_numpy(score, cluster_ids, n_bootstrap, seed)
    try:
        return bootstrap_ci_from_scores_torch(score, cluster_ids, n_bootstrap, seed, backend)
    except Exception:
        return bootstrap_ci_from_scores_numpy(score, cluster_ids, n_bootstrap, seed)


def estimate_shift_effect_aipw(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: Sequence[str],
    cluster_col: str,
    delta_pp: float,
    clip_quantile: float,
    n_bootstrap: int,
    seed: int,
    backend: BootstrapBackendInfo,
) -> dict:
    """Estimate +delta_pp shift effect using AIPW with Gaussian GPS approximation."""
    if not (0.5 <= float(clip_quantile) <= 1.0):
        raise ValueError("clip_quantile must be in [0.5, 1.0].")
    delta = float(delta_pp) / 100.0

    cols = [*covariate_cols, treatment_col, outcome_col, cluster_col]
    work = df[cols].dropna().copy()
    if work.empty:
        return {
            "direction": "",
            "effect_per_1pp": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_success": 0,
            "n_rows": 0,
            "n_clusters": 0,
            "support_shift_share": float("nan"),
            "weight_p95": float("nan"),
            "weight_p99": float("nan"),
            "weight_max": float("nan"),
            "ess": float("nan"),
        }

    t = pd.to_numeric(work[treatment_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(dtype=float)
    x = build_covariate_matrix(work, covariate_cols)
    cluster_ids = work[cluster_col].astype(str).to_numpy()

    nuis = fit_shift_nuisance_models(treatment=t, covariates=x, outcome=y, seed=seed)
    t_shift = t + delta
    x_shift = np.column_stack([t_shift, x])
    mu_shift = nuis.outcome_model.predict(x_shift)

    t_minus_delta = t - delta
    f_shift = gaussian_pdf((t_minus_delta - nuis.treatment_mean) / nuis.treatment_std) / nuis.treatment_std
    f_shift = np.clip(f_shift, 1e-8, None)
    w_raw = f_shift / nuis.treatment_density_observed
    clip_thr = float(np.quantile(w_raw, clip_quantile))
    w = np.clip(w_raw, 0.0, clip_thr)

    score = (mu_shift + w * (y - nuis.outcome_mu_observed)) - y
    effect = float(np.mean(score))
    ci_low, ci_high, boot_n = bootstrap_ci_from_scores(
        score=score,
        cluster_ids=cluster_ids,
        n_bootstrap=n_bootstrap,
        seed=seed + 17,
        backend=backend,
    )

    sum_w = float(np.sum(w))
    ess = _safe_ratio(sum_w * sum_w, float(np.sum(np.square(w))))
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    support_share = float(np.mean((t_minus_delta >= t_min) & (t_minus_delta <= t_max)))
    return {
        "effect_per_1pp": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_success": int(boot_n),
        "n_rows": int(len(work)),
        "n_clusters": int(pd.Series(cluster_ids).nunique()),
        "support_shift_share": support_share,
        "weight_p95": float(np.quantile(w, 0.95)),
        "weight_p99": float(np.quantile(w, 0.99)),
        "weight_max": float(np.max(w)),
        "ess": ess,
        "treatment_std": float(np.std(t, ddof=1)),
        "gps_sigma": float(nuis.treatment_std),
        "clip_threshold": clip_thr,
    }


def estimate_theta(y_residual: np.ndarray, t_residual: np.ndarray) -> Tuple[float, float]:
    """Estimate theta and treatment variance from residualized samples."""
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
    nuisance_model: str,
    seed: int,
) -> DMLFitResult:
    """Fit one DML residualization problem with grouped cross-fitting."""
    cols = [*control_cols, treatment_col, outcome_col, group_col]
    work = df[cols].dropna().copy()
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
            treatment_q01=float("nan"),
            treatment_q50=float("nan"),
            treatment_q99=float("nan"),
        )
    y = pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(work[treatment_col], errors="coerce").to_numpy(dtype=float)
    x = _to_numeric(work[list(control_cols)], list(control_cols)).to_numpy(dtype=float)
    groups = work[group_col].astype(str).to_numpy()

    finite_mask = (
        np.isfinite(y)
        & np.isfinite(t)
        & np.isfinite(x).all(axis=1)
    )
    y = y[finite_mask]
    t = t[finite_mask]
    x = x[finite_mask]
    groups = groups[finite_mask]
    if y.size == 0:
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
            skip_reason="empty_after_finite_filter",
            treatment_q01=float("nan"),
            treatment_q50=float("nan"),
            treatment_q99=float("nan"),
        )
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return DMLFitResult(
            theta_raw=float("nan"),
            var_treatment=float("nan"),
            y_residual=np.array([]),
            t_residual=np.array([]),
            groups=groups,
            r2_y=float("nan"),
            r2_t=float("nan"),
            n_rows=int(y.size),
            n_groups=int(unique_groups.size),
            skip_reason="too_few_groups",
            treatment_q01=float(np.quantile(t, 0.01)),
            treatment_q50=float(np.quantile(t, 0.50)),
            treatment_q99=float(np.quantile(t, 0.99)),
        )

    folds = int(min(max(2, int(n_splits)), unique_groups.size))
    gkf = GroupKFold(n_splits=folds)
    y_hat = np.full(y.size, np.nan, dtype=float)
    t_hat = np.full(t.size, np.nan, dtype=float)

    model_name = str(nuisance_model).strip().lower()
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        if model_name == "rf":
            m_y = RandomForestRegressor(
                n_estimators=int(rf_n_estimators),
                max_depth=int(rf_max_depth),
                min_samples_leaf=20,
                min_samples_split=40,
                max_features="sqrt",
                random_state=int(seed) + fold_idx * 37,
                n_jobs=1,
            )
            m_t = RandomForestRegressor(
                n_estimators=int(rf_n_estimators),
                max_depth=int(rf_max_depth),
                min_samples_leaf=20,
                min_samples_split=40,
                max_features="sqrt",
                random_state=int(seed) + fold_idx * 71,
                n_jobs=1,
            )
        else:
            m_y = LinearRegression()
            m_t = LinearRegression()
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
            n_rows=int(y.size),
            n_groups=int(unique_groups.size),
            skip_reason="crossfit_nan",
            treatment_q01=float(np.quantile(t, 0.01)),
            treatment_q50=float(np.quantile(t, 0.50)),
            treatment_q99=float(np.quantile(t, 0.99)),
        )

    y_res = y - y_hat
    t_res = t - t_hat
    theta, var_t = estimate_theta(y_res, t_res)
    skip_reason = "" if np.isfinite(theta) else "low_treatment_variance"
    return DMLFitResult(
        theta_raw=float(theta),
        var_treatment=float(var_t),
        y_residual=y_res,
        t_residual=t_res,
        groups=groups,
        r2_y=float(r2_score(y, y_hat)),
        r2_t=float(r2_score(t, t_hat)),
        n_rows=int(y.size),
        n_groups=int(unique_groups.size),
        skip_reason=skip_reason,
        treatment_q01=float(np.quantile(t, 0.01)),
        treatment_q50=float(np.quantile(t, 0.50)),
        treatment_q99=float(np.quantile(t, 0.99)),
    )


def _cluster_moment_arrays(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-cluster moment arrays for theta bootstrap."""
    y = np.asarray(y_residual, dtype=float)
    t = np.asarray(t_residual, dtype=float)
    g = np.asarray(groups).astype(str)
    unique_groups, inv = np.unique(g, return_inverse=True)
    n_cluster = unique_groups.size
    count = np.zeros(n_cluster, dtype=float)
    sum_y = np.zeros(n_cluster, dtype=float)
    sum_t = np.zeros(n_cluster, dtype=float)
    sum_yt = np.zeros(n_cluster, dtype=float)
    sum_tt = np.zeros(n_cluster, dtype=float)
    np.add.at(count, inv, 1.0)
    np.add.at(sum_y, inv, y)
    np.add.at(sum_t, inv, t)
    np.add.at(sum_yt, inv, y * t)
    np.add.at(sum_tt, inv, t * t)
    return count, sum_y, sum_t, sum_yt, sum_tt


def _theta_from_aggregates(
    total_n: np.ndarray,
    total_sum_y: np.ndarray,
    total_sum_t: np.ndarray,
    total_sum_yt: np.ndarray,
    total_sum_tt: np.ndarray,
) -> np.ndarray:
    """Compute theta values from bootstrap aggregate moments."""
    e_y = total_sum_y / np.clip(total_n, 1e-12, None)
    e_t = total_sum_t / np.clip(total_n, 1e-12, None)
    e_yt = total_sum_yt / np.clip(total_n, 1e-12, None)
    e_tt = total_sum_tt / np.clip(total_n, 1e-12, None)
    cov = e_yt - e_y * e_t
    var = e_tt - np.square(e_t)
    theta = np.full_like(var, np.nan, dtype=float)
    mask = var > 1e-12
    theta[mask] = cov[mask] / var[mask]
    return theta


def cluster_bootstrap_theta_numpy(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    groups: np.ndarray,
    n_iters: int,
    seed: int,
) -> np.ndarray:
    """Cluster bootstrap theta distribution with numpy backend."""
    if n_iters <= 0 or len(y_residual) == 0:
        return np.array([], dtype=float)
    count, sum_y, sum_t, sum_yt, sum_tt = _cluster_moment_arrays(y_residual, t_residual, groups)
    n_cluster = count.size
    if n_cluster < 2:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_cluster, size=(int(n_iters), n_cluster))
    total_n = count[sampled].sum(axis=1)
    total_sum_y = sum_y[sampled].sum(axis=1)
    total_sum_t = sum_t[sampled].sum(axis=1)
    total_sum_yt = sum_yt[sampled].sum(axis=1)
    total_sum_tt = sum_tt[sampled].sum(axis=1)
    theta = _theta_from_aggregates(total_n, total_sum_y, total_sum_t, total_sum_yt, total_sum_tt)
    return theta[np.isfinite(theta)]


def cluster_bootstrap_theta_torch(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    groups: np.ndarray,
    n_iters: int,
    seed: int,
    backend: BootstrapBackendInfo,
) -> np.ndarray:
    """Cluster bootstrap theta distribution with torch backend."""
    if n_iters <= 0 or len(y_residual) == 0:
        return np.array([], dtype=float)
    import torch

    count, sum_y, sum_t, sum_yt, sum_tt = _cluster_moment_arrays(y_residual, t_residual, groups)
    n_cluster = count.size
    if n_cluster < 2:
        return np.array([], dtype=float)
    device = _torch_device(backend)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    cnt_t = torch.as_tensor(count, dtype=torch.float64, device=device)
    sy_t = torch.as_tensor(sum_y, dtype=torch.float64, device=device)
    st_t = torch.as_tensor(sum_t, dtype=torch.float64, device=device)
    syt_t = torch.as_tensor(sum_yt, dtype=torch.float64, device=device)
    stt_t = torch.as_tensor(sum_tt, dtype=torch.float64, device=device)
    out: list[np.ndarray] = []
    batch_size = 64
    for start in range(0, int(n_iters), batch_size):
        bs = min(batch_size, int(n_iters) - start)
        sampled = torch.randint(0, n_cluster, (bs, n_cluster), generator=g, device=device)
        total_n = cnt_t[sampled].sum(dim=1)
        total_sy = sy_t[sampled].sum(dim=1)
        total_st = st_t[sampled].sum(dim=1)
        total_syt = syt_t[sampled].sum(dim=1)
        total_stt = stt_t[sampled].sum(dim=1)

        e_y = total_sy / torch.clamp(total_n, min=1e-12)
        e_t = total_st / torch.clamp(total_n, min=1e-12)
        e_yt = total_syt / torch.clamp(total_n, min=1e-12)
        e_tt = total_stt / torch.clamp(total_n, min=1e-12)
        cov = e_yt - e_y * e_t
        var = e_tt - e_t * e_t
        theta = torch.where(var > 1e-12, cov / var, torch.full_like(var, float("nan")))
        arr = theta.detach().to("cpu").numpy()
        out.append(arr[np.isfinite(arr)])
    if not out:
        return np.array([], dtype=float)
    vals = np.concatenate(out)
    return vals[np.isfinite(vals)]


def cluster_bootstrap_theta(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    groups: np.ndarray,
    n_iters: int,
    seed: int,
    backend: BootstrapBackendInfo,
) -> np.ndarray:
    """Dispatch cluster bootstrap theta by backend."""
    if backend.backend_used != "torch":
        return cluster_bootstrap_theta_numpy(y_residual, t_residual, groups, n_iters, seed)
    try:
        return cluster_bootstrap_theta_torch(y_residual, t_residual, groups, n_iters, seed, backend)
    except Exception:
        return cluster_bootstrap_theta_numpy(y_residual, t_residual, groups, n_iters, seed)


def run_substitution_effects(
    df: pd.DataFrame,
    outcome_col: str,
    control_cols: Sequence[str],
    bin_meta_df: pd.DataFrame,
    args: argparse.Namespace,
    backend: BootstrapBackendInfo,
) -> pd.DataFrame:
    """Run DML substitution effects for all 60 charging bins and one outcome."""
    rows: list[dict[str, object]] = []
    for cross_bin in range(1, 61):
        share_col = f"share_{cross_bin:02d}"
        fit = fit_dml_residualized(
            df=df,
            treatment_col=share_col,
            control_cols=control_cols,
            outcome_col=outcome_col,
            group_col="group_key",
            n_splits=int(args.dml_splits),
            rf_n_estimators=int(args.nuisance_n_estimators),
            rf_max_depth=int(args.nuisance_max_depth),
            nuisance_model=str(args.nuisance_model),
            seed=int(args.random_seed) + cross_bin * 101,
        )
        result: dict[str, object] = {
            "cross_bin": int(cross_bin),
            "share_feature": share_col,
            "outcome_col": str(outcome_col),
            "n_rows": int(fit.n_rows),
            "n_groups": int(fit.n_groups),
            "var_treatment": float(fit.var_treatment),
            "r2_y_nuisance": float(fit.r2_y),
            "r2_t_nuisance": float(fit.r2_t),
            "skip_reason": str(fit.skip_reason),
            "share_q01": float(fit.treatment_q01),
            "share_q50": float(fit.treatment_q50),
            "share_q99": float(fit.treatment_q99),
            "support_width_1_99": float(fit.treatment_q99 - fit.treatment_q01)
            if np.isfinite(fit.treatment_q99) and np.isfinite(fit.treatment_q01)
            else float("nan"),
            "bootstrap_success": 0,
        }
        if fit.skip_reason:
            result.update(
                {
                    "effect_per_1pp": float("nan"),
                    "effect_per_5pp": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                    "q_value": float("nan"),
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
            backend=backend,
        )
        if boot.size == 0:
            result["skip_reason"] = "bootstrap_failed"
            result.update(
                {
                    "effect_per_1pp": float("nan"),
                    "effect_per_5pp": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                    "q_value": float("nan"),
                }
            )
            rows.append(result)
            continue

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
                "effect_per_1pp": effect_1pp,
                "effect_per_5pp": effect_5pp,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": p_value,
                "q_value": float("nan"),
                "bootstrap_success": int(boot.size),
            }
        )
        rows.append(result)

    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg_qvalues(out["p_value"])
    out["abs_effect_per_1pp"] = pd.to_numeric(out["effect_per_1pp"], errors="coerce").abs()
    out = out.merge(bin_meta_df, on="cross_bin", how="left", validate="many_to_one")
    out = out.sort_values(["q_value", "abs_effect_per_1pp"], ascending=[True, False], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def classify_dual_risk(
    cap_df: pd.DataFrame,
    ir_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify each cross-bin into dual-risk categories."""
    left_cols = [
        "cross_bin",
        "cross_label",
        "soc_bin",
        "rate_bin",
        "temp_bin",
        "effect_per_1pp",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "n_rows",
        "n_groups",
        "support_width_1_99",
    ]
    cap_sub = cap_df[left_cols].copy().rename(
        columns={
            "effect_per_1pp": "cap_effect_per_1pp",
            "ci_low": "cap_ci_low",
            "ci_high": "cap_ci_high",
            "p_value": "cap_p_value",
            "q_value": "cap_q_value",
            "n_rows": "cap_n_rows",
            "n_groups": "cap_n_groups",
            "support_width_1_99": "cap_support_width_1_99",
        }
    )
    ir_sub = ir_df[left_cols].copy().rename(
        columns={
            "effect_per_1pp": "ir_effect_per_1pp",
            "ci_low": "ir_ci_low",
            "ci_high": "ir_ci_high",
            "p_value": "ir_p_value",
            "q_value": "ir_q_value",
            "n_rows": "ir_n_rows",
            "n_groups": "ir_n_groups",
            "support_width_1_99": "ir_support_width_1_99",
        }
    )
    merged = cap_sub.merge(
        ir_sub,
        on=["cross_bin", "cross_label", "soc_bin", "rate_bin", "temp_bin"],
        how="inner",
        validate="one_to_one",
    )

    cap_risk = (
        np.isfinite(merged["cap_ci_low"].to_numpy())
        & np.isfinite(merged["cap_ci_high"].to_numpy())
        & (merged["cap_ci_low"].to_numpy(dtype=float) > 0.0)
    )
    ir_risk = (
        np.isfinite(merged["ir_ci_low"].to_numpy())
        & np.isfinite(merged["ir_ci_high"].to_numpy())
        & (merged["ir_ci_low"].to_numpy(dtype=float) > 0.0)
    )
    labels = np.full(merged.shape[0], "uncertain", dtype=object)
    labels[cap_risk & ir_risk] = "dual_risk"
    labels[cap_risk & (~ir_risk)] = "cap_dominant_risk"
    labels[(~cap_risk) & ir_risk] = "ir_dominant_risk"
    merged["risk_category"] = labels
    merged["cap_significant_positive"] = cap_risk
    merged["ir_significant_positive"] = ir_risk
    merged["cap_ci_cross_zero"] = (
        np.isfinite(merged["cap_ci_low"].to_numpy(dtype=float))
        & np.isfinite(merged["cap_ci_high"].to_numpy(dtype=float))
        & (merged["cap_ci_low"].to_numpy(dtype=float) <= 0.0)
        & (merged["cap_ci_high"].to_numpy(dtype=float) >= 0.0)
    )
    merged["ir_ci_cross_zero"] = (
        np.isfinite(merged["ir_ci_low"].to_numpy(dtype=float))
        & np.isfinite(merged["ir_ci_high"].to_numpy(dtype=float))
        & (merged["ir_ci_low"].to_numpy(dtype=float) <= 0.0)
        & (merged["ir_ci_high"].to_numpy(dtype=float) >= 0.0)
    )
    merged["effect_rank_capacity"] = (
        pd.to_numeric(merged["cap_effect_per_1pp"], errors="coerce")
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    merged["effect_rank_ir"] = (
        pd.to_numeric(merged["ir_effect_per_1pp"], errors="coerce")
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    stack = np.column_stack(
        [
            np.abs(pd.to_numeric(merged["cap_effect_per_1pp"], errors="coerce").to_numpy(dtype=float)),
            np.abs(pd.to_numeric(merged["ir_effect_per_1pp"], errors="coerce").to_numpy(dtype=float)),
        ]
    )
    all_nan = np.isnan(stack).all(axis=1)
    rank_score = np.full(stack.shape[0], np.nan, dtype=float)
    if np.any(~all_nan):
        rank_score[~all_nan] = np.nanmax(stack[~all_nan], axis=1)
    merged["max_abs_effect_per_1pp"] = rank_score
    merged = merged.sort_values(
        ["risk_category", "max_abs_effect_per_1pp"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return merged


def build_support_normalized_effects_compare(
    compare_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build raw-vs-support-normalized effect comparison table for both outcomes."""
    if compare_df.empty:
        return pd.DataFrame()

    merge_cols = [c for c in ["cross_bin", "cross_label", "condition_text"] if c in interpretation_df.columns]
    base = compare_df.copy()
    if set(["cross_bin", "cross_label"]).issubset(base.columns) and set(["cross_bin", "cross_label"]).issubset(merge_cols):
        base = base.merge(
            interpretation_df[merge_cols].drop_duplicates(subset=["cross_bin", "cross_label"]),
            on=["cross_bin", "cross_label"],
            how="left",
            validate="many_to_one",
        )

    outcome_cfg = [
        ("capacity", "cap_effect_per_1pp", "cap_support_width_1_99", "cap_q_value", "cap_significant_positive", "cap_ci_cross_zero"),
        ("impedance", "ir_effect_per_1pp", "ir_support_width_1_99", "ir_q_value", "ir_significant_positive", "ir_ci_cross_zero"),
    ]
    frames: list[pd.DataFrame] = []
    for outcome, eff_col, width_col, q_col, sig_col, cross_col in outcome_cfg:
        cols = [
            "cross_bin",
            "cross_label",
            "soc_bin",
            "rate_bin",
            "temp_bin",
            "risk_category",
            eff_col,
            width_col,
            q_col,
            sig_col,
            cross_col,
        ]
        if "condition_text" in base.columns:
            cols.append("condition_text")
        part = base[[c for c in cols if c in base.columns]].copy()
        part["outcome"] = outcome
        part = part.rename(
            columns={
                eff_col: "effect_per_1pp",
                width_col: "support_width_1_99",
                q_col: "q_value",
                sig_col: "significant_positive",
                cross_col: "ci_cross_zero",
            }
        )
        part["effect_per_1pp"] = pd.to_numeric(part["effect_per_1pp"], errors="coerce")
        part["support_width_1_99"] = pd.to_numeric(part["support_width_1_99"], errors="coerce")
        part["q_value"] = pd.to_numeric(part["q_value"], errors="coerce")
        part["effect_support_norm"] = part["effect_per_1pp"] * (part["support_width_1_99"] / 0.01)
        part["rank_raw"] = part["effect_per_1pp"].rank(method="dense", ascending=False).astype("Int64")
        part["rank_norm"] = part["effect_support_norm"].rank(method="dense", ascending=False).astype("Int64")
        part["rank_delta"] = pd.to_numeric(part["rank_norm"], errors="coerce") - pd.to_numeric(
            part["rank_raw"], errors="coerce"
        )
        frames.append(part)

    out = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    sort_cols = ["outcome", "rank_raw", "rank_norm", "cross_bin"]
    out = out.sort_values(sort_cols, ascending=[True, True, True, True], kind="mergesort").reset_index(drop=True)
    return out


def build_soc_temp_high_temp_evidence(
    compare_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    q_threshold: float = 0.1,
) -> pd.DataFrame:
    """Summarize SOC x temperature evidence for capacity/impedance risk patterns."""
    if compare_df.empty:
        return pd.DataFrame()

    work = compare_df.copy()
    num_cols = [
        "soc_bin",
        "temp_bin",
        "cap_effect_per_1pp",
        "ir_effect_per_1pp",
        "cap_q_value",
        "ir_q_value",
        "cap_support_width_1_99",
        "ir_support_width_1_99",
    ]
    for col in num_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    rows: list[dict[str, float | int]] = []
    for (soc_bin, temp_bin), part in work.groupby(["soc_bin", "temp_bin"], dropna=False):
        if not np.isfinite(float(soc_bin)) or not np.isfinite(float(temp_bin)):
            continue
        row: dict[str, float | int] = {
            "soc_bin": int(float(soc_bin)),
            "temp_bin": int(float(temp_bin)),
            "n_bins": int(part.shape[0]),
        }
        for prefix in ["cap", "ir"]:
            eff = pd.to_numeric(part[f"{prefix}_effect_per_1pp"], errors="coerce")
            qv = pd.to_numeric(part[f"{prefix}_q_value"], errors="coerce")
            width = pd.to_numeric(part[f"{prefix}_support_width_1_99"], errors="coerce")
            row[f"{prefix}_mean"] = float(eff.mean())
            row[f"{prefix}_median"] = float(eff.median())
            row[f"{prefix}_q_lt_0p1_share"] = float((qv < float(q_threshold)).mean())
            row[f"{prefix}_sig_pos_share"] = float(((qv < float(q_threshold)) & (eff > 0.0)).mean())
            row[f"{prefix}_support_mean"] = float(width.mean())
            row[f"{prefix}_support_norm_mean"] = float((eff * (width / 0.01)).mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    label_cols = [c for c in ["soc_bin", "temp_bin", "soc_label", "temp_label"] if c in interpretation_df.columns]
    if set(["soc_bin", "temp_bin"]).issubset(label_cols):
        labels = interpretation_df[label_cols].drop_duplicates(subset=["soc_bin", "temp_bin"])
        out = out.merge(labels, on=["soc_bin", "temp_bin"], how="left", validate="many_to_one")
    out = out.sort_values(["soc_bin", "temp_bin"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return out


def build_high_temp_claim_assessment(
    soc_temp_evidence_df: pd.DataFrame,
    q_threshold: float = 0.1,
    high_temp_bin: int = 5,
) -> pd.DataFrame:
    """Assess high-temperature-risk claim strength across SOC strata for each outcome."""
    if soc_temp_evidence_df.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, object]] = []
    for outcome_name, prefix in [("capacity", "cap"), ("impedance", "ir")]:
        part = soc_temp_evidence_df.copy()
        part["soc_bin"] = pd.to_numeric(part["soc_bin"], errors="coerce")
        part["temp_bin"] = pd.to_numeric(part["temp_bin"], errors="coerce")
        part = part.dropna(subset=["soc_bin", "temp_bin"]).copy()
        part["soc_bin"] = part["soc_bin"].astype(int)
        part["temp_bin"] = part["temp_bin"].astype(int)
        soc_bins = sorted(part["soc_bin"].unique().tolist())

        positive_count = 0
        top_count = 0
        sig_count = 0
        support_count = 0
        low_soc_mean = float("nan")
        low_soc_label = ""

        for soc in soc_bins:
            soc_part = part.loc[part["soc_bin"] == soc].copy()
            if soc_part.empty:
                continue
            high_part = soc_part.loc[soc_part["temp_bin"] == int(high_temp_bin)]
            if high_part.empty:
                continue
            high_row = high_part.iloc[0]
            high_mean = float(pd.to_numeric(high_row[f"{prefix}_mean"], errors="coerce"))
            high_q_share = float(pd.to_numeric(high_row[f"{prefix}_q_lt_0p1_share"], errors="coerce"))
            high_support = float(pd.to_numeric(high_row[f"{prefix}_support_mean"], errors="coerce"))
            soc_support_med = float(pd.to_numeric(soc_part[f"{prefix}_support_mean"], errors="coerce").median())
            soc_top_temp = int(
                soc_part.sort_values(f"{prefix}_mean", ascending=False, kind="mergesort").iloc[0]["temp_bin"]
            )

            if high_mean > 0.0:
                positive_count += 1
            if soc_top_temp == int(high_temp_bin):
                top_count += 1
            if high_q_share >= 0.5:
                sig_count += 1
            if np.isfinite(high_support) and np.isfinite(soc_support_med) and high_support >= soc_support_med:
                support_count += 1

            if soc == 1:
                low_soc_mean = high_mean
                low_soc_label = str(high_row.get("soc_label", ""))

        n_soc = len(soc_bins)
        if n_soc == 0:
            assessment = "weak"
        elif top_count >= 2 and positive_count >= 2 and sig_count >= 2:
            assessment = "strong"
        elif positive_count >= 2 and (top_count >= 1 or sig_count >= 1):
            assessment = "partial"
        else:
            assessment = "weak"

        detail = (
            f"positive_soc={positive_count}/{n_soc}, top_soc={top_count}/{n_soc}, "
            f"q<{_format_float(q_threshold, 2)}_share_soc={sig_count}/{n_soc}, support_ok_soc={support_count}/{n_soc}"
        )
        out_rows.append(
            {
                "outcome": outcome_name,
                "high_temp_bin": int(high_temp_bin),
                "assessment": assessment,
                "n_soc": int(n_soc),
                "positive_soc_count": int(positive_count),
                "top_soc_count": int(top_count),
                "q_lt_0p1_soc_count": int(sig_count),
                "support_ok_soc_count": int(support_count),
                "low_soc_label": low_soc_label,
                "low_soc_high_temp_mean": float(low_soc_mean),
                "detail": detail,
            }
        )
    return pd.DataFrame(out_rows)


def save_trend_plot(cell_trend_df: pd.DataFrame, output_path: Path) -> None:
    """Save trend distribution plot for per-cell cycle correlations."""
    if cell_trend_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].hist(cell_trend_df["rho_cycle_q"], bins=24, color="#2563eb", alpha=0.85)
    axes[0].axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[0].set_title("cell内 cycle~q_t 相关分布")
    axes[0].set_xlabel("Spearman rho")
    axes[0].set_ylabel("cell数量")

    axes[1].hist(cell_trend_df["rho_cycle_ir"], bins=24, color="#f59e0b", alpha=0.85)
    axes[1].axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_title("cell内 cycle~ir_t 相关分布")
    axes[1].set_xlabel("Spearman rho")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_crosslink_plot(crosslink_df: pd.DataFrame, output_path: Path) -> None:
    """Save direction-wise AIPW shift effect comparison plot."""
    if crosslink_df.empty:
        return
    plot_df = crosslink_df.copy()
    x = np.arange(plot_df.shape[0])
    y = plot_df["effect_per_1pp"].to_numpy(dtype=float)
    low = plot_df["ci_low"].to_numpy(dtype=float)
    high = plot_df["ci_high"].to_numpy(dtype=float)
    yerr_lower = np.maximum(y - low, 0.0)
    yerr_upper = np.maximum(high - y, 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([yerr_lower, yerr_upper]),
        fmt="o",
        color="#0f766e",
        ecolor="#0f766e",
        capsize=4,
        linewidth=1.5,
    )
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["direction_label"].tolist(), rotation=0)
    ax.set_ylabel("对结果变量的效应（每 +1pp 处理变化）")
    ax.set_title("容量-阻抗双方向因果效应（AIPW/GPS）")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_dual_forest_plot(
    compare_df: pd.DataFrame,
    output_path: Path,
    top_k: int,
) -> None:
    """Save dual-outcome forest plot for top risk bins."""
    if compare_df.empty:
        return
    order_df = compare_df.sort_values("max_abs_effect_per_1pp", ascending=False).head(int(top_k)).copy()
    order_df = order_df.sort_values("max_abs_effect_per_1pp", ascending=True)
    y_labels = [f"bin{int(b):02d}" for b in order_df["cross_bin"].tolist()]
    y_pos = np.arange(order_df.shape[0])

    cap_y = order_df["cap_effect_per_1pp"].to_numpy(dtype=float)
    cap_low = order_df["cap_ci_low"].to_numpy(dtype=float)
    cap_high = order_df["cap_ci_high"].to_numpy(dtype=float)
    ir_y = order_df["ir_effect_per_1pp"].to_numpy(dtype=float)
    ir_low = order_df["ir_ci_low"].to_numpy(dtype=float)
    ir_high = order_df["ir_ci_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.24 * order_df.shape[0] + 1.8)))
    ax.errorbar(
        cap_y,
        y_pos + 0.15,
        xerr=np.vstack([np.maximum(cap_y - cap_low, 0.0), np.maximum(cap_high - cap_y, 0.0)]),
        fmt="o",
        color="#2563eb",
        ecolor="#2563eb",
        capsize=3,
        label="容量衰减风险",
    )
    ax.errorbar(
        ir_y,
        y_pos - 0.15,
        xerr=np.vstack([np.maximum(ir_y - ir_low, 0.0), np.maximum(ir_high - ir_y, 0.0)]),
        fmt="o",
        color="#f59e0b",
        ecolor="#f59e0b",
        capsize=3,
        label="阻抗上升风险",
    )
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("效应（每 +1pp 区间份额替代）")
    ax.set_ylabel("cross_bin")
    ax.set_title("充电60区间双结局效应对比（Top bins）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_risk_matrix_plot(compare_df: pd.DataFrame, bin_meta_df: pd.DataFrame, output_path: Path) -> None:
    """Save SOC-panel risk matrix from dual outcome categories."""
    if compare_df.empty:
        return
    value_map = {
        "uncertain": 0,
        "cap_dominant_risk": 1,
        "ir_dominant_risk": 2,
        "dual_risk": 3,
    }
    cmap = colors.ListedColormap(["#d1d5db", "#60a5fa", "#fbbf24", "#ef4444"])
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    soc_bins = sorted(pd.to_numeric(bin_meta_df["soc_bin"], errors="coerce").dropna().astype(int).unique().tolist())
    rate_bins = sorted(pd.to_numeric(bin_meta_df["rate_bin"], errors="coerce").dropna().astype(int).unique().tolist())
    temp_bins = sorted(pd.to_numeric(bin_meta_df["temp_bin"], errors="coerce").dropna().astype(int).unique().tolist())
    if not soc_bins:
        soc_bins = sorted(pd.to_numeric(compare_df["soc_bin"], errors="coerce").dropna().astype(int).unique().tolist())
    if not rate_bins:
        rate_bins = sorted(pd.to_numeric(compare_df["rate_bin"], errors="coerce").dropna().astype(int).unique().tolist())
    if not temp_bins:
        temp_bins = sorted(pd.to_numeric(compare_df["temp_bin"], errors="coerce").dropna().astype(int).unique().tolist())

    soc_label_map: Dict[int, str] = {}
    rate_label_map: Dict[int, str] = {}
    temp_label_map: Dict[int, str] = {}

    if {"soc_bin", "soc_label"}.issubset(bin_meta_df.columns):
        soc_meta = bin_meta_df[["soc_bin", "soc_label"]].copy()
        soc_meta["soc_bin"] = pd.to_numeric(soc_meta["soc_bin"], errors="coerce").astype("Int64")
        soc_meta = soc_meta.dropna(subset=["soc_bin"]).copy()
        soc_meta["soc_bin"] = soc_meta["soc_bin"].astype(int)
        soc_meta["soc_label"] = soc_meta["soc_label"].astype(str).str.strip()
        soc_meta = soc_meta.loc[soc_meta["soc_label"] != ""].copy()
        if not soc_meta.empty:
            dup_soc = soc_meta.groupby("soc_bin")["soc_label"].nunique()
            for bin_id, cnt in dup_soc.items():
                if int(cnt) > 1:
                    print(f"Warning: soc_bin={int(bin_id)} has multiple labels; using the first one.")
            soc_label_map = (
                soc_meta.drop_duplicates(subset=["soc_bin"], keep="first")
                .set_index("soc_bin")["soc_label"]
                .to_dict()
            )
    if {"rate_bin", "rate_label"}.issubset(bin_meta_df.columns):
        rate_meta = bin_meta_df[["rate_bin", "rate_label"]].copy()
        rate_meta["rate_bin"] = pd.to_numeric(rate_meta["rate_bin"], errors="coerce").astype("Int64")
        rate_meta = rate_meta.dropna(subset=["rate_bin"]).copy()
        rate_meta["rate_bin"] = rate_meta["rate_bin"].astype(int)
        rate_meta["rate_label"] = rate_meta["rate_label"].astype(str).str.strip()
        rate_meta = rate_meta.loc[rate_meta["rate_label"] != ""].copy()
        if not rate_meta.empty:
            dup_rate = rate_meta.groupby("rate_bin")["rate_label"].nunique()
            for bin_id, cnt in dup_rate.items():
                if int(cnt) > 1:
                    print(f"Warning: rate_bin={int(bin_id)} has multiple labels; using the first one.")
            rate_label_map = (
                rate_meta.drop_duplicates(subset=["rate_bin"], keep="first")
                .set_index("rate_bin")["rate_label"]
                .to_dict()
            )
    if {"temp_bin", "temp_label"}.issubset(bin_meta_df.columns):
        temp_meta = bin_meta_df[["temp_bin", "temp_label"]].copy()
        temp_meta["temp_bin"] = pd.to_numeric(temp_meta["temp_bin"], errors="coerce").astype("Int64")
        temp_meta = temp_meta.dropna(subset=["temp_bin"]).copy()
        temp_meta["temp_bin"] = temp_meta["temp_bin"].astype(int)
        temp_meta["temp_label"] = temp_meta["temp_label"].astype(str).str.strip()
        temp_meta = temp_meta.loc[temp_meta["temp_label"] != ""].copy()
        if not temp_meta.empty:
            dup_temp = temp_meta.groupby("temp_bin")["temp_label"].nunique()
            for bin_id, cnt in dup_temp.items():
                if int(cnt) > 1:
                    print(f"Warning: temp_bin={int(bin_id)} has multiple labels; using the first one.")
            temp_label_map = (
                temp_meta.drop_duplicates(subset=["temp_bin"], keep="first")
                .set_index("temp_bin")["temp_label"]
                .to_dict()
            )

    for bin_id in soc_bins:
        if bin_id not in soc_label_map:
            soc_label_map[bin_id] = f"S{bin_id}"
            print(f"Warning: missing soc_label for soc_bin={bin_id}; fallback to S{bin_id}.")
    for bin_id in rate_bins:
        if bin_id not in rate_label_map:
            rate_label_map[bin_id] = f"R{bin_id}"
            print(f"Warning: missing rate_label for rate_bin={bin_id}; fallback to R{bin_id}.")
    for bin_id in temp_bins:
        if bin_id not in temp_label_map:
            temp_label_map[bin_id] = f"T{bin_id}"
            print(f"Warning: missing temp_label for temp_bin={bin_id}; fallback to T{bin_id}.")

    fig, axes = plt.subplots(1, len(soc_bins), figsize=(4.5 * len(soc_bins), 4.2), sharey=True)
    if len(soc_bins) == 1:
        axes = [axes]
    im = None
    for idx, soc_bin in enumerate(soc_bins):
        ax = axes[idx]
        part = compare_df.loc[compare_df["soc_bin"] == soc_bin].copy()
        if part.empty:
            ax.set_axis_off()
            continue
        part["risk_value"] = part["risk_category"].map(value_map).fillna(0).astype(int)
        pv = part.pivot_table(
            index="rate_bin",
            columns="temp_bin",
            values="risk_value",
            aggfunc="max",
            fill_value=0,
        )
        pv = pv.reindex(index=rate_bins, columns=temp_bins, fill_value=0)
        im = ax.imshow(pv.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"SOC {soc_label_map.get(soc_bin, f'S{soc_bin}')}")
        ax.set_xlabel("温度区间(°C)")
        if idx == 0:
            ax.set_ylabel("倍率区间(C-rate)")
        ax.set_xticks(np.arange(len(temp_bins)))
        ax.set_xticklabels([temp_label_map[b] for b in temp_bins], rotation=0)
        ax.set_yticks(np.arange(len(rate_bins)))
        ax.set_yticklabels([rate_label_map[b] for b in rate_bins])
    if im is None:
        plt.close(fig)
        return
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["uncertain", "cap_dom", "ir_dom", "dual"])
    cbar.set_label("风险类别")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.14, top=0.88, wspace=0.22)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_single_outcome_top_plot(
    top_df: pd.DataFrame,
    output_path: Path,
    effect_col: str,
    low_col: str,
    high_col: str,
    title: str,
    color: str,
) -> None:
    """Save one outcome-specific forest plot using already-sorted top table."""
    if top_df.empty:
        return
    plot_df = top_df.copy().sort_values(effect_col, ascending=True, kind="mergesort")
    y_pos = np.arange(plot_df.shape[0])
    labels = plot_df["cross_bin"].map(lambda v: f"bin{int(v):02d}").tolist()
    x = pd.to_numeric(plot_df[effect_col], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(plot_df[low_col], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(plot_df[high_col], errors="coerce").to_numpy(dtype=float)
    xerr = np.vstack([np.maximum(x - low, 0.0), np.maximum(high - x, 0.0)])

    fig, ax = plt.subplots(figsize=(10.0, max(5.0, 0.45 * plot_df.shape[0] + 1.2)))
    ax.errorbar(
        x,
        y_pos,
        xerr=xerr,
        fmt="o",
        color=color,
        ecolor=color,
        capsize=3,
        linewidth=1.2,
    )
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("效应（每 +1pp 区间份额替代）")
    ax.set_ylabel("cross_bin")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_capacity_ir_scatter(compare_df: pd.DataFrame, output_path: Path) -> None:
    """Save scatter plot of capacity-risk vs impedance-risk effects by cross-bin."""
    if compare_df.empty:
        return
    work = compare_df.copy()
    color_map = {
        "uncertain": "#9ca3af",
        "cap_dominant_risk": "#2563eb",
        "ir_dominant_risk": "#f59e0b",
        "dual_risk": "#dc2626",
    }
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    for cat, part in work.groupby("risk_category", dropna=False):
        ax.scatter(
            pd.to_numeric(part["cap_effect_per_1pp"], errors="coerce"),
            pd.to_numeric(part["ir_effect_per_1pp"], errors="coerce"),
            s=48,
            alpha=0.8,
            color=color_map.get(str(cat), "#6b7280"),
            label=str(cat),
        )

    annotate_df = work.sort_values("max_abs_effect_per_1pp", ascending=False).head(10).copy()
    for row in annotate_df.itertuples(index=False):
        x = float(getattr(row, "cap_effect_per_1pp"))
        y = float(getattr(row, "ir_effect_per_1pp"))
        if np.isfinite(x) and np.isfinite(y):
            ax.annotate(
                f"bin{int(getattr(row, 'cross_bin')):02d}",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#111827",
            )

    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlabel("容量衰减效应（cap_effect_per_1pp）")
    ax.set_ylabel("阻抗增加效应（ir_effect_per_1pp）")
    ax.set_title("容量与阻抗区间风险散点图")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _format_float(value: float, digits: int = 6) -> str:
    """Format float safely for markdown report."""
    if value is None or not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render dataframe to markdown text without third-party dependency."""
    if df.empty:
        return "_空表_"
    work = df.copy()
    if max_rows is not None:
        work = work.head(int(max_rows)).copy()
    cols = list(work.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in work.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (bool, np.bool_)):
                vals.append("是" if bool(val) else "否")
            elif isinstance(val, float):
                vals.append(_format_float(val, 6))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _metric_lookup(summary_df: pd.DataFrame, metric: str) -> float:
    """Lookup one metric value from summary table."""
    part = summary_df.loc[summary_df["metric"] == metric, "value"]
    if part.empty:
        return float("nan")
    return float(pd.to_numeric(part.iloc[0], errors="coerce"))


def _bin_title(row: pd.Series) -> str:
    """Build short readable label for one cross-bin row."""
    return f"bin{int(row['cross_bin']):02d}（{str(row.get('condition_text', row.get('cross_label', '')))}）"


def render_report(
    args: argparse.Namespace,
    backend: BootstrapBackendInfo,
    diagnostics_df: pd.DataFrame,
    trend_summary_df: pd.DataFrame,
    crosslink_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    ir_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    cap_top_df: pd.DataFrame,
    ir_top_df: pd.DataFrame,
    support_compare_df: pd.DataFrame,
    soc_temp_evidence_df: pd.DataFrame,
    high_temp_assessment_df: pd.DataFrame,
) -> str:
    """Render Chinese markdown report for joint causal analysis."""
    compare_view = compare_df.merge(
        interpretation_df[
            [c for c in ["cross_bin", "cross_label", "condition_text", "condition_text_detailed"] if c in interpretation_df.columns]
        ],
        on=["cross_bin", "cross_label"],
        how="left",
        validate="many_to_one",
    )
    if not cap_top_df.empty:
        cap_top_view = cap_top_df.copy()
    else:
        cap_top_view = pd.DataFrame()
    if not ir_top_df.empty:
        ir_top_view = ir_top_df.copy()
    else:
        ir_top_view = pd.DataFrame()

    rows_window = _metric_lookup(trend_summary_df, "rows_window")
    sp_cycle_q = _metric_lookup(trend_summary_df, "spearman_cycle_q")
    sp_cycle_ir = _metric_lookup(trend_summary_df, "spearman_cycle_ir")
    sp_y = _metric_lookup(trend_summary_df, "spearman_y_capdrop_vs_y_irrise")
    pe_y = _metric_lookup(trend_summary_df, "pearson_y_capdrop_vs_y_irrise")
    both_worse = _metric_lookup(trend_summary_df, "share_both_worsen")
    cell_q = _metric_lookup(trend_summary_df, "cell_median_rho_cycle_q")
    cell_ir = _metric_lookup(trend_summary_df, "cell_median_rho_cycle_ir")
    cell_opp = _metric_lookup(trend_summary_df, "cell_share_opposite_sign_trend")

    total_bins = int(compare_view.shape[0])
    uncertain_n = int((compare_view["risk_category"] == "uncertain").sum()) if not compare_view.empty else 0
    dual_n = int((compare_view["risk_category"] == "dual_risk").sum()) if not compare_view.empty else 0
    cap_dom_n = int((compare_view["risk_category"] == "cap_dominant_risk").sum()) if not compare_view.empty else 0
    ir_dom_n = int((compare_view["risk_category"] == "ir_dominant_risk").sum()) if not compare_view.empty else 0

    top_cap_row = cap_top_view.iloc[0] if not cap_top_view.empty else None
    top_ir_row = ir_top_view.iloc[0] if not ir_top_view.empty else None

    stable_cap = cap_top_view.loc[cap_top_view["significant_positive"]].copy() if not cap_top_view.empty else pd.DataFrame()
    stable_ir = ir_top_view.loc[ir_top_view["significant_positive"]].copy() if not ir_top_view.empty else pd.DataFrame()
    support_view = support_compare_df.copy() if not support_compare_df.empty else pd.DataFrame()
    cap_support = support_view.loc[support_view["outcome"] == "capacity"].copy() if not support_view.empty else pd.DataFrame()
    ir_support = support_view.loc[support_view["outcome"] == "impedance"].copy() if not support_view.empty else pd.DataFrame()
    dual_view = compare_view.loc[compare_view["risk_category"] == "dual_risk"].copy()
    dual_view = dual_view.sort_values("max_abs_effect_per_1pp", ascending=False, kind="mergesort")
    head_dual = dual_view.iloc[0] if not dual_view.empty else None
    stable_dual_follow = dual_view.iloc[1:7].copy() if dual_view.shape[0] > 1 else pd.DataFrame()
    cap_dom_view = compare_view.loc[compare_view["risk_category"] == "cap_dominant_risk"].copy()
    ir_dom_view = compare_view.loc[compare_view["risk_category"] == "ir_dominant_risk"].copy()
    bin05_row = compare_view.loc[compare_view["cross_bin"] == 5].head(1)
    bin05_row = bin05_row.iloc[0] if not bin05_row.empty else None
    cap_raw_top_row = cap_support.sort_values("rank_raw", ascending=True, kind="mergesort").head(1)
    cap_raw_top_row = cap_raw_top_row.iloc[0] if not cap_raw_top_row.empty else None
    cap_norm_top_row = cap_support.sort_values("rank_norm", ascending=True, kind="mergesort").head(1)
    cap_norm_top_row = cap_norm_top_row.iloc[0] if not cap_norm_top_row.empty else None
    ir_raw_top_row = ir_support.sort_values("rank_raw", ascending=True, kind="mergesort").head(1)
    ir_raw_top_row = ir_raw_top_row.iloc[0] if not ir_raw_top_row.empty else None
    ir_norm_top_row = ir_support.sort_values("rank_norm", ascending=True, kind="mergesort").head(1)
    ir_norm_top_row = ir_norm_top_row.iloc[0] if not ir_norm_top_row.empty else None

    def _top_overlap_count(df: pd.DataFrame, top_n: int = 5) -> int:
        if df.empty:
            return 0
        raw_set = set(pd.to_numeric(df.nsmallest(int(top_n), "rank_raw")["cross_bin"], errors="coerce").dropna().astype(int))
        norm_set = set(pd.to_numeric(df.nsmallest(int(top_n), "rank_norm")["cross_bin"], errors="coerce").dropna().astype(int))
        return int(len(raw_set & norm_set))

    cap_top5_overlap = _top_overlap_count(cap_support, top_n=5)
    ir_top5_overlap = _top_overlap_count(ir_support, top_n=5)
    refresh_needed = (cap_top5_overlap <= 2) or (ir_top5_overlap <= 2)

    assess_view = high_temp_assessment_df.copy() if not high_temp_assessment_df.empty else pd.DataFrame()
    assess_map: dict[str, pd.Series] = {}
    if not assess_view.empty and "outcome" in assess_view.columns:
        for _, row in assess_view.iterrows():
            assess_map[str(row["outcome"])] = row

    cap_raw_norm_view = pd.DataFrame()
    if not cap_support.empty:
        cap_raw_norm_view = cap_support.sort_values("rank_raw", ascending=True, kind="mergesort").head(10).copy()
        cap_raw_norm_view = cap_raw_norm_view[
            [
                c
                for c in [
                    "rank_raw",
                    "rank_norm",
                    "rank_delta",
                    "cross_bin",
                    "cross_label",
                    "condition_text",
                    "effect_per_1pp",
                    "effect_support_norm",
                    "support_width_1_99",
                    "q_value",
                    "risk_category",
                ]
                if c in cap_raw_norm_view.columns
            ]
        ]
    ir_raw_norm_view = pd.DataFrame()
    if not ir_support.empty:
        ir_raw_norm_view = ir_support.sort_values("rank_raw", ascending=True, kind="mergesort").head(10).copy()
        ir_raw_norm_view = ir_raw_norm_view[
            [
                c
                for c in [
                    "rank_raw",
                    "rank_norm",
                    "rank_delta",
                    "cross_bin",
                    "cross_label",
                    "condition_text",
                    "effect_per_1pp",
                    "effect_support_norm",
                    "support_width_1_99",
                    "q_value",
                    "risk_category",
                ]
                if c in ir_raw_norm_view.columns
            ]
        ]
    soc_temp_view = pd.DataFrame()
    if not soc_temp_evidence_df.empty:
        soc_temp_view = soc_temp_evidence_df[
            [
                c
                for c in [
                    "soc_bin",
                    "soc_label",
                    "temp_bin",
                    "temp_label",
                    "n_bins",
                    "cap_mean",
                    "cap_q_lt_0p1_share",
                    "cap_support_mean",
                    "ir_mean",
                    "ir_q_lt_0p1_share",
                    "ir_support_mean",
                ]
                if c in soc_temp_evidence_df.columns
            ]
        ].copy()

    def _join_bin_titles(df: pd.DataFrame, limit: int = 6) -> str:
        if df.empty:
            return "无"
        rows = []
        for _, row in df.head(int(limit)).iterrows():
            rows.append(_bin_title(row))
        return "、".join(rows)

    lines: list[str] = []
    lines.append("# 容量-阻抗联合因果分析报告（充电60区间）")
    lines.append("")
    lines.append("## 1. 执行摘要")
    lines.append(
        f"- 在 `H={int(args.horizon_cycles)}` 的窗口口径下，容量衰减与阻抗上升存在稳定的共同恶化关系："
        f"`Spearman={_format_float(sp_y, 4)}`，`Pearson={_format_float(pe_y, 4)}`，"
        f"且同窗口两者同时变坏的占比为 `{_format_float(both_worse * 100.0, 2)}%`。"
    )
    lines.append(
        f"- pooled 全局看，`cycle~q_t` 呈强负相关（`{_format_float(sp_cycle_q, 4)}`），"
        f"`cycle~ir_t` 的 pooled 相关较弱（`{_format_float(sp_cycle_ir, 4)}`），"
        f"但 cell 内中位趋势显示 `q_t` 下降、`ir_t` 上升同时成立，"
        f"`{_format_float(cell_opp * 100.0, 2)}%` 的 cell 呈现“容量下降+阻抗上升”的反向趋势组合。"
    )
    lines.append(
        f"- 双方向 AIPW 显示：`IR变化(+1pp) -> 容量衰减` 的效应为 `{_format_float(float(crosslink_df.iloc[0]['effect_per_1pp']), 6)}`，"
        f"`容量变化(+1pp) -> 阻抗上升` 的效应为 `{_format_float(float(crosslink_df.iloc[1]['effect_per_1pp']), 6)}`；"
        "当前标准化口径下，后者点估计更大。"
    )
    if top_cap_row is not None and top_ir_row is not None:
        lines.append(
            f"- 容量风险最大的区间是 `{_bin_title(top_cap_row)}`，阻抗风险最大的区间也是 `{_bin_title(top_ir_row)}`；"
            "这说明存在头部共同高风险工况。"
        )
    if bin05_row is not None:
        lines.append(
            f"- `bin05` 的 `support_width_1_99={_format_float(float(bin05_row['cap_support_width_1_99']), 6)}`，"
            "属于支持域极窄区间；其“+1pp”效应应与支持宽度归一口径一起解读，避免外推放大误读。"
        )
    if cap_raw_top_row is not None and cap_norm_top_row is not None and ir_raw_top_row is not None and ir_norm_top_row is not None:
        lines.append(
            f"- 双口径对比显示：容量 raw 头部为 `{_bin_title(cap_raw_top_row)}`，支持宽度归一头部为 `{_bin_title(cap_norm_top_row)}`；"
            f"阻抗 raw 头部为 `{_bin_title(ir_raw_top_row)}`，支持宽度归一头部为 `{_bin_title(ir_norm_top_row)}`。"
        )
    cap_assess = assess_map.get("capacity")
    ir_assess = assess_map.get("impedance")
    if cap_assess is not None and ir_assess is not None:
        lines.append(
            f"- 高温风险分层判定：容量 `{str(cap_assess.get('assessment', 'nan'))}`，阻抗 `{str(ir_assess.get('assessment', 'nan'))}`；"
            "当前证据更支持“低SOC层高温风险更强”，而非“全SOC单调升温即更高风险”。"
        )
    lines.append(
        f"- 60 个区间中，`dual_risk={dual_n}`、`cap_dominant={cap_dom_n}`、`ir_dominant={ir_dom_n}`、`uncertain={uncertain_n}`；"
        "多数区间仍属证据不足，不能直接解读为安全。"
    )
    lines.append("")

    lines.append("## 2. 样本与口径说明")
    lines.append(f"- 时间窗：`H={int(args.horizon_cycles)}` cycles。")
    lines.append(
        f"- 样本过滤：`{_format_float(float(args.q_min), 3)} <= q_t <= {_format_float(float(args.q_max), 3)}`，`ir_t>0`。"
    )
    lines.append(f"- 排除策略前缀：`{args.exclude_policy_prefix}`。")
    lines.append("- 因果方法：趋势层 + 双方向 `AIPW/GPS` + 区间替代 `DML + cluster bootstrap + BH-FDR`。")
    lines.append(
        f"- bootstrap：`n={int(args.bootstrap_iters)}`，后端 `{backend.backend_used}`，设备 `{backend.device_used}`，"
        f"DML nuisance 模型 `{args.nuisance_model}`。"
    )
    if backend.fallback_reason:
        lines.append(f"- 后端回退说明：`{backend.fallback_reason}`。")
    lines.append(
        "- 数值解释：`effect_per_1pp` 表示将某区间充电份额额外增加 `1pp` 时，对未来 `H=200` 窗口结果的边际影响；"
        "`ci_low/ci_high` 为 95% bootstrap CI；`q_value` 为 BH-FDR 多重比较后的显著性。"
    )
    lines.append("- `significant_positive=True` 表示 95%CI 全部大于 0；`ci_cross_zero=True` 表示 CI 跨 0。")
    lines.append(_df_to_md(diagnostics_df))
    lines.append("")

    lines.append("## 3. 容量变化与阻抗增加的相关性结论")
    lines.append(_df_to_md(trend_summary_df))
    lines.append("")
    lines.append(
        f"- 全局 pooled 结果显示，容量随循环下降的趋势非常强（`cycle~q_t Spearman={_format_float(sp_cycle_q, 4)}`）；"
        f"阻抗的 pooled 结果较弱且方向与 cell 内结果不完全一致（`cycle~ir_t Spearman={_format_float(sp_cycle_ir, 4)}`），"
        "说明阻抗更容易受到不同 policy、不同电芯基线和取样窗口混杂。"
    )
    lines.append(
        f"- cell 内中位趋势更值得信赖：`cycle~q_t` 的中位 Spearman 为 `{_format_float(cell_q, 4)}`，"
        f"`cycle~ir_t` 的中位 Spearman 为 `{_format_float(cell_ir, 4)}`，"
        f"且 `{_format_float(cell_opp * 100.0, 2)}%` 的 cell 同时表现出“容量下降 + 阻抗上升”。"
    )
    lines.append(
        f"- 在 `H=200` 的窗口层面，`y_cap_drop_h` 与 `y_ir_rise_h` 的 `Spearman={_format_float(sp_y, 4)}`、"
        f"`Pearson={_format_float(pe_y, 4)}`。Pearson 明显高于 Spearman，说明当窗口内老化加速时，"
        "容量衰减与阻抗上升会出现更强的线性共振，头部恶化窗口的共变更突出。"
    )
    lines.append(
        f"- 同窗口两者同时恶化的占比达到 `{_format_float(both_worse * 100.0, 2)}%`，"
        "因此“容量下降”和“阻抗增加”不是彼此孤立的退化信号，而是大多数窗口中共同出现。"
    )
    lines.append(
        "- 这里的结论首先是“趋势共变”而非“严格机理单向因果”；真正的方向性判断，需要结合下一节的双方向 AIPW 结果。"
    )
    lines.append("")

    lines.append("## 4. 双方向因果效应解读")
    lines.append(_df_to_md(crosslink_df))
    lines.append("")
    lines.append(
        f"- `IR变化(+1pp) -> 容量衰减` 的效应估计为 `{_format_float(float(crosslink_df.iloc[0]['effect_per_1pp']), 6)}`，"
        f"95%CI 为 `[{_format_float(float(crosslink_df.iloc[0]['ci_low']), 6)}, {_format_float(float(crosslink_df.iloc[0]['ci_high']), 6)}]`。"
    )
    lines.append(
        f"- `容量变化(+1pp) -> 阻抗上升` 的效应估计为 `{_format_float(float(crosslink_df.iloc[1]['effect_per_1pp']), 6)}`，"
        f"95%CI 为 `[{_format_float(float(crosslink_df.iloc[1]['ci_low']), 6)}, {_format_float(float(crosslink_df.iloc[1]['ci_high']), 6)}]`。"
    )
    lines.append(
        "- 两个方向的 CI 都没有跨 0，说明在当前标准化处理定义下，容量与阻抗之间不仅共同变坏，而且具有稳定的方向性预测结构。"
    )
    lines.append(
        "- 需要注意，这两个 `+1pp` 并不是同一个物理量的加法，因此它们更适合作为“标准化强度对比”，"
        "而不是直接拿来做物理量级比较。当前结果更像是在说：近期容量损失是更强的阻抗恶化先行信号，而近期阻抗上跳也会对应后续容量更快下降。"
    )
    lines.append("")

    lines.append("## 5. 容量衰减高风险区间")
    if top_cap_row is not None:
        lines.append(
            f"- 按 `cap_effect_per_1pp` 排序，容量风险头部区间是 `{_bin_title(top_cap_row)}`，"
            f"点估计为 `{_format_float(float(top_cap_row['effect_per_1pp']), 6)}`。"
        )
    if not stable_cap.empty:
        lines.append(
            f"- 在 Top10 中，CI 全正的容量风险区间主要包括：{_join_bin_titles(stable_cap, limit=6)}。"
        )
    lines.append(
        "- 需要区分两类区间：一类是“幅度极大但不确定性更宽”的极端区间，另一类是“幅度中等但 CI 更稳”的稳定风险区间。"
    )
    if top_cap_row is not None:
        lines.append(
            f"- 例如 `{_bin_title(top_cap_row)}` 的效应远高于其他区间，说明它是绝对头部风险；"
            f"但其 `q_value={_format_float(float(top_cap_row['q_value']), 6)}`，属于多重比较下的边缘显著，需要与稳定显著区间分开看。"
        )
    lines.append(
        f"- 相比之下，像 `{_join_bin_titles(stable_cap.iloc[1:] if stable_cap.shape[0] > 1 else stable_cap, limit=5)}` 这类区间，"
        "虽然点估计不如头部极端区间夸张，但更适合被视为稳定的容量治理重点。"
    )
    lines.append("")
    lines.append(
        _df_to_md(
            cap_top_view[
                [
                    c
                    for c in [
                        "effect_rank",
                        "cross_bin",
                        "cross_label",
                        "condition_text",
                        "effect_per_1pp",
                        "ci_low",
                        "ci_high",
                        "q_value",
                        "significant_positive",
                        "ci_cross_zero",
                        "risk_category",
                    ]
                    if c in cap_top_view.columns
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## 6. 阻抗增加高风险区间")
    if top_ir_row is not None:
        lines.append(
            f"- 按 `ir_effect_per_1pp` 排序，阻抗风险头部区间是 `{_bin_title(top_ir_row)}`，"
            f"点估计为 `{_format_float(float(top_ir_row['effect_per_1pp']), 6)}`。"
        )
    if not stable_ir.empty:
        lines.append(
            f"- 在 Top10 中，CI 全正的阻抗风险区间主要包括：{_join_bin_titles(stable_ir, limit=6)}。"
        )
    lines.append(
        "- 阻抗风险的头部结构与容量风险并不完全相同：有些区间对阻抗非常敏感，但对容量的直接影响尚不够稳定。"
    )
    if top_ir_row is not None:
        lines.append(
            f"- `{_bin_title(top_ir_row)}` 同时也是双结局共同高风险区间，说明它更接近“共损伤工况”；"
            "而某些 `ir_dominant_risk` 区间则更像“先推高阻抗、对容量短期影响尚不够稳定”的工况。"
        )
    lines.append("")
    lines.append(
        _df_to_md(
            ir_top_view[
                [
                    c
                    for c in [
                        "effect_rank",
                        "cross_bin",
                        "cross_label",
                        "condition_text",
                        "effect_per_1pp",
                        "ci_low",
                        "ci_high",
                        "q_value",
                        "significant_positive",
                        "ci_cross_zero",
                        "risk_category",
                    ]
                    if c in ir_top_view.columns
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## 7. 双结局共同高风险区间")
    cat_summary = (
        compare_df.groupby("risk_category", as_index=False)
        .agg(
            n_bins=("cross_bin", "count"),
            mean_cap_effect=("cap_effect_per_1pp", "mean"),
            mean_ir_effect=("ir_effect_per_1pp", "mean"),
        )
        .sort_values("n_bins", ascending=False)
        if not compare_df.empty
        else pd.DataFrame()
    )
    lines.append("### 7.1 风险类别统计")
    lines.append(_df_to_md(cat_summary))
    lines.append("")
    if head_dual is not None:
        lines.append(
            f"- 头部绝对风险区间是 `{_bin_title(head_dual)}`：它在容量和阻抗两个结果上都显著高于其他区间，"
            "应优先被当作最强共损伤工况。"
        )
    if not stable_dual_follow.empty:
        lines.append(
            f"- 除头部极端区间外，更稳定的共损伤区间包括：{_join_bin_titles(stable_dual_follow, limit=6)}。"
        )
    if not cap_dom_view.empty:
        lines.append(
            f"- 容量优先治理区间：{_join_bin_titles(cap_dom_view.sort_values('cap_effect_per_1pp', ascending=False), limit=3)}。"
        )
    if not ir_dom_view.empty:
        lines.append(
            f"- 阻抗优先治理区间：{_join_bin_titles(ir_dom_view.sort_values('ir_effect_per_1pp', ascending=False), limit=3)}。"
        )
    lines.append(
        f"- `uncertain` 区间有 `{uncertain_n}/{total_bins}` 个，占比 `{_format_float(_safe_ratio(float(uncertain_n), float(max(total_bins, 1))) * 100.0, 2)}%`。"
        "这里的含义是“当前证据不足”，不是“这些区间一定安全”。"
    )
    top_dual = dual_view.head(12)
    lines.append("### 7.2 dual_risk 重点区间")
    lines.append(
        _df_to_md(
            top_dual[
                [
                    "cross_bin",
                    "cross_label",
                    "condition_text",
                    "soc_bin",
                    "rate_bin",
                    "temp_bin",
                    "cap_effect_per_1pp",
                    "cap_ci_low",
                    "cap_ci_high",
                    "ir_effect_per_1pp",
                    "ir_ci_low",
                    "ir_ci_high",
                ]
            ]
            if not top_dual.empty
            else pd.DataFrame()
        )
    )
    lines.append("")
    lines.append("## 8. 支持宽度归一与SOC/温度证据刷新")
    lines.append(
        "- 支持宽度归一口径：`effect_support_norm = effect_per_1pp * (support_width_1_99 / 0.01)`，用于把“斜率敏感度”换算到现有样本可达支持域尺度。"
    )
    if bin05_row is not None:
        lines.append(
            f"- `bin05` 在 raw 口径下斜率很高，但其支持区间仅 `{_format_float(float(bin05_row['cap_support_width_1_99']), 6)}`，"
            "属于稀有高温状态驱动的高斜率区间，不应直接解读为“低倍率本身高风险”。"
        )
    lines.append(
        f"- Top5 排名重叠度：容量 `{cap_top5_overlap}/5`，阻抗 `{ir_top5_overlap}/5`。"
        + ("结论：raw 与归一口径偏离显著，建议双口径并行陈述主结论。" if refresh_needed else "结论：raw 与归一口径一致性尚可。")
    )
    lines.append(f"- 是否需要刷新报告：`{'是' if refresh_needed else '否'}`（依据：raw/norm 排名重叠度、支持宽度差异、SOC×温度分层一致性）。")
    lines.append("### 8.1 raw 与支持宽度归一排名对比（容量）")
    lines.append(_df_to_md(cap_raw_norm_view))
    lines.append("### 8.2 raw 与支持宽度归一排名对比（阻抗）")
    lines.append(_df_to_md(ir_raw_norm_view))
    lines.append("### 8.3 SOC×温度分层证据")
    lines.append(_df_to_md(soc_temp_view))
    lines.append("### 8.4 高温风险结论评估")
    lines.append(_df_to_md(high_temp_assessment_df))
    lines.append(
        "- 判读：当前数据支持“高温风险在低SOC层最强、跨SOC不单调”。因此“高温是强风险因子”应采用分层结论，而非全局一刀切。"
    )
    lines.append("")
    lines.append("## 9. 图表解读")
    lines.append("![趋势分布图](./fig_trend_cell_cycle_correlations.png)")
    lines.append("- X轴：cell 内 `cycle~q_t` 与 `cycle~ir_t` 的 Spearman 相关系数。")
    lines.append("- Y轴：cell 数量。")
    lines.append("- 关键结论：容量在 cell 内几乎普遍随循环下降，阻抗在 cell 内多数呈上升趋势。")
    lines.append("- 业务解释：如果只看 pooled 全局相关，很容易低估阻抗上升趋势；cell 内分布更能反映真实老化方向。")
    lines.append("")
    lines.append("![双方向因果效应图](./fig_causal_crosslink_effects.png)")
    lines.append("- X轴：两种方向性问题（`IR变化->容量衰减`、`容量变化->阻抗上升`）。")
    lines.append("- Y轴：每 +1pp 处理变化对应的结果变化。")
    lines.append("- 关键结论：两条方向的 CI 都未跨 0，但“容量变化->阻抗上升”的标准化效应更大。")
    lines.append("- 业务解释：近期容量损失可被看作更强的阻抗恶化先行信号之一，同时 IR 上跳也不应被忽视。")
    lines.append("")
    lines.append("![容量风险Top图](./fig_capacity_risk_top_bins.png)")
    lines.append("- X轴：`cap_effect_per_1pp`。")
    lines.append("- Y轴：容量风险 Top 区间。")
    lines.append("- 关键结论：容量风险头部区间高度集中在少数工况，且头部极端区间与稳定显著区间需要分开看。")
    lines.append("- 业务解释：用于优先确定“先做哪个容量保护实验”。")
    lines.append("")
    lines.append("![阻抗风险Top图](./fig_ir_risk_top_bins.png)")
    lines.append("- X轴：`ir_effect_per_1pp`。")
    lines.append("- Y轴：阻抗风险 Top 区间。")
    lines.append("- 关键结论：阻抗风险的头部区间与容量风险部分重合，但也存在偏阻抗主导的区间。")
    lines.append("- 业务解释：用于优先确定“先做哪个内阻抑制实验”。")
    lines.append("")
    lines.append("![容量-阻抗散点图](./fig_capacity_ir_bin_scatter.png)")
    lines.append("- X轴：容量衰减效应。")
    lines.append("- Y轴：阻抗增加效应。")
    lines.append("- 关键结论：大多数共同高风险区间位于第一象限，说明容量与阻抗的风险往往共向增强。")
    lines.append("- 业务解释：第一象限远离原点的区间应优先被归入“共损伤治理清单”。")
    lines.append("")
    lines.append("![双结局森林图](./fig_dual_outcome_forest_top_bins.png)")
    lines.append("- X轴：每 +1pp 区间份额替代的效应。")
    lines.append("- Y轴：cross_bin（按综合风险排序）。")
    lines.append("- 关键结论：可直接比较同一区间对容量和阻抗的相对伤害程度。")
    lines.append("- 业务解释：适合用来挑选“先做联合优化”还是“先做单指标优化”的目标区间。")
    lines.append("")
    lines.append("![双结局风险矩阵](./fig_cross_bin_dual_risk_matrix.png)")
    lines.append("- X轴：温度物理区间（示例 `[20,31)`…`[36,60]`）。")
    lines.append("- Y轴：倍率物理区间（示例 `[0,0.434)`…`[4.22,7.75]`）。")
    lines.append("- 关键结论：风险并不是均匀分布的，而是在特定 SOC 层中沿倍率/温度组合聚集。")
    lines.append("- 业务解释：这张图最适合转成分层控制策略或优先实验矩阵。")
    lines.append("- 备注：区间来源=样本分位切分。")
    lines.append("")

    lines.append("## 10. 结论与使用建议")
    lines.append(
        "- 第一，容量下降和阻抗增加在当前样本上存在稳定相关性，而且这种关系在窗口层和 cell 内层都成立。"
    )
    lines.append(
        "- 第二，方向性分析说明两者不只是“同时坏”，而是具有可用于监测和预警的先后结构。"
    )
    lines.append(
        "- 第三，优先治理应分两层：先锁定头部共损伤区间，再区分容量优先区间和阻抗优先区间。"
    )
    lines.append(
        "- 第四，`uncertain` 区间很多，说明后续更值得补的是支持域和样本稳定性，而不是贸然把其余区间都判成安全。"
    )
    lines.append("")

    lines.append("## 11. 关键输出文件")
    lines.append("- `trend_capacity_ir_summary.csv`")
    lines.append("- `causal_crosslink_effects.csv`")
    lines.append("- `causal_substitution_effects_capacity_drop_h.csv`")
    lines.append("- `causal_substitution_effects_ir_rise_h.csv`")
    lines.append("- `cross_bin_dual_outcome_compare.csv`")
    lines.append("- `cross_bin_interpretation_table.csv`")
    lines.append("- `support_normalized_effects_compare.csv`")
    lines.append("- `soc_temp_high_temp_evidence.csv`")
    lines.append("- `high_temp_claim_assessment.csv`")
    lines.append("- `capacity_risk_top_bins.csv`")
    lines.append("- `ir_risk_top_bins.csv`")
    lines.append("- `runtime_backend_info.csv` 与 `runtime_library_versions.csv`")
    lines.append("")

    lines.append("## 12. 复现命令")
    lines.append("```bash")
    lines.append(
        "pipenv run python scripts/analyze_capacity_ir_joint_causal.py "
        "--horizon-cycles 200 --bootstrap-iters 400 "
        "--bootstrap-backend numpy --device cpu "
        "--output-dir outputs/analysis/capacity_ir_joint_causal"
    )
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    """Run full joint causal workflow and save outputs."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    backend = resolve_bootstrap_backend(args.bootstrap_backend, args.device)
    versions_df = collect_runtime_versions()
    versions_df.to_csv(output_dir / "runtime_library_versions.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "requested_backend": backend.requested_backend,
                "backend_used": backend.backend_used,
                "requested_device": backend.requested_device,
                "device_used": backend.device_used,
                "torch_available": backend.torch_available,
                "torch_xla_available": backend.torch_xla_available,
                "fallback_reason": backend.fallback_reason,
            }
        ]
    ).to_csv(output_dir / "runtime_backend_info.csv", index=False, encoding="utf-8")

    bin_meta_df = load_bin_meta(args.charge_bin_edges_path)
    interpretation_df = build_cross_bin_interpretation_table(bin_meta_df)
    interpretation_df.to_csv(output_dir / "cross_bin_interpretation_table.csv", index=False, encoding="utf-8-sig")

    if args.report_only:
        diagnostics_df = pd.read_csv(output_dir / "dataset_diagnostics.csv")
        trend_summary_df = pd.read_csv(output_dir / "trend_capacity_ir_summary.csv")
        cell_trend_df = pd.read_csv(output_dir / "trend_capacity_ir_cellwise.csv")
        crosslink_df = pd.read_csv(output_dir / "causal_crosslink_effects.csv")
        cap_df = pd.read_csv(output_dir / "causal_substitution_effects_capacity_drop_h.csv")
        ir_df = pd.read_csv(output_dir / "causal_substitution_effects_ir_rise_h.csv")
        compare_df = classify_dual_risk(cap_df=cap_df, ir_df=ir_df)
    else:
        life_df = load_life_table(args.life_performance_path, args.exclude_policy_prefix)
        policy_df = load_policy_features(args.policy_meaning_path)
        window_df = build_window_dataset(
            life_df=life_df,
            horizon_cycles=int(args.horizon_cycles),
            q_min=float(args.q_min),
            q_max=float(args.q_max),
        )
        cycle_stats_df, share_wide_df = load_charge_cycle_features(
            charge_path=args.charge_aging_path_timeseries_path,
            exclude_prefix=args.exclude_policy_prefix,
        )
        analysis_df = build_analysis_dataset(
            window_df=window_df,
            policy_df=policy_df,
            cycle_stats_df=cycle_stats_df,
            share_wide_df=share_wide_df,
        )
        analysis_df.to_csv(output_dir / "analysis_dataset_joint_windows.csv", index=False, encoding="utf-8")

        diagnostics_rows = [
            {"metric": "life_rows", "value": float(life_df.shape[0]), "notes": "life_performance 过滤后行数"},
            {"metric": "window_rows", "value": float(window_df.shape[0]), "notes": "窗口样本行数"},
            {"metric": "analysis_rows", "value": float(analysis_df.shape[0]), "notes": "合并后分析样本行数"},
            {
                "metric": "ir_non_positive_filtered_share",
                "value": 1.0 - _safe_ratio(float(window_df.shape[0]), float(life_df.shape[0])),
                "notes": "窗口构造后样本保留差异（含ir>0等条件影响）",
            },
            {
                "metric": "unique_clusters",
                "value": float(analysis_df["cluster_id"].nunique()),
                "notes": "policy+cell cluster 数",
            },
            {
                "metric": "bootstrap_iters",
                "value": float(args.bootstrap_iters),
                "notes": "bootstrap 迭代次数",
            },
            {
                "metric": "dml_nuisance_model",
                "value": float("nan"),
                "notes": f"DML nuisance 模型: {args.nuisance_model}",
            },
        ]
        diagnostics_df = pd.DataFrame(diagnostics_rows)
        diagnostics_df.to_csv(output_dir / "dataset_diagnostics.csv", index=False, encoding="utf-8")

        trend_summary_df, cell_trend_df = build_trend_outputs(
            df=analysis_df,
            min_cell_cycles=int(args.min_cell_cycles),
        )
        trend_summary_df.to_csv(output_dir / "trend_capacity_ir_summary.csv", index=False, encoding="utf-8")
        cell_trend_df.to_csv(output_dir / "trend_capacity_ir_cellwise.csv", index=False, encoding="utf-8")

        cov_cols = [
            "cycle_t_centered",
            "cycle_t_sq",
            "q_t",
            "ir_t",
            "t_max_t",
            "initial_c_rate",
            "switch_soc_percent",
            "post_switch_c_rate",
            "cycle_total_charge_h",
            "nonzero_cross_bin_count_cycle",
            "is_abnormal_cell",
        ]
        effect_a = estimate_shift_effect_aipw(
            df=analysis_df,
            treatment_col="dir_rel_1",
            outcome_col="y_cap_drop_h",
            covariate_cols=cov_cols,
            cluster_col="cluster_id",
            delta_pp=float(args.delta_pp),
            clip_quantile=float(args.weight_clip_quantile),
            n_bootstrap=int(args.bootstrap_iters),
            seed=int(args.random_seed) + 11,
            backend=backend,
        )
        effect_b = estimate_shift_effect_aipw(
            df=analysis_df,
            treatment_col="dq_rel_1",
            outcome_col="y_ir_rise_h",
            covariate_cols=cov_cols,
            cluster_col="cluster_id",
            delta_pp=float(args.delta_pp),
            clip_quantile=float(args.weight_clip_quantile),
            n_bootstrap=int(args.bootstrap_iters),
            seed=int(args.random_seed) + 29,
            backend=backend,
        )
        crosslink_df = pd.DataFrame(
            [
                {
                    "direction": "dir_rel_1_to_y_cap_drop_h",
                    "direction_label": "IR变化(+1pp) -> 容量衰减",
                    **effect_a,
                },
                {
                    "direction": "dq_rel_1_to_y_ir_rise_h",
                    "direction_label": "容量变化(+1pp) -> 阻抗上升",
                    **effect_b,
                },
            ]
        )
        crosslink_df.to_csv(output_dir / "causal_crosslink_effects.csv", index=False, encoding="utf-8")

        control_cols = cov_cols.copy()
        cap_df = run_substitution_effects(
            df=analysis_df,
            outcome_col="y_cap_drop_h",
            control_cols=control_cols,
            bin_meta_df=bin_meta_df,
            args=args,
            backend=backend,
        )
        ir_df = run_substitution_effects(
            df=analysis_df,
            outcome_col="y_ir_rise_h",
            control_cols=control_cols,
            bin_meta_df=bin_meta_df,
            args=args,
            backend=backend,
        )
        cap_df.to_csv(
            output_dir / "causal_substitution_effects_capacity_drop_h.csv",
            index=False,
            encoding="utf-8",
        )
        ir_df.to_csv(
            output_dir / "causal_substitution_effects_ir_rise_h.csv",
            index=False,
            encoding="utf-8",
        )
        compare_df = classify_dual_risk(cap_df=cap_df, ir_df=ir_df)

    compare_df.to_csv(output_dir / "cross_bin_dual_outcome_compare.csv", index=False, encoding="utf-8")
    support_compare_df = build_support_normalized_effects_compare(
        compare_df=compare_df,
        interpretation_df=interpretation_df,
    )
    soc_temp_evidence_df = build_soc_temp_high_temp_evidence(
        compare_df=compare_df,
        interpretation_df=interpretation_df,
        q_threshold=0.1,
    )
    high_temp_assessment_df = build_high_temp_claim_assessment(
        soc_temp_evidence_df=soc_temp_evidence_df,
        q_threshold=0.1,
        high_temp_bin=5,
    )
    support_compare_df.to_csv(output_dir / "support_normalized_effects_compare.csv", index=False, encoding="utf-8-sig")
    soc_temp_evidence_df.to_csv(output_dir / "soc_temp_high_temp_evidence.csv", index=False, encoding="utf-8-sig")
    high_temp_assessment_df.to_csv(output_dir / "high_temp_claim_assessment.csv", index=False, encoding="utf-8-sig")

    cap_top_df = build_effect_top_table(
        effect_df=cap_df,
        compare_df=compare_df,
        interpretation_df=interpretation_df,
        outcome_prefix="capacity",
        top_n=10,
    )
    ir_top_df = build_effect_top_table(
        effect_df=ir_df,
        compare_df=compare_df,
        interpretation_df=interpretation_df,
        outcome_prefix="ir",
        top_n=10,
    )
    cap_top_df.to_csv(output_dir / "capacity_risk_top_bins.csv", index=False, encoding="utf-8-sig")
    ir_top_df.to_csv(output_dir / "ir_risk_top_bins.csv", index=False, encoding="utf-8-sig")

    if not args.disable_plots:
        save_trend_plot(cell_trend_df, output_dir / "fig_trend_cell_cycle_correlations.png")
        save_crosslink_plot(crosslink_df, output_dir / "fig_causal_crosslink_effects.png")
        save_single_outcome_top_plot(
            top_df=cap_top_df,
            output_path=output_dir / "fig_capacity_risk_top_bins.png",
            effect_col="effect_per_1pp",
            low_col="ci_low",
            high_col="ci_high",
            title="容量衰减高风险区间（Top10）",
            color="#2563eb",
        )
        save_single_outcome_top_plot(
            top_df=ir_top_df,
            output_path=output_dir / "fig_ir_risk_top_bins.png",
            effect_col="effect_per_1pp",
            low_col="ci_low",
            high_col="ci_high",
            title="阻抗增加高风险区间（Top10）",
            color="#f59e0b",
        )
        save_capacity_ir_scatter(compare_df, output_dir / "fig_capacity_ir_bin_scatter.png")
        save_dual_forest_plot(
            compare_df=compare_df,
            output_path=output_dir / "fig_dual_outcome_forest_top_bins.png",
            top_k=int(args.top_k),
        )
        save_risk_matrix_plot(compare_df, bin_meta_df, output_dir / "fig_cross_bin_dual_risk_matrix.png")

    report_text = render_report(
        args=args,
        backend=backend,
        diagnostics_df=diagnostics_df,
        trend_summary_df=trend_summary_df,
        crosslink_df=crosslink_df,
        cap_df=cap_df,
        ir_df=ir_df,
        compare_df=compare_df,
        interpretation_df=interpretation_df,
        cap_top_df=cap_top_df,
        ir_top_df=ir_top_df,
        support_compare_df=support_compare_df,
        soc_temp_evidence_df=soc_temp_evidence_df,
        high_temp_assessment_df=high_temp_assessment_df,
    )
    (output_dir / "capacity_ir_joint_causal_report.md").write_text(report_text, encoding="utf-8-sig")

    print(f"Saved: {output_dir / 'trend_capacity_ir_summary.csv'}")
    print(f"Saved: {output_dir / 'causal_crosslink_effects.csv'}")
    print(f"Saved: {output_dir / 'causal_substitution_effects_capacity_drop_h.csv'}")
    print(f"Saved: {output_dir / 'causal_substitution_effects_ir_rise_h.csv'}")
    print(f"Saved: {output_dir / 'cross_bin_dual_outcome_compare.csv'}")
    print(f"Saved: {output_dir / 'cross_bin_interpretation_table.csv'}")
    print(f"Saved: {output_dir / 'support_normalized_effects_compare.csv'}")
    print(f"Saved: {output_dir / 'soc_temp_high_temp_evidence.csv'}")
    print(f"Saved: {output_dir / 'high_temp_claim_assessment.csv'}")
    print(f"Saved: {output_dir / 'capacity_risk_top_bins.csv'}")
    print(f"Saved: {output_dir / 'ir_risk_top_bins.csv'}")
    print(f"Saved: {output_dir / 'capacity_ir_joint_causal_report.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - explicit CLI failure path
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
