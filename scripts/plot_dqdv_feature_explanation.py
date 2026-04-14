from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt


# =========================
# Config (edit here first)
# =========================
RAW_FILE_PATH = REPO_ROOT / "data" / "raw" / "4_8C-80PER_4_8C" / "cycles_465027.csv"
TARGET_POLICY = "4_8C-80PER_4_8C"
TARGET_CELL_CODE = "465027"
TARGET_CYCLE = 10

VOLTAGE_DRIFT_ACCEPTANCE_V = 0.015

OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "dqdv_feature_explanation"
OUTPUT_COMPARISON_FIG_PATH = OUTPUT_DIR / "dqdv_balanced_comparison_cycle10.png"
OUTPUT_CYCLE_CSV_PATH = OUTPUT_DIR / "dqdv_balanced_peak_comparison_cycle10.csv"
OUTPUT_CELL_METRICS_CSV_PATH = OUTPUT_DIR / "dqdv_balanced_cell_cycle_metrics.csv"
OUTPUT_SUMMARY_MD_PATH = OUTPUT_DIR / "dqdv_balanced_summary.md"


def load_extract_module() -> Any:
    """Load the dQ/dV extraction module for shared logic reuse."""
    module_path = REPO_ROOT / "scripts" / "extract_discharge_dqdv_peak_features.py"
    spec = importlib.util.spec_from_file_location("extract_dqdv_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extraction module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_target_cell_frame(mod: Any) -> pd.DataFrame:
    """Load and filter one target cell's discharge rows from raw data."""
    frame = mod.load_discharge_frame(RAW_FILE_PATH)
    frame = frame.loc[
        (frame["policy"] == TARGET_POLICY)
        & (frame["cell_code"].astype(str) == TARGET_CELL_CODE)
    ].copy()
    if frame.empty:
        raise ValueError(f"No rows found for policy={TARGET_POLICY}, cell_code={TARGET_CELL_CODE}")
    return frame.sort_values(["cycles", "ts"]).reset_index(drop=True)


def get_target_cycle_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the configured target cycle rows."""
    part = frame.loc[frame["cycles"] == TARGET_CYCLE].copy()
    if part.empty:
        raise ValueError(f"No rows found for target cycle={TARGET_CYCLE}")
    return part.sort_values("ts").reset_index(drop=True)


def roughness_score(y: np.ndarray) -> float:
    """Return mean absolute second-order difference as roughness index."""
    if len(y) < 3:
        return float("nan")
    return float(np.mean(np.abs(np.diff(y, 2))))


def normalize_peak(peak: Any) -> Dict[str, float]:
    """Convert one PeakFeature-like object to dict."""
    return {
        "voltage_v": float(peak.voltage_v),
        "height_dqdv": float(peak.height_dqdv),
        "area": float(peak.area),
        "prominence": float(peak.prominence),
        "width_v": float(peak.width_v),
    }


def build_cycle_peak_comparison_df(current_peaks: List[Any], balanced_peaks: List[Any], top_k: int) -> pd.DataFrame:
    """Build rank-aligned peak parameter comparison table for one cycle."""
    rows: List[dict] = []
    for rank in range(1, top_k + 1):
        cur = normalize_peak(current_peaks[rank - 1]) if rank <= len(current_peaks) else {}
        bal = normalize_peak(balanced_peaks[rank - 1]) if rank <= len(balanced_peaks) else {}
        row = {
            "peak_rank": rank,
            "current_voltage_v": cur.get("voltage_v", np.nan),
            "current_height_dqdv": cur.get("height_dqdv", np.nan),
            "current_area": cur.get("area", np.nan),
            "current_prominence": cur.get("prominence", np.nan),
            "current_width_v": cur.get("width_v", np.nan),
            "balanced_voltage_v": bal.get("voltage_v", np.nan),
            "balanced_height_dqdv": bal.get("height_dqdv", np.nan),
            "balanced_area": bal.get("area", np.nan),
            "balanced_prominence": bal.get("prominence", np.nan),
            "balanced_width_v": bal.get("width_v", np.nan),
        }
        row["delta_voltage_v"] = row["balanced_voltage_v"] - row["current_voltage_v"]
        row["delta_height_dqdv"] = row["balanced_height_dqdv"] - row["current_height_dqdv"]
        row["delta_area"] = row["balanced_area"] - row["current_area"]
        row["delta_prominence"] = row["balanced_prominence"] - row["current_prominence"]
        row["delta_width_v"] = row["balanced_width_v"] - row["current_width_v"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cell_cycle_metrics(mod: Any, cell_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute current-vs-balanced diagnostics on all cycles of target cell."""
    rows: List[dict] = []
    for cycle, group in cell_frame.groupby("cycles", sort=True):
        x, y_current, n_window, n_dqdv = mod.build_dqdv_series(group)
        if len(y_current) == 0:
            continue
        y_balanced = mod.apply_balanced_post_smooth(y_current, force=True)

        peaks_current = mod.detect_peaks(x, y_current)
        peaks_balanced = mod.detect_peaks(x, y_balanced)

        peak1_abs_current = (
            float(abs(peaks_current[0].height_dqdv))
            if len(peaks_current) > 0
            else np.nan
        )
        peak1_abs_balanced = (
            float(abs(peaks_balanced[0].height_dqdv))
            if len(peaks_balanced) > 0
            else np.nan
        )
        peak1_voltage_drift = (
            float(peaks_balanced[0].voltage_v - peaks_current[0].voltage_v)
            if len(peaks_current) > 0 and len(peaks_balanced) > 0
            else np.nan
        )

        rows.append(
            {
                "cycles": int(cycle),
                "n_points_window": int(n_window),
                "n_points_dqdv": int(n_dqdv),
                "peak_count_current": int(len(peaks_current)),
                "peak_count_balanced": int(len(peaks_balanced)),
                "peak1_abs_height_current": peak1_abs_current,
                "peak1_abs_height_balanced": peak1_abs_balanced,
                "peak1_voltage_drift_v": peak1_voltage_drift,
                "roughness_current": roughness_score(y_current),
                "roughness_balanced": roughness_score(y_balanced),
            }
        )
    return pd.DataFrame(rows).sort_values("cycles").reset_index(drop=True)


def build_summary_metrics(cycle_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate summary metrics for acceptance checks."""
    if cycle_df.empty:
        raise ValueError("No valid dQ/dV cycles for target cell.")

    cycle_df = cycle_df.copy()
    cycle_df["roughness_reduction_ratio"] = (
        (cycle_df["roughness_current"] - cycle_df["roughness_balanced"])
        / cycle_df["roughness_current"].replace(0.0, np.nan)
    )

    metrics = {
        "valid_cycles": int(len(cycle_df)),
        "peak_cycles_current": int((cycle_df["peak_count_current"] >= 1).sum()),
        "peak_cycles_balanced": int((cycle_df["peak_count_balanced"] >= 1).sum()),
        "median_peak_count_current": float(cycle_df["peak_count_current"].median()),
        "median_peak_count_balanced": float(cycle_df["peak_count_balanced"].median()),
        "max_peak1_abs_current": float(cycle_df["peak1_abs_height_current"].max(skipna=True)),
        "max_peak1_abs_balanced": float(cycle_df["peak1_abs_height_balanced"].max(skipna=True)),
        "roughness_mean_current": float(cycle_df["roughness_current"].mean(skipna=True)),
        "roughness_mean_balanced": float(cycle_df["roughness_balanced"].mean(skipna=True)),
        "roughness_reduction_mean_ratio": float(cycle_df["roughness_reduction_ratio"].mean(skipna=True)),
        "p95_abs_voltage_drift_v": float(
            np.nanpercentile(np.abs(cycle_df["peak1_voltage_drift_v"].to_numpy(dtype=float)), 95)
        ),
    }
    return metrics


def plot_cycle_comparison(
    group: pd.DataFrame,
    voltage_mid: np.ndarray,
    y_current: np.ndarray,
    y_balanced: np.ndarray,
    current_peaks: List[Any],
    balanced_peaks: List[Any],
) -> None:
    """Save one comparison figure for raw/current/balanced dQ/dV."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    ax_raw, ax_curve, ax_bar = axes

    ax_raw.plot(
        group["V"].to_numpy(dtype=float),
        group["ah_dischg"].to_numpy(dtype=float),
        lw=1.2,
        color="#1f77b4",
    )
    ax_raw.set_title("A) Raw Discharge Q-V")
    ax_raw.set_xlabel("Voltage V (V)")
    ax_raw.set_ylabel("Discharge Capacity Q (Ah)")
    ax_raw.set_xlim(3.6, 2.8)
    ax_raw.grid(alpha=0.25, linestyle="--")

    ax_curve.plot(voltage_mid, y_current, lw=1.5, color="#1f77b4", label="current dQ/dV")
    ax_curve.plot(voltage_mid, y_balanced, lw=1.6, color="#ff7f0e", label="balanced dQ/dV (SG15,3)")

    for idx, peak in enumerate(current_peaks, start=1):
        ax_curve.scatter(
            peak.voltage_v,
            peak.height_dqdv,
            s=35,
            color="#1f77b4",
            marker="o",
            zorder=4,
            label=f"current_peak{idx}" if idx == 1 else None,
        )
    for idx, peak in enumerate(balanced_peaks, start=1):
        ax_curve.scatter(
            peak.voltage_v,
            peak.height_dqdv,
            s=45,
            color="#d62728",
            marker="x",
            zorder=5,
            label=f"balanced_peak{idx}" if idx == 1 else None,
        )
    ax_curve.set_title("B) Current vs Balanced dQ/dV")
    ax_curve.set_xlabel("Voltage V (V)")
    ax_curve.set_ylabel("dQ/dV (Ah/V)")
    ax_curve.set_xlim(3.6, 2.8)
    ax_curve.grid(alpha=0.25, linestyle="--")
    ax_curve.legend(loc="best", fontsize=8)

    peak1_abs_current = float(abs(current_peaks[0].height_dqdv)) if len(current_peaks) > 0 else np.nan
    peak1_abs_balanced = float(abs(balanced_peaks[0].height_dqdv)) if len(balanced_peaks) > 0 else np.nan
    peak1_area_current = float(abs(current_peaks[0].area)) if len(current_peaks) > 0 else np.nan
    peak1_area_balanced = float(abs(balanced_peaks[0].area)) if len(balanced_peaks) > 0 else np.nan

    metric_names = ["roughness", "|peak1_height|", "|peak1_area|"]
    current_vals = [roughness_score(y_current), peak1_abs_current, peak1_area_current]
    balanced_vals = [roughness_score(y_balanced), peak1_abs_balanced, peak1_area_balanced]
    x_pos = np.arange(len(metric_names))
    width = 0.36
    ax_bar.bar(x_pos - width / 2, current_vals, width, label="current", color="#1f77b4", alpha=0.75)
    ax_bar.bar(x_pos + width / 2, balanced_vals, width, label="balanced", color="#ff7f0e", alpha=0.75)
    ax_bar.set_title("C) Feature Magnitude Comparison")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metric_names, rotation=0)
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")
    ax_bar.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Balanced dQ/dV Comparison | policy={TARGET_POLICY}, cell={TARGET_CELL_CODE}, cycle={TARGET_CYCLE}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_COMPARISON_FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    cycle_peak_df: pd.DataFrame,
    cell_cycle_df: pd.DataFrame,
    summary_metrics: Dict[str, float],
    cycle_current_peak_count: int,
    cycle_balanced_peak_count: int,
) -> None:
    """Write compact markdown report for balanced dQ/dV evaluation."""
    peak1_row = cycle_peak_df.iloc[0] if not cycle_peak_df.empty else None
    peak1_drift_abs = (
        abs(float(peak1_row["delta_voltage_v"]))
        if peak1_row is not None and np.isfinite(peak1_row["delta_voltage_v"])
        else np.nan
    )
    drift_pass = bool(np.isfinite(peak1_drift_abs) and peak1_drift_abs <= VOLTAGE_DRIFT_ACCEPTANCE_V)

    no_new_empty_pass = summary_metrics["peak_cycles_balanced"] >= summary_metrics["peak_cycles_current"]

    lines: List[str] = []
    lines.append("# Balanced dQ/dV Example Validation")
    lines.append("")
    lines.append(f"- Target: `policy={TARGET_POLICY}`, `cell_code={TARGET_CELL_CODE}`, `cycle={TARGET_CYCLE}`")
    lines.append("- Balanced strategy: second-pass Savitzky-Golay `SG(15,3)` on top of current dQ/dV.")
    lines.append("")
    lines.append("## 1. Cycle-10 Acceptance Checks")
    lines.append("")
    lines.append(f"- current peak count: **{cycle_current_peak_count}**")
    lines.append(f"- balanced peak count: **{cycle_balanced_peak_count}**")
    lines.append(
        f"- peak-1 voltage drift: **{peak1_drift_abs:.6f} V** "
        f"(threshold `{VOLTAGE_DRIFT_ACCEPTANCE_V:.3f}V`, pass={drift_pass})"
    )
    if peak1_row is not None:
        lines.append(
            f"- peak-1 |height|: current `{abs(float(peak1_row['current_height_dqdv'])):.6f}` "
            f"-> balanced `{abs(float(peak1_row['balanced_height_dqdv'])):.6f}`"
        )
        lines.append(
            f"- peak-1 |area|: current `{abs(float(peak1_row['current_area'])):.6f}` "
            f"-> balanced `{abs(float(peak1_row['balanced_area'])):.6f}`"
        )
    lines.append("")
    lines.append("## 2. Whole-Cell (All Cycles) Diagnostics")
    lines.append("")
    lines.append(f"- valid dQ/dV cycles: **{summary_metrics['valid_cycles']}**")
    lines.append(
        f"- peak cycles current/balanced: **{summary_metrics['peak_cycles_current']} / "
        f"{summary_metrics['peak_cycles_balanced']}** (no new empty cycles pass={no_new_empty_pass})"
    )
    lines.append(
        f"- median peak count current/balanced: **{summary_metrics['median_peak_count_current']:.3f} / "
        f"{summary_metrics['median_peak_count_balanced']:.3f}**"
    )
    lines.append(
        f"- max |peak1_height| current/balanced: **{summary_metrics['max_peak1_abs_current']:.6f} / "
        f"{summary_metrics['max_peak1_abs_balanced']:.6f}**"
    )
    lines.append(
        f"- mean roughness current/balanced: **{summary_metrics['roughness_mean_current']:.6f} / "
        f"{summary_metrics['roughness_mean_balanced']:.6f}** "
        f"(mean reduction ratio={summary_metrics['roughness_reduction_mean_ratio']:.2%})"
    )
    lines.append(
        f"- P95 |peak1 voltage drift| across cycles: **{summary_metrics['p95_abs_voltage_drift_v']:.6f} V**"
    )
    lines.append("")
    lines.append("## 3. Output Files")
    lines.append("")
    lines.append(f"- Figure: `{OUTPUT_COMPARISON_FIG_PATH}`")
    lines.append(f"- Cycle peak table: `{OUTPUT_CYCLE_CSV_PATH}`")
    lines.append(f"- Cell cycle metrics: `{OUTPUT_CELL_METRICS_CSV_PATH}`")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run balanced dQ/dV comparison on the configured sample cell/cycle."""
    mod = load_extract_module()
    cell_frame = load_target_cell_frame(mod)
    cycle_group = get_target_cycle_group(cell_frame)

    voltage_mid, y_current, _, _ = mod.build_dqdv_series(cycle_group)
    if len(voltage_mid) == 0 or len(y_current) == 0:
        raise RuntimeError("No valid dQ/dV series for target cycle.")

    y_balanced = mod.apply_balanced_post_smooth(y_current, force=True)
    current_peaks = mod.detect_peaks(voltage_mid, y_current)
    balanced_peaks = mod.detect_peaks(voltage_mid, y_balanced)
    if len(balanced_peaks) == 0:
        raise RuntimeError("No balanced peaks detected on target cycle.")

    cycle_peak_df = build_cycle_peak_comparison_df(current_peaks, balanced_peaks, int(mod.TOP_K_PEAKS))
    cell_cycle_df = compute_cell_cycle_metrics(mod, cell_frame)
    summary_metrics = build_summary_metrics(cell_cycle_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cycle_peak_df.to_csv(OUTPUT_CYCLE_CSV_PATH, index=False, encoding="utf-8")
    cell_cycle_df.to_csv(OUTPUT_CELL_METRICS_CSV_PATH, index=False, encoding="utf-8")

    plot_cycle_comparison(cycle_group, voltage_mid, y_current, y_balanced, current_peaks, balanced_peaks)
    write_summary_markdown(
        cycle_peak_df=cycle_peak_df,
        cell_cycle_df=cell_cycle_df,
        summary_metrics=summary_metrics,
        cycle_current_peak_count=len(current_peaks),
        cycle_balanced_peak_count=len(balanced_peaks),
    )

    print(f"Saved figure: {OUTPUT_COMPARISON_FIG_PATH}")
    print(f"Saved cycle comparison csv: {OUTPUT_CYCLE_CSV_PATH}")
    print(f"Saved cell metrics csv: {OUTPUT_CELL_METRICS_CSV_PATH}")
    print(f"Saved summary markdown: {OUTPUT_SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
