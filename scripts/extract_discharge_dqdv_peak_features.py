from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


# =========================
# Config
# =========================
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

SAMPLE_CURVE_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_curve_points_sample10.csv"
SAMPLE_FEATURE_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_peak_features_sample10.csv"
SAMPLE_SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_extraction_summary_sample10.json"

FULL_FEATURE_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_peak_features_skill_full.csv"
FULL_SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_extraction_summary_skill_full.json"

ENCODING = "utf-8-sig"
CYCLES_FILE_GLOB = "cycles_*.csv"
SUMMARY_FILE_GLOB = "summary_*.csv"

DEFAULT_SAMPLE_CYCLE_COUNT = 10
DEFAULT_RANDOM_SEED = 20260414

CURVE_METHOD = "polyfit"
POLY_DEGREE_MIN = 3
POLY_DEGREE_MAX = 7
VOLTAGE_STEP_V = 0.002
POST_SMOOTH_WINDOW = 7

MIN_POINTS = 15
MIN_DQDV_POINTS = 10
MIN_VOLTAGE_SPAN = 0.05
MIN_CAPACITY_SPAN_AH = 0.01

PEAK_MIN_DISTANCE_POINTS = 5
PEAK_MIN_HEIGHT = 0.0
PEAK_PROMINENCE_ABS_MIN = 0.003
PEAK_PROMINENCE_REL_Q90 = 0.15
TOP_K_PEAKS = 2

FLOAT_EPS = 1e-12

PEAK_SIGFIG_COLUMNS = [
    "main_peak_voltage_v",
    "main_peak_width_v",
    "main_peak_height_dqdv",
    "main_peak_area",
    "main_peak_prominence",
    "main_peak_skewness",
    "second_peak_voltage_v",
    "second_peak_width_v",
    "second_peak_height_dqdv",
    "second_peak_area",
    "second_peak_prominence",
    "second_peak_skewness",
    "main_second_peak_voltage_gap_v",
]

TEMP_SIGFIG_COLUMNS = [
    "temp_max_c",
    "temp_min_c",
    "temp_avg_c",
    "main_peak_temp_max_c",
    "main_peak_temp_min_c",
    "main_peak_temp_avg_c",
    "second_peak_temp_max_c",
    "second_peak_temp_min_c",
    "second_peak_temp_avg_c",
]


@dataclass(frozen=True)
class CycleKey:
    """Unique cycle identifier."""

    policy: str
    cell_code: str
    cycles: int

    def as_tuple(self) -> Tuple[str, str, int]:
        """Convert to tuple for set filtering."""
        return (self.policy, self.cell_code, self.cycles)


@dataclass
class FitOutcome:
    """Best polynomial fit diagnostics."""

    degree: int
    poly: np.polynomial.Polynomial
    r2: float
    mae: float


@dataclass
class PeakFeature:
    """One dQ/dV peak feature bundle."""

    voltage_v: float
    width_v: float
    height_dqdv: float
    area: float
    prominence: float
    skewness: float
    temp_max_c: float
    temp_min_c: float
    temp_avg_c: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Skill-aligned discharge dQ/dV curve+peak extraction."
    )
    parser.add_argument("--mode", choices=["sample10", "smoke", "full"], default="sample10")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_CYCLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-features", type=str, default="")
    parser.add_argument("--output-curve-points", type=str, default="")
    parser.add_argument("--output-summary", type=str, default="")
    parser.add_argument("--export-curve-points", dest="export_curve_points", action="store_true")
    parser.add_argument("--no-export-curve-points", dest="export_curve_points", action="store_false")
    parser.set_defaults(export_curve_points=None)
    return parser.parse_args()


def find_cycles_files() -> List[Path]:
    """List all cycles files."""
    files = sorted(RAW_DIR.rglob(CYCLES_FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {CYCLES_FILE_GLOB!r} under {RAW_DIR}")
    return files


def load_discharge_frame(file_path: Path) -> pd.DataFrame:
    """Load one cycles file and keep discharge rows."""
    usecols = ["policy", "cell_code", "cycles", "ts", "V", "Temper", "ah_dischg", "flag_dischg"]
    frame = pd.read_csv(file_path, usecols=usecols, encoding=ENCODING, low_memory=False)
    frame["cycles"] = pd.to_numeric(frame["cycles"], errors="coerce")
    frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
    frame["V"] = pd.to_numeric(frame["V"], errors="coerce")
    frame["Temper"] = pd.to_numeric(frame["Temper"], errors="coerce")
    frame["ah_dischg"] = pd.to_numeric(frame["ah_dischg"], errors="coerce")
    frame["flag_dischg"] = pd.to_numeric(frame["flag_dischg"], errors="coerce")

    frame = frame.dropna(subset=["policy", "cell_code", "cycles", "ts", "V", "ah_dischg", "flag_dischg"]).copy()
    if frame.empty:
        return frame

    frame = frame.loc[frame["flag_dischg"].astype(int) == 1].copy()
    if frame.empty:
        return frame

    frame["cycles"] = frame["cycles"].astype(int)
    frame["cell_code"] = frame["cell_code"].astype(str)
    return frame.sort_values(["policy", "cell_code", "cycles", "ts"]).reset_index(drop=True)


def collect_sampling_keys() -> pd.DataFrame:
    """Collect cycle keys from summary files for stratified sample mode."""
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(RAW_DIR.rglob(SUMMARY_FILE_GLOB)):
        suffix = summary_path.stem.replace("summary_", "")
        cycles_path = summary_path.with_name(f"cycles_{suffix}.csv")
        if not cycles_path.exists():
            continue
        try:
            part = pd.read_csv(
                summary_path,
                usecols=["policy", "cell_code", "cycle"],
                encoding=ENCODING,
                low_memory=False,
            )
        except Exception:
            continue
        part["cycle"] = pd.to_numeric(part["cycle"], errors="coerce")
        part = part.dropna(subset=["policy", "cell_code", "cycle"]).copy()
        if part.empty:
            continue
        part["cell_code"] = part["cell_code"].astype(str)
        part["cycles"] = part["cycle"].astype(int)
        part["source_file"] = str(cycles_path.resolve())
        rows.extend(part[["policy", "cell_code", "cycles", "source_file"]].to_dict(orient="records"))
    if not rows:
        raise RuntimeError("No sampling keys found from summary files.")
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def stratified_sample_keys(keys_df: pd.DataFrame, sample_count: int, seed: int) -> pd.DataFrame:
    """Round-robin sample across policies with fixed random seed."""
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    if sample_count >= len(keys_df):
        return keys_df.copy()

    rng = Random(seed)
    buckets: Dict[str, List[dict]] = {}
    for policy, grp in keys_df.groupby("policy", sort=True):
        rows = grp.to_dict(orient="records")
        rng.shuffle(rows)
        buckets[str(policy)] = rows

    policy_order = sorted(buckets.keys())
    rng.shuffle(policy_order)

    chosen: List[dict] = []
    seen: Set[Tuple[str, str, int]] = set()
    while len(chosen) < sample_count:
        progressed = False
        for policy in policy_order:
            bag = buckets[policy]
            if not bag:
                continue
            row = bag.pop()
            key = (str(row["policy"]), str(row["cell_code"]), int(row["cycles"]))
            if key in seen:
                continue
            chosen.append(row)
            seen.add(key)
            progressed = True
            if len(chosen) >= sample_count:
                break
        if not progressed:
            break

    if len(chosen) < sample_count:
        raise RuntimeError(f"Insufficient sampled keys: expected={sample_count}, got={len(chosen)}")
    return pd.DataFrame(chosen).sort_values(["policy", "cell_code", "cycles"]).reset_index(drop=True)


def prepare_monotone_qv(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare monotone Q(V) for discharge with oriented voltage."""
    part = group[["ts", "V", "ah_dischg"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if part.empty:
        return np.array([]), np.array([]), np.array([])
    part = part.sort_values("ts").reset_index(drop=True)
    part["voltage"] = part["V"].astype(float)
    part["oriented_voltage"] = -part["voltage"]
    part["capacity"] = part["ah_dischg"].astype(float)
    collapsed = (
        part.groupby("oriented_voltage", as_index=False)
        .agg(voltage=("voltage", "mean"), capacity=("capacity", "max"))
        .sort_values("oriented_voltage")
        .reset_index(drop=True)
    )
    if collapsed.empty:
        return np.array([]), np.array([]), np.array([])
    cap = np.maximum.accumulate(collapsed["capacity"].to_numpy(dtype=float))
    ov = collapsed["oriented_voltage"].to_numpy(dtype=float)
    vv = collapsed["voltage"].to_numpy(dtype=float)
    finite = np.isfinite(ov) & np.isfinite(vv) & np.isfinite(cap)
    return ov[finite], vv[finite], cap[finite]


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 safely."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def fit_best_poly(oriented_v: np.ndarray, capacity: np.ndarray) -> Tuple[Optional[FitOutcome], str]:
    """Fit degree 3..7 and return best fit by R2, MAE, then lower degree."""
    if len(oriented_v) < MIN_POINTS:
        return None, "too_few_points"
    if float(np.ptp(oriented_v)) < MIN_VOLTAGE_SPAN:
        return None, "insufficient_voltage_span"
    if float(np.ptp(capacity)) < MIN_CAPACITY_SPAN_AH:
        return None, "insufficient_capacity_span"

    max_degree = min(POLY_DEGREE_MAX, len(oriented_v) - 1)
    if max_degree < POLY_DEGREE_MIN:
        return None, "too_few_points_for_poly_degree"

    candidates: List[FitOutcome] = []
    for deg in range(POLY_DEGREE_MIN, max_degree + 1):
        try:
            poly = np.polynomial.Polynomial.fit(oriented_v, capacity, deg=deg).convert()
        except Exception:
            continue
        pred = poly(oriented_v)
        if not np.all(np.isfinite(pred)):
            continue
        candidates.append(
            FitOutcome(
                degree=deg,
                poly=poly,
                r2=safe_r2(capacity, pred),
                mae=float(np.mean(np.abs(capacity - pred))),
            )
        )
    if not candidates:
        return None, "fitting_failure"
    candidates.sort(key=lambda c: (-np.nan_to_num(c.r2, nan=-1e18), c.mae, c.degree))
    return candidates[0], ""


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling-mean smoother."""
    if len(y) == 0:
        return np.array([])
    if window <= 1:
        return np.array(y, copy=True)
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def build_dqdv(
    fit: FitOutcome, oriented_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Build resampled curve arrays: voltage, oriented_voltage, capacity_fit, dqdv_raw, dqdv_smooth."""
    vmin = float(np.min(oriented_v))
    vmax = float(np.max(oriented_v))
    if vmax - vmin <= FLOAT_EPS:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), "no_valid_voltage_steps"

    grid = np.arange(vmin, vmax + FLOAT_EPS, VOLTAGE_STEP_V, dtype=float)
    if len(grid) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), "too_few_resampled_points"

    cap = fit.poly(grid)
    if not np.all(np.isfinite(cap)):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), "non_finite_fit_output"

    dq = np.diff(cap)
    dq = np.clip(dq, a_min=0.0, a_max=None)
    dqdv_raw = dq / VOLTAGE_STEP_V
    if len(dqdv_raw) < MIN_DQDV_POINTS:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), "too_few_dqdv_points"
    dqdv_smooth = rolling_mean(dqdv_raw, POST_SMOOTH_WINDOW)

    oriented_mid = 0.5 * (grid[1:] + grid[:-1])
    voltage_mid = -oriented_mid
    cap_mid = 0.5 * (cap[1:] + cap[:-1])
    return voltage_mid, oriented_mid, cap_mid, dqdv_raw, dqdv_smooth, ""


def idx_to_voltage(voltage: np.ndarray, floating_index: float) -> float:
    """Interpolate voltage at floating index."""
    return float(np.interp(floating_index, np.arange(len(voltage), dtype=float), voltage))


def compute_skewness(values: np.ndarray) -> float:
    """Compute skewness with safe handling for near-constant arrays."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    if std <= FLOAT_EPS:
        return 0.0
    centered = (x - mean) / std
    return float(np.mean(centered**3))


def detect_peaks(
    voltage: np.ndarray,
    dqdv_smooth: np.ndarray,
    raw_voltage: np.ndarray,
    raw_temper: np.ndarray,
) -> List[PeakFeature]:
    """Detect and rank peaks by height then prominence."""
    if len(voltage) < 3 or len(dqdv_smooth) < 3:
        return []
    if not np.any(np.isfinite(dqdv_smooth)):
        return []

    q90 = float(np.nanpercentile(dqdv_smooth, 90))
    prominence = max(PEAK_PROMINENCE_ABS_MIN, q90 * PEAK_PROMINENCE_REL_Q90)
    peaks, props = find_peaks(
        dqdv_smooth,
        prominence=prominence,
        distance=PEAK_MIN_DISTANCE_POINTS,
        height=PEAK_MIN_HEIGHT,
    )
    if len(peaks) == 0:
        return []

    widths = peak_widths(dqdv_smooth, peaks, rel_height=0.5)
    left_ips = widths[2]
    right_ips = widths[3]

    results: List[PeakFeature] = []
    for i, pidx in enumerate(peaks):
        left_base = int(props["left_bases"][i])
        right_base = int(props["right_bases"][i])
        if right_base <= left_base:
            continue
        seg_v = voltage[left_base : right_base + 1]
        seg_y = dqdv_smooth[left_base : right_base + 1]
        if len(seg_v) < 2:
            continue
        order = np.argsort(seg_v)
        if hasattr(np, "trapezoid"):
            area = float(np.trapezoid(seg_y[order], seg_v[order]))
        else:
            area = float(np.trapz(seg_y[order], seg_v[order]))
        width_v = abs(idx_to_voltage(voltage, right_ips[i]) - idx_to_voltage(voltage, left_ips[i]))

        seg_v_low = float(np.min(seg_v))
        seg_v_high = float(np.max(seg_v))
        temp_mask = (
            np.isfinite(raw_voltage)
            & np.isfinite(raw_temper)
            & (raw_voltage >= seg_v_low - FLOAT_EPS)
            & (raw_voltage <= seg_v_high + FLOAT_EPS)
        )
        peak_temps = raw_temper[temp_mask]
        if len(peak_temps) > 0:
            temp_max_c = float(np.max(peak_temps))
            temp_min_c = float(np.min(peak_temps))
            temp_avg_c = float(np.mean(peak_temps))
        else:
            temp_max_c = float("nan")
            temp_min_c = float("nan")
            temp_avg_c = float("nan")

        results.append(
            PeakFeature(
                voltage_v=float(voltage[pidx]),
                width_v=width_v,
                height_dqdv=float(dqdv_smooth[pidx]),
                area=area,
                prominence=float(props["prominences"][i]),
                skewness=compute_skewness(seg_y),
                temp_max_c=temp_max_c,
                temp_min_c=temp_min_c,
                temp_avg_c=temp_avg_c,
            )
        )
    results.sort(key=lambda x: (x.height_dqdv, x.prominence), reverse=True)
    return results


def peak_placeholders(prefix: str) -> Dict[str, float]:
    """Create empty peak fields."""
    return {
        f"{prefix}_peak_voltage_v": np.nan,
        f"{prefix}_peak_width_v": np.nan,
        f"{prefix}_peak_height_dqdv": np.nan,
        f"{prefix}_peak_area": np.nan,
        f"{prefix}_peak_prominence": np.nan,
        f"{prefix}_peak_skewness": np.nan,
        f"{prefix}_peak_temp_max_c": np.nan,
        f"{prefix}_peak_temp_min_c": np.nan,
        f"{prefix}_peak_temp_avg_c": np.nan,
    }


def fill_main_second_peaks(row: Dict[str, object], peaks: List[PeakFeature]) -> None:
    """Fill fixed main/second peak fields."""
    row.update(peak_placeholders("main"))
    row.update(peak_placeholders("second"))
    row["main_second_peak_voltage_gap_v"] = np.nan
    if len(peaks) >= 1:
        p = peaks[0]
        row["main_peak_voltage_v"] = p.voltage_v
        row["main_peak_width_v"] = p.width_v
        row["main_peak_height_dqdv"] = p.height_dqdv
        row["main_peak_area"] = p.area
        row["main_peak_prominence"] = p.prominence
        row["main_peak_skewness"] = p.skewness
        row["main_peak_temp_max_c"] = p.temp_max_c
        row["main_peak_temp_min_c"] = p.temp_min_c
        row["main_peak_temp_avg_c"] = p.temp_avg_c
    if len(peaks) >= 2:
        p = peaks[1]
        row["second_peak_voltage_v"] = p.voltage_v
        row["second_peak_width_v"] = p.width_v
        row["second_peak_height_dqdv"] = p.height_dqdv
        row["second_peak_area"] = p.area
        row["second_peak_prominence"] = p.prominence
        row["second_peak_skewness"] = p.skewness
        row["second_peak_temp_max_c"] = p.temp_max_c
        row["second_peak_temp_min_c"] = p.temp_min_c
        row["second_peak_temp_avg_c"] = p.temp_avg_c
        row["main_second_peak_voltage_gap_v"] = abs(
            float(row["main_peak_voltage_v"]) - float(row["second_peak_voltage_v"])
        )


def feature_columns() -> List[str]:
    """Feature CSV column order."""
    return [
        "policy",
        "cell_code",
        "cycles",
        "state",
        "source_file",
        "curve_method",
        "fit_degree",
        "fit_r2",
        "fit_mae",
        "n_points_raw_group",
        "n_points_qv",
        "n_points_dqdv",
        "temp_max_c",
        "temp_min_c",
        "temp_avg_c",
        "voltage_span_v",
        "capacity_span_ah",
        "peak_count_detected",
        "main_peak_voltage_v",
        "main_peak_width_v",
        "main_peak_height_dqdv",
        "main_peak_area",
        "main_peak_prominence",
        "main_peak_skewness",
        "main_peak_temp_max_c",
        "main_peak_temp_min_c",
        "main_peak_temp_avg_c",
        "second_peak_voltage_v",
        "second_peak_width_v",
        "second_peak_height_dqdv",
        "second_peak_area",
        "second_peak_prominence",
        "second_peak_skewness",
        "second_peak_temp_max_c",
        "second_peak_temp_min_c",
        "second_peak_temp_avg_c",
        "main_second_peak_voltage_gap_v",
        "is_valid_curve",
        "invalid_reason",
    ]


def curve_columns() -> List[str]:
    """Curve points CSV column order."""
    return [
        "policy",
        "cell_code",
        "cycles",
        "state",
        "point_index",
        "voltage",
        "oriented_voltage",
        "capacity_fit",
        "raw_dqdv",
        "smoothed_dqdv",
        "curve_method",
        "fit_degree",
        "fit_r2",
        "fit_mae",
    ]


def base_row(
    key: CycleKey,
    source_file: Path,
    n_raw: int,
    n_qv: int,
    v_span: float,
    q_span: float,
    temp_max: float,
    temp_min: float,
    temp_avg: float,
) -> Dict[str, object]:
    """Create base feature row."""
    return {
        "policy": key.policy,
        "cell_code": key.cell_code,
        "cycles": key.cycles,
        "state": "dischg",
        "source_file": str(source_file),
        "curve_method": CURVE_METHOD,
        "fit_degree": np.nan,
        "fit_r2": np.nan,
        "fit_mae": np.nan,
        "n_points_raw_group": int(n_raw),
        "n_points_qv": int(n_qv),
        "n_points_dqdv": 0,
        "temp_max_c": float(temp_max),
        "temp_min_c": float(temp_min),
        "temp_avg_c": float(temp_avg),
        "voltage_span_v": float(v_span),
        "capacity_span_ah": float(q_span),
        "peak_count_detected": 0,
        "is_valid_curve": False,
        "invalid_reason": "",
    }


def extract_cycle(key: CycleKey, group: pd.DataFrame, source_file: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Extract one cycle's curve points and peak features."""
    ov, vv, cap = prepare_monotone_qv(group)
    temp_vals = pd.to_numeric(group["Temper"], errors="coerce").to_numpy(dtype=float)
    temp_vals = temp_vals[np.isfinite(temp_vals)]
    if len(temp_vals) > 0:
        temp_max = float(np.max(temp_vals))
        temp_min = float(np.min(temp_vals))
        temp_avg = float(np.mean(temp_vals))
    else:
        temp_max = float("nan")
        temp_min = float("nan")
        temp_avg = float("nan")
    row = base_row(
        key=key,
        source_file=source_file,
        n_raw=len(group),
        n_qv=len(ov),
        v_span=float(np.ptp(vv)) if len(vv) else 0.0,
        q_span=float(np.ptp(cap)) if len(cap) else 0.0,
        temp_max=temp_max,
        temp_min=temp_min,
        temp_avg=temp_avg,
    )
    curve_rows: List[Dict[str, object]] = []

    if len(ov) == 0:
        row["invalid_reason"] = "missing_state_segment"
        fill_main_second_peaks(row, [])
        return row, curve_rows

    fit, fit_reason = fit_best_poly(ov, cap)
    if fit is None:
        row["invalid_reason"] = fit_reason
        fill_main_second_peaks(row, [])
        return row, curve_rows

    row["fit_degree"] = int(fit.degree)
    row["fit_r2"] = float(fit.r2)
    row["fit_mae"] = float(fit.mae)

    voltage_mid, oriented_mid, cap_mid, dqdv_raw, dqdv_smooth, curve_reason = build_dqdv(fit, ov)
    if curve_reason:
        row["invalid_reason"] = curve_reason
        fill_main_second_peaks(row, [])
        return row, curve_rows

    row["n_points_dqdv"] = int(len(dqdv_smooth))
    raw_voltage = pd.to_numeric(group["V"], errors="coerce").to_numpy(dtype=float)
    raw_temper = pd.to_numeric(group["Temper"], errors="coerce").to_numpy(dtype=float)
    peaks = detect_peaks(voltage_mid, dqdv_smooth, raw_voltage=raw_voltage, raw_temper=raw_temper)
    row["peak_count_detected"] = int(len(peaks))
    fill_main_second_peaks(row, peaks[:TOP_K_PEAKS])

    if len(peaks) == 0:
        row["invalid_reason"] = "no_detected_peak"
        return row, curve_rows

    row["is_valid_curve"] = True
    row["invalid_reason"] = ""

    for idx in range(len(dqdv_smooth)):
        curve_rows.append(
            {
                "policy": key.policy,
                "cell_code": key.cell_code,
                "cycles": key.cycles,
                "state": "dischg",
                "point_index": int(idx),
                "voltage": float(voltage_mid[idx]),
                "oriented_voltage": float(oriented_mid[idx]),
                "capacity_fit": float(cap_mid[idx]),
                "raw_dqdv": float(dqdv_raw[idx]),
                "smoothed_dqdv": float(dqdv_smooth[idx]),
                "curve_method": CURVE_METHOD,
                "fit_degree": int(fit.degree),
                "fit_r2": float(fit.r2),
                "fit_mae": float(fit.mae),
            }
        )
    return row, curve_rows


def run_for_file(file_path: Path, selected_keys: Optional[Set[Tuple[str, str, int]]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int]:
    """Run extraction for one file with optional cycle filter."""
    frame = load_discharge_frame(file_path)
    if frame.empty:
        return [], [], 0
    frows: List[Dict[str, object]] = []
    crows: List[Dict[str, object]] = []
    n_cycles = 0
    for (policy, cell_code, cycles), group in frame.groupby(["policy", "cell_code", "cycles"], sort=False):
        key = CycleKey(str(policy), str(cell_code), int(cycles))
        if selected_keys is not None and key.as_tuple() not in selected_keys:
            continue
        n_cycles += 1
        fr, cr = extract_cycle(key, group, file_path.resolve())
        frows.append(fr)
        crows.extend(cr)
    return frows, crows, n_cycles


def round_to_decimals(value: object, decimals: int) -> object:
    """Round one numeric value to fixed decimal places."""
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(x):
        return x
    return float(np.round(x, decimals))


def save_features(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Save cycle-level features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=feature_columns())
    if not df.empty:
        df = df.sort_values(["policy", "cell_code", "cycles"]).reset_index(drop=True)
        for col in PEAK_SIGFIG_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(lambda v: round_to_decimals(v, 3))
        for col in TEMP_SIGFIG_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(lambda v: round_to_decimals(v, 2))
    df.to_csv(output_path, index=False, encoding="utf-8")


def save_curve_points(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Save point-level curve table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=curve_columns())
    else:
        df = df.reindex(columns=curve_columns())
        df = df.sort_values(["policy", "cell_code", "cycles", "point_index"]).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def build_summary(mode: str, selected_cycle_count: int, selected_file_count: int, feature_rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Build JSON summary payload."""
    df = pd.DataFrame(feature_rows)
    if df.empty:
        valid = 0
        invalid_counts: Dict[str, int] = {}
    else:
        valid = int(df["is_valid_curve"].astype(bool).sum())
        invalid = df.loc[~df["is_valid_curve"].astype(bool), "invalid_reason"].fillna("")
        invalid_counts = invalid[invalid != ""].value_counts().astype(int).to_dict()
    return {
        "mode": mode,
        "curve_method": CURVE_METHOD,
        "poly_degree_min": POLY_DEGREE_MIN,
        "poly_degree_max": POLY_DEGREE_MAX,
        "voltage_step_v": VOLTAGE_STEP_V,
        "post_smooth_window": POST_SMOOTH_WINDOW,
        "top_k_peaks": TOP_K_PEAKS,
        "selected_cycle_count": int(selected_cycle_count),
        "selected_file_count": int(selected_file_count),
        "output_cycle_rows": int(len(feature_rows)),
        "valid_curve_rows": int(valid),
        "invalid_curve_rows": int(len(feature_rows) - valid),
        "invalid_reason_counts": invalid_counts,
    }


def save_summary(payload: Dict[str, object], output_path: Path) -> None:
    """Save summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_outputs(args: argparse.Namespace, mode: str) -> Tuple[Path, Optional[Path], Path, bool]:
    """Resolve output paths and curve-export switch."""
    if mode in {"sample10", "smoke"}:
        fout = Path(args.output_features) if args.output_features else SAMPLE_FEATURE_OUTPUT_PATH
        cout = Path(args.output_curve_points) if args.output_curve_points else SAMPLE_CURVE_OUTPUT_PATH
        sout = Path(args.output_summary) if args.output_summary else SAMPLE_SUMMARY_OUTPUT_PATH
        default_curve = True
    else:
        fout = Path(args.output_features) if args.output_features else FULL_FEATURE_OUTPUT_PATH
        cout = Path(args.output_curve_points) if args.output_curve_points else None
        sout = Path(args.output_summary) if args.output_summary else FULL_SUMMARY_OUTPUT_PATH
        default_curve = False
    export_curve = default_curve if args.export_curve_points is None else bool(args.export_curve_points)
    if not export_curve:
        cout = None
    return fout, cout, sout, export_curve


def main() -> None:
    """Main entry for extraction."""
    args = parse_args()
    mode = "sample10" if args.mode == "smoke" else args.mode
    feature_out, curve_out, summary_out, export_curve = resolve_outputs(args, mode)

    if mode in {"sample10", "smoke"}:
        keys = stratified_sample_keys(collect_sampling_keys(), args.sample_count, args.seed)
        selected_keys = {(str(r.policy), str(r.cell_code), int(r.cycles)) for r in keys.itertuples(index=False)}
        target_files = [Path(p) for p in sorted(set(keys["source_file"].tolist()))]
        selected_cycle_count = len(keys)
        print(f"Mode: {mode} | sample_count={selected_cycle_count} | seed={args.seed}")
    else:
        selected_keys = None
        target_files = find_cycles_files()
        selected_cycle_count = 0
        print(f"Mode: full | files={len(target_files)}")

    all_features: List[Dict[str, object]] = []
    all_curves: List[Dict[str, object]] = []
    processed_cycles = 0
    for fpath in target_files:
        frows, crows, ncycles = run_for_file(fpath, selected_keys)
        all_features.extend(frows)
        if export_curve:
            all_curves.extend(crows)
        processed_cycles += ncycles

    if mode == "full":
        selected_cycle_count = processed_cycles

    save_features(all_features, feature_out)
    if export_curve and curve_out is not None:
        save_curve_points(all_curves, curve_out)
    summary = build_summary(mode, selected_cycle_count, len(target_files), all_features)
    save_summary(summary, summary_out)

    print(f"Processed cycles: {processed_cycles}")
    print(f"Feature rows: {len(all_features)}")
    print(f"Valid curves: {summary['valid_curve_rows']}")
    print(f"Invalid curves: {summary['invalid_curve_rows']}")
    print(f"Saved features: {feature_out}")
    if export_curve and curve_out is not None:
        print(f"Saved curve points: {curve_out}")
    else:
        print("Curve point export: disabled")
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()
