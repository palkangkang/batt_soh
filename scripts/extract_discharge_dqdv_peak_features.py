from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter


# =========================
# Config (edit here first)
# =========================
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
FULL_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_peak_features.csv"
SMOKE_OUTPUT_PATH = OUTPUT_DIR / "discharge_dqdv_peak_features_smoke.csv"

CYCLES_FILE_GLOB = "cycles_*.csv"
RECURSIVE_SEARCH = True
ENCODING = "utf-8-sig"

# Fixed smoke-test reproducibility.
DEFAULT_RANDOM_SEED = 20260410
DEFAULT_SMOKE_FILE_COUNT = 5

# Discharge voltage analysis window (configurable).
DISCHARGE_VOLTAGE_HIGH = 3.6
DISCHARGE_VOLTAGE_LOW = 2.8

# Strict cycle validity checks.
MIN_WINDOW_POINTS = 50
MIN_DQDV_POINTS = 10

# Numeric safety guards.
FLOAT_EPS = 1e-9
MIN_ABS_DV_FOR_DERIVATIVE = 1e-4
MIN_POSITIVE_DT = 1e-9

# SciPy smoothing and peak detection.
ENABLE_SAVGOL = True
SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLYORDER = 3

# Optional second-pass smoothing for "balanced dQ/dV".
# Keep default False so existing full-output behavior stays unchanged
# until explicitly enabled.
ENABLE_BALANCED_POST_SMOOTH = False
BALANCED_SAVGOL_WINDOW = 15
BALANCED_SAVGOL_POLYORDER = 3

PEAK_MIN_DISTANCE_POINTS = 5
PEAK_MIN_HEIGHT = 0.0
PEAK_PROMINENCE_ABS_MIN = 0.003
PEAK_PROMINENCE_REL_Q90 = 0.15
TOP_K_PEAKS = 3


@dataclass
class PeakFeature:
    """Container for one detected dQ/dV peak."""

    voltage_v: float
    height_dqdv: float
    area: float
    prominence: float
    width_v: float


@dataclass
class FileProcessSummary:
    """Per-file extraction summary used for smoke-test reporting."""

    file_path: Path
    grouped_cycles: int
    valid_window_cycles: int
    valid_dqdv_cycles: int
    peak_cycles: int
    output_rows: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for smoke/full execution modes."""
    parser = argparse.ArgumentParser(
        description="Extract discharge dQ/dV peak statistics by policy+cell_code+cycles."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Execution mode. smoke: random subset files. full: all files.",
    )
    parser.add_argument(
        "--smoke-file-count",
        type=int,
        default=DEFAULT_SMOKE_FILE_COUNT,
        help="Number of randomly sampled files in smoke mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used in smoke mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path override.",
    )
    return parser.parse_args()


def find_target_files() -> List[Path]:
    """Find all cycles CSV files under RAW_DIR according to search settings."""
    if RECURSIVE_SEARCH:
        files = sorted(RAW_DIR.rglob(CYCLES_FILE_GLOB))
    else:
        files = sorted(RAW_DIR.glob(CYCLES_FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched {CYCLES_FILE_GLOB!r} under {RAW_DIR}")
    return files


def sample_files(files: Sequence[Path], count: int, seed: int) -> List[Path]:
    """Sample a reproducible subset of files for smoke testing."""
    if count <= 0:
        raise ValueError("smoke-file-count must be > 0")
    if count >= len(files):
        return list(files)
    rng = Random(seed)
    selected = rng.sample(list(files), count)
    selected.sort()
    return selected


def load_discharge_frame(file_path: Path) -> pd.DataFrame:
    """Load one raw file and keep only discharge rows in the configured voltage window."""
    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "ts",
        "V",
        "ah_dischg",
        "flag_dischg",
    ]
    df = pd.read_csv(file_path, usecols=usecols, encoding=ENCODING, low_memory=False)

    required = set(usecols)
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {file_path}: {sorted(missing)}")

    for col in ["cycles", "ts", "V", "ah_dischg", "flag_dischg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["policy", "cell_code", "cycles", "ts", "V", "ah_dischg", "flag_dischg"]).copy()
    if df.empty:
        return df

    df = df.loc[df["flag_dischg"].astype(int) == 1].copy()
    if df.empty:
        return df

    df = df.loc[
        (df["V"] <= DISCHARGE_VOLTAGE_HIGH + FLOAT_EPS)
        & (df["V"] >= DISCHARGE_VOLTAGE_LOW - FLOAT_EPS)
    ].copy()
    if df.empty:
        return df

    df["cycles"] = df["cycles"].astype(int)
    df = df.sort_values(["policy", "cell_code", "cycles", "ts"]).reset_index(drop=True)
    return df


def _safe_savgol_window(n: int, target_window: int, polyorder: int) -> int:
    """Build a valid odd Savitzky-Golay window length for a series length n."""
    if n <= 0:
        return 0
    win = min(target_window, n)
    if win % 2 == 0:
        win -= 1
    if win <= polyorder:
        win = polyorder + 2
        if win % 2 == 0:
            win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win <= polyorder or win <= 2:
        return 0
    return win


def apply_balanced_post_smooth(dqdv: np.ndarray, force: bool = False) -> np.ndarray:
    """
    Apply optional second-pass Savitzky-Golay smoothing.

    Args:
        dqdv: Input dQ/dV series.
        force: Force-enable balanced smoothing regardless of global config.
    """
    enabled = ENABLE_BALANCED_POST_SMOOTH or force
    if not enabled:
        return np.array(dqdv, copy=True)

    win = _safe_savgol_window(len(dqdv), BALANCED_SAVGOL_WINDOW, BALANCED_SAVGOL_POLYORDER)
    if win <= 0:
        return np.array(dqdv, copy=True)

    return savgol_filter(
        np.asarray(dqdv, dtype=float),
        window_length=win,
        polyorder=BALANCED_SAVGOL_POLYORDER,
        mode="interp",
    )


def build_dqdv_series(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build discharge dQ/dV series from one cycle group.

    Returns:
        voltage_mid: voltage axis (descending).
        dqdv: signed dQ/dV values corresponding to voltage_mid.
        n_window_points: row count in the configured voltage window.
        n_dqdv_points: valid derivative point count.
    """
    n_window_points = int(len(group))
    if n_window_points < MIN_WINDOW_POINTS:
        return np.array([]), np.array([]), n_window_points, 0

    v = group["V"].to_numpy(dtype=float)
    q = group["ah_dischg"].to_numpy(dtype=float)
    t = group["ts"].to_numpy(dtype=float)

    dv = np.diff(v)
    dq = np.diff(q)
    dt = np.diff(t)

    valid = (
        np.isfinite(dv)
        & np.isfinite(dq)
        & np.isfinite(dt)
        & (dt > MIN_POSITIVE_DT)
        & (dv < -MIN_ABS_DV_FOR_DERIVATIVE)
        & (dq >= -FLOAT_EPS)
    )

    if not np.any(valid):
        return np.array([]), np.array([]), n_window_points, 0

    v_mid = 0.5 * (v[1:] + v[:-1])
    v_mid = v_mid[valid]
    dqdv = dq[valid] / dv[valid]

    good = np.isfinite(v_mid) & np.isfinite(dqdv)
    v_mid = v_mid[good]
    dqdv = dqdv[good]

    if len(v_mid) == 0:
        return np.array([]), np.array([]), n_window_points, 0

    s = pd.DataFrame({"Vmid": v_mid, "dqdv": dqdv})
    s = s.groupby("Vmid", as_index=False)["dqdv"].mean()
    s = s.sort_values("Vmid", ascending=False).reset_index(drop=True)

    n_dqdv_points = int(len(s))
    if n_dqdv_points < MIN_DQDV_POINTS:
        return np.array([]), np.array([]), n_window_points, n_dqdv_points

    y = s["dqdv"].to_numpy(dtype=float)
    x = s["Vmid"].to_numpy(dtype=float)

    if ENABLE_SAVGOL:
        win = _safe_savgol_window(len(y), SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER)
        if win > 0:
            y = savgol_filter(y, window_length=win, polyorder=SAVGOL_POLYORDER, mode="interp")

    return x, y, n_window_points, n_dqdv_points


def _index_interp_to_voltage(voltage: np.ndarray, floating_index: float) -> float:
    """Interpolate voltage value at a floating sample index."""
    idx_axis = np.arange(len(voltage), dtype=float)
    return float(np.interp(floating_index, idx_axis, voltage))


def detect_peaks(voltage: np.ndarray, dqdv: np.ndarray) -> List[PeakFeature]:
    """
    Detect dominant discharge peaks and return top-K peak statistics.

    Peaks are searched on -dQ/dV so that discharge valleys become positive peaks.
    Returned peak height preserves original dQ/dV sign.
    """
    if len(voltage) < 3 or len(dqdv) < 3:
        return []

    search_signal = -dqdv
    if not np.any(np.isfinite(search_signal)):
        return []

    q90 = float(np.nanpercentile(search_signal, 90))
    dynamic_prominence = max(PEAK_PROMINENCE_ABS_MIN, q90 * PEAK_PROMINENCE_REL_Q90)

    peaks, props = find_peaks(
        search_signal,
        prominence=dynamic_prominence,
        distance=PEAK_MIN_DISTANCE_POINTS,
        height=PEAK_MIN_HEIGHT,
    )
    if len(peaks) == 0:
        return []

    widths = peak_widths(search_signal, peaks, rel_height=0.5)
    left_ips = widths[2]
    right_ips = widths[3]

    candidates: List[Tuple[float, PeakFeature]] = []
    for i, peak_idx in enumerate(peaks):
        left_base = int(props["left_bases"][i])
        right_base = int(props["right_bases"][i])
        if right_base <= left_base:
            continue

        seg_v = voltage[left_base : right_base + 1]
        seg_y = dqdv[left_base : right_base + 1]
        if len(seg_v) < 2:
            continue
        sort_idx = np.argsort(seg_v)
        if hasattr(np, "trapezoid"):
            area = float(np.trapezoid(seg_y[sort_idx], seg_v[sort_idx]))
        else:
            area = float(np.trapz(seg_y[sort_idx], seg_v[sort_idx]))

        v_left = _index_interp_to_voltage(voltage, left_ips[i])
        v_right = _index_interp_to_voltage(voltage, right_ips[i])
        width_v = abs(v_right - v_left)

        prominence = float(props["prominences"][i])
        peak = PeakFeature(
            voltage_v=float(voltage[peak_idx]),
            height_dqdv=float(dqdv[peak_idx]),
            area=area,
            prominence=prominence,
            width_v=width_v,
        )
        candidates.append((prominence, peak))

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[0], abs(x[1].height_dqdv)), reverse=True)
    return [item[1] for item in candidates[:TOP_K_PEAKS]]


def make_empty_peak_fields() -> Dict[str, float]:
    """Build empty placeholder fields for fixed-width top-K peak outputs."""
    fields: Dict[str, float] = {}
    for rank in range(1, TOP_K_PEAKS + 1):
        fields[f"peak{rank}_voltage_v"] = np.nan
        fields[f"peak{rank}_height_dqdv"] = np.nan
        fields[f"peak{rank}_area"] = np.nan
        fields[f"peak{rank}_prominence"] = np.nan
        fields[f"peak{rank}_width_v"] = np.nan
    return fields


def extract_features_from_file(file_path: Path) -> Tuple[List[dict], FileProcessSummary]:
    """Extract one-row-per-cycle dQ/dV peak statistics from one input file."""
    frame = load_discharge_frame(file_path)
    if frame.empty:
        summary = FileProcessSummary(
            file_path=file_path,
            grouped_cycles=0,
            valid_window_cycles=0,
            valid_dqdv_cycles=0,
            peak_cycles=0,
            output_rows=0,
        )
        return [], summary

    group_cols = ["policy", "cell_code", "cycles"]
    rows: List[dict] = []

    grouped_cycles = 0
    valid_window_cycles = 0
    valid_dqdv_cycles = 0
    peak_cycles = 0

    for (policy, cell_code, cycles), group in frame.groupby(group_cols, sort=False):
        grouped_cycles += 1
        x, y, n_window, n_dqdv = build_dqdv_series(group)
        if n_window >= MIN_WINDOW_POINTS:
            valid_window_cycles += 1
        if n_dqdv >= MIN_DQDV_POINTS:
            valid_dqdv_cycles += 1
        if len(x) == 0 or len(y) == 0:
            continue

        peak_input = apply_balanced_post_smooth(y)
        peaks = detect_peaks(x, peak_input)
        if not peaks:
            continue

        peak_cycles += 1
        row = {
            "policy": policy,
            "cell_code": cell_code,
            "cycles": int(cycles),
            "n_points_window": int(n_window),
            "n_points_dqdv": int(n_dqdv),
            "n_peaks_detected": int(len(peaks)),
        }
        row.update(make_empty_peak_fields())

        for rank, peak in enumerate(peaks, start=1):
            row[f"peak{rank}_voltage_v"] = peak.voltage_v
            row[f"peak{rank}_height_dqdv"] = peak.height_dqdv
            row[f"peak{rank}_area"] = peak.area
            row[f"peak{rank}_prominence"] = peak.prominence
            row[f"peak{rank}_width_v"] = peak.width_v
        rows.append(row)

    summary = FileProcessSummary(
        file_path=file_path,
        grouped_cycles=grouped_cycles,
        valid_window_cycles=valid_window_cycles,
        valid_dqdv_cycles=valid_dqdv_cycles,
        peak_cycles=peak_cycles,
        output_rows=len(rows),
    )
    return rows, summary


def deduplicate_rows(rows: List[dict]) -> List[dict]:
    """Ensure one row per policy+cell_code+cycles by deterministic best-row keep."""
    if not rows:
        return rows

    df = pd.DataFrame(rows)
    sort_cols = ["n_peaks_detected", "n_points_dqdv", "n_points_window"]
    df = df.sort_values(sort_cols, ascending=[False, False, False])
    df = df.drop_duplicates(subset=["policy", "cell_code", "cycles"], keep="first")
    df = df.sort_values(["policy", "cell_code", "cycles"]).reset_index(drop=True)
    return df.to_dict(orient="records")


def build_output_fieldnames() -> List[str]:
    """Build ordered CSV schema for output."""
    fields = [
        "policy",
        "cell_code",
        "cycles",
        "n_points_window",
        "n_points_dqdv",
        "n_peaks_detected",
    ]
    for rank in range(1, TOP_K_PEAKS + 1):
        fields.extend(
            [
                f"peak{rank}_voltage_v",
                f"peak{rank}_height_dqdv",
                f"peak{rank}_area",
                f"peak{rank}_prominence",
                f"peak{rank}_width_v",
            ]
        )
    return fields


def save_rows(rows: List[dict], output_path: Path) -> None:
    """Save extracted rows to CSV with stable column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=build_output_fieldnames())
    df.to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    """Entry point for dQ/dV peak extraction."""
    args = parse_args()
    all_files = find_target_files()

    if args.mode == "smoke":
        selected_files = sample_files(all_files, args.smoke_file_count, args.seed)
        output_path = Path(args.output) if args.output else SMOKE_OUTPUT_PATH
    else:
        selected_files = all_files
        output_path = Path(args.output) if args.output else FULL_OUTPUT_PATH

    print(f"Mode: {args.mode}")
    print(f"Input files selected: {len(selected_files)} / {len(all_files)}")
    if args.mode == "smoke":
        print(f"Smoke seed: {args.seed}")

    all_rows: List[dict] = []
    summaries: List[FileProcessSummary] = []
    for fp in selected_files:
        rows, summary = extract_features_from_file(fp)
        all_rows.extend(rows)
        summaries.append(summary)

    all_rows = deduplicate_rows(all_rows)
    save_rows(all_rows, output_path)

    print(f"Output rows: {len(all_rows)}")
    print(f"Saved to: {output_path}")
    print("Per-file summary:")
    for s in summaries:
        print(
            " | ".join(
                [
                    f"file={s.file_path}",
                    f"grouped_cycles={s.grouped_cycles}",
                    f"valid_window_cycles={s.valid_window_cycles}",
                    f"valid_dqdv_cycles={s.valid_dqdv_cycles}",
                    f"peak_cycles={s.peak_cycles}",
                    f"output_rows={s.output_rows}",
                ]
            )
        )


if __name__ == "__main__":
    main()
