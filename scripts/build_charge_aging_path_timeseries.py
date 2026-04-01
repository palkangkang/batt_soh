from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SOC_LABEL_MAP: Dict[int, str] = {
    1: "[0,10)",
    2: "[10,90)",
    3: "[90,100]",
}

RATE_QUANTILES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
TEMP_QUANTILES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

ANOMALY_DT_SMALL_THRESHOLD_S = 10.0
ANOMALY_DT_MEDIUM_THRESHOLD_S = 600.0
ANOMALY_DT_CELL_THRESHOLD_S = 3600.0
TEMP_VALID_MIN_C = 20.0
TEMP_VALID_MAX_C = 60.0

QUANTILE_SAMPLE_PER_FILE = 20000
QUANTILE_SAMPLE_SEED = 20260331

USECOLS_CYCLES = ["policy", "cell_code", "cycles", "ts", "I", "Temper", "soc", "flag_chg"]
USECOLS_LIFE = ["policy", "cell_code", "cycles", "q_discharge"]


@dataclass(frozen=True)
class CellKey:
    """Unique identifier of one cell under one policy."""

    policy: str
    cell_code: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build charging aging-path timeseries with 60 cross bins (soc x rate x temp)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Directory that contains cycles_*.csv under subfolders.",
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=repo_root / "data" / "processed" / "life_performance.csv",
        help="Path to life_performance.csv.",
    )
    parser.add_argument(
        "--out-timeseries",
        type=Path,
        default=repo_root / "data" / "processed" / "charge_aging_path_timeseries.csv",
        help="Output path of cycle-level timeseries table.",
    )
    parser.add_argument(
        "--out-final",
        type=Path,
        default=repo_root / "data" / "processed" / "charge_aging_path_final.csv",
        help="Output path of final cumulative summary table.",
    )
    parser.add_argument(
        "--out-bin-edges",
        type=Path,
        default=repo_root / "data" / "processed" / "charge_aging_path_bin_edges.csv",
        help="Output path of 60-bin edge mapping table.",
    )
    parser.add_argument(
        "--out-anomalies",
        type=Path,
        default=repo_root / "data" / "processed" / "charge_aging_path_ts_anomalies.csv",
        help="Output path of interval-level ts anomaly table.",
    )
    parser.add_argument(
        "--out-abnormal-cells",
        type=Path,
        default=repo_root / "data" / "processed" / "charge_aging_path_abnormal_cells.csv",
        help="Output path of abnormal-cell summary table (dt > 3600s).",
    )
    parser.add_argument(
        "--out-abnormal-timeseries",
        type=Path,
        default=repo_root / "data" / "processed" / "charge_aging_path_timeseries_abnormal_cells.csv",
        help="Output path of abnormal-cell timeseries subset.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding for raw files.",
    )
    parser.add_argument(
        "--time-sig-digits",
        type=int,
        default=1,
        help="Significant digits kept for time columns when saving files.",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=0.3,
        help="Minimum valid q_discharge used for life filtering before base/cycle derivation.",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=1.3,
        help="Maximum valid q_discharge used for life filtering before base/cycle derivation.",
    )
    return parser.parse_args()


def format_sig3(value: float) -> str:
    """Format one numeric value as three significant digits text."""

    return format(float(value), ".3g")


def ensure_monotonic_edges(edges: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure quantile edges are strictly increasing by applying a tiny epsilon shift."""

    out = edges.astype(float).copy()
    for idx in range(1, len(out)):
        if out[idx] <= out[idx - 1]:
            out[idx] = out[idx - 1] + eps
    return out


def round_to_sig(values: pd.Series, digits: int) -> pd.Series:
    """Round numeric series to target significant digits."""

    if digits <= 0:
        raise ValueError("digits must be > 0")

    arr = values.to_numpy(dtype=float, copy=True)
    mask = np.isfinite(arr) & (arr != 0.0)
    if np.any(mask):
        scales = np.power(10.0, digits - 1 - np.floor(np.log10(np.abs(arr[mask]))))
        arr[mask] = np.round(arr[mask] * scales) / scales
    return pd.Series(arr, index=values.index)


def apply_time_sig_digits(df: pd.DataFrame, time_cols: List[str], digits: int) -> pd.DataFrame:
    """Apply significant-digit rounding for selected time columns."""

    out = df.copy()
    for col in time_cols:
        if col in out.columns:
            out[col] = round_to_sig(out[col], digits=digits)
    return out


def assign_soc_bin(soc_mid_percent: np.ndarray) -> np.ndarray:
    """Assign SOC bins (1..3) according to [0,10), [10,90), [90,100]."""

    return np.select(
        [
            (soc_mid_percent >= 0.0) & (soc_mid_percent < 10.0),
            (soc_mid_percent >= 10.0) & (soc_mid_percent < 90.0),
            (soc_mid_percent >= 90.0) & (soc_mid_percent <= 100.0),
        ],
        [1, 2, 3],
        default=0,
    ).astype(np.int16)


def assign_numeric_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values into quantile bins using monotonically increasing edges."""

    inner = edges[1:-1]
    bins = np.searchsorted(inner, values, side="right") + 1
    return bins.astype(np.int16)


def build_cross_bin(soc_bin: np.ndarray, rate_bin: np.ndarray, temp_bin: np.ndarray) -> np.ndarray:
    """Build cross bin index 1..60 by soc x rate x temp."""

    return ((soc_bin - 1) * 20 + (rate_bin - 1) * 5 + temp_bin).astype(np.int16)


def scan_cycles_files(raw_dir: Path) -> Dict[CellKey, Path]:
    """Scan all cycles_*.csv files and map each file to one (policy, cell_code) key."""

    file_map: Dict[CellKey, Path] = {}
    for file_path in sorted(raw_dir.rglob("cycles_*.csv")):
        try:
            first = pd.read_csv(
                file_path,
                usecols=["policy", "cell_code"],
                nrows=1,
                encoding="utf-8-sig",
            )
        except Exception:
            continue
        if first.empty:
            continue
        policy = str(first.at[0, "policy"])
        cell_code = str(first.at[0, "cell_code"])
        file_map[CellKey(policy=policy, cell_code=cell_code)] = file_path
    return file_map


def load_life_tables(
    life_path: Path,
    q_min: float,
    q_max: float,
) -> Tuple[pd.DataFrame, Dict[CellKey, float], Dict[CellKey, List[int]]]:
    """Load life_performance and derive base capacity and cycle list per cell."""

    life = pd.read_csv(life_path, usecols=USECOLS_LIFE, encoding="utf-8")
    life["cell_code"] = life["cell_code"].astype(str)
    life["cycles"] = life["cycles"].astype(int)
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    life = life.dropna(subset=["q_discharge"])
    life = life[
        (life["q_discharge"] >= float(q_min))
        & (life["q_discharge"] <= float(q_max))
    ].copy()

    base = (
        life[life["cycles"] <= 100]
        .groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .mean()
        .rename(columns={"q_discharge": "base_q_discharge_100"})
    )

    base_map: Dict[CellKey, float] = {}
    for row in base.itertuples(index=False):
        key = CellKey(policy=str(row.policy), cell_code=str(row.cell_code))
        val = float(row.base_q_discharge_100)
        if math.isfinite(val) and val > 0:
            base_map[key] = val

    cycle_map: Dict[CellKey, List[int]] = {}
    for (policy, cell_code), part in life.groupby(["policy", "cell_code"], sort=True):
        key = CellKey(policy=str(policy), cell_code=str(cell_code))
        cycles = sorted(int(x) for x in part["cycles"].dropna().unique().tolist())
        cycle_map[key] = cycles

    return life, base_map, cycle_map


def extract_intervals(
    file_path: Path,
    key: CellKey,
    base_q: float,
    encoding: str,
) -> Tuple[pd.DataFrame, int]:
    """Extract charging intervals and return (interval_df, cycles_with_all_temp_abnormal)."""

    if base_q <= 0:
        return pd.DataFrame(), 0

    try:
        df = pd.read_csv(file_path, usecols=USECOLS_CYCLES, encoding=encoding)
    except Exception:
        return pd.DataFrame(), 0

    if df.empty:
        return pd.DataFrame(), 0

    df["cell_code"] = df["cell_code"].astype(str)
    df = df[(df["policy"] == key.policy) & (df["cell_code"] == key.cell_code)]
    df = df[df["flag_chg"] == 1].copy()
    if df.empty:
        return pd.DataFrame(), 0

    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["I"] = pd.to_numeric(df["I"], errors="coerce")
    df["Temper"] = pd.to_numeric(df["Temper"], errors="coerce")
    df["soc"] = pd.to_numeric(df["soc"], errors="coerce")
    df = df.dropna(subset=["cycles", "ts", "I", "Temper", "soc"])
    if df.empty:
        return pd.DataFrame(), 0

    df["cycles"] = df["cycles"].astype(int)
    df = df.sort_values(["cycles", "ts"], kind="mergesort").reset_index(drop=True)

    # Replace abnormal temperature by previous non-abnormal value in-cycle,
    # and backfill only for leading abnormal runs.
    df["Temper_clean"] = df["Temper"].where(
        (df["Temper"] >= TEMP_VALID_MIN_C) & (df["Temper"] <= TEMP_VALID_MAX_C),
        np.nan,
    )
    all_temp_abnormal_cycles = int(
        df.groupby("cycles")["Temper_clean"].apply(lambda s: s.isna().all()).sum()
    )
    df["Temper_clean"] = df.groupby("cycles")["Temper_clean"].transform(lambda s: s.ffill().bfill())

    for col in ["ts", "I", "soc", "Temper_clean"]:
        df[f"{col}_prev"] = df.groupby("cycles")[col].shift(1)

    iv = df.dropna(subset=["ts_prev", "I_prev", "soc_prev", "Temper_clean", "Temper_clean_prev"]).copy()
    if iv.empty:
        return pd.DataFrame(), all_temp_abnormal_cycles

    iv["i_mid"] = (iv["I"] + iv["I_prev"]) / 2.0
    iv = iv[iv["i_mid"] > 0.0].copy()
    if iv.empty:
        return pd.DataFrame(), all_temp_abnormal_cycles

    iv["dt_s"] = iv["ts"] - iv["ts_prev"]
    iv["soc_mid_percent"] = ((iv["soc"] + iv["soc_prev"]) / 2.0) * 100.0
    iv["temp_mid_c"] = (iv["Temper_clean"] + iv["Temper_clean_prev"]) / 2.0
    iv["c_rate_mid"] = iv["i_mid"] / base_q
    iv["soc_bin"] = assign_soc_bin(iv["soc_mid_percent"].to_numpy())
    iv = iv[iv["soc_bin"] > 0].copy()
    if iv.empty:
        return pd.DataFrame(), all_temp_abnormal_cycles

    iv["ts_anomaly_reason"] = np.select(
        [
            iv["dt_s"] <= 0.0,
            iv["dt_s"] > ANOMALY_DT_SMALL_THRESHOLD_S,
        ],
        [
            "non_positive_dt",
            f"large_dt_gt_{int(ANOMALY_DT_SMALL_THRESHOLD_S)}s",
        ],
        default="",
    )

    out = iv[
        [
            "policy",
            "cell_code",
            "cycles",
            "ts_prev",
            "ts",
            "dt_s",
            "soc_bin",
            "soc_mid_percent",
            "c_rate_mid",
            "temp_mid_c",
            "ts_anomaly_reason",
        ]
    ].copy()
    return out, all_temp_abnormal_cycles


def sample_from_intervals(
    interval_df: pd.DataFrame,
    max_n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample c_rate and temp values from one interval dataframe."""

    if interval_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_rows = len(interval_df)
    if n_rows <= max_n:
        part = interval_df
    else:
        idx = rng.choice(n_rows, size=max_n, replace=False)
        part = interval_df.iloc[idx]
    return part["c_rate_mid"].to_numpy(dtype=float), part["temp_mid_c"].to_numpy(dtype=float)


def compute_quantile_edges(
    keys: Iterable[CellKey],
    file_map: Dict[CellKey, Path],
    base_map: Dict[CellKey, float],
    encoding: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global rate/temp quantile edges from sampled intervals across all cells."""

    rng = np.random.default_rng(QUANTILE_SAMPLE_SEED)
    rate_samples: List[np.ndarray] = []
    temp_samples: List[np.ndarray] = []

    for idx, key in enumerate(sorted(keys, key=lambda x: (x.policy, x.cell_code))):
        if key not in file_map or key not in base_map:
            continue
        interval_df, _ = extract_intervals(file_map[key], key, base_map[key], encoding)
        rate_part, temp_part = sample_from_intervals(
            interval_df=interval_df,
            max_n=QUANTILE_SAMPLE_PER_FILE,
            rng=rng,
        )
        if rate_part.size > 0:
            rate_samples.append(rate_part)
            temp_samples.append(temp_part)
        if (idx + 1) % 20 == 0:
            print(f"[pass1] processed {idx + 1} cells")

    if not rate_samples:
        raise RuntimeError("No valid interval sample found for quantile computation.")

    all_rate = np.concatenate(rate_samples)
    all_temp = np.concatenate(temp_samples)
    rate_edges = ensure_monotonic_edges(np.quantile(all_rate, RATE_QUANTILES))
    rate_edges[0] = 0.0
    temp_edges = ensure_monotonic_edges(np.quantile(all_temp, TEMP_QUANTILES))
    return rate_edges, temp_edges


def build_dim_table(rate_edges: np.ndarray, temp_edges: np.ndarray) -> pd.DataFrame:
    """Build 60-bin dimension table with raw and display edges."""

    rows: List[dict] = []
    for soc_bin in [1, 2, 3]:
        soc_label = SOC_LABEL_MAP[soc_bin]
        for rate_bin in [1, 2, 3, 4]:
            rate_lo = float(rate_edges[rate_bin - 1])
            rate_hi = float(rate_edges[rate_bin])
            if rate_bin == 1 and abs(rate_lo) < 1e-9:
                rate_lo = 0.0
            rate_label = f"[{format_sig3(rate_lo)},{format_sig3(rate_hi)}{']' if rate_bin == 4 else ')'}"
            for temp_bin in [1, 2, 3, 4, 5]:
                temp_lo = float(temp_edges[temp_bin - 1])
                temp_hi = float(temp_edges[temp_bin])
                temp_lo_int = int(round(temp_lo))
                temp_hi_int = int(round(temp_hi))
                if temp_hi_int <= temp_lo_int:
                    temp_hi_int = temp_lo_int + 1
                temp_label = f"[{temp_lo_int},{temp_hi_int}{']' if temp_bin == 5 else ')'}"
                cross_bin = (soc_bin - 1) * 20 + (rate_bin - 1) * 5 + temp_bin
                rows.append(
                    {
                        "soc_bin": soc_bin,
                        "rate_bin": rate_bin,
                        "temp_bin": temp_bin,
                        "cross_bin": cross_bin,
                        "soc_label": soc_label,
                        "rate_label": rate_label,
                        "temp_label": temp_label,
                        "cross_label": f"s{soc_bin}_r{rate_bin}_t{temp_bin}",
                        "rate_edge_low_raw": rate_lo,
                        "rate_edge_high_raw": rate_hi,
                        "temp_edge_low_raw": temp_lo,
                        "temp_edge_high_raw": temp_hi,
                        "temp_edge_low_int": temp_lo_int,
                        "temp_edge_high_int": temp_hi_int,
                    }
                )
    dim = pd.DataFrame(rows)
    dim = dim.sort_values(["cross_bin"]).reset_index(drop=True)
    return dim


def summarize_abnormal_cell(interval_df: pd.DataFrame, key: CellKey) -> dict:
    """Summarize abnormal statistics for one cell."""

    gt600 = interval_df[interval_df["dt_s"] > ANOMALY_DT_MEDIUM_THRESHOLD_S]
    gt3600 = interval_df[interval_df["dt_s"] > ANOMALY_DT_CELL_THRESHOLD_S]
    if interval_df.empty:
        return {
            "policy": key.policy,
            "cell_code": key.cell_code,
            "anomaly_count_gt_600s": 0,
            "anomaly_count_gt_3600s": 0,
            "max_dt_s": 0.0,
            "first_anomaly_cycle": "",
            "last_anomaly_cycle": "",
        }

    max_dt = float(interval_df["dt_s"].max())
    first_cycle = int(gt3600["cycles"].min()) if not gt3600.empty else ""
    last_cycle = int(gt3600["cycles"].max()) if not gt3600.empty else ""
    return {
        "policy": key.policy,
        "cell_code": key.cell_code,
        "anomaly_count_gt_600s": int(len(gt600)),
        "anomaly_count_gt_3600s": int(len(gt3600)),
        "max_dt_s": max_dt,
        "first_anomaly_cycle": first_cycle,
        "last_anomaly_cycle": last_cycle,
    }


def build_cell_timeseries(
    key: CellKey,
    cycles: List[int],
    dim_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    is_abnormal_cell: bool,
) -> pd.DataFrame:
    """Build one cell's 60-bin timeseries and cumulative table."""

    cycles_df = pd.DataFrame({"cycles": cycles})
    full = cycles_df.merge(dim_df, how="cross")
    if not agg_df.empty:
        full = full.merge(
            agg_df,
            on=["cycles", "soc_bin", "rate_bin", "temp_bin", "cross_bin"],
            how="left",
        )
    if "cycle_charge_time_h" not in full.columns:
        full["cycle_charge_time_h"] = 0.0
    full["cycle_charge_time_h"] = full["cycle_charge_time_h"].fillna(0.0)
    full["policy"] = key.policy
    full["cell_code"] = key.cell_code
    full["is_abnormal_cell"] = int(is_abnormal_cell)

    full = full.sort_values(["cross_bin", "cycles"], kind="mergesort")
    full["cumulative_charge_time_h"] = full.groupby("cross_bin")["cycle_charge_time_h"].cumsum()

    nonzero_counts = (
        full.groupby("cycles")["cycle_charge_time_h"]
        .apply(lambda s: int((s > 0.0).sum()))
        .rename("nonzero_cross_bin_count_cycle")
        .reset_index()
    )
    full = full.merge(nonzero_counts, on="cycles", how="left")
    full["nonzero_cross_bin_count_cycle"] = full["nonzero_cross_bin_count_cycle"].fillna(0).astype(int)

    full = full.sort_values(["cycles", "cross_bin"], kind="mergesort").reset_index(drop=True)
    return full[
        [
            "policy",
            "cell_code",
            "cycles",
            "soc_bin",
            "rate_bin",
            "temp_bin",
            "cross_bin",
            "soc_label",
            "rate_label",
            "temp_label",
            "cross_label",
            "cycle_charge_time_h",
            "cumulative_charge_time_h",
            "nonzero_cross_bin_count_cycle",
            "is_abnormal_cell",
        ]
    ].copy()


def build_cell_final(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """Build one cell's final summary table across all cycles for each cross bin."""

    final = (
        timeseries_df.groupby(
            [
                "policy",
                "cell_code",
                "soc_bin",
                "rate_bin",
                "temp_bin",
                "cross_bin",
                "soc_label",
                "rate_label",
                "temp_label",
                "cross_label",
                "is_abnormal_cell",
            ],
            as_index=False,
        )
        .agg(
            total_charge_time_h=("cycle_charge_time_h", "sum"),
            final_cumulative_charge_time_h=("cumulative_charge_time_h", "max"),
            max_cycle=("cycles", "max"),
        )
        .sort_values(["policy", "cell_code", "cross_bin"], kind="mergesort")
        .reset_index(drop=True)
    )
    return final


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    mode: str,
    header: bool,
    time_cols: List[str] | None = None,
    time_sig_digits: int = 1,
) -> None:
    """Save dataframe to csv with fixed UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df
    if time_cols:
        out_df = apply_time_sig_digits(out_df, time_cols=time_cols, digits=time_sig_digits)
    out_df.to_csv(
        path,
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8",
        float_format="%.15g",
    )


def main() -> None:
    """Execute full pipeline and export outputs."""

    args = parse_args()
    print("Loading life performance...")
    _, base_map, cycle_map = load_life_tables(
        args.life_path,
        q_min=args.q_min,
        q_max=args.q_max,
    )

    print("Scanning cycles files...")
    file_map = scan_cycles_files(args.raw_dir)
    valid_keys = sorted(set(base_map.keys()).intersection(file_map.keys()), key=lambda x: (x.policy, x.cell_code))
    if not valid_keys:
        raise RuntimeError("No overlapped cells between base_map and cycles files.")
    print(f"Valid cells: {len(valid_keys)}")

    print("Computing quantile edges (pass 1, sampled)...")
    rate_edges, temp_edges = compute_quantile_edges(
        keys=valid_keys,
        file_map=file_map,
        base_map=base_map,
        encoding=args.encoding,
    )
    print("rate_edges:", [round(float(x), 8) for x in rate_edges])
    print("temp_edges:", [round(float(x), 8) for x in temp_edges])

    dim_df = build_dim_table(rate_edges=rate_edges, temp_edges=temp_edges)
    save_dataframe(dim_df, args.out_bin_edges, mode="w", header=True)
    print(f"Saved bin edges: {args.out_bin_edges}")

    # Reset outputs so repeated runs always regenerate cleanly.
    for out_path in [
        args.out_timeseries,
        args.out_final,
        args.out_anomalies,
        args.out_abnormal_cells,
        args.out_abnormal_timeseries,
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

    wrote_timeseries_header = False
    wrote_final_header = False
    wrote_anomaly_header = False
    wrote_abnormal_cell_header = False
    wrote_abnormal_ts_header = False
    total_all_temp_abnormal_cycles = 0

    print("Building outputs (pass 2, full intervals)...")
    for idx, key in enumerate(valid_keys):
        interval_df, all_temp_abnormal_cycles = extract_intervals(
            file_path=file_map[key],
            key=key,
            base_q=base_map[key],
            encoding=args.encoding,
        )
        total_all_temp_abnormal_cycles += all_temp_abnormal_cycles

        anomaly_df = interval_df[interval_df["ts_anomaly_reason"] != ""].copy()
        if not anomaly_df.empty:
            anomaly_out = anomaly_df[
                [
                    "policy",
                    "cell_code",
                    "cycles",
                    "ts_prev",
                    "ts",
                    "dt_s",
                    "soc_bin",
                    "soc_mid_percent",
                    "c_rate_mid",
                    "temp_mid_c",
                    "ts_anomaly_reason",
                ]
            ].copy()
            save_dataframe(
                anomaly_out,
                args.out_anomalies,
                mode="a",
                header=not wrote_anomaly_header,
                time_cols=["ts_prev", "ts", "dt_s"],
                time_sig_digits=args.time_sig_digits,
            )
            wrote_anomaly_header = True

        abnormal_stats = summarize_abnormal_cell(interval_df=interval_df, key=key)
        is_abnormal_cell = abnormal_stats["anomaly_count_gt_3600s"] > 0

        if is_abnormal_cell:
            abnormal_cell_df = pd.DataFrame([abnormal_stats])
            save_dataframe(
                abnormal_cell_df,
                args.out_abnormal_cells,
                mode="a",
                header=not wrote_abnormal_cell_header,
                time_cols=["max_dt_s"],
                time_sig_digits=args.time_sig_digits,
            )
            wrote_abnormal_cell_header = True

        # Binning and cycle aggregation.
        if interval_df.empty:
            agg_df = pd.DataFrame(
                columns=["cycles", "soc_bin", "rate_bin", "temp_bin", "cross_bin", "cycle_charge_time_h"]
            )
        else:
            work = interval_df.copy()
            work["rate_bin"] = assign_numeric_bin(work["c_rate_mid"].to_numpy(dtype=float), rate_edges)
            work["temp_bin"] = assign_numeric_bin(work["temp_mid_c"].to_numpy(dtype=float), temp_edges)
            work["cross_bin"] = build_cross_bin(
                soc_bin=work["soc_bin"].to_numpy(dtype=np.int16),
                rate_bin=work["rate_bin"].to_numpy(dtype=np.int16),
                temp_bin=work["temp_bin"].to_numpy(dtype=np.int16),
            )
            work["cycle_charge_time_h"] = np.maximum(work["dt_s"].to_numpy(dtype=float), 0.0) / 3600.0
            agg_df = (
                work.groupby(
                    ["cycles", "soc_bin", "rate_bin", "temp_bin", "cross_bin"],
                    as_index=False,
                )["cycle_charge_time_h"]
                .sum()
                .sort_values(["cycles", "cross_bin"], kind="mergesort")
                .reset_index(drop=True)
            )

        cycles = cycle_map.get(key, [])
        if not cycles:
            if (idx + 1) % 20 == 0:
                print(f"[pass2] processed {idx + 1}/{len(valid_keys)} cells (no cycles in life map)")
            continue

        cell_timeseries = build_cell_timeseries(
            key=key,
            cycles=cycles,
            dim_df=dim_df,
            agg_df=agg_df,
            is_abnormal_cell=is_abnormal_cell,
        )
        cell_final = build_cell_final(cell_timeseries)

        save_dataframe(
            cell_timeseries,
            args.out_timeseries,
            mode="a",
            header=not wrote_timeseries_header,
            time_cols=["cycle_charge_time_h", "cumulative_charge_time_h"],
            time_sig_digits=args.time_sig_digits,
        )
        wrote_timeseries_header = True

        save_dataframe(
            cell_final,
            args.out_final,
            mode="a",
            header=not wrote_final_header,
            time_cols=["total_charge_time_h", "final_cumulative_charge_time_h"],
            time_sig_digits=args.time_sig_digits,
        )
        wrote_final_header = True

        if is_abnormal_cell:
            save_dataframe(
                cell_timeseries,
                args.out_abnormal_timeseries,
                mode="a",
                header=not wrote_abnormal_ts_header,
                time_cols=["cycle_charge_time_h", "cumulative_charge_time_h"],
                time_sig_digits=args.time_sig_digits,
            )
            wrote_abnormal_ts_header = True

        if (idx + 1) % 20 == 0 or (idx + 1) == len(valid_keys):
            print(f"[pass2] processed {idx + 1}/{len(valid_keys)} cells")

    print("Done.")
    print(f"timeseries: {args.out_timeseries}")
    print(f"final: {args.out_final}")
    print(f"bin_edges: {args.out_bin_edges}")
    print(f"anomalies: {args.out_anomalies}")
    print(f"abnormal_cells: {args.out_abnormal_cells}")
    print(f"abnormal_timeseries: {args.out_abnormal_timeseries}")
    print(f"diagnostic_all_temp_abnormal_cycles: {total_all_temp_abnormal_cycles}")
    print(f"q_min_filter: {args.q_min}")
    print(f"q_max_filter: {args.q_max}")


if __name__ == "__main__":
    main()
