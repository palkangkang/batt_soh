from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# Config (edit here first)
# =========================
RAW_DIR = Path("data/raw")
CHARGE_OUTPUT_PATH = Path("data/processed/charge_interval_features.csv")
DISCHARGE_OUTPUT_PATH = Path("data/processed/discharge_interval_features.csv")

# Recursively process all cycles files under RAW_DIR.
CYCLES_FILE_GLOB = "cycles_*.csv"
RECURSIVE_SEARCH = True

CHARGE_VOLTAGE_START = 3.0
CHARGE_VOLTAGE_END = 3.6
DISCHARGE_VOLTAGE_START = 3.6
DISCHARGE_VOLTAGE_END = 2.8
VOLTAGE_STEP = 0.05

ENCODING = "utf-8-sig"
FLOAT_EPS = 1e-3

# Temperature outlier filtering before average calculation.
ENABLE_TEMPERATURE_OUTLIER_FILTER = True
TEMP_VALID_MIN = -40.0
TEMP_VALID_MAX = 120.0
TEMP_MIN_POINTS_FOR_MAD = 5
TEMP_MAD_Z_THRESHOLD = 3.5


def infer_decimal_places(step: float) -> int:
    s = f"{step:.10f}".rstrip("0").rstrip(".")
    if "." not in s:
        return 1
    return max(1, len(s.split(".")[1]))


RANGE_LABEL_DECIMALS = infer_decimal_places(VOLTAGE_STEP)


def build_range_label(low: float, high: float, is_last: bool) -> str:
    # Guard against label collapse like "[3.0,3.0)" when step precision is higher.
    decimals = RANGE_LABEL_DECIMALS
    lo_txt = f"{low:.{decimals}f}"
    hi_txt = f"{high:.{decimals}f}"
    while lo_txt == hi_txt and decimals < 8:
        decimals += 1
        lo_txt = f"{low:.{decimals}f}"
        hi_txt = f"{high:.{decimals}f}"
    return f"[{lo_txt},{hi_txt}{']' if is_last else ')'}"


@dataclass
class Record:
    ts: float
    v: float
    ah: float
    temper: float


@dataclass
class ModeConfig:
    state: str
    flag_col: str
    ah_col: str
    voltage_start: float
    voltage_end: float
    output_path: Path


@dataclass
class RangeDef:
    low: float
    high: float
    label: str
    is_last: bool


MODES = [
    ModeConfig(
        state="chg",
        flag_col="flag_chg",
        ah_col="ah_chg",
        voltage_start=CHARGE_VOLTAGE_START,
        voltage_end=CHARGE_VOLTAGE_END,
        output_path=CHARGE_OUTPUT_PATH,
    ),
    ModeConfig(
        state="dischg",
        flag_col="flag_dischg",
        ah_col="ah_dischg",
        voltage_start=DISCHARGE_VOLTAGE_START,
        voltage_end=DISCHARGE_VOLTAGE_END,
        output_path=DISCHARGE_OUTPUT_PATH,
    ),
]


def find_target_files() -> List[Path]:
    files = sorted(RAW_DIR.rglob(CYCLES_FILE_GLOB) if RECURSIVE_SEARCH else RAW_DIR.glob(CYCLES_FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No file matched pattern: {CYCLES_FILE_GLOB!r} under {RAW_DIR}")
    return files


def build_ranges(v_start: float, v_end: float, step: float) -> Tuple[str, List[RangeDef]]:
    if abs(v_start - v_end) <= FLOAT_EPS:
        raise ValueError("voltage_start and voltage_end must be different")
    if step <= 0:
        raise ValueError("VOLTAGE_STEP must be > 0")

    direction = "asc" if v_start < v_end else "desc"
    ranges: List[RangeDef] = []

    cur = v_start
    if direction == "asc":
        while cur < v_end - FLOAT_EPS:
            nxt = round(cur + step, 10)
            is_last = abs(nxt - v_end) <= FLOAT_EPS
            ranges.append(
                RangeDef(
                    low=cur,
                    high=nxt,
                    label=build_range_label(cur, nxt, is_last),
                    is_last=is_last,
                )
            )
            cur = nxt
    else:
        while cur > v_end + FLOAT_EPS:
            nxt = round(cur - step, 10)
            is_last = abs(nxt - v_end) <= FLOAT_EPS
            ranges.append(
                RangeDef(
                    low=nxt,
                    high=cur,
                    label=build_range_label(cur, nxt, is_last),
                    is_last=is_last,
                )
            )
            cur = nxt

    return direction, ranges


def is_in_range(v: float, rng: RangeDef, direction: str) -> bool:
    if direction == "asc":
        if rng.is_last:
            return (v >= rng.low - FLOAT_EPS) and (v <= rng.high + FLOAT_EPS)
        return (v >= rng.low - FLOAT_EPS) and (v < rng.high - FLOAT_EPS)

    if rng.is_last:
        return (v >= rng.low - FLOAT_EPS) and (v <= rng.high + FLOAT_EPS)
    return (v > rng.low + FLOAT_EPS) and (v <= rng.high + FLOAT_EPS)


def load_mode_records(file_path: Path, mode: ModeConfig) -> Dict[Tuple[str, str, int], List[Record]]:
    grouped: Dict[Tuple[str, str, int], List[Record]] = {}
    with file_path.open("r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        required = {"policy", "cell_code", "cycles", "ts", "V", "Temper", mode.ah_col, mode.flag_col}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"Missing required columns in {file_path}: {sorted(missing)}")

        for row in reader:
            try:
                if int(float(row[mode.flag_col])) != 1:
                    continue
                policy = row["policy"]
                cell_code = row["cell_code"]
                cycles = int(float(row["cycles"]))
                ts = float(row["ts"])
                v = float(row["V"])
                ah = float(row[mode.ah_col])
                temper = float(row["Temper"])
            except (TypeError, ValueError, KeyError):
                continue

            key = (policy, cell_code, cycles)
            grouped.setdefault(key, []).append(Record(ts=ts, v=v, ah=ah, temper=temper))

    for key in grouped:
        grouped[key].sort(key=lambda r: r.ts)
    return grouped


def is_near_boundary(v: float, target: float) -> bool:
    return abs(v - target) <= FLOAT_EPS


def find_boundary_segments(records: List[Record], start_v: float, end_v: float) -> List[Tuple[int, int]]:
    """
    Pair segments by first-index boundary matching:
    1) Find first index near start_v (within FLOAT_EPS).
    2) From that index onward, find first index near end_v.
    3) Record this pair and continue searching after the matched end index.
    """
    segments: List[Tuple[int, int]] = []
    n = len(records)
    search_from = 0

    while search_from < n:
        start_idx = None
        for i in range(search_from, n):
            if is_near_boundary(records[i].v, start_v):
                start_idx = i
                break
        if start_idx is None:
            break

        end_idx = None
        for j in range(start_idx + 1, n):
            if is_near_boundary(records[j].v, end_v):
                end_idx = j
                break
        if end_idx is None:
            break

        segments.append((start_idx, end_idx))
        search_from = end_idx + 1

    return segments


def filter_temperatures(values: List[float]) -> List[float]:
    if not ENABLE_TEMPERATURE_OUTLIER_FILTER:
        return values

    # 1) Physical range clipping first.
    clipped = [v for v in values if TEMP_VALID_MIN <= v <= TEMP_VALID_MAX]
    if not clipped:
        return []

    # 2) Robust MAD filtering for obvious spikes/drops.
    if len(clipped) < TEMP_MIN_POINTS_FOR_MAD:
        return clipped

    med = statistics.median(clipped)
    deviations = [abs(v - med) for v in clipped]
    mad = statistics.median(deviations)
    if mad <= FLOAT_EPS:
        return clipped

    filtered = [v for v in clipped if abs(0.6745 * (v - med) / mad) <= TEMP_MAD_Z_THRESHOLD]
    return filtered if filtered else clipped


def extract_features(
    grouped: Dict[Tuple[str, str, int], List[Record]],
    mode: ModeConfig,
) -> List[dict]:
    direction, voltage_ranges = build_ranges(mode.voltage_start, mode.voltage_end, VOLTAGE_STEP)
    output_rows: List[dict] = []

    for (policy, cell_code, cycles), records in grouped.items():
        for rng in voltage_ranges:
            if direction == "asc":
                # Example: [3.45,3.50) => first 3.45±eps point to first later 3.50±eps point.
                start_v, end_v = rng.low, rng.high
            else:
                # Example: [3.50,3.45) => first 3.50±eps point to first later 3.45±eps point.
                start_v, end_v = rng.high, rng.low

            valid_segments: List[Tuple[float, float, float]] = []
            for start_idx, end_idx in find_boundary_segments(records, start_v, end_v):
                if end_idx <= start_idx:
                    continue

                start_rec = records[start_idx]
                end_rec = records[end_idx]
                delta_ah = end_rec.ah - start_rec.ah
                duration_s = end_rec.ts - start_rec.ts
                if delta_ah < -FLOAT_EPS or duration_s <= FLOAT_EPS:
                    continue

                seg_temps_raw = [r.temper for r in records[start_idx : end_idx + 1]]
                seg_temps = filter_temperatures(seg_temps_raw)
                if not seg_temps:
                    continue
                avg_temper = sum(seg_temps) / len(seg_temps)
                valid_segments.append((delta_ah, duration_s, avg_temper))

            # Count only validated directional segments.
            total_count = len(valid_segments)
            for idx, (delta_ah, duration_s, avg_temper) in enumerate(valid_segments, start=1):
                output_rows.append(
                    {
                        "state": mode.state,
                        "policy": policy,
                        "cell_code": cell_code,
                        "cycles": cycles,
                        "range": rng.label,
                        "delta_ah": delta_ah,
                        "charge_duration_s": duration_s,
                        "avg_temper": avg_temper,
                        "range_count": idx,
                        "range_total_count": total_count,
                    }
                )

    output_rows.sort(
        key=lambda x: (
            x["state"],
            x["policy"],
            x["cell_code"],
            x["cycles"],
            x["range"],
            x["range_count"],
        )
    )
    return output_rows


def save_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "state",
        "policy",
        "cell_code",
        "cycles",
        "range",
        "delta_ah",
        "charge_duration_s",
        "avg_temper",
        "range_count",
        "range_total_count",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = find_target_files()
    print(f"Input files: {len(files)}")

    for mode in MODES:
        all_rows: List[dict] = []
        for fp in files:
            grouped = load_mode_records(fp, mode)
            all_rows.extend(extract_features(grouped, mode))

        all_rows.sort(
            key=lambda x: (
                x["state"],
                x["policy"],
                x["cell_code"],
                x["cycles"],
                x["range"],
                x["range_count"],
            )
        )
        save_csv(all_rows, mode.output_path)
        print(f"[{mode.state}] Output rows: {len(all_rows)}")
        print(f"[{mode.state}] Saved to: {mode.output_path}")


if __name__ == "__main__":
    main()
