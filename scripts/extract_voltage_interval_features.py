from __future__ import annotations

import csv
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
VOLTAGE_STEP = 0.1

ENCODING = "utf-8-sig"
FLOAT_EPS = 1e-9


@dataclass
class Record:
    ts: float
    v: float
    ah: float


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
                    label=f"[{cur:.1f},{nxt:.1f}{']' if is_last else ')'}",
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
                    label=f"[{cur:.1f},{nxt:.1f}{']' if is_last else ')'}",
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
        required = {"policy", "cell_code", "cycles", "ts", "V", mode.ah_col, mode.flag_col}
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
            except (TypeError, ValueError, KeyError):
                continue

            key = (policy, cell_code, cycles)
            grouped.setdefault(key, []).append(Record(ts=ts, v=v, ah=ah))

    for key in grouped:
        grouped[key].sort(key=lambda r: r.ts)
    return grouped


def is_valid_directional_segment(
    records: List[Record],
    start_idx: int,
    end_idx: int,
    rng: RangeDef,
    direction: str,
) -> bool:
    if end_idx <= start_idx:
        return False

    start_rec = records[start_idx]
    end_rec = records[end_idx]
    prev_v = records[start_idx - 1].v if start_idx > 0 else None
    next_v = records[end_idx + 1].v if end_idx + 1 < len(records) else None

    if direction == "asc":
        entered_from_expected_side = (prev_v is None) or (prev_v <= rng.low + FLOAT_EPS)
        moved_in_expected_direction = end_rec.v > start_rec.v + FLOAT_EPS
        reached_expected_boundary = (
            (next_v is not None and next_v >= rng.high - FLOAT_EPS)
            or (rng.is_last and end_rec.v >= rng.high - FLOAT_EPS)
        )
    else:
        entered_from_expected_side = (prev_v is None) or (prev_v >= rng.high - FLOAT_EPS)
        moved_in_expected_direction = end_rec.v < start_rec.v - FLOAT_EPS
        reached_expected_boundary = (
            (next_v is not None and next_v <= rng.low + FLOAT_EPS)
            or (rng.is_last and end_rec.v <= rng.low + FLOAT_EPS)
        )

    return entered_from_expected_side and moved_in_expected_direction and reached_expected_boundary


def extract_features(
    grouped: Dict[Tuple[str, str, int], List[Record]],
    mode: ModeConfig,
) -> List[dict]:
    direction, voltage_ranges = build_ranges(mode.voltage_start, mode.voltage_end, VOLTAGE_STEP)
    output_rows: List[dict] = []

    for (policy, cell_code, cycles), records in grouped.items():
        for rng in voltage_ranges:
            # Slice contiguous runs in this voltage range, then keep only directional runs.
            runs: List[Tuple[int, int]] = []
            run_start = None
            for i, r in enumerate(records):
                in_rng = is_in_range(r.v, rng, direction)
                if in_rng and run_start is None:
                    run_start = i
                elif (not in_rng) and run_start is not None:
                    runs.append((run_start, i - 1))
                    run_start = None
            if run_start is not None:
                runs.append((run_start, len(records) - 1))

            valid_segments: List[Tuple[float, float]] = []
            for start_idx, end_idx in runs:
                if not is_valid_directional_segment(records, start_idx, end_idx, rng, direction):
                    continue

                start_rec = records[start_idx]
                end_rec = records[end_idx]
                delta_ah = end_rec.ah - start_rec.ah
                duration_s = end_rec.ts - start_rec.ts
                if delta_ah < -FLOAT_EPS or duration_s <= FLOAT_EPS:
                    continue

                valid_segments.append((delta_ah, duration_s))

            # Count only validated directional segments.
            total_count = len(valid_segments)
            for idx, (delta_ah, duration_s) in enumerate(valid_segments, start=1):
                output_rows.append(
                    {
                        "state": mode.state,
                        "policy": policy,
                        "cell_code": cell_code,
                        "cycles": cycles,
                        "range": rng.label,
                        "delta_ah": delta_ah,
                        "charge_duration_s": duration_s,
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
