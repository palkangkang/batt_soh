from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# Default Config (overridable by CLI)
# =========================
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_INPUT_GLOB = "cycles_*.csv"
DEFAULT_RECURSIVE = True

DEFAULT_CHARGE_OUTPUT_NAME = "charge_interval_features.csv"
DEFAULT_DISCHARGE_OUTPUT_NAME = "discharge_interval_features.csv"
DEFAULT_OUTPUT_DIR = Path("data/processed")

DEFAULT_CHARGE_START = 3.0
DEFAULT_CHARGE_END = 3.6
DEFAULT_DISCHARGE_START = 3.6
DEFAULT_DISCHARGE_END = 2.8
DEFAULT_VOLTAGE_STEP = 0.05
DEFAULT_EPS = 1e-3

DEFAULT_ENCODING = "utf-8-sig"

DEFAULT_ENABLE_TEMP_FILTER = True
DEFAULT_TEMP_VALID_MIN = -40.0
DEFAULT_TEMP_VALID_MAX = 120.0
DEFAULT_TEMP_MIN_POINTS_FOR_MAD = 5
DEFAULT_TEMP_MAD_Z_THRESHOLD = 3.5


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract charge/discharge voltage interval features from cycles_*.csv files."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--input-glob", type=str, default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--encoding", type=str, default=DEFAULT_ENCODING)

    parser.add_argument("--mode", choices=["both", "chg", "dischg"], default="both")

    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--charge-output-name", type=str, default=DEFAULT_CHARGE_OUTPUT_NAME)
    parser.add_argument("--discharge-output-name", type=str, default=DEFAULT_DISCHARGE_OUTPUT_NAME)

    parser.add_argument("--charge-start", type=float, default=DEFAULT_CHARGE_START)
    parser.add_argument("--charge-end", type=float, default=DEFAULT_CHARGE_END)
    parser.add_argument("--discharge-start", type=float, default=DEFAULT_DISCHARGE_START)
    parser.add_argument("--discharge-end", type=float, default=DEFAULT_DISCHARGE_END)
    parser.add_argument("--voltage-step", type=float, default=DEFAULT_VOLTAGE_STEP)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)

    parser.add_argument("--disable-temp-filter", action="store_true")
    parser.add_argument("--temp-valid-min", type=float, default=DEFAULT_TEMP_VALID_MIN)
    parser.add_argument("--temp-valid-max", type=float, default=DEFAULT_TEMP_VALID_MAX)
    parser.add_argument("--temp-min-points-for-mad", type=int, default=DEFAULT_TEMP_MIN_POINTS_FOR_MAD)
    parser.add_argument("--temp-mad-z-threshold", type=float, default=DEFAULT_TEMP_MAD_Z_THRESHOLD)

    return parser.parse_args()


def infer_decimal_places(step: float) -> int:
    s = f"{step:.10f}".rstrip("0").rstrip(".")
    if "." not in s:
        return 1
    return max(1, len(s.split(".")[1]))


def build_range_label(low: float, high: float, is_last: bool, decimals: int) -> str:
    # Guard against label collapse like "[3.0,3.0)" when precision is higher than display.
    lo_txt = f"{low:.{decimals}f}"
    hi_txt = f"{high:.{decimals}f}"
    while lo_txt == hi_txt and decimals < 8:
        decimals += 1
        lo_txt = f"{low:.{decimals}f}"
        hi_txt = f"{high:.{decimals}f}"
    return f"[{lo_txt},{hi_txt}{']' if is_last else ')'}"


def build_modes(args: argparse.Namespace) -> List[ModeConfig]:
    modes = [
        ModeConfig(
            state="chg",
            flag_col="flag_chg",
            ah_col="ah_chg",
            voltage_start=args.charge_start,
            voltage_end=args.charge_end,
            output_path=args.out_dir / args.charge_output_name,
        ),
        ModeConfig(
            state="dischg",
            flag_col="flag_dischg",
            ah_col="ah_dischg",
            voltage_start=args.discharge_start,
            voltage_end=args.discharge_end,
            output_path=args.out_dir / args.discharge_output_name,
        ),
    ]

    if args.mode == "chg":
        return [m for m in modes if m.state == "chg"]
    if args.mode == "dischg":
        return [m for m in modes if m.state == "dischg"]
    return modes


def find_target_files(raw_dir: Path, input_glob: str, recursive: bool) -> List[Path]:
    files = sorted(raw_dir.rglob(input_glob) if recursive else raw_dir.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No input file matched {input_glob!r} under {raw_dir}")
    return files


def build_ranges(v_start: float, v_end: float, step: float, eps: float, decimals: int) -> Tuple[str, List[RangeDef]]:
    if abs(v_start - v_end) <= eps:
        raise ValueError("voltage start and end must be different")
    if step <= 0:
        raise ValueError("voltage step must be > 0")

    direction = "asc" if v_start < v_end else "desc"
    ranges: List[RangeDef] = []

    cur = v_start
    if direction == "asc":
        while cur < v_end - eps:
            nxt = round(cur + step, 10)
            is_last = abs(nxt - v_end) <= eps
            ranges.append(
                RangeDef(
                    low=cur,
                    high=nxt,
                    label=build_range_label(cur, nxt, is_last, decimals),
                    is_last=is_last,
                )
            )
            cur = nxt
    else:
        while cur > v_end + eps:
            nxt = round(cur - step, 10)
            is_last = abs(nxt - v_end) <= eps
            ranges.append(
                RangeDef(
                    low=nxt,
                    high=cur,
                    label=build_range_label(cur, nxt, is_last, decimals),
                    is_last=is_last,
                )
            )
            cur = nxt

    return direction, ranges


def is_near_boundary(v: float, target: float, eps: float) -> bool:
    return abs(v - target) <= eps


def find_boundary_segments(records: List[Record], start_v: float, end_v: float, eps: float) -> List[Tuple[int, int]]:
    """
    Pair segments by first-index boundary matching:
    1) Find first index near start_v (within eps).
    2) From that index onward, find first index near end_v.
    3) Record this pair and continue searching after the matched end index.
    """
    segments: List[Tuple[int, int]] = []
    n = len(records)
    search_from = 0

    while search_from < n:
        start_idx = None
        for i in range(search_from, n):
            if is_near_boundary(records[i].v, start_v, eps):
                start_idx = i
                break
        if start_idx is None:
            break

        end_idx = None
        for j in range(start_idx + 1, n):
            if is_near_boundary(records[j].v, end_v, eps):
                end_idx = j
                break
        if end_idx is None:
            break

        segments.append((start_idx, end_idx))
        search_from = end_idx + 1

    return segments


def filter_temperatures(
    values: List[float],
    enable_filter: bool,
    temp_valid_min: float,
    temp_valid_max: float,
    temp_min_points_for_mad: int,
    temp_mad_z_threshold: float,
    eps: float,
) -> List[float]:
    if not enable_filter:
        return values

    clipped = [v for v in values if temp_valid_min <= v <= temp_valid_max]
    if not clipped:
        return []

    if len(clipped) < temp_min_points_for_mad:
        return clipped

    med = statistics.median(clipped)
    deviations = [abs(v - med) for v in clipped]
    mad = statistics.median(deviations)
    if mad <= eps:
        return clipped

    filtered = [v for v in clipped if abs(0.6745 * (v - med) / mad) <= temp_mad_z_threshold]
    return filtered if filtered else clipped


def load_mode_records(file_path: Path, mode: ModeConfig, encoding: str) -> Dict[Tuple, List[Record]]:
    grouped: Dict[Tuple, List[Record]] = {}
    with file_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        required = {"cell_code", "cycles", "ts", "V", "Temper", mode.ah_col, mode.flag_col}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"Missing required columns in {file_path}: {sorted(missing)}")

        has_policy_column = "policy" in (reader.fieldnames or [])
        for row in reader:
            try:
                if int(float(row[mode.flag_col])) != 1:
                    continue
                cell_code = row["cell_code"]
                cycles = int(float(row["cycles"]))
                ts = float(row["ts"])
                v = float(row["V"])
                ah = float(row[mode.ah_col])
                temper = float(row["Temper"])
            except (TypeError, ValueError, KeyError):
                continue

            policy = (row.get("policy") or "").strip() if has_policy_column else ""
            # Conditional grouping key:
            # - with policy value: ("p", policy, cell_code, cycles)
            # - without policy: ("n", cell_code, cycles)
            # This prevents accidental merging between policy and non-policy records.
            if policy:
                key = ("p", policy, cell_code, cycles)
            else:
                key = ("n", cell_code, cycles)
            grouped.setdefault(key, []).append(Record(ts=ts, v=v, ah=ah, temper=temper))

    for key in grouped:
        grouped[key].sort(key=lambda r: r.ts)
    return grouped


def extract_features(
    grouped: Dict[Tuple, List[Record]],
    mode: ModeConfig,
    voltage_step: float,
    eps: float,
    decimals: int,
    enable_temp_filter: bool,
    temp_valid_min: float,
    temp_valid_max: float,
    temp_min_points_for_mad: int,
    temp_mad_z_threshold: float,
) -> List[dict]:
    direction, voltage_ranges = build_ranges(mode.voltage_start, mode.voltage_end, voltage_step, eps, decimals)
    output_rows: List[dict] = []

    for group_key, records in grouped.items():
        if group_key[0] == "p":
            _, _, cell_code, cycles = group_key
        else:
            _, cell_code, cycles = group_key
        for rng in voltage_ranges:
            if direction == "asc":
                start_v, end_v = rng.low, rng.high
            else:
                start_v, end_v = rng.high, rng.low

            valid_segments: List[Tuple[float, float, float]] = []
            for start_idx, end_idx in find_boundary_segments(records, start_v, end_v, eps):
                if end_idx <= start_idx:
                    continue

                start_rec = records[start_idx]
                end_rec = records[end_idx]
                delta_ah = end_rec.ah - start_rec.ah
                duration_s = end_rec.ts - start_rec.ts
                if delta_ah < -eps or duration_s <= eps:
                    continue

                seg_temps_raw = [r.temper for r in records[start_idx : end_idx + 1]]
                seg_temps = filter_temperatures(
                    seg_temps_raw,
                    enable_filter=enable_temp_filter,
                    temp_valid_min=temp_valid_min,
                    temp_valid_max=temp_valid_max,
                    temp_min_points_for_mad=temp_min_points_for_mad,
                    temp_mad_z_threshold=temp_mad_z_threshold,
                    eps=eps,
                )
                if not seg_temps:
                    continue

                avg_temper = sum(seg_temps) / len(seg_temps)
                valid_segments.append((delta_ah, duration_s, avg_temper))

            total_count = len(valid_segments)
            for idx, (delta_ah, duration_s, avg_temper) in enumerate(valid_segments, start=1):
                output_rows.append(
                    {
                        "state": mode.state,
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
    args = parse_args()
    recursive = not args.non_recursive
    modes = build_modes(args)
    decimals = infer_decimal_places(args.voltage_step)

    files = find_target_files(args.raw_dir, args.input_glob, recursive)
    print(f"Input files: {len(files)}")

    for mode in modes:
        all_rows: List[dict] = []
        for fp in files:
            grouped = load_mode_records(fp, mode, args.encoding)
            all_rows.extend(
                extract_features(
                    grouped=grouped,
                    mode=mode,
                    voltage_step=args.voltage_step,
                    eps=args.eps,
                    decimals=decimals,
                    enable_temp_filter=(not args.disable_temp_filter),
                    temp_valid_min=args.temp_valid_min,
                    temp_valid_max=args.temp_valid_max,
                    temp_min_points_for_mad=args.temp_min_points_for_mad,
                    temp_mad_z_threshold=args.temp_mad_z_threshold,
                )
            )

        save_csv(all_rows, mode.output_path)
        print(f"[{mode.state}] Output rows: {len(all_rows)}")
        print(f"[{mode.state}] Saved to: {mode.output_path}")


if __name__ == "__main__":
    main()
