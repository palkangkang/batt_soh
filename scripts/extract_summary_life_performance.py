from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple


# =========================
# Config (edit here first)
# =========================
RAW_DIR = Path("data/raw")
SUMMARY_FILE_GLOB = "summary_*.csv"
RECURSIVE_SEARCH = True
ENCODING = "utf-8-sig"

OUTPUT_PATH = Path("data/processed/life_performance.csv")
FLOAT_EPS = 1e-9


def find_summary_files() -> list[Path]:
    files = sorted(
        RAW_DIR.rglob(SUMMARY_FILE_GLOB) if RECURSIVE_SEARCH else RAW_DIR.glob(SUMMARY_FILE_GLOB)
    )
    if not files:
        raise FileNotFoundError(f"No files matched {SUMMARY_FILE_GLOB!r} under {RAW_DIR}")
    return files


def extract_rows(files: list[Path]) -> tuple[list[dict], int]:
    required = {"policy", "cell_code", "cycle", "QDischarge", "Tmax", "IR"}
    dedup: Dict[Tuple[str, str, int], dict] = {}
    conflict_count = 0

    for fp in files:
        with fp.open("r", encoding=ENCODING, newline="") as f:
            reader = csv.DictReader(f)
            missing = required.difference(set(reader.fieldnames or []))
            if missing:
                raise KeyError(f"Missing required columns in {fp}: {sorted(missing)}")

            for row in reader:
                try:
                    policy = row["policy"]
                    cell_code = row["cell_code"]
                    cycles = int(float(row["cycle"]))
                    q_discharge = float(row["QDischarge"])
                    t_max = float(row["Tmax"])
                    ir = float(row["IR"])
                except (TypeError, ValueError, KeyError):
                    continue

                key = (policy, cell_code, cycles)
                new_row = {
                    "policy": policy,
                    "cell_code": cell_code,
                    "cycles": cycles,
                    "q_discharge": q_discharge,
                    "t_max": t_max,
                    "ir": ir,
                }

                if key not in dedup:
                    dedup[key] = new_row
                    continue

                old = dedup[key]
                same_q = abs(old["q_discharge"] - q_discharge) <= FLOAT_EPS
                same_t = abs(old["t_max"] - t_max) <= FLOAT_EPS
                same_ir = abs(old["ir"] - ir) <= FLOAT_EPS
                if not (same_q and same_t and same_ir):
                    conflict_count += 1
                    dedup[key] = new_row

    rows = list(dedup.values())
    rows.sort(key=lambda x: (x["policy"], x["cell_code"], x["cycles"]))
    return rows, conflict_count


def save_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "cell_code", "cycles", "q_discharge", "t_max", "ir"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = find_summary_files()
    rows, conflict_count = extract_rows(files)
    save_csv(rows, OUTPUT_PATH)
    print(f"Input files: {len(files)}")
    print(f"Output rows (unique by policy+cell_code+cycles): {len(rows)}")
    print(f"Conflicting duplicate keys replaced: {conflict_count}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
