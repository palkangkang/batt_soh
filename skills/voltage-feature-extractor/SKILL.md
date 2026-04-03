---
name: voltage-feature-extractor
description: Extract charge/discharge voltage-interval features from cycles_*.csv battery data with configurable voltage partitions and voltage matching tolerance. Use when you need range-based delta Ah, duration, and average temperature features grouped by cell_code and cycles (with optional policy-aware internal grouping), while keeping the batt_soh thread-compatible boundary-pairing logic.
---

# Voltage Feature Extractor

## Overview

Use this skill to build charge/discharge voltage-interval feature CSV files from `cycles_*.csv` files.

This skill assumes input columns have the same meanings as this thread's dataset.

Required columns: `cell_code`, `cycles`, `ts`, `V`, `Temper`, `flag_chg`, `flag_dischg`, `ah_chg`, `ah_dischg`.

Optional column: `policy`.

## Quick Start

Run from the repository root:

```bash
python skills/voltage-feature-extractor/scripts/extract_voltage_features.py
```

Default outputs:

- `data/processed/charge_interval_features.csv`
- `data/processed/discharge_interval_features.csv`

## Configure Voltage Partitions and Threshold

Use CLI overrides when needed:

```bash
python skills/voltage-feature-extractor/scripts/extract_voltage_features.py \
  --voltage-step 0.05 \
  --eps 0.001 \
  --charge-start 3.0 \
  --charge-end 3.6 \
  --discharge-start 3.6 \
  --discharge-end 2.8
```

Common options:

- `--raw-dir`: input root directory
- `--input-glob`: input file glob (default `cycles_*.csv`)
- `--mode`: `both`, `chg`, or `dischg`
- `--out-dir`: output directory
- `--charge-output-name`, `--discharge-output-name`: output file names
- `--eps`: voltage boundary matching tolerance (for boundary-point pairing)

## Output Schema

Each output row keeps this schema:

- `state`
- `cell_code`
- `cycles`
- `range`
- `delta_ah`
- `charge_duration_s`
- `avg_temper`
- `range_count`
- `range_total_count`

Notes:

- Output does not include `policy`.
- If `policy` exists in input, it is still used internally in grouping to avoid accidental cross-policy merging.

## Logic Contract (Thread-Aligned)

The implementation is intentionally aligned to this thread's final logic:

- Boundary-point first-index pairing is used, not generic contiguous-in-range scans.
- Charge and discharge directions are paired differently.
- Voltage label formatting avoids collapsed labels like `[3.0,3.0)`.
- Temperature average is computed after outlier handling.

See detailed rules in:

- `references/logic-spec.md`
