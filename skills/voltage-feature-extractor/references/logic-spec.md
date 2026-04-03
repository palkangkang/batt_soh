# Logic Specification

This document defines the extraction behavior implemented by `scripts/extract_voltage_features.py`.

## Inputs

Each input CSV row is expected to expose the same semantics as this thread's dataset.

- Required identity/time: `cell_code`, `cycles`, `ts`
- Optional identity: `policy`
- Signals: `V`, `Temper`
- State flags: `flag_chg`, `flag_dischg`
- Capacity integrals: `ah_chg`, `ah_dischg`

## Mode-Specific Filtering

- Charge mode keeps rows where `flag_chg == 1`, and uses `ah_chg`.
- Discharge mode keeps rows where `flag_dischg == 1`, and uses `ah_dischg`.

Rows are always sorted by `ts` ascending inside each group.

Grouping key is conditional:

- if `policy` exists and has value: `(policy, cell_code, cycles)`
- if `policy` missing/empty: `(cell_code, cycles)`

To avoid accidental merging between policy and non-policy records in mixed inputs, grouping tracks these two cases separately.

## Voltage Partitioning

For each mode, ranges are generated from:

- start voltage
- end voltage
- step

Charge default: `3.0 -> 3.6`.
Discharge default: `3.6 -> 2.8`.

Range labels use dynamic decimal precision based on `step`, with collapse guard to prevent labels like `[x,x)`.

## Boundary First-Index Pairing (Core Rule)

Tolerance `eps` is used for boundary matching (`abs(V - boundary) <= eps`).

For each range:

- Charge range `[low, high)`:
  - find the first index near `low`
  - then find the first later index near `high`
- Discharge range `[high, low)`:
  - find the first index near `high`
  - then find the first later index near `low`

After a pair is found, continue searching from `end_idx + 1`.

## Feature Computation Per Paired Segment

For each `(start_idx, end_idx)`:

- `delta_ah = ah[end_idx] - ah[start_idx]`
- `charge_duration_s = ts[end_idx] - ts[start_idx]`
- `avg_temper = mean(filtered_temperatures_in_segment)`

Segment validation:

- `end_idx > start_idx`
- `delta_ah >= -eps`
- `charge_duration_s > eps`
- filtered temperature list is non-empty

## Temperature Outlier Handling

When enabled:

1. Physical clipping: keep only `[temp_valid_min, temp_valid_max]`.
2. If enough points exist, apply MAD-based robust filtering.
3. Fallback to clipped values if MAD result is empty.

## Output Rows

Output schema:

- `state`
- `cell_code`
- `cycles`
- `range`
- `delta_ah`
- `charge_duration_s`
- `avg_temper`
- `range_count`
- `range_total_count`

Sort key:

`state, cell_code, cycles, range, range_count`
