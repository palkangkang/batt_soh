# Schema Contract (No Policy)

## 1) life_performance.csv

Required columns:

- `cell_code`
- `cycles`
- target alias columns: exactly one of
  - `q_discharge`
  - `Q_dischg`
  - `q_dischg`

Strict alias rule:

- if none of the three target aliases exist: fail
- if two or more target aliases exist: fail
- if exactly one exists: normalize to internal name `q_discharge`

Additional strict checks:

- `cycles` must be castable to integer
- `q_discharge` must be castable to numeric
- if `policy` column is absent: key `cell_code + cycles` must be unique
- if `policy` column is present: key `policy + cell_code + cycles` must be unique (policy only used as alignment key, not as model feature)

## 2) charge_interval_features.csv / discharge_interval_features.csv

Required columns:

- `cell_code`
- `cycles`
- `range`
- `delta_ah`
- `range_count`

Required processing rule:

- keep only rows where `range_count == 1`

Aggregation rule before merge:

- group by `cell_code + cycles + range`
- aggregate `delta_ah_sum = sum(delta_ah)`

## 3) Excluded variables

Even if present in source files, the following must not be used in model inputs or correlation controls:

- `initial_c_rate`
- `switch_soc_percent`
- `post_switch_c_rate`
- any policy-derived feature columns

## 4) Join key and target

- preferred join key:
  - `policy + cell_code + cycles` when both sides contain `policy`
  - otherwise `cell_code + cycles`
- target: normalized `q_discharge`

## 5) Output constraints

- unified summary folder: `outputs/analysis/correlation_no_policy`
- markdown files allowed in this folder: exactly one (`correlation_summary_no_policy.md`)
- image format: PNG
