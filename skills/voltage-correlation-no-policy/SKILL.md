---
name: voltage-correlation-no-policy
description: Run correlation analysis between charge/discharge voltage-interval features and discharge capacity without any policy triplet variables. Use when you need first-occurrence range-based correlation tables, robustness plots, and a single unified markdown report that compares charge-only, discharge-only, and charge-plus-discharge feature groups.
---

# Voltage Correlation No Policy

## Overview

Use this skill to analyze correlation between voltage-interval features and `q_discharge` while **strictly excluding policy triplet parameters**.

This skill is for:

- charge-range feature correlation vs `q_discharge`
- discharge-range feature correlation vs `q_discharge`
- unified model-level comparison: `charge_only`, `discharge_only`, `charge_plus_discharge`

This skill is **not** for policy-aware analysis.

## Required Inputs

- `data/processed/life_performance.csv`
- `data/processed/charge_interval_features.csv`
- `data/processed/discharge_interval_features.csv`

For detailed schema/validation contract, read:

- `references/schema_contract.md`

## Hard Constraints

- Never read or use `policy_meaning.csv`.
- Never use policy triplet fields: `initial_c_rate`, `switch_soc_percent`, `post_switch_c_rate`.
- Only use first-occurrence interval rows: `range_count == 1`.
- Unified summary directory keeps exactly one markdown report: `correlation_summary_no_policy.md`.
- If source files include `policy`, it can be used only as join key (`policy + cell_code + cycles`) to avoid key collision; it must never enter correlation/model features.

## Runtime Environment

Run with the repository-approved environment:

```bash
cd C:\Users\pal\pyenv\ds_env
pipenv run python C:\Users\pal\projects\batt_soh\skills\voltage-correlation-no-policy\scripts\analyze_charge_vs_q_discharge_no_policy.py
pipenv run python C:\Users\pal\projects\batt_soh\skills\voltage-correlation-no-policy\scripts\analyze_discharge_vs_q_discharge_no_policy.py
pipenv run python C:\Users\pal\projects\batt_soh\skills\voltage-correlation-no-policy\scripts\summarize_correlation_no_policy.py
```

## Outputs

- Charge analysis:
  - `outputs/analysis/charge_feature_q_discharge_corr_no_policy/correlation_by_range.csv`
  - `outputs/analysis/charge_feature_q_discharge_corr_no_policy/merged_dataset_overview.csv`
  - `outputs/analysis/charge_feature_q_discharge_corr_no_policy/*.png`
- Discharge analysis:
  - `outputs/analysis/discharge_feature_q_discharge_corr_no_policy/univariate_correlation.csv`
  - `outputs/analysis/discharge_feature_q_discharge_corr_no_policy/feature_coverage_summary.csv`
  - `outputs/analysis/discharge_feature_q_discharge_corr_no_policy/*.png`
- Unified summary:
  - `outputs/analysis/correlation_no_policy/combo_correlation_summary.csv`
  - `outputs/analysis/correlation_no_policy/combo_correlation_uplift.csv`
  - `outputs/analysis/correlation_no_policy/combo_correlation_comparison.png`
  - `outputs/analysis/correlation_no_policy/correlation_summary_no_policy.md`

## Notes

- Image files are saved as PNG.
- Chinese labels are used when compatible CJK fonts are available; otherwise scripts fall back to safe fonts.
- `life_performance` target alias compatibility is strict: exactly one of `q_discharge`, `Q_dischg`, `q_dischg` must exist.
