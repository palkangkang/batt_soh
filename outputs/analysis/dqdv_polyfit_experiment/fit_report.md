# dQ/dV Polynomial HR Experiment Report

## 1. Experiment Setup

- sample_count: **5**
- fixed_sample: `4_8C-80PER_4_8C | 465027 | 10`
- random_seed: `20260410`
- degree_range: `7~9`
- high_sensitivity_voltage_window: `2.5V~3.4V`
- discharge_dqdv_transform: `abs(dQ/dV)`
- hr_voltage_step: `0.002 V`

## 2. Degree Selection Score

- `score = 0.60*norm(RMSE_high) + 0.25*norm(1-R2_q) + 0.15*norm(RMSE_q)`
- tie-break: lower `RMSE_high`, then lower `degree`.

## 3. Promotion Gates

- valid_samples: `5` (threshold `4`)
- median(R2_q_best): `0.998196` (threshold `0.996`)
- median(nonnegative_consistency): `1.000000` (threshold `0.98`)
- promote_to_full: **True**

## 4. Sample Result Snapshot

| sample_id | best_degree | r2_q_best | rmse_q_best | rmse_dqdv_high_best | sign_consistency | passes_physical_checks |
| --- | --- | --- | --- | --- | --- | --- |
| 4_8C-80PER_4_8C|465027|10 | 9 | 0.997543 | 0.015737 | 1.551501 | 1.000000 | 1 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE|737345|638 | 9 | 0.998196 | 0.012729 | 0.707494 | 1.000000 | 1 |
| 5C_67PER_4C_NEWSTRUCTURE|737329|626 | 9 | 0.998362 | 0.011534 | 0.646756 | 1.000000 | 1 |
| 5_6C_19PER_4_6C_NEWSTRUCTURE|737287|238 | 9 | 0.997798 | 0.014359 | 0.782461 | 1.000000 | 1 |
| 6C-50PER_3C|460507|685 | 9 | 0.998657 | 0.009731 | 0.524186 | 1.000000 | 1 |

## 5. Output Files

- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_polyfit_experiment\sample_selection.csv`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_polyfit_experiment\per_degree_metrics.csv`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_polyfit_experiment\best_degree_summary.csv`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_polyfit_experiment\highres_dqdv_curves.csv`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_polyfit_experiment\fit_report.md`