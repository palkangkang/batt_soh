# Balanced dQ/dV Example Validation

- Target: `policy=4_8C-80PER_4_8C`, `cell_code=465027`, `cycle=10`
- Balanced strategy: second-pass Savitzky-Golay `SG(15,3)` on top of current dQ/dV.

## 1. Cycle-10 Acceptance Checks

- current peak count: **3**
- balanced peak count: **1**
- peak-1 voltage drift: **0.000964 V** (threshold `0.015V`, pass=True)
- peak-1 |height|: current `13.133595` -> balanced `11.640604`
- peak-1 |area|: current `1.053363` -> balanced `1.053534`

## 2. Whole-Cell (All Cycles) Diagnostics

- valid dQ/dV cycles: **868**
- peak cycles current/balanced: **868 / 868** (no new empty cycles pass=True)
- median peak count current/balanced: **1.000 / 1.000**
- max |peak1_height| current/balanced: **13.133595 / 11.640604**
- mean roughness current/balanced: **0.085360 / 0.010968** (mean reduction ratio=86.93%)
- P95 |peak1 voltage drift| across cycles: **0.005365 V**

## 3. Output Files

- Figure: `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_feature_explanation\dqdv_balanced_comparison_cycle10.png`
- Cycle peak table: `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_feature_explanation\dqdv_balanced_peak_comparison_cycle10.csv`
- Cell cycle metrics: `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_feature_explanation\dqdv_balanced_cell_cycle_metrics.csv`