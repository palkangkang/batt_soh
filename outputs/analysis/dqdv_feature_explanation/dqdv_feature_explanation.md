# dQ/dV Feature Extraction Notes (Discharge)

Example target: `policy=4_8C-80PER_4_8C`, `cell_code=465027`, `cycles=10`

![dQ/dV feature illustration](C:/Users/pal/projects/batt_soh/outputs/analysis/dqdv_feature_explanation/dqdv_feature_extraction_illustration.png)

## 1. Calculation Workflow

1. Keep discharge rows (`flag_dischg=1`), sort by `ts`, and clip to the voltage window (`3.6V~2.8V` by default).
2. Compute first-order difference:

```text
dQ/dV_i = (Q_{i+1} - Q_i) / (V_{i+1} - V_i)
```

3. Keep valid differentials only (`dt>0`, `dV<0`, and `|dV|` above threshold), then aggregate duplicate voltages.
4. Apply Savitzky-Golay smoothing, then detect peaks on `-dQ/dV` using `find_peaks`.
5. Export Top-3 peak statistics:
   - Position: `peak*_voltage_v`
   - Height: `peak*_height_dqdv` (signed)
   - Area: `peak*_area` (signed integral between left/right bases)
   - Prominence/width: `peak*_prominence`, `peak*_width_v`

## 2. Example Values (from exported feature file)

| Field | Value |
|---|---:|
| `n_points_window` | 251 |
| `n_points_dqdv` | 186 |
| `n_peaks_detected` | 3 |
| `peak1_voltage_v` | 3.152537 |
| `peak1_height_dqdv` | -13.133595 |
| `peak1_area` | -1.053363 |
| `peak1_prominence` | 12.834676 |
| `peak1_width_v` | 0.007522 |
| `peak2_voltage_v` | 3.157142 |
| `peak2_height_dqdv` | -7.396323 |
| `peak2_area` | -0.234973 |
| `peak3_voltage_v` | 3.144680 |
| `peak3_height_dqdv` | -6.934502 |
| `peak3_area` | -0.728197 |

## 3. Peak-1 Area Definition

- Peak-1 base index range: `[0, 185]`, approx voltage range `[3.4338, 2.8086] V`.
- Signed area is computed by numeric integration of `dQ/dV` over voltage: peak-1 area is `-1.053363 Ah` in this example.