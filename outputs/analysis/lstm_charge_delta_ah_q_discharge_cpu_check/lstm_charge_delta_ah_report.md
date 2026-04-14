# LSTM Report: charge delta_ah -> q_discharge

## 1. Run Summary
- Time: 2026-04-07 16:18:23
- Python: `C:\Users\pal\.virtualenvs\colab-OixbOpvz\Scripts\python.exe`
- Device: `cpu`
- Window size: `30`
- Input dim per step: `24` (`12 delta_ah + 12 mask`)
- Label filter: `0.3 <= q_discharge <= 1.3`

## 2. Data Profile
- Cycle-level merged rows: **139,718**
- Train windows: **512**
- Valid windows: **256**
- Voltage ranges:
  - `[3.00,3.05)`
  - `[3.05,3.10)`
  - `[3.10,3.15)`
  - `[3.15,3.20)`
  - `[3.20,3.25)`
  - `[3.25,3.30)`
  - `[3.30,3.35)`
  - `[3.35,3.40)`
  - `[3.40,3.45)`
  - `[3.45,3.50)`
  - `[3.50,3.55)`
  - `[3.55,3.60]`

## 3. Metrics
| set_type | n_windows | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|---:|
| train | 512 | 0.59704924 | 0.772690 | 0.771244 | -250.715775 |
| valid | 256 | 0.59735537 | 0.772888 | 0.771183 | -212.059601 |

## 4. Key Figures
- Best epoch by valid loss: **1**
![loss_curve](./loss_curve.png)

![valid_scatter](./valid_scatter.png)

## 5. Notes
- This run uses only charge interval `delta_ah` features.
- Missing intervals are handled by zero-fill + explicit mask channels.