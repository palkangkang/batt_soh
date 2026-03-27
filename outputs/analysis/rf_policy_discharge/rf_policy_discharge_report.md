# RandomForest: policy + first-occurrence discharge features

## 1. Data and constraints
- Run time: 2026-03-19 13:11:15
- Python executable: `C:\Users\pal\.virtualenvs\colab-OixbOpvz\Scripts\python.exe`
- Font fallback: `DejaVu Sans`
- Outlier filter: remove samples with `q_discharge > 1.5`
- Life rows before/after outlier filter: **140,623 / 140,612**
- Outlier rows removed: **11**
- Total cycle-level rows used: **140,282**
- First-occurrence rule: only use rows with `range_count == 1`
- Dropped unusable feature columns (all-NaN in train): **9**
- Model strategy: RandomForest ensemble average (3 seeds)
- Ensemble seeds: `[20260318, 20260319, 20260320]`
- Policy-wise winsorization on discharge-derived features: **Disabled**

## 2. Feature design
- Policy triad features: **3**
- Discharge interval `delta_ah` features: **16**
- Discharge interval duration features: **16**
- Discharge interval temperature features: **16**
- Discharge group-stat features: **12**
- Include `cycles` feature: **No**
- Total feature count: **63**

## 3. Train vs valid metrics
| set | n_rows | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| train | 98451 | 0.004947 | 0.008355 | 0.976387 |
| valid | 41831 | 0.014514 | 0.019594 | 0.861875 |

## 4. Scatter plot
![train_valid_scatter](./fit_scatter_train_valid.png)