# Quick tuning report (RandomForest, no cycles)

## 1. Search setup
- Run time: 2026-03-19 13:06:55
- Trials: **8**
- Use cycles feature: **False**
- Policy-wise winsorization in tuning: **False**
- Rows train/valid: **98451 / 41831**
- Feature count: **54**
- Objective: maximize `valid_r2`

## 2. Best trial
- Best valid R2: **0.862192**, valid RMSE: **0.019571**, valid MAE: **0.014512**
- Train R2: **0.976241** (gap=0.114048)
- Best params:
  - `n_estimators=100`
  - `max_depth=24`
  - `min_samples_leaf=3`
  - `min_samples_split=10`
  - `max_features=0.3`
  - `max_samples=0.85`
  - `criterion=squared_error`

## 3. Top 10 trials
| rank | valid_r2 | valid_rmse | valid_mae | train_r2 | r2_gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.862192 | 0.019571 | 0.014512 | 0.976241 | 0.114048 |
| 2 | 0.860783 | 0.019671 | 0.014606 | 0.978591 | 0.117808 |
| 3 | 0.860422 | 0.019697 | 0.014588 | 0.968567 | 0.108146 |
| 4 | 0.857762 | 0.019884 | 0.014743 | 0.963609 | 0.105847 |
| 5 | 0.856703 | 0.019958 | 0.014875 | 0.962481 | 0.105778 |
| 6 | 0.855575 | 0.020036 | 0.014770 | 0.972251 | 0.116676 |
| 7 | 0.850592 | 0.020379 | 0.015272 | 0.944223 | 0.093631 |
| 8 | 0.830923 | 0.021679 | 0.016248 | 0.916223 | 0.085300 |