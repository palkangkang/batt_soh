# dQdV vs 电压区间 DeltaAh：LSTM 本地结果对比（Colab Final）

## 1. 数据对齐与口径声明

- 主比较集合：两模型验证集键交集（`policy+cell_code+cycles`）。
- 双口径并行：`q_discharge` 直接比较；`retention` 对 deltaAh 侧按与 dQdV 相同规则换算。
- retention换算规则：先筛选 `0.3<=q_discharge<=1.3`，每个 `policy+cell_code` 用前 `5` 个有效循环的 `q_discharge` 中位数作为 `q_ref`。

| item | value |
| --- | --- |
| dQdV valid rows | 41874 |
| deltaAh valid rows | 41539 |
| intersection rows | 41539 |
| dQdV-only rows | 335 |
| deltaAh-only rows | 0 |
| q_ref missing in delta conversion | 0 |
| retention NaN rows in delta conversion | 0 |

## 2. 技术路线差异

| dimension | dqdv | delta_ah |
| --- | --- | --- |
| 特征语义 | 放电 dQ/dV 主峰形态 + 主峰温度 + cycle_index_norm | 充电电压区间 delta_ah + mask |
| 时间步输入维度 | 10 | 24 |
| 缺失处理 | 数值强制转换并以0填充（无显式mask） | 零填充 + 显式mask通道 |
| 训练目标 | retention（同时回写pred_q_discharge） | q_discharge |
| 标签过滤 | 0.3<=q<=1.3, retention∈[0.3,1.1], q_ref_cycles=5 | 0.3<=q<=1.3 |
| 模型超参 | hidden=192, layers=2, dropout=0.1, lr=0.0005 | hidden=128, layers=2, dropout=0.1, lr=0.001 |
| 收敛行为 | run_config最佳轮次=6, log最小valid_loss轮次=16 | run_config最佳轮次=20, log最小valid_loss轮次=20 |

- dQdV 报告输入维度行：`- 每个时间步输入维度：`10`（主峰9维 + cycle_index_norm）`
- deltaAh 报告输入维度行：`- 每个时间步输入维度：`24`（`12维 delta_ah + 12维 mask`）`
- valid_loss记录最小值：dQdV=0.000112993，deltaAh=0.001051025。
- run_config保存的best_valid_loss：dQdV=0.000156026，deltaAh=0.001051025。

## 3. 最终效果总览（主结果：交集样本）

| eval_scope | target | aggregation | method | n_samples | n_groups | mse | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intersection | q_discharge | macro | delta_ah_interval_lstm | 41539 | 52 | 0.001292 | 0.032443 | 0.027317 | 0.204996 |
| intersection | q_discharge | macro | dqdv_main_peak_lstm | 41539 | 52 | 0.000222 | 0.013445 | 0.010607 | 0.839849 |
| intersection | q_discharge | weighted | delta_ah_interval_lstm | 41539 | 52 | 0.001051 | 0.032420 | 0.025174 | 0.613330 |
| intersection | q_discharge | weighted | dqdv_main_peak_lstm | 41539 | 52 | 0.000176 | 0.013283 | 0.009734 | 0.935087 |
| intersection | retention | macro | delta_ah_interval_lstm | 41539 | 52 | 0.001120 | 0.030188 | 0.025417 | 0.204996 |
| intersection | retention | macro | dqdv_main_peak_lstm | 41539 | 52 | 0.000192 | 0.012512 | 0.009872 | 0.839849 |
| intersection | retention | weighted | delta_ah_interval_lstm | 41539 | 52 | 0.000905 | 0.030087 | 0.023384 | 0.560086 |
| intersection | retention | weighted | dqdv_main_peak_lstm | 41539 | 52 | 0.000153 | 0.012364 | 0.009052 | 0.925716 |

关键差异（交集主结果）:
- q_discharge/weighted: dQdV 相比 deltaAh，MSE 下降 0.000875 (83.212%)，R2 提升 0.321757。
- q_discharge/macro: dQdV 相比 deltaAh，MSE 下降 0.001071 (82.852%)，R2 提升 0.634853。
- retention/weighted: dQdV 相比 deltaAh，MSE 下降 0.000752 (83.114%)，R2 提升 0.365630。
- retention/macro: dQdV 相比 deltaAh，MSE 下降 0.000927 (82.817%)，R2 提升 0.634853。

## 4. 分层比较（交集样本）

### 4.1 按循环分位段

| cycle_bin | target | method | n_samples | mse | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0.999, 162.0] | q_discharge | delta_ah_interval_lstm | 8329 | 0.000897 | 0.029943 | 0.020481 | -0.112357 |
| (0.999, 162.0] | q_discharge | dqdv_main_peak_lstm | 8329 | 0.000295 | 0.017175 | 0.012806 | 0.634021 |
| (0.999, 162.0] | retention | delta_ah_interval_lstm | 8329 | 0.000771 | 0.027775 | 0.019037 | -6.604625 |
| (0.999, 162.0] | retention | dqdv_main_peak_lstm | 8329 | 0.000255 | 0.015976 | 0.011905 | -1.515933 |
| (162.0, 326.0] | q_discharge | delta_ah_interval_lstm | 8323 | 0.001051 | 0.032424 | 0.024819 | 0.071919 |
| (162.0, 326.0] | q_discharge | dqdv_main_peak_lstm | 8323 | 0.000151 | 0.012304 | 0.009241 | 0.866353 |
| (162.0, 326.0] | retention | delta_ah_interval_lstm | 8323 | 0.000905 | 0.030087 | 0.023047 | -1.178984 |
| (162.0, 326.0] | retention | dqdv_main_peak_lstm | 8323 | 0.000132 | 0.011489 | 0.008600 | 0.682267 |
| (326.0, 501.0] | q_discharge | delta_ah_interval_lstm | 8283 | 0.001058 | 0.032527 | 0.025887 | 0.590478 |
| (326.0, 501.0] | q_discharge | dqdv_main_peak_lstm | 8283 | 0.000156 | 0.012472 | 0.008187 | 0.939791 |
| (326.0, 501.0] | retention | delta_ah_interval_lstm | 8283 | 0.000907 | 0.030123 | 0.024000 | 0.573959 |
| (326.0, 501.0] | retention | dqdv_main_peak_lstm | 8283 | 0.000133 | 0.011542 | 0.007584 | 0.937451 |
| (501.0, 734.0] | q_discharge | delta_ah_interval_lstm | 8301 | 0.000965 | 0.031058 | 0.025492 | 0.454934 |
| (501.0, 734.0] | q_discharge | dqdv_main_peak_lstm | 8301 | 0.000100 | 0.010012 | 0.007967 | 0.943356 |
| (501.0, 734.0] | retention | delta_ah_interval_lstm | 8301 | 0.000825 | 0.028717 | 0.023640 | 0.438593 |
| (501.0, 734.0] | retention | dqdv_main_peak_lstm | 8301 | 0.000086 | 0.009298 | 0.007407 | 0.941140 |
| (734.0, 2237.0] | q_discharge | delta_ah_interval_lstm | 8303 | 0.001285 | 0.035848 | 0.029209 | 0.507702 |
| (734.0, 2237.0] | q_discharge | dqdv_main_peak_lstm | 8303 | 0.000180 | 0.013404 | 0.010453 | 0.931175 |
| (734.0, 2237.0] | retention | delta_ah_interval_lstm | 8303 | 0.001118 | 0.033435 | 0.027212 | 0.519655 |
| (734.0, 2237.0] | retention | dqdv_main_peak_lstm | 8303 | 0.000157 | 0.012532 | 0.009750 | 0.932515 |

### 4.2 按工况组（VARCHARGE vs 非VARCHARGE）

| policy_group | target | method | n_samples | mse | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NON_VARCHARGE | q_discharge | delta_ah_interval_lstm | 39905 | 0.000995 | 0.031544 | 0.024629 | 0.628428 |
| NON_VARCHARGE | q_discharge | dqdv_main_peak_lstm | 39905 | 0.000162 | 0.012748 | 0.009450 | 0.939319 |
| NON_VARCHARGE | retention | delta_ah_interval_lstm | 39905 | 0.000852 | 0.029190 | 0.022834 | 0.589491 |
| NON_VARCHARGE | retention | dqdv_main_peak_lstm | 39905 | 0.000140 | 0.011842 | 0.008774 | 0.932440 |
| VARCHARGE | q_discharge | delta_ah_interval_lstm | 1634 | 0.002418 | 0.049175 | 0.038497 | 0.056920 |
| VARCHARGE | q_discharge | dqdv_main_peak_lstm | 1634 | 0.000517 | 0.022737 | 0.016669 | 0.798384 |
| VARCHARGE | retention | delta_ah_interval_lstm | 1634 | 0.002203 | 0.046941 | 0.036811 | -0.372016 |
| VARCHARGE | retention | dqdv_main_peak_lstm | 1634 | 0.000461 | 0.021476 | 0.015829 | 0.712804 |

## 5. 一致性校验与附录

### 5.1 与原train_valid_metrics.csv一致性（各自全样本，q_discharge）

| method | saved_mse | recalc_mse | abs_diff_mse | saved_rmse | recalc_rmse | abs_diff_rmse | saved_mae | recalc_mae | abs_diff_mae | saved_r2 | recalc_r2 | abs_diff_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqdv_main_peak_lstm | 0.000180 | 0.000180 | 0.000000 | 0.013421 | 0.013421 | 0.000000 | 0.009797 | 0.009797 | 0.000000 | 0.935550 | 0.935550 | 0.000000 |
| delta_ah_interval_lstm | 0.001051 | 0.001051 | 0.000000 | 0.032420 | 0.032420 | 0.000000 | 0.025174 | 0.025174 | 0.000000 | 0.613330 | 0.613330 | 0.000000 |

### 5.2 全样本附录（不用于主优劣判定）

| eval_scope | target | aggregation | method | n_samples | n_groups | mse | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqdv_full_valid | q_discharge | macro | dqdv_main_peak_lstm | 41874 | 52 | 0.000222 | 0.013467 | 0.010607 | 0.841564 |
| dqdv_full_valid | q_discharge | weighted | dqdv_main_peak_lstm | 41874 | 52 | 0.000180 | 0.013421 | 0.009797 | 0.935550 |
| dqdv_full_valid | retention | macro | dqdv_main_peak_lstm | 41874 | 52 | 0.000193 | 0.012532 | 0.009871 | 0.841564 |
| dqdv_full_valid | retention | weighted | dqdv_main_peak_lstm | 41874 | 52 | 0.000156 | 0.012491 | 0.009110 | 0.926793 |
| delta_full_valid | q_discharge | macro | delta_ah_interval_lstm | 41539 | 52 | 0.001292 | 0.032443 | 0.027317 | 0.204996 |
| delta_full_valid | q_discharge | weighted | delta_ah_interval_lstm | 41539 | 52 | 0.001051 | 0.032420 | 0.025174 | 0.613330 |
| delta_full_valid | retention | macro | delta_ah_interval_lstm | 41539 | 52 | 0.001120 | 0.030188 | 0.025417 | 0.204996 |
| delta_full_valid | retention | weighted | delta_ah_interval_lstm | 41539 | 52 | 0.000905 | 0.030087 | 0.023384 | 0.560086 |

## 6. 结论与风险

- 在交集主评估集上，dQdV路线在 q 与 retention 两口径下均明显优于 deltaAh 路线。
- 该结论在窗口级加权与 policy+cell 宏平均两种统计方式下方向一致，稳健性较高。
- 风险1：deltaAh 的 retention 为后验换算，不是其训练原生目标；该口径用于业务解释有效，但不等同于直接训练 retention。
- 风险2：dQdV run_config记录的best_epoch与log最小loss轮次不一致，可能由`min_delta`择优逻辑触发，应在复训时统一best定义。
- 风险3：交集评估最公平，但会排除 dQdV 独有的335条验证样本；附录全样本结果已保留用于完整性参考。
