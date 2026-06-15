# compact2 / compact3 / compact4 是否需要扩展的短链路判断报告

## 1. 结论摘要

- 本次采用固定 LightGBM 表格模型、同一 train/valid split、同一 55维推荐工况特征包，比较 compact2、compact3、compact4 三种 dQdV target pack。
- 部署链路 `55维工况 -> 预测dQdV -> retention` 的最佳 target pack 为 `compact4_area_height_voltage_skewness`，valid R2 = `0.897059`。
- oracle 上限 `真实dQdV -> retention` 的最佳 target pack 为 `compact4_area_height_voltage_skewness`，valid R2 = `0.918027`。
- compact4 相比 compact2 的 deployable R2 变化为 `0.036851`。
- direct 55维 retention baseline 仍显著高于所有 dQdV bridge，说明当前不应把 compact3/compact4 扩展视为主预测路径替代 direct retention。

## 2. target pack 定义

| target_pack | dQdV targets |
| --- | --- |
| compact2_area_height | main_peak_area, main_peak_height_dqdv |
| compact3_area_height_voltage | main_peak_area, main_peak_height_dqdv, main_peak_voltage_v |
| compact4_area_height_voltage_skewness | main_peak_area, main_peak_height_dqdv, main_peak_voltage_v, main_peak_skewness |

## 3. 汇总指标

| target_pack | target_dim | recommended55_dqdv_mean_valid_r2 | full159_dqdv_mean_valid_r2 | oracle_bridge_valid_r2 | deployable_bridge_55_valid_r2 | deployable_bridge_159_valid_r2 | direct55_valid_r2 | direct159_valid_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact2_area_height | 2 | 0.919244 | 0.919362 | 0.867750 | 0.860208 | 0.866331 | 0.939471 | 0.945561 |
| compact3_area_height_voltage | 3 | 0.861383 | 0.864816 | 0.889140 | 0.892527 | 0.888976 | 0.942919 | 0.943898 |
| compact4_area_height_voltage_skewness | 4 | 0.781099 | 0.793294 | 0.918027 | 0.897059 | 0.903932 | 0.941887 | 0.946469 |

![retention_r2_by_target_pack](C:/Users/pal/projects/batt_soh/outputs/analysis/compact_target_pack_retention_decision/retention_r2_by_target_pack.png)

![dqdv_mean_r2_by_target_pack](C:/Users/pal/projects/batt_soh/outputs/analysis/compact_target_pack_retention_decision/dqdv_mean_r2_by_target_pack.png)

![recommended55_dqdv_target_r2_heatmap](C:/Users/pal/projects/batt_soh/outputs/analysis/compact_target_pack_retention_decision/recommended55_dqdv_target_r2_heatmap.png)

## 4. 分目标 dQdV 可预测性

| target_pack | input_pack | target | model_name | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact2_area_height | full159 | main_peak_area | full159_to_dqdv_lightgbm | 0.926660 | 0.030568 | 0.021829 | 0.000934 |
| compact2_area_height | full159 | main_peak_height_dqdv | full159_to_dqdv_lightgbm | 0.912064 | 0.110981 | 0.078356 | 0.012317 |
| compact2_area_height | recommended55 | main_peak_area | recommended55_to_dqdv_lightgbm | 0.927339 | 0.030427 | 0.022412 | 0.000926 |
| compact2_area_height | recommended55 | main_peak_height_dqdv | recommended55_to_dqdv_lightgbm | 0.911149 | 0.111558 | 0.080024 | 0.012445 |
| compact3_area_height_voltage | full159 | main_peak_area | full159_to_dqdv_lightgbm | 0.926801 | 0.030539 | 0.021910 | 0.000933 |
| compact3_area_height_voltage | full159 | main_peak_height_dqdv | full159_to_dqdv_lightgbm | 0.913629 | 0.109989 | 0.077133 | 0.012098 |
| compact3_area_height_voltage | full159 | main_peak_voltage_v | full159_to_dqdv_lightgbm | 0.754016 | 0.009264 | 0.005882 | 0.000086 |
| compact3_area_height_voltage | recommended55 | main_peak_area | recommended55_to_dqdv_lightgbm | 0.923623 | 0.031195 | 0.022847 | 0.000973 |
| compact3_area_height_voltage | recommended55 | main_peak_height_dqdv | recommended55_to_dqdv_lightgbm | 0.911030 | 0.111632 | 0.079716 | 0.012462 |
| compact3_area_height_voltage | recommended55 | main_peak_voltage_v | recommended55_to_dqdv_lightgbm | 0.749497 | 0.009349 | 0.005995 | 0.000087 |
| compact4_area_height_voltage_skewness | full159 | main_peak_area | full159_to_dqdv_lightgbm | 0.926551 | 0.030591 | 0.021828 | 0.000936 |
| compact4_area_height_voltage_skewness | full159 | main_peak_height_dqdv | full159_to_dqdv_lightgbm | 0.911816 | 0.111138 | 0.078035 | 0.012352 |
| compact4_area_height_voltage_skewness | full159 | main_peak_skewness | full159_to_dqdv_lightgbm | 0.580333 | 0.016245 | 0.008723 | 0.000264 |
| compact4_area_height_voltage_skewness | full159 | main_peak_voltage_v | full159_to_dqdv_lightgbm | 0.754476 | 0.009255 | 0.005880 | 0.000086 |
| compact4_area_height_voltage_skewness | recommended55 | main_peak_area | recommended55_to_dqdv_lightgbm | 0.922735 | 0.031376 | 0.022950 | 0.000984 |
| compact4_area_height_voltage_skewness | recommended55 | main_peak_height_dqdv | recommended55_to_dqdv_lightgbm | 0.907749 | 0.113672 | 0.080959 | 0.012921 |
| compact4_area_height_voltage_skewness | recommended55 | main_peak_skewness | recommended55_to_dqdv_lightgbm | 0.549659 | 0.016828 | 0.009271 | 0.000283 |
| compact4_area_height_voltage_skewness | recommended55 | main_peak_voltage_v | recommended55_to_dqdv_lightgbm | 0.744254 | 0.009446 | 0.006085 | 0.000089 |

## 5. retention 链路对比

| target_pack | stage | input_pack | model_name | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact2_area_height | direct_retention | full159 | direct_retention_159 | 0.945561 | 0.010772 | 0.006697 | 0.000116 |
| compact2_area_height | direct_retention | recommended55 | direct_retention_55 | 0.939471 | 0.011358 | 0.007306 | 0.000129 |
| compact2_area_height | retention_bridge | predicted_dqdv_full159 | deployable_bridge_159 | 0.866331 | 0.016879 | 0.011543 | 0.000285 |
| compact2_area_height | retention_bridge | predicted_dqdv_recommended55 | deployable_bridge_55 | 0.860208 | 0.017261 | 0.011662 | 0.000298 |
| compact2_area_height | retention_bridge | true_dqdv | oracle_bridge | 0.867750 | 0.016789 | 0.011640 | 0.000282 |
| compact3_area_height_voltage | direct_retention | full159 | direct_retention_159 | 0.943898 | 0.010935 | 0.006766 | 0.000120 |
| compact3_area_height_voltage | direct_retention | recommended55 | direct_retention_55 | 0.942919 | 0.011030 | 0.007200 | 0.000122 |
| compact3_area_height_voltage | retention_bridge | predicted_dqdv_full159 | deployable_bridge_159 | 0.888976 | 0.015383 | 0.010287 | 0.000237 |
| compact3_area_height_voltage | retention_bridge | predicted_dqdv_recommended55 | deployable_bridge_55 | 0.892527 | 0.015135 | 0.010118 | 0.000229 |
| compact3_area_height_voltage | retention_bridge | true_dqdv | oracle_bridge | 0.889140 | 0.015371 | 0.010577 | 0.000236 |
| compact4_area_height_voltage_skewness | direct_retention | full159 | direct_retention_159 | 0.946469 | 0.010681 | 0.006728 | 0.000114 |
| compact4_area_height_voltage_skewness | direct_retention | recommended55 | direct_retention_55 | 0.941887 | 0.011129 | 0.007291 | 0.000124 |
| compact4_area_height_voltage_skewness | retention_bridge | predicted_dqdv_full159 | deployable_bridge_159 | 0.903932 | 0.014309 | 0.009653 | 0.000205 |
| compact4_area_height_voltage_skewness | retention_bridge | predicted_dqdv_recommended55 | deployable_bridge_55 | 0.897059 | 0.014812 | 0.009921 | 0.000219 |
| compact4_area_height_voltage_skewness | retention_bridge | true_dqdv | oracle_bridge | 0.918027 | 0.013218 | 0.008835 | 0.000175 |

## 6. 判断

- 需要 compact3 吗：需要作为最低增强版。`deployable_bridge_55` 从 compact2 的 `0.860208` 提升到 compact3 的 `0.892527`，增益为 `+0.032319`，说明 `main_peak_voltage_v` 对 retention 有明显信息增益。
- 需要 compact4 吗：如果继续走 dQdV 中介路线，建议优先 compact4。compact4 的 `oracle_bridge` R2 为 `0.918027`，`deployable_bridge_55` R2 为 `0.897059`，均为三组 target pack 中最高；相比 compact3，deployable R2 仍有 `+0.004532` 的小幅提升。
- compact4 的代价：`recommended55 -> dQdV` 平均 valid R2 从 compact3 的 `0.861383` 降到 compact4 的 `0.781099`，主要由 `main_peak_skewness` 较难预测造成，其 valid R2 为 `0.549659`。因此 compact4 不适合简单追求 dQdV 平均 R2，但适合追求 retention bridge 的信息完整性。
- direct baseline 仍然更强：compact4 的 `deployable_bridge_55` R2 为 `0.897059`，仍低于 `direct_retention_55` 的 `0.941887`。因此 compact4 可以作为解释性中介或多任务辅助，不建议替代 direct retention 主路径。
- 最终建议：后续若继续做 dQdV 中介，不要再优先 compact2；短期建议用 compact4 做中介表征，并同时保留 compact3 作为更稳健的低维备选。下一步重点不是直接上长窗口 LSTM，而是验证 compact4 中 `skewness` 的噪声鲁棒性，以及尝试 weighted loss 或 predicted-dQdV-aware bridge。
