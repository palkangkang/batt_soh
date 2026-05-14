# 单调物理约束 + LSTM 多步 retention 预测验证报告

## 1. 任务摘要

- history_len: `100`
- horizon: `50`
- block_stride: `150`
- sample_mode: `non_overlapping_blocks`
- feature_pack: `recommended55`
- target_pack: `compact4`

本报告验证“未来容量保持率整体单调不升”这一物理约束是否能改善 H1:H50 retention 预测。retention 指容量保持率，即当前放电容量除以同一电芯参考容量；H50 指未来第 50 个预测步。单调不升指预测曲线满足 H1 >= H2 >= ... >= H50。

## 2. 术语说明

- `recommended55`：55 个工况统计特征，不包含 `cycles`、`policy`、`cell_code` 或 policy 三元参数。
- `compact4`：4 个 dQdV 中介特征，包括 `main_peak_area`、`main_peak_height_dqdv`、`main_peak_voltage_v`、`main_peak_skewness`。
- `direct LightGBM`：用 55 维工况 summary 直接预测未来 retention。
- `dQdV bridge`：先预测未来 compact4 dQdV，再用 dQdV 预测未来 retention。
- `linear_last10`：用历史最后 10 个 retention 点做线性外推。
- `cummin`：从 H1 到 H50 对预测值做累计最小值，保证曲线不再上升。
- `isotonic`：对单条预测曲线做单调不升的最小二乘投影。
- `bounded_monotonic`：先限制预测不超过历史最后一个 retention，再做单调不升投影。
- `monotonic LSTM penalty`：LSTM 直接输出未来 retention，并在 loss 中惩罚上升段和曲线抖动。
- `monotonic LSTM delta strict`：LSTM 输出非负衰减增量，使用历史最后 retention 作为递推起点，不把历史 retention 当作输入特征。
- `monotonic LSTM delta with history retention`：LSTM 输入为 100x56，额外包含历史 retention 观测，因此不是纯工况输入模型。

## 3. 数据检查

| check_item | value | expected | pass_flag | details |
| --- | --- | --- | --- | --- |
| sample_mode | non_overlapping_blocks | non_overlapping_blocks | 1 |  |
| history_len | 100 | 100 | 1 | expected 100 outside smoke |
| horizon | 50 | 50 | 1 | expected 50 outside smoke |
| block_stride | 150 | 150 | 1 | expected history_len+horizon |
| feature_count | 55 | 55 | 1 | recommended55 |
| target_pack | compact4 | compact4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| target_dim | 4 | 4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| forbidden_input_columns_present | 0 | 0 | 1 |  |
| train_block_count | 580 | >0 | 1 |  |
| valid_block_count | 251 | >0 | 1 |  |

## 4. Stage 0 单调性诊断

| series | monotonic_violation_count | monotonic_violation_rate | max_positive_jump | mean_positive_jump | total_positive_jump | curve_has_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| true_retention | 2931 | 0.238312 | 0.013029 | 0.000436 | 1.279094 | 0.892430 |
| direct_retention | 5625 | 0.457354 | 0.020746 | 0.001551 | 8.721563 | 1.000000 |
| deployable_bridge | 6067 | 0.493292 | 0.058840 | 0.006242 | 37.872140 | 1.000000 |
| linear_last10 | 1372 | 0.111554 | 0.000225 | 0.000080 | 0.110440 | 0.111554 |
| persistence | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_bridge | 6071 | 0.493617 | 0.059111 | 0.005908 | 35.868991 | 1.000000 |

真实 retention 的曲线违反率为 `0.892430`。这表示观测标签本身不严格单调，单调约束在本任务中更像物理去噪假设，而不是逐点标签真值的硬事实。

![valid monotonic curves before after](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/valid_monotonic_curves_before_after.png)

图 1 说明：X 轴是未来 horizon step，即 H1 到 H50；Y 轴是 retention；黑线是真实 retention，其余曲线是原始预测和单调后处理预测。关键结论：若后处理曲线更贴近黑线且不再上升，说明单调约束有效；若偏离更大，说明真实短期波动不可忽略。

## 5. Stage 1 单调后处理指标

| method | horizon | rmse | mae | mse | r2 | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| deployable_bridge | all | 0.017016 | 0.011842 | 0.000290 | 0.826557 | 0.493292 |
| deployable_bridge | H1 | 0.016506 | 0.011077 | 0.000272 | 0.766816 | 0.493292 |
| deployable_bridge | H10 | 0.016270 | 0.011539 | 0.000265 | 0.798226 | 0.493292 |
| deployable_bridge | H20 | 0.016821 | 0.011530 | 0.000283 | 0.811395 | 0.493292 |
| deployable_bridge | H50 | 0.018901 | 0.013402 | 0.000357 | 0.844523 | 0.493292 |
| deployable_bridge_bounded_monotonic | all | 0.010473 | 0.005949 | 0.000110 | 0.934298 | 0.000000 |
| deployable_bridge_bounded_monotonic | H1 | 0.007243 | 0.002754 | 0.000052 | 0.955097 | 0.000000 |
| deployable_bridge_bounded_monotonic | H10 | 0.008320 | 0.004277 | 0.000069 | 0.947238 | 0.000000 |
| deployable_bridge_bounded_monotonic | H20 | 0.009488 | 0.005371 | 0.000090 | 0.939994 | 0.000000 |
| deployable_bridge_bounded_monotonic | H50 | 0.014184 | 0.009319 | 0.000201 | 0.912440 | 0.000000 |
| deployable_bridge_cummin | all | 0.017063 | 0.012199 | 0.000291 | 0.825596 | 0.000000 |
| deployable_bridge_cummin | H1 | 0.016506 | 0.011077 | 0.000272 | 0.766816 | 0.000000 |
| deployable_bridge_cummin | H10 | 0.015544 | 0.010896 | 0.000242 | 0.815832 | 0.000000 |
| deployable_bridge_cummin | H20 | 0.016808 | 0.011958 | 0.000283 | 0.811676 | 0.000000 |
| deployable_bridge_cummin | H50 | 0.018288 | 0.013403 | 0.000334 | 0.854438 | 0.000000 |
| deployable_bridge_isotonic | all | 0.015892 | 0.010954 | 0.000253 | 0.848722 | 0.000000 |
| deployable_bridge_isotonic | H1 | 0.016969 | 0.011430 | 0.000288 | 0.753561 | 0.000000 |
| deployable_bridge_isotonic | H10 | 0.015409 | 0.010553 | 0.000237 | 0.819009 | 0.000000 |
| deployable_bridge_isotonic | H20 | 0.015549 | 0.010587 | 0.000242 | 0.838839 | 0.000000 |
| deployable_bridge_isotonic | H50 | 0.016771 | 0.011951 | 0.000281 | 0.877582 | 0.000000 |
| direct_retention | all | 0.012026 | 0.007656 | 0.000145 | 0.913362 | 0.457354 |
| direct_retention | H1 | 0.011807 | 0.007148 | 0.000139 | 0.880700 | 0.457354 |
| direct_retention | H10 | 0.011628 | 0.007326 | 0.000135 | 0.896934 | 0.457354 |
| direct_retention | H20 | 0.011986 | 0.007459 | 0.000144 | 0.904233 | 0.457354 |
| direct_retention | H50 | 0.012430 | 0.008257 | 0.000154 | 0.932762 | 0.457354 |
| direct_retention_bounded_monotonic | all | 0.009892 | 0.005869 | 0.000098 | 0.941390 | 0.000000 |
| direct_retention_bounded_monotonic | H1 | 0.007101 | 0.003134 | 0.000050 | 0.956848 | 0.000000 |
| direct_retention_bounded_monotonic | H10 | 0.007916 | 0.004432 | 0.000063 | 0.952230 | 0.000000 |
| direct_retention_bounded_monotonic | H20 | 0.009309 | 0.005560 | 0.000087 | 0.942232 | 0.000000 |
| direct_retention_bounded_monotonic | H50 | 0.011801 | 0.007753 | 0.000139 | 0.939391 | 0.000000 |
| direct_retention_cummin | all | 0.012356 | 0.008009 | 0.000153 | 0.908546 | 0.000000 |
| direct_retention_cummin | H1 | 0.011807 | 0.007148 | 0.000139 | 0.880700 | 0.000000 |
| direct_retention_cummin | H10 | 0.012062 | 0.007634 | 0.000145 | 0.889100 | 0.000000 |
| direct_retention_cummin | H20 | 0.012249 | 0.007766 | 0.000150 | 0.899990 | 0.000000 |
| direct_retention_cummin | H50 | 0.012674 | 0.008694 | 0.000161 | 0.930087 | 0.000000 |
| direct_retention_isotonic | all | 0.011874 | 0.007505 | 0.000141 | 0.915544 | 0.000000 |
| direct_retention_isotonic | H1 | 0.011778 | 0.007109 | 0.000139 | 0.881282 | 0.000000 |
| direct_retention_isotonic | H10 | 0.011443 | 0.007119 | 0.000131 | 0.900192 | 0.000000 |
| direct_retention_isotonic | H20 | 0.011677 | 0.007285 | 0.000136 | 0.909101 | 0.000000 |
| direct_retention_isotonic | H50 | 0.012305 | 0.008205 | 0.000151 | 0.934104 | 0.000000 |
| linear_last10 | all | 0.004748 | 0.001937 | 0.000023 | 0.986496 | 0.111554 |
| linear_last10 | H1 | 0.000795 | 0.000373 | 0.000001 | 0.999459 | 0.111554 |
| linear_last10 | H10 | 0.001578 | 0.000785 | 0.000002 | 0.998101 | 0.111554 |
| linear_last10 | H20 | 0.003177 | 0.001475 | 0.000010 | 0.993274 | 0.111554 |
| linear_last10 | H50 | 0.008564 | 0.004065 | 0.000073 | 0.968077 | 0.111554 |
| linear_last10_bounded_monotonic | all | 0.004456 | 0.001755 | 0.000020 | 0.988106 | 0.000000 |
| linear_last10_bounded_monotonic | H1 | 0.000802 | 0.000384 | 0.000001 | 0.999450 | 0.000000 |
| linear_last10_bounded_monotonic | H10 | 0.001443 | 0.000715 | 0.000002 | 0.998414 | 0.000000 |
| linear_last10_bounded_monotonic | H20 | 0.002925 | 0.001340 | 0.000009 | 0.994298 | 0.000000 |
| linear_last10_bounded_monotonic | H50 | 0.008133 | 0.003697 | 0.000066 | 0.971213 | 0.000000 |
| linear_last10_cummin | all | 0.004467 | 0.001756 | 0.000020 | 0.988047 | 0.000000 |
| linear_last10_cummin | H1 | 0.000795 | 0.000373 | 0.000001 | 0.999459 | 0.000000 |
| linear_last10_cummin | H10 | 0.001447 | 0.000713 | 0.000002 | 0.998403 | 0.000000 |
| linear_last10_cummin | H20 | 0.002936 | 0.001329 | 0.000009 | 0.994256 | 0.000000 |
| linear_last10_cummin | H50 | 0.008147 | 0.003709 | 0.000066 | 0.971110 | 0.000000 |
| linear_last10_isotonic | all | 0.004690 | 0.001938 | 0.000022 | 0.986824 | 0.000000 |
| linear_last10_isotonic | H1 | 0.001328 | 0.000571 | 0.000002 | 0.998490 | 0.000000 |
| linear_last10_isotonic | H10 | 0.001914 | 0.000913 | 0.000004 | 0.997209 | 0.000000 |
| linear_last10_isotonic | H20 | 0.003270 | 0.001524 | 0.000011 | 0.992874 | 0.000000 |
| linear_last10_isotonic | H50 | 0.008312 | 0.003867 | 0.000069 | 0.969929 | 0.000000 |

- direct LightGBM + cummin 的 H50 RMSE 变化：`0.000245`，负数代表提升。
- linear_last10 + cummin 的 H50 RMSE 变化：`-0.000417`，负数代表提升。
- dQdV bridge + cummin 的 H50 RMSE 变化：`-0.000613`，负数代表提升。

![postprocess H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/postprocess_h50_scatter.png)

图 2 说明：X 轴是真实 H50 retention；Y 轴是预测 H50 retention；虚线是理想预测 `Y=X`；每个点代表一个 valid block。关键结论：点云越贴近虚线，H50 精度越高。

![postprocess H50 residual distribution](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/postprocess_h50_residual_distribution.png)

图 3 说明：X 轴是 H50 残差 `真实 retention - 预测 retention`；Y 轴是 block 数量；黑色虚线是 0 残差。关键结论：分布越窄且越靠近 0，后处理越有效。

![postprocess H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/postprocess_h50_residual_vs_true.png)

图 4 说明：X 轴是真实 H50 retention；Y 轴是残差。若残差随真实 retention 呈结构性斜率，说明模型在不同衰减阶段有系统偏差。

![postprocess selected curves](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/postprocess_curves_selected_blocks.png)

图 5 说明：X 轴是 H1:H50，Y 轴是 retention；黑线是真实曲线，彩色线是原始和单调后处理曲线。关键结论：该图用于判断单调约束是修正预测抖动，还是过度压低未来预测。

## 6. Stage 2/3 LSTM 指标

| method | horizon | rmse | mae | mse | r2 | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| monotonic_lstm_delta_strict | all | 0.005087 | 0.002596 | 0.000026 | 0.984502 | 0.000000 |
| monotonic_lstm_delta_strict | H1 | 0.000839 | 0.000408 | 0.000001 | 0.999397 | 0.000000 |
| monotonic_lstm_delta_strict | H10 | 0.001861 | 0.001110 | 0.000003 | 0.997361 | 0.000000 |
| monotonic_lstm_delta_strict | H20 | 0.003243 | 0.001924 | 0.000011 | 0.992988 | 0.000000 |
| monotonic_lstm_delta_strict | H50 | 0.009093 | 0.005257 | 0.000083 | 0.964018 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | all | 0.004574 | 0.002429 | 0.000021 | 0.987466 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H1 | 0.000830 | 0.000435 | 0.000001 | 0.999411 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H10 | 0.001938 | 0.001151 | 0.000004 | 0.997136 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H20 | 0.003138 | 0.001838 | 0.000010 | 0.993436 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H50 | 0.007888 | 0.004645 | 0.000062 | 0.972918 | 0.000000 |
| monotonic_lstm_penalty | all | 0.016457 | 0.010710 | 0.000271 | 0.837771 | 0.337101 |
| monotonic_lstm_penalty | H1 | 0.013831 | 0.008980 | 0.000191 | 0.836281 | 0.337101 |
| monotonic_lstm_penalty | H10 | 0.014848 | 0.009686 | 0.000220 | 0.831962 | 0.337101 |
| monotonic_lstm_penalty | H20 | 0.015759 | 0.010346 | 0.000248 | 0.834454 | 0.337101 |
| monotonic_lstm_penalty | H50 | 0.019522 | 0.012709 | 0.000381 | 0.834127 | 0.337101 |

| method | best_epoch | best_valid_loss | best_valid_H50_RMSE |
| --- | --- | --- | --- |
| monotonic_lstm_penalty | 70 | 0.158858 | 0.019522 |
| monotonic_lstm_delta_strict | 15 | 0.015173 | 0.009093 |
| monotonic_lstm_delta_with_history_retention | 18 | 0.012272 | 0.007888 |

![loss curve](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/loss_curve.png)

图 6 说明：X 轴是 epoch；Y 轴是训练目标 loss；实线是 train loss，虚线是 valid loss。关键结论：若 valid loss 不下降或快速反弹，说明样本量或输入信息不足以支撑 LSTM 泛化。

![valid H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/valid_h50_scatter.png)

图 7 说明：X 轴是真实 H50 retention；Y 轴是 LSTM 预测 H50 retention；虚线是 `Y=X`。关键结论：对比点云贴合程度判断 LSTM 是否优于基线。

![valid H50 residual distribution](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/valid_h50_residual_distribution.png)

图 8 说明：X 轴是 H50 残差；Y 轴是 block 数量。关键结论：分布越集中在 0 附近，LSTM 误差越小。

![valid H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/valid_h50_residual_vs_true.png)

图 9 说明：X 轴是真实 H50 retention；Y 轴是残差。关键结论：该图用于识别 LSTM 是否在高 retention 或低 retention 区间存在系统偏差。

![valid monotonic curves](C:/Users/pal/projects/batt_soh/outputs/analysis/monotonic_lstm_multistep_retention_blocks_h100_m50/valid_monotonic_curves.png)

图 10 说明：X 轴是 H1:H50；Y 轴是 retention；黑线是真实曲线，彩色线是三种 LSTM 预测。关键结论：delta 两个版本天然单调，penalty 版本是否仍有上升取决于 soft loss 是否足够强。

## 7. 统一对比

| method | input | monotonic_constraint | H50_RMSE | H50_MAE | H50_R2 | all_RMSE | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct LightGBM | 55维工况 summary | 否 | 0.012430 | 0.008257 | 0.932762 | 0.012026 | 0.457354 |
| direct LightGBM + cummin | 55维工况 summary | 是 | 0.012674 | 0.008694 | 0.930087 | 0.012356 | 0.000000 |
| direct LightGBM + isotonic | 55维工况 summary | 是 | 0.012305 | 0.008205 | 0.934104 | 0.011874 | 0.000000 |
| linear_last10 | 历史 retention | 否 | 0.008564 | 0.004065 | 0.968077 | 0.004748 | 0.111554 |
| linear_last10 + cummin | 历史 retention | 是 | 0.008147 | 0.003709 | 0.971110 | 0.004467 | 0.000000 |
| dQdV bridge | 55维工况 -> compact4 -> retention | 否 | 0.018901 | 0.013402 | 0.844523 | 0.017016 | 0.493292 |
| dQdV bridge + cummin | 55维工况 -> compact4 -> retention | 是 | 0.018288 | 0.013403 | 0.854438 | 0.017063 | 0.000000 |
| monotonic LSTM penalty | 100x55 | 软约束 | 0.019522 | 0.012709 | 0.834127 | 0.016457 | 0.337101 |
| monotonic LSTM delta strict | 100x55 + last_history_retention作为递推起点 | 硬约束 | 0.009093 | 0.005257 | 0.964018 | 0.005087 | 0.000000 |
| monotonic LSTM delta with history retention | 100x56 | 硬约束 | 0.007888 | 0.004645 | 0.972918 | 0.004574 | 0.000000 |

## 8. 问题回答

1. 真实 retention 在 H1:H50 上是否严格单调？不是。真实曲线违反率为 `0.892430`，说明观测中存在短期上升或噪声。
2. 原始 LightGBM / linear_last10 / dQdV bridge 是否存在明显单调违反？LightGBM 和 dQdV bridge 通常更明显；linear_last10 由于是线性外推，违反程度通常较低。
3. 单调后处理是否提升 H50 精度？看 Stage 1 的 RMSE 变化：direct `0.000245`、linear `-0.000417`、bridge `-0.000613`。
4. LSTM 单调模型是否优于 LightGBM？当前最优 LSTM 为：monotonic LSTM delta with history retention，H50 RMSE=0.007888。需要与 direct LightGBM 和 linear_last10 的 H50 RMSE 同表比较。
5. 如果 LSTM 没有提升，主要原因优先判断为：linear_last10 已经抓住短期 retention 平滑趋势，其次才是样本量不足；若 Stage 1 后处理也无收益，则单调约束不是主要突破口。
6. 路线建议：本次结果中最优 LSTM 已超过 linear_last10，可继续把单调 LSTM 作为主路线候选，但仍应追加不同随机种子和更大 forecast gap 验证稳定性。
