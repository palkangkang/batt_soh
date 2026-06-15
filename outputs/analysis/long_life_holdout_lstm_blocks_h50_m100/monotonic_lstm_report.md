# 单调物理约束 + LSTM 多步 retention 预测验证报告

## 1. 任务摘要

- split_name: `long_life_holdout`
- train_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv`
- valid_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv`
- history_len: `50`
- horizon: `100`
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
| split_name | long_life_holdout | long_life_holdout | 1 | declared train/valid split |
| train_split_path | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv | 1 |  |
| valid_split_path | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv | 1 |  |
| sample_mode | non_overlapping_blocks | non_overlapping_blocks | 1 |  |
| history_len | 50 | >0 | 1 | positive history length |
| horizon | 100 | >0 | 1 | positive forecast horizon |
| block_stride | 150 | 150 | 1 | expected history_len+horizon |
| feature_count | 55 | 55 | 1 | recommended55 |
| target_pack | compact4 | compact4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| target_dim | 4 | 4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| forbidden_input_columns_present | 0 | 0 | 1 |  |
| train_block_count | 519 | >0 | 1 |  |
| valid_block_count | 312 | >0 | 1 |  |

## 4. Stage 0 单调性诊断

| series | monotonic_violation_count | monotonic_violation_rate | max_positive_jump | mean_positive_jump | total_positive_jump | curve_has_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| true_retention | 7748 | 0.250842 | 0.125197 | 0.000382 | 2.962247 | 0.900641 |
| direct_retention | 14624 | 0.473452 | 0.021532 | 0.001873 | 27.394229 | 1.000000 |
| deployable_bridge | 15153 | 0.490579 | 0.061294 | 0.006236 | 94.494328 | 1.000000 |
| linear_last10 | 3949 | 0.127849 | 0.002189 | 0.000102 | 0.403864 | 0.128205 |
| persistence | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_bridge | 15242 | 0.493460 | 0.092603 | 0.005137 | 78.295122 | 1.000000 |

真实 retention 的曲线违反率为 `0.900641`。这表示观测标签本身不严格单调，单调约束在本任务中更像物理去噪假设，而不是逐点标签真值的硬事实。

![valid monotonic curves before after](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/valid_monotonic_curves_before_after.png)

图 1 说明：X 轴是未来 horizon step，即 H1 到 H50；Y 轴是 retention；黑线是真实 retention，其余曲线是原始预测和单调后处理预测。关键结论：若后处理曲线更贴近黑线且不再上升，说明单调约束有效；若偏离更大，说明真实短期波动不可忽略。

## 5. Stage 1 单调后处理指标

| method | horizon | rmse | mae | mse | r2 | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| deployable_bridge | all | 0.018591 | 0.012463 | 0.000346 | 0.671990 | 0.490579 |
| deployable_bridge | H1 | 0.019730 | 0.012725 | 0.000389 | 0.430347 | 0.490579 |
| deployable_bridge | H10 | 0.019741 | 0.013307 | 0.000390 | 0.464928 | 0.490579 |
| deployable_bridge | H20 | 0.017293 | 0.011677 | 0.000299 | 0.618393 | 0.490579 |
| deployable_bridge | H50 | 0.021112 | 0.013843 | 0.000446 | 0.557233 | 0.490579 |
| deployable_bridge | H100 | 0.019241 | 0.012629 | 0.000370 | 0.760365 | 0.490579 |
| deployable_bridge_bounded_monotonic | all | 0.009284 | 0.004934 | 0.000086 | 0.918197 | 0.000000 |
| deployable_bridge_bounded_monotonic | H1 | 0.002340 | 0.000829 | 0.000005 | 0.991985 | 0.000000 |
| deployable_bridge_bounded_monotonic | H10 | 0.003623 | 0.001917 | 0.000013 | 0.981972 | 0.000000 |
| deployable_bridge_bounded_monotonic | H20 | 0.004510 | 0.002645 | 0.000020 | 0.974042 | 0.000000 |
| deployable_bridge_bounded_monotonic | H50 | 0.008460 | 0.004927 | 0.000072 | 0.928903 | 0.000000 |
| deployable_bridge_bounded_monotonic | H100 | 0.016368 | 0.010233 | 0.000268 | 0.826576 | 0.000000 |
| deployable_bridge_cummin | all | 0.015245 | 0.011287 | 0.000232 | 0.779452 | 0.000000 |
| deployable_bridge_cummin | H1 | 0.019730 | 0.012725 | 0.000389 | 0.430347 | 0.000000 |
| deployable_bridge_cummin | H10 | 0.014032 | 0.009874 | 0.000197 | 0.729645 | 0.000000 |
| deployable_bridge_cummin | H20 | 0.012928 | 0.009685 | 0.000167 | 0.786737 | 0.000000 |
| deployable_bridge_cummin | H50 | 0.015301 | 0.011457 | 0.000234 | 0.767427 | 0.000000 |
| deployable_bridge_cummin | H100 | 0.019256 | 0.014435 | 0.000371 | 0.759996 | 0.000000 |
| deployable_bridge_isotonic | all | 0.017584 | 0.011624 | 0.000309 | 0.706567 | 0.000000 |
| deployable_bridge_isotonic | H1 | 0.021245 | 0.014108 | 0.000451 | 0.339459 | 0.000000 |
| deployable_bridge_isotonic | H10 | 0.018322 | 0.012240 | 0.000336 | 0.539087 | 0.000000 |
| deployable_bridge_isotonic | H20 | 0.017214 | 0.011652 | 0.000296 | 0.621900 | 0.000000 |
| deployable_bridge_isotonic | H50 | 0.017659 | 0.011687 | 0.000312 | 0.690235 | 0.000000 |
| deployable_bridge_isotonic | H100 | 0.017800 | 0.011633 | 0.000317 | 0.794921 | 0.000000 |
| direct_retention | all | 0.015331 | 0.010792 | 0.000235 | 0.776953 | 0.473452 |
| direct_retention | H1 | 0.011477 | 0.008158 | 0.000132 | 0.807248 | 0.473452 |
| direct_retention | H10 | 0.011923 | 0.008375 | 0.000142 | 0.804818 | 0.473452 |
| direct_retention | H20 | 0.012503 | 0.008910 | 0.000156 | 0.800530 | 0.473452 |
| direct_retention | H50 | 0.014085 | 0.010116 | 0.000198 | 0.802915 | 0.473452 |
| direct_retention | H100 | 0.020482 | 0.014922 | 0.000420 | 0.728449 | 0.473452 |
| direct_retention_bounded_monotonic | all | 0.013932 | 0.009597 | 0.000194 | 0.815805 | 0.000000 |
| direct_retention_bounded_monotonic | H1 | 0.007683 | 0.004663 | 0.000059 | 0.913625 | 0.000000 |
| direct_retention_bounded_monotonic | H10 | 0.008504 | 0.005673 | 0.000072 | 0.900693 | 0.000000 |
| direct_retention_bounded_monotonic | H20 | 0.009400 | 0.006564 | 0.000088 | 0.887251 | 0.000000 |
| direct_retention_bounded_monotonic | H50 | 0.012470 | 0.009080 | 0.000156 | 0.845526 | 0.000000 |
| direct_retention_bounded_monotonic | H100 | 0.021134 | 0.015778 | 0.000447 | 0.710903 | 0.000000 |
| direct_retention_cummin | all | 0.016509 | 0.012394 | 0.000273 | 0.741338 | 0.000000 |
| direct_retention_cummin | H1 | 0.011477 | 0.008158 | 0.000132 | 0.807248 | 0.000000 |
| direct_retention_cummin | H10 | 0.012403 | 0.009135 | 0.000154 | 0.788760 | 0.000000 |
| direct_retention_cummin | H20 | 0.013069 | 0.009953 | 0.000171 | 0.782043 | 0.000000 |
| direct_retention_cummin | H50 | 0.015588 | 0.012138 | 0.000243 | 0.758612 | 0.000000 |
| direct_retention_cummin | H100 | 0.022964 | 0.017940 | 0.000527 | 0.658647 | 0.000000 |
| direct_retention_isotonic | all | 0.015203 | 0.010716 | 0.000231 | 0.780641 | 0.000000 |
| direct_retention_isotonic | H1 | 0.011346 | 0.007825 | 0.000129 | 0.811622 | 0.000000 |
| direct_retention_isotonic | H10 | 0.011691 | 0.008208 | 0.000137 | 0.812331 | 0.000000 |
| direct_retention_isotonic | H20 | 0.012056 | 0.008544 | 0.000145 | 0.814520 | 0.000000 |
| direct_retention_isotonic | H50 | 0.013987 | 0.010013 | 0.000196 | 0.805658 | 0.000000 |
| direct_retention_isotonic | H100 | 0.021185 | 0.015812 | 0.000449 | 0.709498 | 0.000000 |
| linear_last10 | all | 0.011745 | 0.003214 | 0.000138 | 0.869080 | 0.127849 |
| linear_last10 | H1 | 0.000988 | 0.000283 | 0.000001 | 0.998572 | 0.127849 |
| linear_last10 | H10 | 0.002877 | 0.000787 | 0.000008 | 0.988632 | 0.127849 |
| linear_last10 | H20 | 0.004647 | 0.001307 | 0.000022 | 0.972439 | 0.127849 |
| linear_last10 | H50 | 0.010376 | 0.003035 | 0.000108 | 0.893050 | 0.127849 |
| linear_last10 | H100 | 0.020048 | 0.006783 | 0.000402 | 0.739842 | 0.127849 |
| linear_last10_bounded_monotonic | all | 0.008730 | 0.002548 | 0.000076 | 0.927672 | 0.000000 |
| linear_last10_bounded_monotonic | H1 | 0.001605 | 0.000292 | 0.000003 | 0.996229 | 0.000000 |
| linear_last10_bounded_monotonic | H10 | 0.002449 | 0.000656 | 0.000006 | 0.991763 | 0.000000 |
| linear_last10_bounded_monotonic | H20 | 0.003660 | 0.001032 | 0.000013 | 0.982906 | 0.000000 |
| linear_last10_bounded_monotonic | H50 | 0.007727 | 0.002381 | 0.000060 | 0.940682 | 0.000000 |
| linear_last10_bounded_monotonic | H100 | 0.015047 | 0.005464 | 0.000226 | 0.853440 | 0.000000 |
| linear_last10_cummin | all | 0.008743 | 0.002573 | 0.000076 | 0.927462 | 0.000000 |
| linear_last10_cummin | H1 | 0.000988 | 0.000283 | 0.000001 | 0.998572 | 0.000000 |
| linear_last10_cummin | H10 | 0.002474 | 0.000679 | 0.000006 | 0.991595 | 0.000000 |
| linear_last10_cummin | H20 | 0.003687 | 0.001067 | 0.000014 | 0.982657 | 0.000000 |
| linear_last10_cummin | H50 | 0.007748 | 0.002406 | 0.000060 | 0.940368 | 0.000000 |
| linear_last10_cummin | H100 | 0.015059 | 0.005491 | 0.000227 | 0.853222 | 0.000000 |
| linear_last10_isotonic | all | 0.011159 | 0.003211 | 0.000125 | 0.881824 | 0.000000 |
| linear_last10_isotonic | H1 | 0.006774 | 0.000896 | 0.000046 | 0.932857 | 0.000000 |
| linear_last10_isotonic | H10 | 0.007075 | 0.001300 | 0.000050 | 0.931263 | 0.000000 |
| linear_last10_isotonic | H20 | 0.007655 | 0.001700 | 0.000059 | 0.925222 | 0.000000 |
| linear_last10_isotonic | H50 | 0.010418 | 0.003041 | 0.000109 | 0.892174 | 0.000000 |
| linear_last10_isotonic | H100 | 0.016588 | 0.006136 | 0.000275 | 0.821901 | 0.000000 |

- direct LightGBM + cummin 的 H100 RMSE 变化：`0.002482`，负数代表提升。
- linear_last10 + cummin 的 H100 RMSE 变化：`-0.004989`，负数代表提升。
- dQdV bridge + cummin 的 H100 RMSE 变化：`0.000015`，负数代表提升。

![postprocess H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/postprocess_h50_scatter.png)

图 2 说明：X 轴是真实 H100 retention；Y 轴是预测 H100 retention；虚线是理想预测 `Y=X`；每个点代表一个 valid block。关键结论：点云越贴近虚线，H100 精度越高。

![postprocess H50 residual distribution](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/postprocess_h50_residual_distribution.png)

图 3 说明：X 轴是 H100 残差 `真实 retention - 预测 retention`；Y 轴是 block 数量；黑色虚线是 0 残差。关键结论：分布越窄且越靠近 0，后处理越有效。

![postprocess H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/postprocess_h50_residual_vs_true.png)

图 4 说明：X 轴是真实 H100 retention；Y 轴是残差。若残差随真实 retention 呈结构性斜率，说明模型在不同衰减阶段有系统偏差。

![postprocess selected curves](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/postprocess_curves_selected_blocks.png)

图 5 说明：X 轴是 H1:H50，Y 轴是 retention；黑线是真实曲线，彩色线是原始和单调后处理曲线。关键结论：该图用于判断单调约束是修正预测抖动，还是过度压低未来预测。

## 6. Stage 2/3 LSTM 指标

| method | horizon | rmse | mae | mse | r2 | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| monotonic_lstm_delta_strict | all | 0.008589 | 0.005010 | 0.000074 | 0.929996 | 0.000000 |
| monotonic_lstm_delta_strict | H1 | 0.001619 | 0.000318 | 0.000003 | 0.996162 | 0.000000 |
| monotonic_lstm_delta_strict | H10 | 0.002550 | 0.001160 | 0.000007 | 0.991069 | 0.000000 |
| monotonic_lstm_delta_strict | H20 | 0.003340 | 0.002025 | 0.000011 | 0.985764 | 0.000000 |
| monotonic_lstm_delta_strict | H50 | 0.007141 | 0.004879 | 0.000051 | 0.949340 | 0.000000 |
| monotonic_lstm_delta_strict | H100 | 0.015180 | 0.010216 | 0.000230 | 0.850852 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | all | 0.007506 | 0.004540 | 0.000056 | 0.946531 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H1 | 0.001618 | 0.000321 | 0.000003 | 0.996170 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H10 | 0.002505 | 0.001095 | 0.000006 | 0.991381 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H20 | 0.003223 | 0.001871 | 0.000010 | 0.986742 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H50 | 0.006279 | 0.004352 | 0.000039 | 0.960839 | 0.000000 |
| monotonic_lstm_delta_with_history_retention | H100 | 0.012995 | 0.009225 | 0.000169 | 0.890696 | 0.000000 |
| monotonic_lstm_penalty | all | 0.023539 | 0.015610 | 0.000554 | 0.474153 | 0.470150 |
| monotonic_lstm_penalty | H1 | 0.019255 | 0.012646 | 0.000371 | 0.457441 | 0.470150 |
| monotonic_lstm_penalty | H10 | 0.018571 | 0.012068 | 0.000345 | 0.526428 | 0.470150 |
| monotonic_lstm_penalty | H20 | 0.019616 | 0.012819 | 0.000385 | 0.508990 | 0.470150 |
| monotonic_lstm_penalty | H50 | 0.023616 | 0.015955 | 0.000558 | 0.445973 | 0.470150 |
| monotonic_lstm_penalty | H100 | 0.031175 | 0.021389 | 0.000972 | 0.370902 | 0.470150 |

| method | best_epoch | best_valid_loss | best_valid_H50_RMSE |
| --- | --- | --- | --- |
| monotonic_lstm_penalty | 39 | 0.385901 | 0.031175 |
| monotonic_lstm_delta_strict | 1 | 0.050899 | 0.015180 |
| monotonic_lstm_delta_with_history_retention | 46 | 0.038878 | 0.012995 |

![loss curve](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/loss_curve.png)

图 6 说明：X 轴是 epoch；Y 轴是训练目标 loss；实线是 train loss，虚线是 valid loss。关键结论：若 valid loss 不下降或快速反弹，说明样本量或输入信息不足以支撑 LSTM 泛化。

![valid H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/valid_h50_scatter.png)

图 7 说明：X 轴是真实 H100 retention；Y 轴是 LSTM 预测 H100 retention；虚线是 `Y=X`。关键结论：对比点云贴合程度判断 LSTM 是否优于基线。

![valid H50 residual distribution](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/valid_h50_residual_distribution.png)

图 8 说明：X 轴是 H100 残差；Y 轴是 block 数量。关键结论：分布越集中在 0 附近，LSTM 误差越小。

![valid H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/valid_h50_residual_vs_true.png)

图 9 说明：X 轴是真实 H100 retention；Y 轴是残差。关键结论：该图用于识别 LSTM 是否在高 retention 或低 retention 区间存在系统偏差。

![valid monotonic curves](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_blocks_h50_m100/valid_monotonic_curves.png)

图 10 说明：X 轴是 H1:H50；Y 轴是 retention；黑线是真实曲线，彩色线是三种 LSTM 预测。关键结论：delta 两个版本天然单调，penalty 版本是否仍有上升取决于 soft loss 是否足够强。

## 7. 统一对比

| method | input | monotonic_constraint | H50_RMSE | H50_MAE | H50_R2 | all_RMSE | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct LightGBM | 55维工况 summary | 否 | 0.020482 | 0.014922 | 0.728449 | 0.015331 | 0.473452 |
| direct LightGBM + cummin | 55维工况 summary | 是 | 0.022964 | 0.017940 | 0.658647 | 0.016509 | 0.000000 |
| direct LightGBM + isotonic | 55维工况 summary | 是 | 0.021185 | 0.015812 | 0.709498 | 0.015203 | 0.000000 |
| linear_last10 | 历史 retention | 否 | 0.020048 | 0.006783 | 0.739842 | 0.011745 | 0.127849 |
| linear_last10 + cummin | 历史 retention | 是 | 0.015059 | 0.005491 | 0.853222 | 0.008743 | 0.000000 |
| dQdV bridge | 55维工况 -> compact4 -> retention | 否 | 0.019241 | 0.012629 | 0.760365 | 0.018591 | 0.490579 |
| dQdV bridge + cummin | 55维工况 -> compact4 -> retention | 是 | 0.019256 | 0.014435 | 0.759996 | 0.015245 | 0.000000 |
| monotonic LSTM penalty | 100x55 | 软约束 | 0.031175 | 0.021389 | 0.370902 | 0.023539 | 0.470150 |
| monotonic LSTM delta strict | 100x55 + last_history_retention作为递推起点 | 硬约束 | 0.015180 | 0.010216 | 0.850852 | 0.008589 | 0.000000 |
| monotonic LSTM delta with history retention | 100x56 | 硬约束 | 0.012995 | 0.009225 | 0.890696 | 0.007506 | 0.000000 |

## 8. 问题回答

1. 真实 retention 在 H1:H50 上是否严格单调？不是。真实曲线违反率为 `0.900641`，说明观测中存在短期上升或噪声。
2. 原始 LightGBM / linear_last10 / dQdV bridge 是否存在明显单调违反？LightGBM 和 dQdV bridge 通常更明显；linear_last10 由于是线性外推，违反程度通常较低。
3. 单调后处理是否提升 H50 精度？看 Stage 1 的 RMSE 变化：direct `0.002482`、linear `-0.004989`、bridge `0.000015`。
4. LSTM 单调模型是否优于 LightGBM？当前最优 LSTM 为：monotonic LSTM delta with history retention，H100 RMSE=0.012995。需要与 direct LightGBM 和 linear_last10 的 H50 RMSE 同表比较。
5. 如果 LSTM 没有提升，主要原因优先判断为：linear_last10 已经抓住短期 retention 平滑趋势，其次才是样本量不足；若 Stage 1 后处理也无收益，则单调约束不是主要突破口。
6. 路线建议：本次结果中最优 LSTM 已超过 linear_last10，可继续把单调 LSTM 作为主路线候选，但仍应追加不同随机种子和更大 forecast gap 验证稳定性。
