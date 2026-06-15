# 单调物理约束 + LSTM 多步 retention 预测验证报告

## 1. 任务摘要

- split_name: `long_life_holdout`
- train_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv`
- valid_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv`
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
| split_name | long_life_holdout | long_life_holdout | 1 | declared train/valid split |
| train_split_path | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv | 1 |  |
| valid_split_path | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv | 1 |  |
| sample_mode | non_overlapping_blocks | non_overlapping_blocks | 1 |  |
| history_len | 100 | >0 | 1 | positive history length |
| horizon | 50 | >0 | 1 | positive forecast horizon |
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
| true_retention | 3697 | 0.241824 | 0.062391 | 0.000351 | 1.297946 | 0.875000 |
| direct_retention | 7127 | 0.466183 | 0.016615 | 0.001504 | 10.717915 | 1.000000 |
| deployable_bridge | 7621 | 0.498496 | 0.053855 | 0.005478 | 41.744636 | 1.000000 |
| linear_last10 | 1176 | 0.076923 | 0.000154 | 0.000039 | 0.046399 | 0.076923 |
| persistence | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| oracle_bridge | 7508 | 0.491104 | 0.052058 | 0.005263 | 39.512989 | 1.000000 |

真实 retention 的曲线违反率为 `0.875000`。这表示观测标签本身不严格单调，单调约束在本任务中更像物理去噪假设，而不是逐点标签真值的硬事实。

![valid monotonic curves before after](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/valid_monotonic_curves_before_after.png)

图 1 说明：X 轴是未来 horizon step，即 H1 到 H50；Y 轴是 retention；黑线是真实 retention，其余曲线是原始预测和单调后处理预测。关键结论：若后处理曲线更贴近黑线且不再上升，说明单调约束有效；若偏离更大，说明真实短期波动不可忽略。

## 5. Stage 1 单调后处理指标

| method | horizon | rmse | mae | mse | r2 | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| deployable_bridge | all | 0.019789 | 0.013653 | 0.000392 | 0.689862 | 0.498496 |
| deployable_bridge | H1 | 0.018977 | 0.012698 | 0.000360 | 0.650260 | 0.498496 |
| deployable_bridge | H10 | 0.020428 | 0.014033 | 0.000417 | 0.617812 | 0.498496 |
| deployable_bridge | H20 | 0.019708 | 0.013587 | 0.000388 | 0.673891 | 0.498496 |
| deployable_bridge | H50 | 0.019420 | 0.013809 | 0.000377 | 0.755876 | 0.498496 |
| deployable_bridge_bounded_monotonic | all | 0.006097 | 0.003304 | 0.000037 | 0.970557 | 0.000000 |
| deployable_bridge_bounded_monotonic | H1 | 0.002549 | 0.000909 | 0.000006 | 0.993687 | 0.000000 |
| deployable_bridge_bounded_monotonic | H10 | 0.003249 | 0.001816 | 0.000011 | 0.990330 | 0.000000 |
| deployable_bridge_bounded_monotonic | H20 | 0.004708 | 0.002788 | 0.000022 | 0.981393 | 0.000000 |
| deployable_bridge_bounded_monotonic | H50 | 0.009278 | 0.005798 | 0.000086 | 0.944276 | 0.000000 |
| deployable_bridge_cummin | all | 0.015067 | 0.010793 | 0.000227 | 0.820206 | 0.000000 |
| deployable_bridge_cummin | H1 | 0.018977 | 0.012698 | 0.000360 | 0.650260 | 0.000000 |
| deployable_bridge_cummin | H10 | 0.014976 | 0.010704 | 0.000224 | 0.794599 | 0.000000 |
| deployable_bridge_cummin | H20 | 0.014723 | 0.010625 | 0.000217 | 0.817995 | 0.000000 |
| deployable_bridge_cummin | H50 | 0.014675 | 0.010600 | 0.000215 | 0.860613 | 0.000000 |
| deployable_bridge_isotonic | all | 0.019055 | 0.013064 | 0.000363 | 0.712417 | 0.000000 |
| deployable_bridge_isotonic | H1 | 0.020919 | 0.014461 | 0.000438 | 0.575017 | 0.000000 |
| deployable_bridge_isotonic | H10 | 0.019455 | 0.013366 | 0.000379 | 0.653355 | 0.000000 |
| deployable_bridge_isotonic | H20 | 0.019061 | 0.013028 | 0.000363 | 0.694969 | 0.000000 |
| deployable_bridge_isotonic | H50 | 0.017036 | 0.011744 | 0.000290 | 0.812133 | 0.000000 |
| direct_retention | all | 0.013409 | 0.009400 | 0.000180 | 0.857594 | 0.466183 |
| direct_retention | H1 | 0.013197 | 0.008563 | 0.000174 | 0.830868 | 0.466183 |
| direct_retention | H10 | 0.012701 | 0.008680 | 0.000161 | 0.852263 | 0.466183 |
| direct_retention | H20 | 0.013370 | 0.009242 | 0.000179 | 0.849919 | 0.466183 |
| direct_retention | H50 | 0.014049 | 0.010485 | 0.000197 | 0.872243 | 0.466183 |
| direct_retention_bounded_monotonic | all | 0.009860 | 0.006947 | 0.000097 | 0.923009 | 0.000000 |
| direct_retention_bounded_monotonic | H1 | 0.006722 | 0.003983 | 0.000045 | 0.956122 | 0.000000 |
| direct_retention_bounded_monotonic | H10 | 0.007616 | 0.005265 | 0.000058 | 0.946874 | 0.000000 |
| direct_retention_bounded_monotonic | H20 | 0.009032 | 0.006494 | 0.000082 | 0.931506 | 0.000000 |
| direct_retention_bounded_monotonic | H50 | 0.013028 | 0.009850 | 0.000170 | 0.890135 | 0.000000 |
| direct_retention_cummin | all | 0.013756 | 0.010093 | 0.000189 | 0.850138 | 0.000000 |
| direct_retention_cummin | H1 | 0.013197 | 0.008563 | 0.000174 | 0.830868 | 0.000000 |
| direct_retention_cummin | H10 | 0.012995 | 0.009304 | 0.000169 | 0.845337 | 0.000000 |
| direct_retention_cummin | H20 | 0.013458 | 0.009849 | 0.000181 | 0.847936 | 0.000000 |
| direct_retention_cummin | H50 | 0.014908 | 0.011570 | 0.000222 | 0.856136 | 0.000000 |
| direct_retention_isotonic | all | 0.013334 | 0.009339 | 0.000178 | 0.859194 | 0.000000 |
| direct_retention_isotonic | H1 | 0.013279 | 0.008446 | 0.000176 | 0.828756 | 0.000000 |
| direct_retention_isotonic | H10 | 0.012698 | 0.008625 | 0.000161 | 0.852326 | 0.000000 |
| direct_retention_isotonic | H20 | 0.013065 | 0.009061 | 0.000171 | 0.856684 | 0.000000 |
| direct_retention_isotonic | H50 | 0.014305 | 0.010843 | 0.000205 | 0.867542 | 0.000000 |
| linear_last10 | all | 0.003443 | 0.001156 | 0.000012 | 0.990611 | 0.076923 |
| linear_last10 | H1 | 0.001632 | 0.000353 | 0.000003 | 0.997414 | 0.076923 |
| linear_last10 | H10 | 0.001643 | 0.000509 | 0.000003 | 0.997529 | 0.076923 |
| linear_last10 | H20 | 0.002728 | 0.000943 | 0.000007 | 0.993754 | 0.076923 |
| linear_last10 | H50 | 0.005370 | 0.002197 | 0.000029 | 0.981336 | 0.076923 |
| linear_last10_bounded_monotonic | all | 0.003373 | 0.001089 | 0.000011 | 0.990989 | 0.000000 |
| linear_last10_bounded_monotonic | H1 | 0.001639 | 0.000368 | 0.000003 | 0.997391 | 0.000000 |
| linear_last10_bounded_monotonic | H10 | 0.001625 | 0.000486 | 0.000003 | 0.997583 | 0.000000 |
| linear_last10_bounded_monotonic | H20 | 0.002678 | 0.000889 | 0.000007 | 0.993980 | 0.000000 |
| linear_last10_bounded_monotonic | H50 | 0.005234 | 0.002059 | 0.000027 | 0.982268 | 0.000000 |
| linear_last10_cummin | all | 0.003373 | 0.001087 | 0.000011 | 0.990990 | 0.000000 |
| linear_last10_cummin | H1 | 0.001632 | 0.000353 | 0.000003 | 0.997414 | 0.000000 |
| linear_last10_cummin | H10 | 0.001619 | 0.000485 | 0.000003 | 0.997599 | 0.000000 |
| linear_last10_cummin | H20 | 0.002676 | 0.000888 | 0.000007 | 0.993989 | 0.000000 |
| linear_last10_cummin | H50 | 0.005234 | 0.002057 | 0.000027 | 0.982267 | 0.000000 |
| linear_last10_isotonic | all | 0.003429 | 0.001156 | 0.000012 | 0.990689 | 0.000000 |
| linear_last10_isotonic | H1 | 0.001694 | 0.000418 | 0.000003 | 0.997213 | 0.000000 |
| linear_last10_isotonic | H10 | 0.001710 | 0.000553 | 0.000003 | 0.997323 | 0.000000 |
| linear_last10_isotonic | H20 | 0.002748 | 0.000959 | 0.000008 | 0.993659 | 0.000000 |
| linear_last10_isotonic | H50 | 0.005289 | 0.002127 | 0.000028 | 0.981896 | 0.000000 |

- direct LightGBM + cummin 的 H50 RMSE 变化：`0.000859`，负数代表提升。
- linear_last10 + cummin 的 H50 RMSE 变化：`-0.000136`，负数代表提升。
- dQdV bridge + cummin 的 H50 RMSE 变化：`-0.004746`，负数代表提升。

![postprocess H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/postprocess_h50_scatter.png)

图 2 说明：X 轴是真实 H50 retention；Y 轴是预测 H50 retention；虚线是理想预测 `Y=X`；每个点代表一个 valid block。关键结论：点云越贴近虚线，H50 精度越高。

![postprocess H50 residual distribution](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/postprocess_h50_residual_distribution.png)

图 3 说明：X 轴是 H50 残差 `真实 retention - 预测 retention`；Y 轴是 block 数量；黑色虚线是 0 残差。关键结论：分布越窄且越靠近 0，后处理越有效。

![postprocess H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/postprocess_h50_residual_vs_true.png)

图 4 说明：X 轴是真实 H50 retention；Y 轴是残差。若残差随真实 retention 呈结构性斜率，说明模型在不同衰减阶段有系统偏差。

![postprocess selected curves](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/postprocess_curves_selected_blocks.png)

图 5 说明：X 轴是 H1:H50，Y 轴是 retention；黑线是真实曲线，彩色线是原始和单调后处理曲线。关键结论：该图用于判断单调约束是修正预测抖动，还是过度压低未来预测。

## 6. Stage 2/3 LSTM 指标

| method | horizon | rmse | mae | mse | r2 | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| monotonic_lstm_delta_last_retention_only | all | 0.005072 | 0.003698 | 0.000026 | 0.979624 | 0.000000 |
| monotonic_lstm_delta_last_retention_only | H1 | 0.001161 | 0.000341 | 0.000001 | 0.998691 | 0.000000 |
| monotonic_lstm_delta_last_retention_only | H10 | 0.001854 | 0.001616 | 0.000003 | 0.996851 | 0.000000 |
| monotonic_lstm_delta_last_retention_only | H20 | 0.003273 | 0.002747 | 0.000011 | 0.991006 | 0.000000 |
| monotonic_lstm_delta_last_retention_only | H50 | 0.008406 | 0.006857 | 0.000071 | 0.954261 | 0.000000 |

| method | best_epoch | best_valid_loss | best_valid_H50_RMSE |
| --- | --- | --- | --- |
| monotonic_lstm_delta_last_retention_only | 1 | 0.013196 | 0.008406 |

![loss curve](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/loss_curve.png)

图 6 说明：X 轴是 epoch；Y 轴是训练目标 loss；实线是 train loss，虚线是 valid loss。关键结论：若 valid loss 不下降或快速反弹，说明样本量或输入信息不足以支撑 LSTM 泛化。

![valid H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/valid_h50_scatter.png)

图 7 说明：X 轴是真实 H50 retention；Y 轴是 LSTM 预测 H50 retention；虚线是 `Y=X`。关键结论：对比点云贴合程度判断 LSTM 是否优于基线。

![valid H50 residual distribution](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/valid_h50_residual_distribution.png)

图 8 说明：X 轴是 H50 残差；Y 轴是 block 数量。关键结论：分布越集中在 0 附近，LSTM 误差越小。

![valid H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/valid_h50_residual_vs_true.png)

图 9 说明：X 轴是真实 H50 retention；Y 轴是残差。关键结论：该图用于识别 LSTM 是否在高 retention 或低 retention 区间存在系统偏差。

![valid monotonic curves](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50/valid_monotonic_curves.png)

图 10 说明：X 轴是 H1:H50；Y 轴是 retention；黑线是真实曲线，彩色线是三种 LSTM 预测。关键结论：delta 两个版本天然单调，penalty 版本是否仍有上升取决于 soft loss 是否足够强。

## 7. 统一对比

| method | input | monotonic_constraint | H50_RMSE | H50_MAE | H50_R2 | all_RMSE | monotonic_violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct LightGBM | 55维工况 summary | 否 | 0.014049 | 0.010485 | 0.872243 | 0.013409 | 0.466183 |
| direct LightGBM + cummin | 55维工况 summary | 是 | 0.014908 | 0.011570 | 0.856136 | 0.013756 | 0.000000 |
| direct LightGBM + isotonic | 55维工况 summary | 是 | 0.014305 | 0.010843 | 0.867542 | 0.013334 | 0.000000 |
| linear_last10 | 历史 retention | 否 | 0.005370 | 0.002197 | 0.981336 | 0.003443 | 0.076923 |
| linear_last10 + cummin | 历史 retention | 是 | 0.005234 | 0.002057 | 0.982267 | 0.003373 | 0.000000 |
| dQdV bridge | 55维工况 -> compact4 -> retention | 否 | 0.019420 | 0.013809 | 0.755876 | 0.019789 | 0.498496 |
| dQdV bridge + cummin | 55维工况 -> compact4 -> retention | 是 | 0.014675 | 0.010600 | 0.860613 | 0.015067 | 0.000000 |
| monotonic LSTM delta last retention only | 1x1 last_history_retention | 硬约束 | 0.008406 | 0.006857 | 0.954261 | 0.005072 | 0.000000 |

## 8. 问题回答

1. 真实 retention 在 H1:H50 上是否严格单调？不是。真实曲线违反率为 `0.875000`，说明观测中存在短期上升或噪声。
2. 原始 LightGBM / linear_last10 / dQdV bridge 是否存在明显单调违反？LightGBM 和 dQdV bridge 通常更明显；linear_last10 由于是线性外推，违反程度通常较低。
3. 单调后处理是否提升 H50 精度？看 Stage 1 的 RMSE 变化：direct `0.000859`、linear `-0.000136`、bridge `-0.004746`。
4. LSTM 单调模型是否优于 LightGBM？当前最优 LSTM 为：monotonic LSTM delta last retention only，H50 RMSE=0.008406。需要与 direct LightGBM 和 linear_last10 的 H50 RMSE 同表比较。
5. 如果 LSTM 没有提升，主要原因优先判断为：linear_last10 已经抓住短期 retention 平滑趋势，其次才是样本量不足；若 Stage 1 后处理也无收益，则单调约束不是主要突破口。
6. 路线建议：若正式 H100/M50 结果仍显示 LSTM 未超过 linear_last10，则不建议把 LSTM 作为主预测模型，更建议回到 linear_last10 或 LightGBM + 单调后处理，并把 LSTM 作为残差修正候选。
