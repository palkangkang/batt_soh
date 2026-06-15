# 159维工况特征 -> dQdV 相关性与可预测性分析报告

## 结论摘要

- 推荐后续优先 target pack：`compact2`。compact2 在比较分数中最高；median valid R2=0.9112，Top稳定性均值=0.6420。
- 推荐后续 feature pack：使用 `recommended_feature_pack_union.csv` 的去冗余 union 特征包，当前包含 55 个特征。
- 输入侧已排除 `cycles`、`cycle_index_norm`、policy 标签与 policy 三元参数。

## 数据检查

| check_item | value | pass | details |
| --- | --- | --- | --- |
| input_feature_pack | charge_crossbin_discharge_capacity_stats | 1 | must use charge_crossbin_discharge_capacity_stats |
| input_feature_dim | 159 | 1 | expected 60 charge cumulative + 60 charge increment + 16 discharge inc + 16 discharge cum + 7 summary stats |
| excluded_input_columns_present |  | 1 | cycles, cycle_index_norm, policy labels and policy numeric parameters must not be input features |
| target_pack | compact_peak_shape_height_no_width | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| merged_cycle_rows | 138806 | 1 |  |
| train_cycle_rows | 98686 | 1 |  |
| valid_cycle_rows | 40120 | 1 |  |
| policy_cell_count | 183 | 1 |  |
| charge_cross_bin_feature_dim | 60 | 1 |  |
| discharge_range_count | 16 | 1 |  |


## main_peak_area

### 最相关且最稳定的工况特征

| feature | stability_score | global_spearman | train_spearman | valid_spearman | within_cell_spearman_median | diff_spearman | mutual_information_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discharge_inc_delta_ah_3p15_to_3p10 | 0.7509 | 0.7777 | 0.7803 | 0.7688 | 0.9218 | 0.1285 | 0.8896 |
| discharge_cum_delta_ah_2p85_to_2p80 | 0.6732 | -0.6236 | -0.6213 | -0.6408 | -0.9895 | -0.0036 | 0.8249 |
| discharge_cum_delta_ah_3p00_to_2p95 | 0.6651 | -0.6189 | -0.6309 | -0.5944 | -0.9884 | -0.0089 | 0.8245 |
| discharge_cum_delta_ah_3p05_to_3p00 | 0.6577 | -0.6262 | -0.6290 | -0.6363 | -0.9896 | -0.0204 | 0.6108 |
| discharge_cum_delta_ah_3p25_to_3p20 | 0.6508 | -0.5666 | -0.5790 | -0.5590 | -0.9813 | -0.0080 | 0.8707 |
| discharge_cum_delta_ah_3p10_to_3p05 | 0.6383 | -0.5933 | -0.5932 | -0.6126 | -0.9897 | 0.0386 | 0.5542 |
| discharge_cum_delta_ah_3p15_to_3p10 | 0.6294 | -0.5571 | -0.5552 | -0.5850 | -0.9897 | 0.0987 | 0.5551 |
| discharge_cum_total_delta_ah | 0.6205 | -0.5502 | -0.5504 | -0.5719 | -0.9897 | 0.0520 | 0.5491 |
| discharge_cum_total_duration_s | 0.6177 | -0.5390 | -0.5457 | -0.5489 | -0.9897 | 0.0518 | 0.5378 |
| discharge_cum_delta_ah_2p95_to_2p90 | 0.6125 | -0.4968 | -0.4859 | -0.5262 | -0.9732 | -0.0029 | 0.8423 |
| discharge_cum_delta_ah_3p40_to_3p35 | 0.5916 | -0.4451 | -0.4465 | -0.4663 | -0.9880 | 0.0037 | 0.7887 |
| discharge_cum_delta_ah_3p35_to_3p30 | 0.5872 | -0.4496 | -0.4492 | -0.4649 | -0.9886 | 0.0078 | 0.7176 |

### 疑似老化进度伪相关

无。


## main_peak_height_dqdv

### 最相关且最稳定的工况特征

| feature | stability_score | global_spearman | train_spearman | valid_spearman | within_cell_spearman_median | diff_spearman | mutual_information_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discharge_inc_delta_ah_3p15_to_3p10 | 0.7623 | 0.7959 | 0.8009 | 0.7786 | 0.9176 | 0.1041 | 1.0000 |
| discharge_cum_delta_ah_3p05_to_3p00 | 0.6439 | -0.6186 | -0.6213 | -0.6290 | -0.9832 | -0.0098 | 0.5316 |
| discharge_cum_delta_ah_2p85_to_2p80 | 0.6438 | -0.6135 | -0.6125 | -0.6271 | -0.9831 | -0.0023 | 0.5794 |
| discharge_cum_delta_ah_3p00_to_2p95 | 0.6379 | -0.6166 | -0.6272 | -0.5957 | -0.9816 | -0.0036 | 0.5677 |
| discharge_cum_delta_ah_3p10_to_3p05 | 0.6270 | -0.5815 | -0.5825 | -0.5981 | -0.9832 | 0.0328 | 0.5062 |
| discharge_cum_delta_ah_3p15_to_3p10 | 0.6124 | -0.5444 | -0.5438 | -0.5687 | -0.9832 | 0.0612 | 0.4964 |
| discharge_cum_delta_ah_3p25_to_3p20 | 0.6111 | -0.5521 | -0.5687 | -0.5333 | -0.9749 | -0.0018 | 0.5960 |
| discharge_cum_total_delta_ah | 0.6061 | -0.5386 | -0.5404 | -0.5560 | -0.9832 | 0.0266 | 0.4929 |
| discharge_cum_total_duration_s | 0.5971 | -0.5244 | -0.5365 | -0.5197 | -0.9832 | 0.0264 | 0.4805 |
| discharge_cum_delta_ah_2p95_to_2p90 | 0.5819 | -0.4946 | -0.4847 | -0.5217 | -0.9668 | -0.0011 | 0.5532 |
| discharge_cum_delta_ah_3p40_to_3p35 | 0.5603 | -0.4509 | -0.4546 | -0.4668 | -0.9819 | 0.0008 | 0.4515 |
| discharge_cum_delta_ah_3p35_to_3p30 | 0.5549 | -0.4436 | -0.4448 | -0.4546 | -0.9825 | 0.0053 | 0.4245 |

### 疑似老化进度伪相关

无。


## main_peak_voltage_v

### 最相关且最稳定的工况特征

| feature | stability_score | global_spearman | train_spearman | valid_spearman | within_cell_spearman_median | diff_spearman | mutual_information_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discharge_inc_delta_ah_3p15_to_3p10 | 0.5858 | 0.5107 | 0.5180 | 0.4927 | 0.7811 | 0.0343 | 0.8173 |
| charge_cross_bin_cum_41_h | 0.5751 | -0.5893 | -0.6089 | -0.5474 | -0.8348 | 0.0008 | 0.3895 |
| discharge_cum_delta_ah_3p25_to_3p20 | 0.5183 | -0.3060 | -0.3059 | -0.3278 | -0.8665 | -0.0033 | 0.8673 |
| charge_cross_bin_cum_38_h | 0.5173 | -0.4709 | -0.4890 | -0.4306 | -0.8711 | 0.0043 | 0.2482 |
| charge_cross_bin_cum_37_h | 0.5082 | -0.4983 | -0.5281 | -0.4024 | -0.8273 | 0.0024 | 0.2987 |
| discharge_cum_delta_ah_3p00_to_2p95 | 0.5073 | -0.3571 | -0.3437 | -0.3988 | -0.8629 | -0.0044 | 0.6066 |
| charge_cross_bin_cum_16_h | 0.5011 | -0.4725 | -0.4875 | -0.4254 | -0.7829 | 0.0057 | 0.2370 |
| charge_cross_bin_cum_07_h | 0.4978 | -0.4923 | -0.5308 | -0.3776 | -0.8370 | -0.0040 | 0.2813 |
| charge_cross_bin_cum_36_h | 0.4936 | -0.4666 | -0.4844 | -0.4093 | -0.7796 | 0.0036 | 0.2317 |
| charge_cross_bin_cum_06_h | 0.4936 | -0.4822 | -0.5126 | -0.3906 | -0.7864 | 0.0080 | 0.2784 |
| charge_cross_bin_cum_02_h | 0.4916 | -0.4799 | -0.5232 | -0.3551 | -0.8686 | 0.0013 | 0.2642 |
| charge_cross_bin_cum_27_h | 0.4869 | -0.4567 | -0.4866 | -0.3846 | -0.7805 | 0.0026 | 0.2758 |

### 疑似老化进度伪相关

无。


## main_peak_skewness

### 最相关且最稳定的工况特征

| feature | stability_score | global_spearman | train_spearman | valid_spearman | within_cell_spearman_median | diff_spearman | mutual_information_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| discharge_inc_delta_ah_3p15_to_3p10 | 0.6580 | -0.6515 | -0.6562 | -0.6390 | -0.8382 | -0.0759 | 0.7367 |
| discharge_cum_delta_ah_3p00_to_2p95 | 0.5399 | 0.3817 | 0.3836 | 0.3841 | 0.8997 | 0.0049 | 0.6694 |
| charge_cross_bin_cum_38_h | 0.5393 | 0.4847 | 0.4879 | 0.4940 | 0.8871 | 0.0012 | 0.2382 |
| discharge_cum_delta_ah_3p25_to_3p20 | 0.5210 | 0.3220 | 0.3373 | 0.3073 | 0.8981 | -0.0038 | 0.8131 |
| charge_cross_bin_cum_37_h | 0.5122 | 0.4558 | 0.4661 | 0.4269 | 0.8451 | -0.0031 | 0.2626 |
| discharge_cum_delta_ah_2p85_to_2p80 | 0.5031 | 0.3284 | 0.3118 | 0.3894 | 0.8980 | 0.0026 | 0.6594 |
| charge_cross_bin_cum_41_h | 0.4988 | 0.4482 | 0.4837 | 0.3747 | 0.8795 | 0.0030 | 0.2753 |
| discharge_cum_delta_ah_2p95_to_2p90 | 0.4984 | 0.2953 | 0.2737 | 0.3532 | 0.8511 | 0.0017 | 0.8479 |
| charge_cross_bin_cum_39_h | 0.4924 | 0.4332 | 0.4055 | 0.5389 | 0.8665 | 0.0062 | 0.2171 |
| charge_cross_bin_cum_17_h | 0.4867 | 0.3996 | 0.4079 | 0.3769 | 0.8592 | 0.0047 | 0.2087 |
| discharge_cum_delta_ah_3p05_to_3p00 | 0.4801 | 0.3383 | 0.3288 | 0.3867 | 0.8997 | 0.0031 | 0.3477 |
| charge_cross_bin_cum_16_h | 0.4627 | 0.4002 | 0.4106 | 0.3615 | 0.7596 | -0.0021 | 0.1819 |

### 疑似老化进度伪相关

无。


## Top特征包 vs 全量特征包

下表展示每个 target 和 feature pack 的最佳 valid 指标，模型在 `ridge`、`elasticnet`、`random_forest`、`hist_gradient_boosting` 中择优。

| target | feature_pack | model_name | n_features | mse | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_peak_area | full_feature_pack | hist_gradient_boosting | 159 | 0.0008 | 0.0276 | 0.0200 | 0.9383 |
| main_peak_area | top40_stable | hist_gradient_boosting | 40 | 0.0010 | 0.0320 | 0.0236 | 0.9173 |
| main_peak_area | redundancy_pruned_top | hist_gradient_boosting | 40 | 0.0011 | 0.0325 | 0.0241 | 0.9146 |
| main_peak_area | top20_stable | hist_gradient_boosting | 20 | 0.0012 | 0.0343 | 0.0250 | 0.9046 |
| main_peak_height_dqdv | full_feature_pack | hist_gradient_boosting | 159 | 0.0096 | 0.0982 | 0.0707 | 0.9279 |
| main_peak_height_dqdv | redundancy_pruned_top | hist_gradient_boosting | 40 | 0.0123 | 0.1110 | 0.0815 | 0.9079 |
| main_peak_height_dqdv | top40_stable | hist_gradient_boosting | 40 | 0.0126 | 0.1123 | 0.0827 | 0.9057 |
| main_peak_height_dqdv | top20_stable | hist_gradient_boosting | 20 | 0.0130 | 0.1139 | 0.0837 | 0.9030 |
| main_peak_skewness | full_feature_pack | hist_gradient_boosting | 159 | 0.0002 | 0.0152 | 0.0080 | 0.6139 |
| main_peak_skewness | redundancy_pruned_top | hist_gradient_boosting | 40 | 0.0002 | 0.0152 | 0.0082 | 0.6113 |
| main_peak_skewness | top40_stable | hist_gradient_boosting | 40 | 0.0002 | 0.0153 | 0.0082 | 0.6062 |
| main_peak_skewness | top20_stable | hist_gradient_boosting | 20 | 0.0002 | 0.0153 | 0.0083 | 0.6054 |
| main_peak_voltage_v | full_feature_pack | hist_gradient_boosting | 159 | 0.0001 | 0.0089 | 0.0056 | 0.7610 |
| main_peak_voltage_v | redundancy_pruned_top | hist_gradient_boosting | 40 | 0.0001 | 0.0091 | 0.0058 | 0.7492 |
| main_peak_voltage_v | top40_stable | hist_gradient_boosting | 40 | 0.0001 | 0.0097 | 0.0064 | 0.7166 |
| main_peak_voltage_v | top20_stable | hist_gradient_boosting | 20 | 0.0001 | 0.0103 | 0.0070 | 0.6839 |


## compact2/compact3/compact4 判断

- 当前推荐：`compact2`。
- 若目标是最小可解释闭环，`compact2` 成本最低，只覆盖面积与高度。
- 若希望把主峰电压漂移纳入桥接，`compact3` 是复杂度和信息量的折中。
- 若 `main_peak_skewness` 的 valid R2 与稳定性没有明显拖累，`compact4` 更适合保留峰形非对称信息；反之应先用 compact3。