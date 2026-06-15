# recommended feature pack + compact2 dQdV retention 短链路验证报告

## 1. 摘要
- 入选 compact2 预测器：`recommended55_hist_gradient_boosting`
- 入选 compact2->retention bridge：`oracle_bridge_candidate_lightgbm`
- recommended55 compact2 最佳 valid 平均 R2：`0.919829`
- oracle bridge valid R2：`0.867750`
- deployable bridge valid R2：`0.854967`
- direct recommended55 retention valid R2：`0.944597`
- full159 direct retention valid R2：`0.946936`

## 2. 数据检查
| check_item | value | pass_flag | details |
| --- | --- | --- | --- |
| recommended_feature_count | 55 | 1 | expected 55 |
| full159_feature_count | 159 | 1 | expected 159 |
| forbidden_input_columns_present | 0 | 1 |  |
| merged_cycle_rows | 140560 | 1 |  |
| train_cycle_rows | 98686 | 1 |  |
| valid_cycle_rows | 41874 | 1 |  |
| train_policy_cell_count | 135 | 1 |  |
| valid_policy_cell_count | 52 | 1 |  |
| split_overlap_zero | 1 | 1 | overlap_count=0 |
| compact2_target_dim | 2 | 1 | main_peak_area,main_peak_height_dqdv |
| retention_range_check | 1 | 1 |  |
| charge_cross_bin_feature_dim | 60 | 1 |  |
| discharge_range_count | 16 | 1 |  |

## 3. compact2 可预测性
| model_name | mean_valid_r2 |
| --- | --- |
| recommended55_hist_gradient_boosting | 0.919829 |
| full159_lightgbm | 0.919673 |
| full159_hist_gradient_boosting | 0.918231 |
| recommended55_lightgbm | 0.916493 |
| recommended55_elasticnet | 0.592513 |
| recommended55_ridge | 0.408901 |
| full159_elasticnet | -1009976.477991 |
| full159_ridge | -1849484.886134 |

## 4. retention 链路指标
| model_name | set_type | source_model | mse | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_bridge | train | lightgbm | 0.000245 | 0.015664 | 0.010940 | 0.897382 |
| oracle_bridge | valid | lightgbm | 0.000282 | 0.016789 | 0.011640 | 0.867750 |
| deployable_bridge | train | lightgbm | 0.000273 | 0.016533 | 0.011409 | 0.885684 |
| deployable_bridge | valid | lightgbm | 0.000309 | 0.017581 | 0.011959 | 0.854967 |
| full159_compact2_bridge | train | lightgbm | 0.000272 | 0.016480 | 0.011387 | 0.886419 |
| full159_compact2_bridge | valid | lightgbm | 0.000312 | 0.017668 | 0.011817 | 0.853536 |
| direct_retention_baseline | train | lightgbm | 0.000016 | 0.003997 | 0.002496 | 0.993319 |
| direct_retention_baseline | valid | lightgbm | 0.000118 | 0.010866 | 0.007132 | 0.944597 |
| full159_direct_retention | train | lightgbm | 0.000014 | 0.003742 | 0.002334 | 0.994143 |
| full159_direct_retention | valid | lightgbm | 0.000113 | 0.010635 | 0.006722 | 0.946936 |

## 5. 必答结论
- 55维工况 -> compact2：最佳模型 `recommended55_hist_gradient_boosting` 的 valid 平均 R2 为 `0.919829`，说明 compact2 在 cycle 级表格特征下已经足够可预测。
- 真实 compact2 -> retention 上限：oracle bridge valid R2 为 `0.867750`。
- 预测 compact2 -> retention 部署链路：deployable bridge valid R2 为 `0.854967`。
- oracle 到 deployable 的 R2 损失：`0.012783`。
- direct retention baseline 对比：direct valid R2 为 `0.944597`，比 deployable bridge 高 `0.089630`。
- full159 compact2 bridge valid R2：`0.853536`。
- full159 direct retention valid R2：`0.946936`。
- 建议：direct retention baseline 明显更强，compact2 暂时更适合作解释层，不建议优先投入 compact2 长窗口训练。

<!-- compact2_extended_comparison_start -->
## 7. dQdV 预测对比：55维 vs 159维

| 输入特征 | dQdV target | 模型 | R2 | RMSE | MAE | MSE |
| --- | --- | --- | --- | --- | --- | --- |
| recommended55 | main_peak_area | recommended55_hist_gradient_boosting | 0.925221 | 0.030867 | 0.022634 | 0.000953 |
| recommended55 | main_peak_height_dqdv | recommended55_hist_gradient_boosting | 0.914437 | 0.109474 | 0.078085 | 0.011985 |
| full159 | main_peak_area | full159_lightgbm | 0.927873 | 0.030315 | 0.021587 | 0.000919 |
| full159 | main_peak_height_dqdv | full159_lightgbm | 0.911472 | 0.111354 | 0.077879 | 0.012400 |

`recommended55_hist_gradient_boosting` 代表 55维推荐特征包的最佳 dQdV 预测模型；`full159_lightgbm` 代表 159维全量工况特征包的 dQdV 预测对照。从 valid 指标看，55维在两个 compact2 目标上已经接近 159维，说明推荐特征包保留了主要的 dQdV 可预测信息。

![compact2_valid_scatter_55_vs_159](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/compact2_valid_scatter_55_vs_159.png)

![compact2_valid_residual_55_vs_159](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/compact2_valid_residual_55_vs_159.png)

## 8. dQdV 中介到 retention：真实 dQdV vs 55维预测 dQdV vs 159维预测 dQdV

| 链路 | 输入到 bridge 的 dQdV | bridge 模型 | R2 | RMSE | MAE | MSE |
| --- | --- | --- | --- | --- | --- | --- |
| oracle_bridge | 真实 compact2 | lightgbm | 0.867750 | 0.016789 | 0.011640 | 0.000282 |
| deployable_bridge | 55维预测 compact2 | lightgbm | 0.854967 | 0.017581 | 0.011959 | 0.000309 |
| full159_compact2_bridge | 159维预测 compact2 | lightgbm | 0.853536 | 0.017668 | 0.011817 | 0.000312 |

`oracle_bridge` 使用真实 compact2，是中介表征对 retention 的上限参考，不是部署路径；`deployable_bridge` 使用 55维工况预测 compact2；`full159_compact2_bridge` 使用 159维工况预测 compact2。当前 oracle 到 deployable 的 R2 损失为 `0.012783`，55维预测 dQdV 与159维预测 dQdV 传递到 retention 后几乎相当。

![retention_bridge_valid_scatter_oracle_55_159](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/retention_bridge_valid_scatter_oracle_55_159.png)

![retention_bridge_valid_residual_oracle_55_159](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/retention_bridge_valid_residual_oracle_55_159.png)

## 9. 直接 retention 预测：55维 vs 159维

| 链路 | 输入特征 | 模型 | R2 | RMSE | MAE | MSE |
| --- | --- | --- | --- | --- | --- | --- |
| direct_retention_baseline | recommended55 | lightgbm | 0.944597 | 0.010866 | 0.007132 | 0.000118 |
| full159_direct_retention | full159 | lightgbm | 0.946936 | 0.010635 | 0.006722 | 0.000113 |

`direct_retention_baseline` 是 55维推荐特征包直接预测 retention；`full159_direct_retention` 是 159维特征直接预测 retention。当前 direct 路径显著强于 dQdV bridge 路径，而 159维 direct 相比55维 direct 的提升很小。

![retention_direct_valid_scatter_55_vs_159](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/retention_direct_valid_scatter_55_vs_159.png)

![retention_direct_valid_residual_55_vs_159](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/retention_direct_valid_residual_55_vs_159.png)

## 10. 汇总结论

| 问题 | 最优/对照结果 | 结论 |
| --- | --- | --- |
| 55维能否预测 compact2 dQdV | recommended55 compact2 valid mean R2 = 0.919829 | 能，已具备较好可预测性 |
| 159维是否明显优于55维预测 compact2 | full159 compact2 valid mean R2 = 0.919673 | 不明显优于55维 |
| 预测 dQdV bridge 是否接近真实 dQdV oracle | 0.854967 vs 0.867750 | 损失较小 |
| 55维预测 dQdV 与159维预测 dQdV 的 retention 差异 | 0.854967 vs 0.853536 | 二者几乎相当 |
| direct retention 是否优于 dQdV bridge | 0.944597 vs 0.854967 | 明显优于 |
| full159 direct 是否明显优于55维 direct | 0.946936 vs 0.944597 | 提升很小 |

综合判断：55维推荐特征包对 compact2 的预测已经足够强；compact2 中介链路的信息损失主要不在 `工况 -> dQdV`，而在 `compact2 dQdV` 对 retention 的表达上限低于 direct retention 模型。当前不建议优先投入 compact2 长窗口训练；更应先分析 direct retention baseline 强的原因，或扩展到 compact3/compact4 检查是否存在更强 dQdV 状态表征。
<!-- compact2_extended_comparison_end -->

## 11. 原有图表
![compact2_predicted_vs_true](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/compact2_predicted_vs_true.png)

![retention_predicted_vs_true](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/retention_predicted_vs_true.png)

![bridge_r2_comparison](C:/Users/pal/projects/batt_soh/outputs/analysis/interval_feature_pack_compact2_retention_bridge/bridge_r2_comparison.png)
