# recommended55 + compact4 多步未来预测短闭环报告

## 1. 摘要
- split_name: `long_life_holdout`
- train_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv`
- valid_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv`
- history_len: `100`
- horizon: `50`
- block_stride: `150`
- block_stage_filter: `none`
- feature_pack: `recommended55`
- target_pack: `compact4`
- history_representation: `summary`
- all-horizon oracle bridge R2: `0.737944`
- all-horizon deployable bridge R2: `0.689862`
- all-horizon direct retention R2: `0.857594`
- all-horizon linear_last10 baseline R2: `0.990611`

## 2. 术语与代称解释
- `recommended55`：从相关性分析中筛选出的 55 个工况统计特征，不包含 `cycles`、`policy` 或 policy 三元参数。
- `compact4`：4 个 dQdV 中介特征，包含 `main_peak_area`、`main_peak_height_dqdv`、`main_peak_voltage_v`、`main_peak_skewness`。
- `dQdV`：放电容量-电压曲线的微分特征，用于描述电芯退化相关的峰形状态。
- `retention`：容量保持率，定义为当前 `q_discharge / q_ref`，其中 `q_ref` 是同一电芯前若干有效循环的参考容量。
- `history_len` 或 `N`：模型可见的历史 cycle 数，本报告为 100。
- `horizon` 或 `M`：要预测的未来 cycle 数，本报告为 50。
- `block_stride`：相邻样本块起点间隔，本报告为 `N+M=150`，用于构造非重叠未来预测样本。
- `H1/H10/H20/H50`：未来第 1/10/20/50 个预测步；`all` 表示把 H1 到 H50 全部预测点合并计算指标。
- `train/valid`：训练集/验证集，来自 `long_life_holdout`，按 `policy + cell_code` 电芯组合划分，不按单个 cycle 随机混切。
- `LightGBM`：本报告使用的表格树模型，用历史工况摘要预测未来 dQdV 或 retention。
- `summary`：历史 100 个 cycle 的特征压缩方式，包括 last、mean、std、min、max、delta、slope。
- `baseline`：不经过本任务中介模型的对照方法，用于判断复杂链路是否真的有增益。
- `oracle_bridge`：使用真实未来 dQdV 预测未来 retention，是中介表征的上限参考，部署时不可直接获得。
- `deployable_bridge`：使用工况预测出来的未来 dQdV 再预测未来 retention，是 dQdV 中介链路的可部署版本。
- `direct_retention`：直接用历史工况摘要预测未来 retention，不经过 dQdV 中介。
- `persistence`：朴素基线，假设未来 retention 等于历史最后一个 retention。
- `linear_last10`：朴素趋势基线，用历史最后 10 个 retention 点线性外推未来 retention。
- `R2/RMSE/MAE/MSE`：R2 越高越好；RMSE、MAE、MSE 越低越好。
- `residual`：残差，统一定义为 `true - predicted`；残差接近 0 说明预测误差小。

## 3. 数据检查
| check_item | value | pass_flag | details |
| --- | --- | --- | --- |
| split_name | long_life_holdout | 1 | declared train/valid split |
| train_split_path | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv | 1 |  |
| valid_split_path | C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv | 1 |  |
| sample_mode | non_overlapping_blocks | 1 | 1:N -> N+1:N+M fixed block samples |
| history_len | 100 | 1 | positive history length |
| horizon | 50 | 1 | positive forecast horizon |
| block_stride | 150 | 1 | expected history_len+horizon |
| block_stage_filter | none | 1 | none or early_train_late_valid |
| train_relative_input_end_max | 0.9445061043285239 | 1 | threshold=0.450 |
| valid_relative_input_start_min | 0.0004470272686633885 | 1 | threshold=0.550 |
| feature_count | 55 | 1 |  |
| include_history_retention_summary | 1 | 1 | optional LightGBM history-retention route flag |
| history_retention_summary_feature_count | 7 | 1 | last,mean,std,min,max,delta,slope when enabled |
| include_last_retention_only | 1 | 1 | optional LightGBM last-retention-only route flag |
| last_retention_only_feature_count | 1 | 1 | single last historical retention scalar when enabled |
| forbidden_input_columns_present | 0 | 1 |  |
| target_dim | 4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| train_policy_cell_count | 147 | 1 |  |
| valid_policy_cell_count | 40 | 1 |  |
| split_overlap_zero | 1 | 1 | overlap_count=0 |
| train_block_count | 519 | 1 |  |
| valid_block_count | 312 | 1 |  |

## 4. dQdV 多步预测指标
| method | target | horizon | n_rows | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interval_to_dqdv_lightgbm | main_peak_area | all | 15600 | 0.857307 | 0.028629 | 0.022011 | 0.000820 |
| interval_to_dqdv_lightgbm | main_peak_area | H1 | 312 | 0.885806 | 0.023876 | 0.019099 | 0.000570 |
| interval_to_dqdv_lightgbm | main_peak_area | H5 | 312 | 0.885460 | 0.023978 | 0.019141 | 0.000575 |
| interval_to_dqdv_lightgbm | main_peak_area | H10 | 312 | 0.881893 | 0.025078 | 0.019657 | 0.000629 |
| interval_to_dqdv_lightgbm | main_peak_area | H20 | 312 | 0.863959 | 0.027429 | 0.021007 | 0.000752 |
| interval_to_dqdv_lightgbm | main_peak_area | H50 | 312 | 0.824360 | 0.033903 | 0.026170 | 0.001149 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | all | 15600 | 0.819112 | 0.109434 | 0.083931 | 0.011976 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H1 | 312 | 0.808544 | 0.106868 | 0.086235 | 0.011421 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H5 | 312 | 0.843307 | 0.095510 | 0.074633 | 0.009122 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H10 | 312 | 0.812696 | 0.107928 | 0.084326 | 0.011649 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H20 | 312 | 0.837034 | 0.101626 | 0.078501 | 0.010328 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H50 | 312 | 0.782259 | 0.127586 | 0.100545 | 0.016278 |
| interval_to_dqdv_lightgbm | main_peak_skewness | all | 15600 | 0.107228 | 0.015902 | 0.009966 | 0.000253 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H1 | 312 | 0.120534 | 0.011753 | 0.009943 | 0.000138 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H5 | 312 | 0.123275 | 0.011891 | 0.009911 | 0.000141 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H10 | 312 | 0.152562 | 0.011995 | 0.009944 | 0.000144 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H20 | 312 | 0.327804 | 0.010606 | 0.008660 | 0.000112 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H50 | 312 | 0.157757 | 0.012178 | 0.009861 | 0.000148 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | all | 15600 | 0.555333 | 0.008458 | 0.006566 | 0.000072 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H1 | 312 | 0.606991 | 0.007853 | 0.006335 | 0.000062 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H5 | 312 | 0.569457 | 0.007989 | 0.006238 | 0.000064 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H10 | 312 | 0.550179 | 0.008440 | 0.006547 | 0.000071 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H20 | 312 | 0.532956 | 0.008508 | 0.006613 | 0.000072 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H50 | 312 | 0.504333 | 0.009243 | 0.007359 | 0.000085 |

![dqdv_r2_by_horizon_target](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/dqdv_r2_by_horizon_target.png)

**图 dQdV R2 by horizon 说明**：X 轴是未来预测步 `horizon_step`，Y 轴是验证集 R2；每条线代表一个 dQdV 特征。关键结论：`main_peak_area` 和 `main_peak_height_dqdv` 最稳定，`main_peak_skewness` 最弱但仍有可预测性。

![valid_dqdv_scatter_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/valid_dqdv_scatter_selected_horizons.png)

**图 dQdV scatter 说明**：X 轴是真实 dQdV 特征值，Y 轴是预测 dQdV 特征值，黑色虚线是理想预测 `Y=X`。点越贴近虚线，预测越准。关键结论：面积和峰高散点更贴近虚线，skewness 离散更明显。

![valid_dqdv_residual_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/valid_dqdv_residual_selected_horizons.png)

**图 dQdV residual 说明**：X 轴是残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。分布越集中在 0 附近，误差越小。关键结论：H50 的残差分布比短 horizon 更宽，说明远期 dQdV 预测不确定性增大。

## 5. retention 多步预测链路指标
| method | horizon | n_rows | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- |
| deployable_bridge | all | 15600 | 0.689862 | 0.019789 | 0.013653 | 0.000392 |
| deployable_bridge | H1 | 312 | 0.650260 | 0.018977 | 0.012698 | 0.000360 |
| deployable_bridge | H5 | 312 | 0.653168 | 0.019015 | 0.013299 | 0.000362 |
| deployable_bridge | H10 | 312 | 0.617812 | 0.020428 | 0.014033 | 0.000417 |
| deployable_bridge | H20 | 312 | 0.673891 | 0.019708 | 0.013587 | 0.000388 |
| deployable_bridge | H50 | 312 | 0.755876 | 0.019420 | 0.013809 | 0.000377 |
| direct_retention | all | 15600 | 0.857594 | 0.013409 | 0.009400 | 0.000180 |
| direct_retention | H1 | 312 | 0.830868 | 0.013197 | 0.008563 | 0.000174 |
| direct_retention | H5 | 312 | 0.837609 | 0.013011 | 0.008673 | 0.000169 |
| direct_retention | H10 | 312 | 0.852263 | 0.012701 | 0.008680 | 0.000161 |
| direct_retention | H20 | 312 | 0.849919 | 0.013370 | 0.009242 | 0.000179 |
| direct_retention | H50 | 312 | 0.872243 | 0.014049 | 0.010485 | 0.000197 |
| direct_retention_last_only | all | 15600 | 0.946000 | 0.008257 | 0.004670 | 0.000068 |
| direct_retention_last_only | H1 | 312 | 0.986888 | 0.003674 | 0.000945 | 0.000014 |
| direct_retention_last_only | H5 | 312 | 0.990518 | 0.003144 | 0.001307 | 0.000010 |
| direct_retention_last_only | H10 | 312 | 0.989036 | 0.003460 | 0.001828 | 0.000012 |
| direct_retention_last_only | H20 | 312 | 0.976955 | 0.005239 | 0.003398 | 0.000027 |
| direct_retention_last_only | H50 | 312 | 0.860273 | 0.014692 | 0.009737 | 0.000216 |
| direct_retention_with_history_summary | all | 15600 | 0.977660 | 0.005311 | 0.002569 | 0.000028 |
| direct_retention_with_history_summary | H1 | 312 | 0.987502 | 0.003587 | 0.001264 | 0.000013 |
| direct_retention_with_history_summary | H5 | 312 | 0.986632 | 0.003733 | 0.001486 | 0.000014 |
| direct_retention_with_history_summary | H10 | 312 | 0.983773 | 0.004209 | 0.001858 | 0.000018 |
| direct_retention_with_history_summary | H20 | 312 | 0.981916 | 0.004641 | 0.002327 | 0.000022 |
| direct_retention_with_history_summary | H50 | 312 | 0.968521 | 0.006974 | 0.003724 | 0.000049 |
| linear_last10 | all | 15600 | 0.990611 | 0.003443 | 0.001156 | 0.000012 |
| linear_last10 | H1 | 312 | 0.997414 | 0.001632 | 0.000353 | 0.000003 |
| linear_last10 | H5 | 312 | 0.997231 | 0.001699 | 0.000439 | 0.000003 |
| linear_last10 | H10 | 312 | 0.997529 | 0.001643 | 0.000509 | 0.000003 |
| linear_last10 | H20 | 312 | 0.993754 | 0.002728 | 0.000943 | 0.000007 |
| linear_last10 | H50 | 312 | 0.981336 | 0.005370 | 0.002197 | 0.000029 |
| oracle_bridge | all | 15600 | 0.737944 | 0.018190 | 0.013350 | 0.000331 |
| oracle_bridge | H1 | 312 | 0.619301 | 0.019799 | 0.014258 | 0.000392 |
| oracle_bridge | H5 | 312 | 0.660857 | 0.018803 | 0.013785 | 0.000354 |
| oracle_bridge | H10 | 312 | 0.690348 | 0.018388 | 0.013233 | 0.000338 |
| oracle_bridge | H20 | 312 | 0.725716 | 0.018075 | 0.012951 | 0.000327 |
| oracle_bridge | H50 | 312 | 0.782428 | 0.018334 | 0.013713 | 0.000336 |
| persistence | all | 15600 | 0.970750 | 0.006077 | 0.003034 | 0.000037 |
| persistence | H1 | 312 | 0.998660 | 0.001174 | 0.000319 | 0.000001 |
| persistence | H5 | 312 | 0.997876 | 0.001488 | 0.000745 | 0.000002 |
| persistence | H10 | 312 | 0.996256 | 0.002022 | 0.001183 | 0.000004 |
| persistence | H20 | 312 | 0.986991 | 0.003936 | 0.002293 | 0.000015 |
| persistence | H50 | 312 | 0.929039 | 0.010470 | 0.006020 | 0.000110 |

![retention_r2_by_horizon](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/retention_r2_by_horizon.png)

**图 retention R2 by horizon 说明**：X 轴是未来预测步，Y 轴是 retention 的验证集 R2；每条线代表一种预测路径或基线。关键结论：`linear_last10` 全程最强，`direct_retention` 明显强于 `deployable_bridge`。

![retention_rmse_by_horizon](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/retention_rmse_by_horizon.png)

**图 retention RMSE by horizon 说明**：X 轴是未来预测步，Y 轴是 RMSE；越低表示误差越小。关键结论：`linear_last10` 误差最低，说明容量保持率在 50 cycle 内非常平滑，简单趋势外推已经很强。

![valid_retention_scatter_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/valid_retention_scatter_selected_horizons.png)

**图 retention scatter 说明**：X 轴是真实 retention，Y 轴是预测 retention，黑色虚线是理想预测 `Y=X`。关键结论：`linear_last10` 最贴近虚线；`deployable_bridge` 的散点更分散，说明 dQdV 中介链路传递到 retention 后仍有误差损失。

![valid_retention_residual_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50/valid_retention_residual_selected_horizons.png)

**图 retention residual 说明**：X 轴是 retention 残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。关键结论：`linear_last10` 残差最集中，`deployable_bridge` 残差更宽，尤其在 H50 仍落后于 direct 和趋势外推。

## 6. 结论
- 本实验采用非重叠 block，重点评估未来预测而不是相邻滑窗拟合。
- 预测 dQdV 传递到 retention 后的 all-horizon R2 损失为 `0.048082`。
- direct retention 相比 deployable bridge 的 all-horizon R2 优势为 `0.167732`。
- H50 上 deployable bridge R2 为 `0.755876`，direct retention R2 为 `0.872243`。
- persistence all-horizon R2 为 `0.970750`，linear_last10 all-horizon R2 为 `0.990611`；朴素外推基线非常强，说明 retention 在 50 cycle 预测窗口内非常平滑。
- 当前不建议直接进入 LSTM/TCN/Transformer 长训练。更低成本的下一步是预测相对 `linear_last10` 的 residual/delta，或增加 forecast gap，再判断深度时序模型是否真正提供增益。
- compact4 dQdV 仍有解释价值，但在当前多步未来预测口径下，不应作为主预测路径替代 direct retention 或朴素趋势外推。
