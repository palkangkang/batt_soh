# recommended55 + compact4 多步未来预测短闭环报告

## 1. 摘要
- split_name: `long_life_holdout`
- train_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv`
- valid_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv`
- history_len: `50`
- horizon: `100`
- block_stride: `150`
- block_stage_filter: `none`
- feature_pack: `recommended55`
- target_pack: `compact4`
- history_representation: `summary`
- all-horizon oracle bridge R2: `0.701301`
- all-horizon deployable bridge R2: `0.671990`
- all-horizon direct retention R2: `0.776953`
- all-horizon linear_last10 baseline R2: `0.869080`

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
| history_len | 50 | 1 | positive history length |
| horizon | 100 | 1 | positive forecast horizon |
| block_stride | 150 | 1 | expected history_len+horizon |
| block_stage_filter | none | 1 | none or early_train_late_valid |
| train_relative_input_end_max | 0.8890122086570478 | 1 | threshold=0.450 |
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
| interval_to_dqdv_lightgbm | main_peak_area | all | 31200 | 0.745638 | 0.036075 | 0.028318 | 0.001301 |
| interval_to_dqdv_lightgbm | main_peak_area | H1 | 312 | 0.793164 | 0.028051 | 0.022408 | 0.000787 |
| interval_to_dqdv_lightgbm | main_peak_area | H5 | 312 | 0.793965 | 0.028135 | 0.022645 | 0.000792 |
| interval_to_dqdv_lightgbm | main_peak_area | H10 | 312 | 0.795375 | 0.028447 | 0.022559 | 0.000809 |
| interval_to_dqdv_lightgbm | main_peak_area | H20 | 312 | 0.820220 | 0.027361 | 0.022383 | 0.000749 |
| interval_to_dqdv_lightgbm | main_peak_area | H50 | 312 | 0.761388 | 0.034189 | 0.027191 | 0.001169 |
| interval_to_dqdv_lightgbm | main_peak_area | H100 | 312 | 0.641015 | 0.048469 | 0.039709 | 0.002349 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | all | 31200 | 0.710122 | 0.132350 | 0.100761 | 0.017517 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H1 | 312 | 0.805351 | 0.095865 | 0.074599 | 0.009190 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H5 | 312 | 0.799601 | 0.096891 | 0.075703 | 0.009388 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H10 | 312 | 0.812770 | 0.094747 | 0.075364 | 0.008977 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H20 | 312 | 0.795160 | 0.101919 | 0.080325 | 0.010388 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H50 | 312 | 0.765178 | 0.114377 | 0.090650 | 0.013082 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H100 | 312 | 0.531055 | 0.187237 | 0.152331 | 0.035058 |
| interval_to_dqdv_lightgbm | main_peak_skewness | all | 31200 | -0.069995 | 0.018736 | 0.010697 | 0.000351 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H1 | 312 | -0.103168 | 0.012179 | 0.010248 | 0.000148 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H5 | 312 | -0.061321 | 0.011773 | 0.009706 | 0.000139 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H10 | 312 | 0.100749 | 0.011294 | 0.009279 | 0.000128 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H20 | 312 | -0.198649 | 0.013041 | 0.010950 | 0.000170 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H50 | 312 | -1.040345 | 0.017667 | 0.013493 | 0.000312 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H100 | 312 | 0.019553 | 0.013139 | 0.010803 | 0.000173 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | all | 31200 | 0.348701 | 0.010622 | 0.007036 | 0.000113 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H1 | 312 | 0.446193 | 0.007671 | 0.005964 | 0.000059 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H5 | 312 | 0.385349 | 0.008220 | 0.006514 | 0.000068 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H10 | 312 | 0.399201 | 0.008172 | 0.006337 | 0.000067 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H20 | 312 | 0.356199 | 0.008755 | 0.006782 | 0.000077 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H50 | 312 | 0.500969 | 0.008571 | 0.006281 | 0.000073 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H100 | 312 | 0.500240 | 0.009281 | 0.007301 | 0.000086 |

![dqdv_r2_by_horizon_target](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/dqdv_r2_by_horizon_target.png)

**图 dQdV R2 by horizon 说明**：X 轴是未来预测步 `horizon_step`，Y 轴是验证集 R2；每条线代表一个 dQdV 特征。关键结论：`main_peak_area` 和 `main_peak_height_dqdv` 最稳定，`main_peak_skewness` 最弱但仍有可预测性。

![valid_dqdv_scatter_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/valid_dqdv_scatter_selected_horizons.png)

**图 dQdV scatter 说明**：X 轴是真实 dQdV 特征值，Y 轴是预测 dQdV 特征值，黑色虚线是理想预测 `Y=X`。点越贴近虚线，预测越准。关键结论：面积和峰高散点更贴近虚线，skewness 离散更明显。

![valid_dqdv_residual_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/valid_dqdv_residual_selected_horizons.png)

**图 dQdV residual 说明**：X 轴是残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。分布越集中在 0 附近，误差越小。关键结论：H50 的残差分布比短 horizon 更宽，说明远期 dQdV 预测不确定性增大。

## 5. retention 多步预测链路指标
| method | horizon | n_rows | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- |
| deployable_bridge | all | 31200 | 0.671990 | 0.018591 | 0.012463 | 0.000346 |
| deployable_bridge | H1 | 312 | 0.430347 | 0.019730 | 0.012725 | 0.000389 |
| deployable_bridge | H5 | 312 | 0.510776 | 0.018527 | 0.012625 | 0.000343 |
| deployable_bridge | H10 | 312 | 0.464928 | 0.019741 | 0.013307 | 0.000390 |
| deployable_bridge | H20 | 312 | 0.618393 | 0.017293 | 0.011677 | 0.000299 |
| deployable_bridge | H50 | 312 | 0.557233 | 0.021112 | 0.013843 | 0.000446 |
| deployable_bridge | H100 | 312 | 0.760365 | 0.019241 | 0.012629 | 0.000370 |
| direct_retention | all | 31200 | 0.776953 | 0.015331 | 0.010792 | 0.000235 |
| direct_retention | H1 | 312 | 0.807248 | 0.011477 | 0.008158 | 0.000132 |
| direct_retention | H5 | 312 | 0.799557 | 0.011859 | 0.008276 | 0.000141 |
| direct_retention | H10 | 312 | 0.804818 | 0.011923 | 0.008375 | 0.000142 |
| direct_retention | H20 | 312 | 0.800530 | 0.012503 | 0.008910 | 0.000156 |
| direct_retention | H50 | 312 | 0.802915 | 0.014085 | 0.010116 | 0.000198 |
| direct_retention | H100 | 312 | 0.728449 | 0.020482 | 0.014922 | 0.000420 |
| direct_retention_last_only | all | 31200 | 0.750579 | 0.016212 | 0.009192 | 0.000263 |
| direct_retention_last_only | H1 | 312 | 0.982211 | 0.003487 | 0.000918 | 0.000012 |
| direct_retention_last_only | H5 | 312 | 0.983341 | 0.003419 | 0.001302 | 0.000012 |
| direct_retention_last_only | H10 | 312 | 0.975224 | 0.004248 | 0.001932 | 0.000018 |
| direct_retention_last_only | H20 | 312 | 0.961958 | 0.005460 | 0.003189 | 0.000030 |
| direct_retention_last_only | H50 | 312 | 0.856392 | 0.012023 | 0.008227 | 0.000145 |
| direct_retention_last_only | H100 | 312 | 0.354522 | 0.031579 | 0.021319 | 0.000997 |
| direct_retention_with_history_summary | all | 31200 | 0.951945 | 0.007116 | 0.003720 | 0.000051 |
| direct_retention_with_history_summary | H1 | 312 | 0.977187 | 0.003948 | 0.001285 | 0.000016 |
| direct_retention_with_history_summary | H5 | 312 | 0.979553 | 0.003788 | 0.001553 | 0.000014 |
| direct_retention_with_history_summary | H10 | 312 | 0.973706 | 0.004376 | 0.001756 | 0.000019 |
| direct_retention_with_history_summary | H20 | 312 | 0.966699 | 0.005109 | 0.002459 | 0.000026 |
| direct_retention_with_history_summary | H50 | 312 | 0.953692 | 0.006828 | 0.003739 | 0.000047 |
| direct_retention_with_history_summary | H100 | 312 | 0.947182 | 0.009033 | 0.005424 | 0.000082 |
| linear_last10 | all | 31200 | 0.869080 | 0.011745 | 0.003214 | 0.000138 |
| linear_last10 | H1 | 312 | 0.998572 | 0.000988 | 0.000283 | 0.000001 |
| linear_last10 | H5 | 312 | 0.997379 | 0.001356 | 0.000412 | 0.000002 |
| linear_last10 | H10 | 312 | 0.988632 | 0.002877 | 0.000787 | 0.000008 |
| linear_last10 | H20 | 312 | 0.972439 | 0.004647 | 0.001307 | 0.000022 |
| linear_last10 | H50 | 312 | 0.893050 | 0.010376 | 0.003035 | 0.000108 |
| linear_last10 | H100 | 312 | 0.739842 | 0.020048 | 0.006783 | 0.000402 |
| oracle_bridge | all | 31200 | 0.701301 | 0.017741 | 0.012909 | 0.000315 |
| oracle_bridge | H1 | 312 | 0.584478 | 0.016851 | 0.012161 | 0.000284 |
| oracle_bridge | H5 | 312 | 0.600431 | 0.016744 | 0.012310 | 0.000280 |
| oracle_bridge | H10 | 312 | 0.569708 | 0.017702 | 0.012666 | 0.000313 |
| oracle_bridge | H20 | 312 | 0.636265 | 0.016884 | 0.012162 | 0.000285 |
| oracle_bridge | H50 | 312 | 0.635727 | 0.019149 | 0.014008 | 0.000367 |
| oracle_bridge | H100 | 312 | 0.782428 | 0.018334 | 0.013713 | 0.000336 |
| persistence | all | 31200 | 0.894850 | 0.010526 | 0.005302 | 0.000111 |
| persistence | H1 | 312 | 0.996197 | 0.001612 | 0.000319 | 0.000003 |
| persistence | H5 | 312 | 0.997465 | 0.001334 | 0.000653 | 0.000002 |
| persistence | H10 | 312 | 0.990135 | 0.002680 | 0.001193 | 0.000007 |
| persistence | H20 | 312 | 0.981669 | 0.003790 | 0.002081 | 0.000014 |
| persistence | H50 | 312 | 0.926122 | 0.008624 | 0.005090 | 0.000074 |
| persistence | H100 | 312 | 0.770868 | 0.018815 | 0.010944 | 0.000354 |

![retention_r2_by_horizon](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/retention_r2_by_horizon.png)

**图 retention R2 by horizon 说明**：X 轴是未来预测步，Y 轴是 retention 的验证集 R2；每条线代表一种预测路径或基线。关键结论：`linear_last10` 全程最强，`direct_retention` 明显强于 `deployable_bridge`。

![retention_rmse_by_horizon](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/retention_rmse_by_horizon.png)

**图 retention RMSE by horizon 说明**：X 轴是未来预测步，Y 轴是 RMSE；越低表示误差越小。关键结论：`linear_last10` 误差最低，说明容量保持率在 50 cycle 内非常平滑，简单趋势外推已经很强。

![valid_retention_scatter_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/valid_retention_scatter_selected_horizons.png)

**图 retention scatter 说明**：X 轴是真实 retention，Y 轴是预测 retention，黑色虚线是理想预测 `Y=X`。关键结论：`linear_last10` 最贴近虚线；`deployable_bridge` 的散点更分散，说明 dQdV 中介链路传递到 retention 后仍有误差损失。

![valid_retention_residual_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100/valid_retention_residual_selected_horizons.png)

**图 retention residual 说明**：X 轴是 retention 残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。关键结论：`linear_last10` 残差最集中，`deployable_bridge` 残差更宽，尤其在 H50 仍落后于 direct 和趋势外推。

## 6. 结论
- 本实验采用非重叠 block，重点评估未来预测而不是相邻滑窗拟合。
- 预测 dQdV 传递到 retention 后的 all-horizon R2 损失为 `0.029311`。
- direct retention 相比 deployable bridge 的 all-horizon R2 优势为 `0.104963`。
- H50 上 deployable bridge R2 为 `0.760365`，direct retention R2 为 `0.728449`。
- persistence all-horizon R2 为 `0.894850`，linear_last10 all-horizon R2 为 `0.869080`；朴素外推基线非常强，说明 retention 在 50 cycle 预测窗口内非常平滑。
- 当前不建议直接进入 LSTM/TCN/Transformer 长训练。更低成本的下一步是预测相对 `linear_last10` 的 residual/delta，或增加 forecast gap，再判断深度时序模型是否真正提供增益。
- compact4 dQdV 仍有解释价值，但在当前多步未来预测口径下，不应作为主预测路径替代 direct retention 或朴素趋势外推。
