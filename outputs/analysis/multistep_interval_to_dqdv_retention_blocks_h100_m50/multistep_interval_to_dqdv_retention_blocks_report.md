# recommended55 + compact4 多步未来预测短闭环报告

## 1. 摘要
- history_len: `100`
- horizon: `50`
- block_stride: `150`
- feature_pack: `recommended55`
- target_pack: `compact4`
- history_representation: `summary`
- all-horizon oracle bridge R2: `0.863171`
- all-horizon deployable bridge R2: `0.826557`
- all-horizon direct retention R2: `0.913362`
- all-horizon linear_last10 baseline R2: `0.986496`

## 2. 术语与代称解释
- `recommended55`：从相关性分析中筛选出的 55 个工况统计特征，不包含 `cycles`、`policy` 或 policy 三元参数。
- `compact4`：4 个 dQdV 中介特征，包含 `main_peak_area`、`main_peak_height_dqdv`、`main_peak_voltage_v`、`main_peak_skewness`。
- `dQdV`：放电容量-电压曲线的微分特征，用于描述电芯退化相关的峰形状态。
- `retention`：容量保持率，定义为当前 `q_discharge / q_ref`，其中 `q_ref` 是同一电芯前若干有效循环的参考容量。
- `history_len` 或 `N`：模型可见的历史 cycle 数，本报告为 100。
- `horizon` 或 `M`：要预测的未来 cycle 数，本报告为 50。
- `block_stride`：相邻样本块起点间隔，本报告为 `N+M=150`，用于构造非重叠未来预测样本。
- `H1/H10/H20/H50`：未来第 1/10/20/50 个预测步；`all` 表示把 H1 到 H50 全部预测点合并计算指标。
- `train/valid`：训练集/验证集，按 `policy + cell_code` 电芯组合划分，不按单个 cycle 随机混切。
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
| history_len | 100 | 1 | expected 100 outside smoke |
| horizon | 50 | 1 | expected 50 outside smoke |
| block_stride | 150 | 1 | expected history_len+horizon |
| feature_count | 55 | 1 | nan |
| forbidden_input_columns_present | 0 | 1 | nan |
| target_dim | 4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| train_policy_cell_count | 135 | 1 | nan |
| valid_policy_cell_count | 52 | 1 | nan |
| split_overlap_zero | 1 | 1 | overlap_count=0 |
| train_block_count | 580 | 1 | nan |
| valid_block_count | 251 | 1 | nan |

## 4. dQdV 多步预测指标
| method | target | horizon | n_rows | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interval_to_dqdv_lightgbm | main_peak_area | all | 12550 | 0.944907 | 0.024055 | 0.018203 | 0.000579 |
| interval_to_dqdv_lightgbm | main_peak_area | H1 | 251 | 0.955624 | 0.019620 | 0.015067 | 0.000385 |
| interval_to_dqdv_lightgbm | main_peak_area | H5 | 251 | 0.953071 | 0.020377 | 0.015868 | 0.000415 |
| interval_to_dqdv_lightgbm | main_peak_area | H10 | 251 | 0.949330 | 0.021689 | 0.016739 | 0.000470 |
| interval_to_dqdv_lightgbm | main_peak_area | H20 | 251 | 0.949393 | 0.022410 | 0.017446 | 0.000502 |
| interval_to_dqdv_lightgbm | main_peak_area | H50 | 251 | 0.930554 | 0.029472 | 0.021953 | 0.000869 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | all | 12550 | 0.929040 | 0.090439 | 0.068051 | 0.008179 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H1 | 251 | 0.936419 | 0.079621 | 0.059883 | 0.006340 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H5 | 251 | 0.932457 | 0.081924 | 0.060759 | 0.006711 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H10 | 251 | 0.933007 | 0.083237 | 0.062967 | 0.006928 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H20 | 251 | 0.937721 | 0.081662 | 0.062276 | 0.006669 |
| interval_to_dqdv_lightgbm | main_peak_height_dqdv | H50 | 251 | 0.918656 | 0.104998 | 0.080045 | 0.011025 |
| interval_to_dqdv_lightgbm | main_peak_skewness | all | 12550 | 0.723527 | 0.011139 | 0.008119 | 0.000124 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H1 | 251 | 0.724854 | 0.009684 | 0.007825 | 0.000094 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H5 | 251 | 0.745452 | 0.009842 | 0.007384 | 0.000097 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H10 | 251 | 0.735983 | 0.010317 | 0.007747 | 0.000106 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H20 | 251 | 0.748725 | 0.010587 | 0.008415 | 0.000112 |
| interval_to_dqdv_lightgbm | main_peak_skewness | H50 | 251 | 0.718718 | 0.012097 | 0.008793 | 0.000146 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | all | 12550 | 0.819711 | 0.007294 | 0.005763 | 0.000053 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H1 | 251 | 0.824177 | 0.006885 | 0.005516 | 0.000047 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H5 | 251 | 0.838409 | 0.006495 | 0.005089 | 0.000042 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H10 | 251 | 0.814482 | 0.007034 | 0.005439 | 0.000049 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H20 | 251 | 0.832958 | 0.006759 | 0.005344 | 0.000046 |
| interval_to_dqdv_lightgbm | main_peak_voltage_v | H50 | 251 | 0.817516 | 0.007851 | 0.006193 | 0.000062 |

![dqdv_r2_by_horizon_target](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/dqdv_r2_by_horizon_target.png)

**图 dQdV R2 by horizon 说明**：X 轴是未来预测步 `horizon_step`，Y 轴是验证集 R2；每条线代表一个 dQdV 特征。关键结论：`main_peak_area` 和 `main_peak_height_dqdv` 最稳定，`main_peak_skewness` 最弱但仍有可预测性。

![valid_dqdv_scatter_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/valid_dqdv_scatter_selected_horizons.png)

**图 dQdV scatter 说明**：X 轴是真实 dQdV 特征值，Y 轴是预测 dQdV 特征值，黑色虚线是理想预测 `Y=X`。点越贴近虚线，预测越准。关键结论：面积和峰高散点更贴近虚线，skewness 离散更明显。

![valid_dqdv_residual_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/valid_dqdv_residual_selected_horizons.png)

**图 dQdV residual 说明**：X 轴是残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。分布越集中在 0 附近，误差越小。关键结论：H50 的残差分布比短 horizon 更宽，说明远期 dQdV 预测不确定性增大。

## 5. retention 多步预测链路指标
| method | horizon | n_rows | r2 | rmse | mae | mse |
| --- | --- | --- | --- | --- | --- | --- |
| deployable_bridge | all | 12550 | 0.826557 | 0.017016 | 0.011842 | 0.000290 |
| deployable_bridge | H1 | 251 | 0.766816 | 0.016506 | 0.011077 | 0.000272 |
| deployable_bridge | H5 | 251 | 0.769443 | 0.016802 | 0.011483 | 0.000282 |
| deployable_bridge | H10 | 251 | 0.798226 | 0.016270 | 0.011539 | 0.000265 |
| deployable_bridge | H20 | 251 | 0.811395 | 0.016821 | 0.011530 | 0.000283 |
| deployable_bridge | H50 | 251 | 0.844523 | 0.018901 | 0.013402 | 0.000357 |
| direct_retention | all | 12550 | 0.913362 | 0.012026 | 0.007656 | 0.000145 |
| direct_retention | H1 | 251 | 0.880700 | 0.011807 | 0.007148 | 0.000139 |
| direct_retention | H5 | 251 | 0.885194 | 0.011856 | 0.007165 | 0.000141 |
| direct_retention | H10 | 251 | 0.896934 | 0.011628 | 0.007326 | 0.000135 |
| direct_retention | H20 | 251 | 0.904233 | 0.011986 | 0.007459 | 0.000144 |
| direct_retention | H50 | 251 | 0.932762 | 0.012430 | 0.008257 | 0.000154 |
| linear_last10 | all | 12550 | 0.986496 | 0.004748 | 0.001937 | 0.000023 |
| linear_last10 | H1 | 251 | 0.999459 | 0.000795 | 0.000373 | 0.000001 |
| linear_last10 | H5 | 251 | 0.999220 | 0.000977 | 0.000541 | 0.000001 |
| linear_last10 | H10 | 251 | 0.998101 | 0.001578 | 0.000785 | 0.000002 |
| linear_last10 | H20 | 251 | 0.993274 | 0.003177 | 0.001475 | 0.000010 |
| linear_last10 | H50 | 251 | 0.968077 | 0.008564 | 0.004065 | 0.000073 |
| oracle_bridge | all | 12550 | 0.863171 | 0.015114 | 0.010623 | 0.000228 |
| oracle_bridge | H1 | 251 | 0.815539 | 0.014681 | 0.010151 | 0.000216 |
| oracle_bridge | H5 | 251 | 0.822841 | 0.014728 | 0.010562 | 0.000217 |
| oracle_bridge | H10 | 251 | 0.826756 | 0.015076 | 0.010675 | 0.000227 |
| oracle_bridge | H20 | 251 | 0.840850 | 0.015452 | 0.010760 | 0.000239 |
| oracle_bridge | H50 | 251 | 0.899122 | 0.015225 | 0.011113 | 0.000232 |
| persistence | all | 12550 | 0.924405 | 0.011234 | 0.005414 | 0.000126 |
| persistence | H1 | 251 | 0.999283 | 0.000916 | 0.000470 | 0.000001 |
| persistence | H5 | 251 | 0.997086 | 0.001889 | 0.001114 | 0.000004 |
| persistence | H10 | 251 | 0.990365 | 0.003555 | 0.002053 | 0.000013 |
| persistence | H20 | 251 | 0.965039 | 0.007242 | 0.004132 | 0.000052 |
| persistence | H50 | 251 | 0.826033 | 0.019993 | 0.011012 | 0.000400 |

![retention_r2_by_horizon](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/retention_r2_by_horizon.png)

**图 retention R2 by horizon 说明**：X 轴是未来预测步，Y 轴是 retention 的验证集 R2；每条线代表一种预测路径或基线。关键结论：`linear_last10` 全程最强，`direct_retention` 明显强于 `deployable_bridge`。

![retention_rmse_by_horizon](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/retention_rmse_by_horizon.png)

**图 retention RMSE by horizon 说明**：X 轴是未来预测步，Y 轴是 RMSE；越低表示误差越小。关键结论：`linear_last10` 误差最低，说明容量保持率在 50 cycle 内非常平滑，简单趋势外推已经很强。

![valid_retention_scatter_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/valid_retention_scatter_selected_horizons.png)

**图 retention scatter 说明**：X 轴是真实 retention，Y 轴是预测 retention，黑色虚线是理想预测 `Y=X`。关键结论：`linear_last10` 最贴近虚线；`deployable_bridge` 的散点更分散，说明 dQdV 中介链路传递到 retention 后仍有误差损失。

![valid_retention_residual_selected_horizons](C:/Users/pal/projects/batt_soh/outputs/analysis/multistep_interval_to_dqdv_retention_blocks_h100_m50/valid_retention_residual_selected_horizons.png)

**图 retention residual 说明**：X 轴是 retention 残差 `true - predicted`，Y 轴是样本数量，黑色虚线是 0 残差。关键结论：`linear_last10` 残差最集中，`deployable_bridge` 残差更宽，尤其在 H50 仍落后于 direct 和趋势外推。

## 6. 结论
- 本实验采用非重叠 block，重点评估未来预测而不是相邻滑窗拟合。
- 预测 dQdV 传递到 retention 后的 all-horizon R2 损失为 `0.036614`。
- direct retention 相比 deployable bridge 的 all-horizon R2 优势为 `0.086805`。
- H50 上 deployable bridge R2 为 `0.844523`，direct retention R2 为 `0.932762`。
- persistence all-horizon R2 为 `0.924405`，linear_last10 all-horizon R2 为 `0.986496`；朴素外推基线非常强，说明 retention 在 50 cycle 预测窗口内非常平滑。
- 当前不建议直接进入 LSTM/TCN/Transformer 长训练。更低成本的下一步是预测相对 `linear_last10` 的 residual/delta，或增加 forecast gap，再判断深度时序模型是否真正提供增益。
- compact4 dQdV 仍有解释价值，但在当前多步未来预测口径下，不应作为主预测路径替代 direct retention 或朴素趋势外推。
