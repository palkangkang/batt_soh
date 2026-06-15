# LSTM 残差修正多步 retention 预测报告

## 1. 任务目的

本报告验证 LSTM 是否能降低已有基线模型在 H50 retention 预测上的残差。H50 指未来第 50 个 cycle 的容量保持率预测；retention 指容量保持率，即当前放电容量除以前 5 个有效 cycle 的参考容量中位数。

本任务不使用 `cycles`、`cycle_index_norm`、`policy`、`cell_code`、`initial_c_rate`、`switch_soc_percent`、`post_switch_c_rate` 作为模型输入。`policy` 和 `cell_code` 只用于分组、划分训练集与验证集、以及报告定位。

## 2. 术语说明

- `55维 recommended feature pack`：相关性分析得到的 55 个工况统计特征，包含充电 cross-bin 累计/增量信息、放电区间容量信息和少量放电汇总统计。
- `LightGBM`：梯度提升树模型，适合中小样本表格特征；本任务中它是原始直接预测基线。
- `linear_last10`：用历史最后 10 个已观测 retention 做线性拟合，再外推未来 H1 到 H50。
- `dQdV compact4`：四个 dQdV 中介特征，包括主峰面积、主峰高度、主峰电压和主峰偏度。
- `deployable_bridge`：先用工况预测 compact4 dQdV，再用预测 dQdV 预测 retention 的可部署中介路线。
- `残差修正`：先得到基线预测，再让 LSTM 预测 `真实值 - 基线预测值`，最终结果为 `基线预测 + LSTM预测残差`。

## 3. 数据检查

| check_item | value | pass_flag | details |
| --- | --- | --- | --- |
| history_len | 100 | 1 | expected 100 outside smoke |
| horizon | 50 | 1 | expected 50 outside smoke |
| block_stride | 150 | 1 | expected history_len+horizon |
| feature_count | 55 | 1 | recommended55 only |
| forbidden_input_columns_present | 0 | 1 |  |
| target_dim | 4 | 1 | main_peak_area,main_peak_height_dqdv,main_peak_voltage_v,main_peak_skewness |
| train_block_count | 580 | 1 |  |
| valid_block_count | 251 | 1 |  |
| residual_scheme_count | 3 | 1 | direct_retention,linear_last10,deployable_bridge |
| uses_policy_or_cycle_as_input | 0 | 1 | policy/cell/cycle are metadata only |

## 4. H50 核心结果

| method | rmse | mae | mse | r2 |
| --- | --- | --- | --- | --- |
| direct_retention | 0.012213 | 0.008096 | 0.000149 | 0.935084 |
| oracle_bridge | 0.015225 | 0.011113 | 0.000232 | 0.899122 |
| deployable_bridge | 0.017719 | 0.012295 | 0.000314 | 0.863364 |
| persistence | 0.019993 | 0.011012 | 0.000400 | 0.826033 |
| linear_last10 | 0.008564 | 0.004065 | 0.000073 | 0.968077 |
| direct_retention_lstm_residual | 0.012215 | 0.008095 | 0.000149 | 0.935062 |
| linear_last10_lstm_residual | 0.008444 | 0.005051 | 0.000071 | 0.968970 |
| deployable_bridge_lstm_residual | 0.017104 | 0.011609 | 0.000293 | 0.872673 |

关键对比：

- direct LightGBM 残差修正：RMSE 从 0.012213 到 0.012215，升高 0.000002。
- linear_last10 残差修正：RMSE 从 0.008564 到 0.008444，降低 0.000121。
- dQdV bridge 残差修正：RMSE 从 0.017719 到 0.017104，降低 0.000614。

## 5. 多 horizon 指标

| method | horizon | horizon_step | rmse | mae | mse | r2 |
| --- | --- | --- | --- | --- | --- | --- |
| direct_retention | all | 0 | 0.011999 | 0.007641 | 0.000144 | 0.913761 |
| direct_retention | H1 | 1 | 0.011871 | 0.007294 | 0.000141 | 0.879403 |
| direct_retention | H10 | 10 | 0.011698 | 0.007377 | 0.000137 | 0.895685 |
| direct_retention | H20 | 20 | 0.011674 | 0.007418 | 0.000136 | 0.909158 |
| direct_retention | H50 | 50 | 0.012213 | 0.008096 | 0.000149 | 0.935084 |
| oracle_bridge | all | 0 | 0.015114 | 0.010623 | 0.000228 | 0.863171 |
| oracle_bridge | H1 | 1 | 0.014681 | 0.010151 | 0.000216 | 0.815539 |
| oracle_bridge | H10 | 10 | 0.015076 | 0.010675 | 0.000227 | 0.826756 |
| oracle_bridge | H20 | 20 | 0.015452 | 0.010760 | 0.000239 | 0.840850 |
| oracle_bridge | H50 | 50 | 0.015225 | 0.011113 | 0.000232 | 0.899122 |
| deployable_bridge | all | 0 | 0.016952 | 0.011847 | 0.000287 | 0.827863 |
| deployable_bridge | H1 | 1 | 0.015961 | 0.011248 | 0.000255 | 0.781980 |
| deployable_bridge | H10 | 10 | 0.016033 | 0.011402 | 0.000257 | 0.804058 |
| deployable_bridge | H20 | 20 | 0.016513 | 0.011589 | 0.000273 | 0.818226 |
| deployable_bridge | H50 | 50 | 0.017719 | 0.012295 | 0.000314 | 0.863364 |
| persistence | all | 0 | 0.011234 | 0.005414 | 0.000126 | 0.924405 |
| persistence | H1 | 1 | 0.000916 | 0.000470 | 0.000001 | 0.999283 |
| persistence | H10 | 10 | 0.003555 | 0.002053 | 0.000013 | 0.990365 |
| persistence | H20 | 20 | 0.007242 | 0.004132 | 0.000052 | 0.965039 |
| persistence | H50 | 50 | 0.019993 | 0.011012 | 0.000400 | 0.826033 |
| linear_last10 | all | 0 | 0.004748 | 0.001937 | 0.000023 | 0.986496 |
| linear_last10 | H1 | 1 | 0.000795 | 0.000373 | 0.000001 | 0.999459 |
| linear_last10 | H10 | 10 | 0.001578 | 0.000785 | 0.000002 | 0.998101 |
| linear_last10 | H20 | 20 | 0.003177 | 0.001475 | 0.000010 | 0.993274 |
| linear_last10 | H50 | 50 | 0.008564 | 0.004065 | 0.000073 | 0.968077 |
| direct_retention_lstm_residual | all | 0 | 0.011998 | 0.007638 | 0.000144 | 0.913778 |
| direct_retention_lstm_residual | H1 | 1 | 0.011877 | 0.007300 | 0.000141 | 0.879265 |
| direct_retention_lstm_residual | H10 | 10 | 0.011691 | 0.007372 | 0.000137 | 0.895811 |
| direct_retention_lstm_residual | H20 | 20 | 0.011677 | 0.007422 | 0.000136 | 0.909114 |
| direct_retention_lstm_residual | H50 | 50 | 0.012215 | 0.008095 | 0.000149 | 0.935062 |
| linear_last10_lstm_residual | all | 0 | 0.005086 | 0.002771 | 0.000026 | 0.984508 |
| linear_last10_lstm_residual | H1 | 1 | 0.001042 | 0.000654 | 0.000001 | 0.999071 |
| linear_last10_lstm_residual | H10 | 10 | 0.001976 | 0.001323 | 0.000004 | 0.997024 |
| linear_last10_lstm_residual | H20 | 20 | 0.003749 | 0.002333 | 0.000014 | 0.990631 |
| linear_last10_lstm_residual | H50 | 50 | 0.008444 | 0.005051 | 0.000071 | 0.968970 |
| deployable_bridge_lstm_residual | all | 0 | 0.016196 | 0.011125 | 0.000262 | 0.842865 |
| deployable_bridge_lstm_residual | H1 | 1 | 0.015223 | 0.010414 | 0.000232 | 0.801679 |
| deployable_bridge_lstm_residual | H10 | 10 | 0.015334 | 0.010740 | 0.000235 | 0.820774 |
| deployable_bridge_lstm_residual | H20 | 20 | 0.015790 | 0.010897 | 0.000249 | 0.833811 |
| deployable_bridge_lstm_residual | H50 | 50 | 0.017104 | 0.011609 | 0.000293 | 0.872673 |

## 6. LSTM 训练过程

| scheme | best_epoch | best_valid_loss |
| --- | --- | --- |
| direct_retention | 7 | 4064.593262 |
| linear_last10 | 9 | 0.145758 |
| deployable_bridge | 30 | 22.950155 |

![LSTM loss curve](C:/Users/pal/projects/batt_soh/outputs/analysis/lstm_residual_multistep_retention_h100_m50/loss_curve.png)

图 1 说明：横轴是 epoch，表示训练轮次；纵轴是标准化残差上的加权 MSE，数值越低表示 LSTM 对残差拟合越好。实线为训练集，虚线为验证集。若验证集 loss 下降不明显或反弹，说明残差可能噪声较强或样本不足。

## 7. H50 散点图

![H50 scatter](C:/Users/pal/projects/batt_soh/outputs/analysis/lstm_residual_multistep_retention_h100_m50/h50_retention_scatter.png)

图 2 说明：横轴是真实 H50 retention，纵轴是预测 H50 retention。虚线是理想预测线 `Y=X`。点越靠近虚线，预测越准确；若残差修正后点云更贴近虚线，说明 LSTM 残差模型有效。

## 8. H50 残差分布

![H50 residual histogram](C:/Users/pal/projects/batt_soh/outputs/analysis/lstm_residual_multistep_retention_h100_m50/h50_residual_distribution.png)

图 3 说明：横轴是残差 `真实 retention - 预测 retention`，纵轴是样本块数量。分布越集中在 0 附近，说明预测误差越小；分布整体偏正或偏负，说明模型存在系统性低估或高估。

## 9. H50 残差随真实 retention 变化

![H50 residual vs true](C:/Users/pal/projects/batt_soh/outputs/analysis/lstm_residual_multistep_retention_h100_m50/h50_residual_vs_true.png)

图 4 说明：横轴是真实 H50 retention，纵轴是残差。若残差随真实 retention 呈明显斜率或分段结构，说明模型在不同老化阶段存在系统性偏差；若 LSTM 修正后该结构减弱，说明时序信息补到了 LightGBM 或线性外推的盲区。

## 10. 结论

本实验的核心判断标准不是 LSTM 单独能否预测 retention，而是 LSTM 是否能降低 direct LightGBM、linear_last10 或 dQdV bridge 的 H50 残差。如果残差修正不能降低 H50 RMSE，说明当前误差更可能来自噪声、未来工况不可观测或样本分布差异，而不是单纯缺少 LSTM 时序建模。
