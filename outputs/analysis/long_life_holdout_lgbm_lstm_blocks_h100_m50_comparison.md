# long_life_holdout H100/M50 工况统计 -> retention LightGBM/LSTM 评估汇总

## 1. 直接执行

- split_name: `long_life_holdout`
- train_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv`
- valid_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv`
- 样本口径：`history_len=100`，`horizon=50`，`block_stride=150`，`sample_mode=non_overlapping_blocks`。
- LightGBM-history 输出目录：`C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_history_retention_blocks_h100_m50`
- LSTM baseline_source: `loaded:C:\Users\pal\projects\batt_soh\outputs\analysis\long_life_holdout_lgbm_blocks_h100_m50`
- 路线示意图：按用户确认的科研论文中文流程图风格，使用 Codex 内置图片生成工具生成，并复制到项目图表目录。

## 2. 路线总表与示意图

| H50排名 | 路线 | 方法 | 输入信息 | 是否使用历史retention | 可部署性/口径 | H10_RMSE | H10_R2 | H50_RMSE | H50_R2 | ALL_RMSE | ALL_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | trend baseline | linear_last10 | 历史最后10点 retention 线性外推 | 是 | 低成本强基线 | 0.001643 | 0.997529 | 0.005370 | 0.981336 | 0.003443 | 0.990611 |
| 2 | LSTM pure operational | LSTM delta strict 100x55 | 100x55工况序列 + last retention递推起点 | 否 | 纯工况序列主对照 | 0.001427 | 0.998136 | 0.006508 | 0.972583 | 0.003896 | 0.987979 |
| 3 | LSTM enhanced | LSTM delta 100x56 history-retention-enhanced | 100x55工况序列 + 历史retention通道 | 是 | 历史retention增强序列 | 0.001559 | 0.997775 | 0.006921 | 0.968999 | 0.004171 | 0.986224 |
| 4 | LightGBM enhanced | LightGBM + history retention summary | 55维工况 summary + 7维历史retention summary | 是 | 历史retention增强 tabular | 0.004209 | 0.983773 | 0.006974 | 0.968521 | 0.005311 | 0.977660 |
| 5 | trend baseline | persistence | 历史最后一个 retention | 是 | 低成本基线 | 0.002022 | 0.996256 | 0.010470 | 0.929039 | 0.006077 | 0.970750 |
| 6 | LightGBM | LightGBM direct | 55维工况 summary | 否 | 纯工况 tabular | 0.012619 | 0.854161 | 0.014691 | 0.860301 | 0.013461 | 0.856480 |

### 2.1 trend baseline

![trend baseline route diagram](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_trend_baseline_gpt_image2.png)

图 2-1 说明：该路线图已按科研论文中文流程图风格刷新；该路线只使用历史 retention 的平滑趋势，代表最低成本强基线。

### 2.2 LightGBM

![LightGBM route diagram](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_lightgbm_gpt_image2.png)

图 2-2 说明：该路线图已按科研论文中文流程图风格刷新；LightGBM 路线使用 tabular summary，其中增强版额外加入 7 个历史 retention summary 特征。

### 2.3 纯工况 LSTM

![pure operational LSTM route diagram](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_lstm_pure_operational_gpt_image2.png)

图 2-3 说明：该路线图已按科研论文中文流程图风格刷新；纯工况 LSTM 使用 `100x55` 工况统计序列，并用 last retention 作为递推起点，不把历史 retention 作为输入通道。

### 2.4 历史 retention 增强 LSTM

![history retention enhanced LSTM route diagram](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_lstm_history_retention_gpt_image2.png)

图 2-4 说明：该路线图已按科研论文中文流程图风格刷新；增强 LSTM 使用 `100x56`，历史 retention 是显式输入通道，结论必须单独标注。

## 3. 图像证据

![H50 RMSE and R2 bar comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_v2_h50_rmse_r2_bar.png)

图 3-1 说明：左图是 H50 RMSE，越低越好；右图是 H50 R2，越高越好。

![R2 by horizon comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_v2_r2_by_horizon.png)

图 3-2 说明：X 轴是未来 horizon step，Y 轴是 valid R2，用于观察全预测窗口的泛化趋势。

![RMSE by horizon comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_v2_rmse_by_horizon.png)

图 3-3 说明：X 轴是未来 horizon step，Y 轴是 valid RMSE，越低表示误差越小。

## 4. H10/H50/ALL 指标与散点残差图

| 路线 | 方法 | horizon | n_rows | MSE | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trend baseline | linear_last10 | H10 | 312 | 0.000003 | 0.001643 | 0.000509 | 0.997529 |
| trend baseline | linear_last10 | H50 | 312 | 0.000029 | 0.005370 | 0.002197 | 0.981336 |
| trend baseline | linear_last10 | ALL | 15600 | 0.000012 | 0.003443 | 0.001156 | 0.990611 |
| trend baseline | persistence | H10 | 312 | 0.000004 | 0.002022 | 0.001183 | 0.996256 |
| trend baseline | persistence | H50 | 312 | 0.000110 | 0.010470 | 0.006020 | 0.929039 |
| trend baseline | persistence | ALL | 15600 | 0.000037 | 0.006077 | 0.003034 | 0.970750 |
| LightGBM | LightGBM direct | H10 | 312 | 0.000159 | 0.012619 | 0.008637 | 0.854161 |
| LightGBM | LightGBM direct | H50 | 312 | 0.000216 | 0.014691 | 0.010817 | 0.860301 |
| LightGBM | LightGBM direct | ALL | 15600 | 0.000181 | 0.013461 | 0.009420 | 0.856480 |
| LightGBM enhanced | LightGBM + history retention summary | H10 | 312 | 0.000018 | 0.004209 | 0.001858 | 0.983773 |
| LightGBM enhanced | LightGBM + history retention summary | H50 | 312 | 0.000049 | 0.006974 | 0.003724 | 0.968521 |
| LightGBM enhanced | LightGBM + history retention summary | ALL | 15600 | 0.000028 | 0.005311 | 0.002569 | 0.977660 |
| LSTM pure operational | LSTM delta strict 100x55 | H10 | 312 | 0.000002 | 0.001427 | 0.000984 | 0.998136 |
| LSTM pure operational | LSTM delta strict 100x55 | H50 | 312 | 0.000042 | 0.006508 | 0.004721 | 0.972583 |
| LSTM pure operational | LSTM delta strict 100x55 | ALL | 15600 | 0.000015 | 0.003896 | 0.002434 | 0.987979 |
| LSTM enhanced | LSTM delta 100x56 history-retention-enhanced | H10 | 312 | 0.000002 | 0.001559 | 0.001092 | 0.997775 |
| LSTM enhanced | LSTM delta 100x56 history-retention-enhanced | H50 | 312 | 0.000048 | 0.006921 | 0.005063 | 0.968999 |
| LSTM enhanced | LSTM delta 100x56 history-retention-enhanced | ALL | 15600 | 0.000017 | 0.004171 | 0.002659 | 0.986224 |

![H10 scatter comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_scatter_h10.png)

![H10 residual comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_residual_h10.png)

![H50 scatter comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_scatter_h50.png)

![H50 residual comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_residual_h50.png)

![ALL scatter comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_scatter_all.png)

![ALL residual comparison](long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_residual_all.png)


## 5. 证据链检查

| 证据项 | 状态 | 关键值 |
| --- | --- | --- |
| 数据切分 | PASS | split_name=long_life_holdout |
| 样本块 | PASS | non_overlapping_blocks |
| 窗口 | PASS | history_len=100, horizon=50 |
| 步长 | PASS | block_stride=150 |
| 工况特征 | PASS | feature_count=55 |
| 历史retention增强 | PASS | LightGBM + 7维历史retention summary |
| split重合 | PASS | train/valid policy-cell overlap=0 |
| LSTM baseline契约 | PASS | LSTM加载long_life LightGBM baseline |

## 6. 直接回答：LSTM + 历史 retention 是否优于 LightGBM + 历史 retention？

- 直接回答：按 H50 RMSE，`LSTM + 历史 retention` 更好。
- H50 上 `LSTM delta 100x56 history-retention-enhanced` RMSE=`0.006921`、R2=`0.968999`；`LightGBM + history retention summary` RMSE=`0.006974`、R2=`0.968521`。
- H50 RMSE 差值为 `0.000053`，ALL RMSE 差值为 `0.001140`；正数表示 LSTM-history 误差更低。
- 但该胜利必须标注为 `100x56 history-retention-enhanced`，不能写成“仅工况统计信息”的胜利。

## 7. 图表与产物索引

| 产物 | 路径 | 存在 | bytes |
| --- | --- | --- | --- |
| H50 RMSE/R2柱状图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_v2_h50_rmse_r2_bar.png | true | 75181 |
| 跨路线R2曲线 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_v2_r2_by_horizon.png | true | 155277 |
| 跨路线RMSE曲线 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_v2_rmse_by_horizon.png | true | 185247 |
| H10散点图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_scatter_h10.png | true | 419037 |
| H10残差图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_residual_h10.png | true | 173565 |
| H50散点图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_scatter_h50.png | true | 396655 |
| H50残差图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_residual_h50.png | true | 176319 |
| ALL散点图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_scatter_all.png | true | 601907 |
| ALL残差图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/comparison_residual_all.png | true | 177374 |
| trend baseline路线示意图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_trend_baseline_gpt_image2.png | true | 1229214 |
| LightGBM路线示意图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_lightgbm_gpt_image2.png | true | 1167612 |
| 纯工况LSTM路线示意图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_lstm_pure_operational_gpt_image2.png | true | 1171571 |
| 历史retention增强LSTM路线示意图 | long_life_holdout_lgbm_lstm_blocks_h100_m50_figures/route_diagrams/route_lstm_history_retention_gpt_image2.png | true | 1180520 |

## 8. 深度交互

- 这次新增的 LightGBM-history 才是回答“LightGBM + 历史 retention”的同口径证据，不能继续用 `linear_last10` 或 pure LightGBM 代替。
- 若 LSTM-history 胜出，合理表述是“历史 retention 增强的序列模型胜出”；若要证明纯工况统计序列更强，应继续看 `100x55` LSTM 与不含历史 retention 的 LightGBM。
- `linear_last10` 仍需要保留，因为它代表短期 H50 retention 平滑趋势的最低成本解释。
