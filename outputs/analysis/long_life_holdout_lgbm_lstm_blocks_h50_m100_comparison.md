# long_life_holdout H50/M100 工况统计 -> retention LightGBM/LSTM 评估汇总

## 1. 直接执行

- split_name: `long_life_holdout`
- train_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/train_policy_cell_samples_long_life_holdout.csv`
- valid_split_path: `C:/Users/pal/projects/batt_soh/data/processed/extrapolation_splits/valid_policy_cell_samples_long_life_holdout.csv`
- 样本口径：`history_len=50`，`horizon=100`，`block_stride=150`，`sample_mode=non_overlapping_blocks`。
- LightGBM-history 输出目录：`C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_history_retention_blocks_h50_m100`
- LSTM baseline_source: `loaded:C:\Users\pal\projects\batt_soh\outputs\analysis\long_life_holdout_lgbm_history_retention_blocks_h50_m100`
- 路线示意图：主路线沿用用户确认的科研论文中文流程图风格；last-only 消融图由脚本生成并写入同一图表目录。

## 2. 路线总表与示意图

| H100排名 | 路线 | 方法 | 输入信息 | 是否使用历史retention | 可部署性/口径 | H10_RMSE | H10_R2 | H50_RMSE | H50_R2 | H100_RMSE | H100_R2 | ALL_RMSE | ALL_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LightGBM enhanced | LightGBM + history retention summary | 55维工况 summary + 7维历史retention summary | 是 | 历史retention增强 tabular | 0.004376 | 0.973706 | 0.006828 | 0.953692 | 0.009033 | 0.947182 | 0.007116 | 0.951945 |
| 2 | LSTM enhanced | LSTM delta 50x56 history-retention-enhanced | 50x55工况序列 + 历史retention通道 | 是 | 历史retention增强序列 | 0.002505 | 0.991381 | 0.006279 | 0.960839 | 0.012995 | 0.890696 | 0.007506 | 0.946531 |
| 3 | LSTM pure operational | LSTM delta strict 50x55 | 50x55工况序列 + last retention递推起点 | 否 | 纯工况序列主对照 | 0.002550 | 0.991069 | 0.007141 | 0.949340 | 0.015180 | 0.850852 | 0.008589 | 0.929996 |
| 4 | last retention only ablation | LSTM delta 1x1 last retention only | 1x1 last retention标量 | 是，仅last | 仅last retention消融 LSTM | 0.002721 | 0.989837 | 0.007342 | 0.946452 | 0.015326 | 0.847961 | 0.008826 | 0.926082 |
| 5 | trend baseline | persistence | 历史最后一个 retention | 是 | 低成本基线 | 0.002680 | 0.990135 | 0.008624 | 0.926122 | 0.018815 | 0.770868 | 0.010526 | 0.894850 |
| 6 | trend baseline | linear_last10 | 历史最后10点 retention 线性外推 | 是 | 低成本强基线 | 0.002877 | 0.988632 | 0.010376 | 0.893050 | 0.020048 | 0.739842 | 0.011745 | 0.869080 |
| 7 | LightGBM | LightGBM direct | 55维工况 summary | 否 | 纯工况 tabular | 0.011923 | 0.804818 | 0.014085 | 0.802915 | 0.020482 | 0.728449 | 0.015331 | 0.776953 |
| 8 | last retention only ablation | LightGBM last retention only | last retention标量 | 是，仅last | 仅last retention消融 tabular | 0.004248 | 0.975224 | 0.012023 | 0.856392 | 0.031579 | 0.354522 | 0.016212 | 0.750579 |

### 2.0 输入数据内容及含义

表 2-0 用来区分“基础工况特征”“历史retention增强”和“last-retention-only消融”三种不同输入口径，避免把模型结构收益和输入信息量收益混写。

| 路线 | 输入形态 | 输入内容 | 含义 |
| --- | --- | --- | --- |
| trend baseline / linear_last10 | 10个历史retention点 | 历史窗口末端最后10个capacity retention观测值 | 只利用容量保持率的局部平滑趋势，作为低成本强基线；不使用工况统计。 |
| trend baseline / persistence | 1个last retention标量 | 历史窗口最后一个capacity retention观测值 | 假设未来保持率等于当前状态，衡量模型是否超过最朴素起点基线。 |
| LightGBM direct | 385维tabular summary | 50个历史cycle内的55个工况基础特征，逐列压缩为last/mean/std/min/max/delta/slope七类统计量。 | 把工况时间序列压成表格摘要，不输入历史retention；用于检验工况统计本身的预测力。 |
| LightGBM + history retention summary | 392维tabular summary | 385维工况summary + 历史retention的last/mean/std/min/max/delta/slope七类summary。 | 把历史retention作为7个统计特征加入LightGBM，但不保留完整retention时间序列。 |
| LSTM pure operational | 50x55工况序列 + last retention递推起点 | 50个历史cycle的55个工况通道；last retention只用于monotonic delta递推起点，不作为输入通道。 | 保留工况时序结构，检验序列模型是否能从工况变化中获得额外泛化收益。 |
| LSTM history-retention-enhanced | 50x56序列 | 50x55工况序列 + 1个历史retention通道。 | 显式输入历史retention全序列；若胜出，结论应标注为history-retention-enhanced，不属于纯工况胜利。 |
| LightGBM last retention only | 1维tabular | 只输入历史窗口最后一个retention标量；禁用55维工况summary、历史retention全序列和7维history summary。 | 同口径消融：只看last retention能否预测未来M100保持率曲线。 |
| LSTM last retention only | 1x1标量序列 | 只输入历史窗口最后一个retention标量，并通过单调delta结构从该起点向未来递推。 | 在短历史H50、预测M100且输入严格限制为last retention时，检验LSTM结构是否比LightGBM更会利用这个起点做未来曲线外推。 |

### 2.1 trend baseline

![trend baseline route diagram](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_trend_baseline_gpt_image2.png)

图 2-1 说明：该路线图已按科研论文中文流程图风格刷新；该路线只使用历史 retention 的平滑趋势，代表最低成本强基线。

### 2.2 LightGBM

![LightGBM route diagram](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_lightgbm_gpt_image2.png)

图 2-2 说明：该路线图已按科研论文中文流程图风格刷新；LightGBM 路线使用 tabular summary，其中增强版额外加入 7 个历史 retention summary 特征。

### 2.3 纯工况 LSTM

![pure operational LSTM route diagram](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_lstm_pure_operational_gpt_image2.png)

图 2-3 说明：该路线图已按科研论文中文流程图风格刷新；纯工况 LSTM 使用 `50x55` 工况统计序列，并用 last retention 作为递推起点，不把历史 retention 作为输入通道。

### 2.4 历史 retention 增强 LSTM

![history retention enhanced LSTM route diagram](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_lstm_history_retention_gpt_image2.png)

图 2-4 说明：该路线图已按科研论文中文流程图风格刷新；增强 LSTM 使用 `50x56`，历史 retention 是显式输入通道，结论必须单独标注。

### 2.5 last retention only 消融

![last retention only ablation route diagram](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_last_retention_only_ablation.png)

图 2-5 说明：该图为脚本生成的科研流程图；LightGBM 与 LSTM 都只接收同一个 last retention 标量，不接收工况统计和历史 retention 序列。

## 3. 图像证据

![H100 RMSE and R2 bar comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_v2_h100_rmse_r2_bar.png)

图 3-1 说明：左图是 H100 RMSE，越低越好；右图是 H100 R2，越高越好。

![R2 by horizon comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_v2_r2_by_horizon.png)

图 3-2 说明：X 轴是未来 horizon step，Y 轴是 valid R2，用于观察全预测窗口的泛化趋势。

![RMSE by horizon comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_v2_rmse_by_horizon.png)

图 3-3 说明：X 轴是未来 horizon step，Y 轴是 valid RMSE，越低表示误差越小。

## 4. H10/H50/H100/ALL 指标与散点残差图

| 路线 | 方法 | horizon | n_rows | MSE | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trend baseline | linear_last10 | H10 | 312 | 0.000008 | 0.002877 | 0.000787 | 0.988632 |
| trend baseline | linear_last10 | H50 | 312 | 0.000108 | 0.010376 | 0.003035 | 0.893050 |
| trend baseline | linear_last10 | H100 | 312 | 0.000402 | 0.020048 | 0.006783 | 0.739842 |
| trend baseline | linear_last10 | ALL | 31200 | 0.000138 | 0.011745 | 0.003214 | 0.869080 |
| trend baseline | persistence | H10 | 312 | 0.000007 | 0.002680 | 0.001193 | 0.990135 |
| trend baseline | persistence | H50 | 312 | 0.000074 | 0.008624 | 0.005090 | 0.926122 |
| trend baseline | persistence | H100 | 312 | 0.000354 | 0.018815 | 0.010944 | 0.770868 |
| trend baseline | persistence | ALL | 31200 | 0.000111 | 0.010526 | 0.005302 | 0.894850 |
| LightGBM | LightGBM direct | H10 | 312 | 0.000142 | 0.011923 | 0.008375 | 0.804818 |
| LightGBM | LightGBM direct | H50 | 312 | 0.000198 | 0.014085 | 0.010116 | 0.802915 |
| LightGBM | LightGBM direct | H100 | 312 | 0.000420 | 0.020482 | 0.014922 | 0.728449 |
| LightGBM | LightGBM direct | ALL | 31200 | 0.000235 | 0.015331 | 0.010792 | 0.776953 |
| LightGBM enhanced | LightGBM + history retention summary | H10 | 312 | 0.000019 | 0.004376 | 0.001756 | 0.973706 |
| LightGBM enhanced | LightGBM + history retention summary | H50 | 312 | 0.000047 | 0.006828 | 0.003739 | 0.953692 |
| LightGBM enhanced | LightGBM + history retention summary | H100 | 312 | 0.000082 | 0.009033 | 0.005424 | 0.947182 |
| LightGBM enhanced | LightGBM + history retention summary | ALL | 31200 | 0.000051 | 0.007116 | 0.003720 | 0.951945 |
| last retention only ablation | LightGBM last retention only | H10 | 312 | 0.000018 | 0.004248 | 0.001932 | 0.975224 |
| last retention only ablation | LightGBM last retention only | H50 | 312 | 0.000145 | 0.012023 | 0.008227 | 0.856392 |
| last retention only ablation | LightGBM last retention only | H100 | 312 | 0.000997 | 0.031579 | 0.021319 | 0.354522 |
| last retention only ablation | LightGBM last retention only | ALL | 31200 | 0.000263 | 0.016212 | 0.009192 | 0.750579 |
| LSTM pure operational | LSTM delta strict 50x55 | H10 | 312 | 0.000007 | 0.002550 | 0.001160 | 0.991069 |
| LSTM pure operational | LSTM delta strict 50x55 | H50 | 312 | 0.000051 | 0.007141 | 0.004879 | 0.949340 |
| LSTM pure operational | LSTM delta strict 50x55 | H100 | 312 | 0.000230 | 0.015180 | 0.010216 | 0.850852 |
| LSTM pure operational | LSTM delta strict 50x55 | ALL | 31200 | 0.000074 | 0.008589 | 0.005010 | 0.929996 |
| LSTM enhanced | LSTM delta 50x56 history-retention-enhanced | H10 | 312 | 0.000006 | 0.002505 | 0.001095 | 0.991381 |
| LSTM enhanced | LSTM delta 50x56 history-retention-enhanced | H50 | 312 | 0.000039 | 0.006279 | 0.004352 | 0.960839 |
| LSTM enhanced | LSTM delta 50x56 history-retention-enhanced | H100 | 312 | 0.000169 | 0.012995 | 0.009225 | 0.890696 |
| LSTM enhanced | LSTM delta 50x56 history-retention-enhanced | ALL | 31200 | 0.000056 | 0.007506 | 0.004540 | 0.946531 |
| last retention only ablation | LSTM delta 1x1 last retention only | H10 | 312 | 0.000007 | 0.002721 | 0.001581 | 0.989837 |
| last retention only ablation | LSTM delta 1x1 last retention only | H50 | 312 | 0.000054 | 0.007342 | 0.005865 | 0.946452 |
| last retention only ablation | LSTM delta 1x1 last retention only | H100 | 312 | 0.000235 | 0.015326 | 0.012266 | 0.847961 |
| last retention only ablation | LSTM delta 1x1 last retention only | ALL | 31200 | 0.000078 | 0.008826 | 0.006226 | 0.926082 |

![H10 scatter comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_h10.png)

![H10 residual comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_h10.png)

![H50 scatter comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_h50.png)

![H50 residual comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_h50.png)

![H100 scatter comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_h100.png)

![H100 residual comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_h100.png)

![ALL scatter comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_all.png)

![ALL residual comparison](long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_all.png)


## 5. 证据链检查

| 证据项 | 状态 | 关键值 |
| --- | --- | --- |
| 数据切分 | PASS | split_name=long_life_holdout |
| 样本块 | PASS | non_overlapping_blocks |
| 窗口 | PASS | history_len=50, horizon=100 |
| 步长 | PASS | block_stride=150 |
| 工况特征 | PASS | feature_count=55 |
| 历史retention增强 | PASS | LightGBM + 7维历史retention summary |
| split重合 | PASS | train/valid policy-cell overlap=0 |
| LSTM baseline契约 | PASS | LSTM加载long_life LightGBM baseline |
| LightGBM last-only | PASS | 输入仅包含 last_retention_only_feature_count=1 的消融路线 |
| LSTM last-only | PASS | 输入为 1x1 last retention 标量序列 |

## 6. 直接回答：LSTM + 历史 retention 是否优于 LightGBM + 历史 retention？

- 直接回答：按 H100 RMSE，`LightGBM + 历史 retention summary` 更好。
- H100 上 `LSTM delta 50x56 history-retention-enhanced` RMSE=`0.012995`、R2=`0.890696`；`LightGBM + history retention summary` RMSE=`0.009033`、R2=`0.947182`。
- H100 RMSE 差值为 `-0.003962`，ALL RMSE 差值为 `-0.000390`；正数表示 LSTM-history 误差更低。
- 但该胜利必须标注为 `50x56 history-retention-enhanced`，不能写成“仅工况统计信息”的胜利。

## 7. last retention only 消融结论

- 直接回答：只给 last retention 标量时，按 H100 RMSE，`LSTM last-retention-only` 更好。
- 任务结论：在短历史 H50、预测 M100 的任务里，如果输入严格限制为 last retention，LSTM 的单调 delta 结构比 LightGBM 更会利用这个起点做未来曲线外推。
- H100 上 `LSTM delta 1x1 last retention only` RMSE=`0.015326`、R2=`0.847961`；`LightGBM last retention only` RMSE=`0.031579`、R2=`0.354522`。
- H100 RMSE 差值为 `0.016253`，ALL RMSE 差值为 `0.007386`；正数表示 LSTM last-only 误差更低。
- 该消融不包含 55维工况统计、不包含历史 retention 全序列，也不包含 7维 history summary，因此可用于回答“单纯 last retention”问题。

## 8. 图表与产物索引

| 产物 | 路径 | 存在 | bytes |
| --- | --- | --- | --- |
| H100 RMSE/R2柱状图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_v2_h100_rmse_r2_bar.png | true | 87381 |
| 跨路线R2曲线 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_v2_r2_by_horizon.png | true | 215394 |
| 跨路线RMSE曲线 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_v2_rmse_by_horizon.png | true | 249869 |
| H10散点图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_h10.png | true | 544718 |
| H10残差图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_h10.png | true | 228757 |
| H50散点图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_h50.png | true | 581696 |
| H50残差图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_h50.png | true | 236538 |
| H100散点图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_h100.png | true | 538455 |
| H100残差图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_h100.png | true | 239961 |
| ALL散点图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_scatter_all.png | true | 927439 |
| ALL残差图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/comparison_residual_all.png | true | 271252 |
| trend baseline路线示意图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_trend_baseline_gpt_image2.png | true | 138476 |
| LightGBM路线示意图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_lightgbm_gpt_image2.png | true | 146128 |
| 纯工况LSTM路线示意图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_lstm_pure_operational_gpt_image2.png | true | 145930 |
| 历史retention增强LSTM路线示意图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_lstm_history_retention_gpt_image2.png | true | 161144 |
| last retention only消融路线示意图 | long_life_holdout_lgbm_lstm_blocks_h50_m100_figures/route_diagrams/route_last_retention_only_ablation.png | true | 169310 |

## 9. 深度交互

- 这次新增的 LightGBM-history 才是回答“LightGBM + 历史 retention”的同口径证据，不能继续用 `linear_last10` 或 pure LightGBM 代替。
- 若 LSTM-history 胜出，合理表述是“历史 retention 增强的序列模型胜出”；若要证明纯工况统计序列更强，应继续看 `50x55` LSTM 与不含历史 retention 的 LightGBM。
- last retention only 消融是回答“单纯 last retention”问题的同口径证据，不应与 `100x55` 工况序列或 `100x56` 历史序列增强结果混写。
- `linear_last10` 仍需要保留，因为它代表短期 H50 retention 平滑趋势的最低成本解释。
