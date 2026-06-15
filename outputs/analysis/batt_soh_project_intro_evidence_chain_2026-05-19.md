# batt_soh 项目介绍证据链底稿（2026-05-19）

## 一句话结论

本项目已经形成一条可追溯的离线 SOH 研究链路：从原始运行工况中提取可部署特征，用电压曲线和 dQ/dV 表征连接物理退化状态，用传统 ML 与 LSTM 验证预测能力，再用因果推断解释哪些策略区间可能带来老化风险，最后回到计算量、输入可得性和部署边界。

## 主证据链

运行工况 -> 曲线/区间表征 -> retention/q_discharge 预测 -> 因果风险解释 -> 计算量与部署权衡。

| 链路环节 | 本地证据 | 当前结论 | 使用边界 |
|---|---|---|---|
| 数据口径 | `data/processed/train_policy_cell_samples.csv`, `data/processed/valid_policy_cell_samples.csv` | 以 `policy + cell_code` 为样本单元，训练/验证为 135/52。 | 适合跨电芯/策略验证，不等同在线部署闭环。 |
| 标签口径 | `outputs/analysis/lstm_dqdv_retention_grid_colab_final/train_valid_metrics.csv` | `q_discharge` 是绝对容量，`retention` 是跨电芯可比较 SOH 语言。 | retention 依赖参考容量定义，解释时需说明归一化口径。 |
| 曲线表征 | `outputs/analysis/dqdv_feature_retention_correlation/correlation_global.csv`, `scripts/extract_discharge_dqdv_peak_features.py` | dQ/dV 主峰面积、prominence、高度与 retention 的 Spearman 为 0.8575 / 0.8566 / 0.8565。 | dQ/dV 采集和计算要求高，在线部署成本高于工况统计。 |
| 工况表征 | `scripts/build_charge_aging_path_timeseries.py`, `scripts/analyze_interval_features_to_dqdv.py` | SOC x 倍率 x 温度 60 个 cross-bin 和 159 维统计可压缩为 55 维推荐包。 | 55 维更接近可部署输入，但会牺牲部分曲线细节。 |
| 单步预测 | `outputs/analysis/model_benchmark_policy_discharge/small_model_benchmark_metrics.csv`, `outputs/analysis/lstm_dqdv_retention_grid_colab_final/train_valid_metrics.csv` | XGBoost/RF 容量 valid R2 为 0.877722 / 0.861875；dQ/dV LSTM retention valid R2 为 0.926793。 | 模型胜负必须绑定输入信息量，不能只比较模型名称。 |
| Bridge | `outputs/analysis/compact_target_pack_retention_decision/compact_target_pack_summary.csv` | compact4 oracle/deployable/direct55 R2 为 0.918027 / 0.897059 / 0.941887。 | 工况到 dQ/dV bridge 有解释价值；当前主预测线仍是 direct retention。 |
| 多步预测 | `outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_comparison.md` | H50/M100 下 LightGBM-history H100 R2=0.947182，LSTM-history H100 R2=0.890696。 | 结论属于 long-life holdout 与指定窗口，不外推成所有场景。 |
| 因果解释 | `outputs/analysis/causal_initial_rate_effect/causal_effect_global_window_mean.csv`, `outputs/analysis/causal_rate_temp_mediation/mediation_contribution_summary.csv`, `outputs/analysis/capacity_ir_joint_causal/trend_capacity_ir_summary.csv` | +1C 平均倍率效应为 0.014658，95% CI 0.012841~0.016559；温度中介占比 0.0304。 | 因果结果是离线策略风险解释，不是已验证在线控制策略。 |

## 证据链讲法

1. 运行工况不是结论，只是退化暴露的原始坐标。倍率、温度、SOC 和电压共同决定每个 cycle 暴露在什么物理/控制条件下。
2. 表征层把高频曲线和长期历史压成可建模对象。电压区间 Ah/deltaAh 保留曲线局部变化；dQ/dV 主峰特征压缩为低维健康代理；SOC x 倍率 x 温度 cross-bin 把长期运行轨迹压成累计/增量特征。
3. 预测层验证这些表征是否能解释容量和容量保持率。传统 ML 给出强表格基线，LSTM 用时序结构验证曲线/工况序列是否还有增益。
4. 因果层把预测转化为风险解释。GPS+AIPW 用于估计倍率升高对未来容量下降的方向和量级，DML cross-bin 用于识别候选高风险运行区间，温度中介结果用于约束而不是夸大温度解释。
5. 部署层重新审视输入可得性和计算量。dQ/dV 表征精度强但成本更高；55 维工况包和趋势基线更接近在线部署的低成本路径。

## 核心数值与来源字段

| 结论 | 数值 | 来源文件 | 字段 |
|---|---:|---|---|
| dQ/dV 主峰面积与 retention 相关性 | 0.8575 | `outputs/analysis/dqdv_feature_retention_correlation/correlation_global.csv` | `feature=main_peak_area, spearman_rho` |
| dQ/dV 主峰 prominence 与 retention 相关性 | 0.8566 | `outputs/analysis/dqdv_feature_retention_correlation/correlation_global.csv` | `feature=main_peak_prominence, spearman_rho` |
| dQ/dV 主峰高度与 retention 相关性 | 0.8565 | `outputs/analysis/dqdv_feature_retention_correlation/correlation_global.csv` | `feature=main_peak_height_dqdv, spearman_rho` |
| dQ/dV LSTM 单步 retention valid R2/RMSE | 0.926793 / 0.012491 | `outputs/analysis/lstm_dqdv_retention_grid_colab_final/train_valid_metrics.csv` | `target=retention,set_type=valid,r2,rmse` |
| dQ/dV LSTM 单步 q_discharge valid R2/RMSE | 0.935550 / 0.013421 | `outputs/analysis/lstm_dqdv_retention_grid_colab_final/train_valid_metrics.csv` | `target=q_discharge,set_type=valid,r2,rmse` |
| XGBoost 容量 valid R2/RMSE | 0.877722 / 0.018436 | `outputs/analysis/model_benchmark_policy_discharge/small_model_benchmark_metrics.csv` | `model_name=xgboost,valid_r2,valid_rmse` |
| RandomForest 容量 valid R2/RMSE | 0.861875 / 0.019594 | `outputs/analysis/model_benchmark_policy_discharge/small_model_benchmark_metrics.csv` | `model_name=random_forest,valid_r2,valid_rmse` |
| compact4 oracle bridge valid R2 | 0.918027 | `outputs/analysis/compact_target_pack_retention_decision/compact_target_pack_summary.csv` | `target_pack=compact4,oracle_bridge_valid_r2` |
| compact4 deployable 55 bridge valid R2 | 0.897059 | `outputs/analysis/compact_target_pack_retention_decision/compact_target_pack_summary.csv` | `target_pack=compact4,deployable_bridge_55_valid_r2` |
| compact4 direct55 valid R2 | 0.941887 | `outputs/analysis/compact_target_pack_retention_decision/compact_target_pack_summary.csv` | `target_pack=compact4,direct55_valid_r2` |
| H50/M100 LightGBM-history H100 R2/RMSE | 0.947182 / 0.009033 | `outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_comparison.md` | H100 排名表 |
| H50/M100 LSTM-history H100 R2/RMSE | 0.890696 / 0.012995 | `outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_comparison.md` | H100 排名表 |
| 平均倍率 +1C 对未来相对容量下降效应 | 0.014658 | `outputs/analysis/causal_initial_rate_effect/causal_effect_global_window_mean.csv` | `effect_plus_1c` |
| +1C 效应 95% CI | 0.012841~0.016559 | `outputs/analysis/causal_initial_rate_effect/causal_effect_global_window_mean.csv` | `ci_low,ci_high` |
| 温度中介 NIE/TE | 0.0304 | `outputs/analysis/causal_rate_temp_mediation/mediation_contribution_summary.csv` | `treatment_mode=window_mean,nie_share` |
| 容量下降与阻抗上升 Spearman/Pearson | 0.6354 / 0.8641 | `outputs/analysis/capacity_ir_joint_causal/trend_capacity_ir_summary.csv` | `spearman_y_capdrop_vs_y_irrise, pearson_y_capdrop_vs_y_irrise` |
| 同窗口容量与阻抗同时恶化占比 | 73.16% | `outputs/analysis/capacity_ir_joint_causal/trend_capacity_ir_summary.csv` | `share_both_worsen` |

## PPT 页级 storyboard

| 页 | action title | 主 exhibit | 本地证据 |
|---:|---|---|---|
| 1 | 运行工况已经被整理成一条可追溯的 SOH 证据链 | 证据链标题 | 本底稿 |
| 2 | 单个模型不是结论，证据链才是项目当前最有价值的产物 | 全链路图 | `outputs/analysis/battery_life_research_summary_v2.md` |
| 3 | 按 policy + cell_code 划分让验证更接近跨策略/跨电芯泛化 | train/valid 口径表 | `data/processed/train_policy_cell_samples.csv`, `data/processed/valid_policy_cell_samples.csv` |
| 4 | retention 把绝对容量转换成跨电芯可比较的 SOH 语言 | label 框架 | `outputs/analysis/lstm_dqdv_retention_grid_colab_final/train_valid_metrics.csv` |
| 5 | 倍率、温度、SOC、电压共同定义每个 cycle 的退化暴露 | 工况坐标图 | 原始字段与脚本定义 |
| 6 | 60 个 SOC-倍率-温度区间把长期运行历史压成可部署统计量 | cross-bin 结构图 | `scripts/build_charge_aging_path_timeseries.py` |
| 7 | 电压区间 Ah 特征保留局部曲线变化，但不是最强寿命表征 | 电压区间示意 | `scripts/extract_voltage_interval_features.py` |
| 8 | dQ/dV 主峰特征把放电曲线压缩成强寿命代理量 | dQ/dV 特征图和相关性 | `outputs/analysis/dqdv_feature_retention_correlation/correlation_global.csv` |
| 9 | 树模型基线证明容量预测不能只用线性关系解释 | XGBoost/RF R2 | `outputs/analysis/model_benchmark_policy_discharge/small_model_benchmark_metrics.csv` |
| 10 | dQ/dV LSTM 用低维输入取得当前最清晰的单步 retention 精度 | LSTM 指标与散点 | `outputs/analysis/lstm_dqdv_retention_grid_colab_final/train_valid_metrics.csv` |
| 11 | 工况到 dQ/dV 的 bridge 有解释价值，但 direct retention 仍更强 | compact4 R2 对比 | `outputs/analysis/compact_target_pack_retention_decision/compact_target_pack_summary.csv` |
| 12 | 159 维到 55 维的压缩让工况特征更接近部署约束 | 维度压缩图 | `scripts/analyze_interval_features_to_dqdv.py` |
| 13 | 1:N 估计 N 的比较必须先区分输入信息量 | 输入信息分层表 | 长窗口报告 |
| 14 | 短预测窗口下，历史 retention 趋势本身是非常强的低成本基线 | H100/M50 结论图 | `outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h100_m50_comparison.md` |
| 15 | 预测窗口拉长后，LightGBM-history 在 H100 endpoint 上优于 LSTM-history | H50/M100 H100 排名 | `outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_comparison.md` |
| 16 | 只给 last retention 时，LSTM delta 结构更会利用递推起点 | last-only 消融 | `outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_comparison.md` |
| 17 | GPS+AIPW 将倍率升高与未来容量下降的方向定量化 | +1C 因果曲线 | `outputs/analysis/causal_initial_rate_effect/causal_effect_global_window_mean.csv` |
| 18 | 温度中介存在，但当前数据不支持把温度写成主中介通道 | 中介分解 | `outputs/analysis/causal_rate_temp_mediation/mediation_contribution_summary.csv` |
| 19 | DML cross-bin 结果适合形成候选高风险工况清单 | cross-bin forest | `outputs/analysis/charge_bin_substitution_causal` |
| 20 | 容量和阻抗联合恶化让风险解释不再只依赖容量曲线 | 双风险矩阵 | `outputs/analysis/capacity_ir_joint_causal/trend_capacity_ir_summary.csv` |
| 21 | 部署权衡的第一问题是输入可得性，而不是模型名称 | 输入可得性 x 精度矩阵 | 本地结果综合 |
| 22 | 传统 ML、LSTM 和趋势基线分别占据不同成本-信息位置 | 模型权衡表 | 本地结果综合 |
| 23 | 当前最稳妥主线是 dQ/dV 表征、retention 目标和工况可部署特征的组合 | 结论框架 | 全部证据 |
| 24 | References / Evidence Map | 本地证据与行业文献 | 本地报告 + 文献锚点 |

## 行业知识锚点

- Severson et al., Nature Energy, 2019：早期循环统计和曲线特征可用于寿命预测。
- Attia et al., Nature, 2020：充电策略、快速筛选与寿命优化需要把策略变量和退化结果连接起来。
- ICA/DVA 相关综述与应用文献：dQ/dV、峰位、峰面积等曲线表征常用于诊断活性材料损失、锂库存损失等退化状态。
- 阻抗/SOH 文献：容量衰减和阻抗上升共同影响 BMS 安全与功率能力，容量单指标不足以覆盖全部风险。

## 边界声明

- 本材料不把因果推断结果表述为已经在线验证的控制策略。
- 本材料不把 LightGBM 与 LSTM 的胜负脱离输入信息量、预测窗口和任务口径讨论。
- 本材料不把 smoke 或局部产物当作正式性能结论；所有数值均回溯到已有本地报告或 CSV。
