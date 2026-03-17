# Script Catalog

本文件用于汇总仓库内脚本文件的用途、输入与输出，便于快速检索与交接。

| 所在目录 | 脚本名称 | 脚本用途 | 需要的输入数据 | 产生的文件或图片 |
|---|---|---|---|---|
| `scripts` | `extract_summary_life_performance.py` | 从 `summary_*.csv` 提取循环级寿命指标，生成统一的 `life_performance.csv` | `data/raw/**/summary_*.csv` | `data/processed/life_performance.csv` |
| `scripts` | `extract_voltage_interval_features.py` | 从 `cycles_*.csv` 提取充/放电电压区间特征（`delta_ah`、时长、区间出现次数） | `data/raw/**/cycles_*.csv` | `data/processed/charge_interval_features.csv`<br>`data/processed/discharge_interval_features.csv` |
| `scripts` | `split_train_valid_by_policy_cell.py` | 按 `policy + cell_code` 粒度划分训练/验证样本，并附带 policy 三元参数 | `data/processed/life_performance.csv`<br>`data/processed/policy_meaning.csv` | `data/processed/train_policy_cell_samples.csv`<br>`data/processed/valid_policy_cell_samples.csv` |
| `scripts` | `plot_train_valid_histograms.py` | 绘制训练/验证划分在关键特征上的分布对比直方图，并导出统计表 | `data/processed/train_policy_cell_samples.csv`<br>`data/processed/valid_policy_cell_samples.csv` | `data/processed/train_valid_hist_compare.png`<br>`data/processed/train_valid_hist_stats.csv` |
| `scripts` | `analyze_charge_range_vs_q_discharge.py` | 分析首次出现充电区间特征与放电容量相关性（全局 + policy 分层），生成表格、图和报告 | `data/processed/charge_interval_features.csv`<br>`data/processed/life_performance.csv` | `outputs/analysis/charge_feature_q_discharge_corr/correlation_by_range.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/merged_dataset_overview.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/coefficients_by_range.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/robustness_by_range.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/top3_scatter.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/report_charge_vs_q_discharge.md`<br>`outputs/analysis/charge_feature_q_discharge_corr/correlation_by_policy_range.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/range_policy_stratified_summary.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/policy_spearman_heatmap.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/policy_spearman_boxplot.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/report_policy_stratified.md` |
| `scripts` | `compare_three_group_models.py` | 对比三组模型（仅充电特征 / 仅 policy 三元参数 / 组合特征），输出增益报告，并补充按寿命阶段（早/中/晚）分段增益 | `data/processed/life_performance.csv`<br>`data/processed/charge_interval_features.csv`<br>`data/processed/train_policy_cell_samples.csv`<br>`data/processed/valid_policy_cell_samples.csv` | `outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_group_metrics.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_uplift_summary.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/valid_predictions_by_model.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/policy_level_metrics_by_model.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_metrics_comparison.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_uplift_comparison.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/report_three_group_model_comparison.md`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_metrics_by_model.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_uplift_summary.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_model_metrics.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_uplift.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/report_three_group_model_comparison_by_lifecycle.md` |
| `src/models` | `model.py` | 模型模块占位文件（当前为空） | 暂无 | 暂无 |
| `src/utils` | `data_loader.py` | 数据加载模块占位文件（当前为空） | 暂无 | 暂无 |
| `train` | `train.py` | 训练入口占位文件（当前为空） | 暂无 | 暂无 |
| `test` | `test.py` | 测试/评估入口占位文件（当前为空） | 暂无 | 暂无 |

## 维护建议

- 新增脚本后，同步补充本表。
- 若脚本输入/输出路径变化，优先更新本表再提交代码。
- 占位文件开始实现后，及时将“用途/输入/输出”从“暂无”更新为实际内容。
