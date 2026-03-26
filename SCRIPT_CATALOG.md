# Script Catalog

本文档用于表格化描述仓库内脚本文件的用途、输入与输出，便于检索与复现。

| 所在目录 | 脚本名称 | 脚本用途 | 需要的输入数据 | 产生的文件或图片 |
|---|---|---|---|---|
| `scripts` | `extract_summary_life_performance.py` | 从 `summary_*.csv` 提取循环级寿命表现指标，生成统一放电容量表。 | `data/raw/**/summary_*.csv` | `data/processed/life_performance.csv` |
| `scripts` | `extract_voltage_interval_features.py` | 从 `cycles_*.csv` 提取充/放电电压区间特征（`delta_ah`、持续时间、区间出现次数等）。 | `data/raw/**/cycles_*.csv` | `data/processed/charge_interval_features.csv`<br>`data/processed/discharge_interval_features.csv` |
| `scripts` | `split_train_valid_by_policy_cell.py` | 按 `policy + cell_code` 粒度划分训练/验证样本，并关联 policy 三元参数。 | `data/processed/life_performance.csv`<br>`data/processed/policy_meaning.csv` | `data/processed/train_policy_cell_samples.csv`<br>`data/processed/valid_policy_cell_samples.csv` |
| `scripts` | `plot_train_valid_histograms.py` | 绘制训练/验证样本在关键特征上的分布对比直方图并导出统计表。 | `data/processed/train_policy_cell_samples.csv`<br>`data/processed/valid_policy_cell_samples.csv` | `data/processed/train_valid_hist_compare.png`<br>`data/processed/train_valid_hist_stats.csv` |
| `scripts` | `analyze_charge_range_vs_q_discharge.py` | 分析“充电区间特征（首现区间）与放电容量”的相关性（全局 + policy 分层）。 | `data/processed/charge_interval_features.csv`<br>`data/processed/life_performance.csv` | `outputs/analysis/charge_feature_q_discharge_corr/correlation_by_range.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/merged_dataset_overview.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/coefficients_by_range.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/robustness_by_range.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/top3_scatter.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/correlation_by_policy_range.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/range_policy_stratified_summary.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/policy_spearman_heatmap.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/policy_spearman_boxplot.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/report_charge_vs_q_discharge.md`（中间产物）<br>`outputs/analysis/charge_feature_q_discharge_corr/report_policy_stratified.md`（中间产物） |
| `scripts` | `analyze_discharge_policy_vs_q_discharge.py` | 分析“policy 三元参数 + 放电区间特征（首现区间）与放电容量”的相关性。 | `data/processed/discharge_interval_features.csv`<br>`data/processed/life_performance.csv`<br>`data/processed/policy_meaning.csv` | `outputs/analysis/discharge_policy_q_discharge_corr/univariate_correlation.csv`<br>`outputs/analysis/discharge_policy_q_discharge_corr/discharge_partial_corr_given_policy.csv`<br>`outputs/analysis/discharge_policy_q_discharge_corr/multiple_correlation_model_summary.csv`<br>`outputs/analysis/discharge_policy_q_discharge_corr/combined_model_uplift.csv`<br>`outputs/analysis/discharge_policy_q_discharge_corr/combined_model_standardized_coefficients.csv`<br>`outputs/analysis/discharge_policy_q_discharge_corr/feature_coverage_summary.csv`<br>`outputs/analysis/discharge_policy_q_discharge_corr/univariate_spearman_top12.png`<br>`outputs/analysis/discharge_policy_q_discharge_corr/partial_corr_top10.png`<br>`outputs/analysis/discharge_policy_q_discharge_corr/multiple_correlation_comparison.png`<br>`outputs/analysis/discharge_policy_q_discharge_corr/report_discharge_policy_vs_q_discharge.md`（中间产物） |
| `scripts` | `compare_three_group_models.py` | 对比三组模型（仅充电特征 / 仅 policy 参数 / 组合特征）并输出总体与生命周期分段增益。 | `data/processed/life_performance.csv`<br>`data/processed/charge_interval_features.csv`<br>`data/processed/train_policy_cell_samples.csv`<br>`data/processed/valid_policy_cell_samples.csv` | `outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_group_metrics.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_uplift_summary.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/valid_predictions_by_model.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/policy_level_metrics_by_model.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_metrics_comparison.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/model_uplift_comparison.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/report_three_group_model_comparison.md`（中间产物）<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_metrics_by_model.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_uplift_summary.csv`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_model_metrics.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/lifecycle_stage_uplift.png`<br>`outputs/analysis/charge_feature_q_discharge_corr/model_group_comparison/report_three_group_model_comparison_by_lifecycle.md`（中间产物） |
| `src/models` | `model.py` | 模型模块占位文件（当前为空骨架）。 | 暂无 | 暂无 |
| `src/utils` | `data_loader.py` | 数据加载模块占位文件（当前为空骨架）。 | 暂无 | 暂无 |
| `train` | `train.py` | 训练入口占位文件（当前为空骨架）。 | 暂无 | 暂无 |
| `test` | `test.py` | 测试/评估入口占位文件（当前为空骨架）。 | 暂无 | 暂无 |

## 必要文件与运行机制（非脚本）

- `requirements.txt`：当前本地解释器环境依赖快照（用于快速复现）。
- `google_colab_requirements.txt`：与 Colab 对齐的依赖清单（用于云端训练/推理环境）。
- `AGENTS.md`：工程协作规范与执行边界，包含解释器约定与日志要求。
- `logs/`：会话与任务日志目录，按天记录为 `logs/session_YYYY-MM-DD.md`（仅维护命名规则，不在本文逐条枚举每日日志文件）。

## 维护说明

- 本仓库默认运行环境：`C:\Users\pal\pyenv\colab`（建议在该目录执行 `pipenv run python <script>`）。
- 新增、删除或重命名任何 `.py` 文件后，应同步刷新本表。
- 若脚本输入/输出路径变化，应优先更新本表，再提交代码变更。
