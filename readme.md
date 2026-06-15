# batt_soh

`batt_soh` 是一个面向电池 SOH、容量衰减、dQ/dV 表征和工况因果分析的本地研究工程。仓库以脚本化流程为主，覆盖数据整理、特征工程、传统机器学习、序列模型、多步保持率预测、策略风险解释和报告沉淀。

## 项目目标

- 建立 `policy + cell_code` 粒度的数据划分与训练验证口径。
- 从容量、放电区间统计、充电区间统计、dQ/dV 主峰、温度和阻抗等变量中构建可解释特征。
- 对比 RF、XGBoost、LightGBM、LSTM 等模型在容量和 retention 预测任务中的表现。
- 区分预测建模、表征桥接和观测因果推断，避免把相关性、预测性能和可干预策略混写。
- 将关键实验结果沉淀为 Markdown 报告，便于复盘、迁移到 Colab 和后续论文式整理。

## 目录概览

| 路径 | 说明 |
|---|---|
| `data/raw/` | 原始数据留存区，原则上只读。 |
| `data/processed/` | 清洗、划分、特征化后的训练和分析数据。 |
| `scripts/` | 数据处理、建模训练、评估、因果分析和报告生成脚本。 |
| `src/` | 可复用源码骨架和工具逻辑。 |
| `train/` | 训练入口骨架。 |
| `test/` | 测试或评估入口骨架。 |
| `outputs/analysis/` | 分析报告、指标表、图表和模型产物。仓库只选择性跟踪关键报告与轻量结果。 |
| `logs/` | Codex 协作与任务会话日志，按 `logs/session_YYYY-MM-DD.md` 命名。 |
| `AGENTS.md` | 本工程的人机协作规则、执行边界和项目约定。 |

## 主要研究线索

### 1. 数据底座与容量标签

工程围绕固定训练/验证划分展开，样本粒度以 `policy + cell_code` 为主。容量任务中的核心观测量包括 `q_discharge` 和由参考容量派生的 `retention`，相关数据主要位于 `data/processed/`。

### 2. dQ/dV 表征与 SOH 预测

dQ/dV 主峰面积、峰高、峰位、偏度等特征用于描述容量曲线局部形态，并与 retention 预测、目标特征包选择和桥接建模结合。

### 3. 工况特征到容量/retention 的建模

项目包含传统机器学习和序列模型两类路线。传统模型用于建立低成本、可审计的非线性基线；LSTM 与单调后处理用于多步 retention 外推和长寿命 holdout 评估。

### 4. 观测因果与策略风险解释

因果分析脚本围绕倍率、温度、SOC 区间、容量衰减和阻抗上升等变量展开，用于生成候选策略风险排序和受控实验建议。此类结论应理解为观测数据下的策略解释，不等同于已验证上线策略。

## 代表性专项报告

| 报告 | 主题 |
|---|---|
| [电池寿命预估与衰减因果推断技术报告](outputs/analysis/battery_life_decay_causal_full_report_2026-05-29.md) | 汇总数据底座、预测建模、dQ/dV 表征、多步外推和因果解释。 |
| [159维工况特征到 dQ/dV 相关性与可预测性分析](outputs/analysis/interval_features_to_dqdv_correlation/interval_features_to_dqdv_correlation_report.md) | 分析工况特征与 dQ/dV target 的相关性、稳定性和推荐特征包。 |
| [容量-阻抗联合因果分析报告](outputs/analysis/capacity_ir_joint_causal/capacity_ir_joint_causal_report.md) | 从容量衰减和阻抗上升双结局角度评估高风险工况区间。 |
| [long_life_holdout H100/M50 LightGBM/LSTM 评估汇总](outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h100_m50_comparison.md) | 对比趋势基线、LightGBM、LSTM 和历史 retention 增强路线。 |
| [compact target pack retention 决策报告](outputs/analysis/compact_target_pack_retention_decision/compact_target_pack_retention_decision_report.md) | 评估 compact target pack 在 dQ/dV 和 retention 任务中的取舍。 |
| [因果推断逐步指南](outputs/analysis/causal_inference_step_by_step_guide.md) | 解释项目中观测因果分析的基本步骤、口径和使用边界。 |

## 复现与使用建议

1. 先阅读 `AGENTS.md`，确认本仓库的执行边界、Python 环境约定和日志规则。
2. 只读检查数据结构时，优先从 `data/processed/` 和已有 `outputs/analysis/*.md` 报告入手。
3. 运行脚本前先查看脚本参数和输出目录，避免覆盖已有产物。
4. 对长时间训练或 Colab 迁移任务，先跑小样本 smoke 或契约检查，再执行正式训练。
5. 大体积中间预测表、图片批量产物和临时 smoke 结果不默认纳入远端仓库。

## 重要边界

- 本仓库中的分析结论依赖当前数据划分、目标定义和脚本版本。
- 预测指标不自动构成因果结论；观测因果估计也不等同受控实验。
- `outputs/analysis/` 中的产物并非全部托管到远端，远端内容以关键报告、轻量表格和必要图表为主。
