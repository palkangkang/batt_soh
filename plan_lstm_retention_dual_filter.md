# LSTM 改为容量保持率并实施顺序双过滤计划（最终参数版）

## Summary
将 LSTM 主目标改为 `retention`，并固定双过滤顺序：  
1. 先做绝对容量过滤（`q_discharge`）。  
2. 再做相对容量过滤（`retention`）。  

默认值更新为：`retention: 0.30~1.10`。  
`q_ref` 维持每个 `policy+cell_code` 前 5 循环中位数。

## Implementation Changes
1. 标签与过滤流程（训练/调参共用）
- 统一流程：
1. 清洗 `q_discharge`（数值化、去空、`>0`）。
2. 绝对过滤：`q_min <= q_discharge <= q_max`。
3. 基于绝对过滤后样本计算 `q_ref`（前 `q_ref_cycles=5` 循环中位数）。
4. 计算 `retention = q_discharge / q_ref`。
5. 相对过滤：`retention_min <= retention <= retention_max`（默认 `0.30~1.10`）。
- 训练目标 `y` 改为 `retention`。

2. CLI 与接口
- `train_lstm_charge_delta_ah.py` 新增：
- `--retention-min` default `0.30`
- `--retention-max` default `1.10`
- `--q-ref-cycles` default `5`
- `tune_lstm_charge_delta_ah_grid.py` 同步新增并透传以上参数。
- 恢复签名（train/tune）纳入 `retention_min/retention_max/q_ref_cycles`，防止错配恢复。

3. 输出口径
- `valid_predictions.csv` 至少包含：
- `q_discharge`, `q_ref`, `retention_true`, `pred_retention`, `pred_q_discharge`, `residual_retention`, `residual_q_discharge`
- 指标同时输出：
- retention 主目标指标（MSE/RMSE/MAE/R2）
- 回推 Ah 指标（`pred_q_discharge = pred_retention * q_ref`）
- 报告与图表标题切换为保持率主目标，并写明过滤顺序与默认阈值。

4. Colab 同步
- `batt_soh_nn_train_colab.ipynb` 的调参与训练命令补充：
- `--retention-min 0.30 --retention-max 1.10 --q-ref-cycles 5`
- `colab_tuning_and_inference_guide.md` 同步更新默认 retention 区间。

## Test Plan
1. 语法与参数
- 两脚本 `py_compile` 通过。
- `--help` 显示 retention 默认值为 `0.30~1.10`。

2. 顺序过滤验证
- 检查统计链条：
- 原始样本数
- 绝对过滤后样本数
- 可计算 `q_ref` 样本数
- 相对过滤后样本数
- 验证最终样本全部满足两类区间条件。

3. 训练与恢复
- 小样本冒烟跑通 train+tune。
- 中断后同参数可恢复；改 retention 参数会触发签名不匹配。

4. 产物完整性
- 现有标准产物继续生成，且预测表与报告包含 retention 新字段和说明。

## Assumptions
- 绝对过滤默认仍为 `q_discharge: 0.3~1.3`（未变）。
- 相对过滤默认更新为 `retention: 0.30~1.10`。
- `q_ref` 固定“前 5 循环中位数”。
- 特征、切分和 LSTM 架构保持不变。
