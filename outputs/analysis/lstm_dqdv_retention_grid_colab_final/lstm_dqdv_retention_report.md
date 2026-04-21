# LSTM 训练报告：dQdV 主峰特征拟合容量保持率

## 1. 运行摘要
- 运行时间：2026-04-17 06:36:11
- Python 解释器：`/usr/bin/python3.12`
- 设备：`cuda`
- 序列模式：`prefix_full`
- 特征包：`main_peak_temp_cycle`
- q 绝对过滤：`0.3 <= q_discharge <= 1.3`
- retention 过滤：`0.3 <= retention <= 1.1`，`q_ref`=前 `5` 个有效循环中位数
- checkpoint 快照间隔：每 `10` 轮

## 2. 数据概览
- 合并后 cycle 级样本数：**140,560**
- 训练样本数：**98,686**
- 验证样本数：**41,874**
- 每个时间步输入维度：`10`（主峰9维 + cycle_index_norm）

## 3. 指标结果
| target | set_type | n_samples | MSE | RMSE | MAE | R2 |
|---|---|---:|---:|---:|---:|---:|
| retention | train | 98686 | 0.00018578 | 0.013630 | 0.008847 | 0.922302 |
| retention | valid | 41874 | 0.00015603 | 0.012491 | 0.009110 | 0.926793 |
| q_discharge | train | 98686 | 0.00021460 | 0.014649 | 0.009489 | 0.927824 |
| q_discharge | valid | 41874 | 0.00018013 | 0.013421 | 0.009797 | 0.935550 |

## 4. 图表
- 最佳 epoch：**6**
![loss_curve](./loss_curve.png)

![valid_scatter](./valid_scatter.png)