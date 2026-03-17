# 处理后数据字典 / Processed Data Dictionary

适用范围：`data/processed` 目录下的特征与汇总文件。

- `charge_interval_features.csv`
- `discharge_interval_features.csv`
- `life_performance.csv`

## A. 区间特征文件字段定义

适用文件：

- `data/processed/charge_interval_features.csv`
- `data/processed/discharge_interval_features.csv`

这两个文件列结构一致，差异主要体现在 `state` 与 `range` 取值方向。

| 列名 | 中文名称 | English Name | 单位 | 详细说明 |
|---|---|---|---|---|
| `state` | 状态 | State | 无 | 特征来源状态：`chg` 表示充电特征，`dischg` 表示放电特征。 |
| `policy` | 策略名称 | Test Policy | 无 | 测试工况/控制策略标识，来源于原始数据 `policy`。 |
| `cell_code` | 电芯编号 | Cell Code | 无 | 电芯唯一标识，来源于原始数据 `cell_code`。 |
| `cycles` | 循环序号 | Cycle Index | 次（count） | 当前记录对应的循环编号。 |
| `range` | 电压区间 | Voltage Range | 伏特（V） | 特征提取所在电压区间标签。充电示例：`[3.0,3.1)`；放电示例：`[3.6,3.5)`。 |
| `delta_ah` | 区间容量差 | Capacity Delta in Range | 安时（Ah） | 单次区间片段内累计容量差（结束值 - 起始值）。充电取自 `ah_chg`，放电取自 `ah_dischg`。 |
| `charge_duration_s` | 区间时长 | Duration in Range | 秒（s） | 单次区间片段内时间差（结束 `ts` - 起始 `ts`）。列名沿用 `charge_duration_s`，当 `state=dischg` 时表示放电区间时长。 |
| `range_count` | 区间出现序号 | Range Occurrence Index | 次（count） | 同一 `state + policy + cell_code + cycles + range` 下该片段第几次出现（从 1 开始）。 |
| `range_total_count` | 区间总出现次数 | Total Occurrences per Range | 次（count） | 同一 `state + policy + cell_code + cycles + range` 下，通过方向性筛选后的有效片段总数。 |

区间特征提取规则摘要：

1. 充电特征仅保留“低压到高压”的上升过程，电压区间 `3.0V -> 3.6V`，步长 `0.1V`。
2. 放电特征仅保留“高压到低压”的下降过程，电压区间 `3.6V -> 2.8V`，步长 `0.1V`。
3. 若同一区间在同一循环中出现多次，会拆分为多行并通过 `range_count`/`range_total_count` 标记。

## B. 寿命表现文件字段定义

适用文件：

- `data/processed/life_performance.csv`

该文件按 `policy + cell_code + cycles` 粒度汇总循环级寿命相关指标。

| 列名 | 中文名称 | English Name | 单位 | 详细说明 |
|---|---|---|---|---|
| `policy` | 策略名称 | Test Policy | 无 | 测试工况/控制策略标识。 |
| `cell_code` | 电芯编号 | Cell Code | 无 | 电芯唯一标识。 |
| `cycles` | 循环序号 | Cycle Index | 次（count） | 循环编号（来源于 summary 文件中的 `cycle` 字段）。 |
| `q_discharge` | 放电容量 | Discharge Capacity | 安时（Ah，建议再与设备定义核对） | 循环级放电容量指标，来源于 `summary_*.csv` 的 `QDischarge`。 |
| `t_max` | 最高温度 | Maximum Temperature | 摄氏度（°C） | 循环级最高温度指标，来源于 `summary_*.csv` 的 `Tmax`。 |

## 维护建议

1. 新增或修改特征列时，同步更新本字典并注明来源脚本。
2. 若特征提取规则变化（电压区间、步长、方向筛选），需同步更新“区间特征提取规则摘要”。
3. 对 `q_discharge` 等关键指标，建议在建模前做单位与物理范围复核。
