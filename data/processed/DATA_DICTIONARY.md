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
| `range` | 电压区间 | Voltage Range | 伏特（V） | 特征提取所在电压区间标签。当前步长为 `0.05V`，充电示例：`[3.00,3.05)`；放电示例：`[3.60,3.55)`。 |
| `delta_ah` | 区间容量差 | Capacity Delta in Range | 安时（Ah） | 单次区间片段内累计容量差（结束值 - 起始值）。充电取自 `ah_chg`，放电取自 `ah_dischg`。 |
| `charge_duration_s` | 区间时长 | Duration in Range | 秒（s） | 单次区间片段内时间差（结束 `ts` - 起始 `ts`）。列名沿用 `charge_duration_s`，当 `state=dischg` 时表示放电区间时长。 |
| `avg_temper` | 区间平均温度 | Average Temperature in Range | 摄氏度（°C） | 单次区间片段内温度均值。先做异常值处理（物理范围裁剪 + MAD 鲁棒过滤），再对保留采样点求平均。 |
| `range_count` | 区间出现序号 | Range Occurrence Index | 次（count） | 同一 `state + policy + cell_code + cycles + range` 下该片段第几次出现（从 1 开始）。 |
| `range_total_count` | 区间总出现次数 | Total Occurrences per Range | 次（count） | 同一 `state + policy + cell_code + cycles + range` 下，通过方向性筛选后的有效片段总数。 |

区间特征提取规则摘要：

1. 电压区间步长为 `0.05V`；充电区间 `3.0V -> 3.6V`，放电区间 `3.6V -> 2.8V`。
2. 每个区间按“边界点首索引配对”提取：
   - 充电：先取第一个 `lower_bound±0.001V` 点索引，再取其后第一个 `upper_bound±0.001V` 点索引。
   - 放电：先取第一个 `upper_bound±0.001V` 点索引，再取其后第一个 `lower_bound±0.001V` 点索引。
3. 若同一区间在同一循环中出现多次有效边界配对，会拆分为多行，并通过 `range_count`/`range_total_count` 标记。
4. `avg_temper` 在计算前会先剔除明显异常温度点：先按物理范围过滤，再按 MAD 规则过滤离群点。

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

## C. 充电老化路径统计文件字段定义

适用文件：

- `data/processed/charge_aging_path_timeseries.csv`
- `data/processed/charge_aging_path_final.csv`
- `data/processed/charge_aging_path_bin_edges.csv`
- `data/processed/charge_aging_path_ts_anomalies.csv`
- `data/processed/charge_aging_path_abnormal_cells.csv`
- `data/processed/charge_aging_path_timeseries_abnormal_cells.csv`

口径补充（v6）：

1. 温度清洗：`Temper < 20°C` 或 `Temper > 60°C` 视为异常；按同一 `cycle` 的时间顺序先前向填充（`ffill`），再后向填充（`bfill`）替代。
2. 倍率边界：`rate` 分箱下边界强制为 `0.0`，其余边界由全局分位确定。
3. 时间写盘精度：所有输出文件中的时间列在保存时统一保留 `1` 位有效数字（内部计算仍使用全精度）。
4. 生命周期标签过滤：先在 `life_performance.csv` 中按 `q_min <= q_discharge <= q_max`（默认 `0.3 <= q_discharge <= 1.3`）过滤，再据此计算 `base_q_discharge_100` 与 `cycle_map`；超出区间的 cycle 不参与充电老化路径特征构建。

### C1. `charge_aging_path_timeseries.csv`

按 `policy + cell_code + cycles + cross_bin` 粒度输出充电累计路径时序（每个 cycle 固定 60 个 `cross_bin`，`cycles` 集来自区间过滤后的生命周期标签）。

| 列名 | 中文名称 | 单位 | 说明 |
|---|---|---|---|
| `policy` | 策略名称 | 无 | 测试工况标识。 |
| `cell_code` | 电芯编号 | 无 | 电芯唯一标识。 |
| `cycles` | 循环序号 | 次 | 循环编号。 |
| `soc_bin` | SOC 分箱编号 | 无 | 1~3，对应 `[0,10)`,`[10,90)`,`[90,100]`。 |
| `rate_bin` | 倍率分箱编号 | 无 | 1~4（全局四分位）。 |
| `temp_bin` | 温度分箱编号 | 无 | 1~5（全局五分位）。 |
| `cross_bin` | 交叉分箱编号 | 无 | 1~60，定义：`(soc_bin-1)*20 + (rate_bin-1)*5 + temp_bin`。 |
| `soc_label` | SOC 分箱标签 | 无 | 分箱文本标签。 |
| `rate_label` | 倍率分箱标签 | 无 | 3 位有效数字区间标签，首段下边界固定从 `0` 开始。 |
| `temp_label` | 温度分箱标签 | °C | 四舍五入到整数的区间标签。 |
| `cross_label` | 交叉分箱标签 | 无 | 形如 `s1_r2_t3`。 |
| `cycle_charge_time_h` | 当循环充电时长 | 小时（h） | 当前 cycle 在该交叉分箱的充电时间（仅统计 `flag_chg=1` 且 `I_mid>0`，写盘保留 1 位有效数字）。 |
| `cumulative_charge_time_h` | 累计充电时长 | 小时（h） | 到当前 cycle 为止该交叉分箱的累计充电时长（写盘保留 1 位有效数字）。 |
| `nonzero_cross_bin_count_cycle` | 当循环非零分箱数 | 个 | 当前 cycle 的 60 个 `cross_bin` 中，`cycle_charge_time_h > 0` 的数量。 |
| `is_abnormal_cell` | 异常电芯标记 | 0/1 | 若该电芯存在任一 `dt_s > 3600` 的充电时间跳变，则为 1。 |

### C2. `charge_aging_path_final.csv`

按 `policy + cell_code + cross_bin` 聚合的终态累计结果（仅来自区间过滤后 cycle 集的充电统计）。

| 列名 | 中文名称 | 单位 | 说明 |
|---|---|---|---|
| `policy` | 策略名称 | 无 | 同上。 |
| `cell_code` | 电芯编号 | 无 | 同上。 |
| `soc_bin` / `rate_bin` / `temp_bin` / `cross_bin` | 分箱编号 | 无 | 同 `timeseries` 定义。 |
| `soc_label` / `rate_label` / `temp_label` / `cross_label` | 分箱标签 | 无 | 同 `timeseries` 定义。 |
| `is_abnormal_cell` | 异常电芯标记 | 0/1 | 同 `timeseries`。 |
| `total_charge_time_h` | 总充电时长 | 小时（h） | 全寿命该分箱累计充电时长（写盘保留 1 位有效数字）。 |
| `final_cumulative_charge_time_h` | 终态累计充电时长 | 小时（h） | 等价于 `cumulative_charge_time_h` 的末值（写盘保留 1 位有效数字）。 |
| `max_cycle` | 最大循环序号 | 次 | 该电芯覆盖到的最大循环编号。 |

### C3. `charge_aging_path_bin_edges.csv`

60 个交叉分箱的映射与边界定义表。

| 列名 | 中文名称 | 单位 | 说明 |
|---|---|---|---|
| `soc_bin` / `rate_bin` / `temp_bin` / `cross_bin` | 分箱编号 | 无 | 同上。 |
| `soc_label` / `rate_label` / `temp_label` / `cross_label` | 分箱标签 | 无 | 同上。 |
| `rate_edge_low_raw` / `rate_edge_high_raw` | 倍率原始边界 | C-rate | 未格式化的浮点边界值，首段下边界固定 `0.0`。 |
| `temp_edge_low_raw` / `temp_edge_high_raw` | 温度原始边界 | °C | 未取整的浮点边界值。 |
| `temp_edge_low_int` / `temp_edge_high_int` | 温度显示边界 | °C | 四舍五入后的整数边界。 |

### C4. `charge_aging_path_ts_anomalies.csv`

充电区间级时间跳变异常明细（用于诊断与异常电芯标记，主统计不剔除）。

| 列名 | 中文名称 | 单位 | 说明 |
|---|---|---|---|
| `policy` | 策略名称 | 无 | 同上。 |
| `cell_code` | 电芯编号 | 无 | 同上。 |
| `cycles` | 循环序号 | 次 | 同上。 |
| `ts_prev` / `ts` | 相邻时间戳 | 秒（s） | 区间起止 `ts`（写盘保留 1 位有效数字）。 |
| `dt_s` | 相邻时间差 | 秒（s） | `ts - ts_prev`（写盘保留 1 位有效数字）。 |
| `soc_bin` | SOC 分箱编号 | 无 | 同上。 |
| `soc_mid_percent` | 区间中点 SOC | % | 相邻点平均 SOC（0~100）。 |
| `c_rate_mid` | 区间中点倍率 | C-rate | 相邻点平均电流除以基准容量。 |
| `temp_mid_c` | 区间中点温度 | °C | 清洗后温度（20~60°C 异常替代）在相邻点上的平均值。 |
| `ts_anomaly_reason` | 异常类型 | 文本 | `non_positive_dt` 或 `large_dt_gt_10s`。 |

### C5. `charge_aging_path_abnormal_cells.csv`

异常电芯（`dt_s > 3600`）级别清单（用于 `is_abnormal_cell` 标记，主表仍保留异常电芯数据）。

| 列名 | 中文名称 | 单位 | 说明 |
|---|---|---|---|
| `policy` | 策略名称 | 无 | 同上。 |
| `cell_code` | 电芯编号 | 无 | 同上。 |
| `anomaly_count_gt_600s` | 中等跳变次数 | 次 | `dt_s > 600` 的区间数量。 |
| `anomaly_count_gt_3600s` | 超大跳变次数 | 次 | `dt_s > 3600` 的区间数量。 |
| `max_dt_s` | 最大时间跳变 | 秒（s） | 当前电芯最大 `dt_s`（写盘保留 1 位有效数字）。 |
| `first_anomaly_cycle` | 首次超大跳变循环 | 次/空 | 首次出现 `dt_s > 3600` 的循环编号。 |
| `last_anomaly_cycle` | 最后超大跳变循环 | 次/空 | 最后出现 `dt_s > 3600` 的循环编号。 |

### C6. `charge_aging_path_timeseries_abnormal_cells.csv`

`timeseries` 的异常电芯子集，列结构与 `charge_aging_path_timeseries.csv` 完全一致，仅包含 `is_abnormal_cell=1` 的电芯记录，用于异常样本单独分析。

## D. 放电 dQ/dV 峰特征文件字段定义

适用文件：

- `data/processed/discharge_dqdv_peak_features.csv`
- `data/processed/discharge_dqdv_peak_features_smoke.csv`（冒烟测试子集，列结构一致）

该文件按 `policy + cell_code + cycles` 粒度保存放电状态下 dQ/dV 曲线的统计峰参数，不保存原始 dQ/dV 全曲线。

| 列名 | 中文名称 | English Name | 单位 | 详细说明 |
|---|---|---|---|---|
| `policy` | 策略名称 | Test Policy | 无 | 测试工况/控制策略标识，来源于原始 `cycles_*.csv`。 |
| `cell_code` | 电芯编号 | Cell Code | 无 | 电芯唯一标识。 |
| `cycles` | 循环序号 | Cycle Index | 次（count） | 当前记录对应循环编号。 |
| `n_points_window` | 电压窗口有效点数 | Window Point Count | 点（count） | 电压窗口内（默认 `3.6V~2.8V`）且 `flag_dischg=1` 的有效时序点数量。 |
| `n_points_dqdv` | 有效导数点数 | Valid dQ/dV Point Count | 点（count） | 通过差分与方向性过滤后用于构建 dQ/dV 的离散点数量。 |
| `n_peaks_detected` | 识别峰数量 | Detected Peak Count | 个（count） | 该循环识别到的有效峰数量（输出仅保留 `>=1` 的循环）。 |
| `peak1_voltage_v` | 主峰电压位置 | Peak-1 Voltage Position | 伏特（V） | 主峰（最显著峰）对应电压位置。 |
| `peak1_height_dqdv` | 主峰峰高 | Peak-1 Height (dQ/dV) | Ah/V | 主峰处 dQ/dV 值，保留原始符号（放电常见为负值）。 |
| `peak1_area` | 主峰面积 | Peak-1 Area | Ah | 主峰局部积分面积（保留原始符号）。 |
| `peak1_prominence` | 主峰显著性 | Peak-1 Prominence | Ah/V | 峰显著性指标（基于 `-dQ/dV` 峰检测得到）。 |
| `peak1_width_v` | 主峰半高宽 | Peak-1 Width | 伏特（V） | 峰的半高宽换算到电压轴后的宽度。 |
| `peak2_voltage_v` ~ `peak3_width_v` | 次峰参数 | Peak-2/3 Parameters | 同上 | 第二、第三显著峰参数；若不足 3 个峰，对应列为空。 |

提取规则摘要：

1. 输入范围：递归读取 `data/raw/**/cycles_*.csv`，仅使用放电段（`flag_dischg=1`）并按 `ts` 排序。
2. dQ/dV 构建：使用 `ah_dischg` 与电压 `V` 做离散差分，保留原始 dQ/dV 符号。
3. 边界与有效性：默认严格阈值，要求窗口点数不少于 `50`，有效 dQ/dV 点不少于 `10`。
4. 峰识别：采用 SciPy（Savitzky-Golay 平滑 + `find_peaks`），并仅输出每循环 Top-3 峰统计。
5. 过滤原则：未识别到有效峰的 `policy + cell_code + cycles` 组合不写入输出文件。
