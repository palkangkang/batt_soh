# 原始数据字典 / Raw Data Dictionary

适用范围：`data/raw` 下两类原始文件。

- 明细时序文件：`cycles_*.csv`
- 循环汇总文件：`summary_*.csv`

## A. `cycles_*.csv` 字段定义

| 列名 | 中文名称 | English Name | 单位 | 详细说明 |
|---|---|---|---|---|
| `policy` | 策略名称 | Test Policy | 无 | 测试工况/控制策略标识，通常与文件夹策略名一致（如 `4_8C_80PER_4_8C`）。 |
| `cell_code` | 电芯编号 | Cell Code | 无 | 电芯唯一标识，通常与文件名 `cycles_xxx.csv` 的 `xxx` 对应。 |
| `cycles` | 循环序号 | Cycle Index | 次（count） | 当前记录所在循环编号。 |
| `ts` | 时间戳（循环内） | Timestamp (within cycle) | 秒（s） | 当前循环内累计时间进度。 |
| `I` | 电流 | Current | 安培（A） | 电池端电流，一般正值为充电、负值为放电。 |
| `V` | 电压 | Voltage | 伏特（V） | 电池端电压采样值。 |
| `Temper` | 温度 | Temperature | 摄氏度（°C） | 温度传感器读数。 |
| `flag_chg` | 充电标记 | Charge Flag | 0/1 | `1` 表示充电阶段，`0` 表示非充电阶段。 |
| `flag_dischg` | 放电标记 | Discharge Flag | 0/1 | `1` 表示放电阶段，`0` 表示非放电阶段。 |
| `soc` | 荷电状态 | State of Charge (SoC) | 比例（0~1） | 剩余电量比例估计值。 |
| `ah_chg` | 累计充电容量 | Charged Capacity Accumulated | 安时（Ah） | 充电方向累计容量积分值。 |
| `ah_dischg` | 累计放电容量 | Discharged Capacity Accumulated | 安时（Ah） | 放电方向累计容量积分值。 |
| `ah_total` | 净累计容量 | Net Capacity Accumulated | 安时（Ah） | 通常可理解为 `ah_chg - ah_dischg`。 |
| `wh_chg` | 累计充电能量 | Charged Energy Accumulated | 瓦时（Wh） | 充电方向累计能量积分值。 |
| `wh_dischg` | 累计放电能量 | Discharged Energy Accumulated | 瓦时（Wh） | 放电方向累计能量积分值。 |
| `wh_total` | 净累计能量 | Net Energy Accumulated | 瓦时（Wh） | 通常可理解为 `wh_chg - wh_dischg`。 |

## B. `summary_*.csv` 字段定义

| 列名 | 中文名称 | English Name | 单位 | 详细说明 |
|---|---|---|---|---|
| `policy` | 策略名称 | Test Policy | 无 | 与 `cycles_*.csv` 中一致，表示实验工况策略。 |
| `cell_code` | 电芯编号 | Cell Code | 无 | 与 `cycles_*.csv` 中一致，表示电芯唯一标识。 |
| `cycle` | 循环序号 | Cycle Index | 次（count） | 循环编号（与 `cycles` 含义一致，仅命名不同）。 |
| `QDischarge` | 放电容量 | Discharge Capacity | 安时（Ah） | 该循环放电容量汇总值，可用于 SOH/衰减分析。 |
| `QCharge` | 充电容量 | Charge Capacity | 安时（Ah） | 该循环充电容量汇总值，可用于效率与容量对比分析。 |
| `IR` | 内阻指标 | Internal Resistance | 欧姆（Ohm） | 循环级内阻/等效内阻指标，数值量级与内阻特征相符。 |
| `Tmax` | 最高温度 | Maximum Temperature | 摄氏度（°C） | 循环内最高温度。 |
| `Tavg` | 平均温度 | Average Temperature | 摄氏度（°C） | 循环内平均温度。 |
| `Tmin` | 最低温度 | Minimum Temperature | 摄氏度（°C） | 循环内最低温度。 |
| `chargetime` | 充电时长 | Charge Time | 时长单位（分钟） | 每循环的充电时长汇总值；从数值分布看可能为分钟尺度，建议与你的实验系统定义对齐确认。 |

## 备注

1. `summary_*.csv` 的 `policy`、`cell_code`、`cycle` 与 `cycles_*.csv` 可按电芯与循环维度关联。
2. 温度列（`Temper`、`Tmax`、`Tavg`、`Tmin`）可能出现极端异常值（如 400、-270），建模前建议做异常值清洗。
3. 若后续新增原始文件结构，建议先做 schema 校验，再更新本字典。
