# dQ/dV LSTM 特征包参考说明

本文档整理当前 dQ/dV + LSTM 容量保持率任务中使用过的三组输入特征包，便于后续建模、消融、端侧部署评估和 Colab 复现实验统一口径。

对应训练脚本：`scripts/train_lstm_dqdv_retention.py`

## 1. 特征包总览

| 特征包名称 | 输入维度 | 训练脚本参数 | 主要用途 | 是否包含 `cycle_index_norm` | 是否包含温度特征 |
|---|---:|---|---|---|---|
| 全量特征 | 10 | `--feature-pack main_peak_temp_cycle` | 实验室精度上界、完整主峰信息基线 | 是 | 是 |
| 紧凑4特征 | 4 | `--feature-pack compact_peak_shape` | 最低输入维度、低成本端侧方案基线 | 否 | 否 |
| 紧凑5特征 | 5 | `--feature-pack compact_peak_shape_height` | 当前推荐的工程折中方案 | 否 | 否 |

三个特征包都基于放电 dQ/dV 主峰特征表：

- `data/processed/discharge_dqdv_peak_features_skill_full.csv`
- 字段定义参考：`data/processed/DATA_DICTIONARY.md`

注意：`cycle_index_norm` 不是 dQ/dV 原始峰特征，而是训练脚本中根据同一 `policy + cell_code` 下循环序列位置派生的归一化循环位置特征。

## 2. 全量特征：`main_peak_temp_cycle`

全量特征包共 10 维，包含 9 个 dQ/dV 主峰特征和 1 个归一化循环位置特征。

| 顺序 | 字段名 | 含义 | 单位 | 获取来源 | 工程备注 |
|---:|---|---|---|---|---|
| 1 | `main_peak_voltage_v` | 主峰峰位电压 | V | dQ/dV 主峰检测 | 主峰在电压轴上的位置，反映峰位迁移。 |
| 2 | `main_peak_width_v` | 主峰半高宽 | V | dQ/dV 主峰检测 | 反映峰形展宽，通常与老化过程有关。 |
| 3 | `main_peak_height_dqdv` | 主峰峰高 | Ah/V | dQ/dV 主峰检测 | 主峰强度，端侧获取成本低。 |
| 4 | `main_peak_area` | 主峰局部积分面积 | Ah | 主峰左右基线区间积分 | 反映主峰区间容量贡献。 |
| 5 | `main_peak_prominence` | 主峰显著性 | Ah/V | 主峰相对基线计算 | 对峰邻域和基线定义较敏感。 |
| 6 | `main_peak_skewness` | 主峰局部曲线偏度 | 无 | 主峰左右基线区间统计 | 描述峰形非对称性。 |
| 7 | `main_peak_temp_max_c` | 主峰区间最高温度 | °C | 主峰区间对应原始放电点温度 | 依赖温度通道与峰区间对齐。 |
| 8 | `main_peak_temp_min_c` | 主峰区间最低温度 | °C | 主峰区间对应原始放电点温度 | 依赖温度通道与峰区间对齐。 |
| 9 | `main_peak_temp_avg_c` | 主峰区间平均温度 | °C | 主峰区间对应原始放电点温度 | 依赖温度通道与峰区间对齐。 |
| 10 | `cycle_index_norm` | 同一电芯序列内归一化循环位置 | 无 | 训练脚本派生 | 非传感器特征，泛化部署需谨慎。 |

适用场景：

- 需要建立最高精度基线。
- 数据中有完整温度通道，且不同任务之间循环周期定义一致。
- 离线分析或实验室复现实验。

不建议作为泛化端侧首选的原因：

- 包含 `cycle_index_norm`，对循环编号和寿命阶段定义敏感。
- 包含温度统计项，要求端侧温度采样与 dQ/dV 主峰区间稳定对齐。
- 包含 `main_peak_prominence`，对峰邻域、基线和噪声处理更敏感。

## 3. 紧凑4特征：`compact_peak_shape`

紧凑4特征包共 4 维，只保留主峰形状和峰位相关的低成本特征。

| 顺序 | 字段名 | 含义 | 单位 | 获取来源 | 工程备注 |
|---:|---|---|---|---|---|
| 1 | `main_peak_area` | 主峰局部积分面积 | Ah | 主峰左右基线区间积分 | 容量贡献相关，信息密度较高。 |
| 2 | `main_peak_skewness` | 主峰局部曲线偏度 | 无 | 主峰左右基线区间统计 | 描述峰形非对称性。 |
| 3 | `main_peak_voltage_v` | 主峰峰位电压 | V | dQ/dV 主峰检测 | 描述峰位迁移。 |
| 4 | `main_peak_width_v` | 主峰半高宽 | V | dQ/dV 主峰检测 | 描述峰形展宽。 |

适用场景：

- 希望尽量降低输入维度和端侧特征抽取复杂度。
- 不希望依赖循环周期标签、温度通道或基线敏感的 prominence。
- 作为 compact 系列的最低成本对照基线。

工程特点：

- 特征获取链路最简单。
- 不依赖 `cycle_index_norm`，泛化约束更少。
- 当前训练结果显示精度低于紧凑5特征，因此更适合作为极低成本备选。

## 4. 紧凑5特征：`compact_peak_shape_height`

紧凑5特征包共 5 维，在紧凑4特征基础上增加主峰峰高。

| 顺序 | 字段名 | 含义 | 单位 | 获取来源 | 工程备注 |
|---:|---|---|---|---|---|
| 1 | `main_peak_area` | 主峰局部积分面积 | Ah | 主峰左右基线区间积分 | 容量贡献相关，信息密度较高。 |
| 2 | `main_peak_skewness` | 主峰局部曲线偏度 | 无 | 主峰左右基线区间统计 | 描述峰形非对称性。 |
| 3 | `main_peak_voltage_v` | 主峰峰位电压 | V | dQ/dV 主峰检测 | 描述峰位迁移。 |
| 4 | `main_peak_width_v` | 主峰半高宽 | V | dQ/dV 主峰检测 | 描述峰形展宽。 |
| 5 | `main_peak_height_dqdv` | 主峰峰高 | Ah/V | dQ/dV 主峰检测 | 主峰定位后几乎可直接得到，增量成本低。 |

适用场景：

- 希望在不引入循环周期、温度和 prominence 的前提下提升 compact4 精度。
- 面向端侧或半端侧部署，需要控制特征获取复杂度。
- 作为当前 compact 系列的推荐工程方案。

工程特点：

- 相比 compact4，只新增一个容易获取的峰高特征。
- 不依赖 `cycle_index_norm`，更适合泛化到循环标签不稳定或不同使用工况的数据。
- 不依赖温度特征，降低传感器同步和峰区间温度统计要求。
- 不使用 `main_peak_prominence`，减少对峰邻域和基线定义的敏感性。

## 5. 三组特征包的关系

```text
main_peak_temp_cycle
├─ dQ/dV主峰形状/峰位/幅值特征：voltage, width, height, area, prominence, skewness
├─ 主峰区间温度统计：temp_max, temp_min, temp_avg
└─ 序列位置派生特征：cycle_index_norm

compact_peak_shape
└─ area, skewness, voltage, width

compact_peak_shape_height
└─ compact_peak_shape + height
```

从输入压缩角度看：

- 全量特征到紧凑4特征：去掉温度、prominence、峰高和 `cycle_index_norm`，输入维度从 10 降到 4。
- 紧凑4特征到紧凑5特征：只恢复 `main_peak_height_dqdv`，输入维度从 4 增到 5。
- 紧凑5特征保留了低成本峰形信息，同时避免了 `cycle_index_norm` 的泛化风险。

## 6. 后续任务引用建议

建模脚本应优先引用 feature pack 名称，而不是在新任务中手写列名：

```bash
--feature-pack main_peak_temp_cycle
--feature-pack compact_peak_shape
--feature-pack compact_peak_shape_height
```

若新任务不复用现有脚本，建议仍保持同样的列顺序，避免 checkpoint、归一化器、可视化报告之间出现隐性不一致。

推荐使用口径：

- 精度上界或离线完整信息基线：`main_peak_temp_cycle`
- 极低输入维度和最低算力基线：`compact_peak_shape`
- 当前工程折中方案：`compact_peak_shape_height`

## 7. 注意事项

- `cycle_index_norm` 会扩大模型对循环位置标签的依赖，不建议在泛化端侧任务中作为默认输入。
- 温度特征只有在温度采样、峰区间对齐和工况定义稳定时才建议使用。
- `main_peak_prominence` 对 dQ/dV 曲线平滑、峰邻域和基线定义敏感，跨数据集迁移前应重新验证。
- compact 特征包不等于不需要预处理；仍需稳定完成放电容量积分、dQ/dV 平滑、主峰定位和峰区间统计。
- 若后续切换特征包训练 LSTM，应使用独立输出目录，避免 checkpoint 与 `run_config.json` 混用。

