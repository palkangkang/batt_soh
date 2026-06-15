# LightGBM 多步容量保持率预测报告（固定起点，5类工况特征）

## 1. 运行摘要
- 运行时间：2026-05-09 09:56:59
- Python解释器：`C:\Users\pal\.virtualenvs\colab-OixbOpvz\Scripts\python.exe`
- 字体回退：`DejaVu Sans`
- 窗口模式：`fixed_origin`
- 任务口径：`1:100 -> 101:150`
- retention口径：`q_ref=前5个有效循环中位数`，过滤 `q∈[0.3,1.3]`，`retention∈[0.3,1.1]`

## 2. 特征口径
- 充电cross-bin累计：**60** 列
- 充电cross-bin当前增量：**60** 列
- 放电当前区间容量增量：**16** 列
- 放电累计区间容量：**16** 列
- 放电汇总统计：**7** 列
- raw特征维度：**159**
- 聚合后特征维度（last/mean/std/slope）：**636**
- 放电区间口径：`range_count == 1`

## 3. 数据规模
- merged cycle级样本：**140,282**
- 训练组/验证组：**134 / 52**
- 可构造窗口组（train/valid）：**103 / 28**
- 训练窗口数：**103**
- 验证窗口数：**28**
- charge/discharge cycle行数：**140,565 / 140,292**

固定起点说明：每个 `policy+cell_code` 最多贡献 1 个样本，输入严格为 cycles `1:N`，目标严格为 cycles `N+1:N+M`。

## 4. 指标结果
| set_type | aggregation | horizon | n_windows | n_points | n_groups | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | weighted | 1 | 103 | 103 | 103 | 0.002107 | 0.006610 | 0.712726 |
| train | group_macro | 1 | 103 | 103 | 103 | 0.002107 | 0.002107 | nan |
| train | weighted | 10 | 103 | 103 | 103 | 0.002390 | 0.007490 | 0.706418 |
| train | group_macro | 10 | 103 | 103 | 103 | 0.002390 | 0.002390 | nan |
| train | weighted | 50 | 103 | 103 | 103 | 0.003404 | 0.012043 | 0.683191 |
| train | group_macro | 50 | 103 | 103 | 103 | 0.003404 | 0.003404 | nan |
| train | weighted | all | 103 | 5150 | 103 | 0.002780 | 0.009429 | 0.690682 |
| train | group_macro | all | 103 | 5150 | 103 | 0.002780 | 0.002871 | -11.930703 |
| valid | weighted | 1 | 28 | 28 | 28 | 0.004694 | 0.006228 | 0.485332 |
| valid | group_macro | 1 | 28 | 28 | 28 | 0.004694 | 0.004694 | nan |
| valid | weighted | 10 | 28 | 28 | 28 | 0.004736 | 0.006475 | 0.509669 |
| valid | group_macro | 10 | 28 | 28 | 28 | 0.004736 | 0.004736 | nan |
| valid | weighted | 50 | 28 | 28 | 28 | 0.005387 | 0.007673 | 0.616001 |
| valid | group_macro | 50 | 28 | 28 | 28 | 0.005387 | 0.005387 | nan |
| valid | weighted | all | 28 | 1400 | 28 | 0.005155 | 0.007267 | 0.521131 |
| valid | group_macro | all | 28 | 1400 | 28 | 0.005155 | 0.005346 | -91.878117 |

## 5. 模型配置
- 模型：每个 horizon 独立训练一个 `LGBMRegressor`
- n_estimators：`50`
- learning_rate：`0.05`
- num_leaves/max_depth：`31` / `6`
- min_child_samples：`5`
- subsample/colsample_bytree：`0.8` / `0.8`
- reg_alpha/reg_lambda：`0.0` / `1.0`
- n_jobs：`4`

## 6. 结论
- 短期预测（h=1）R2：train=0.712726，valid=0.485332，gap=0.227394。
- 长期预测（h=50）R2：train=0.683191，valid=0.616001，gap=0.067190。
- 验证集 `all` 指标：weighted R2=0.521131，group-macro R2=-91.878117。
- 固定起点下每个电芯只有一个窗口，weighted 与 group-macro 更接近等权电芯评估。

## 7. 数据一致性检查
| check_item | pass_flag | value |
|---|---:|---:|
| check_split_overlap_zero | 1 | 0 |
| check_target_after_input | 1 | 1 |
| check_consecutive_horizon | 1 | 1 |
| check_feature_dim_159_raw | 1 | 159 |
| check_feature_dim_636_aggregated | 1 | 636 |
| check_no_nan_inf_features | 1 | 1 |
| check_fixed_origin_input_1_to_N | 1 | 1 |
| check_fixed_origin_target_N_plus_1_to_N_plus_M | 1 | 1 |

## 8. 散点图
![valid_retention_scatter_horizons](./valid_retention_scatter_horizons.png)