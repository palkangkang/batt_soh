# LightGBM 滑窗 vs 固定起点对比报告

## 1. 运行摘要
- 生成时间：2026-05-09 10:00:27
- 滑窗口径：每个电芯生成多个 `s:s+N-1 -> s+N:s+N+M-1` 样本。
- 固定起点口径：每个电芯最多 1 个 `1:N -> N+1:N+M` 样本。
- 固定起点使用 `min_child_samples=5` 适配小样本；滑窗正式结果复用已有产物。

## 2. 有效样本数量
| window_mode | N | M | train_windows | valid_windows | train_groups_available | valid_groups_available |
|---|---:|---:|---:|---:|---:|---:|
| rolling | 100 | 50 | 73826 | 32279 | 132 | 51 |
| fixed_origin | 100 | 50 | 103 | 28 | 103 | 28 |

## 3. 验证集预测精度
| window_mode | aggregation | horizon | n_windows | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|---:|
| fixed_origin | group_macro | 1 | 28 | 0.004694 | 0.004694 | nan |
| fixed_origin | group_macro | 10 | 28 | 0.004736 | 0.004736 | nan |
| fixed_origin | group_macro | 50 | 28 | 0.005387 | 0.005387 | nan |
| fixed_origin | group_macro | all | 28 | 0.005155 | 0.005346 | -91.878117 |
| fixed_origin | weighted | 1 | 28 | 0.004694 | 0.006228 | 0.485332 |
| fixed_origin | weighted | 10 | 28 | 0.004736 | 0.006475 | 0.509669 |
| fixed_origin | weighted | 50 | 28 | 0.005387 | 0.007673 | 0.616001 |
| fixed_origin | weighted | all | 28 | 0.005155 | 0.007267 | 0.521131 |
| rolling | group_macro | 1 | 32279 | 0.007637 | 0.009386 | 0.631960 |
| rolling | group_macro | 10 | 32279 | 0.007664 | 0.009448 | 0.726487 |
| rolling | group_macro | 50 | 32279 | 0.009139 | 0.011541 | 0.803530 |
| rolling | group_macro | all | 32279 | 0.008177 | 0.010243 | 0.784243 |
| rolling | weighted | 1 | 32279 | 0.006613 | 0.009092 | 0.916402 |
| rolling | weighted | 10 | 32279 | 0.006749 | 0.009204 | 0.922484 |
| rolling | weighted | 50 | 32279 | 0.008261 | 0.011189 | 0.929445 |
| rolling | weighted | all | 32279 | 0.007334 | 0.009921 | 0.926646 |

## 4. 散点图
![rolling_valid_scatter](../lgbm_operational_multistep_retention/valid_retention_scatter_horizons.png)
![fixed_origin_valid_scatter](../lgbm_operational_multistep_retention_fixed_origin/valid_retention_scatter_horizons.png)

## 5. 结论说明
- 滑窗样本量远大于固定起点，适合学习滚动退化动态，但 window-weighted 指标可能偏乐观。
- 固定起点更接近严格早期预测任务，但 train/valid 样本只有电芯组数量级，指标更容易受单个电芯影响。
- 固定起点下每个电芯最多一个窗口，MAE/RMSE 的 weighted 与 group_macro 通常更接近。
- 固定起点的单 horizon group-macro R2 不可定义，因为每个电芯组在该 horizon 只有 1 个点；`all` horizon 的 group-macro R2 仍可参考。
- 本对比重点是样本构造口径差异，不是严格超参公平竞赛。