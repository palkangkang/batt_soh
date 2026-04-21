# 容量-阻抗联合因果分析报告（充电60区间）

## 1. 分析设定
- 时间窗：`H=200` cycles。
- 样本过滤：`0.300 <= q_t <= 1.300`，`ir_t>0`。
- 排除策略前缀：`VARCHARGE`。
- 因果方法：趋势层 + 双方向 `AIPW/GPS` + 区间替代 `DML + cluster bootstrap + BH-FDR`。
- bootstrap：`n=5`，后端 `torch`，设备 `cpu`。
- DML nuisance 模型：`linear`。
- 后端回退说明：`torch_xla_not_available`。

## 2. 数据诊断
| metric | value | notes |
|---|---|---|
| life_rows | 138867.000000 | life_performance 过滤后行数 |
| window_rows | 98893.000000 | 窗口样本行数 |
| analysis_rows | 98893.000000 | 合并后分析样本行数 |
| ir_non_positive_filtered_share | 0.287858 | 窗口构造后样本保留差异（含ir>0等条件影响） |
| unique_clusters | 173.000000 | policy+cell cluster 数 |
| bootstrap_iters | 5.000000 | bootstrap 迭代次数 |
| dml_nuisance_model | nan | DML nuisance 模型: linear |

## 3. 趋势联动结果
| metric | value | notes |
|---|---|---|
| rows_window | 98893.000000 | 窗口样本行数 |
| spearman_cycle_q | -0.709202 | 全局 cycle~q_t Spearman |
| spearman_cycle_ir | -0.223128 | 全局 cycle~ir_t Spearman |
| spearman_y_capdrop_vs_y_irrise | 0.635436 | 全局 y_cap_drop_h 与 y_ir_rise_h Spearman |
| pearson_y_capdrop_vs_y_irrise | 0.864076 | 全局 y_cap_drop_h 与 y_ir_rise_h Pearson |
| share_both_worsen | 0.731579 | 同窗口容量衰减>0 且阻抗上升>0 占比 |
| cell_median_rho_cycle_q | -0.994224 | cell 内 cycle~q_t Spearman 中位数 |
| cell_median_rho_cycle_ir | 0.558026 | cell 内 cycle~ir_t Spearman 中位数 |
| cell_share_opposite_sign_trend | 0.848837 | cell 内容量下降+阻抗上升趋势占比 |

![趋势分布图](./fig_trend_cell_cycle_correlations.png)
- X轴：cell内 Spearman 相关系数。
- Y轴：cell 数量。
- 关键结论：用于验证“容量下降、阻抗上升”是否为跨cell普遍趋势。
- 业务解释：若两者方向稳定且同向恶化共现率高，可支撑联合健康指标设计。

## 4. 双方向因果效应（+1pp）
| direction | direction_label | effect_per_1pp | ci_low | ci_high | bootstrap_success | n_rows | n_clusters | support_shift_share | weight_p95 | weight_p99 | weight_max | ess | treatment_std | gps_sigma | clip_threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dir_rel_1_to_y_cap_drop_h | IR变化(+1pp) -> 容量衰减 | 0.002614 | 0.002446 | 0.003536 | 5 | 98893 | 173 | 0.999990 | 1.196593 | 14.533334 | 14.535072 | 11048.824716 | 0.006563 | 0.006511 | 14.535072 |
| dq_rel_1_to_y_ir_rise_h | 容量变化(+1pp) -> 阻抗上升 | 0.028035 | 0.025866 | 0.028954 | 5 | 98893 | 173 | 0.999990 | 0.448105 | 0.565768 | 0.565775 | 97544.077073 | 0.007093 | 0.007090 | 0.565775 |

![双方向因果效应图](./fig_causal_crosslink_effects.png)
- X轴：因果方向（`IR变化->容量衰减`、`容量变化->IR上升`）。
- Y轴：每 +1pp 处理变化对应的结果变化。
- 关键结论：比较两方向效应大小与CI，避免把共同趋势误解为单向因果。
- 业务解释：用于判断优先控制“热-阻抗路径”还是“容量衰减前置信号路径”。

## 5. 充电60区间双结局风险
### 5.1 风险类别统计
| risk_category | n_bins | mean_cap_effect | mean_ir_effect |
|---|---|---|---|
| uncertain | 40 | -0.086875 | -0.073444 |
| dual_risk | 15 | 0.099123 | 0.280317 |
| cap_dominant_risk | 3 | 0.000429 | 0.000132 |
| ir_dominant_risk | 2 | 0.000183 | 0.000427 |

### 5.2 dual_risk 重点区间
| cross_bin | cross_label | soc_bin | rate_bin | temp_bin | cap_effect_per_1pp | cap_ci_low | cap_ci_high | ir_effect_per_1pp | ir_ci_low | ir_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | s1_r1_t5 | 1 | 1 | 5 | 1.246480 | 0.210515 | 1.273666 | 3.787507 | 2.856400 | 4.171602 |
| 51 | s3_r3_t1 | 3 | 3 | 1 | 0.140462 | 0.048619 | 0.426728 | 0.265315 | 0.180821 | 0.450330 |
| 8 | s1_r2_t3 | 1 | 2 | 3 | 0.056905 | 0.028179 | 0.057178 | 0.090377 | 0.047093 | 0.144998 |
| 7 | s1_r2_t2 | 1 | 2 | 2 | 0.013630 | 0.005946 | 0.025062 | 0.024989 | 0.003100 | 0.027539 |
| 55 | s3_r3_t5 | 3 | 3 | 5 | 0.004522 | 0.002045 | 0.015066 | 0.010172 | 0.009407 | 0.023397 |
| 54 | s3_r3_t4 | 3 | 3 | 4 | 0.007990 | 0.000381 | 0.010405 | 0.009683 | 0.000891 | 0.016217 |
| 16 | s1_r4_t1 | 1 | 4 | 1 | 0.003564 | 0.003133 | 0.005320 | 0.004386 | 0.003016 | 0.006267 |
| 17 | s1_r4_t2 | 1 | 4 | 2 | 0.004344 | 0.003335 | 0.004621 | 0.002970 | 0.002026 | 0.003549 |
| 23 | s2_r1_t3 | 2 | 1 | 3 | 0.001969 | 0.001871 | 0.004440 | 0.002479 | 0.002692 | 0.006491 |
| 18 | s1_r4_t3 | 1 | 4 | 3 | 0.002128 | 0.001466 | 0.002442 | 0.001531 | 0.000540 | 0.002028 |
| 36 | s2_r4_t1 | 2 | 4 | 1 | 0.001553 | 0.001221 | 0.002435 | 0.002104 | 0.001951 | 0.003658 |
| 37 | s2_r4_t2 | 2 | 4 | 2 | 0.001551 | 0.000971 | 0.002366 | 0.001534 | 0.000863 | 0.002509 |

![双结局森林图](./fig_dual_outcome_forest_top_bins.png)
- X轴：每 +1pp 区间份额替代的效应。
- Y轴：cross_bin（按风险强度排序）。
- 关键结论：对比同一工况区间在容量衰减与阻抗上升上的差异。
- 业务解释：指导优先治理“共损伤区间”而非只看单一指标。

![双结局风险矩阵](./fig_cross_bin_dual_risk_matrix.png)
- X轴：温度分位 bin（T1~T5）。
- Y轴：倍率分位 bin（R1~R4）。
- 关键结论：展示 SOC 分层下 dual/cap-dominant/ir-dominant/uncertain 空间分布。
- 业务解释：可直接映射成分层控制策略与试验优先级。

## 6. 关键输出文件
- `trend_capacity_ir_summary.csv`
- `causal_crosslink_effects.csv`
- `causal_substitution_effects_capacity_drop_h.csv`
- `causal_substitution_effects_ir_rise_h.csv`
- `cross_bin_dual_outcome_compare.csv`
- `runtime_backend_info.csv` 与 `runtime_library_versions.csv`

## 7. 复现命令
```bash
pipenv run python scripts/analyze_capacity_ir_joint_causal.py --horizon-cycles 200 --bootstrap-iters 400 --bootstrap-backend numpy --device cpu --output-dir outputs/analysis/capacity_ir_joint_causal
```