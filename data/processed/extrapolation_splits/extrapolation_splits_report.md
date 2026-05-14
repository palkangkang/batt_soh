# 外推评估划分报告

## 1. 目的

本报告生成补充的 train/valid 划分，用于检验模型是否具备长寿命外推、寿命阶段外推或工况策略族外推能力。它不替换原有 balanced split。

## 2. 术语说明

- `balanced`：当前仓库已有的分层覆盖型划分，训练集和验证集都覆盖不同寿命与工况策略。
- `long_life_holdout`：把 `max_cycles` 大于等于阈值的长寿命电芯放入验证集，用于检验长寿命外推。
- `policy_family_holdout`：按策略族留出验证集，用于检验未见工况策略族上的泛化。
- `max_cycles`：同一 `policy + cell_code` 电芯样本可观测到的最大循环数，只用于划分和报告，不作为模型输入。
- `split_overlap_count`：训练集与验证集重叠的 `policy + cell_code` 数量，电芯级划分应为 0。

## 3. 参数

- long_life_threshold: `1000`
- policy_family_mode: `high_initial_rate`
- high_initial_rate_threshold: `5.0`

## 4. 划分摘要

| split | set_type | n_policy_cell | n_policy | max_cycles_min | max_cycles_q25 | max_cycles_median | max_cycles_q75 | max_cycles_max | max_cycles_mean | count_ge800 | count_ge1000 | split_overlap_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | train | 135 | 81 | 101.000 | 494.500 | 562.000 | 884.000 | 2189.000 | 731.296 | 51 | 27 | 0 |
| balanced | valid | 52 | 47 | 240.000 | 531.250 | 809.500 | 986.000 | 2237.000 | 805.731 | 27 | 13 | 0 |
| long_life_holdout | train | 147 | 82 | 101.000 | 487.000 | 536.000 | 809.000 | 988.000 | 613.517 | 38 | 0 | 0 |
| long_life_holdout | valid | 40 | 16 | 1001.000 | 1052.250 | 1154.500 | 1270.250 | 2237.000 | 1260.900 | 40 | 40 | 0 |
| policy_family_holdout | train | 70 | 36 | 101.000 | 484.000 | 523.500 | 860.250 | 2237.000 | 722.529 | 21 | 16 | 0 |
| policy_family_holdout | valid | 117 | 49 | 280.000 | 513.000 | 787.000 | 931.000 | 1934.000 | 769.624 | 57 | 24 | 0 |

## 5. 使用建议

- 先在 balanced split 上复现实验，再在 long_life_holdout 上检查 LightGBM 与 LSTM 的性能降幅。
- 如果 LSTM 在 long_life_holdout 或后续 late-stage block 过滤下下降更少，才说明它可能具有时序外推优势。
- 不要为了让某个模型获胜而替换划分；每个划分都必须对应一个真实部署问题。
