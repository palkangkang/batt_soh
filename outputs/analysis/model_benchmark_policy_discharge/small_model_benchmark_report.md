# 小型模型基准报告（policy + 放电特征）

## 1. 评测设置
- 运行时间：2026-03-19 15:28:48
- Python解释器：`C:\Users\pal\.virtualenvs\colab-OixbOpvz\Scripts\python.exe`
- 字体回退：`DejaVu Sans`
- 训练/验证样本：**98,451 / 41,831**
- 特征维度（剔除全空前/后）：**63 / 54**
- 剔除训练集全空特征列：**9**
- 数据口径：`q_discharge<=1.5`、`range_count==1`、不使用 `cycles`

## 2. 模型与来源
- 新训练模型：`linear_regression`、`ridge_alpha_1`、`elastic_net`、`decision_tree`、`extra_trees`
- 复用已有结果：`random_forest`、`xgboost`

## 3. 指标对比（按验证集 R2 排序）
| 排名 | 模型 | 来源 | valid_R2 | valid_RMSE | valid_MAE | train_R2 |
|---:|---|---|---:|---:|---:|---:|
| 1 | xgboost | existing_result | 0.877722 | 0.018436 | 0.013580 | 0.981723 |
| 2 | random_forest | existing_result | 0.861875 | 0.019594 | 0.014514 | 0.976387 |
| 3 | extra_trees | fresh_train | 0.831886 | 0.021617 | 0.016086 | 0.975261 |
| 4 | decision_tree | fresh_train | 0.816410 | 0.022590 | 0.016300 | 0.946712 |
| 5 | elastic_net | fresh_train | -543.619867 | 1.230365 | 0.031456 | 0.740755 |
| 6 | linear_regression | fresh_train | -576.804637 | 1.267295 | 0.031784 | 0.741853 |
| 7 | ridge_alpha_1 | fresh_train | -786.861620 | 1.479832 | 0.033267 | 0.741096 |

## 4. 可视化
![benchmark_metrics](./small_model_benchmark_metrics.png)