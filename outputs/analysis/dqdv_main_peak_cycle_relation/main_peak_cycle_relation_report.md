# dQ/dV 主峰与循环关系报告

## 1) 数据覆盖情况

- 输入总行数: **140577**
- 有效行数 (`is_valid_curve=True`): **140572**
- 有效率: **99.9964%**
- 组合数 (`policy+cell_code`): **187**
- 每组周期数 (min/p50/p90/max): **101 / 689.0 / 1155.8 / 2237**

## 2) 校验检查

- expected_long_rows = combo_count x feature_count = **1683**, actual = **1683**
- spearman_rho 越界数量: **0**
- spearman_pvalue 越界数量: **0**
- 抽样复算 spot_check 通过: **True** (policy=5_4C-40PER_3_6C, cell=464898, feature=main_peak_voltage_v)
- 抽样复算绝对误差 (delta/slope/rho): **0.000e+00 / 0.000e+00 / 0.000e+00**

## 3) 各特征上升/下降最快的组合

### 特征: `main_peak_voltage_v`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| VARCHARGE_4C_100CYCLES | 737324 | 0.0018 | -0.2248 | 0.0180 | 0.5831 | 240 |
| VARCHARGE_2C_100CYCLES | 737301 | 0.0010 | 0.1108 | 0.0380 | 1.2254 | 813 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737297 | -0.0005 | -0.4562 | -0.0020 | -0.0638 | 2237 |
| 5C_67PER_4C_NEWSTRUCTURE | 737334 | -0.0008 | -0.4438 | -0.0290 | -0.9274 | 1934 |
| 3_6C-80PER_3_6C | 460486 | -0.0008 | -0.3604 | 0.0060 | 0.1929 | 1177 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 2C_10PER_6C | 460605 | -0.0456 | -0.9668 | -0.0880 | -2.8433 | 170 |
| VARCHARGE_8C_100CYCLES | 737290 | -0.0281 | -0.9710 | -0.0780 | -2.4880 | 280 |
| 1C_4PER_6C | 460518 | -0.0251 | -0.9747 | -0.0790 | -2.5296 | 326 |
| 2C_7PER_5_5C | 460673 | -0.0213 | -0.9794 | -0.0760 | -2.4390 | 361 |
| 3_6C_9PER_5C | 720452 | -0.0195 | -0.9775 | -0.0840 | -2.6923 | 414 |

### 特征: `main_peak_width_v`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 2C_10PER_6C | 460605 | 0.0144 | 0.8914 | 0.0250 | 7.4405 | 170 |
| 4_8C_80PER_4_8C_SLOWCYCLE | 737351 | 0.0031 | 0.3032 | 0.0060 | 1.8405 | 101 |
| 7C-40PER_3_6C | 460642 | 0.0027 | 0.7645 | 0.0220 | 6.7901 | 623 |
| 8C-35PER_3_6C | 460656 | 0.0026 | 0.7864 | 0.0090 | 2.7108 | 597 |
| 7C-40PER_3C | 460601 | 0.0025 | 0.7734 | 0.0240 | 7.4074 | 646 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 4_65C_44PER_5C | 737362 | -0.0025 | -0.8471 | -0.0170 | -5.1205 | 493 |
| 5_6C_26PER_4_5C | 737363 | -0.0025 | -0.7509 | -0.0150 | -4.5455 | 429 |
| 3_6C_30PER_6C | 734853 | -0.0022 | -0.7725 | -0.0070 | -2.1538 | 461 |
| 4_65C_69PER_6C | 737384 | -0.0022 | -0.7413 | -0.0140 | -4.2424 | 483 |
| 3_6C_9PER_5C | 720452 | -0.0019 | -0.5497 | -0.0110 | -3.3639 | 414 |

### 特征: `main_peak_height_dqdv`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| VARCHARGE_2C_100CYCLES | 737301 | -0.0090 | -0.2912 | 0.1180 | 3.3137 | 813 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737297 | -0.0233 | -0.9402 | -0.6250 | -16.0421 | 2237 |
| 3_6C-80PER_3_6C | 460623 | -0.0235 | -0.8461 | -0.1470 | -4.0044 | 1175 |
| 3_6C-80PER_3_6C | 460486 | -0.0245 | -0.8702 | -0.2070 | -5.6946 | 1177 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737299 | -0.0252 | -0.9519 | -0.7160 | -19.8834 | 2189 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 2C_10PER_6C | 460605 | -0.7276 | -0.9859 | -1.2890 | -36.9871 | 170 |
| VARCHARGE_8C_100CYCLES | 737290 | -0.5296 | -0.9953 | -1.5940 | -42.6317 | 280 |
| 1C_4PER_6C | 460518 | -0.4963 | -0.9961 | -1.6960 | -45.7513 | 326 |
| 2C_7PER_5_5C | 460673 | -0.4446 | -0.9903 | -1.6140 | -44.8209 | 361 |
| 3_6C_9PER_5C | 737391 | -0.4182 | -0.9954 | -1.9170 | -49.4582 | 409 |

### 特征: `main_peak_area`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| VARCHARGE_2C_100CYCLES | 737301 | -0.0029 | -0.7583 | 0.0280 | 2.4648 | 813 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737297 | -0.0063 | -0.9902 | -0.1630 | -13.5046 | 2237 |
| 3_6C-80PER_3_6C | 460623 | -0.0074 | -0.9346 | -0.0700 | -5.9072 | 1175 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737299 | -0.0077 | -0.9864 | -0.2250 | -19.3299 | 2189 |
| 3_6C-80PER_3_6C | 460486 | -0.0078 | -0.9403 | -0.0730 | -6.2181 | 1177 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 2C_10PER_6C | 460605 | -0.1971 | -0.9939 | -0.3540 | -31.1072 | 170 |
| VARCHARGE_8C_100CYCLES | 737290 | -0.1697 | -0.9950 | -0.4880 | -41.1467 | 280 |
| 1C_4PER_6C | 460518 | -0.1589 | -0.9976 | -0.5300 | -44.8393 | 326 |
| 2C_7PER_5_5C | 460673 | -0.1409 | -0.9927 | -0.5140 | -44.2341 | 361 |
| 3_6C_9PER_5C | 737391 | -0.1279 | -0.9983 | -0.5810 | -48.4975 | 409 |

### 特征: `main_peak_prominence`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| VARCHARGE_2C_100CYCLES | 737301 | -0.0090 | -0.2912 | 0.1180 | 3.3137 | 813 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737297 | -0.0233 | -0.9402 | -0.6250 | -16.0421 | 2237 |
| 3_6C-80PER_3_6C | 460623 | -0.0235 | -0.8461 | -0.1470 | -4.0044 | 1175 |
| 3_6C-80PER_3_6C | 460486 | -0.0245 | -0.8702 | -0.2070 | -5.6946 | 1177 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737299 | -0.0252 | -0.9519 | -0.7160 | -19.8834 | 2189 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 2C_10PER_6C | 460605 | -0.7276 | -0.9859 | -1.2890 | -36.9871 | 170 |
| VARCHARGE_8C_100CYCLES | 737290 | -0.5296 | -0.9953 | -1.5940 | -42.6317 | 280 |
| 1C_4PER_6C | 460518 | -0.4963 | -0.9961 | -1.6960 | -45.7513 | 326 |
| 2C_7PER_5_5C | 460673 | -0.4446 | -0.9903 | -1.6140 | -44.8209 | 361 |
| 3_6C_9PER_5C | 737391 | -0.4182 | -0.9954 | -1.9170 | -49.4582 | 409 |

### 特征: `main_peak_skewness`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 2C_10PER_6C | 460605 | 0.1022 | 0.9514 | 0.1770 | 58.2237 | 170 |
| VARCHARGE_8C_100CYCLES | 737290 | 0.0485 | 0.9790 | 0.1320 | 41.7722 | 280 |
| 2C_7PER_5_5C | 460673 | 0.0432 | 0.9858 | 0.1580 | 49.8423 | 361 |
| 1C_4PER_6C | 460518 | 0.0404 | 0.9719 | 0.1420 | 44.0994 | 326 |
| 3_6C_9PER_5C | 720452 | 0.0340 | 0.9900 | 0.1270 | 40.1899 | 414 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| VARCHARGE_4C_100CYCLES | 737324 | -0.0019 | 0.1731 | -0.0100 | -3.4364 | 240 |
| VARCHARGE_2C_100CYCLES | 737301 | -0.0004 | -0.0478 | -0.0180 | -5.6075 | 813 |
| 3_6C-80PER_3_6C | 460623 | -0.0000 | 0.0686 | 0.0200 | 6.1920 | 1175 |
| 4_8C_80PER_4_8C_NEWSTRUCTURE | 737297 | 0.0002 | 0.2135 | 0.0010 | 0.3247 | 2237 |
| 3_6C-80PER_3_6C | 460486 | 0.0002 | 0.1448 | 0.0060 | 1.8809 | 1177 |

### 特征: `main_peak_temp_max_c`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 4_8C_80PER_4_8C_SLOWCYCLE | 737351 | 87.4449 | 0.0247 | 670.0000 | 248.1481 | 101 |
| VARCHARGE_4C_100CYCLES | 737324 | 1.4697 | 0.3768 | 5.6400 | 18.6941 | 240 |
| 2C_10PER_6C | 460605 | 0.6465 | 0.3986 | 1.6900 | 4.7889 | 170 |
| 5_2C_50PER_4_25C | 720603 | 0.5567 | 0.6964 | 0.9700 | 2.6751 | 439 |
| 5_4C-80PER_5_4C | 463871 | 0.4715 | 0.7555 | 1.2400 | 3.5217 | 557 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 3_6C_9PER_5C | 460451 | -0.8832 | -0.3314 | -7.5300 | -20.2965 | 595 |
| 5_6C_26PER_4_5C | 737370 | -0.8279 | -0.8079 | -4.9100 | -13.3026 | 476 |
| 5_6C_36PER_4_3C_NEWSTRUCTURE | 737213 | -0.6803 | -0.2167 | -5.2900 | -14.2052 | 827 |
| 4_65C_19PER_4_85C | 460603 | -0.6019 | -0.6883 | -2.4700 | -7.0875 | 520 |
| 4_65C_44PER_5C | 737292 | -0.5532 | -0.3226 | -4.6600 | -12.3772 | 517 |

### 特征: `main_peak_temp_min_c`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 4_8C_80PER_4_8C_SLOWCYCLE | 737351 | 6.9203 | 0.1098 | 0.0000 | 0.0000 | 101 |
| VARCHARGE_4C_100CYCLES | 737324 | 1.7170 | 0.3990 | 6.9500 | 26.1082 | 240 |
| 3_7C_31PER_5_9C_NEWSTRUCTURE | 737234 | 0.5848 | 0.9793 | 2.1000 | 6.5319 | 540 |
| 5_4C-80PER_5_4C | 464002 | 0.4925 | 0.6706 | 1.1600 | 3.8133 | 532 |
| 5_4C-80PER_5_4C | 463871 | 0.4821 | 0.6098 | 1.1000 | 3.5866 | 557 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| VARCHARGE_8C_100CYCLES | 737290 | -0.6892 | -0.7797 | -2.0000 | -6.0441 | 280 |
| 3_6C_9PER_5C | 720452 | -0.3234 | -0.7581 | -0.9300 | -2.9693 | 414 |
| 3_6C_80PER_3_6C_SLOWCYCLE | 737235 | -0.2712 | -0.0977 | -1.1600 | -3.8196 | 101 |
| 80PER_3_6C | 460623 | -0.2398 | -0.1590 | -9.1500 | -29.9902 | 1060 |
| 80PER_4C | 464977 | -0.2264 | -0.1900 | -0.6500 | -2.2169 | 208 |

### 特征: `main_peak_temp_avg_c`

上升最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 4_8C_80PER_4_8C_SLOWCYCLE | 737351 | 33.7828 | 0.1238 | 269.2500 | 99.7222 | 101 |
| VARCHARGE_4C_100CYCLES | 737324 | 1.6122 | 0.4975 | 6.3900 | 22.7645 | 240 |
| 5_4C-80PER_5_4C | 463871 | 0.4533 | 0.6454 | 1.3000 | 3.9707 | 557 |
| 5_4C-80PER_5_4C | 464002 | 0.4520 | 0.6675 | 1.1600 | 3.6205 | 532 |
| 3_7C_31PER_5_9C_NEWSTRUCTURE | 737229 | 0.4306 | 0.5400 | 4.7400 | 14.2129 | 666 |

下降最快 Top:
| policy | cell_code | slope_per_100_cycles | spearman_rho | delta_abs | delta_pct | n_cycles |
| --- | --- | --- | --- | --- | --- | --- |
| 3_6C_80PER_3_6C_SLOWCYCLE | 737235 | -0.6424 | -0.2216 | -2.6800 | -8.1707 | 101 |
| 3_6C_9PER_5C | 460451 | -0.4896 | -0.3100 | -4.3000 | -12.7559 | 595 |
| 5_6C_26PER_4_5C | 737370 | -0.4385 | -0.6559 | -2.7800 | -8.1573 | 476 |
| 80PER_4C | 464977 | -0.3740 | -0.3840 | -0.6900 | -2.3225 | 208 |
| 5_6C_36PER_4_3C_NEWSTRUCTURE | 737213 | -0.3603 | -0.1836 | -2.5200 | -7.3256 | 827 |

## 4) 各特征 Spearman 分布摘要

| feature | median_rho | q1_rho | q3_rho | iqr_rho |
| --- | --- | --- | --- | --- |
| main_peak_voltage_v | -0.8715 | -0.9663 | -0.7591 | 0.2072 |
| main_peak_width_v | 0.3999 | -0.0132 | 0.6265 | 0.6397 |
| main_peak_height_dqdv | -0.9910 | -0.9966 | -0.9832 | 0.0134 |
| main_peak_area | -0.9979 | -0.9989 | -0.9920 | 0.0070 |
| main_peak_prominence | -0.9910 | -0.9966 | -0.9832 | 0.0134 |
| main_peak_skewness | 0.9207 | 0.7764 | 0.9735 | 0.1970 |
| main_peak_temp_max_c | 0.3377 | 0.0062 | 0.5787 | 0.5725 |
| main_peak_temp_min_c | 0.2836 | -0.0339 | 0.5584 | 0.5923 |
| main_peak_temp_avg_c | 0.2469 | -0.0146 | 0.5124 | 0.5270 |

## 5) 输出文件

- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_main_peak_cycle_relation\main_peak_cycle_relation_long.csv`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_main_peak_cycle_relation\main_peak_cycle_relation_wide.csv`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_main_peak_cycle_relation\main_peak_cycle_relation_report.md`
- `C:\Users\pal\projects\batt_soh\outputs\analysis\dqdv_main_peak_cycle_relation\main_peak_cycle_relation_overview.png`


