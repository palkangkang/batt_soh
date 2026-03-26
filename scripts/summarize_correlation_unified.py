from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# =========================
# Config
# =========================
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

LIFE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
POLICY_PATH = REPO_ROOT / "data" / "processed" / "policy_meaning.csv"
CHARGE_PATH = REPO_ROOT / "data" / "processed" / "charge_interval_features.csv"
DISCHARGE_PATH = REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv"

CHARGE_RESULT_DIR = REPO_ROOT / "outputs" / "analysis" / "charge_feature_q_discharge_corr"
DISCHARGE_RESULT_DIR = REPO_ROOT / "outputs" / "analysis" / "discharge_policy_q_discharge_corr"

OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "correlation_unified"
ENCODING = "utf-8-sig"
FIRST_OCCURRENCE_RANGE_COUNT = 1


MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402


def setup_fonts() -> List[str]:
    candidates = ["fonts-noto-cjk"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected = [f for f in candidates if f in installed]
    if not selected:
        selected = ["DejaVu Sans"]
    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    return selected


def parse_range_low(range_label: str) -> float:
    try:
        body = str(range_label).strip()[1:]
        return float(body.split(",")[0])
    except Exception:
        return float("inf")


def sanitize_range_label(range_label: str) -> str:
    return (
        str(range_label)
        .replace("[", "")
        .replace("]", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def build_wide_features(
    path: Path,
    prefix: str,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    df = pd.read_csv(
        path,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"],
    )
    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["delta_ah"] = pd.to_numeric(df["delta_ah"], errors="coerce")
    df["range_count"] = pd.to_numeric(df["range_count"], errors="coerce")
    df = df.dropna(subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]).copy()
    df["cycles"] = df["cycles"].astype(int)
    df["range_count"] = df["range_count"].astype(int)

    df = df.loc[df["range_count"] == FIRST_OCCURRENCE_RANGE_COUNT].copy()
    agg = (
        df.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(delta_ah_sum=("delta_ah", "sum"))
    )
    ranges = sorted(agg["range"].dropna().unique().tolist(), key=parse_range_low)
    agg["feature_name"] = agg["range"].map(lambda r: f"{prefix}_{sanitize_range_label(r)}")
    wide = (
        agg.pivot_table(
            index=["policy", "cell_code", "cycles"],
            columns="feature_name",
            values="delta_ah_sum",
            aggfunc="mean",
        )
        .reset_index()
    )
    cols = [f"{prefix}_{sanitize_range_label(r)}" for r in ranges]
    cols = [c for c in cols if c in wide.columns]
    wide = wide[["policy", "cell_code", "cycles"] + cols]
    label_map = {f"{prefix}_{sanitize_range_label(r)}": str(r) for r in ranges}
    return wide, cols, label_map


def build_unified_dataset() -> Tuple[pd.DataFrame, List[str], List[str], List[str], Dict[str, str]]:
    life = pd.read_csv(
        LIFE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life["cycles"] = life["cycles"].astype(int)

    policy = pd.read_csv(
        POLICY_PATH,
        encoding=ENCODING,
        usecols=["policy", "initial_c_rate", "switch_soc_percent", "post_switch_c_rate"],
    ).drop_duplicates(subset=["policy"], keep="first")
    for c in ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]:
        policy[c] = pd.to_numeric(policy[c], errors="coerce")

    charge_wide, charge_cols, charge_label_map = build_wide_features(CHARGE_PATH, "charge_delta_ah")
    discharge_wide, discharge_cols, discharge_label_map = build_wide_features(DISCHARGE_PATH, "discharge_delta_ah")

    data = (
        life.merge(charge_wide, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(discharge_wide, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(policy, on="policy", how="left", validate="many_to_one")
    )
    policy_cols = ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]
    data = data.dropna(subset=["q_discharge"] + policy_cols).copy()

    display_map = {}
    display_map.update(charge_label_map)
    display_map.update(discharge_label_map)
    return data, charge_cols, discharge_cols, policy_cols, display_map


def fit_combo(data: pd.DataFrame, feature_cols: List[str], target_col: str = "q_discharge") -> dict:
    X_df = data[feature_cols].copy()
    valid_cols = [c for c in feature_cols if X_df[c].notna().any()]
    if not valid_cols:
        raise RuntimeError("No valid features after filtering all-NaN columns.")

    X = X_df[valid_cols].to_numpy(dtype=float)
    y = data[target_col].to_numpy(dtype=float)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lr", LinearRegression()),
        ]
    )
    model.fit(X, y)
    pred = model.predict(X)
    r = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan")
    r2 = float(max(r * r, 0.0))
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "n_samples": len(y),
        "n_features": len(valid_cols),
        "multiple_r": r,
        "r2_in_sample": r2,
        "mae_in_sample": mae,
        "rmse_in_sample": rmse,
    }


def save_comparison_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    order = ["policy_plus_charge", "policy_plus_discharge", "policy_plus_charge_plus_discharge"]
    labels = {
        "policy_plus_charge": "policy+充电",
        "policy_plus_discharge": "policy+放电",
        "policy_plus_charge_plus_discharge": "policy+充电+放电",
    }
    df = summary_df.set_index("combo_name").loc[order].reset_index()
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - w / 2, df["multiple_r"], width=w, label="Multiple R", color="#0ea5e9")
    ax.bar(x + w / 2, df["r2_in_sample"], width=w, label="R²", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in df["combo_name"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("不同特征组合与放电容量关联强度对比（统一口径）")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def try_load_charge_summary() -> pd.DataFrame:
    p = CHARGE_RESULT_DIR / "correlation_by_range.csv"
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8")
        cols = [c for c in ["variable_display", "spearman_rho", "pearson_r", "pearson_policy_demean", "n_samples"] if c in df.columns]
        return df[cols].copy()
    return pd.DataFrame()


def try_load_discharge_summary() -> pd.DataFrame:
    p = DISCHARGE_RESULT_DIR / "univariate_correlation.csv"
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8")
        cols = [c for c in ["variable_display", "variable_type", "spearman_rho", "pearson_r", "n_samples"] if c in df.columns]
        return df[cols].copy()
    return pd.DataFrame()


def render_unified_report(
    report_path: Path,
    fonts: List[str],
    data: pd.DataFrame,
    charge_cols: List[str],
    discharge_cols: List[str],
    policy_cols: List[str],
    summary_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    charge_summary_df: pd.DataFrame,
    discharge_summary_df: pd.DataFrame,
) -> None:
    name_map = {
        "policy_plus_charge": "policy+充电特征",
        "policy_plus_discharge": "policy+放电特征",
        "policy_plus_charge_plus_discharge": "policy+充电+放电特征",
    }
    lines: list[str] = []
    lines.append("# 相关性汇总报告（统一口径）")
    lines.append("")
    lines.append("## 1. 分析范围")
    lines.append("- 汇总对象：")
    lines.append("  - policy + 充电特征")
    lines.append("  - policy + 放电特征")
    lines.append("  - policy + 充电特征 + 放电特征（新增）")
    lines.append("- 统一口径：")
    lines.append("  - 充/放电区间特征均使用第一次出现（`range_count == 1`）")
    lines.append("  - 同一线性相关框架比较（`Multiple R`、`R²`、`MAE`、`RMSE`）")
    lines.append("")
    lines.append("## 2. 数据规模与特征")
    lines.append(f"- 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 中文字体回退链：`{', '.join(fonts)}`")
    lines.append(f"- 样本点（cycle级）：**{len(data):,}**")
    lines.append(f"- 充电特征数：**{len(charge_cols)}**；放电特征数：**{len(discharge_cols)}**；policy参数数：**{len(policy_cols)}**")
    lines.append("")
    lines.append("## 3. 组合关联性对比（统一方法）")
    lines.append("| 组合 | n | 特征数 | Multiple R | R² | MAE | RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {name_map.get(r['combo_name'], r['combo_name'])} | {int(r['n_samples'])} | {int(r['n_features'])} | "
            f"{r['multiple_r']:.4f} | {r['r2_in_sample']:.4f} | {r['mae_in_sample']:.6f} | {r['rmse_in_sample']:.6f} |"
        )
    lines.append("")
    lines.append("### 3.1 与新增组合（policy+充电+放电）的增益")
    lines.append("| 对比基线 | ΔR² | ΔMultiple R | MAE改善(%) | RMSE改善(%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in uplift_df.iterrows():
        lines.append(
            f"| {name_map.get(r['compare_to'], r['compare_to'])} | {r['delta_r2']:.5f} | {r['delta_multiple_r']:.5f} | "
            f"{r['mae_improve_pct']:.2%} | {r['rmse_improve_pct']:.2%} |"
        )
    lines.append("")
    lines.append("## 4. 迁移内容：policy+充电特征相关性摘要")
    if charge_summary_df.empty:
        lines.append("- 未发现历史充电相关性结果文件。")
    else:
        lines.append("| 区间 | Spearman | Pearson | Policy-demeaned Pearson | n |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in charge_summary_df.iterrows():
            lines.append(
                f"| {r.get('variable_display', '')} | {float(r.get('spearman_rho', np.nan)):.4f} | "
                f"{float(r.get('pearson_r', np.nan)):.4f} | {float(r.get('pearson_policy_demean', np.nan)):.4f} | "
                f"{int(r.get('n_samples', 0))} |"
            )
    lines.append("")
    lines.append("## 5. 迁移内容：policy+放电特征相关性摘要")
    if discharge_summary_df.empty:
        lines.append("- 未发现历史放电相关性结果文件。")
    else:
        lines.append("| 变量 | 类型 | Spearman | Pearson | n |")
        lines.append("|---|---|---:|---:|---:|")
        for _, r in discharge_summary_df.iterrows():
            lines.append(
                f"| {r.get('variable_display', '')} | {r.get('variable_type', '')} | "
                f"{float(r.get('spearman_rho', np.nan)):.4f} | {float(r.get('pearson_r', np.nan)):.4f} | "
                f"{int(r.get('n_samples', 0))} |"
            )
    lines.append("")
    lines.append("## 6. 图表")
    lines.append("![组合关联强度对比](./combo_correlation_comparison.png)")
    lines.append("")
    lines.append("## 7. 结论（客观）")
    best = summary_df.sort_values("r2_in_sample", ascending=False).iloc[0]
    lines.append(f"- 在统一比较口径下，R²最高组合为：**{name_map.get(best['combo_name'], best['combo_name'])}**。")
    lines.append("- 新增组合可用于判断充/放电特征是否存在互补信息。")
    lines.append("- 本报告为相关性汇总，若用于预测泛化结论，建议配合独立验证集评估。")

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fonts = setup_fonts()

    data, charge_cols, discharge_cols, policy_cols, _ = build_unified_dataset()
    if data.empty:
        raise RuntimeError("Unified dataset is empty.")

    combos = {
        "policy_plus_charge": policy_cols + charge_cols,
        "policy_plus_discharge": policy_cols + discharge_cols,
        "policy_plus_charge_plus_discharge": policy_cols + charge_cols + discharge_cols,
    }
    summary_rows = []
    for combo_name, cols in combos.items():
        row = fit_combo(data, cols)
        row["combo_name"] = combo_name
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("combo_name").reset_index(drop=True)

    idx = {r["combo_name"]: r for _, r in summary_df.iterrows()}
    comb = idx["policy_plus_charge_plus_discharge"]
    uplift_rows = []
    for base_name in ["policy_plus_charge", "policy_plus_discharge"]:
        base = idx[base_name]
        uplift_rows.append(
            {
                "compare_to": base_name,
                "delta_r2": comb["r2_in_sample"] - base["r2_in_sample"],
                "delta_multiple_r": comb["multiple_r"] - base["multiple_r"],
                "mae_improve_pct": (base["mae_in_sample"] - comb["mae_in_sample"]) / max(base["mae_in_sample"], 1e-12),
                "rmse_improve_pct": (base["rmse_in_sample"] - comb["rmse_in_sample"]) / max(base["rmse_in_sample"], 1e-12),
            }
        )
    uplift_df = pd.DataFrame(uplift_rows)

    charge_summary_df = try_load_charge_summary()
    discharge_summary_df = try_load_discharge_summary()

    out_summary_csv = OUTPUT_DIR / "combo_correlation_summary.csv"
    out_uplift_csv = OUTPUT_DIR / "combo_correlation_uplift.csv"
    out_plot_png = OUTPUT_DIR / "combo_correlation_comparison.png"
    out_report_md = OUTPUT_DIR / "correlation_summary_unified.md"

    summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")
    uplift_df.to_csv(out_uplift_csv, index=False, encoding="utf-8")
    save_comparison_plot(summary_df, out_plot_png)
    render_unified_report(
        report_path=out_report_md,
        fonts=fonts,
        data=data,
        charge_cols=charge_cols,
        discharge_cols=discharge_cols,
        policy_cols=policy_cols,
        summary_df=summary_df,
        uplift_df=uplift_df,
        charge_summary_df=charge_summary_df,
        discharge_summary_df=discharge_summary_df,
    )

    print(f"Saved: {out_summary_csv}")
    print(f"Saved: {out_uplift_csv}")
    print(f"Saved: {out_plot_png}")
    print(f"Saved: {out_report_md}")
    print(
        f"Rows={len(data)} | features charge/discharge/policy="
        f"{len(charge_cols)}/{len(discharge_cols)}/{len(policy_cols)}"
    )


if __name__ == "__main__":
    main()
