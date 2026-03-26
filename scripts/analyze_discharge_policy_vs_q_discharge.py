from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Config (edit here first)
# =========================
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DISCHARGE_FEATURE_PATH = REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv"
LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
POLICY_MEANING_PATH = REPO_ROOT / "data" / "processed" / "policy_meaning.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "discharge_policy_q_discharge_corr"

ENCODING = "utf-8-sig"
FIRST_OCCURRENCE_RANGE_COUNT = 1
RANDOM_SEED = 20260317
LOW_COVERAGE_WARN_THRESHOLD = 0.10


MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402


def parse_range_low(range_label: str) -> float:
    # [3.6,3.5) -> 3.6
    try:
        body = range_label.strip()[1:]
        return float(body.split(",")[0])
    except Exception:
        return float("inf")


def sanitize_range_label(range_label: str) -> str:
    # [3.6,3.5) -> 3p6_3p5
    return (
        range_label.replace("[", "")
        .replace("]", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


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


def safe_corr(method, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    try:
        r, p = method(x, y)
        if np.isfinite(r) and np.isfinite(p):
            return float(r), float(p)
    except Exception:
        pass
    return float("nan"), float("nan")


def build_dataset() -> tuple[pd.DataFrame, List[str], List[str], Dict[str, str]]:
    life = pd.read_csv(
        LIFE_PERFORMANCE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    discharge = pd.read_csv(
        DISCHARGE_FEATURE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"],
    )
    policy = pd.read_csv(
        POLICY_MEANING_PATH,
        encoding=ENCODING,
        usecols=["policy", "initial_c_rate", "switch_soc_percent", "post_switch_c_rate"],
    )

    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    discharge["cycles"] = pd.to_numeric(discharge["cycles"], errors="coerce")
    discharge["delta_ah"] = pd.to_numeric(discharge["delta_ah"], errors="coerce")
    discharge["range_count"] = pd.to_numeric(discharge["range_count"], errors="coerce")
    policy["initial_c_rate"] = pd.to_numeric(policy["initial_c_rate"], errors="coerce")
    policy["switch_soc_percent"] = pd.to_numeric(policy["switch_soc_percent"], errors="coerce")
    policy["post_switch_c_rate"] = pd.to_numeric(policy["post_switch_c_rate"], errors="coerce")

    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    discharge = discharge.dropna(
        subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]
    ).copy()
    policy = policy.dropna(subset=["policy"]).drop_duplicates(subset=["policy"], keep="first").copy()

    life["cycles"] = life["cycles"].astype(int)
    discharge["cycles"] = discharge["cycles"].astype(int)
    discharge["range_count"] = discharge["range_count"].astype(int)

    discharge = discharge.loc[discharge["range_count"] == FIRST_OCCURRENCE_RANGE_COUNT].copy()
    discharge_agg = (
        discharge.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(delta_ah_first=("delta_ah", "sum"))
    )

    range_order = sorted(discharge_agg["range"].dropna().unique().tolist(), key=parse_range_low, reverse=True)
    discharge_agg["feature_name"] = discharge_agg["range"].map(
        lambda r: f"discharge_delta_ah_{sanitize_range_label(str(r))}"
    )
    discharge_wide = (
        discharge_agg.pivot_table(
            index=["policy", "cell_code", "cycles"],
            columns="feature_name",
            values="delta_ah_first",
            aggfunc="mean",
        )
        .reset_index()
    )
    discharge_feature_cols = [f"discharge_delta_ah_{sanitize_range_label(str(r))}" for r in range_order]
    discharge_feature_cols = [c for c in discharge_feature_cols if c in discharge_wide.columns]
    discharge_wide = discharge_wide[["policy", "cell_code", "cycles"] + discharge_feature_cols]
    feature_label_map = {
        f"discharge_delta_ah_{sanitize_range_label(str(r))}": str(r)
        for r in range_order
    }

    data = (
        life.merge(discharge_wide, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(policy, on="policy", how="left", validate="many_to_one")
        .copy()
    )
    policy_cols = ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]
    return data, discharge_feature_cols, policy_cols, feature_label_map


def calc_univariate_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    policy_cols: List[str],
    feature_label_map: Dict[str, str],
    target_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    var_order = policy_cols + feature_cols
    for var in var_order:
        var_display = feature_label_map.get(var, var)
        part = df[[var, target_col]].dropna()
        x = part[var].to_numpy(dtype=float)
        y = part[target_col].to_numpy(dtype=float)
        pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
        rows.append(
            {
                "variable": var,
                "variable_display": var_display,
                "variable_type": "policy_param" if var in policy_cols else "discharge_feature",
                "n_samples": len(part),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman_rho"].abs()
    out = out.sort_values(["variable_type", "abs_spearman"], ascending=[True, False]).reset_index(drop=True)
    return out


def calc_feature_coverage(
    df: pd.DataFrame,
    discharge_feature_cols: List[str],
    policy_cols: List[str],
    feature_label_map: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    total = len(df)
    for v in policy_cols + discharge_feature_cols:
        v_display = feature_label_map.get(v, v)
        n = int(df[v].notna().sum())
        rows.append(
            {
                "variable": v,
                "variable_display": v_display,
                "variable_type": "policy_param" if v in policy_cols else "discharge_feature",
                "non_null_samples": n,
                "coverage_ratio": n / max(total, 1),
            }
        )
    out = pd.DataFrame(rows).sort_values(["variable_type", "coverage_ratio"], ascending=[True, False]).reset_index(drop=True)
    return out


def calc_partial_corr_discharge_given_policy(
    df: pd.DataFrame,
    discharge_feature_cols: List[str],
    policy_cols: List[str],
    feature_label_map: Dict[str, str],
    target_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for var in discharge_feature_cols:
        var_display = feature_label_map.get(var, var)
        sub = df[[var, target_col] + policy_cols].dropna()
        if len(sub) < 30:
            rows.append(
                {
                    "variable": var,
                    "variable_display": var_display,
                    "n_samples": len(sub),
                    "partial_pearson_r_given_policy": float("nan"),
                    "partial_pearson_p_given_policy": float("nan"),
                }
            )
            continue

        x = sub[var].to_numpy(dtype=float)
        y = sub[target_col].to_numpy(dtype=float)
        z = sub[policy_cols].to_numpy(dtype=float)

        reg_x = LinearRegression().fit(z, x)
        reg_y = LinearRegression().fit(z, y)
        rx = x - reg_x.predict(z)
        ry = y - reg_y.predict(z)
        r, p = safe_corr(stats.pearsonr, rx, ry)
        rows.append(
            {
                "variable": var,
                "variable_display": var_display,
                "n_samples": len(sub),
                "partial_pearson_r_given_policy": r,
                "partial_pearson_p_given_policy": p,
            }
        )

    out = pd.DataFrame(rows)
    out["abs_partial_r"] = out["partial_pearson_r_given_policy"].abs()
    out = out.sort_values("abs_partial_r", ascending=False).reset_index(drop=True)
    return out


def model_r2_summary(
    df: pd.DataFrame,
    discharge_feature_cols: List[str],
    policy_cols: List[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Use a stable, large-sample base set: y and policy params available.
    # Discharge features are median-imputed inside the pipeline.
    base = df[[target_col] + policy_cols + discharge_feature_cols].copy()
    base = base.dropna(subset=[target_col] + policy_cols).copy()
    if base.empty:
        raise RuntimeError("No rows available after requiring y + policy parameters.")

    X_policy = base[policy_cols].to_numpy(dtype=float)
    X_discharge = base[discharge_feature_cols].to_numpy(dtype=float)
    X_combined = base[discharge_feature_cols + policy_cols].to_numpy(dtype=float)
    y = base[target_col].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lr", LinearRegression()),
        ]
    )

    rows = []
    pred_cols = {}
    for name, X in [
        ("policy_only", X_policy),
        ("discharge_only", X_discharge),
        ("combined_discharge_plus_policy", X_combined),
    ]:
        model.fit(X, y)
        pred = model.predict(X)
        r2 = float(stats.pearsonr(y, pred).statistic ** 2) if len(y) > 2 else float("nan")
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        mae = float(np.mean(np.abs(y - pred)))
        rows.append(
            {
                "model_name": name,
                "n_samples_complete_case": len(base),
                "n_features": X.shape[1],
                "r2_in_sample": r2,
                "multiple_r": math.sqrt(max(r2, 0.0)),
                "mae_in_sample": mae,
                "rmse_in_sample": rmse,
            }
        )
        pred_cols[name] = pred

    summary = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    uplift = []
    idx = {r["model_name"]: r for _, r in summary.iterrows()}
    comb = idx["combined_discharge_plus_policy"]
    for base in ["policy_only", "discharge_only"]:
        b = idx[base]
        uplift.append(
            {
                "compare_to": base,
                "delta_r2": comb["r2_in_sample"] - b["r2_in_sample"],
                "delta_multiple_r": comb["multiple_r"] - b["multiple_r"],
                "mae_improve_pct": (b["mae_in_sample"] - comb["mae_in_sample"]) / max(b["mae_in_sample"], 1e-12),
                "rmse_improve_pct": (b["rmse_in_sample"] - comb["rmse_in_sample"]) / max(b["rmse_in_sample"], 1e-12),
            }
        )
    uplift_df = pd.DataFrame(uplift)
    return summary, uplift_df


def standardized_coefficients(
    df: pd.DataFrame,
    features: List[str],
    feature_label_map: Dict[str, str],
    target_col: str,
) -> pd.DataFrame:
    sub = df[features + [target_col]].copy()
    sub = sub.dropna(subset=[target_col]).copy()
    if sub.empty:
        return pd.DataFrame()

    X_raw = sub[features].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float).reshape(-1, 1)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_raw)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    ys = scaler_y.fit_transform(y).ravel()
    lr = LinearRegression().fit(Xs, ys)
    coef = lr.coef_
    out = pd.DataFrame({"variable": features, "std_coef": coef})
    out["variable_display"] = out["variable"].map(lambda x: feature_label_map.get(x, x))
    out["abs_std_coef"] = out["std_coef"].abs()
    out = out.sort_values("abs_std_coef", ascending=False).reset_index(drop=True)
    return out


def save_plots(
    uni_df: pd.DataFrame,
    partial_df: pd.DataFrame,
    model_df: pd.DataFrame,
    out_uni_png: Path,
    out_partial_png: Path,
    out_model_png: Path,
) -> None:
    # 1) Univariate Spearman bar
    top = uni_df.sort_values("abs_spearman", ascending=False).head(12).copy()
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    colors = ["#0ea5e9" if t == "policy_param" else "#22c55e" for t in top["variable_type"]]
    ax.bar(np.arange(len(top)), top["spearman_rho"], color=colors)
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top["variable_display"], rotation=30, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title("单变量 Spearman 相关性（Top 12）")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_uni_png, format="png")
    plt.close(fig)

    # 2) Partial corr bar
    top_p = partial_df.sort_values("abs_partial_r", ascending=False).head(10).copy()
    fig2, ax2 = plt.subplots(figsize=(11.4, 5.2))
    ax2.bar(np.arange(len(top_p)), top_p["partial_pearson_r_given_policy"], color="#f59e0b")
    ax2.axhline(0, color="#64748b", linewidth=1)
    ax2.set_xticks(np.arange(len(top_p)))
    ax2.set_xticklabels(top_p["variable_display"], rotation=30, ha="right")
    ax2.set_ylabel("Partial Pearson r")
    ax2.set_title("控制 policy 三元参数后的偏相关（Top 10）")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_partial_png, format="png")
    plt.close(fig2)

    # 3) Model comparison
    order = ["policy_only", "discharge_only", "combined_discharge_plus_policy"]
    mdf = model_df.set_index("model_name").loc[order].reset_index()
    labels = ["仅policy", "仅放电区间", "放电区间+policy"]
    x = np.arange(len(labels))
    w = 0.35
    fig3, ax3 = plt.subplots(figsize=(9.2, 4.8))
    ax3.bar(x - w / 2, mdf["multiple_r"], width=w, color="#0ea5e9", label="Multiple R")
    ax3.bar(x + w / 2, mdf["r2_in_sample"], width=w, color="#22c55e", label="R²")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("组合相关性强度对比（线性模型）")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="best")
    fig3.tight_layout()
    fig3.savefig(out_model_png, format="png")
    plt.close(fig3)


def render_report(
    report_path: Path,
    font_list: List[str],
    data_df: pd.DataFrame,
    discharge_feature_cols: List[str],
    policy_cols: List[str],
    uni_df: pd.DataFrame,
    partial_df: pd.DataFrame,
    model_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# 放电区间特征 + Policy 三元参数与放电容量相关性分析")
    lines.append("")
    lines.append("## 1. 分析目标")
    lines.append("- 评估“放电时不同电压区间的容量差异特征 + policy 三元参数”与 `q_discharge` 的相关性。")
    lines.append("- 放电区间特征口径：仅使用首次出现（`range_count == 1`）。")
    lines.append("")
    lines.append("## 2. 数据口径与规模")
    lines.append(f"- 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 中文字体回退链：`{', '.join(font_list)}`")
    lines.append(f"- 合并后样本点（cycle 级）：**{len(data_df):,}**")
    lines.append(f"- 放电区间特征数：**{len(discharge_feature_cols)}**")
    lines.append(f"- policy 参数数：**{len(policy_cols)}**")
    low_cov = coverage_df.loc[
        (coverage_df["variable_type"] == "discharge_feature")
        & (coverage_df["coverage_ratio"] < LOW_COVERAGE_WARN_THRESHOLD)
    ]
    if not low_cov.empty:
        low_names = ", ".join([f"{r['variable_display']}({r['coverage_ratio']:.2%})" for _, r in low_cov.iterrows()])
        lines.append(f"- 低覆盖率放电区间特征（< {LOW_COVERAGE_WARN_THRESHOLD:.0%}）：{low_names}。")
    lines.append("")

    lines.append("## 3. 单变量相关（与 q_discharge）")
    lines.append("| 变量 | 类型 | n | Pearson | Spearman |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in uni_df.iterrows():
        lines.append(
            f"| {r['variable_display']} | {r['variable_type']} | {int(r['n_samples'])} | "
            f"{r['pearson_r']:.4f} | {r['spearman_rho']:.4f} |"
        )
    lines.append("")

    lines.append("## 4. 控制 policy 三元参数后的放电特征偏相关")
    lines.append("| 变量 | n | Partial Pearson r | p-value |")
    lines.append("|---|---:|---:|---:|")
    for _, r in partial_df.iterrows():
        lines.append(
            f"| {r['variable_display']} | {int(r['n_samples'])} | "
            f"{r['partial_pearson_r_given_policy']:.4f} | {r['partial_pearson_p_given_policy']:.3e} |"
        )
    lines.append("")

    lines.append("## 5. 组合相关性（线性模型 Multiple R / R²）")
    lines.append("| 模型 | n | 特征数 | Multiple R | R² | MAE | RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in model_df.iterrows():
        lines.append(
            f"| {r['model_name']} | {int(r['n_samples_complete_case'])} | {int(r['n_features'])} | "
            f"{r['multiple_r']:.4f} | {r['r2_in_sample']:.4f} | {r['mae_in_sample']:.6f} | {r['rmse_in_sample']:.6f} |"
        )
    lines.append("")

    lines.append("### 5.1 组合模型增益")
    lines.append("| 对比基线 | ΔR² | ΔMultiple R | MAE改善(%) | RMSE改善(%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in uplift_df.iterrows():
        lines.append(
            f"| {r['compare_to']} | {r['delta_r2']:.5f} | {r['delta_multiple_r']:.5f} | "
            f"{r['mae_improve_pct']:.2%} | {r['rmse_improve_pct']:.2%} |"
        )
    lines.append("")

    if not coef_df.empty:
        lines.append("## 6. 组合模型标准化系数（绝对值排序）")
        lines.append("| 变量 | Std Coef |")
        lines.append("|---|---:|")
        for _, r in coef_df.head(12).iterrows():
            lines.append(f"| {r['variable_display']} | {r['std_coef']:.4f} |")
        lines.append("")

    lines.append("## 7. 图表")
    lines.append("![单变量相关 Top12](./univariate_spearman_top12.png)")
    lines.append("")
    lines.append("![偏相关 Top10](./partial_corr_top10.png)")
    lines.append("")
    lines.append("![组合相关性对比](./multiple_correlation_comparison.png)")
    lines.append("")

    lines.append("## 8. 结果解读（客观）")
    best_model = model_df.sort_values("r2_in_sample", ascending=False).iloc[0]
    lines.append(
        f"- 在当前口径下，组合模型 `combined_discharge_plus_policy` 的相关性强度为 "
        f"`Multiple R={model_df.loc[model_df['model_name']=='combined_discharge_plus_policy','multiple_r'].iloc[0]:.4f}`，"
        f"`R²={model_df.loc[model_df['model_name']=='combined_discharge_plus_policy','r2_in_sample'].iloc[0]:.4f}`。"
    )
    lines.append(f"- 本次三组中 R² 最优模型为：`{best_model['model_name']}`。")
    lines.append("- 偏相关用于判断“在控制 policy 三元参数后”放电区间特征的独立线性关联强度。")
    if not low_cov.empty:
        lines.append("- 低覆盖率特征的相关系数不稳定，解读时应降低权重，优先参考高覆盖区间。")
    lines.append("- 注意：本报告主要是相关性与解释性分析，不等同于严格泛化性能评估。")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font_list = setup_fonts()
    data, discharge_feature_cols, policy_cols, feature_label_map = build_dataset()
    if data.empty:
        raise RuntimeError("Merged dataset is empty.")

    target_col = "q_discharge"
    uni_df = calc_univariate_correlations(
        data, discharge_feature_cols, policy_cols, feature_label_map, target_col
    )
    coverage_df = calc_feature_coverage(data, discharge_feature_cols, policy_cols, feature_label_map)
    partial_df = calc_partial_corr_discharge_given_policy(
        data, discharge_feature_cols, policy_cols, feature_label_map, target_col
    )
    model_df, uplift_df = model_r2_summary(data, discharge_feature_cols, policy_cols, target_col)
    coef_df = standardized_coefficients(
        data, discharge_feature_cols + policy_cols, feature_label_map, target_col
    )

    out_uni_csv = OUTPUT_DIR / "univariate_correlation.csv"
    out_partial_csv = OUTPUT_DIR / "discharge_partial_corr_given_policy.csv"
    out_model_csv = OUTPUT_DIR / "multiple_correlation_model_summary.csv"
    out_uplift_csv = OUTPUT_DIR / "combined_model_uplift.csv"
    out_coef_csv = OUTPUT_DIR / "combined_model_standardized_coefficients.csv"
    out_cov_csv = OUTPUT_DIR / "feature_coverage_summary.csv"
    out_uni_png = OUTPUT_DIR / "univariate_spearman_top12.png"
    out_partial_png = OUTPUT_DIR / "partial_corr_top10.png"
    out_model_png = OUTPUT_DIR / "multiple_correlation_comparison.png"
    out_report_md = OUTPUT_DIR / "report_discharge_policy_vs_q_discharge.md"

    uni_df.to_csv(out_uni_csv, index=False, encoding="utf-8")
    partial_df.to_csv(out_partial_csv, index=False, encoding="utf-8")
    model_df.to_csv(out_model_csv, index=False, encoding="utf-8")
    uplift_df.to_csv(out_uplift_csv, index=False, encoding="utf-8")
    coef_df.to_csv(out_coef_csv, index=False, encoding="utf-8")
    coverage_df.to_csv(out_cov_csv, index=False, encoding="utf-8")

    save_plots(
        uni_df=uni_df,
        partial_df=partial_df,
        model_df=model_df,
        out_uni_png=out_uni_png,
        out_partial_png=out_partial_png,
        out_model_png=out_model_png,
    )
    render_report(
        report_path=out_report_md,
        font_list=font_list,
        data_df=data,
        discharge_feature_cols=discharge_feature_cols,
        policy_cols=policy_cols,
        uni_df=uni_df,
        partial_df=partial_df,
        model_df=model_df,
        uplift_df=uplift_df,
        coef_df=coef_df,
        coverage_df=coverage_df,
    )

    print(f"Saved: {out_uni_csv}")
    print(f"Saved: {out_partial_csv}")
    print(f"Saved: {out_model_csv}")
    print(f"Saved: {out_uplift_csv}")
    print(f"Saved: {out_coef_csv}")
    print(f"Saved: {out_cov_csv}")
    print(f"Saved: {out_uni_png}")
    print(f"Saved: {out_partial_png}")
    print(f"Saved: {out_model_png}")
    print(f"Saved: {out_report_md}")
    print(f"Rows merged: {len(data)} | discharge_features={len(discharge_feature_cols)} | policy_features={len(policy_cols)}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
