from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    build_wide_features,
    choose_merge_keys,
    load_interval_features_first_occurrence,
    load_life_performance_with_alias,
    setup_fonts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize no-policy charge/discharge correlation results into a single report."
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "life_performance.csv",
    )
    parser.add_argument(
        "--charge-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_interval_features.csv",
    )
    parser.add_argument(
        "--discharge-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv",
    )
    parser.add_argument(
        "--charge-result-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "charge_feature_q_discharge_corr_no_policy",
    )
    parser.add_argument(
        "--discharge-result-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "discharge_feature_q_discharge_corr_no_policy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "correlation_no_policy",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--first-occurrence", type=int, default=1)
    return parser.parse_args()


def fit_combo(data: pd.DataFrame, feature_cols: list[str]) -> dict:
    X_df = data[feature_cols].copy()
    valid_cols = [c for c in feature_cols if X_df[c].notna().any()]
    if not valid_cols:
        raise RuntimeError("No valid features after filtering all-NaN columns.")

    X = X_df[valid_cols].to_numpy(dtype=float)
    y = data["q_discharge"].to_numpy(dtype=float)

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


def save_comparison_plot(summary_df: pd.DataFrame, out_path: Path, zh: bool) -> None:
    import matplotlib.pyplot as plt

    order = ["charge_only", "discharge_only", "charge_plus_discharge"]
    labels_zh = {
        "charge_only": "仅充电特征",
        "discharge_only": "仅放电特征",
        "charge_plus_discharge": "充电+放电特征",
    }
    labels_en = {
        "charge_only": "Charge Only",
        "discharge_only": "Discharge Only",
        "charge_plus_discharge": "Charge + Discharge",
    }
    labels = labels_zh if zh else labels_en

    df = summary_df.set_index("combo_name").loc[order].reset_index()
    x = np.arange(len(df))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - w / 2, df["multiple_r"], width=w, label="Multiple R", color="#0ea5e9")
    ax.bar(x + w / 2, df["r2_in_sample"], width=w, label="R²", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in df["combo_name"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("充放电特征组合相关性对比" if zh else "Correlation Strength by Feature Group")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def _try_load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8")
    return pd.DataFrame()


def render_report(
    report_path: Path,
    fonts: list[str],
    data: pd.DataFrame,
    charge_cols: list[str],
    discharge_cols: list[str],
    summary_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
) -> None:
    name_map = {
        "charge_only": "仅充电特征",
        "discharge_only": "仅放电特征",
        "charge_plus_discharge": "充电+放电特征",
    }

    lines: list[str] = []
    lines.append("# 相关性汇总报告（无Policy）")
    lines.append("")
    lines.append("## 1. 分析范围")
    lines.append("- 仅分析：充电电压区间特征、放电电压区间特征，以及两者联合。")
    lines.append("- 明确不纳入：任何 policy 三元参数字段。")
    lines.append("- 区间口径：仅使用 `range_count == 1` 的首次出现区间特征。")
    lines.append("")
    lines.append("## 2. 数据规模与特征")
    lines.append(f"- 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{sys.executable}`")
    lines.append(f"- 中文字体回退链：`{', '.join(fonts)}`")
    lines.append(f"- 样本点（cycle级）：**{len(data):,}**")
    lines.append(f"- 充电特征数：**{len(charge_cols)}**；放电特征数：**{len(discharge_cols)}**")
    lines.append("")
    lines.append("## 3. 组合关联性对比")
    lines.append("| 组合 | n | 特征数 | Multiple R | R² | MAE | RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {name_map.get(r['combo_name'], r['combo_name'])} | {int(r['n_samples'])} | {int(r['n_features'])} | "
            f"{r['multiple_r']:.4f} | {r['r2_in_sample']:.4f} | {r['mae_in_sample']:.6f} | {r['rmse_in_sample']:.6f} |"
        )

    lines.append("")
    lines.append("### 3.1 相对联合模型（充电+放电）的增益")
    lines.append("| 对比基线 | ΔR² | ΔMultiple R | MAE改善(%) | RMSE改善(%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in uplift_df.iterrows():
        lines.append(
            f"| {name_map.get(r['compare_to'], r['compare_to'])} | {r['delta_r2']:.5f} | {r['delta_multiple_r']:.5f} | "
            f"{r['mae_improve_pct']:.2%} | {r['rmse_improve_pct']:.2%} |"
        )

    lines.append("")
    lines.append("## 4. 充电区间相关性摘要")
    if charge_df.empty:
        lines.append("- 未检测到 `correlation_by_range.csv`。")
    else:
        part = charge_df.copy()
        keep = [c for c in ["variable_display", "spearman_rho", "pearson_r", "n_samples"] if c in part.columns]
        part = part[keep].copy()
        lines.append("| 区间 | Spearman | Pearson | n |")
        lines.append("|---|---:|---:|---:|")
        for _, r in part.iterrows():
            lines.append(
                f"| {r.get('variable_display', '')} | {float(r.get('spearman_rho', np.nan)):.4f} | "
                f"{float(r.get('pearson_r', np.nan)):.4f} | {int(r.get('n_samples', 0))} |"
            )

    lines.append("")
    lines.append("## 5. 放电区间相关性摘要")
    if discharge_df.empty:
        lines.append("- 未检测到 `univariate_correlation.csv`。")
    else:
        part = discharge_df.copy()
        keep = [c for c in ["variable_display", "spearman_rho", "pearson_r", "n_samples"] if c in part.columns]
        part = part[keep].copy()
        lines.append("| 区间 | Spearman | Pearson | n |")
        lines.append("|---|---:|---:|---:|")
        for _, r in part.iterrows():
            lines.append(
                f"| {r.get('variable_display', '')} | {float(r.get('spearman_rho', np.nan)):.4f} | "
                f"{float(r.get('pearson_r', np.nan)):.4f} | {int(r.get('n_samples', 0))} |"
            )

    lines.append("")
    lines.append("## 6. 图表")
    lines.append("![组合关联强度对比](./combo_correlation_comparison.png)")
    lines.append("")
    lines.append("## 7. 结论")
    best = summary_df.sort_values("r2_in_sample", ascending=False).iloc[0]
    lines.append(f"- R²最高组合：**{name_map.get(best['combo_name'], best['combo_name'])}**。")
    lines.append("- 该结果仅反映无 policy 条件下的统计关联，不直接等价于泛化预测能力。")

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fonts, has_cjk = setup_fonts(REPO_ROOT)

    life = load_life_performance_with_alias(args.life_path, args.encoding)
    charge_agg = load_interval_features_first_occurrence(args.charge_path, args.first_occurrence, args.encoding)
    discharge_agg = load_interval_features_first_occurrence(args.discharge_path, args.first_occurrence, args.encoding)

    merge_keys = choose_merge_keys(charge_agg.columns, life.columns)
    charge_wide, charge_cols, _ = build_wide_features(
        charge_agg,
        "charge_delta_ah",
        reverse=False,
        key_cols=merge_keys,
    )
    merge_keys_2 = choose_merge_keys(discharge_agg.columns, life.columns)
    discharge_wide, discharge_cols, _ = build_wide_features(
        discharge_agg,
        "discharge_delta_ah",
        reverse=True,
        key_cols=merge_keys_2,
    )

    data = (
        life.merge(charge_wide, on=merge_keys, how="inner")
        .merge(discharge_wide, on=choose_merge_keys(discharge_wide.columns, life.columns), how="inner")
        .copy()
    )
    if data.empty:
        raise RuntimeError("Unified dataset is empty after joining charge/discharge/life tables.")

    combos = {
        "charge_only": charge_cols,
        "discharge_only": discharge_cols,
        "charge_plus_discharge": charge_cols + discharge_cols,
    }
    summary_rows = []
    for combo_name, cols in combos.items():
        row = fit_combo(data, cols)
        row["combo_name"] = combo_name
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    idx = {r["combo_name"]: r for _, r in summary_df.iterrows()}
    comb = idx["charge_plus_discharge"]
    uplift_df = pd.DataFrame(
        [
            {
                "compare_to": "charge_only",
                "delta_r2": comb["r2_in_sample"] - idx["charge_only"]["r2_in_sample"],
                "delta_multiple_r": comb["multiple_r"] - idx["charge_only"]["multiple_r"],
                "mae_improve_pct": (idx["charge_only"]["mae_in_sample"] - comb["mae_in_sample"]) / max(idx["charge_only"]["mae_in_sample"], 1e-12),
                "rmse_improve_pct": (idx["charge_only"]["rmse_in_sample"] - comb["rmse_in_sample"]) / max(idx["charge_only"]["rmse_in_sample"], 1e-12),
            },
            {
                "compare_to": "discharge_only",
                "delta_r2": comb["r2_in_sample"] - idx["discharge_only"]["r2_in_sample"],
                "delta_multiple_r": comb["multiple_r"] - idx["discharge_only"]["multiple_r"],
                "mae_improve_pct": (idx["discharge_only"]["mae_in_sample"] - comb["mae_in_sample"]) / max(idx["discharge_only"]["mae_in_sample"], 1e-12),
                "rmse_improve_pct": (idx["discharge_only"]["rmse_in_sample"] - comb["rmse_in_sample"]) / max(idx["discharge_only"]["rmse_in_sample"], 1e-12),
            },
        ]
    )

    charge_df = _try_load_csv(args.charge_result_dir / "correlation_by_range.csv")
    discharge_df = _try_load_csv(args.discharge_result_dir / "univariate_correlation.csv")

    out_summary_csv = args.output_dir / "combo_correlation_summary.csv"
    out_uplift_csv = args.output_dir / "combo_correlation_uplift.csv"
    out_plot_png = args.output_dir / "combo_correlation_comparison.png"
    out_report_md = args.output_dir / "correlation_summary_no_policy.md"

    # Keep only one markdown in this summary directory.
    for p in args.output_dir.glob("*.md"):
        if p.resolve() != out_report_md.resolve():
            p.unlink(missing_ok=True)

    summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")
    uplift_df.to_csv(out_uplift_csv, index=False, encoding="utf-8")
    save_comparison_plot(summary_df, out_plot_png, has_cjk)
    render_report(
        report_path=out_report_md,
        fonts=fonts,
        data=data,
        charge_cols=charge_cols,
        discharge_cols=discharge_cols,
        summary_df=summary_df,
        uplift_df=uplift_df,
        charge_df=charge_df,
        discharge_df=discharge_df,
    )

    print(f"Saved: {out_summary_csv}")
    print(f"Saved: {out_uplift_csv}")
    print(f"Saved: {out_plot_png}")
    print(f"Saved: {out_report_md}")
    print(f"Rows={len(data)} | features charge/discharge={len(charge_cols)}/{len(discharge_cols)}")


if __name__ == "__main__":
    main()
