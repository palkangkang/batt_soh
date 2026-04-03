from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    choose_merge_keys,
    load_interval_features_first_occurrence,
    load_life_performance_with_alias,
    parse_range_low,
    safe_corr,
    sanitize_range_label,
    setup_fonts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze discharge voltage-interval features vs q_discharge without policy variables."
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "life_performance.csv",
    )
    parser.add_argument(
        "--discharge-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "discharge_feature_q_discharge_corr_no_policy",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--first-occurrence", type=int, default=1)
    return parser.parse_args()


def compute_stats(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for range_label, part in merged.groupby("range", sort=False):
        x = part["delta_ah_sum"].to_numpy(dtype=float)
        y = part["q_discharge"].to_numpy(dtype=float)
        n = len(part)

        pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
        kendall_tau, kendall_p = safe_corr(stats.kendalltau, x, y)

        if n >= 5:
            x_lo, x_hi = np.quantile(x, [0.05, 0.95])
            y_lo, y_hi = np.quantile(y, [0.05, 0.95])
            x_w = np.clip(x, x_lo, x_hi)
            y_w = np.clip(y, y_lo, y_hi)
            pearson_winsor, _ = safe_corr(stats.pearsonr, x_w, y_w)
        else:
            pearson_winsor = float("nan")

        if n >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            try:
                lin = stats.linregress(x, y)
                lin_r2 = float(lin.rvalue**2)
                lin_p = float(lin.pvalue)
            except Exception:
                lin_r2 = float("nan")
                lin_p = float("nan")
        else:
            lin_r2 = float("nan")
            lin_p = float("nan")

        rows.append(
            {
                "variable": f"discharge_delta_ah_{sanitize_range_label(range_label)}",
                "variable_display": str(range_label),
                "variable_type": "discharge_feature",
                "n_samples": int(n),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p,
                "pearson_winsor_5_95": pearson_winsor,
                "lin_r2": lin_r2,
                "lin_p": lin_p,
            }
        )

    out = pd.DataFrame(rows)
    out["range_low"] = out["variable_display"].map(parse_range_low)
    out["abs_spearman"] = out["spearman_rho"].abs()
    out = out.sort_values(["range_low"]).reset_index(drop=True)
    return out


def save_top12_plot(stats_df: pd.DataFrame, path: Path, zh: bool) -> None:
    import matplotlib.pyplot as plt

    use = stats_df.sort_values("abs_spearman", ascending=False).head(12).copy()
    use = use.sort_values("abs_spearman", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(use["variable_display"], use["spearman_rho"], color="#0284c7")
    ax.set_xlabel("Spearman")
    ax.set_title("放电区间Top12相关性" if zh else "Top-12 Discharge Correlations")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def save_robust_plot(stats_df: pd.DataFrame, path: Path, zh: bool) -> None:
    import matplotlib.pyplot as plt

    use = stats_df.sort_values("abs_spearman", ascending=False).head(12).copy()
    use = use.sort_values("abs_spearman", ascending=True)

    y = np.arange(len(use))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.barh(y - 0.2, use["pearson_r"], height=0.35, label="Pearson", color="#0ea5e9")
    ax.barh(y + 0.2, use["pearson_winsor_5_95"], height=0.35, label="Winsor(5-95)", color="#22c55e")
    ax.set_yticks(y)
    ax.set_yticklabels(use["variable_display"].tolist())
    ax.set_xlabel("Correlation")
    ax.set_title("稳健性对比Top12" if zh else "Robustness Top-12")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, has_cjk = setup_fonts(REPO_ROOT)

    life = load_life_performance_with_alias(args.life_path, args.encoding)
    discharge_agg = load_interval_features_first_occurrence(
        args.discharge_path,
        first_occurrence=args.first_occurrence,
        encoding=args.encoding,
    )

    merge_keys = choose_merge_keys(discharge_agg.columns, life.columns)
    merged = discharge_agg.merge(life, on=merge_keys, how="inner")
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after joining discharge features with life labels.")

    stats_df = compute_stats(merged)
    if stats_df.empty:
        raise RuntimeError("No correlation rows produced from discharge features.")

    coverage_df = (
        merged.groupby("range", as_index=False)
        .agg(non_null_samples=("q_discharge", "size"))
        .rename(columns={"range": "variable_display"})
    )
    coverage_df["variable"] = coverage_df["variable_display"].map(
        lambda r: f"discharge_delta_ah_{sanitize_range_label(r)}"
    )
    coverage_df["variable_type"] = "discharge_feature"
    coverage_df["coverage_ratio_vs_life"] = coverage_df["non_null_samples"] / max(len(life), 1)
    coverage_df["range_low"] = coverage_df["variable_display"].map(parse_range_low)
    coverage_df = coverage_df.sort_values("range_low").reset_index(drop=True)

    out_uni_csv = args.output_dir / "univariate_correlation.csv"
    out_cov_csv = args.output_dir / "feature_coverage_summary.csv"
    out_uni_png = args.output_dir / "univariate_spearman_top12.png"
    out_robust_png = args.output_dir / "robustness_by_range.png"

    stats_df.to_csv(out_uni_csv, index=False, encoding="utf-8")
    coverage_df.to_csv(out_cov_csv, index=False, encoding="utf-8")
    save_top12_plot(stats_df, out_uni_png, has_cjk)
    save_robust_plot(stats_df, out_robust_png, has_cjk)

    print(f"Saved: {out_uni_csv}")
    print(f"Saved: {out_cov_csv}")
    print(f"Saved: {out_uni_png}")
    print(f"Saved: {out_robust_png}")
    print(f"Rows merged: {len(merged)} | discharge_features={stats_df.shape[0]}")


if __name__ == "__main__":
    main()
