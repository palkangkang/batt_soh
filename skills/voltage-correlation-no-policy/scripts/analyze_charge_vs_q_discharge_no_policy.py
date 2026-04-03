from __future__ import annotations

import argparse
import math
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
        description="Analyze charge voltage-interval features vs q_discharge without policy variables."
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "charge_feature_q_discharge_corr_no_policy",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--first-occurrence", type=int, default=1)
    parser.add_argument("--scatter-sample-n", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260403)
    return parser.parse_args()


def fisher_pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def compute_stats(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for range_label, part in merged.groupby("range", sort=False):
        x = part["delta_ah_sum"].to_numpy(dtype=float)
        y = part["q_discharge"].to_numpy(dtype=float)
        n = len(part)

        pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
        kendall_tau, kendall_p = safe_corr(stats.kendalltau, x, y)
        ci_low, ci_high = fisher_pearson_ci(pearson_r, n)

        if n >= 5:
            x_lo, x_hi = np.quantile(x, [0.05, 0.95])
            y_lo, y_hi = np.quantile(y, [0.05, 0.95])
            x_w = np.clip(x, x_lo, x_hi)
            y_w = np.clip(y, y_lo, y_hi)
            pearson_winsor, _ = safe_corr(stats.pearsonr, x_w, y_w)
        else:
            pearson_winsor = float("nan")

        try:
            lin = stats.linregress(x, y)
            slope = float(lin.slope)
            intercept = float(lin.intercept)
            lin_r2 = float(lin.rvalue**2)
            lin_p = float(lin.pvalue)
        except Exception:
            slope = float("nan")
            intercept = float("nan")
            lin_r2 = float("nan")
            lin_p = float("nan")

        rows.append(
            {
                "variable": f"charge_delta_ah_{sanitize_range_label(range_label)}",
                "variable_display": str(range_label),
                "n_samples": int(n),
                "x_mean": float(np.mean(x)),
                "x_std": float(np.std(x, ddof=1)) if n > 1 else float("nan"),
                "y_mean": float(np.mean(y)),
                "y_std": float(np.std(y, ddof=1)) if n > 1 else float("nan"),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "pearson_ci_low": ci_low,
                "pearson_ci_high": ci_high,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p,
                "pearson_winsor_5_95": pearson_winsor,
                "lin_slope": slope,
                "lin_intercept": intercept,
                "lin_r2": lin_r2,
                "lin_p": lin_p,
            }
        )

    out = pd.DataFrame(rows)
    out["range_low"] = out["variable_display"].map(parse_range_low)
    out = out.sort_values("range_low").reset_index(drop=True)
    return out


def save_coeff_plot(stats_df: pd.DataFrame, path: Path, zh: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    x = np.arange(len(stats_df))
    ax.plot(x, stats_df["spearman_rho"], marker="o", linewidth=1.5, label="Spearman")
    ax.plot(x, stats_df["pearson_r"], marker="s", linewidth=1.5, label="Pearson")
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["variable_display"].tolist(), rotation=40, ha="right")
    ax.set_ylabel("相关系数" if zh else "Correlation")
    ax.set_title("充电电压区间与放电容量相关性" if zh else "Charge Range vs Q_discharge Correlation")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def save_robust_plot(stats_df: pd.DataFrame, path: Path, zh: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    x = np.arange(len(stats_df))
    w = 0.4
    ax.bar(x - w / 2, stats_df["pearson_r"], width=w, label="Pearson", color="#0ea5e9")
    ax.bar(x + w / 2, stats_df["pearson_winsor_5_95"], width=w, label="Winsor(5-95)", color="#22c55e")
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["variable_display"].tolist(), rotation=40, ha="right")
    ax.set_ylabel("相关系数" if zh else "Correlation")
    ax.set_title("稳健性对比（原始Pearson vs Winsor）" if zh else "Robustness: Pearson vs Winsor")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def save_top3_scatter(
    merged: pd.DataFrame,
    stats_df: pd.DataFrame,
    path: Path,
    sample_n: int,
    random_seed: int,
    zh: bool,
) -> None:
    import matplotlib.pyplot as plt

    use = stats_df.copy()
    use["abs_spearman"] = use["spearman_rho"].abs()
    top = use.sort_values("abs_spearman", ascending=False).head(3)
    if top.empty:
        return

    fig, axes = plt.subplots(1, len(top), figsize=(5.4 * len(top), 4.2), squeeze=False)
    rng = np.random.default_rng(random_seed)

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[0, i]
        label = row["variable_display"]
        part = merged.loc[merged["range"] == label, ["delta_ah_sum", "q_discharge"]].dropna().copy()
        if len(part) > sample_n:
            idx = rng.choice(part.index.to_numpy(), size=sample_n, replace=False)
            part = part.loc[idx]

        x = part["delta_ah_sum"].to_numpy(dtype=float)
        y = part["q_discharge"].to_numpy(dtype=float)
        ax.scatter(x, y, s=8, alpha=0.25)

        if len(part) >= 3:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 80)
            ys = m * xs + b
            ax.plot(xs, ys, color="#ef4444", linewidth=1.5)

        ax.set_xlabel("区间容量差(Ah)" if zh else "Delta Ah")
        ax.set_ylabel("放电容量(Ah)" if zh else "Q_discharge")
        ax.set_title(f"{label}\nSpearman={row['spearman_rho']:.3f}")
        ax.grid(alpha=0.2)

    fig.suptitle("Top3区间散点图" if zh else "Top-3 Range Scatter", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, has_cjk = setup_fonts(REPO_ROOT)

    life = load_life_performance_with_alias(args.life_path, args.encoding)
    charge_agg = load_interval_features_first_occurrence(
        args.charge_path,
        first_occurrence=args.first_occurrence,
        encoding=args.encoding,
    )

    merge_keys = choose_merge_keys(charge_agg.columns, life.columns)
    merged = charge_agg.merge(life, on=merge_keys, how="inner")
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after joining charge features with life labels.")

    stats_df = compute_stats(merged)
    if stats_df.empty:
        raise RuntimeError("No correlation rows produced from charge features.")

    overview_df = (
        merged.groupby("range", as_index=False)
        .agg(
            n_samples=("q_discharge", "size"),
            delta_ah_mean=("delta_ah_sum", "mean"),
            delta_ah_std=("delta_ah_sum", "std"),
            q_discharge_mean=("q_discharge", "mean"),
            q_discharge_std=("q_discharge", "std"),
        )
    )
    overview_df["range_low"] = overview_df["range"].map(parse_range_low)
    overview_df = overview_df.sort_values("range_low").reset_index(drop=True)

    out_csv = args.output_dir / "correlation_by_range.csv"
    out_overview_csv = args.output_dir / "merged_dataset_overview.csv"
    out_coef_png = args.output_dir / "coefficients_by_range.png"
    out_robust_png = args.output_dir / "robustness_by_range.png"
    out_scatter_png = args.output_dir / "top3_scatter.png"

    stats_df.to_csv(out_csv, index=False, encoding="utf-8")
    overview_df.to_csv(out_overview_csv, index=False, encoding="utf-8")
    save_coeff_plot(stats_df, out_coef_png, has_cjk)
    save_robust_plot(stats_df, out_robust_png, has_cjk)
    save_top3_scatter(
        merged=merged,
        stats_df=stats_df,
        path=out_scatter_png,
        sample_n=args.scatter_sample_n,
        random_seed=args.random_seed,
        zh=has_cjk,
    )

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_overview_csv}")
    print(f"Saved: {out_coef_png}")
    print(f"Saved: {out_robust_png}")
    print(f"Saved: {out_scatter_png}")
    print(f"Merged samples: {len(merged)}")


if __name__ == "__main__":
    main()
