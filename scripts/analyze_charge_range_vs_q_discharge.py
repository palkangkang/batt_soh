from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


# =========================
# Config (edit here first)
# =========================
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
CHARGE_FEATURE_PATH = REPO_ROOT / "data" / "processed" / "charge_interval_features.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "charge_feature_q_discharge_corr"

ENCODING = "utf-8-sig"
RANDOM_SEED = 20260317
SCATTER_SAMPLE_N = 5000
WINSOR_LOWER_Q = 0.05
WINSOR_UPPER_Q = 0.95
PYTHON_ENV_HOME = Path(r"C:\Users\pal\pyenv\colab")
FIRST_OCCURRENCE_RANGE_COUNT = 1
POLICY_MIN_SAMPLES = 80


# Matplotlib config must be set before importing pyplot.
MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402


@dataclass
class RangeStats:
    range_label: str
    n_samples: int
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    pearson_r: float
    pearson_p: float
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    pearson_winsor_5_95: float
    pearson_policy_demean: float
    lin_slope: float
    lin_intercept: float
    lin_r2: float
    lin_p: float


def parse_range_low(range_label: str) -> float:
    # Example: [3.0,3.1) -> 3.0
    try:
        body = range_label.strip()[1:]
        low = body.split(",")[0]
        return float(low)
    except Exception:
        return float("inf")


def sanitize_range_label(range_label: str) -> str:
    # Example: [3.0,3.1) -> 3p0_3p1
    return (
        range_label.replace("[", "")
        .replace("]", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def build_charge_variable_name(range_label: str) -> str:
    return f"charge_delta_ah_{sanitize_range_label(range_label)}"


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


def fisher_pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def safe_corr(func, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        r, p = func(x, y)
        if np.isfinite(r) and np.isfinite(p):
            return float(r), float(p)
    except Exception:
        pass
    return float("nan"), float("nan")


def compute_range_stats(df: pd.DataFrame) -> list[RangeStats]:
    result: list[RangeStats] = []
    for range_label, part in df.groupby("range", sort=False):
        x = part["delta_ah_sum"].to_numpy(dtype=float)
        y = part["q_discharge"].to_numpy(dtype=float)
        n = len(part)
        if n < 5:
            continue

        pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
        kendall_tau, kendall_p = safe_corr(stats.kendalltau, x, y)
        ci_low, ci_high = fisher_pearson_ci(pearson_r, n)

        x_lo, x_hi = np.quantile(x, [WINSOR_LOWER_Q, WINSOR_UPPER_Q])
        y_lo, y_hi = np.quantile(y, [WINSOR_LOWER_Q, WINSOR_UPPER_Q])
        x_w = np.clip(x, x_lo, x_hi)
        y_w = np.clip(y, y_lo, y_hi)
        pearson_winsor, _ = safe_corr(stats.pearsonr, x_w, y_w)

        # Remove policy-level mean effects.
        x_dm = part["delta_ah_sum"] - part.groupby("policy")["delta_ah_sum"].transform("mean")
        y_dm = part["q_discharge"] - part.groupby("policy")["q_discharge"].transform("mean")
        pearson_policy_dm, _ = safe_corr(
            stats.pearsonr,
            x_dm.to_numpy(dtype=float),
            y_dm.to_numpy(dtype=float),
        )

        try:
            lin = stats.linregress(x, y)
            slope = float(lin.slope)
            intercept = float(lin.intercept)
            r2 = float(lin.rvalue**2)
            lin_p = float(lin.pvalue)
        except Exception:
            slope = float("nan")
            intercept = float("nan")
            r2 = float("nan")
            lin_p = float("nan")

        result.append(
            RangeStats(
                range_label=range_label,
                n_samples=n,
                x_mean=float(np.mean(x)),
                x_std=float(np.std(x, ddof=1)),
                y_mean=float(np.mean(y)),
                y_std=float(np.std(y, ddof=1)),
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                pearson_ci_low=ci_low,
                pearson_ci_high=ci_high,
                spearman_rho=spearman_rho,
                spearman_p=spearman_p,
                kendall_tau=kendall_tau,
                kendall_p=kendall_p,
                pearson_winsor_5_95=pearson_winsor,
                pearson_policy_demean=pearson_policy_dm,
                lin_slope=slope,
                lin_intercept=intercept,
                lin_r2=r2,
                lin_p=lin_p,
            )
        )
    return result


def compute_policy_layer_stats(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    rows = []
    for (policy, range_label), part in df.groupby(["policy", "range"], sort=False):
        x = part["delta_ah_sum"].to_numpy(dtype=float)
        y = part["q_discharge"].to_numpy(dtype=float)
        n = len(part)
        if n < 5:
            continue

        pearson_r, pearson_p = safe_corr(stats.pearsonr, x, y)
        spearman_rho, spearman_p = safe_corr(stats.spearmanr, x, y)
        kendall_tau, kendall_p = safe_corr(stats.kendalltau, x, y)
        ci_low, ci_high = fisher_pearson_ci(pearson_r, n)

        try:
            lin = stats.linregress(x, y)
            lin_r2 = float(lin.rvalue**2)
            lin_p = float(lin.pvalue)
        except Exception:
            lin_r2 = float("nan")
            lin_p = float("nan")

        rows.append(
            {
                "policy": policy,
                "range_label": range_label,
                "n_samples": int(n),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "pearson_ci_low": ci_low,
                "pearson_ci_high": ci_high,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_tau,
                "kendall_p": kendall_p,
                "lin_r2": lin_r2,
                "lin_p": lin_p,
                "is_eligible": bool(n >= min_samples),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["range_low"] = out["range_label"].map(parse_range_low)
    out = out.sort_values(["range_low", "policy"]).reset_index(drop=True)
    return out


def build_policy_range_summary(
    policy_stats_df: pd.DataFrame,
    overall_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    if policy_stats_df.empty:
        return pd.DataFrame()

    overall_spearman_map = dict(zip(overall_stats_df["range_label"], overall_stats_df["spearman_rho"]))
    all_policy_cnt = policy_stats_df["policy"].nunique()
    rows = []

    for range_label, part_all in policy_stats_df.groupby("range_label", sort=False):
        part = part_all.loc[part_all["is_eligible"]].copy()
        total_pairs = len(part_all)
        eligible_pairs = len(part)
        overall_spearman = float(overall_spearman_map.get(range_label, np.nan))

        if eligible_pairs == 0:
            rows.append(
                {
                    "range_label": range_label,
                    "n_policies_total": int(all_policy_cnt),
                    "n_policies_eligible": 0,
                    "eligible_ratio": 0.0,
                    "overall_spearman": overall_spearman,
                    "median_spearman": float("nan"),
                    "spearman_q25": float("nan"),
                    "spearman_q75": float("nan"),
                    "spearman_min": float("nan"),
                    "spearman_max": float("nan"),
                    "spearman_iqr": float("nan"),
                    "positive_ratio": float("nan"),
                    "negative_ratio": float("nan"),
                    "same_sign_ratio_vs_overall": float("nan"),
                    "n_policy_pairs_total": int(total_pairs),
                    "n_policy_pairs_eligible": int(eligible_pairs),
                }
            )
            continue

        s = part["spearman_rho"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        s_q25, s_q75 = np.quantile(s, [0.25, 0.75])
        pos_ratio = float(np.mean(s > 0))
        neg_ratio = float(np.mean(s < 0))

        if np.isfinite(overall_spearman) and overall_spearman != 0:
            same_sign_ratio = float(np.mean(np.sign(s) == np.sign(overall_spearman)))
        else:
            same_sign_ratio = float("nan")

        rows.append(
            {
                "range_label": range_label,
                "n_policies_total": int(all_policy_cnt),
                "n_policies_eligible": int(part["policy"].nunique()),
                "eligible_ratio": float(part["policy"].nunique() / max(all_policy_cnt, 1)),
                "overall_spearman": overall_spearman,
                "median_spearman": float(np.median(s)),
                "spearman_q25": float(s_q25),
                "spearman_q75": float(s_q75),
                "spearman_min": float(np.min(s)),
                "spearman_max": float(np.max(s)),
                "spearman_iqr": float(s_q75 - s_q25),
                "positive_ratio": pos_ratio,
                "negative_ratio": neg_ratio,
                "same_sign_ratio_vs_overall": same_sign_ratio,
                "n_policy_pairs_total": int(total_pairs),
                "n_policy_pairs_eligible": int(eligible_pairs),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["range_low"] = out["range_label"].map(parse_range_low)
    out = out.sort_values("range_low").reset_index(drop=True)
    return out


def save_policy_spearman_heatmap(policy_stats_df: pd.DataFrame, path: Path) -> None:
    eligible = policy_stats_df.loc[policy_stats_df["is_eligible"]].copy()
    if eligible.empty:
        return

    range_order = sorted(eligible["range_label"].unique(), key=parse_range_low)
    policy_order = (
        eligible.groupby("policy")["n_samples"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    heat_df = (
        eligible.pivot_table(index="policy", columns="range_label", values="spearman_rho", aggfunc="mean")
        .reindex(index=policy_order, columns=range_order)
    )

    fig_h = max(6.0, min(22.0, 0.22 * len(policy_order) + 2.0))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    matrix = heat_df.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(range_order)))
    ax.set_xticklabels(range_order)
    ax.set_yticks(np.arange(len(policy_order)))
    y_font_size = 6 if len(policy_order) > 50 else 8
    ax.set_yticklabels(policy_order, fontsize=y_font_size)
    ax.set_xlabel("充电电压区间")
    ax.set_ylabel("Policy")
    ax.set_title(f"Policy 分层 Spearman 相关性热力图（n >= {POLICY_MIN_SAMPLES}）")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def save_policy_spearman_boxplot(
    policy_stats_df: pd.DataFrame,
    overall_stats_df: pd.DataFrame,
    path: Path,
) -> None:
    eligible = policy_stats_df.loc[policy_stats_df["is_eligible"]].copy()
    if eligible.empty:
        return

    range_order = sorted(eligible["range_label"].unique(), key=parse_range_low)
    data = [
        eligible.loc[eligible["range_label"] == r, "spearman_rho"].dropna().to_numpy(dtype=float)
        for r in range_order
    ]
    overall_map = dict(zip(overall_stats_df["range_label"], overall_stats_df["spearman_rho"]))
    overall_vals = [overall_map.get(r, np.nan) for r in range_order]

    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    box = ax.boxplot(data, tick_labels=range_order, showfliers=False, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#bfdbfe")
        patch.set_alpha(0.7)

    x = np.arange(1, len(range_order) + 1)
    ax.plot(x, overall_vals, marker="D", color="#dc2626", linewidth=1.4, label="全局 Spearman")
    ax.axhline(0.0, color="#64748b", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_xlabel("充电电压区间")
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"Policy 分层 Spearman 分布（箱线图，n >= {POLICY_MIN_SAMPLES}）")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def render_policy_markdown_report(
    report_path: Path,
    policy_stats_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_samples: int,
) -> None:
    lines: list[str] = []
    lines.append("# Policy 分层相关性补充报告")
    lines.append("")
    lines.append("## 1. 目的与范围")
    lines.append("- 目的：在保留全局汇总结果的基础上，补充按 `policy` 分层的相关性分析。")
    lines.append("- 口径：仅使用首次出现区间特征（`range_count == 1`），并对每个 `policy + range` 独立计算相关系数。")
    lines.append(f"- 纳入阈值：仅对样本数 `n >= {min_samples}` 的 `policy + range` 组合做分层稳健性统计。")
    lines.append("")

    total_policy_pairs = len(policy_stats_df)
    eligible_pairs = int(policy_stats_df["is_eligible"].sum()) if not policy_stats_df.empty else 0
    total_policies = int(policy_stats_df["policy"].nunique()) if not policy_stats_df.empty else 0
    eligible_policies = (
        int(policy_stats_df.loc[policy_stats_df["is_eligible"], "policy"].nunique())
        if not policy_stats_df.empty
        else 0
    )
    lines.append("## 2. 样本覆盖")
    lines.append(f"- `policy + range` 组合总数：**{total_policy_pairs:,}**")
    lines.append(f"- 满足阈值的组合数：**{eligible_pairs:,}**")
    lines.append(f"- 覆盖 policy 总数：**{total_policies:,}**")
    lines.append(f"- 至少一个区间满足阈值的 policy 数：**{eligible_policies:,}**")
    lines.append("")

    lines.append("## 3. 各区间分层稳健性摘要")
    lines.append(
        "| 电压区间 | 合格Policy数/总Policy数 | 全局Spearman | Policy中位数Spearman | IQR | 同号比例(相对全局) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['range_label']} | {int(r['n_policies_eligible'])}/{int(r['n_policies_total'])} | "
            f"{r['overall_spearman']:.4f} | {r['median_spearman']:.4f} | {r['spearman_iqr']:.4f} | "
            f"{r['same_sign_ratio_vs_overall']:.2%} |"
        )
    lines.append("")

    if not summary_df.empty:
        most_stable = summary_df.sort_values("same_sign_ratio_vs_overall", ascending=False).iloc[0]
        most_diverse = summary_df.sort_values("spearman_iqr", ascending=False).iloc[0]
        lines.append("## 4. 结果解读（客观描述）")
        lines.append(
            f"- 同号一致性最高区间：**{most_stable['range_label']}**，同号比例 "
            f"{most_stable['same_sign_ratio_vs_overall']:.2%}。"
        )
        lines.append(
            f"- policy 间离散度最高区间（Spearman IQR）：**{most_diverse['range_label']}**，"
            f"IQR={most_diverse['spearman_iqr']:.4f}。"
        )
        lines.append("- 若某区间同号比例高且 IQR 小，可视为跨策略更稳健；反之说明策略依赖更强。")
        lines.append("")

    lines.append("## 5. 图表")
    lines.append("![Policy分层 Spearman 热力图](./policy_spearman_heatmap.png)")
    lines.append("")
    lines.append("![Policy分层 Spearman 箱线图](./policy_spearman_boxplot.png)")
    lines.append("")
    lines.append("## 6. 文件说明")
    lines.append("- `correlation_by_policy_range.csv`：每个 `policy + range` 的分层相关统计。")
    lines.append("- `range_policy_stratified_summary.csv`：按电压区间聚合后的分层稳健性摘要。")
    lines.append("- 本报告为补充分析，不替换既有全局报告。")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_coefficients_plot(stats_df: pd.DataFrame, path: Path) -> None:
    stats_df = stats_df.sort_values("range_low", ascending=True).reset_index(drop=True)
    x_idx = np.arange(len(stats_df))
    w = 0.24

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(x_idx - w, stats_df["pearson_r"], width=w, label="Pearson", color="#2563eb")
    ax.bar(x_idx, stats_df["spearman_rho"], width=w, label="Spearman", color="#f59e0b")
    ax.bar(x_idx + w, stats_df["kendall_tau"], width=w, label="Kendall", color="#16a34a")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(stats_df["range_label"], rotation=0)
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="#64748b", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel("相关系数")
    ax.set_xlabel("充电电压区间")
    ax.set_title("各电压区间与放电容量的相关系数对比")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def save_robustness_plot(stats_df: pd.DataFrame, path: Path) -> None:
    stats_df = stats_df.sort_values("range_low", ascending=True).reset_index(drop=True)
    x_idx = np.arange(len(stats_df))

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(x_idx, stats_df["pearson_r"], marker="o", label="Pearson", color="#2563eb")
    ax.plot(
        x_idx,
        stats_df["pearson_winsor_5_95"],
        marker="o",
        label="Winsorized Pearson (5%-95%)",
        color="#f59e0b",
    )
    ax.plot(
        x_idx,
        stats_df["pearson_policy_demean"],
        marker="o",
        label="Policy-demeaned Pearson",
        color="#16a34a",
    )
    ax.fill_between(
        x_idx,
        stats_df["pearson_ci_low"],
        stats_df["pearson_ci_high"],
        color="#93c5fd",
        alpha=0.22,
        label="Pearson 95% CI",
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(stats_df["range_label"])
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="#64748b", linewidth=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel("相关系数")
    ax.set_xlabel("充电电压区间")
    ax.set_title("相关性稳健性对比")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def save_scatter_top3(merged_df: pd.DataFrame, stats_df: pd.DataFrame, path: Path) -> None:
    top3 = (
        stats_df.assign(abs_spearman=stats_df["spearman_rho"].abs())
        .sort_values("abs_spearman", ascending=False)
        .head(3)
    )
    if top3.empty:
        return

    fig, axes = plt.subplots(1, len(top3), figsize=(5.6 * len(top3), 4.6))
    if len(top3) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top3.iterrows()):
        rng = row["range_label"]
        part = merged_df.loc[merged_df["range"] == rng, ["delta_ah_sum", "q_discharge"]].copy()
        if len(part) > SCATTER_SAMPLE_N:
            part = part.sample(n=SCATTER_SAMPLE_N, random_state=RANDOM_SEED)

        x = part["delta_ah_sum"].to_numpy(dtype=float)
        y = part["q_discharge"].to_numpy(dtype=float)

        ax.scatter(x, y, s=8, alpha=0.23, color="#2563eb", edgecolors="none")
        try:
            lin = stats.linregress(x, y)
            xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            yy = lin.slope * xx + lin.intercept
            ax.plot(xx, yy, color="#dc2626", linewidth=1.8, label="线性拟合")
        except Exception:
            pass

        ax.set_title(
            f"{rng}\nPearson={row['pearson_r']:.3f}, Spearman={row['spearman_rho']:.3f}, n={int(row['n_samples'])}"
        )
        ax.set_xlabel("区间容量差 delta_ah_sum (Ah)")
        ax.set_ylabel("放电容量 q_discharge (Ah)")
        ax.grid(alpha=0.25, linestyle="--")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("代表区间散点图（抽样）", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)


def render_markdown_report(
    report_path: Path,
    python_env_home: Path,
    python_executable: str,
    font_list: List[str],
    life_count: int,
    charge_raw_count: int,
    charge_first_occ_count: int,
    charge_agg_count: int,
    merged_count: int,
    unique_policies: int,
    unique_cells: int,
    stats_df: pd.DataFrame,
) -> None:
    stats_df_sorted = stats_df.sort_values("range_low", ascending=True).reset_index(drop=True)
    ranking = stats_df.assign(abs_spearman=stats_df["spearman_rho"].abs()).sort_values(
        "abs_spearman", ascending=False
    )

    lines: list[str] = []
    lines.append("# 充电电压区间容量差与放电容量相关性探索报告")
    lines.append("")
    lines.append("## 1. 任务与口径")
    lines.append("- 目标：分析充电特征在不同电压阶段的容量差异（`delta_ah_sum`）与循环放电容量（`q_discharge`）的相关性。")
    lines.append("- 合并键：`policy + cell_code + cycles`。")
    lines.append("- 聚合方式：对 `charge_interval_features` 中同一 `policy+cell_code+cycles+range` 的 `delta_ah` 求和，得到 `delta_ah_sum`。")
    lines.append("- 本次分析仅使用首次出现区间特征：`range_count == 1`。")
    lines.append("")
    lines.append("## 2. 运行环境与数据规模")
    lines.append(f"- 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 环境入口目录：`{python_env_home}`（通过 `pipenv run python` 调用）")
    lines.append(f"- Python 解释器：`{python_executable}`")
    lines.append(f"- 中文字体回退链（已检测）：`{', '.join(font_list)}`")
    lines.append(f"- `life_performance` 行数：**{life_count:,}**")
    lines.append(f"- `charge_interval_features` 原始行数：**{charge_raw_count:,}**")
    lines.append(
        f"- `charge_interval_features` 首次出现筛选后行数（range_count == {FIRST_OCCURRENCE_RANGE_COUNT}）："
        f"**{charge_first_occ_count:,}**（保留率 {charge_first_occ_count / max(charge_raw_count, 1):.2%}）"
    )
    lines.append(f"- `charge` 按 cycle+range 聚合后行数：**{charge_agg_count:,}**")
    lines.append(f"- 合并后分析样本点数：**{merged_count:,}**")
    lines.append(f"- 覆盖策略数：**{unique_policies:,}**；覆盖电芯数：**{unique_cells:,}**")
    lines.append("")
    lines.append("## 3. 统计方法")
    lines.append("- `Pearson`：线性相关性。")
    lines.append("- `Spearman`：单调相关性（对非线性更稳健）。")
    lines.append("- `Kendall tau`：秩相关稳健度补充。")
    lines.append("- `Winsorized Pearson (5%-95%)`：削弱极值影响后再计算 Pearson。")
    lines.append("- `Policy-demeaned Pearson`：按策略去均值后计算，降低策略层面系统差异影响。")
    lines.append("- 线性回归：给出斜率、截距、R²、p 值。")
    lines.append("")
    lines.append("## 4. 各区间相关性明细")
    lines.append(
        "| 电压区间 | n | Pearson (95% CI) | Spearman | Kendall | Winsorized Pearson | Policy-demeaned Pearson | 线性R² |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in stats_df_sorted.iterrows():
        lines.append(
            f"| {r['range_label']} | {int(r['n_samples'])} | "
            f"{r['pearson_r']:.4f} [{r['pearson_ci_low']:.4f}, {r['pearson_ci_high']:.4f}] | "
            f"{r['spearman_rho']:.4f} | {r['kendall_tau']:.4f} | "
            f"{r['pearson_winsor_5_95']:.4f} | {r['pearson_policy_demean']:.4f} | {r['lin_r2']:.4f} |"
        )
    lines.append("")
    lines.append("## 5. 结果解读（客观描述）")
    if not ranking.empty:
        top = ranking.iloc[0]
        lines.append(
            f"- 按 `|Spearman|` 排序，相关性最高区间为 **{top['range_label']}**，"
            f"Spearman={top['spearman_rho']:.4f}，Pearson={top['pearson_r']:.4f}。"
        )
    if len(ranking) >= 2:
        second = ranking.iloc[1]
        lines.append(
            f"- 次高区间为 **{second['range_label']}**，Spearman={second['spearman_rho']:.4f}，Pearson={second['pearson_r']:.4f}。"
        )
    neg = stats_df_sorted.loc[stats_df_sorted["pearson_r"] < 0]
    if not neg.empty:
        neg_desc = ", ".join([f"{r}({v:.3f})" for r, v in zip(neg["range_label"], neg["pearson_r"])])
        lines.append(f"- 存在负相关区间：{neg_desc}。")
    lines.append("- 各方法方向大体一致时，可认为区间相关性具备一定稳健性。")
    lines.append("")
    lines.append("## 6. 图表")
    lines.append("![各电压区间相关系数对比](./coefficients_by_range.png)")
    lines.append("")
    lines.append("![稳健性方法对比](./robustness_by_range.png)")
    lines.append("")
    lines.append("![代表区间散点图](./top3_scatter.png)")
    lines.append("")
    lines.append("## 7. 局限性与后续建议")
    lines.append("- 相关分析不代表因果关系。")
    lines.append("- 当前分析未引入时间切分与训练/验证隔离，不能直接代替模型泛化评估。")
    lines.append("- 如用于特征筛选，建议补充：按策略分层验证、按生命周期阶段分段验证、与温度/内阻协变量联合建模。")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_fonts = setup_fonts()

    life_df = pd.read_csv(
        LIFE_PERFORMANCE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    charge_df = pd.read_csv(
        CHARGE_FEATURE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"],
    )

    life_df["cycles"] = pd.to_numeric(life_df["cycles"], errors="coerce")
    life_df["q_discharge"] = pd.to_numeric(life_df["q_discharge"], errors="coerce")
    charge_df["cycles"] = pd.to_numeric(charge_df["cycles"], errors="coerce")
    charge_df["delta_ah"] = pd.to_numeric(charge_df["delta_ah"], errors="coerce")
    charge_df["range_count"] = pd.to_numeric(charge_df["range_count"], errors="coerce")

    life_df = life_df.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    charge_df = charge_df.dropna(
        subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]
    ).copy()
    life_df["cycles"] = life_df["cycles"].astype(int)
    charge_df["cycles"] = charge_df["cycles"].astype(int)
    charge_df["range_count"] = charge_df["range_count"].astype(int)

    life_count = len(life_df)
    charge_raw_count = len(charge_df)
    charge_df = charge_df.loc[charge_df["range_count"] == FIRST_OCCURRENCE_RANGE_COUNT].copy()
    charge_first_occ_count = len(charge_df)

    charge_agg = (
        charge_df.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(delta_ah_sum=("delta_ah", "sum"), segment_count=("delta_ah", "size"))
    )

    merged = charge_agg.merge(
        life_df,
        on=["policy", "cell_code", "cycles"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["delta_ah_sum", "q_discharge"]).copy()

    if merged.empty:
        raise RuntimeError("Merged dataframe is empty. Please check key alignment and source files.")

    merged["range_low"] = merged["range"].map(parse_range_low)
    merged = merged.sort_values(["range_low", "policy", "cell_code", "cycles"]).reset_index(drop=True)

    stats_list = compute_range_stats(merged)
    if not stats_list:
        raise RuntimeError("No valid range statistics were generated.")

    stats_df = pd.DataFrame([s.__dict__ for s in stats_list])
    stats_df["range_low"] = stats_df["range_label"].map(parse_range_low)
    stats_df = stats_df.sort_values("range_low", ascending=True).reset_index(drop=True)
    stats_df.insert(0, "variable", stats_df["range_label"].map(build_charge_variable_name))
    stats_df.insert(1, "variable_display", stats_df["range_label"])
    stats_df.insert(2, "range", stats_df["range_label"])

    stats_out = OUTPUT_DIR / "correlation_by_range.csv"
    merged_out = OUTPUT_DIR / "merged_dataset_overview.csv"
    report_out = OUTPUT_DIR / "report_charge_vs_q_discharge.md"
    coef_png = OUTPUT_DIR / "coefficients_by_range.png"
    robust_png = OUTPUT_DIR / "robustness_by_range.png"
    scatter_png = OUTPUT_DIR / "top3_scatter.png"
    policy_stats_out = OUTPUT_DIR / "correlation_by_policy_range.csv"
    policy_summary_out = OUTPUT_DIR / "range_policy_stratified_summary.csv"
    policy_heatmap_png = OUTPUT_DIR / "policy_spearman_heatmap.png"
    policy_boxplot_png = OUTPUT_DIR / "policy_spearman_boxplot.png"
    policy_report_out = OUTPUT_DIR / "report_policy_stratified.md"

    stats_df.to_csv(stats_out, index=False, encoding="utf-8")

    merged_preview = (
        merged.groupby("range", as_index=False)
        .agg(
            n_samples=("delta_ah_sum", "size"),
            delta_ah_mean=("delta_ah_sum", "mean"),
            delta_ah_std=("delta_ah_sum", "std"),
            q_discharge_mean=("q_discharge", "mean"),
            q_discharge_std=("q_discharge", "std"),
            mean_segment_count=("segment_count", "mean"),
        )
        .assign(range_low=lambda d: d["range"].map(parse_range_low))
        .sort_values("range_low")
        .drop(columns=["range_low"])
    )
    merged_preview.insert(0, "variable", merged_preview["range"].map(build_charge_variable_name))
    merged_preview.insert(1, "variable_display", merged_preview["range"])
    merged_preview.to_csv(merged_out, index=False, encoding="utf-8")

    save_coefficients_plot(stats_df, coef_png)
    save_robustness_plot(stats_df, robust_png)
    save_scatter_top3(merged, stats_df, scatter_png)

    policy_stats_df = compute_policy_layer_stats(merged, min_samples=POLICY_MIN_SAMPLES)
    if policy_stats_df.empty:
        raise RuntimeError("No policy-layer statistics were generated.")
    policy_stats_df.insert(1, "variable", policy_stats_df["range_label"].map(build_charge_variable_name))
    policy_stats_df.insert(2, "variable_display", policy_stats_df["range_label"])
    policy_stats_df.to_csv(policy_stats_out, index=False, encoding="utf-8")

    policy_summary_df = build_policy_range_summary(policy_stats_df, stats_df)
    if policy_summary_df.empty:
        raise RuntimeError("No policy-layer summary was generated.")
    policy_summary_df.insert(1, "variable", policy_summary_df["range_label"].map(build_charge_variable_name))
    policy_summary_df.insert(2, "variable_display", policy_summary_df["range_label"])
    policy_summary_df.to_csv(policy_summary_out, index=False, encoding="utf-8")
    save_policy_spearman_heatmap(policy_stats_df, policy_heatmap_png)
    save_policy_spearman_boxplot(policy_stats_df, stats_df, policy_boxplot_png)
    render_policy_markdown_report(
        report_path=policy_report_out,
        policy_stats_df=policy_stats_df,
        summary_df=policy_summary_df,
        min_samples=POLICY_MIN_SAMPLES,
    )

    render_markdown_report(
        report_path=report_out,
        python_env_home=PYTHON_ENV_HOME,
        python_executable=os.environ.get("PYTHON_EXECUTABLE", ""),
        font_list=selected_fonts,
        life_count=life_count,
        charge_raw_count=charge_raw_count,
        charge_first_occ_count=charge_first_occ_count,
        charge_agg_count=len(charge_agg),
        merged_count=len(merged),
        unique_policies=int(merged["policy"].nunique()),
        unique_cells=int(merged["cell_code"].nunique()),
        stats_df=stats_df,
    )

    print(f"Saved: {stats_out}")
    print(f"Saved: {merged_out}")
    print(f"Saved: {coef_png}")
    print(f"Saved: {robust_png}")
    print(f"Saved: {scatter_png}")
    print(f"Saved: {report_out}")
    print(f"Saved: {policy_stats_out}")
    print(f"Saved: {policy_summary_out}")
    print(f"Saved: {policy_heatmap_png}")
    print(f"Saved: {policy_boxplot_png}")
    print(f"Saved: {policy_report_out}")
    print(f"Merged samples: {len(merged)}")


if __name__ == "__main__":
    # Keep a stable executable path in report when run via pipenv.
    os.environ.setdefault("PYTHON_EXECUTABLE", os.path.realpath(os.sys.executable))
    main()
