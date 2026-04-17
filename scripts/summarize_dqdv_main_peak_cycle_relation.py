from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# =========================
# Config (edit here first)
# =========================
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "dqdv_main_peak_cycle_relation"

OUTPUT_LONG_CSV = "main_peak_cycle_relation_long.csv"
OUTPUT_WIDE_CSV = "main_peak_cycle_relation_wide.csv"
OUTPUT_REPORT_MD = "main_peak_cycle_relation_report.md"
OUTPUT_OVERVIEW_PNG = "main_peak_cycle_relation_overview.png"

ENCODING = "utf-8-sig"
RANDOM_SEED = 20260415
REPORT_TOP_N = 5
FLOAT_EPS = 1e-12

MAIN_PEAK_FEATURES: List[str] = [
    "main_peak_voltage_v",
    "main_peak_width_v",
    "main_peak_height_dqdv",
    "main_peak_area",
    "main_peak_prominence",
    "main_peak_skewness",
    "main_peak_temp_max_c",
    "main_peak_temp_min_c",
    "main_peak_temp_avg_c",
]

FEATURE_LABEL_MAP: Dict[str, str] = {
    "main_peak_voltage_v": "peak_voltage",
    "main_peak_width_v": "peak_width",
    "main_peak_height_dqdv": "peak_height",
    "main_peak_area": "peak_area",
    "main_peak_prominence": "peak_prominence",
    "main_peak_skewness": "peak_skewness",
    "main_peak_temp_max_c": "peak_temp_max",
    "main_peak_temp_min_c": "peak_temp_min",
    "main_peak_temp_avg_c": "peak_temp_avg",
}


MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class TrendMetrics:
    """Container for cycle-trend metrics of one feature in one combo."""

    n_cycles: int
    cycle_start: float
    cycle_end: float
    value_start: float
    value_end: float
    delta_abs: float
    delta_pct: float
    spearman_rho: float
    spearman_pvalue: float
    slope_per_cycle: float
    slope_per_100_cycles: float
    linear_r2: float


@dataclass
class ValidationSummary:
    """Container for validation checks."""

    expected_long_rows: int
    actual_long_rows: int
    rho_out_of_range_count: int
    pvalue_out_of_range_count: int
    spot_check_passed: bool
    spot_check_policy: str
    spot_check_cell_code: str
    spot_check_feature: str
    spot_check_abs_diff_delta: float
    spot_check_abs_diff_slope: float
    spot_check_abs_diff_rho: float


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize per-(policy,cell_code) main-peak feature relations vs cycles."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Input full dQ/dV feature table path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for summary artifacts.",
    )
    return parser.parse_args()


def to_numeric_bool(series: pd.Series) -> pd.Series:
    """Convert a possibly mixed-type bool-like column into strict bool."""
    if series.dtype == bool:
        return series
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "t"})
    )
    return normalized


def load_input_table(input_path: Path) -> pd.DataFrame:
    """Load and normalize the required columns from input CSV."""
    usecols = ["policy", "cell_code", "cycles", "is_valid_curve"] + MAIN_PEAK_FEATURES
    df = pd.read_csv(input_path, usecols=usecols, encoding=ENCODING, low_memory=False)

    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["is_valid_curve"] = to_numeric_bool(df["is_valid_curve"])
    for col in MAIN_PEAK_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["policy", "cell_code", "cycles"]).copy()
    df["cycles"] = df["cycles"].astype(int)
    df["cell_code"] = df["cell_code"].astype(str)
    return df


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 with zero-variance guard."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_trend_metrics(cycles: np.ndarray, values: np.ndarray) -> TrendMetrics:
    """Compute relation metrics between cycles and one feature series."""
    if len(cycles) == 0:
        return TrendMetrics(
            n_cycles=0,
            cycle_start=float("nan"),
            cycle_end=float("nan"),
            value_start=float("nan"),
            value_end=float("nan"),
            delta_abs=float("nan"),
            delta_pct=float("nan"),
            spearman_rho=float("nan"),
            spearman_pvalue=float("nan"),
            slope_per_cycle=float("nan"),
            slope_per_100_cycles=float("nan"),
            linear_r2=float("nan"),
        )

    order = np.argsort(cycles)
    x = cycles[order].astype(float)
    y = values[order].astype(float)

    n_cycles = int(len(x))
    cycle_start = float(x[0])
    cycle_end = float(x[-1])
    value_start = float(y[0])
    value_end = float(y[-1])
    delta_abs = float(value_end - value_start)
    delta_pct = float("nan") if abs(value_start) <= FLOAT_EPS else float(delta_abs / abs(value_start) * 100.0)

    if len(x) >= 2:
        rho, pvalue = stats.spearmanr(x, y)
        spearman_rho = float(rho) if np.isfinite(rho) else float("nan")
        spearman_pvalue = float(pvalue) if np.isfinite(pvalue) else float("nan")
    else:
        spearman_rho = float("nan")
        spearman_pvalue = float("nan")

    if len(x) >= 2 and float(np.ptp(x)) > FLOAT_EPS:
        slope, intercept = np.polyfit(x, y, deg=1)
        pred = slope * x + intercept
        linear_r2 = safe_r2(y_true=y, y_pred=pred)
        slope_per_cycle = float(slope)
        slope_per_100_cycles = float(slope * 100.0)
    else:
        linear_r2 = float("nan")
        slope_per_cycle = float("nan")
        slope_per_100_cycles = float("nan")

    return TrendMetrics(
        n_cycles=n_cycles,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        value_start=value_start,
        value_end=value_end,
        delta_abs=delta_abs,
        delta_pct=delta_pct,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pvalue,
        slope_per_cycle=slope_per_cycle,
        slope_per_100_cycles=slope_per_100_cycles,
        linear_r2=linear_r2,
    )


def build_long_summary(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Build long-form summary table: one row per combo per feature."""
    rows: List[Dict[str, object]] = []
    group_cols = ["policy", "cell_code"]
    for (policy, cell_code), group in valid_df.groupby(group_cols, sort=True):
        part = group.sort_values("cycles").reset_index(drop=True)
        combo_key = f"{policy}|{cell_code}"
        for feat in MAIN_PEAK_FEATURES:
            feat_part = part[["cycles", feat]].dropna().copy()
            metrics = compute_trend_metrics(
                cycles=feat_part["cycles"].to_numpy(dtype=float),
                values=feat_part[feat].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "policy": policy,
                    "cell_code": str(cell_code),
                    "combo_key": combo_key,
                    "feature": feat,
                    "feature_label": FEATURE_LABEL_MAP.get(feat, feat),
                    "n_cycles": metrics.n_cycles,
                    "cycle_start": metrics.cycle_start,
                    "cycle_end": metrics.cycle_end,
                    "value_start": metrics.value_start,
                    "value_end": metrics.value_end,
                    "delta_abs": metrics.delta_abs,
                    "delta_pct": metrics.delta_pct,
                    "spearman_rho": metrics.spearman_rho,
                    "spearman_pvalue": metrics.spearman_pvalue,
                    "slope_per_cycle": metrics.slope_per_cycle,
                    "slope_per_100_cycles": metrics.slope_per_100_cycles,
                    "linear_r2": metrics.linear_r2,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["policy", "cell_code", "feature"]).reset_index(drop=True)
    return out


def build_wide_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Build wide-form summary table from long-form rows."""
    metric_cols = [
        "n_cycles",
        "cycle_start",
        "cycle_end",
        "value_start",
        "value_end",
        "delta_abs",
        "delta_pct",
        "spearman_rho",
        "spearman_pvalue",
        "slope_per_cycle",
        "slope_per_100_cycles",
        "linear_r2",
    ]
    pivot = long_df.pivot_table(
        index=["policy", "cell_code"],
        columns="feature",
        values=metric_cols,
        aggfunc="first",
    )
    pivot = pivot.sort_index(axis=1)
    pivot.columns = [f"{feat}__{metric}" for metric, feat in pivot.columns]
    out = pivot.reset_index().sort_values(["policy", "cell_code"]).reset_index(drop=True)
    return out


def format_value(value: object, digits: int = 4) -> str:
    """Format numbers for markdown reporting."""
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def make_markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Render a dataframe slice into markdown table lines."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in df.itertuples(index=False):
        vals: List[str] = []
        for col in columns:
            val = getattr(row, col)
            if isinstance(val, (float, np.floating)):
                vals.append(format_value(val, digits=4))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_validation_summary(
    long_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    combo_count: int,
) -> ValidationSummary:
    """Run structural and spot-check validations."""
    expected_rows = combo_count * len(MAIN_PEAK_FEATURES)
    actual_rows = int(len(long_df))

    rho = long_df["spearman_rho"]
    rho_invalid = int(((rho < -1.0) | (rho > 1.0)).fillna(False).sum())

    pval = long_df["spearman_pvalue"]
    pval_invalid = int(((pval < 0.0) | (pval > 1.0)).fillna(False).sum())

    spot_row = (
        long_df.dropna(subset=["delta_abs", "slope_per_cycle", "spearman_rho"])
        .sample(n=1, random_state=RANDOM_SEED)
        .iloc[0]
    )
    policy = str(spot_row["policy"])
    cell_code = str(spot_row["cell_code"])
    feature = str(spot_row["feature"])

    group = (
        valid_df.loc[
            (valid_df["policy"] == policy) & (valid_df["cell_code"].astype(str) == cell_code),
            ["cycles", feature],
        ]
        .dropna()
        .sort_values("cycles")
    )
    recomputed = compute_trend_metrics(
        cycles=group["cycles"].to_numpy(dtype=float),
        values=group[feature].to_numpy(dtype=float),
    )

    diff_delta = abs(float(spot_row["delta_abs"]) - recomputed.delta_abs)
    diff_slope = abs(float(spot_row["slope_per_cycle"]) - recomputed.slope_per_cycle)
    diff_rho = abs(float(spot_row["spearman_rho"]) - recomputed.spearman_rho)
    spot_passed = bool(diff_delta <= 1e-10 and diff_slope <= 1e-10 and diff_rho <= 1e-10)

    return ValidationSummary(
        expected_long_rows=expected_rows,
        actual_long_rows=actual_rows,
        rho_out_of_range_count=rho_invalid,
        pvalue_out_of_range_count=pval_invalid,
        spot_check_passed=spot_passed,
        spot_check_policy=policy,
        spot_check_cell_code=cell_code,
        spot_check_feature=feature,
        spot_check_abs_diff_delta=diff_delta,
        spot_check_abs_diff_slope=diff_slope,
        spot_check_abs_diff_rho=diff_rho,
    )


def plot_overview(long_df: pd.DataFrame, output_path: Path) -> None:
    """Generate overview figure: slope distribution + spearman heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": [1.1, 1.4]})
    ax_box, ax_heat = axes

    slope_data = [
        long_df.loc[long_df["feature"] == feat, "slope_per_100_cycles"].dropna().to_numpy(dtype=float)
        for feat in MAIN_PEAK_FEATURES
    ]
    labels = [FEATURE_LABEL_MAP.get(feat, feat) for feat in MAIN_PEAK_FEATURES]

    ax_box.boxplot(slope_data, tick_labels=labels, showfliers=False)
    ax_box.axhline(0.0, color="#777", linestyle="--", linewidth=1.2)
    ax_box.set_title("Slope Distribution by Feature (per 100 cycles)")
    ax_box.set_ylabel("slope_per_100_cycles")
    ax_box.tick_params(axis="x", rotation=35)
    ax_box.grid(axis="y", alpha=0.25, linestyle="--")

    heat = long_df.pivot_table(
        index="combo_key",
        columns="feature",
        values="spearman_rho",
        aggfunc="first",
    )
    heat = heat.reindex(columns=MAIN_PEAK_FEATURES)
    heat = heat.assign(_mean=heat.mean(axis=1)).sort_values("_mean", ascending=False).drop(columns="_mean")
    heat_values = heat.to_numpy(dtype=float)

    im = ax_heat.imshow(heat_values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax_heat.set_title("Spearman Rho Heatmap by Combo and Feature")
    ax_heat.set_xlabel("feature")
    ax_heat.set_ylabel("policy|cell_code (sorted by mean rho)")
    ax_heat.set_xticks(np.arange(len(MAIN_PEAK_FEATURES)))
    ax_heat.set_xticklabels(labels, rotation=35, ha="right")

    n_rows = heat_values.shape[0]
    if n_rows <= 30:
        y_ticks = np.arange(n_rows)
    else:
        step = max(1, math.ceil(n_rows / 20))
        y_ticks = np.arange(0, n_rows, step)
    ax_heat.set_yticks(y_ticks)
    ax_heat.set_yticklabels([heat.index[i] for i in y_ticks], fontsize=8)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("spearman_rho")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    input_rows: int,
    valid_rows: int,
    combo_count: int,
    cycles_per_combo: pd.Series,
    long_df: pd.DataFrame,
    checks: ValidationSummary,
    output_dir: Path,
) -> None:
    """Write markdown report with coverage and trend highlights."""
    lines: List[str] = []
    valid_ratio = valid_rows / max(input_rows, 1)

    lines.append("# dQ/dV Main Peak vs Cycle Relation Report")
    lines.append("")
    lines.append("## 1) Data Coverage")
    lines.append("")
    lines.append(f"- input_rows: **{input_rows}**")
    lines.append(f"- valid_rows (`is_valid_curve=True`): **{valid_rows}**")
    lines.append(f"- valid_ratio: **{valid_ratio:.4%}**")
    lines.append(f"- combo_count (`policy+cell_code`): **{combo_count}**")
    lines.append(
        f"- cycles_per_combo (min/p50/p90/max): "
        f"**{int(cycles_per_combo.min())} / {float(cycles_per_combo.quantile(0.5)):.1f} / "
        f"{float(cycles_per_combo.quantile(0.9)):.1f} / {int(cycles_per_combo.max())}**"
    )
    lines.append("")
    lines.append("## 2) Validation Checks")
    lines.append("")
    lines.append(
        f"- expected_long_rows = combo_count x feature_count = "
        f"**{checks.expected_long_rows}**, actual = **{checks.actual_long_rows}**"
    )
    lines.append(f"- spearman_rho out-of-range count: **{checks.rho_out_of_range_count}**")
    lines.append(f"- spearman_pvalue out-of-range count: **{checks.pvalue_out_of_range_count}**")
    lines.append(
        f"- spot_check passed: **{checks.spot_check_passed}** "
        f"(policy={checks.spot_check_policy}, cell={checks.spot_check_cell_code}, feature={checks.spot_check_feature})"
    )
    lines.append(
        f"- spot_check abs diff (delta/slope/rho): "
        f"**{checks.spot_check_abs_diff_delta:.3e} / {checks.spot_check_abs_diff_slope:.3e} / {checks.spot_check_abs_diff_rho:.3e}**"
    )
    lines.append("")
    lines.append("## 3) Top Rising and Falling Combos by Feature")
    lines.append("")

    for feat in MAIN_PEAK_FEATURES:
        feat_df = long_df.loc[long_df["feature"] == feat].copy()
        rise = feat_df.sort_values("slope_per_100_cycles", ascending=False).head(REPORT_TOP_N).copy()
        fall = feat_df.sort_values("slope_per_100_cycles", ascending=True).head(REPORT_TOP_N).copy()
        keep_cols = [
            "policy",
            "cell_code",
            "slope_per_100_cycles",
            "spearman_rho",
            "delta_abs",
            "delta_pct",
            "n_cycles",
        ]

        lines.append(f"### Feature: `{feat}`")
        lines.append("")
        lines.append("Top Rising:")
        lines.extend(make_markdown_table(rise[keep_cols], keep_cols))
        lines.append("")
        lines.append("Top Falling:")
        lines.extend(make_markdown_table(fall[keep_cols], keep_cols))
        lines.append("")

    lines.append("## 4) Spearman Distribution Summary by Feature")
    lines.append("")
    dist_rows: List[Dict[str, object]] = []
    for feat in MAIN_PEAK_FEATURES:
        s = long_df.loc[long_df["feature"] == feat, "spearman_rho"].dropna()
        dist_rows.append(
            {
                "feature": feat,
                "median_rho": float(s.median()) if len(s) > 0 else float("nan"),
                "q1_rho": float(s.quantile(0.25)) if len(s) > 0 else float("nan"),
                "q3_rho": float(s.quantile(0.75)) if len(s) > 0 else float("nan"),
                "iqr_rho": float(s.quantile(0.75) - s.quantile(0.25)) if len(s) > 0 else float("nan"),
            }
        )
    dist_df = pd.DataFrame(dist_rows)
    dist_cols = ["feature", "median_rho", "q1_rho", "q3_rho", "iqr_rho"]
    lines.extend(make_markdown_table(dist_df[dist_cols], dist_cols))
    lines.append("")

    lines.append("## 5) Output Files")
    lines.append("")
    lines.append(f"- `{output_dir / OUTPUT_LONG_CSV}`")
    lines.append(f"- `{output_dir / OUTPUT_WIDE_CSV}`")
    lines.append(f"- `{output_dir / OUTPUT_REPORT_MD}`")
    lines.append(f"- `{output_dir / OUTPUT_OVERVIEW_PNG}`")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run full summarization pipeline for main-peak cycle relations."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = load_input_table(input_path)
    input_rows = int(len(df))
    valid_df = df.loc[df["is_valid_curve"].astype(bool)].copy()
    valid_df = valid_df.sort_values(["policy", "cell_code", "cycles"]).reset_index(drop=True)
    valid_rows = int(len(valid_df))

    combo_count = int(valid_df[["policy", "cell_code"]].drop_duplicates().shape[0])
    cycles_per_combo = valid_df.groupby(["policy", "cell_code"]).size()

    long_df = build_long_summary(valid_df)
    wide_df = build_wide_summary(long_df)
    checks = build_validation_summary(long_df=long_df, valid_df=valid_df, combo_count=combo_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / OUTPUT_LONG_CSV
    wide_path = output_dir / OUTPUT_WIDE_CSV
    report_path = output_dir / OUTPUT_REPORT_MD
    fig_path = output_dir / OUTPUT_OVERVIEW_PNG

    long_df.to_csv(long_path, index=False, encoding="utf-8")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8")
    plot_overview(long_df=long_df, output_path=fig_path)
    write_report(
        report_path=report_path,
        input_rows=input_rows,
        valid_rows=valid_rows,
        combo_count=combo_count,
        cycles_per_combo=cycles_per_combo,
        long_df=long_df,
        checks=checks,
        output_dir=output_dir,
    )

    print(f"input_rows={input_rows}")
    print(f"valid_rows={valid_rows}")
    print(f"combo_count={combo_count}")
    print(f"long_rows={len(long_df)}")
    print(f"wide_rows={len(wide_df)}")
    print(f"long_csv={long_path}")
    print(f"wide_csv={wide_path}")
    print(f"report_md={report_path}")
    print(f"overview_png={fig_path}")
    print(f"validation_expected_long_rows={checks.expected_long_rows}")
    print(f"validation_actual_long_rows={checks.actual_long_rows}")
    print(f"validation_rho_out_of_range={checks.rho_out_of_range_count}")
    print(f"validation_pvalue_out_of_range={checks.pvalue_out_of_range_count}")
    print(f"validation_spot_check_passed={checks.spot_check_passed}")


if __name__ == "__main__":
    main()
