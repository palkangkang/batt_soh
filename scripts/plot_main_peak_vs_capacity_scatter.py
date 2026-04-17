from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# =========================
# Config
# =========================
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_DQDV_INPUT = REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv"
DEFAULT_CAPACITY_INPUT = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "dqdv_main_peak_capacity_scatter"

DEFAULT_POLICIES_PER_PAGE = 12
DEFAULT_DPI = 220

ENCODING = "utf-8-sig"
SUBPLOT_ROWS = 4
SUBPLOT_COLS = 3
POINT_ALPHA = 0.55
POINT_SIZE = 8
MAX_LEGEND_CELLS = 8

REQUIRED_SOURCE_COLUMNS: List[str] = [
    "policy",
    "cell_code",
    "cycles",
    "q_discharge",
    "main_peak_voltage_v",
    "main_peak_skewness",
]


MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402


@dataclass(frozen=True)
class MetricPlotSpec:
    """One plotting target."""

    y_col: str
    y_label: str
    filename_prefix: str


@dataclass
class PolicyPointSummary:
    """Point summary for one policy and one metric."""

    n_points: int
    n_cells: int


@dataclass
class PageMeta:
    """Metadata for one output page."""

    page_idx: int
    file_name: str
    policies: List[str]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot paged scatter of main-peak voltage/skewness vs q_discharge by policy."
    )
    parser.add_argument(
        "--dqdv-input",
        type=str,
        default=str(DEFAULT_DQDV_INPUT),
        help="Input dQdV feature CSV path.",
    )
    parser.add_argument(
        "--capacity-input",
        type=str,
        default=str(DEFAULT_CAPACITY_INPUT),
        help="Input life_performance CSV path with q_discharge.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write scatter artifacts.",
    )
    parser.add_argument(
        "--policies-per-page",
        type=int,
        default=DEFAULT_POLICIES_PER_PAGE,
        help="Number of policies per page. Default is 12 (4x3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PNG DPI.",
    )
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    """Normalize mixed bool-like values to strict bool."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def load_dqdv_table(path: Path) -> pd.DataFrame:
    """Load dQdV feature table using required columns."""
    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "is_valid_curve",
        "main_peak_voltage_v",
        "main_peak_skewness",
    ]
    df = pd.read_csv(path, usecols=usecols, encoding=ENCODING, low_memory=False)
    df["policy"] = df["policy"].astype(str)
    df["cell_code"] = df["cell_code"].astype(str)
    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["is_valid_curve"] = to_bool(df["is_valid_curve"])
    df["main_peak_voltage_v"] = pd.to_numeric(df["main_peak_voltage_v"], errors="coerce")
    df["main_peak_skewness"] = pd.to_numeric(df["main_peak_skewness"], errors="coerce")
    df = df.dropna(subset=["policy", "cell_code", "cycles"]).copy()
    df["cycles"] = df["cycles"].astype(int)
    return df


def load_capacity_table(path: Path) -> pd.DataFrame:
    """Load capacity table with q_discharge."""
    usecols = ["policy", "cell_code", "cycles", "q_discharge"]
    df = pd.read_csv(path, usecols=usecols, encoding=ENCODING, low_memory=False)
    df["policy"] = df["policy"].astype(str)
    df["cell_code"] = df["cell_code"].astype(str)
    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["q_discharge"] = pd.to_numeric(df["q_discharge"], errors="coerce")
    df = df.dropna(subset=["policy", "cell_code", "cycles"]).copy()
    df["cycles"] = df["cycles"].astype(int)
    return df


def build_source_data(dqdv_df: pd.DataFrame, cap_df: pd.DataFrame) -> pd.DataFrame:
    """Filter valid curves and build merged source data."""
    valid_dqdv = dqdv_df.loc[dqdv_df["is_valid_curve"]].copy()
    merged = valid_dqdv.merge(
        cap_df,
        how="inner",
        on=["policy", "cell_code", "cycles"],
        validate="one_to_one",
    )
    merged = merged[
        ["policy", "cell_code", "cycles", "q_discharge", "main_peak_voltage_v", "main_peak_skewness"]
    ].copy()
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="stable").reset_index(drop=True)
    return merged


def split_pages(items: Sequence[str], page_size: int) -> List[List[str]]:
    """Split ordered items into pages."""
    return [list(items[i : i + page_size]) for i in range(0, len(items), page_size)]


def build_policy_color_maps(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build per-policy stable cell_code color maps."""
    color_maps: Dict[str, Dict[str, str]] = {}
    for policy, group in df.groupby("policy", sort=True):
        cell_codes = sorted(group["cell_code"].astype(str).unique().tolist())
        if len(cell_codes) <= 20:
            cmap = plt.get_cmap("tab20", max(len(cell_codes), 1))
        else:
            cmap = plt.get_cmap("nipy_spectral", len(cell_codes))
        mapping = {cell: mcolors.to_hex(cmap(i)) for i, cell in enumerate(cell_codes)}
        color_maps[str(policy)] = mapping
    return color_maps


def plot_metric_pages(
    df: pd.DataFrame,
    output_dir: Path,
    policies: Sequence[str],
    color_maps: Dict[str, Dict[str, str]],
    spec: MetricPlotSpec,
    policies_per_page: int,
    dpi: int,
) -> Tuple[List[PageMeta], Dict[str, PolicyPointSummary]]:
    """Render paged scatter PNGs for one metric."""
    page_policies = split_pages(policies, policies_per_page)
    policy_groups: Dict[str, pd.DataFrame] = {
        str(policy): sub_df.copy() for policy, sub_df in df.groupby("policy", sort=False)
    }
    page_meta: List[PageMeta] = []
    point_summary: Dict[str, PolicyPointSummary] = {}

    rows = SUBPLOT_ROWS if policies_per_page == SUBPLOT_ROWS * SUBPLOT_COLS else int(
        math.ceil(policies_per_page / SUBPLOT_COLS)
    )
    cols = SUBPLOT_COLS
    total_axes = rows * cols

    for page_idx, policy_list in enumerate(page_policies, start=1):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 5.2, rows * 3.9),
            dpi=dpi,
            constrained_layout=False,
        )
        axes_array = np.array(axes).reshape(-1)

        for ax_idx, policy in enumerate(policy_list):
            ax = axes_array[ax_idx]
            data = policy_groups.get(policy, pd.DataFrame(columns=df.columns))
            data = data.dropna(subset=["q_discharge", spec.y_col]).copy()

            unique_cells = sorted(data["cell_code"].astype(str).unique().tolist())
            n_cells = len(unique_cells)
            n_points = int(len(data))
            point_summary[policy] = PolicyPointSummary(n_points=n_points, n_cells=n_cells)

            color_map = color_maps.get(policy, {})
            if n_points > 0:
                for cell_code in unique_cells:
                    cell_data = data.loc[data["cell_code"] == cell_code]
                    ax.scatter(
                        cell_data["q_discharge"].to_numpy(),
                        cell_data[spec.y_col].to_numpy(),
                        s=POINT_SIZE,
                        alpha=POINT_ALPHA,
                        linewidths=0,
                        color=color_map.get(cell_code, "#1f77b4"),
                        label=cell_code if n_cells <= MAX_LEGEND_CELLS else None,
                    )
            else:
                ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes, fontsize=9)

            ax.set_title(f"{policy} | n_cells={n_cells} | n_points={n_points}", fontsize=9)
            ax.set_xlabel("q_discharge", fontsize=8)
            ax.set_ylabel(spec.y_label, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.2, linewidth=0.5)

            if 0 < n_cells <= MAX_LEGEND_CELLS:
                ax.legend(loc="best", fontsize=6, frameon=False, ncol=1)
            elif n_cells > MAX_LEGEND_CELLS:
                ax.text(
                    0.03,
                    0.97,
                    "Legend hidden (n_cells>8)",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="#666666",
                )

        for ax_idx in range(len(policy_list), total_axes):
            axes_array[ax_idx].set_visible(False)

        fig.suptitle(
            f"{spec.y_label} vs q_discharge by policy (page {page_idx}/{len(page_policies)})",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        file_name = f"{spec.filename_prefix}_page_{page_idx:02d}.png"
        out_path = output_dir / file_name
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

        page_meta.append(PageMeta(page_idx=page_idx, file_name=file_name, policies=list(policy_list)))

    return page_meta, point_summary


def write_index_markdown(
    path: Path,
    dqdv_path: Path,
    capacity_path: Path,
    output_dir: Path,
    source_df: pd.DataFrame,
    policies: Sequence[str],
    metric_page_map: Dict[str, List[PageMeta]],
    metric_point_summary: Dict[str, Dict[str, PolicyPointSummary]],
    policies_per_page: int,
) -> None:
    """Write index markdown containing coverage and page mapping."""
    combo_count = int(source_df[["policy", "cell_code"]].drop_duplicates().shape[0])
    lines: List[str] = []
    lines.append("# Main Peak vs Capacity Scatter Index")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- dQdV input: `{dqdv_path}`")
    lines.append(f"- Capacity input: `{capacity_path}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Source rows: {len(source_df)}")
    lines.append(f"- Policy count: {len(policies)}")
    lines.append(f"- Policy+cell_code combos: {combo_count}")
    lines.append(f"- Policies per page: {policies_per_page}")
    lines.append("- Legend rule: show legend only when `n_cells <= 8`; otherwise hidden in subplot.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `scatter_source_data.csv`")
    for key in ["main_peak_voltage_v", "main_peak_skewness"]:
        for page in metric_page_map[key]:
            lines.append(f"- `{page.file_name}`")
    lines.append("- `scatter_index.md`")
    lines.append("")

    for metric_key, title in [
        ("main_peak_voltage_v", "main_peak_voltage_v Pages"),
        ("main_peak_skewness", "main_peak_skewness Pages"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Page | File | Policies |")
        lines.append("| --- | --- | --- |")
        for meta in metric_page_map[metric_key]:
            lines.append(
                f"| {meta.page_idx} | `{meta.file_name}` | {', '.join(meta.policies)} |"
            )
        lines.append("")

    lines.append("## Policy Point Summary")
    lines.append("")
    lines.append("| policy | n_points_voltage | n_points_skewness | n_cells |")
    lines.append("| --- | --- | --- | --- |")
    for policy in policies:
        voltage_summary = metric_point_summary["main_peak_voltage_v"][policy]
        skew_summary = metric_point_summary["main_peak_skewness"][policy]
        n_cells = max(voltage_summary.n_cells, skew_summary.n_cells)
        lines.append(
            f"| {policy} | {voltage_summary.n_points} | {skew_summary.n_points} | {n_cells} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_checks(
    source_df: pd.DataFrame,
    policies: Sequence[str],
    metric_page_map: Dict[str, List[PageMeta]],
    metric_point_summary: Dict[str, Dict[str, PolicyPointSummary]],
) -> None:
    """Run structural and coverage checks."""
    if source_df.empty:
        raise RuntimeError("scatter_source_data.csv would be empty.")

    missing_cols = [col for col in REQUIRED_SOURCE_COLUMNS if col not in source_df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required source columns: {missing_cols}")

    voltage_pages = metric_page_map["main_peak_voltage_v"]
    skew_pages = metric_page_map["main_peak_skewness"]
    if len(voltage_pages) != len(skew_pages):
        raise RuntimeError("Two metric groups have different page counts.")

    expected_policy_set = set(policies)
    for metric_key, pages in metric_page_map.items():
        page_policy_list = [policy for page in pages for policy in page.policies]
        if len(page_policy_list) != len(expected_policy_set):
            raise RuntimeError(f"{metric_key} does not contain expected policy count.")
        if set(page_policy_list) != expected_policy_set:
            raise RuntimeError(f"{metric_key} policy coverage mismatch.")
        if len(page_policy_list) != len(set(page_policy_list)):
            raise RuntimeError(f"{metric_key} has duplicated policies across pages.")

    for policy in policies:
        source_count = int(
            source_df.loc[source_df["policy"] == policy, ["q_discharge", "main_peak_voltage_v"]]
            .dropna()
            .shape[0]
        )
        if source_count != metric_point_summary["main_peak_voltage_v"][policy].n_points:
            raise RuntimeError(f"Point count mismatch for policy={policy} in voltage summary.")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    dqdv_path = Path(args.dqdv_input).resolve()
    capacity_path = Path(args.capacity_input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.policies_per_page <= 0:
        raise ValueError("--policies-per-page must be positive.")

    dqdv_df = load_dqdv_table(dqdv_path)
    cap_df = load_capacity_table(capacity_path)
    source_df = build_source_data(dqdv_df, cap_df)
    source_output_path = output_dir / "scatter_source_data.csv"
    source_df.to_csv(source_output_path, index=False, encoding=ENCODING)

    policies = sorted(source_df["policy"].astype(str).unique().tolist())
    color_maps = build_policy_color_maps(source_df)

    metric_specs: List[MetricPlotSpec] = [
        MetricPlotSpec(
            y_col="main_peak_voltage_v",
            y_label="main_peak_voltage_v",
            filename_prefix="main_peak_voltage_vs_capacity",
        ),
        MetricPlotSpec(
            y_col="main_peak_skewness",
            y_label="main_peak_skewness",
            filename_prefix="main_peak_skewness_vs_capacity",
        ),
    ]

    metric_page_map: Dict[str, List[PageMeta]] = {}
    metric_point_summary: Dict[str, Dict[str, PolicyPointSummary]] = {}

    for spec in metric_specs:
        page_meta, point_summary = plot_metric_pages(
            df=source_df,
            output_dir=output_dir,
            policies=policies,
            color_maps=color_maps,
            spec=spec,
            policies_per_page=args.policies_per_page,
            dpi=args.dpi,
        )
        metric_page_map[spec.y_col] = page_meta
        metric_point_summary[spec.y_col] = point_summary

    run_checks(
        source_df=source_df,
        policies=policies,
        metric_page_map=metric_page_map,
        metric_point_summary=metric_point_summary,
    )

    index_path = output_dir / "scatter_index.md"
    write_index_markdown(
        path=index_path,
        dqdv_path=dqdv_path,
        capacity_path=capacity_path,
        output_dir=output_dir,
        source_df=source_df,
        policies=policies,
        metric_page_map=metric_page_map,
        metric_point_summary=metric_point_summary,
        policies_per_page=args.policies_per_page,
    )

    print(f"[done] source_rows={len(source_df)} policies={len(policies)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
