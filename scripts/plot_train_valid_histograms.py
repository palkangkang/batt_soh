from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Config (edit here first)
# =========================
TRAIN_CSV = PROJECT_ROOT / "data/processed/train_policy_cell_samples.csv"
VALID_CSV = PROJECT_ROOT / "data/processed/valid_policy_cell_samples.csv"

OUT_FIG = PROJECT_ROOT / "data/processed/train_valid_hist_compare.png"
OUT_STATS = PROJECT_ROOT / "data/processed/train_valid_hist_stats.csv"

FEATURE_CONFIG: Dict[str, Dict[str, float | str]] = {
    "initial_c_rate": {"bin_width": 0.5, "title": "Initial C-rate"},
    "switch_soc_percent": {"bin_width": 5.0, "title": "Switch SOC (%)"},
    "post_switch_c_rate": {"bin_width": 0.25, "title": "Post-switch C-rate"},
    "max_cycles": {"bin_width": 100.0, "title": "Max Cycles"},
}


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_bins(train_vals: pd.Series, valid_vals: pd.Series, bin_width: float) -> np.ndarray:
    combined = pd.concat([train_vals.dropna(), valid_vals.dropna()], ignore_index=True)
    if combined.empty:
        return np.array([0.0, 1.0], dtype=float)

    min_v = combined.min()
    max_v = combined.max()
    left = np.floor(min_v / bin_width) * bin_width
    right = np.ceil(max_v / bin_width) * bin_width
    if np.isclose(left, right):
        right = left + bin_width
    bins = np.arange(left, right + bin_width, bin_width, dtype=float)
    return bins


def pct_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    return np.ones(n) * (100.0 / n)


def describe(values: pd.Series) -> dict:
    non_na = values.dropna()
    if non_na.empty:
        return {
            "n": 0,
            "missing": int(values.isna().sum()),
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(non_na.shape[0]),
        "missing": int(values.isna().sum()),
        "mean": float(non_na.mean()),
        "median": float(non_na.median()),
        "std": float(non_na.std(ddof=1)) if non_na.shape[0] > 1 else 0.0,
        "min": float(non_na.min()),
        "max": float(non_na.max()),
    }


def main() -> None:
    df_train = pd.read_csv(TRAIN_CSV)
    df_valid = pd.read_csv(VALID_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    stats_rows = []
    for idx, (feature, cfg) in enumerate(FEATURE_CONFIG.items()):
        ax = axes[idx]
        train_vals = to_numeric(df_train[feature])
        valid_vals = to_numeric(df_valid[feature])
        bins = build_bins(train_vals, valid_vals, float(cfg["bin_width"]))

        train_non_na = train_vals.dropna()
        valid_non_na = valid_vals.dropna()

        if not train_non_na.empty:
            ax.hist(
                train_non_na,
                bins=bins,
                weights=pct_weights(train_non_na.shape[0]),
                alpha=0.60,
                color="#4e79a7",
                edgecolor="white",
                label="Train (%)",
            )
        if not valid_non_na.empty:
            ax.hist(
                valid_non_na,
                bins=bins,
                weights=pct_weights(valid_non_na.shape[0]),
                alpha=0.60,
                color="#f28e2b",
                edgecolor="white",
                label="Valid (%)",
            )

        train_desc = describe(train_vals)
        valid_desc = describe(valid_vals)
        ax.set_title(
            f"{cfg['title']}\n"
            f"train n={train_desc['n']}, miss={train_desc['missing']} | "
            f"valid n={valid_desc['n']}, miss={valid_desc['missing']}",
            fontsize=10,
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Percentage (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        stats_rows.append({"feature": feature, "split": "train", **train_desc})
        stats_rows.append({"feature": feature, "split": "valid", **valid_desc})

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Train vs Valid Histogram Comparison", fontsize=14, y=1.02)
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_STATS, index=False, encoding="utf-8")

    print(f"Saved figure: {OUT_FIG}")
    print(f"Saved stats: {OUT_STATS}")


if __name__ == "__main__":
    main()
