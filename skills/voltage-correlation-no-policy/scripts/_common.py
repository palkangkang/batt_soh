from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

MPL_CONFIG_DIR: Path | None = None


def _ensure_mpl_config_dir(repo_root: Path) -> None:
    global MPL_CONFIG_DIR
    if MPL_CONFIG_DIR is not None:
        return
    MPL_CONFIG_DIR = repo_root / "outputs" / ".mplconfig"
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def setup_fonts(repo_root: Path) -> Tuple[List[str], bool]:
    _ensure_mpl_config_dir(repo_root)
    from matplotlib import font_manager, rcParams  # local import after MPLCONFIGDIR

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected = [f for f in candidates if f in installed]
    has_cjk = len(selected) > 0
    if not selected:
        selected = ["DejaVu Sans"]

    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    return selected, has_cjk


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


def safe_corr(method: Callable, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    try:
        out = method(x, y)
        r = float(out[0])
        p = float(out[1])
        if np.isfinite(r) and np.isfinite(p):
            return r, p
    except Exception:
        pass
    return float("nan"), float("nan")


def ensure_required_columns(columns: Sequence[str], required: Sequence[str], file_label: str) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"{file_label} missing required columns: {missing}")


def load_life_performance_with_alias(
    life_path: Path,
    encoding: str = "utf-8-sig",
    allow_policy_key: bool = True,
) -> pd.DataFrame:
    target_candidates = ["q_discharge", "Q_dischg", "q_dischg"]
    header = pd.read_csv(life_path, nrows=0, encoding=encoding)
    cols = header.columns.tolist()
    ensure_required_columns(cols, ["cell_code", "cycles"], "life_performance")

    present_targets = [c for c in target_candidates if c in cols]
    if len(present_targets) == 0:
        raise ValueError(
            "life_performance must contain exactly one target column among "
            "['q_discharge','Q_dischg','q_dischg'], but found none."
        )
    if len(present_targets) > 1:
        raise ValueError(
            "life_performance must contain exactly one target column among "
            "['q_discharge','Q_dischg','q_dischg'], but found multiple: "
            f"{present_targets}"
        )

    target_col = present_targets[0]
    usecols = ["cell_code", "cycles", target_col]
    has_policy_col = allow_policy_key and ("policy" in cols)
    if has_policy_col:
        usecols = ["policy"] + usecols

    df = pd.read_csv(life_path, encoding=encoding, usecols=usecols)
    if target_col != "q_discharge":
        df = df.rename(columns={target_col: "q_discharge"})

    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["q_discharge"] = pd.to_numeric(df["q_discharge"], errors="coerce")
    base_subset = ["cell_code", "cycles", "q_discharge"]
    if has_policy_col:
        base_subset = ["policy"] + base_subset
    df = df.dropna(subset=base_subset).copy()
    if df.empty:
        raise ValueError("life_performance has no valid rows after parsing cycles/q_discharge.")

    df["cycles"] = df["cycles"].astype(int)

    uniq_keys = ["cell_code", "cycles"]
    if has_policy_col:
        uniq_keys = ["policy"] + uniq_keys
    dup = df.duplicated(subset=uniq_keys, keep=False)
    if dup.any():
        dup_count = int(dup.sum())
        raise ValueError(
            "life_performance requires unique key "
            f"'{'+'.join(uniq_keys)}'. "
            f"Found duplicated rows: {dup_count}"
        )

    keep_cols = uniq_keys + ["q_discharge"]
    return df[keep_cols]


def load_interval_features_first_occurrence(
    feature_path: Path,
    first_occurrence: int = 1,
    encoding: str = "utf-8-sig",
    allow_policy_key: bool = True,
) -> pd.DataFrame:
    required = ["cell_code", "cycles", "range", "delta_ah", "range_count"]
    header = pd.read_csv(feature_path, nrows=0, encoding=encoding)
    cols = header.columns.tolist()
    ensure_required_columns(cols, required, feature_path.name)

    usecols = required.copy()
    has_policy_col = allow_policy_key and ("policy" in cols)
    if has_policy_col:
        usecols = ["policy"] + usecols

    df = pd.read_csv(feature_path, encoding=encoding, usecols=usecols)
    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["delta_ah"] = pd.to_numeric(df["delta_ah"], errors="coerce")
    df["range_count"] = pd.to_numeric(df["range_count"], errors="coerce")
    drop_subset = required.copy()
    if has_policy_col:
        drop_subset = ["policy"] + drop_subset
    df = df.dropna(subset=drop_subset).copy()
    if df.empty:
        raise ValueError(f"{feature_path.name} has no valid rows after parsing required fields.")

    df["cycles"] = df["cycles"].astype(int)
    df["range_count"] = df["range_count"].astype(int)

    df = df.loc[df["range_count"] == int(first_occurrence)].copy()
    if df.empty:
        raise ValueError(
            f"{feature_path.name} has no rows where range_count == {int(first_occurrence)}."
        )

    group_keys = ["cell_code", "cycles", "range"]
    if has_policy_col:
        group_keys = ["policy"] + group_keys
    agg = df.groupby(group_keys, as_index=False).agg(delta_ah_sum=("delta_ah", "sum"))
    if agg.empty:
        raise ValueError(f"{feature_path.name} aggregation result is empty.")
    return agg


def build_wide_features(
    agg: pd.DataFrame,
    prefix: str,
    reverse: bool = False,
    key_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    ranges = sorted(agg["range"].dropna().unique().tolist(), key=parse_range_low, reverse=reverse)
    agg2 = agg.copy()
    agg2["feature_name"] = agg2["range"].map(lambda r: f"{prefix}_{sanitize_range_label(r)}")

    if key_cols is None:
        key_cols = ["cell_code", "cycles"]
    wide = agg2.pivot_table(
        index=key_cols,
        columns="feature_name",
        values="delta_ah_sum",
        aggfunc="mean",
    ).reset_index()
    cols = [f"{prefix}_{sanitize_range_label(r)}" for r in ranges]
    cols = [c for c in cols if c in wide.columns]
    wide = wide[key_cols + cols]
    label_map = {f"{prefix}_{sanitize_range_label(r)}": str(r) for r in ranges}
    return wide, cols, label_map


def choose_merge_keys(left_cols: Sequence[str], right_cols: Sequence[str]) -> list[str]:
    base = ["cell_code", "cycles"]
    if "policy" in left_cols and "policy" in right_cols:
        return ["policy"] + base
    return base
