"""Create policy-cell splits for long-life and policy-family extrapolation tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from train_interval_to_dqdv_retention_pipeline import ENCODING, REPO_ROOT


KEY_COLS = ["policy", "cell_code"]
SPLIT_COLUMNS = [
    "policy",
    "cell_code",
    "initial_c_rate",
    "switch_soc_percent",
    "post_switch_c_rate",
    "max_cycles",
    "policy_note",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Create supplementary extrapolation train/valid splits without replacing the balanced split."
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "life_performance.csv",
    )
    parser.add_argument(
        "--policy-meaning-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "policy_meaning.csv",
    )
    parser.add_argument(
        "--balanced-train-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv",
    )
    parser.add_argument(
        "--balanced-valid-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "extrapolation_splits",
    )
    parser.add_argument("--long-life-threshold", type=int, default=1000)
    parser.add_argument(
        "--policy-family-mode",
        choices=["variable_charge", "high_initial_rate"],
        default="high_initial_rate",
    )
    parser.add_argument("--high-initial-rate-threshold", type=float, default=5.0)
    return parser.parse_args()


def load_policy_cell_samples(life_path: Path, policy_path: Path) -> pd.DataFrame:
    """Build one policy-cell sample table with max cycle count and policy metadata."""

    life = pd.read_csv(life_path, encoding=ENCODING)
    required = {"policy", "cell_code", "cycles"}
    missing = required.difference(life.columns)
    if missing:
        raise KeyError(f"Missing required life columns: {sorted(missing)}")
    life = life.loc[:, ["policy", "cell_code", "cycles"]].copy()
    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life = life.dropna(subset=["policy", "cell_code", "cycles"])
    samples = (
        life.groupby(KEY_COLS, as_index=False)["cycles"]
        .max()
        .rename(columns={"cycles": "max_cycles"})
    )
    samples["max_cycles"] = samples["max_cycles"].astype(int)

    policy = pd.read_csv(policy_path, encoding=ENCODING)
    for col in ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate", "note"]:
        if col not in policy.columns:
            policy[col] = ""
    policy_meta = policy.loc[
        :, ["policy", "initial_c_rate", "switch_soc_percent", "post_switch_c_rate", "note"]
    ].rename(columns={"note": "policy_note"})
    merged = samples.merge(policy_meta, on="policy", how="left")
    for col in ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate", "policy_note"]:
        merged[col] = merged[col].fillna("")
    return merged.loc[:, SPLIT_COLUMNS].sort_values(KEY_COLS, kind="mergesort").reset_index(drop=True)


def key_set(frame: pd.DataFrame) -> set[Tuple[str, str]]:
    """Return a set of policy-cell keys."""

    return set(frame.loc[:, KEY_COLS].astype(str).apply(tuple, axis=1))


def write_split(train: pd.DataFrame, valid: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Write train and valid split CSV files for one split name."""

    train_path = out_dir / f"train_policy_cell_samples_{name}.csv"
    valid_path = out_dir / f"valid_policy_cell_samples_{name}.csv"
    train.loc[:, SPLIT_COLUMNS].to_csv(train_path, index=False, encoding=ENCODING)
    valid.loc[:, SPLIT_COLUMNS].to_csv(valid_path, index=False, encoding=ENCODING)


def describe_values(values: Sequence[int]) -> Dict[str, float]:
    """Return compact distribution statistics for max cycle counts."""

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "min": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "ge800": 0,
            "ge1000": 0,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "ge800": int(np.sum(arr >= 800)),
        "ge1000": int(np.sum(arr >= 1000)),
    }


def make_policy_family_mask(samples: pd.DataFrame, mode: str, high_rate: float) -> pd.Series:
    """Return the valid-family mask for a policy-family holdout."""

    if mode == "variable_charge":
        note = samples["policy_note"].astype(str)
        policy = samples["policy"].astype(str)
        return note.str.contains("variable_charge", case=False, na=False) | policy.str.startswith("VARCHARGE_")
    initial = pd.to_numeric(samples["initial_c_rate"], errors="coerce")
    return initial >= float(high_rate)


def build_summary_rows(split_name: str, train: pd.DataFrame, valid: pd.DataFrame) -> List[Dict[str, object]]:
    """Build summary rows for one split."""

    rows: List[Dict[str, object]] = []
    overlap = len(key_set(train).intersection(key_set(valid)))
    for set_type, frame in [("train", train), ("valid", valid)]:
        stats = describe_values(frame["max_cycles"].astype(int).tolist())
        rows.append(
            {
                "split": split_name,
                "set_type": set_type,
                "n_policy_cell": int(len(frame)),
                "n_policy": int(frame["policy"].nunique()),
                "max_cycles_min": stats["min"],
                "max_cycles_q25": stats["q25"],
                "max_cycles_median": stats["median"],
                "max_cycles_q75": stats["q75"],
                "max_cycles_max": stats["max"],
                "max_cycles_mean": stats["mean"],
                "count_ge800": stats["ge800"],
                "count_ge1000": stats["ge1000"],
                "split_overlap_count": int(overlap),
            }
        )
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    """Render a dataframe as a Markdown table."""

    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _idx, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def build_report(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    """Build a Chinese Markdown report explaining the split purpose and diagnostics."""

    lines = [
        "# 外推评估划分报告",
        "",
        "## 1. 目的",
        "",
        "本报告生成补充的 train/valid 划分，用于检验模型是否具备长寿命外推、寿命阶段外推或工况策略族外推能力。它不替换原有 balanced split。",
        "",
        "## 2. 术语说明",
        "",
        "- `balanced`：当前仓库已有的分层覆盖型划分，训练集和验证集都覆盖不同寿命与工况策略。",
        "- `long_life_holdout`：把 `max_cycles` 大于等于阈值的长寿命电芯放入验证集，用于检验长寿命外推。",
        "- `policy_family_holdout`：按策略族留出验证集，用于检验未见工况策略族上的泛化。",
        "- `max_cycles`：同一 `policy + cell_code` 电芯样本可观测到的最大循环数，只用于划分和报告，不作为模型输入。",
        "- `split_overlap_count`：训练集与验证集重叠的 `policy + cell_code` 数量，电芯级划分应为 0。",
        "",
        "## 3. 参数",
        "",
        f"- long_life_threshold: `{int(args.long_life_threshold)}`",
        f"- policy_family_mode: `{args.policy_family_mode}`",
        f"- high_initial_rate_threshold: `{float(args.high_initial_rate_threshold)}`",
        "",
        "## 4. 划分摘要",
        "",
        markdown_table(summary),
        "",
        "## 5. 使用建议",
        "",
        "- 先在 balanced split 上复现实验，再在 long_life_holdout 上检查 LightGBM 与 LSTM 的性能降幅。",
        "- 如果 LSTM 在 long_life_holdout 或后续 late-stage block 过滤下下降更少，才说明它可能具有时序外推优势。",
        "- 不要为了让某个模型获胜而替换划分；每个划分都必须对应一个真实部署问题。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Create extrapolation split CSV files and summary report."""

    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_policy_cell_samples(args.life_path, args.policy_meaning_path)

    balanced_train = pd.read_csv(args.balanced_train_path, encoding=ENCODING)
    balanced_valid = pd.read_csv(args.balanced_valid_path, encoding=ENCODING)
    for frame in [balanced_train, balanced_valid]:
        for col in SPLIT_COLUMNS:
            if col not in frame.columns:
                frame[col] = ""
    balanced_train = balanced_train.loc[:, SPLIT_COLUMNS].copy()
    balanced_valid = balanced_valid.loc[:, SPLIT_COLUMNS].copy()

    long_valid = samples.loc[samples["max_cycles"].astype(int) >= int(args.long_life_threshold)].copy()
    long_train = samples.loc[samples["max_cycles"].astype(int) < int(args.long_life_threshold)].copy()

    family_mask = make_policy_family_mask(samples, str(args.policy_family_mode), float(args.high_initial_rate_threshold))
    family_valid = samples.loc[family_mask].copy()
    family_train = samples.loc[~family_mask].copy()

    split_map = {
        "balanced": (balanced_train, balanced_valid),
        "long_life_holdout": (long_train, long_valid),
        "policy_family_holdout": (family_train, family_valid),
    }
    summary_rows: List[Dict[str, object]] = []
    for name, (train, valid) in split_map.items():
        if train.empty or valid.empty:
            raise RuntimeError(f"Split {name} has empty train or valid set.")
        if key_set(train).intersection(key_set(valid)):
            raise RuntimeError(f"Split {name} has train/valid policy-cell overlap.")
        write_split(train, valid, out_dir, name)
        summary_rows.extend(build_summary_rows(name, train, valid))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "split_summary.csv", index=False, encoding=ENCODING)
    config = {
        "life_path": str(args.life_path),
        "policy_meaning_path": str(args.policy_meaning_path),
        "balanced_train_path": str(args.balanced_train_path),
        "balanced_valid_path": str(args.balanced_valid_path),
        "long_life_threshold": int(args.long_life_threshold),
        "policy_family_mode": str(args.policy_family_mode),
        "high_initial_rate_threshold": float(args.high_initial_rate_threshold),
        "output_dir": str(out_dir),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding=ENCODING)
    (out_dir / "extrapolation_splits_report.md").write_text(build_report(summary, args), encoding=ENCODING)
    print(f"Saved extrapolation splits to: {out_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
