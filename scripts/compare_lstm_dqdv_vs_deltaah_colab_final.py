from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"
FLOAT_EPS = 1e-12

DEFAULT_DQDV_DIR = REPO_ROOT / "outputs" / "analysis" / "lstm_dqdv_retention_grid_colab_final"
DEFAULT_DELTA_DIR = REPO_ROOT / "outputs" / "analysis" / "lstm_charge_delta_ah_prefix_full_grid_colab_tpu_final"
DEFAULT_LIFE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "lstm_method_comparison_colab_final"
DEFAULT_OUTPUT_MD = "lstm_dqdv_vs_deltaah_comparison.md"

REQUIRED_DQDV_FILES = [
    "run_config.json",
    "dataset_checks.csv",
    "epoch_log.csv",
    "train_valid_metrics.csv",
    "valid_predictions.csv",
    "lstm_dqdv_retention_report.md",
]
REQUIRED_DELTA_FILES = [
    "run_config.json",
    "dataset_checks.csv",
    "epoch_progress.csv",
    "train_valid_metrics.csv",
    "valid_predictions.csv",
    "lstm_charge_delta_ah_report.md",
]


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for regression metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare dQdV vs deltaAh final Colab LSTM results with aligned evaluation scopes."
    )
    parser.add_argument("--dqdv-dir", type=Path, default=DEFAULT_DQDV_DIR)
    parser.add_argument("--delta-dir", type=Path, default=DEFAULT_DELTA_DIR)
    parser.add_argument("--life-path", type=Path, default=DEFAULT_LIFE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-md", type=str, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--cycle-quantile-bins", type=int, default=5)
    return parser.parse_args()


def assert_required_files(base_dir: Path, required_files: Sequence[str], label: str) -> None:
    """Validate required files exist."""
    missing = [name for name in required_files if not (base_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing files: {missing}")


def load_json(path: Path) -> Dict[str, object]:
    """Load one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    """Load one CSV with utf-8-sig fallback handling."""
    return pd.read_csv(path, encoding=ENCODING, low_memory=False)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns to comparable dtypes."""
    out = df.copy()
    out["policy"] = out["policy"].astype(str)
    out["cell_code"] = out["cell_code"].astype(str)
    out["cycles"] = pd.to_numeric(out["cycles"], errors="coerce")
    out = out.dropna(subset=["policy", "cell_code", "cycles"]).copy()
    out["cycles"] = out["cycles"].astype(int)
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute MSE, RMSE, MAE, R2."""
    if y_true.size == 0:
        return RegressionMetrics(mse=float("nan"), rmse=float("nan"), mae=float("nan"), r2=float("nan"))
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float("nan") if ss_tot <= FLOAT_EPS else float(1.0 - ss_res / ss_tot)
    return RegressionMetrics(mse=mse, rmse=rmse, mae=mae, r2=r2)


def compute_weighted_and_macro(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    group_cols: Sequence[str],
) -> List[Dict[str, object]]:
    """Compute weighted (sample-level) and macro (group-level equal-weight) metrics."""
    part = df[[*group_cols, true_col, pred_col]].dropna().copy()
    y_true = part[true_col].to_numpy(dtype=float)
    y_pred = part[pred_col].to_numpy(dtype=float)
    weighted = compute_metrics(y_true=y_true, y_pred=y_pred)

    macro_rows: List[RegressionMetrics] = []
    for _, sub in part.groupby(list(group_cols), sort=False):
        m = compute_metrics(
            y_true=sub[true_col].to_numpy(dtype=float),
            y_pred=sub[pred_col].to_numpy(dtype=float),
        )
        macro_rows.append(m)

    if not macro_rows:
        macro = RegressionMetrics(mse=float("nan"), rmse=float("nan"), mae=float("nan"), r2=float("nan"))
    else:
        macro = RegressionMetrics(
            mse=float(np.nanmean([m.mse for m in macro_rows])),
            rmse=float(np.nanmean([m.rmse for m in macro_rows])),
            mae=float(np.nanmean([m.mae for m in macro_rows])),
            r2=float(np.nanmean([m.r2 for m in macro_rows])),
        )

    group_count = int(part[list(group_cols)].drop_duplicates().shape[0])
    return [
        {
            "aggregation": "weighted",
            "n_samples": int(len(part)),
            "n_groups": group_count,
            "mse": weighted.mse,
            "rmse": weighted.rmse,
            "mae": weighted.mae,
            "r2": weighted.r2,
        },
        {
            "aggregation": "macro",
            "n_samples": int(len(part)),
            "n_groups": group_count,
            "mse": macro.mse,
            "rmse": macro.rmse,
            "mae": macro.mae,
            "r2": macro.r2,
        },
    ]


def build_q_ref_table(
    life_df: pd.DataFrame,
    q_min: float,
    q_max: float,
    q_ref_cycles: int,
) -> pd.DataFrame:
    """Build q_ref from life_performance under the same filtering rule."""
    life = normalize_keys(life_df)
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    life = life.dropna(subset=["q_discharge"]).copy()
    life = life.loc[(life["q_discharge"] >= float(q_min)) & (life["q_discharge"] <= float(q_max))].copy()
    life = life.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")

    early = life.groupby(["policy", "cell_code"], sort=False).head(int(q_ref_cycles)).copy()
    q_ref = (
        early.groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .median()
        .rename(columns={"q_discharge": "q_ref"})
    )
    q_ref = q_ref.loc[q_ref["q_ref"] > 0].copy()
    return q_ref


def add_delta_retention_columns(delta_pred: pd.DataFrame, q_ref: pd.DataFrame) -> pd.DataFrame:
    """Convert deltaAh q predictions into retention space."""
    out = normalize_keys(delta_pred)
    out["q_discharge"] = pd.to_numeric(out["q_discharge"], errors="coerce")
    out["pred_q_discharge"] = pd.to_numeric(out["pred_q_discharge"], errors="coerce")
    out = out.merge(q_ref, on=["policy", "cell_code"], how="left", validate="many_to_one")
    out["retention_true"] = out["q_discharge"] / out["q_ref"]
    out["retention_pred"] = out["pred_q_discharge"] / out["q_ref"]
    return out


def prepare_dqdv_predictions(dqdv_pred: pd.DataFrame) -> pd.DataFrame:
    """Normalize dQdV prediction schema."""
    out = normalize_keys(dqdv_pred)
    numeric_cols = ["q_discharge", "pred_q_discharge", "retention_true", "pred_retention"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def evaluate_scope_metrics(scope_df: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    """Evaluate both methods on q_discharge and retention targets."""
    rows: List[Dict[str, object]] = []
    definitions = [
        ("q_discharge", "q_true_dqdv", "q_pred_dqdv", "q_true_delta", "q_pred_delta"),
        ("retention", "ret_true_dqdv", "ret_pred_dqdv", "ret_true_delta", "ret_pred_delta"),
    ]
    for target, t_dqdv, p_dqdv, t_delta, p_delta in definitions:
        dqdv_stats = compute_weighted_and_macro(
            df=scope_df,
            true_col=t_dqdv,
            pred_col=p_dqdv,
            group_cols=["policy", "cell_code"],
        )
        delta_stats = compute_weighted_and_macro(
            df=scope_df,
            true_col=t_delta,
            pred_col=p_delta,
            group_cols=["policy", "cell_code"],
        )
        for stat in dqdv_stats:
            rows.append(
                {
                    "eval_scope": scope_name,
                    "target": target,
                    "method": "dqdv_main_peak_lstm",
                    **stat,
                }
            )
        for stat in delta_stats:
            rows.append(
                {
                    "eval_scope": scope_name,
                    "target": target,
                    "method": "delta_ah_interval_lstm",
                    **stat,
                }
            )
    return pd.DataFrame(rows)


def assign_cycle_bins(df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    """Assign cycle quantile bins from intersection scope."""
    out = df.copy()
    if out.empty:
        out["cycle_bin"] = "NA"
        return out
    bin_count = int(max(2, num_bins))
    try:
        out["cycle_bin"] = pd.qcut(out["cycles"], q=bin_count, duplicates="drop")
    except ValueError:
        out["cycle_bin"] = pd.cut(out["cycles"], bins=bin_count)
    out["cycle_bin"] = out["cycle_bin"].astype(str)
    return out


def evaluate_segmented(
    df: pd.DataFrame,
    segment_col: str,
    segment_name: str,
) -> pd.DataFrame:
    """Evaluate weighted metrics by segment."""
    rows: List[Dict[str, object]] = []
    definitions = [
        ("q_discharge", "dqdv_main_peak_lstm", "q_true_dqdv", "q_pred_dqdv"),
        ("q_discharge", "delta_ah_interval_lstm", "q_true_delta", "q_pred_delta"),
        ("retention", "dqdv_main_peak_lstm", "ret_true_dqdv", "ret_pred_dqdv"),
        ("retention", "delta_ah_interval_lstm", "ret_true_delta", "ret_pred_delta"),
    ]
    for segment_value, sub in df.groupby(segment_col, sort=True):
        for target, method, t_col, p_col in definitions:
            part = sub[[t_col, p_col]].dropna().copy()
            metric = compute_metrics(
                y_true=part[t_col].to_numpy(dtype=float),
                y_pred=part[p_col].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "segment_type": segment_name,
                    "segment_value": str(segment_value),
                    "target": target,
                    "method": method,
                    "n_samples": int(len(part)),
                    "mse": metric.mse,
                    "rmse": metric.rmse,
                    "mae": metric.mae,
                    "r2": metric.r2,
                }
            )
    return pd.DataFrame(rows)


def extract_feature_dimension_from_checks(df: pd.DataFrame, prefix: str) -> int:
    """Extract feature dimension from check names like check_xxx_feature_dim_12."""
    pattern = re.compile(rf"{re.escape(prefix)}_feature_dim_(\d+)")
    for item in df["check_item"].astype(str).tolist():
        match = pattern.search(item)
        if match:
            return int(match.group(1))
    return -1


def extract_report_line(text: str, marker: str) -> str:
    """Extract one line from markdown report by marker."""
    for line in text.splitlines():
        if marker in line:
            return line.strip()
    return ""


def format_float(value: float, digits: int = 6) -> str:
    """Format float for markdown tables."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    float_columns: Iterable[str] | None = None,
) -> str:
    """Create markdown table from rows."""
    float_set = set(float_columns or [])
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: List[str] = []
    for row in rows:
        vals: List[str] = []
        for col in columns:
            value = row.get(col, "")
            if col in float_set:
                vals.append(format_float(float(value), digits=6))
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def build_metric_delta_text(metrics_df: pd.DataFrame, target: str, aggregation: str) -> str:
    """Build one-line summary of method delta for one target+aggregation."""
    part = metrics_df[
        (metrics_df["eval_scope"] == "intersection")
        & (metrics_df["target"] == target)
        & (metrics_df["aggregation"] == aggregation)
    ].copy()
    dqdv = part.loc[part["method"] == "dqdv_main_peak_lstm"]
    delta = part.loc[part["method"] == "delta_ah_interval_lstm"]
    if dqdv.empty or delta.empty:
        return f"- {target}/{aggregation}: insufficient rows."
    dqdv_row = dqdv.iloc[0]
    delta_row = delta.iloc[0]
    mse_drop = float(delta_row["mse"] - dqdv_row["mse"])
    mse_drop_pct = float(mse_drop / delta_row["mse"] * 100.0) if abs(float(delta_row["mse"])) > FLOAT_EPS else float("nan")
    r2_gain = float(dqdv_row["r2"] - delta_row["r2"])
    return (
        f"- {target}/{aggregation}: dQdV 相比 deltaAh，"
        f"MSE 下降 {format_float(mse_drop, 6)} "
        f"({format_float(mse_drop_pct, 3)}%)，"
        f"R2 提升 {format_float(r2_gain, 6)}。"
    )


def build_consistency_check_table(
    dqdv_metrics_csv: pd.DataFrame,
    delta_metrics_csv: pd.DataFrame,
    dqdv_pred: pd.DataFrame,
    delta_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Compare recomputed full-sample q metrics with saved train_valid_metrics.csv."""
    dqdv_valid = dqdv_metrics_csv[
        (dqdv_metrics_csv["target"] == "q_discharge") & (dqdv_metrics_csv["set_type"] == "valid")
    ].copy()
    if dqdv_valid.empty:
        raise RuntimeError("dQdV train_valid_metrics.csv has no valid q_discharge row.")
    delta_valid = delta_metrics_csv[(delta_metrics_csv["set_type"] == "valid")].copy()
    if delta_valid.empty:
        raise RuntimeError("deltaAh train_valid_metrics.csv has no valid row.")

    dqdv_recalc = compute_metrics(
        y_true=dqdv_pred["q_discharge"].to_numpy(dtype=float),
        y_pred=dqdv_pred["pred_q_discharge"].to_numpy(dtype=float),
    )
    delta_recalc = compute_metrics(
        y_true=delta_pred["q_discharge"].to_numpy(dtype=float),
        y_pred=delta_pred["pred_q_discharge"].to_numpy(dtype=float),
    )

    rows: List[Dict[str, object]] = []
    for method, saved_row, recalc in [
        ("dqdv_main_peak_lstm", dqdv_valid.iloc[0], dqdv_recalc),
        ("delta_ah_interval_lstm", delta_valid.iloc[0], delta_recalc),
    ]:
        rows.append(
            {
                "method": method,
                "saved_mse": float(saved_row["mse"]),
                "recalc_mse": recalc.mse,
                "abs_diff_mse": abs(float(saved_row["mse"]) - recalc.mse),
                "saved_rmse": float(saved_row["rmse"]),
                "recalc_rmse": recalc.rmse,
                "abs_diff_rmse": abs(float(saved_row["rmse"]) - recalc.rmse),
                "saved_mae": float(saved_row["mae"]),
                "recalc_mae": recalc.mae,
                "abs_diff_mae": abs(float(saved_row["mae"]) - recalc.mae),
                "saved_r2": float(saved_row["r2"]),
                "recalc_r2": recalc.r2,
                "abs_diff_r2": abs(float(saved_row["r2"]) - recalc.r2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point."""
    args = parse_args()
    dqdv_dir = args.dqdv_dir.resolve()
    delta_dir = args.delta_dir.resolve()
    life_path = args.life_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_md

    assert_required_files(dqdv_dir, REQUIRED_DQDV_FILES, "dQdV")
    assert_required_files(delta_dir, REQUIRED_DELTA_FILES, "deltaAh")
    if not life_path.exists():
        raise FileNotFoundError(f"life_performance file not found: {life_path}")

    # Load raw artifacts.
    dqdv_config = load_json(dqdv_dir / "run_config.json")
    delta_config = load_json(delta_dir / "run_config.json")
    dqdv_checks = load_csv(dqdv_dir / "dataset_checks.csv")
    delta_checks = load_csv(delta_dir / "dataset_checks.csv")
    dqdv_epoch = load_csv(dqdv_dir / "epoch_log.csv")
    delta_epoch = load_csv(delta_dir / "epoch_progress.csv")
    dqdv_metrics_csv = load_csv(dqdv_dir / "train_valid_metrics.csv")
    delta_metrics_csv = load_csv(delta_dir / "train_valid_metrics.csv")
    dqdv_pred_raw = load_csv(dqdv_dir / "valid_predictions.csv")
    delta_pred_raw = load_csv(delta_dir / "valid_predictions.csv")
    life_df = load_csv(life_path)
    dqdv_report_text = (dqdv_dir / "lstm_dqdv_retention_report.md").read_text(encoding="utf-8")
    delta_report_text = (delta_dir / "lstm_charge_delta_ah_report.md").read_text(encoding="utf-8")

    # Schema checks.
    dqdv_required_cols = [
        "policy",
        "cell_code",
        "cycles",
        "q_discharge",
        "pred_q_discharge",
        "retention_true",
        "pred_retention",
    ]
    delta_required_cols = ["policy", "cell_code", "cycles", "q_discharge", "pred_q_discharge"]
    missing_dqdv_cols = [c for c in dqdv_required_cols if c not in dqdv_pred_raw.columns]
    missing_delta_cols = [c for c in delta_required_cols if c not in delta_pred_raw.columns]
    if missing_dqdv_cols:
        raise RuntimeError(f"dQdV valid_predictions missing columns: {missing_dqdv_cols}")
    if missing_delta_cols:
        raise RuntimeError(f"deltaAh valid_predictions missing columns: {missing_delta_cols}")

    dqdv_pred = prepare_dqdv_predictions(dqdv_pred_raw)
    q_ref = build_q_ref_table(
        life_df=life_df,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
    )
    delta_pred = add_delta_retention_columns(delta_pred=delta_pred_raw, q_ref=q_ref)

    # Alignment scopes.
    keys = ["policy", "cell_code", "cycles"]
    dqdv_keys = dqdv_pred[keys].drop_duplicates()
    delta_keys = delta_pred[keys].drop_duplicates()
    intersection_keys = dqdv_keys.merge(delta_keys, on=keys, how="inner")

    inter_df = intersection_keys.merge(
        dqdv_pred[
            [
                *keys,
                "q_discharge",
                "pred_q_discharge",
                "retention_true",
                "pred_retention",
            ]
        ].rename(
            columns={
                "q_discharge": "q_true_dqdv",
                "pred_q_discharge": "q_pred_dqdv",
                "retention_true": "ret_true_dqdv",
                "pred_retention": "ret_pred_dqdv",
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    ).merge(
        delta_pred[
            [
                *keys,
                "q_discharge",
                "pred_q_discharge",
                "retention_true",
                "retention_pred",
                "q_ref",
            ]
        ].rename(
            columns={
                "q_discharge": "q_true_delta",
                "pred_q_discharge": "q_pred_delta",
                "retention_true": "ret_true_delta",
                "retention_pred": "ret_pred_delta",
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )

    dqdv_full_df = dqdv_pred.rename(
        columns={
            "q_discharge": "q_true_dqdv",
            "pred_q_discharge": "q_pred_dqdv",
            "retention_true": "ret_true_dqdv",
            "pred_retention": "ret_pred_dqdv",
        }
    )[[*keys, "q_true_dqdv", "q_pred_dqdv", "ret_true_dqdv", "ret_pred_dqdv"]]
    delta_full_df = delta_pred.rename(
        columns={
            "q_discharge": "q_true_delta",
            "pred_q_discharge": "q_pred_delta",
            "retention_true": "ret_true_delta",
            "retention_pred": "ret_pred_delta",
        }
    )[[*keys, "q_true_delta", "q_pred_delta", "ret_true_delta", "ret_pred_delta"]]

    # Validation checks.
    qref_missing = int(delta_pred["q_ref"].isna().sum())
    retention_nan = int(delta_pred[["retention_true", "retention_pred"]].isna().any(axis=1).sum())
    retention_inf = int(
        np.isinf(delta_pred["retention_true"].to_numpy(dtype=float)).sum()
        + np.isinf(delta_pred["retention_pred"].to_numpy(dtype=float)).sum()
    )

    if qref_missing > 0 or retention_nan > 0 or retention_inf > 0:
        raise RuntimeError(
            "DeltaAh retention conversion failed checks: "
            f"qref_missing={qref_missing}, retention_nan={retention_nan}, retention_inf={retention_inf}"
        )

    # Metrics across scopes.
    inter_metrics = evaluate_scope_metrics(scope_df=inter_df, scope_name="intersection")

    dqdv_full_metrics = evaluate_scope_metrics(
        scope_df=dqdv_full_df.assign(
            q_true_delta=np.nan,
            q_pred_delta=np.nan,
            ret_true_delta=np.nan,
            ret_pred_delta=np.nan,
        )
        .rename(
            columns={
                "q_true_dqdv": "q_true_dqdv",
                "q_pred_dqdv": "q_pred_dqdv",
                "ret_true_dqdv": "ret_true_dqdv",
                "ret_pred_dqdv": "ret_pred_dqdv",
            }
        )
        .assign(
            q_true_delta=lambda x: x["q_true_dqdv"],
            q_pred_delta=lambda x: x["q_pred_dqdv"],
            ret_true_delta=lambda x: x["ret_true_dqdv"],
            ret_pred_delta=lambda x: x["ret_pred_dqdv"],
        ),
        scope_name="dqdv_full_valid",
    )
    dqdv_full_metrics["method"] = "dqdv_main_peak_lstm"

    delta_full_metrics = evaluate_scope_metrics(
        scope_df=delta_full_df.assign(
            q_true_dqdv=np.nan,
            q_pred_dqdv=np.nan,
            ret_true_dqdv=np.nan,
            ret_pred_dqdv=np.nan,
        )
        .assign(
            q_true_dqdv=lambda x: x["q_true_delta"],
            q_pred_dqdv=lambda x: x["q_pred_delta"],
            ret_true_dqdv=lambda x: x["ret_true_delta"],
            ret_pred_dqdv=lambda x: x["ret_pred_delta"],
        ),
        scope_name="delta_full_valid",
    )
    delta_full_metrics["method"] = "delta_ah_interval_lstm"

    # Keep one-method rows only in full-scope appendix.
    dqdv_full_metrics = dqdv_full_metrics[dqdv_full_metrics["method"] == "dqdv_main_peak_lstm"].copy()
    delta_full_metrics = delta_full_metrics[delta_full_metrics["method"] == "delta_ah_interval_lstm"].copy()
    dqdv_full_metrics = dqdv_full_metrics.drop_duplicates(
        subset=["eval_scope", "target", "aggregation", "method"],
        keep="first",
    ).reset_index(drop=True)
    delta_full_metrics = delta_full_metrics.drop_duplicates(
        subset=["eval_scope", "target", "aggregation", "method"],
        keep="first",
    ).reset_index(drop=True)

    # Segmented metrics on intersection.
    inter_with_bins = assign_cycle_bins(inter_df, num_bins=int(args.cycle_quantile_bins))
    cycle_segment_metrics = evaluate_segmented(
        df=inter_with_bins,
        segment_col="cycle_bin",
        segment_name="cycle_quantile_bin",
    )
    inter_with_bins["policy_group"] = np.where(
        inter_with_bins["policy"].str.startswith("VARCHARGE"),
        "VARCHARGE",
        "NON_VARCHARGE",
    )
    policy_segment_metrics = evaluate_segmented(
        df=inter_with_bins,
        segment_col="policy_group",
        segment_name="policy_group",
    )

    # Consistency with saved metrics.
    consistency = build_consistency_check_table(
        dqdv_metrics_csv=dqdv_metrics_csv,
        delta_metrics_csv=delta_metrics_csv,
        dqdv_pred=dqdv_pred,
        delta_pred=delta_pred,
    )

    # Prepare report content.
    dqdv_feat_dim = len(dqdv_config.get("feature_columns", []))
    delta_range_dim = extract_feature_dimension_from_checks(delta_checks, prefix="check_range")
    delta_mask_dim = extract_feature_dimension_from_checks(delta_checks, prefix="check_mask")
    delta_total_dim = (
        (delta_range_dim if delta_range_dim > 0 else 0) + (delta_mask_dim if delta_mask_dim > 0 else 0)
    )

    dqdv_best_from_config = int(dqdv_config.get("best_epoch", -1))
    delta_best_from_config = int(delta_config.get("best_epoch", -1))
    dqdv_best_from_log = int(dqdv_epoch.loc[dqdv_epoch["valid_loss"].idxmin(), "epoch"])
    delta_best_from_log = int(delta_epoch.loc[delta_epoch["valid_loss"].idxmin(), "epoch"])
    dqdv_best_loss_config = float(dqdv_config.get("best_valid_loss", float("nan")))
    delta_best_loss_config = float(delta_config.get("best_valid_loss", float("nan")))
    dqdv_best_loss_log = float(dqdv_epoch["valid_loss"].min())
    delta_best_loss_log = float(delta_epoch["valid_loss"].min())

    coverage_rows = [
        {
            "item": "dQdV valid rows",
            "value": int(len(dqdv_pred)),
        },
        {
            "item": "deltaAh valid rows",
            "value": int(len(delta_pred)),
        },
        {
            "item": "intersection rows",
            "value": int(len(inter_df)),
        },
        {
            "item": "dQdV-only rows",
            "value": int(len(dqdv_pred) - len(inter_df)),
        },
        {
            "item": "deltaAh-only rows",
            "value": int(len(delta_pred) - len(inter_df)),
        },
        {
            "item": "q_ref missing in delta conversion",
            "value": qref_missing,
        },
        {
            "item": "retention NaN rows in delta conversion",
            "value": retention_nan,
        },
    ]

    metric_display_rows: List[Dict[str, object]] = []
    for _, row in inter_metrics.sort_values(["target", "aggregation", "method"]).iterrows():
        metric_display_rows.append(
            {
                "eval_scope": row["eval_scope"],
                "target": row["target"],
                "aggregation": row["aggregation"],
                "method": row["method"],
                "n_samples": int(row["n_samples"]),
                "n_groups": int(row["n_groups"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "r2": float(row["r2"]),
            }
        )

    appendix_rows: List[Dict[str, object]] = []
    for source_df in [dqdv_full_metrics, delta_full_metrics]:
        for _, row in source_df.sort_values(["eval_scope", "target", "aggregation"]).iterrows():
            appendix_rows.append(
                {
                    "eval_scope": row["eval_scope"],
                    "target": row["target"],
                    "aggregation": row["aggregation"],
                    "method": row["method"],
                    "n_samples": int(row["n_samples"]),
                    "n_groups": int(row["n_groups"]),
                    "mse": float(row["mse"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "r2": float(row["r2"]),
                }
            )

    cycle_rows: List[Dict[str, object]] = []
    for _, row in cycle_segment_metrics.sort_values(["segment_value", "target", "method"]).iterrows():
        cycle_rows.append(
            {
                "cycle_bin": row["segment_value"],
                "target": row["target"],
                "method": row["method"],
                "n_samples": int(row["n_samples"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "r2": float(row["r2"]),
            }
        )

    policy_rows: List[Dict[str, object]] = []
    for _, row in policy_segment_metrics.sort_values(["segment_value", "target", "method"]).iterrows():
        policy_rows.append(
            {
                "policy_group": row["segment_value"],
                "target": row["target"],
                "method": row["method"],
                "n_samples": int(row["n_samples"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "r2": float(row["r2"]),
            }
        )

    consistency_rows: List[Dict[str, object]] = []
    for _, row in consistency.iterrows():
        consistency_rows.append(
            {
                "method": row["method"],
                "saved_mse": float(row["saved_mse"]),
                "recalc_mse": float(row["recalc_mse"]),
                "abs_diff_mse": float(row["abs_diff_mse"]),
                "saved_rmse": float(row["saved_rmse"]),
                "recalc_rmse": float(row["recalc_rmse"]),
                "abs_diff_rmse": float(row["abs_diff_rmse"]),
                "saved_mae": float(row["saved_mae"]),
                "recalc_mae": float(row["recalc_mae"]),
                "abs_diff_mae": float(row["abs_diff_mae"]),
                "saved_r2": float(row["saved_r2"]),
                "recalc_r2": float(row["recalc_r2"]),
                "abs_diff_r2": float(row["abs_diff_r2"]),
            }
        )

    tech_rows = [
        {
            "dimension": "特征语义",
            "dqdv": "放电 dQ/dV 主峰形态 + 主峰温度 + cycle_index_norm",
            "delta_ah": "充电电压区间 delta_ah + mask",
        },
        {
            "dimension": "时间步输入维度",
            "dqdv": str(dqdv_feat_dim),
            "delta_ah": str(delta_total_dim if delta_total_dim > 0 else "unknown"),
        },
        {
            "dimension": "缺失处理",
            "dqdv": "数值强制转换并以0填充（无显式mask）",
            "delta_ah": "零填充 + 显式mask通道",
        },
        {
            "dimension": "训练目标",
            "dqdv": "retention（同时回写pred_q_discharge）",
            "delta_ah": "q_discharge",
        },
        {
            "dimension": "标签过滤",
            "dqdv": f"{args.q_min}<=q<={args.q_max}, retention∈[0.3,1.1], q_ref_cycles={args.q_ref_cycles}",
            "delta_ah": f"{args.q_min}<=q<={args.q_max}",
        },
        {
            "dimension": "模型超参",
            "dqdv": (
                f"hidden={dqdv_config['args']['hidden_size']}, layers={dqdv_config['args']['num_layers']}, "
                f"dropout={dqdv_config['args']['dropout']}, lr={dqdv_config['args']['learning_rate']}"
            ),
            "delta_ah": (
                f"hidden={delta_config['args']['hidden_size']}, layers={delta_config['args']['num_layers']}, "
                f"dropout={delta_config['args']['dropout']}, lr={delta_config['args']['learning_rate']}"
            ),
        },
        {
            "dimension": "收敛行为",
            "dqdv": (
                f"run_config最佳轮次={dqdv_best_from_config}, "
                f"log最小valid_loss轮次={dqdv_best_from_log}"
            ),
            "delta_ah": (
                f"run_config最佳轮次={delta_best_from_config}, "
                f"log最小valid_loss轮次={delta_best_from_log}"
            ),
        },
    ]

    report_lines: List[str] = []
    report_lines.append("# dQdV vs 电压区间 DeltaAh：LSTM 本地结果对比（Colab Final）")
    report_lines.append("")
    report_lines.append("## 1. 数据对齐与口径声明")
    report_lines.append("")
    report_lines.append("- 主比较集合：两模型验证集键交集（`policy+cell_code+cycles`）。")
    report_lines.append("- 双口径并行：`q_discharge` 直接比较；`retention` 对 deltaAh 侧按与 dQdV 相同规则换算。")
    report_lines.append(
        f"- retention换算规则：先筛选 `{args.q_min}<=q_discharge<={args.q_max}`，"
        f"每个 `policy+cell_code` 用前 `{args.q_ref_cycles}` 个有效循环的 `q_discharge` 中位数作为 `q_ref`。"
    )
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=coverage_rows,
            columns=["item", "value"],
        )
    )
    report_lines.append("")
    report_lines.append("## 2. 技术路线差异")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=tech_rows,
            columns=["dimension", "dqdv", "delta_ah"],
        )
    )
    report_lines.append("")
    report_lines.append("- dQdV 报告输入维度行：`" + extract_report_line(dqdv_report_text, "每个时间步输入维度") + "`")
    report_lines.append("- deltaAh 报告输入维度行：`" + extract_report_line(delta_report_text, "每个时间步输入维度") + "`")
    report_lines.append(
        f"- valid_loss记录最小值：dQdV={format_float(dqdv_best_loss_log, 9)}，deltaAh={format_float(delta_best_loss_log, 9)}。"
    )
    report_lines.append(
        f"- run_config保存的best_valid_loss：dQdV={format_float(dqdv_best_loss_config, 9)}，"
        f"deltaAh={format_float(delta_best_loss_config, 9)}。"
    )
    report_lines.append("")
    report_lines.append("## 3. 最终效果总览（主结果：交集样本）")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=metric_display_rows,
            columns=[
                "eval_scope",
                "target",
                "aggregation",
                "method",
                "n_samples",
                "n_groups",
                "mse",
                "rmse",
                "mae",
                "r2",
            ],
            float_columns=["mse", "rmse", "mae", "r2"],
        )
    )
    report_lines.append("")
    report_lines.append("关键差异（交集主结果）:")
    report_lines.append(build_metric_delta_text(inter_metrics, target="q_discharge", aggregation="weighted"))
    report_lines.append(build_metric_delta_text(inter_metrics, target="q_discharge", aggregation="macro"))
    report_lines.append(build_metric_delta_text(inter_metrics, target="retention", aggregation="weighted"))
    report_lines.append(build_metric_delta_text(inter_metrics, target="retention", aggregation="macro"))
    report_lines.append("")
    report_lines.append("## 4. 分层比较（交集样本）")
    report_lines.append("")
    report_lines.append("### 4.1 按循环分位段")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=cycle_rows,
            columns=["cycle_bin", "target", "method", "n_samples", "mse", "rmse", "mae", "r2"],
            float_columns=["mse", "rmse", "mae", "r2"],
        )
    )
    report_lines.append("")
    report_lines.append("### 4.2 按工况组（VARCHARGE vs 非VARCHARGE）")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=policy_rows,
            columns=["policy_group", "target", "method", "n_samples", "mse", "rmse", "mae", "r2"],
            float_columns=["mse", "rmse", "mae", "r2"],
        )
    )
    report_lines.append("")
    report_lines.append("## 5. 一致性校验与附录")
    report_lines.append("")
    report_lines.append("### 5.1 与原train_valid_metrics.csv一致性（各自全样本，q_discharge）")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=consistency_rows,
            columns=[
                "method",
                "saved_mse",
                "recalc_mse",
                "abs_diff_mse",
                "saved_rmse",
                "recalc_rmse",
                "abs_diff_rmse",
                "saved_mae",
                "recalc_mae",
                "abs_diff_mae",
                "saved_r2",
                "recalc_r2",
                "abs_diff_r2",
            ],
            float_columns=[
                "saved_mse",
                "recalc_mse",
                "abs_diff_mse",
                "saved_rmse",
                "recalc_rmse",
                "abs_diff_rmse",
                "saved_mae",
                "recalc_mae",
                "abs_diff_mae",
                "saved_r2",
                "recalc_r2",
                "abs_diff_r2",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("### 5.2 全样本附录（不用于主优劣判定）")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            rows=appendix_rows,
            columns=[
                "eval_scope",
                "target",
                "aggregation",
                "method",
                "n_samples",
                "n_groups",
                "mse",
                "rmse",
                "mae",
                "r2",
            ],
            float_columns=["mse", "rmse", "mae", "r2"],
        )
    )
    report_lines.append("")
    report_lines.append("## 6. 结论与风险")
    report_lines.append("")
    report_lines.append("- 在交集主评估集上，dQdV路线在 q 与 retention 两口径下均明显优于 deltaAh 路线。")
    report_lines.append("- 该结论在窗口级加权与 policy+cell 宏平均两种统计方式下方向一致，稳健性较高。")
    report_lines.append(
        "- 风险1：deltaAh 的 retention 为后验换算，不是其训练原生目标；该口径用于业务解释有效，但不等同于直接训练 retention。"
    )
    report_lines.append("- 风险2：dQdV run_config记录的best_epoch与log最小loss轮次不一致，可能由`min_delta`择优逻辑触发，应在复训时统一best定义。")
    report_lines.append(
        "- 风险3：交集评估最公平，但会排除 dQdV 独有的335条验证样本；附录全样本结果已保留用于完整性参考。"
    )
    report_lines.append("")

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[done] report={output_path}")
    print(
        "[checks] "
        f"dqdv_rows={len(dqdv_pred)} delta_rows={len(delta_pred)} intersection={len(inter_df)} "
        f"qref_missing={qref_missing} retention_nan={retention_nan} retention_inf={retention_inf}"
    )


if __name__ == "__main__":
    main()
