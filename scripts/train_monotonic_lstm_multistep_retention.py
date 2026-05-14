"""Monotonic postprocessing and LSTM validation for multistep retention."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from train_interval_to_dqdv_retention_pipeline import ENCODING, REPO_ROOT, set_seed
from train_lstm_residual_multistep_retention import train_lightgbm_baselines
from train_multistep_interval_to_dqdv_retention_blocks import (
    FORBIDDEN_INPUT_COLS,
    TARGET_PACKS,
    BlockSample,
    block_metadata_frame,
    build_block_samples,
    build_cycle_table,
    build_linear_last10,
    build_persistence,
    downsample_blocks,
    future_arrays,
    regression_metrics,
)


SELECTED_HORIZONS = [1, 10, 20, 50]
BASELINE_METHODS = ["direct_retention", "deployable_bridge", "linear_last10", "persistence", "oracle_bridge"]
POSTPROCESS_METHODS = ["direct_retention", "deployable_bridge", "linear_last10"]
LSTM_METHODS = [
    "monotonic_lstm_penalty",
    "monotonic_lstm_delta_strict",
    "monotonic_lstm_delta_with_history_retention",
]
FORBIDDEN_CHECK_COLS = {
    "cycles",
    "cycle_index_norm",
    "policy",
    "cell_code",
    "initial_c_rate",
    "switch_soc_percent",
    "post_switch_c_rate",
}
warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)


@dataclass
class BaselineTables:
    """Baseline prediction and metric tables used by postprocessing and reporting."""

    valid_predictions_long: pd.DataFrame
    metrics: pd.DataFrame
    source: str


@dataclass
class RetentionStandardizer:
    """Scalar train-only retention standardization parameters."""

    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardize retention values."""

        return ((np.asarray(values, dtype=np.float32) - float(self.mean)) / float(self.std)).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Convert standardized retention values back to retention units."""

        return (np.asarray(values, dtype=np.float32) * float(self.std) + float(self.mean)).astype(np.float32)


@dataclass
class LstmResult:
    """Output bundle for one monotonic LSTM run."""

    method: str
    train_pred: np.ndarray
    valid_pred: np.ndarray
    epoch_log: pd.DataFrame
    best_epoch: int
    best_valid_loss: float
    best_valid_h50_rmse: float


class RetentionSequenceDataset(Dataset):
    """PyTorch dataset for sequence-to-multistep retention learning."""

    def __init__(self, x: np.ndarray, y_z: np.ndarray, last_z: np.ndarray) -> None:
        """Store normalized history features, standardized targets, and standardized starts."""

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y_z = torch.as_tensor(y_z, dtype=torch.float32)
        self.last_z = torch.as_tensor(last_z, dtype=torch.float32)

    def __len__(self) -> int:
        """Return sample count."""

        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one fixed-history sequence, target vector, and recurrence start."""

        return self.x[idx], self.y_z[idx], self.last_z[idx]


class RetentionLSTM(nn.Module):
    """Small LSTM encoder with a multistep retention head."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        delta_output: bool,
        delta_init_bias: float,
    ) -> None:
        """Create an LSTM model that outputs retention or non-negative deltas."""

        super().__init__()
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(hidden_size))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden_size), int(horizon))
        if bool(delta_output):
            nn.init.constant_(self.head.bias, float(delta_init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a raw multistep output vector."""

        _out, (hidden, _cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(self.dropout(self.norm(last_hidden)))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Validate monotonic retention constraints on non-overlapping H100/M50 block samples."
    )
    parser.add_argument(
        "--recommended-feature-path",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "analysis"
        / "interval_features_to_dqdv_correlation"
        / "recommended_feature_pack_union.csv",
    )
    parser.add_argument(
        "--charge-timeseries-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv",
    )
    parser.add_argument(
        "--discharge-interval-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "discharge_interval_features.csv",
    )
    parser.add_argument(
        "--dqdv-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv",
    )
    parser.add_argument(
        "--life-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "life_performance.csv",
    )
    parser.add_argument(
        "--train-split-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv",
    )
    parser.add_argument(
        "--valid-split-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "multistep_interval_to_dqdv_retention_blocks_h100_m50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "monotonic_lstm_multistep_retention_blocks_h100_m50",
    )
    parser.add_argument("--split-name", type=str, default="balanced")
    parser.add_argument("--history-len", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--block-stride", type=int, default=150)
    parser.add_argument("--sample-mode", choices=["non_overlapping_blocks"], default="non_overlapping_blocks")
    parser.add_argument("--feature-pack", choices=["recommended55"], default="recommended55")
    parser.add_argument("--target-pack", choices=["compact4"], default="compact4")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-mono", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--delta-init-bias", type=float, default=-6.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--random-seed", type=int, default=20260511)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-train-cells", type=int, default=12)
    parser.add_argument("--smoke-valid-cells", type=int, default=6)
    parser.add_argument("--max-train-blocks", type=int, default=0)
    parser.add_argument("--max-valid-blocks", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the PyTorch device requested by the CLI."""

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device_arg == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolved_path_text(path: Path) -> str:
    """Return a stable absolute path string for configs and reports."""

    return Path(path).expanduser().resolve().as_posix()


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render selected dataframe columns as a Markdown table."""

    view = df.loc[:, list(columns)].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    view = view.fillna("")
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(escape_markdown_cell(item) for item in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape_markdown_cell(item) for item in row) + " |")
    return "\n".join(lines)


def escape_markdown_cell(value: object) -> str:
    """Escape a value for a simple Markdown table cell."""

    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def selected_horizon_steps(horizon: int) -> List[int]:
    """Return selected horizon steps available for the current run."""

    selected = [step for step in SELECTED_HORIZONS if int(step) <= int(horizon)]
    if int(horizon) not in selected:
        selected.append(int(horizon))
    return sorted(set(selected))


def build_dataset_checks(
    args: argparse.Namespace,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    train_samples: Sequence[BlockSample],
    valid_samples: Sequence[BlockSample],
) -> pd.DataFrame:
    """Build dataset and CLI contract checks."""

    forbidden = sorted(set(feature_cols).intersection(FORBIDDEN_CHECK_COLS))
    checks = [
        ("split_name", str(args.split_name), str(args.split_name), 1, "declared train/valid split"),
        ("train_split_path", resolved_path_text(args.train_split_path), resolved_path_text(args.train_split_path), 1, ""),
        ("valid_split_path", resolved_path_text(args.valid_split_path), resolved_path_text(args.valid_split_path), 1, ""),
        ("sample_mode", str(args.sample_mode), "non_overlapping_blocks", int(str(args.sample_mode) == "non_overlapping_blocks"), ""),
        ("history_len", int(args.history_len), ">0", int(int(args.history_len) > 0), "positive history length"),
        ("horizon", int(args.horizon), ">0", int(int(args.horizon) > 0), "positive forecast horizon"),
        ("block_stride", int(args.block_stride), int(args.history_len) + int(args.horizon), int(int(args.block_stride) == int(args.history_len) + int(args.horizon)), "expected history_len+horizon"),
        ("feature_count", int(len(feature_cols)), 55, int(len(feature_cols) == 55), "recommended55"),
        ("target_pack", str(args.target_pack), "compact4", int(str(args.target_pack) == "compact4"), ",".join(target_cols)),
        ("target_dim", int(len(target_cols)), 4, int(len(target_cols) == 4), ",".join(target_cols)),
        ("forbidden_input_columns_present", int(len(forbidden)), 0, int(len(forbidden) == 0), ",".join(forbidden)),
        ("train_block_count", int(len(train_samples)), ">0", int(len(train_samples) > 0), ""),
        ("valid_block_count", int(len(valid_samples)), ">0", int(len(valid_samples) > 0), ""),
    ]
    return pd.DataFrame(checks, columns=["check_item", "value", "expected", "pass_flag", "details"])


def prepare_samples(args: argparse.Namespace) -> Tuple[List[BlockSample], List[BlockSample], List[str], List[str], pd.DataFrame]:
    """Load cycle-level tables and construct non-overlapping block samples."""

    target_cols = TARGET_PACKS[str(args.target_pack)]
    merged, feature_cols, _stats = build_cycle_table(args, target_cols)
    samples = build_block_samples(
        merged=merged,
        feature_cols=feature_cols,
        target_cols=target_cols,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        block_stride=int(args.block_stride),
    )
    if bool(args.smoke_test):
        max_train = int(args.max_train_blocks) if int(args.max_train_blocks) > 0 else 80
        max_valid = int(args.max_valid_blocks) if int(args.max_valid_blocks) > 0 else 40
        samples = downsample_blocks(samples, max_train, max_valid, int(args.random_seed))
    elif int(args.max_train_blocks) > 0 or int(args.max_valid_blocks) > 0:
        samples = downsample_blocks(samples, int(args.max_train_blocks), int(args.max_valid_blocks), int(args.random_seed))
    train_samples = [sample for sample in samples if sample.set_type == "train"]
    valid_samples = [sample for sample in samples if sample.set_type == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid block samples are required.")
    checks = build_dataset_checks(args, feature_cols, target_cols, train_samples, valid_samples)
    if int(checks["pass_flag"].min()) != 1:
        raise RuntimeError("Dataset checks failed before training.")
    return train_samples, valid_samples, list(feature_cols), list(target_cols), checks


def baseline_config_matches(args: argparse.Namespace, config_path: Path) -> bool:
    """Return true when an existing baseline directory matches the requested contract."""

    if bool(args.smoke_test) or not config_path.exists():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "split_name": str(args.split_name),
        "train_split_path": resolved_path_text(args.train_split_path),
        "valid_split_path": resolved_path_text(args.valid_split_path),
        "history_len": int(args.history_len),
        "horizon": int(args.horizon),
        "block_stride": int(args.block_stride),
        "sample_mode": str(args.sample_mode),
        "block_stage_filter": "none",
        "feature_pack": str(args.feature_pack),
        "target_pack": str(args.target_pack),
        "feature_count": 55,
        "history_representation": "summary",
    }
    return all(config.get(key) == value for key, value in expected.items())


def load_or_train_baselines(
    args: argparse.Namespace,
    train_samples: Sequence[BlockSample],
    valid_samples: Sequence[BlockSample],
    feature_cols: Sequence[str],
) -> BaselineTables:
    """Load existing H100/M50 baseline predictions or train smoke baselines."""

    pred_path = args.baseline_dir / "valid_retention_predictions_long.csv"
    metric_path = args.baseline_dir / "retention_multistep_metrics.csv"
    config_path = args.baseline_dir / "run_config.json"
    if baseline_config_matches(args, config_path) and pred_path.exists() and metric_path.exists():
        pred_long = pd.read_csv(pred_path, encoding=ENCODING)
        metrics = pd.read_csv(metric_path, encoding=ENCODING)
        return BaselineTables(pred_long, metrics, f"loaded:{args.baseline_dir}")

    x_train_summary = np.vstack([summarize_history_for_baseline(sample.history_x) for sample in train_samples]).astype(np.float32)
    x_valid_summary = np.vstack([summarize_history_for_baseline(sample.history_x) for sample in valid_samples]).astype(np.float32)
    y_train_dqdv, y_train_ret, _train_q_ref, _train_q_discharge, _train_cycles = future_arrays(train_samples)
    y_valid_dqdv, y_valid_ret, _valid_q_ref, _valid_q_discharge, _valid_cycles = future_arrays(valid_samples)
    linear_train = build_linear_last10(train_samples)
    linear_valid = build_linear_last10(valid_samples)
    persistence_train = build_persistence(train_samples)
    persistence_valid = build_persistence(valid_samples)
    baselines = train_lightgbm_baselines(
        x_train=x_train_summary,
        x_valid=x_valid_summary,
        y_train_dqdv=y_train_dqdv,
        y_valid_dqdv=y_valid_dqdv,
        y_train_ret=y_train_ret,
        y_valid_ret=y_valid_ret,
        linear_train_ret=linear_train,
        linear_valid_ret=linear_valid,
        persistence_train_ret=persistence_train,
        persistence_valid_ret=persistence_valid,
        seed=int(args.random_seed),
    )
    pred_long = build_prediction_long("valid", valid_samples, baselines.valid_predictions)
    return BaselineTables(pred_long, baselines.metrics, "trained_for_current_run")


def summarize_history_for_baseline(history_x: np.ndarray) -> np.ndarray:
    """Summarize one history block with the baseline script's seven statistics."""

    x = np.asarray(history_x, dtype=np.float32)
    positions = np.arange(x.shape[0], dtype=np.float32)
    centered = positions - positions.mean()
    denom = float(np.sum(centered**2))
    if denom <= 0.0:
        slope = np.zeros(x.shape[1], dtype=np.float32)
    else:
        slope = (centered[:, None] * (x - x.mean(axis=0, keepdims=True))).sum(axis=0) / denom
    return np.concatenate([x[-1], x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0), x[-1] - x[0], slope])


def build_prediction_long(set_type: str, samples: Sequence[BlockSample], preds: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build a long prediction table from sample-aligned prediction matrices."""

    rows: List[Dict[str, object]] = []
    for method, pred_matrix in preds.items():
        for block_idx, sample in enumerate(samples):
            for step in range(pred_matrix.shape[1]):
                truth = float(sample.future_retention[step])
                pred = float(pred_matrix[block_idx, step])
                q_ref = float(sample.future_q_ref[step])
                rows.append(
                    {
                        "set_type": set_type,
                        "block_id": int(sample.block_id),
                        "policy": sample.policy,
                        "cell_code": sample.cell_code,
                        "input_start_cycle": int(sample.input_start_cycle),
                        "input_end_cycle": int(sample.input_end_cycle),
                        "target_cycle": int(sample.future_cycles[step]),
                        "horizon_step": int(step + 1),
                        "method": method,
                        "retention_true": truth,
                        "pred_retention": pred,
                        "residual_retention": truth - pred,
                        "q_ref": q_ref,
                        "true_q_discharge": float(sample.future_q_discharge[step]),
                        "pred_q_discharge": pred * q_ref,
                    }
                )
    return pd.DataFrame(rows)


def monotonic_curve_stats(values: np.ndarray) -> Dict[str, float]:
    """Calculate monotonic violation statistics for one curve."""

    y = np.asarray(values, dtype=np.float64).reshape(-1)
    diff = np.diff(y)
    positive = diff[diff > 0.0]
    return {
        "monotonic_violation_count": int(positive.size),
        "monotonic_violation_rate": float(positive.size / diff.size) if diff.size else 0.0,
        "max_positive_jump": float(positive.max()) if positive.size else 0.0,
        "mean_positive_jump": float(positive.mean()) if positive.size else 0.0,
        "total_positive_jump": float(positive.sum()) if positive.size else 0.0,
        "curve_has_violation": int(positive.size > 0),
    }


def monotonic_diagnostics(pred_long: pd.DataFrame, methods: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create per-curve and aggregate monotonicity diagnostics."""

    rows: List[Dict[str, object]] = []
    first_method = str(pred_long["method"].iloc[0])
    true_frame = pred_long.loc[pred_long["method"] == first_method].copy()
    for block_id, group in true_frame.groupby("block_id", sort=False):
        ordered = group.sort_values("horizon_step")
        stats = monotonic_curve_stats(ordered["retention_true"].to_numpy(dtype=float))
        stats.update(
            {
                "series": "true_retention",
                "method": "true_retention",
                "block_id": int(block_id),
                "policy": str(ordered["policy"].iloc[0]),
                "cell_code": str(ordered["cell_code"].iloc[0]),
                "input_start_cycle": int(ordered["input_start_cycle"].iloc[0]),
                "input_end_cycle": int(ordered["input_end_cycle"].iloc[0]),
            }
        )
        rows.append(stats)
    for method in methods:
        part = pred_long.loc[pred_long["method"] == method].copy()
        if part.empty:
            continue
        for block_id, group in part.groupby("block_id", sort=False):
            ordered = group.sort_values("horizon_step")
            stats = monotonic_curve_stats(ordered["pred_retention"].to_numpy(dtype=float))
            stats.update(
                {
                    "series": method,
                    "method": method,
                    "block_id": int(block_id),
                    "policy": str(ordered["policy"].iloc[0]),
                    "cell_code": str(ordered["cell_code"].iloc[0]),
                    "input_start_cycle": int(ordered["input_start_cycle"].iloc[0]),
                    "input_end_cycle": int(ordered["input_end_cycle"].iloc[0]),
                }
            )
            rows.append(stats)
    diagnostics = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for series, part in diagnostics.groupby("series", sort=False):
        total_pairs = int(part.shape[0] * max(int(pred_long["horizon_step"].nunique()) - 1, 1))
        count = int(part["monotonic_violation_count"].sum())
        summary_rows.append(
            {
                "series": series,
                "monotonic_violation_count": count,
                "monotonic_violation_rate": float(count / total_pairs) if total_pairs else 0.0,
                "max_positive_jump": float(part["max_positive_jump"].max()) if len(part) else 0.0,
                "mean_positive_jump": safe_positive_jump_mean(part),
                "total_positive_jump": float(part["total_positive_jump"].sum()) if len(part) else 0.0,
                "curve_has_violation_rate": float(part["curve_has_violation"].mean()) if len(part) else 0.0,
                "curve_count": int(len(part)),
            }
        )
    return diagnostics, pd.DataFrame(summary_rows)


def safe_positive_jump_mean(part: pd.DataFrame) -> float:
    """Recover aggregate mean positive jump from per-curve totals and counts."""

    total = float(part["total_positive_jump"].sum()) if len(part) else 0.0
    count = int(part["monotonic_violation_count"].sum()) if len(part) else 0
    return float(total / count) if count > 0 else 0.0


def cummin_projection(values: np.ndarray) -> np.ndarray:
    """Project a curve by taking the cumulative minimum from H1 to HM."""

    return np.minimum.accumulate(np.asarray(values, dtype=np.float64)).astype(np.float32)


def isotonic_decreasing_projection(values: np.ndarray) -> np.ndarray:
    """Fit an L2 isotonic regression curve constrained to be non-increasing."""

    y = np.asarray(values, dtype=np.float64).reshape(-1)
    x = np.arange(1, y.size + 1, dtype=np.float64)
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    return model.fit_transform(x, y).astype(np.float32)


def apply_monotonic_postprocessing(pred_long: pd.DataFrame, block_meta: pd.DataFrame) -> pd.DataFrame:
    """Apply cummin, isotonic, and bounded monotonic projections to baseline predictions."""

    meta_last = block_meta.set_index("block_id")["last_history_retention"].to_dict()
    rows: List[pd.DataFrame] = []
    keep_methods = [method for method in POSTPROCESS_METHODS if method in set(pred_long["method"].astype(str))]
    for method in keep_methods:
        part = pred_long.loc[pred_long["method"] == method].copy()
        rows.append(part.copy())
        for block_id, group in part.groupby("block_id", sort=False):
            ordered = group.sort_values("horizon_step").copy()
            original = ordered["pred_retention"].to_numpy(dtype=float)
            projections = {
                f"{method}_cummin": cummin_projection(original),
                f"{method}_isotonic": isotonic_decreasing_projection(original),
                f"{method}_bounded_monotonic": isotonic_decreasing_projection(
                    np.minimum(original, float(meta_last.get(block_id, original[0])))
                ),
            }
            for new_method, pred in projections.items():
                new_rows = ordered.copy()
                new_rows["method"] = new_method
                new_rows["pred_retention"] = pred
                new_rows["residual_retention"] = new_rows["retention_true"].to_numpy(dtype=float) - pred
                if "q_ref" in new_rows.columns:
                    new_rows["pred_q_discharge"] = new_rows["pred_retention"].to_numpy(dtype=float) * new_rows["q_ref"].to_numpy(dtype=float)
                rows.append(new_rows)
    return pd.concat(rows, ignore_index=True)


def metric_rows_from_predictions(pred_long: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Calculate H-wise and all-horizon metrics from a long prediction table."""

    rows: List[Dict[str, object]] = []
    set_types = pred_long["set_type"].unique().tolist() if "set_type" in pred_long.columns else ["valid"]
    for set_type in set_types:
        set_part = pred_long if set_type == "valid" and "set_type" not in pred_long.columns else pred_long.loc[pred_long["set_type"] == set_type]
        for method, method_part in set_part.groupby("method", sort=False):
            for step, group in method_part.groupby("horizon_step", sort=True):
                row = regression_metrics(group["retention_true"].to_numpy(dtype=float), group["pred_retention"].to_numpy(dtype=float))
                row.update(
                    {
                        "stage": stage,
                        "method": method,
                        "set_type": set_type,
                        "target": "retention",
                        "horizon": f"H{int(step)}",
                        "horizon_step": int(step),
                    }
                )
                rows.append(row)
            row = regression_metrics(
                method_part["retention_true"].to_numpy(dtype=float),
                method_part["pred_retention"].to_numpy(dtype=float),
            )
            row.update(
                {
                    "stage": stage,
                    "method": method,
                    "set_type": set_type,
                    "target": "retention",
                    "horizon": "all",
                    "horizon_step": 0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def attach_monotonic_rates(metrics: pd.DataFrame, pred_long: pd.DataFrame) -> pd.DataFrame:
    """Attach aggregate monotonic violation rates to every metric row."""

    eval_frame = pred_long.loc[pred_long["set_type"] == "valid"].copy() if "set_type" in pred_long.columns else pred_long.copy()
    _diag, summary = monotonic_diagnostics(eval_frame, sorted(eval_frame["method"].astype(str).unique()))
    rate_map = summary.set_index("series")["monotonic_violation_rate"].to_dict()
    curve_map = summary.set_index("series")["curve_has_violation_rate"].to_dict()
    result = metrics.copy()
    result["monotonic_violation_rate"] = result["method"].map(rate_map).fillna(np.nan)
    result["curve_has_violation_rate"] = result["method"].map(curve_map).fillna(np.nan)
    return result


def make_lstm_input(
    samples: Sequence[BlockSample],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    retention_standardizer: RetentionStandardizer,
    include_history_retention: bool,
) -> np.ndarray:
    """Create normalized LSTM sequence input from block history arrays."""

    x = np.stack([sample.history_x for sample in samples]).astype(np.float32)
    x_norm = (x - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)
    if not include_history_retention:
        return x_norm.astype(np.float32)
    history_ret = np.stack([sample.history_retention for sample in samples]).astype(np.float32)
    history_ret_z = retention_standardizer.transform(history_ret).reshape(history_ret.shape[0], history_ret.shape[1], 1)
    return np.concatenate([x_norm, history_ret_z], axis=2).astype(np.float32)


def compute_feature_standardizer(samples: Sequence[BlockSample]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate train-only feature mean and standard deviation."""

    stacked = np.stack([sample.history_x for sample in samples]).astype(np.float32)
    mean = np.nanmean(stacked, axis=(0, 1)).astype(np.float32)
    std = np.nanstd(stacked, axis=(0, 1)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def compute_retention_standardizer(y_train: np.ndarray) -> RetentionStandardizer:
    """Calculate train-only scalar retention standardization."""

    mean = float(np.nanmean(np.asarray(y_train, dtype=np.float32)))
    std = float(np.nanstd(np.asarray(y_train, dtype=np.float32)))
    if std < 1e-6:
        std = 1.0
    return RetentionStandardizer(mean=mean, std=std)


def prediction_from_raw(raw: torch.Tensor, last_z: torch.Tensor, mode: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convert model raw output into standardized retention predictions."""

    if mode == "penalty":
        return raw, None
    if mode == "delta":
        delta = F.softplus(raw)
        pred_z = last_z.reshape(-1, 1) - torch.cumsum(delta, dim=1)
        return pred_z, delta
    raise ValueError(f"Unknown LSTM mode: {mode}")


def monotonic_violation_loss(pred_z: torch.Tensor) -> torch.Tensor:
    """Penalize positive future retention jumps."""

    if pred_z.shape[1] <= 1:
        return torch.zeros((), dtype=pred_z.dtype, device=pred_z.device)
    return torch.relu(pred_z[:, 1:] - pred_z[:, :-1]).pow(2).mean()


def smoothness_loss(pred_z: torch.Tensor) -> torch.Tensor:
    """Penalize second-order roughness in predicted retention."""

    if pred_z.shape[1] <= 2:
        return torch.zeros((), dtype=pred_z.dtype, device=pred_z.device)
    return (pred_z[:, 2:] - 2.0 * pred_z[:, 1:-1] + pred_z[:, :-2]).pow(2).mean()


def delta_smoothness_loss(delta: Optional[torch.Tensor]) -> torch.Tensor:
    """Penalize first-order roughness in non-negative deltas."""

    if delta is None or delta.shape[1] <= 1:
        if delta is None:
            return torch.zeros(())
        return torch.zeros((), dtype=delta.dtype, device=delta.device)
    return (delta[:, 1:] - delta[:, :-1]).pow(2).mean()


def numpy_monotonic_violation_rate(pred: np.ndarray) -> float:
    """Calculate positive-difference violation rate for a prediction matrix."""

    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape[1] <= 1:
        return 0.0
    diff = np.diff(arr, axis=1)
    return float(np.mean(diff > 0.0))


def h50_rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate H-last RMSE."""

    err = np.asarray(y_true, dtype=np.float64)[:, -1] - np.asarray(y_pred, dtype=np.float64)[:, -1]
    return float(np.sqrt(np.mean(err**2)))


def append_epoch_row(path: Path, row: Mapping[str, object]) -> None:
    """Append one epoch log row to CSV so progress is durable."""

    frame = pd.DataFrame([dict(row)])
    header = not path.exists()
    frame.to_csv(path, mode="a", header=header, index=False, encoding=ENCODING)


def train_one_lstm(
    method: str,
    mode: str,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train_z: np.ndarray,
    y_valid_z: np.ndarray,
    y_train_actual: np.ndarray,
    y_valid_actual: np.ndarray,
    last_train_z: np.ndarray,
    last_valid_z: np.ndarray,
    retention_standardizer: RetentionStandardizer,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    epoch_log_path: Path,
) -> LstmResult:
    """Train one monotonic LSTM variant."""

    train_ds = RetentionSequenceDataset(x_train, y_train_z, last_train_z)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    model = RetentionLSTM(
        input_dim=int(x_train.shape[2]),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        horizon=int(args.horizon),
        delta_output=mode == "delta",
        delta_init_bias=float(args.delta_init_bias),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    x_valid_tensor = torch.as_tensor(x_valid, dtype=torch.float32, device=device)
    y_valid_tensor = torch.as_tensor(y_valid_z, dtype=torch.float32, device=device)
    last_valid_tensor = torch.as_tensor(last_valid_z, dtype=torch.float32, device=device)
    method_dir = out_dir / "checkpoints" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_valid = math.inf
    best_epoch = 0
    best_h50 = math.inf
    bad_epochs = 0
    rows: List[Dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batch_losses: List[float] = []
        for xb, yb, lastb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            lastb = lastb.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            pred_z, delta = prediction_from_raw(raw, lastb, mode)
            mse = F.mse_loss(pred_z, yb)
            if mode == "penalty":
                loss = mse + float(args.lambda_mono) * monotonic_violation_loss(pred_z) + float(args.lambda_smooth) * smoothness_loss(pred_z)
            else:
                loss = mse + float(args.lambda_smooth) * delta_smoothness_loss(delta)
            loss.backward()
            if float(args.gradient_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            raw_valid = model(x_valid_tensor)
            pred_valid_z_tensor, delta_valid = prediction_from_raw(raw_valid, last_valid_tensor, mode)
            valid_mse_z = F.mse_loss(pred_valid_z_tensor, y_valid_tensor)
            if mode == "penalty":
                valid_loss_tensor = (
                    valid_mse_z
                    + float(args.lambda_mono) * monotonic_violation_loss(pred_valid_z_tensor)
                    + float(args.lambda_smooth) * smoothness_loss(pred_valid_z_tensor)
                )
            else:
                valid_loss_tensor = valid_mse_z + float(args.lambda_smooth) * delta_smoothness_loss(delta_valid)
            pred_valid_z = pred_valid_z_tensor.detach().cpu().numpy().astype(np.float32)
        pred_valid_actual = retention_standardizer.inverse(pred_valid_z)
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        valid_loss = float(valid_loss_tensor.detach().cpu().item())
        valid_mse_actual = float(np.mean((np.asarray(y_valid_actual, dtype=np.float32) - pred_valid_actual) ** 2))
        valid_violation_rate = numpy_monotonic_violation_rate(pred_valid_actual)
        valid_h50_rmse = h50_rmse_np(y_valid_actual, pred_valid_actual)
        row = {
            "method": method,
            "epoch": int(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_mse": valid_mse_actual,
            "valid_monotonic_violation_rate": valid_violation_rate,
            "valid_H50_RMSE": valid_h50_rmse,
        }
        rows.append(row)
        append_epoch_row(epoch_log_path, row)
        print(
            f"[{method}] epoch={epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            f"valid_mse={valid_mse_actual:.8f} valid_violation={valid_violation_rate:.6f} "
            f"valid_H50_RMSE={valid_h50_rmse:.6f}",
            flush=True,
        )
        checkpoint = {
            "method": method,
            "mode": mode,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_loss": valid_loss,
            "valid_H50_RMSE": valid_h50_rmse,
            "retention_mean": float(retention_standardizer.mean),
            "retention_std": float(retention_standardizer.std),
            "args": vars(args),
        }
        torch.save(checkpoint, method_dir / "latest.pt")
        if valid_loss < best_valid - 1e-10:
            best_valid = valid_loss
            best_epoch = int(epoch)
            best_h50 = valid_h50_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(checkpoint, method_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(args.patience):
            print(f"[{method}] early stopping at epoch={epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_raw = model(torch.as_tensor(x_train, dtype=torch.float32, device=device))
        valid_raw = model(x_valid_tensor)
        train_z, _train_delta = prediction_from_raw(train_raw, torch.as_tensor(last_train_z, dtype=torch.float32, device=device), mode)
        valid_z, _valid_delta = prediction_from_raw(valid_raw, last_valid_tensor, mode)
    train_pred = retention_standardizer.inverse(train_z.detach().cpu().numpy().astype(np.float32))
    valid_pred = retention_standardizer.inverse(valid_z.detach().cpu().numpy().astype(np.float32))
    return LstmResult(method, train_pred, valid_pred, pd.DataFrame(rows), best_epoch, float(best_valid), float(best_h50))


def train_lstm_methods(
    args: argparse.Namespace,
    train_samples: Sequence[BlockSample],
    valid_samples: Sequence[BlockSample],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train all monotonic LSTM variants and return predictions, metrics, and logs."""

    device = resolve_device(str(args.device))
    y_train_dqdv, y_train_ret, _train_q_ref, _train_q_discharge, _train_cycles = future_arrays(train_samples)
    y_valid_dqdv, y_valid_ret, _valid_q_ref, _valid_q_discharge, _valid_cycles = future_arrays(valid_samples)
    del y_train_dqdv, y_valid_dqdv
    retention_standardizer = compute_retention_standardizer(y_train_ret)
    y_train_z = retention_standardizer.transform(y_train_ret)
    y_valid_z = retention_standardizer.transform(y_valid_ret)
    last_train = np.asarray([sample.history_retention[-1] for sample in train_samples], dtype=np.float32)
    last_valid = np.asarray([sample.history_retention[-1] for sample in valid_samples], dtype=np.float32)
    last_train_z = retention_standardizer.transform(last_train)
    last_valid_z = retention_standardizer.transform(last_valid)
    feature_mean, feature_std = compute_feature_standardizer(train_samples)
    epoch_log_path = out_dir / "epoch_log.csv"
    if epoch_log_path.exists():
        epoch_log_path.unlink()
    strict_train_x = make_lstm_input(train_samples, feature_mean, feature_std, retention_standardizer, False)
    strict_valid_x = make_lstm_input(valid_samples, feature_mean, feature_std, retention_standardizer, False)
    history_train_x = make_lstm_input(train_samples, feature_mean, feature_std, retention_standardizer, True)
    history_valid_x = make_lstm_input(valid_samples, feature_mean, feature_std, retention_standardizer, True)
    print(f"Training monotonic LSTM methods on {device}...", flush=True)
    results = [
        train_one_lstm(
            method="monotonic_lstm_penalty",
            mode="penalty",
            x_train=strict_train_x,
            x_valid=strict_valid_x,
            y_train_z=y_train_z,
            y_valid_z=y_valid_z,
            y_train_actual=y_train_ret,
            y_valid_actual=y_valid_ret,
            last_train_z=last_train_z,
            last_valid_z=last_valid_z,
            retention_standardizer=retention_standardizer,
            args=args,
            device=device,
            out_dir=out_dir,
            epoch_log_path=epoch_log_path,
        ),
        train_one_lstm(
            method="monotonic_lstm_delta_strict",
            mode="delta",
            x_train=strict_train_x,
            x_valid=strict_valid_x,
            y_train_z=y_train_z,
            y_valid_z=y_valid_z,
            y_train_actual=y_train_ret,
            y_valid_actual=y_valid_ret,
            last_train_z=last_train_z,
            last_valid_z=last_valid_z,
            retention_standardizer=retention_standardizer,
            args=args,
            device=device,
            out_dir=out_dir,
            epoch_log_path=epoch_log_path,
        ),
        train_one_lstm(
            method="monotonic_lstm_delta_with_history_retention",
            mode="delta",
            x_train=history_train_x,
            x_valid=history_valid_x,
            y_train_z=y_train_z,
            y_valid_z=y_valid_z,
            y_train_actual=y_train_ret,
            y_valid_actual=y_valid_ret,
            last_train_z=last_train_z,
            last_valid_z=last_valid_z,
            retention_standardizer=retention_standardizer,
            args=args,
            device=device,
            out_dir=out_dir,
            epoch_log_path=epoch_log_path,
        ),
    ]
    train_pred_map = {result.method: result.train_pred for result in results}
    valid_pred_map = {result.method: result.valid_pred for result in results}
    pred_long = pd.concat(
        [
            build_prediction_long("train", train_samples, train_pred_map),
            build_prediction_long("valid", valid_samples, valid_pred_map),
        ],
        ignore_index=True,
    )
    metrics = attach_monotonic_rates(metric_rows_from_predictions(pred_long, "monotonic_lstm_retention_prediction"), pred_long)
    epoch_log = pd.read_csv(epoch_log_path, encoding=ENCODING)
    epoch_summary = pd.DataFrame(
        [
            {
                "method": result.method,
                "best_epoch": int(result.best_epoch),
                "best_valid_loss": float(result.best_valid_loss),
                "best_valid_H50_RMSE": float(result.best_valid_h50_rmse),
            }
            for result in results
        ]
    )
    best_method = str(epoch_summary.sort_values("best_valid_H50_RMSE").iloc[0]["method"])
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_src = checkpoint_dir / best_method / "best.pt"
    latest_src = checkpoint_dir / results[-1].method / "latest.pt"
    if best_src.exists():
        shutil.copyfile(best_src, checkpoint_dir / "best.pt")
    if latest_src.exists():
        shutil.copyfile(latest_src, checkpoint_dir / "latest.pt")
    standardizer_config = {
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_std": feature_std.astype(float).tolist(),
        "retention_mean": float(retention_standardizer.mean),
        "retention_std": float(retention_standardizer.std),
        "strict_input_dim": int(strict_train_x.shape[2]),
        "history_retention_input_dim": int(history_train_x.shape[2]),
    }
    (out_dir / "standardization_config.json").write_text(
        json.dumps(standardizer_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pred_long, metrics, epoch_log, epoch_summary


def sample_for_plot(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a deterministic bounded plot sample."""

    if len(frame) <= int(max_rows):
        return frame.copy()
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(frame), size=int(max_rows), replace=False))
    return frame.iloc[keep].copy()


def add_identity_line(ax: object, x_values: pd.Series, y_values: pd.Series) -> None:
    """Add a Y=X reference line to a scatter axis."""

    values = np.concatenate([x_values.to_numpy(dtype=float), y_values.to_numpy(dtype=float)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = (high - low) * 0.04 if high > low else 0.01
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)


def metric_lookup(metrics: pd.DataFrame, method: str, horizon_step: int, metric: str) -> float:
    """Look up one validation metric value."""

    rows = metrics.loc[
        (metrics["set_type"] == "valid")
        & (metrics["method"] == method)
        & (metrics["horizon_step"] == int(horizon_step))
    ]
    if rows.empty:
        return float("nan")
    return float(rows[metric].iloc[0])


def save_h50_scatter(pred_long: pd.DataFrame, metrics: pd.DataFrame, methods: Sequence[str], out_path: Path, horizon: int, seed: int) -> None:
    """Save H-last true-vs-predicted scatter plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.3 * n_rows), squeeze=False)
    valid = pred_long if "set_type" not in pred_long.columns else pred_long.loc[pred_long["set_type"] == "valid"]
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        part = valid.loc[(valid["method"] == method) & (valid["horizon_step"] == int(horizon))].copy()
        part = sample_for_plot(part, 5000, int(seed) + idx)
        ax.scatter(part["retention_true"], part["pred_retention"], s=14, alpha=0.42, edgecolors="none")
        add_identity_line(ax, part["retention_true"], part["pred_retention"])
        rmse = metric_lookup(metrics, method, int(horizon), "rmse")
        r2 = metric_lookup(metrics, method, int(horizon), "r2")
        ax.set_title(f"{method} H{int(horizon)} RMSE={rmse:.4f}, R2={r2:.3f}")
        ax.set_xlabel(f"X: true retention at future H{int(horizon)}")
        ax.set_ylabel(f"Y: predicted retention at future H{int(horizon)}")
        ax.grid(True, linestyle="--", alpha=0.25)
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_h50_residual_hist(pred_long: pd.DataFrame, metrics: pd.DataFrame, methods: Sequence[str], out_path: Path, horizon: int, seed: int) -> None:
    """Save H-last residual distribution plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.3 * n_rows), squeeze=False)
    valid = pred_long if "set_type" not in pred_long.columns else pred_long.loc[pred_long["set_type"] == "valid"]
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        part = valid.loc[(valid["method"] == method) & (valid["horizon_step"] == int(horizon))].copy()
        part = sample_for_plot(part, 5000, int(seed) + idx)
        ax.hist(part["residual_retention"].dropna(), bins=45, color="#4477AA", alpha=0.82)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        rmse = metric_lookup(metrics, method, int(horizon), "rmse")
        ax.set_title(f"{method} H{int(horizon)} residual, RMSE={rmse:.4f}")
        ax.set_xlabel("X: residual = true retention - predicted retention")
        ax.set_ylabel("Y: block count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_h50_residual_vs_true(pred_long: pd.DataFrame, methods: Sequence[str], out_path: Path, horizon: int, seed: int) -> None:
    """Save H-last residual versus true retention plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.3 * n_rows), squeeze=False)
    valid = pred_long if "set_type" not in pred_long.columns else pred_long.loc[pred_long["set_type"] == "valid"]
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        part = valid.loc[(valid["method"] == method) & (valid["horizon_step"] == int(horizon))].copy()
        part = sample_for_plot(part, 5000, int(seed) + idx)
        ax.scatter(part["retention_true"], part["residual_retention"], s=14, alpha=0.42, edgecolors="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{method} H{int(horizon)} residual vs true")
        ax.set_xlabel(f"X: true retention at future H{int(horizon)}")
        ax.set_ylabel("Y: residual = true - predicted")
        ax.grid(True, linestyle="--", alpha=0.25)
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def choose_curve_blocks(pred_long: pd.DataFrame, method: str, limit: int) -> List[int]:
    """Choose representative blocks with large H-last absolute error."""

    valid = pred_long if "set_type" not in pred_long.columns else pred_long.loc[pred_long["set_type"] == "valid"]
    if method not in set(valid["method"].astype(str)):
        return sorted(valid["block_id"].drop_duplicates().astype(int).head(limit).tolist())
    horizon = int(valid["horizon_step"].max())
    part = valid.loc[(valid["method"] == method) & (valid["horizon_step"] == horizon)].copy()
    part["abs_error"] = (part["retention_true"] - part["pred_retention"]).abs()
    return part.sort_values("abs_error", ascending=False)["block_id"].astype(int).head(limit).tolist()


def save_curve_plot(
    pred_long: pd.DataFrame,
    methods: Sequence[str],
    out_path: Path,
    title: str,
    seed_method: str,
    max_blocks: int = 4,
) -> None:
    """Save selected H1:HM retention curves with explicit axis semantics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = pred_long if "set_type" not in pred_long.columns else pred_long.loc[pred_long["set_type"] == "valid"]
    block_ids = choose_curve_blocks(valid, seed_method, max_blocks)
    fig, axes = plt.subplots(len(block_ids), 1, figsize=(11.0, 3.3 * len(block_ids)), squeeze=False)
    first_method = str(valid["method"].iloc[0])
    for row_idx, block_id in enumerate(block_ids):
        ax = axes[row_idx][0]
        true_part = valid.loc[(valid["block_id"] == block_id) & (valid["method"] == first_method)].sort_values("horizon_step")
        ax.plot(true_part["horizon_step"], true_part["retention_true"], color="black", linewidth=2.2, label="true retention")
        for method in methods:
            part = valid.loc[(valid["block_id"] == block_id) & (valid["method"] == method)].sort_values("horizon_step")
            if part.empty:
                continue
            ax.plot(part["horizon_step"], part["pred_retention"], linewidth=1.5, marker="o", markersize=2.8, label=method)
        ax.set_title(f"{title}: block_id={block_id}")
        ax.set_xlabel("X: future horizon step H1 to HM")
        ax.set_ylabel("Y: retention")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_loss_curve(epoch_log: pd.DataFrame, out_path: Path) -> None:
    """Save train and validation loss curves for all LSTM methods."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for method, part in epoch_log.groupby("method", sort=False):
        part = part.sort_values("epoch")
        ax.plot(part["epoch"], part["train_loss"], marker="o", linewidth=1.5, label=f"{method} train")
        ax.plot(part["epoch"], part["valid_loss"], marker="s", linestyle="--", linewidth=1.5, label=f"{method} valid")
    ax.set_xlabel("X: epoch")
    ax.set_ylabel("Y: training objective loss")
    ax.set_title("Monotonic LSTM loss curve")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def comparison_rows(
    baseline_metrics: pd.DataFrame,
    post_metrics: pd.DataFrame,
    lstm_metrics: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Build the unified method comparison table required by the report."""

    combined = pd.concat([baseline_metrics, post_metrics, lstm_metrics], ignore_index=True)
    combined = combined.loc[combined["set_type"] == "valid"].copy()
    method_specs = [
        ("direct_retention", "direct LightGBM", "55维工况 summary", "否"),
        ("direct_retention_cummin", "direct LightGBM + cummin", "55维工况 summary", "是"),
        ("direct_retention_isotonic", "direct LightGBM + isotonic", "55维工况 summary", "是"),
        ("linear_last10", "linear_last10", "历史 retention", "否"),
        ("linear_last10_cummin", "linear_last10 + cummin", "历史 retention", "是"),
        ("deployable_bridge", "dQdV bridge", "55维工况 -> compact4 -> retention", "否"),
        ("deployable_bridge_cummin", "dQdV bridge + cummin", "55维工况 -> compact4 -> retention", "是"),
        ("monotonic_lstm_penalty", "monotonic LSTM penalty", "100x55", "软约束"),
        ("monotonic_lstm_delta_strict", "monotonic LSTM delta strict", "100x55 + last_history_retention作为递推起点", "硬约束"),
        ("monotonic_lstm_delta_with_history_retention", "monotonic LSTM delta with history retention", "100x56", "硬约束"),
    ]
    rows: List[Dict[str, object]] = []
    for method, label, input_desc, mono_desc in method_specs:
        h_rows = combined.loc[(combined["method"] == method) & (combined["horizon_step"] == int(horizon))]
        all_rows = combined.loc[(combined["method"] == method) & (combined["horizon"] == "all")]
        if h_rows.empty:
            continue
        h_row = h_rows.iloc[0]
        all_row = all_rows.iloc[0] if not all_rows.empty else h_row
        rows.append(
            {
                "method": label,
                "raw_method": method,
                "input": input_desc,
                "monotonic_constraint": mono_desc,
                "H50_RMSE": float(h_row["rmse"]),
                "H50_MAE": float(h_row["mae"]),
                "H50_R2": float(h_row["r2"]),
                "all_RMSE": float(all_row["rmse"]),
                "monotonic_violation_rate": float(h_row.get("monotonic_violation_rate", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    args: argparse.Namespace,
    checks: pd.DataFrame,
    monotonic_summary: pd.DataFrame,
    post_metrics: pd.DataFrame,
    lstm_metrics: pd.DataFrame,
    epoch_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    out_dir: Path,
) -> str:
    """Build the final Chinese Markdown report."""

    horizon = int(args.horizon)
    img = out_dir.resolve().as_posix()
    h50_name = f"H{horizon}"
    true_summary = monotonic_summary.loc[monotonic_summary["series"] == "true_retention"]
    true_curve_rate = float(true_summary["curve_has_violation_rate"].iloc[0]) if not true_summary.empty else float("nan")

    def h50_rmse(label: str) -> float:
        """Return H-last RMSE by display label."""

        rows = comparison.loc[comparison["method"] == label]
        if rows.empty:
            return float("nan")
        return float(rows["H50_RMSE"].iloc[0])

    linear_rmse = h50_rmse("linear_last10")
    best_lstm = comparison.loc[comparison["raw_method"].isin(LSTM_METHODS)].sort_values("H50_RMSE").head(1)
    best_lstm_text = "未运行"
    if not best_lstm.empty:
        best_lstm_text = f"{best_lstm['method'].iloc[0]}，{h50_name} RMSE={float(best_lstm['H50_RMSE'].iloc[0]):.6f}"

    direct_base = h50_rmse("direct LightGBM")
    direct_cummin = h50_rmse("direct LightGBM + cummin")
    linear_cummin = h50_rmse("linear_last10 + cummin")
    bridge_base = h50_rmse("dQdV bridge")
    bridge_cummin = h50_rmse("dQdV bridge + cummin")
    direct_delta = direct_cummin - direct_base if np.isfinite(direct_cummin) and np.isfinite(direct_base) else float("nan")
    linear_delta = linear_cummin - linear_rmse if np.isfinite(linear_cummin) and np.isfinite(linear_rmse) else float("nan")
    bridge_delta = bridge_cummin - bridge_base if np.isfinite(bridge_cummin) and np.isfinite(bridge_base) else float("nan")

    recommendation = (
        "若正式 H100/M50 结果仍显示 LSTM 未超过 linear_last10，则不建议把 LSTM 作为主预测模型，"
        "更建议回到 linear_last10 或 LightGBM + 单调后处理，并把 LSTM 作为残差修正候选。"
    )
    if not best_lstm.empty and np.isfinite(linear_rmse) and float(best_lstm["H50_RMSE"].iloc[0]) < linear_rmse:
        recommendation = (
            "本次结果中最优 LSTM 已超过 linear_last10，可继续把单调 LSTM 作为主路线候选，"
            "但仍应追加不同随机种子和更大 forecast gap 验证稳定性。"
        )

    lines = [
        "# 单调物理约束 + LSTM 多步 retention 预测验证报告",
        "",
        "## 1. 任务摘要",
        "",
        f"- split_name: `{args.split_name}`",
        f"- train_split_path: `{resolved_path_text(args.train_split_path)}`",
        f"- valid_split_path: `{resolved_path_text(args.valid_split_path)}`",
        f"- history_len: `{int(args.history_len)}`",
        f"- horizon: `{int(args.horizon)}`",
        f"- block_stride: `{int(args.block_stride)}`",
        f"- sample_mode: `{args.sample_mode}`",
        f"- feature_pack: `{args.feature_pack}`",
        f"- target_pack: `{args.target_pack}`",
        "",
        "本报告验证“未来容量保持率整体单调不升”这一物理约束是否能改善 H1:H50 retention 预测。retention 指容量保持率，即当前放电容量除以同一电芯参考容量；H50 指未来第 50 个预测步。单调不升指预测曲线满足 H1 >= H2 >= ... >= H50。",
        "",
        "## 2. 术语说明",
        "",
        "- `recommended55`：55 个工况统计特征，不包含 `cycles`、`policy`、`cell_code` 或 policy 三元参数。",
        "- `compact4`：4 个 dQdV 中介特征，包括 `main_peak_area`、`main_peak_height_dqdv`、`main_peak_voltage_v`、`main_peak_skewness`。",
        "- `direct LightGBM`：用 55 维工况 summary 直接预测未来 retention。",
        "- `dQdV bridge`：先预测未来 compact4 dQdV，再用 dQdV 预测未来 retention。",
        "- `linear_last10`：用历史最后 10 个 retention 点做线性外推。",
        "- `cummin`：从 H1 到 H50 对预测值做累计最小值，保证曲线不再上升。",
        "- `isotonic`：对单条预测曲线做单调不升的最小二乘投影。",
        "- `bounded_monotonic`：先限制预测不超过历史最后一个 retention，再做单调不升投影。",
        "- `monotonic LSTM penalty`：LSTM 直接输出未来 retention，并在 loss 中惩罚上升段和曲线抖动。",
        "- `monotonic LSTM delta strict`：LSTM 输出非负衰减增量，使用历史最后 retention 作为递推起点，不把历史 retention 当作输入特征。",
        "- `monotonic LSTM delta with history retention`：LSTM 输入为 100x56，额外包含历史 retention 观测，因此不是纯工况输入模型。",
        "",
        "## 3. 数据检查",
        "",
        markdown_table(checks, ["check_item", "value", "expected", "pass_flag", "details"]),
        "",
        "## 4. Stage 0 单调性诊断",
        "",
        markdown_table(monotonic_summary, ["series", "monotonic_violation_count", "monotonic_violation_rate", "max_positive_jump", "mean_positive_jump", "total_positive_jump", "curve_has_violation_rate"]),
        "",
        f"真实 retention 的曲线违反率为 `{true_curve_rate:.6f}`。这表示观测标签本身不严格单调，单调约束在本任务中更像物理去噪假设，而不是逐点标签真值的硬事实。",
        "",
        f"![valid monotonic curves before after]({img}/valid_monotonic_curves_before_after.png)",
        "",
        "图 1 说明：X 轴是未来 horizon step，即 H1 到 H50；Y 轴是 retention；黑线是真实 retention，其余曲线是原始预测和单调后处理预测。关键结论：若后处理曲线更贴近黑线且不再上升，说明单调约束有效；若偏离更大，说明真实短期波动不可忽略。",
        "",
        "## 5. Stage 1 单调后处理指标",
        "",
        markdown_table(
            post_metrics.loc[
                (post_metrics["set_type"] == "valid")
                & ((post_metrics["horizon_step"].isin(selected_horizon_steps(horizon))) | (post_metrics["horizon"] == "all"))
            ].sort_values(["method", "horizon_step"]),
            ["method", "horizon", "rmse", "mae", "mse", "r2", "monotonic_violation_rate"],
        ),
        "",
        f"- direct LightGBM + cummin 的 {h50_name} RMSE 变化：`{direct_delta:.6f}`，负数代表提升。",
        f"- linear_last10 + cummin 的 {h50_name} RMSE 变化：`{linear_delta:.6f}`，负数代表提升。",
        f"- dQdV bridge + cummin 的 {h50_name} RMSE 变化：`{bridge_delta:.6f}`，负数代表提升。",
        "",
        f"![postprocess H50 scatter]({img}/postprocess_h50_scatter.png)",
        "",
        f"图 2 说明：X 轴是真实 {h50_name} retention；Y 轴是预测 {h50_name} retention；虚线是理想预测 `Y=X`；每个点代表一个 valid block。关键结论：点云越贴近虚线，{h50_name} 精度越高。",
        "",
        f"![postprocess H50 residual distribution]({img}/postprocess_h50_residual_distribution.png)",
        "",
        f"图 3 说明：X 轴是 {h50_name} 残差 `真实 retention - 预测 retention`；Y 轴是 block 数量；黑色虚线是 0 残差。关键结论：分布越窄且越靠近 0，后处理越有效。",
        "",
        f"![postprocess H50 residual vs true]({img}/postprocess_h50_residual_vs_true.png)",
        "",
        f"图 4 说明：X 轴是真实 {h50_name} retention；Y 轴是残差。若残差随真实 retention 呈结构性斜率，说明模型在不同衰减阶段有系统偏差。",
        "",
        f"![postprocess selected curves]({img}/postprocess_curves_selected_blocks.png)",
        "",
        "图 5 说明：X 轴是 H1:H50，Y 轴是 retention；黑线是真实曲线，彩色线是原始和单调后处理曲线。关键结论：该图用于判断单调约束是修正预测抖动，还是过度压低未来预测。",
        "",
        "## 6. Stage 2/3 LSTM 指标",
        "",
        markdown_table(
            lstm_metrics.loc[
                (lstm_metrics["set_type"] == "valid")
                & ((lstm_metrics["horizon_step"].isin(selected_horizon_steps(horizon))) | (lstm_metrics["horizon"] == "all"))
            ].sort_values(["method", "horizon_step"]),
            ["method", "horizon", "rmse", "mae", "mse", "r2", "monotonic_violation_rate"],
        ),
        "",
        markdown_table(epoch_summary, ["method", "best_epoch", "best_valid_loss", "best_valid_H50_RMSE"]),
        "",
        f"![loss curve]({img}/loss_curve.png)",
        "",
        "图 6 说明：X 轴是 epoch；Y 轴是训练目标 loss；实线是 train loss，虚线是 valid loss。关键结论：若 valid loss 不下降或快速反弹，说明样本量或输入信息不足以支撑 LSTM 泛化。",
        "",
        f"![valid H50 scatter]({img}/valid_h50_scatter.png)",
        "",
        f"图 7 说明：X 轴是真实 {h50_name} retention；Y 轴是 LSTM 预测 {h50_name} retention；虚线是 `Y=X`。关键结论：对比点云贴合程度判断 LSTM 是否优于基线。",
        "",
        f"![valid H50 residual distribution]({img}/valid_h50_residual_distribution.png)",
        "",
        f"图 8 说明：X 轴是 {h50_name} 残差；Y 轴是 block 数量。关键结论：分布越集中在 0 附近，LSTM 误差越小。",
        "",
        f"![valid H50 residual vs true]({img}/valid_h50_residual_vs_true.png)",
        "",
        f"图 9 说明：X 轴是真实 {h50_name} retention；Y 轴是残差。关键结论：该图用于识别 LSTM 是否在高 retention 或低 retention 区间存在系统偏差。",
        "",
        f"![valid monotonic curves]({img}/valid_monotonic_curves.png)",
        "",
        "图 10 说明：X 轴是 H1:H50；Y 轴是 retention；黑线是真实曲线，彩色线是三种 LSTM 预测。关键结论：delta 两个版本天然单调，penalty 版本是否仍有上升取决于 soft loss 是否足够强。",
        "",
        "## 7. 统一对比",
        "",
        markdown_table(comparison, ["method", "input", "monotonic_constraint", "H50_RMSE", "H50_MAE", "H50_R2", "all_RMSE", "monotonic_violation_rate"]),
        "",
        "## 8. 问题回答",
        "",
        f"1. 真实 retention 在 H1:H50 上是否严格单调？不是。真实曲线违反率为 `{true_curve_rate:.6f}`，说明观测中存在短期上升或噪声。",
        "2. 原始 LightGBM / linear_last10 / dQdV bridge 是否存在明显单调违反？LightGBM 和 dQdV bridge 通常更明显；linear_last10 由于是线性外推，违反程度通常较低。",
        f"3. 单调后处理是否提升 H50 精度？看 Stage 1 的 RMSE 变化：direct `{direct_delta:.6f}`、linear `{linear_delta:.6f}`、bridge `{bridge_delta:.6f}`。",
        f"4. LSTM 单调模型是否优于 LightGBM？当前最优 LSTM 为：{best_lstm_text}。需要与 direct LightGBM 和 linear_last10 的 H50 RMSE 同表比较。",
        "5. 如果 LSTM 没有提升，主要原因优先判断为：linear_last10 已经抓住短期 retention 平滑趋势，其次才是样本量不足；若 Stage 1 后处理也无收益，则单调约束不是主要突破口。",
        f"6. 路线建议：{recommendation}",
    ]
    return "\n".join(lines) + "\n"


def write_run_config(
    args: argparse.Namespace,
    out_dir: Path,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    train_samples: Sequence[BlockSample],
    valid_samples: Sequence[BlockSample],
    baseline_source: str,
) -> None:
    """Write reproducibility configuration."""

    config = {
        "split_name": str(args.split_name),
        "train_split_path": resolved_path_text(args.train_split_path),
        "valid_split_path": resolved_path_text(args.valid_split_path),
        "baseline_dir": resolved_path_text(args.baseline_dir),
        "history_len": int(args.history_len),
        "horizon": int(args.horizon),
        "block_stride": int(args.block_stride),
        "sample_mode": str(args.sample_mode),
        "feature_pack": str(args.feature_pack),
        "feature_count": int(len(feature_cols)),
        "target_pack": str(args.target_pack),
        "target_cols": list(target_cols),
        "forbidden_input_columns": sorted(FORBIDDEN_CHECK_COLS),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "lambda_mono": float(args.lambda_mono),
        "lambda_smooth": float(args.lambda_smooth),
        "baseline_source": baseline_source,
        "train_blocks": int(len(train_samples)),
        "valid_blocks": int(len(valid_samples)),
        "input_feature_columns": list(feature_cols),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, object]:
    """Run monotonic diagnostics, postprocessing, LSTM training, and reporting."""

    set_seed(int(args.random_seed))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_samples, valid_samples, feature_cols, target_cols, checks = prepare_samples(args)
    checks.to_csv(out_dir / "dataset_checks.csv", index=False, encoding=ENCODING)
    block_meta = block_metadata_frame([*train_samples, *valid_samples])
    block_meta.to_csv(out_dir / "block_samples.csv", index=False, encoding=ENCODING)
    pd.DataFrame({"rank": np.arange(1, len(feature_cols) + 1), "feature": list(feature_cols)}).to_csv(
        out_dir / "feature_columns.csv",
        index=False,
        encoding=ENCODING,
    )

    print(f"Train blocks={len(train_samples)}, valid blocks={len(valid_samples)}", flush=True)
    print("Loading or training baselines...", flush=True)
    baselines = load_or_train_baselines(args, train_samples, valid_samples, feature_cols)
    valid_baseline = baselines.valid_predictions_long.copy()
    if "set_type" not in valid_baseline.columns:
        valid_baseline.insert(0, "set_type", "valid")
    valid_baseline = valid_baseline.loc[valid_baseline["method"].isin(BASELINE_METHODS)].copy()

    diag, summary = monotonic_diagnostics(valid_baseline, BASELINE_METHODS)
    diag.to_csv(out_dir / "monotonic_diagnostics.csv", index=False, encoding=ENCODING)
    summary.to_csv(out_dir / "monotonic_violation_summary.csv", index=False, encoding=ENCODING)

    post_long = apply_monotonic_postprocessing(valid_baseline, block_meta)
    post_metrics = attach_monotonic_rates(metric_rows_from_predictions(post_long, "monotonic_postprocess_retention_prediction"), post_long)
    post_long.to_csv(out_dir / "postprocess_monotonic_predictions_long.csv", index=False, encoding=ENCODING)
    post_metrics.to_csv(out_dir / "postprocess_monotonic_metrics.csv", index=False, encoding=ENCODING)

    print("Training monotonic LSTM models...", flush=True)
    lstm_long, lstm_metrics, epoch_log, epoch_summary = train_lstm_methods(args, train_samples, valid_samples, out_dir)
    lstm_long.to_csv(out_dir / "train_valid_predictions_long.csv", index=False, encoding=ENCODING)
    lstm_metrics.to_csv(out_dir / "train_valid_metrics_by_horizon.csv", index=False, encoding=ENCODING)
    epoch_summary.to_csv(out_dir / "lstm_epoch_summary.csv", index=False, encoding=ENCODING)
    epoch_log.to_csv(out_dir / "loss_curve.csv", index=False, encoding=ENCODING)

    horizon = int(args.horizon)
    post_plot_methods = [
        "direct_retention",
        "direct_retention_cummin",
        "direct_retention_isotonic",
        "linear_last10",
        "linear_last10_cummin",
        "deployable_bridge",
        "deployable_bridge_cummin",
    ]
    post_plot_methods = [method for method in post_plot_methods if method in set(post_long["method"].astype(str))]
    lstm_plot_methods = [method for method in LSTM_METHODS if method in set(lstm_long["method"].astype(str))]
    save_curve_plot(
        post_long,
        ["direct_retention", "direct_retention_cummin", "direct_retention_isotonic"],
        out_dir / "valid_monotonic_curves_before_after.png",
        "Before/after monotonic postprocess",
        "direct_retention",
    )
    save_h50_scatter(post_long, post_metrics, post_plot_methods, out_dir / "postprocess_h50_scatter.png", horizon, int(args.random_seed))
    save_h50_residual_hist(
        post_long,
        post_metrics,
        post_plot_methods,
        out_dir / "postprocess_h50_residual_distribution.png",
        horizon,
        int(args.random_seed),
    )
    save_h50_residual_vs_true(
        post_long,
        post_plot_methods,
        out_dir / "postprocess_h50_residual_vs_true.png",
        horizon,
        int(args.random_seed),
    )
    save_curve_plot(
        post_long,
        ["direct_retention", "direct_retention_cummin", "linear_last10", "linear_last10_cummin", "deployable_bridge", "deployable_bridge_cummin"],
        out_dir / "postprocess_curves_selected_blocks.png",
        "Postprocess selected curves",
        "direct_retention",
    )
    save_loss_curve(epoch_log, out_dir / "loss_curve.png")
    save_h50_scatter(lstm_long, lstm_metrics, lstm_plot_methods, out_dir / "valid_h50_scatter.png", horizon, int(args.random_seed))
    save_h50_residual_hist(
        lstm_long,
        lstm_metrics,
        lstm_plot_methods,
        out_dir / "valid_h50_residual_distribution.png",
        horizon,
        int(args.random_seed),
    )
    save_h50_residual_vs_true(
        lstm_long,
        lstm_plot_methods,
        out_dir / "valid_h50_residual_vs_true.png",
        horizon,
        int(args.random_seed),
    )
    save_curve_plot(
        lstm_long,
        lstm_plot_methods,
        out_dir / "valid_monotonic_curves.png",
        "Monotonic LSTM selected curves",
        "monotonic_lstm_penalty",
    )

    baseline_metrics = attach_monotonic_rates(baselines.metrics, valid_baseline)
    comparison = comparison_rows(baseline_metrics, post_metrics, lstm_metrics, horizon)
    comparison.to_csv(out_dir / "method_comparison_summary.csv", index=False, encoding=ENCODING)
    report = build_report(args, checks, summary, post_metrics, lstm_metrics, epoch_summary, comparison, out_dir)
    (out_dir / "monotonic_lstm_report.md").write_text(report, encoding=ENCODING)
    write_run_config(args, out_dir, feature_cols, target_cols, train_samples, valid_samples, baselines.source)
    print(f"Saved outputs to: {out_dir}", flush=True)
    if not comparison.empty:
        print(comparison[["method", "H50_RMSE", "H50_R2", "all_RMSE", "monotonic_violation_rate"]].to_string(index=False), flush=True)
    return {
        "output_dir": str(out_dir),
        "train_blocks": int(len(train_samples)),
        "valid_blocks": int(len(valid_samples)),
        "baseline_source": baselines.source,
    }


def main() -> None:
    """CLI entrypoint."""

    run(parse_args())


if __name__ == "__main__":
    main()
