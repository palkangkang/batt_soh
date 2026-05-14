from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"
sys.path.append(str(REPO_ROOT))

import scripts.train_lstm_charge_delta_ah as base_lstm
import scripts.train_lstm_dqdv_retention as dqdv_retention


MAIN_PEAK_FEATURE_COLUMNS: List[str] = list(dqdv_retention.MAIN_PEAK_FEATURE_COLUMNS)
COMPACT_PEAK_SHAPE_FEATURE_COLUMNS: List[str] = [
    "main_peak_area",
    "main_peak_skewness",
    "main_peak_voltage_v",
    "main_peak_width_v",
]
COMPACT_PEAK_SHAPE_HEIGHT_FEATURE_COLUMNS: List[str] = [
    *COMPACT_PEAK_SHAPE_FEATURE_COLUMNS,
    "main_peak_height_dqdv",
]
MAIN_PEAK_TEMP_CYCLE_FEATURE_COLUMNS: List[str] = [
    *MAIN_PEAK_FEATURE_COLUMNS,
    "cycle_log_scaled",
]
FEATURE_PACK_COLUMNS: Dict[str, List[str]] = {
    "compact_peak_shape": list(COMPACT_PEAK_SHAPE_FEATURE_COLUMNS),
    "compact_peak_shape_height": list(COMPACT_PEAK_SHAPE_HEIGHT_FEATURE_COLUMNS),
    "main_peak_temp_cycle": list(MAIN_PEAK_TEMP_CYCLE_FEATURE_COLUMNS),
}
FEATURE_PACK_DESCRIPTIONS: Dict[str, str] = {
    "compact_peak_shape": "四个dQdV主峰形状特征，不含cycle_index_norm",
    "compact_peak_shape_height": "四个dQdV主峰形状特征 + 主峰高度，不含cycle_index_norm",
    "main_peak_temp_cycle": "主峰9维 + cycle_log_scaled，多步无泄漏全量特征变体",
}


@dataclass
class MultiStepSequence:
    """Container for one policy-cell sequence."""

    policy: str
    cell_code: str
    set_type: str
    cycles: np.ndarray
    x: np.ndarray
    retention: np.ndarray
    q_discharge: np.ndarray
    q_ref: np.ndarray


@dataclass
class MultiStepMeta:
    """Metadata for one 1:N -> N+1:N+H sample."""

    policy: str
    cell_code: str
    input_cycle: int
    target_start_cycle: int
    target_end_cycle: int
    set_type: str


@dataclass
class MetricRow:
    """Regression metric row for a method, split, aggregation, and horizon."""

    target: str
    method: str
    set_type: str
    aggregation: str
    horizon: str
    n_windows: int
    n_groups: int
    mse: float
    rmse: float
    mae: float
    r2: float


class MultiStepDqdvDataset(Dataset):
    """Dataset whose sample uses dQdV history through N to predict N+1:N+H."""

    def __init__(
        self,
        sequences: Mapping[Tuple[str, str], MultiStepSequence],
        min_history: int,
        horizon_steps: int,
        max_windows: int | None,
        seed: int,
    ) -> None:
        self._sequences = dict(sequences)
        self._min_history = int(min_history)
        self._horizon_steps = int(horizon_steps)
        self._refs: List[Tuple[Tuple[str, str], int]] = []
        self._metas: List[MultiStepMeta] = []
        self._lengths: List[int] = []
        self._build_refs()
        self._sort_by_length()
        if max_windows is not None and max_windows > 0 and len(self._refs) > max_windows:
            self._downsample(max_windows=max_windows, seed=seed)
            self._sort_by_length()

    def __len__(self) -> int:
        """Return number of multistep windows."""

        return len(self._refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return one prefix input and future retention target."""

        key, input_idx = self._refs[index]
        seq = self._sequences[key]
        seq_len = int(input_idx + 1)
        x_prefix = seq.x[:seq_len]
        y_future = seq.retention[input_idx + 1 : input_idx + 1 + self._horizon_steps]
        return (
            torch.from_numpy(x_prefix),
            torch.from_numpy(y_future.astype(np.float32)),
            index,
            seq_len,
        )

    @property
    def metas(self) -> Sequence[MultiStepMeta]:
        """Return metadata for all samples."""

        return self._metas

    @property
    def lengths(self) -> Sequence[int]:
        """Return input prefix lengths for all samples."""

        return self._lengths

    @property
    def horizon_steps(self) -> int:
        """Return future horizon length."""

        return self._horizon_steps

    def target_arrays(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return retention, q_discharge, and q_ref targets for selected sample indices."""

        ret_rows: List[np.ndarray] = []
        q_rows: List[np.ndarray] = []
        qref_rows: List[np.ndarray] = []
        for raw_idx in indices.tolist():
            key, input_idx = self._refs[int(raw_idx)]
            seq = self._sequences[key]
            target_slice = slice(input_idx + 1, input_idx + 1 + self._horizon_steps)
            ret_rows.append(seq.retention[target_slice].astype(np.float32))
            q_rows.append(seq.q_discharge[target_slice].astype(np.float32))
            qref_rows.append(seq.q_ref[target_slice].astype(np.float32))
        return np.stack(ret_rows), np.stack(q_rows), np.stack(qref_rows)

    def baseline_arrays(self, indices: np.ndarray, method: str) -> np.ndarray:
        """Return baseline future retention predictions for selected sample indices."""

        pred_rows: List[np.ndarray] = []
        for raw_idx in indices.tolist():
            key, input_idx = self._refs[int(raw_idx)]
            seq = self._sequences[key]
            if method == "persistence":
                pred_rows.append(np.full(self._horizon_steps, float(seq.retention[input_idx]), dtype=np.float32))
            elif method == "linear_last10":
                pred_rows.append(self._linear_last10(seq=seq, input_idx=input_idx))
            else:
                raise ValueError(f"Unknown baseline method: {method}")
        return np.stack(pred_rows)

    def _build_refs(self) -> None:
        """Build valid references with complete consecutive future horizons."""

        refs: List[Tuple[Tuple[str, str], int]] = []
        metas: List[MultiStepMeta] = []
        lengths: List[int] = []
        for key, seq in self._sequences.items():
            n_rows = int(seq.retention.shape[0])
            last_input_idx = n_rows - self._horizon_steps - 1
            if last_input_idx < self._min_history - 1:
                continue
            for input_idx in range(self._min_history - 1, last_input_idx + 1):
                target_cycles = seq.cycles[input_idx + 1 : input_idx + 1 + self._horizon_steps]
                expected_cycles = seq.cycles[input_idx] + np.arange(1, self._horizon_steps + 1, dtype=np.int32)
                if not np.array_equal(target_cycles.astype(np.int32), expected_cycles):
                    continue
                refs.append((key, input_idx))
                metas.append(
                    MultiStepMeta(
                        policy=seq.policy,
                        cell_code=seq.cell_code,
                        input_cycle=int(seq.cycles[input_idx]),
                        target_start_cycle=int(target_cycles[0]),
                        target_end_cycle=int(target_cycles[-1]),
                        set_type=seq.set_type,
                    )
                )
                lengths.append(int(input_idx + 1))
        self._refs = refs
        self._metas = metas
        self._lengths = lengths

    def _downsample(self, max_windows: int, seed: int) -> None:
        """Randomly sample a stable subset of windows."""

        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self._refs), size=int(max_windows), replace=False))
        self._refs = [self._refs[int(i)] for i in keep]
        self._metas = [self._metas[int(i)] for i in keep]
        self._lengths = [self._lengths[int(i)] for i in keep]

    def _sort_by_length(self) -> None:
        """Sort samples by prefix length to reduce batch padding cost."""

        order = sorted(range(len(self._refs)), key=lambda i: (self._lengths[i], self._metas[i].policy, self._metas[i].cell_code))
        self._refs = [self._refs[i] for i in order]
        self._metas = [self._metas[i] for i in order]
        self._lengths = [self._lengths[i] for i in order]

    def _linear_last10(self, seq: MultiStepSequence, input_idx: int) -> np.ndarray:
        """Extrapolate future retention from the last ten observed history points."""

        start_idx = max(0, int(input_idx) - 9)
        x_hist = seq.cycles[start_idx : input_idx + 1].astype(np.float64)
        y_hist = seq.retention[start_idx : input_idx + 1].astype(np.float64)
        x_future = seq.cycles[input_idx + 1 : input_idx + 1 + self._horizon_steps].astype(np.float64)
        if x_hist.size < 2 or float(np.ptp(x_hist)) <= 0.0:
            return np.full(self._horizon_steps, float(seq.retention[input_idx]), dtype=np.float32)
        slope, intercept = np.polyfit(x_hist, y_hist, deg=1)
        return (slope * x_future + intercept).astype(np.float32)


class LSTMMultiHorizonRegressor(nn.Module):
    """LSTM encoder with a direct multi-horizon regression head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon_steps: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        proj_hidden = max(16, hidden_size // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, horizon_steps),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a padded prefix sequence and predict all future horizons."""

        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        return self.head(last_hidden)


class ShortHorizonAwareMSELoss(nn.Module):
    """Weighted MSE loss with optional short-horizon bias corrections."""

    def __init__(
        self,
        horizon_weights: torch.Tensor,
        short_horizon_count: int,
        short_bias_penalty: float,
        short_underprediction_penalty: float,
    ) -> None:
        super().__init__()
        self.register_buffer("horizon_weights", horizon_weights.reshape(1, -1).float())
        self.short_horizon_count = int(short_horizon_count)
        self.short_bias_penalty = float(short_bias_penalty)
        self.short_underprediction_penalty = float(short_underprediction_penalty)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute horizon-aware loss for a batch of multi-step predictions."""

        residual = pred - target
        loss = torch.mean(torch.square(residual) * self.horizon_weights)
        short_count = min(max(int(self.short_horizon_count), 0), int(pred.shape[1]))
        if short_count <= 0:
            return loss
        short_residual = residual[:, :short_count]
        if self.short_bias_penalty > 0.0:
            loss = loss + self.short_bias_penalty * torch.mean(torch.square(torch.mean(short_residual, dim=0)))
        if self.short_underprediction_penalty > 0.0:
            underprediction = torch.relu(target[:, :short_count] - pred[:, :short_count])
            loss = loss + self.short_underprediction_penalty * torch.mean(torch.square(underprediction))
        return loss


def build_horizon_loss_weights(horizon_steps: int, short_horizon_count: int, short_horizon_weight: float) -> np.ndarray:
    """Build mean-one horizon weights that emphasize early forecast steps."""

    if horizon_steps < 1:
        raise ValueError("--horizon-steps must be >= 1")
    if short_horizon_count < 0:
        raise ValueError("--short-horizon-count must be >= 0")
    if short_horizon_weight <= 0.0:
        raise ValueError("--short-horizon-weight must be > 0")
    weights = np.ones(int(horizon_steps), dtype=np.float32)
    short_count = min(int(short_horizon_count), int(horizon_steps))
    if short_count > 0:
        weights[:short_count] = np.float32(short_horizon_weight)
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0:
        raise ValueError("Horizon loss weights must have positive mean")
    return weights / np.float32(mean_weight)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multistep dQdV training."""

    parser = argparse.ArgumentParser(
        description="Train dQdV LSTM 1:N -> N+1:N+H retention trajectory predictor."
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "lstm_dqdv_multistep_h50",
    )
    parser.add_argument(
        "--feature-pack",
        type=str,
        choices=sorted(FEATURE_PACK_COLUMNS),
        default="compact_peak_shape_height",
        help=(
            "Feature pack: compact_peak_shape (4 dims), compact_peak_shape_height (5 dims), "
            "or main_peak_temp_cycle (9 observed main-peak/temp features + leakage-safe cycle_log_scaled)."
        ),
    )
    parser.add_argument("--horizon-steps", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=30)
    parser.add_argument("--cycle-log-scale", type=float, default=3000.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--short-horizon-count",
        type=int,
        default=10,
        help="Number of earliest horizons receiving short-step loss emphasis. Set 0 to recover uniform all-window loss.",
    )
    parser.add_argument(
        "--short-horizon-weight",
        type=float,
        default=3.0,
        help="Relative MSE weight for horizons 1:short_horizon_count before mean-one normalization.",
    )
    parser.add_argument(
        "--short-bias-penalty",
        type=float,
        default=0.25,
        help="Penalty coefficient for mean prediction bias on short horizons.",
    )
    parser.add_argument(
        "--short-underprediction-penalty",
        type=float,
        default=0.0,
        help="Optional extra MSE penalty for underprediction on short horizons.",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument(
        "--resume-interrupted",
        dest="resume_interrupted",
        action="store_true",
        default=True,
        help="Resume epoch-level interrupted training from latest state file.",
    )
    parser.add_argument(
        "--no-resume-interrupted",
        dest="resume_interrupted",
        action="store_false",
        help="Disable resume and restart from epoch 1.",
    )
    parser.add_argument("--best-state-file", type=str, default="best.pt")
    parser.add_argument("--latest-state-file", type=str, default="latest.pt")
    parser.add_argument("--epoch-log-file", type=str, default="epoch_log.csv")
    parser.add_argument("--status-file", type=str, default="runtime_status.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def get_feature_columns(feature_pack: str) -> List[str]:
    """Return input feature columns for a named feature pack."""

    if feature_pack not in FEATURE_PACK_COLUMNS:
        valid = ", ".join(sorted(FEATURE_PACK_COLUMNS))
        raise ValueError(f"Unknown feature_pack={feature_pack!r}. Valid values: {valid}")
    return list(FEATURE_PACK_COLUMNS[feature_pack])


def describe_feature_pack(feature_pack: str) -> str:
    """Return human-readable feature pack description."""

    return FEATURE_PACK_DESCRIPTIONS.get(feature_pack, feature_pack)


def set_seed(seed: int) -> None:
    """Set all random seeds used by the training pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_cycle_log_scaled(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Add a cycle feature that does not depend on each cell's max cycle."""

    if float(scale) <= 0:
        raise ValueError("--cycle-log-scale must be positive.")
    out = df.copy()
    cycles = pd.to_numeric(out["cycles"], errors="coerce").astype(float)
    out["cycle_log_scaled"] = (np.log1p(cycles) / math.log1p(float(scale))).astype(np.float32)
    return out


def coerce_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Coerce model inputs and targets to finite float columns."""

    out = df.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float32)
    out["retention"] = pd.to_numeric(out["retention"], errors="coerce").astype(np.float32)
    out["q_discharge"] = pd.to_numeric(out["q_discharge"], errors="coerce").astype(np.float32)
    out["q_ref"] = pd.to_numeric(out["q_ref"], errors="coerce").astype(np.float32)
    return out


def build_sequences(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[Tuple[str, str], MultiStepSequence]:
    """Build per-cell sequences for multistep forecasting."""

    seq_map: Dict[Tuple[str, str], MultiStepSequence] = {}
    for (policy, cell_code), part in merged.groupby(["policy", "cell_code"], sort=False):
        part = part.sort_values("cycles", kind="mergesort").copy()
        set_types = part["set_type"].dropna().unique().tolist()
        if len(set_types) != 1:
            raise RuntimeError(f"Split leakage detected for key {(policy, cell_code)}.")
        seq_map[(str(policy), str(cell_code))] = MultiStepSequence(
            policy=str(policy),
            cell_code=str(cell_code),
            set_type=str(set_types[0]),
            cycles=part["cycles"].to_numpy(dtype=np.int32),
            x=part[list(feature_cols)].to_numpy(dtype=np.float32),
            retention=part["retention"].to_numpy(dtype=np.float32),
            q_discharge=part["q_discharge"].to_numpy(dtype=np.float32),
            q_ref=part["q_ref"].to_numpy(dtype=np.float32),
        )
    return seq_map


def split_sequences(
    seq_map: Mapping[Tuple[str, str], MultiStepSequence],
) -> Tuple[Dict[Tuple[str, str], MultiStepSequence], Dict[Tuple[str, str], MultiStepSequence]]:
    """Split per-cell sequences into train and validation mappings."""

    train_map: Dict[Tuple[str, str], MultiStepSequence] = {}
    valid_map: Dict[Tuple[str, str], MultiStepSequence] = {}
    for key, seq in seq_map.items():
        if seq.set_type == "train":
            train_map[key] = seq
        elif seq.set_type == "valid":
            valid_map[key] = seq
        else:
            raise RuntimeError(f"Unknown set_type: {seq.set_type}")
    return train_map, valid_map


def collate_multistep_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length prefix inputs and keep descending length order."""

    sorted_batch = sorted(batch, key=lambda item: item[3], reverse=True)
    xs = [item[0] for item in sorted_batch]
    ys = torch.stack([item[1] for item in sorted_batch], dim=0)
    idxs = torch.tensor([int(item[2]) for item in sorted_batch], dtype=torch.long)
    lengths = torch.tensor([int(item[3]) for item in sorted_batch], dtype=torch.long)
    padded_x = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return padded_x, ys, idxs, lengths


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a DataLoader for multistep samples."""

    return DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_multistep_batch,
    )


def build_resume_signature_payload(args: argparse.Namespace, device: torch.device, feature_cols: Sequence[str]) -> Dict[str, Any]:
    """Build deterministic payload used to validate resumed runs."""

    return {
        "feature_pack": str(args.feature_pack),
        "feature_columns": list(feature_cols),
        "input_size": int(len(feature_cols)),
        "horizon_steps": int(args.horizon_steps),
        "min_history": int(args.min_history),
        "cycle_log_scale": float(args.cycle_log_scale),
        "device": str(device),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "short_horizon_count": int(args.short_horizon_count),
        "short_horizon_weight": float(args.short_horizon_weight),
        "short_bias_penalty": float(args.short_bias_penalty),
        "short_underprediction_penalty": float(args.short_underprediction_penalty),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "max_train_windows": int(args.max_train_windows),
        "max_valid_windows": int(args.max_valid_windows),
        "q_min": float(args.q_min),
        "q_max": float(args.q_max),
        "q_ref_cycles": int(args.q_ref_cycles),
        "retention_min": float(args.retention_min),
        "retention_max": float(args.retention_max),
        "seed": int(args.seed),
    }


def build_resume_signature(payload: Mapping[str, Any]) -> str:
    """Hash a resume payload into a stable signature."""

    canonical = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""

    model.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch, _, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch, lengths)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = int(y_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Evaluate average loss on one loader."""

    model.eval()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch, _, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch, lengths)
        loss = criterion(pred, y_batch)
        batch_size = int(y_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict over one DataLoader."""

    model.eval()
    true_rows: List[np.ndarray] = []
    pred_rows: List[np.ndarray] = []
    idx_rows: List[np.ndarray] = []
    for x_batch, y_batch, idx_batch, lengths in loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch, lengths).detach().cpu().numpy().astype(np.float32)
        pred_rows.append(pred)
        true_rows.append(y_batch.numpy().astype(np.float32))
        idx_rows.append(idx_batch.numpy().astype(np.int64))
    return (
        np.concatenate(true_rows, axis=0),
        np.concatenate(pred_rows, axis=0),
        np.concatenate(idx_rows, axis=0),
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 with NaN fallback for degenerate samples."""

    if y_true.size < 2:
        return float("nan")
    if float(np.nanmax(y_true) - np.nanmin(y_true)) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def calc_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute MSE, RMSE, MAE, and R2."""

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = safe_r2(y_true, y_pred)
    return mse, rmse, mae, r2


def macro_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Sequence[Tuple[str, str]],
) -> Tuple[int, float, float, float, float]:
    """Compute equal-weight policy-cell macro metrics."""

    group_df = pd.DataFrame(
        {
            "group": [f"{p}||{c}" for p, c in groups],
            "y_true": y_true.astype(float),
            "y_pred": y_pred.astype(float),
        }
    )
    rows: List[Tuple[float, float, float, float]] = []
    for _, part in group_df.groupby("group", sort=False):
        yt = part["y_true"].to_numpy(dtype=float)
        yp = part["y_pred"].to_numpy(dtype=float)
        rows.append(calc_basic_metrics(yt, yp))
    arr = np.asarray(rows, dtype=float)
    return (
        int(len(rows)),
        float(np.nanmean(arr[:, 0])),
        float(math.sqrt(np.nanmean(arr[:, 0]))),
        float(np.nanmean(arr[:, 2])),
        float(np.nanmean(arr[:, 3])),
    )


def build_metric_rows(
    target: str,
    method: str,
    set_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metas: Sequence[MultiStepMeta],
    selected_horizons: Sequence[int],
) -> List[MetricRow]:
    """Build weighted and macro metric rows for selected horizons and all points."""

    rows: List[MetricRow] = []
    groups = [(m.policy, m.cell_code) for m in metas]
    horizon_count = int(y_true.shape[1])
    for horizon in selected_horizons:
        if horizon > horizon_count:
            continue
        h_idx = int(horizon) - 1
        yt = y_true[:, h_idx]
        yp = y_pred[:, h_idx]
        mse, rmse, mae, r2 = calc_basic_metrics(yt, yp)
        n_groups, mmse, mrmse, mmae, mr2 = macro_group_metrics(yt, yp, groups)
        rows.append(
            MetricRow(target, method, set_type, "weighted", str(horizon), int(yt.size), n_groups, mse, rmse, mae, r2)
        )
        rows.append(
            MetricRow(
                target,
                method,
                set_type,
                "macro_policy_cell",
                str(horizon),
                int(yt.size),
                n_groups,
                mmse,
                mrmse,
                mmae,
                mr2,
            )
        )

    yt_all = y_true.reshape(-1)
    yp_all = y_pred.reshape(-1)
    all_groups: List[Tuple[str, str]] = []
    for meta in metas:
        all_groups.extend([(meta.policy, meta.cell_code)] * horizon_count)
    mse, rmse, mae, r2 = calc_basic_metrics(yt_all, yp_all)
    n_groups, mmse, mrmse, mmae, mr2 = macro_group_metrics(yt_all, yp_all, all_groups)
    rows.append(MetricRow(target, method, set_type, "weighted", "all", int(yt_all.size), n_groups, mse, rmse, mae, r2))
    rows.append(
        MetricRow(
            target,
            method,
            set_type,
            "macro_policy_cell",
            "all",
            int(yt_all.size),
            n_groups,
            mmse,
            mrmse,
            mmae,
            mr2,
        )
    )
    return rows


def build_predictions_long(
    dataset: MultiStepDqdvDataset,
    idxs: np.ndarray,
    model_pred_ret: np.ndarray,
    persistence_pred_ret: np.ndarray,
    linear_pred_ret: np.ndarray,
) -> pd.DataFrame:
    """Build long-form validation predictions for all horizons."""

    true_ret, true_q, q_ref = dataset.target_arrays(idxs)
    metas = [dataset.metas[int(i)] for i in idxs.tolist()]
    rows: List[Dict[str, object]] = []
    for sample_i, meta in enumerate(metas):
        for h_idx in range(dataset.horizon_steps):
            qref_val = float(q_ref[sample_i, h_idx])
            model_ret = float(model_pred_ret[sample_i, h_idx])
            persistence_ret = float(persistence_pred_ret[sample_i, h_idx])
            linear_ret = float(linear_pred_ret[sample_i, h_idx])
            true_ret_val = float(true_ret[sample_i, h_idx])
            true_q_val = float(true_q[sample_i, h_idx])
            model_q = float(model_ret * qref_val)
            persistence_q = float(persistence_ret * qref_val)
            linear_q = float(linear_ret * qref_val)
            rows.append(
                {
                    "policy": meta.policy,
                    "cell_code": meta.cell_code,
                    "input_cycle": int(meta.input_cycle),
                    "horizon": int(h_idx + 1),
                    "target_cycle": int(meta.input_cycle + h_idx + 1),
                    "retention_true": true_ret_val,
                    "q_discharge_true": true_q_val,
                    "pred_retention_model": model_ret,
                    "pred_q_discharge_model": model_q,
                    "pred_retention_persistence": persistence_ret,
                    "pred_q_discharge_persistence": persistence_q,
                    "pred_retention_linear_last10": linear_ret,
                    "pred_q_discharge_linear_last10": linear_q,
                    "residual_retention_model": true_ret_val - model_ret,
                    "residual_q_discharge_model": true_q_val - model_q,
                    "residual_retention_persistence": true_ret_val - persistence_ret,
                    "residual_q_discharge_persistence": true_q_val - persistence_q,
                    "residual_retention_linear_last10": true_ret_val - linear_ret,
                    "residual_q_discharge_linear_last10": true_q_val - linear_q,
                    "q_ref": qref_val,
                }
            )
    return pd.DataFrame(rows)


def build_dataset_checks(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    train_dataset: MultiStepDqdvDataset,
    valid_dataset: MultiStepDqdvDataset,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Build consistency checks proving split integrity and no future-feature leakage."""

    train_keys = set(
        merged.loc[merged["set_type"] == "train", ["policy", "cell_code"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )
    valid_keys = set(
        merged.loc[merged["set_type"] == "valid", ["policy", "cell_code"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )
    feature_mat = merged[list(feature_cols)].to_numpy(dtype=np.float32)
    all_metas = [*train_dataset.metas, *valid_dataset.metas]
    target_after_input = all(
        int(m.input_cycle) < int(m.target_start_cycle) <= int(m.target_end_cycle) for m in all_metas
    )
    expected_horizon = all(
        int(m.target_end_cycle) - int(m.input_cycle) == int(args.horizon_steps) for m in all_metas
    )
    lengths = np.asarray([*train_dataset.lengths, *valid_dataset.lengths], dtype=int)
    checks = [
        ("check_feature_dim_expected", int(feature_mat.shape[1] == len(feature_cols))),
        ("check_feature_nan_free", int(np.isfinite(feature_mat).all())),
        ("check_split_overlap_zero", int(len(train_keys.intersection(valid_keys)) == 0)),
        ("check_input_max_cycle_lt_target_min_cycle", int(target_after_input)),
        ("check_future_dqdv_not_used_by_dataset", int(target_after_input)),
        ("check_target_cycle_consecutive_horizon", int(expected_horizon)),
        ("check_cycle_feature_not_cell_max_norm", int("cycle_index_norm" not in feature_cols)),
        ("check_cycle_log_scale_fixed_positive", int(float(args.cycle_log_scale) > 0.0)),
        ("check_min_history_respected", int(lengths.size == 0 or int(lengths.min()) >= int(args.min_history))),
        ("check_horizon_steps_positive", int(int(args.horizon_steps) > 0)),
    ]
    return pd.DataFrame(checks, columns=["check_item", "pass_flag"])


def save_loss_plot(loss_df: pd.DataFrame, out_path: Path) -> None:
    """Save train/valid loss curve."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss", color="#2563eb")
    ax.plot(loss_df["epoch"], loss_df["valid_loss"], label="valid_loss", color="#dc2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean horizon MSE")
    ax.set_title("dQdV Multistep LSTM Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def save_horizon_scatter_plot(
    pred_long: pd.DataFrame,
    out_path: Path,
    horizons: Sequence[int],
    target: str,
) -> None:
    """Save true-vs-pred scatter panels for selected horizons."""

    import matplotlib.pyplot as plt

    if target == "q_discharge":
        true_col = "q_discharge_true"
        pred_col = "pred_q_discharge_model"
        axis_label = "q_discharge"
    elif target == "retention":
        true_col = "retention_true"
        pred_col = "pred_retention_model"
        axis_label = "retention"
    else:
        raise ValueError(f"Unsupported scatter target: {target}")

    n_cols = len(horizons)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.8), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        part = pred_long.loc[pred_long["horizon"] == int(horizon)]
        if part.empty:
            ax.set_axis_off()
            continue
        y_true = part[true_col].to_numpy(dtype=float)
        y_pred = part[pred_col].to_numpy(dtype=float)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.scatter(y_true, y_pred, s=8, alpha=0.35, color="#2563eb")
        ax.plot([lo, hi], [lo, hi], "--", color="#dc2626", linewidth=1.2)
        ax.set_title(f"h={horizon}")
        ax.set_xlabel(f"True {axis_label}")
        ax.set_ylabel(f"Pred {axis_label}")
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def build_report(
    args: argparse.Namespace,
    device: torch.device,
    feature_cols: Sequence[str],
    merged_rows: int,
    train_rows: int,
    valid_rows: int,
    best_epoch: int,
    metrics_df: pd.DataFrame,
    checks_df: pd.DataFrame,
) -> str:
    """Build a Markdown report for the multistep dQdV run."""

    lines: List[str] = []
    lines.append("# LSTM 训练报告：dQdV 多步容量保持率预测")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 解释器：`{os.path.realpath(sys.executable)}`")
    lines.append(f"- 设备：`{device.type}`")
    lines.append(f"- 任务口径：`1:N -> N+1:N+{int(args.horizon_steps)}`")
    lines.append(f"- 最小历史长度：`{int(args.min_history)}`")
    lines.append(f"- 特征包：`{args.feature_pack}`，{describe_feature_pack(str(args.feature_pack))}")
    if "cycle_log_scaled" in set(feature_cols):
        lines.append(
            f"- cycle 特征：`cycle_log_scaled=log1p(cycles)/log1p({float(args.cycle_log_scale):.1f})`"
        )
    else:
        lines.append("- cycle 特征：未使用 `cycle_index_norm` 或其他循环位置特征")
    lines.append(f"- q 过滤：`{args.q_min} <= q_discharge <= {args.q_max}`")
    lines.append(
        f"- retention 过滤：`{args.retention_min} <= retention <= {args.retention_max}`，"
        f"`q_ref`=前 `{args.q_ref_cycles}` 个有效循环中位数"
    )
    lines.append(
        "- 训练损失："
        f"前 `{int(args.short_horizon_count)}` 个 horizon 的 MSE 相对权重 `{float(args.short_horizon_weight):.3g}`，"
        f"短步长 bias penalty `{float(args.short_bias_penalty):.3g}`，"
        f"underprediction penalty `{float(args.short_underprediction_penalty):.3g}`"
    )
    lines.append("")
    lines.append("## 2. 数据概览")
    lines.append(f"- 合并后 cycle 级样本数：**{merged_rows:,}**")
    lines.append(f"- 训练多步窗口数：**{train_rows:,}**")
    lines.append(f"- 验证多步窗口数：**{valid_rows:,}**")
    lines.append(f"- 每个时间步输入维度：`{len(feature_cols)}`")
    lines.append("- 输入特征：")
    for col in feature_cols:
        lines.append(f"  - `{col}`")
    lines.append("")
    lines.append("## 3. 关键指标")
    final_horizon = str(int(args.horizon_steps))
    key = metrics_df.loc[
        (metrics_df["set_type"] == "valid")
        & (metrics_df["aggregation"] == "weighted")
        & (metrics_df["horizon"].isin(["1", "10", final_horizon, "all"]))
    ].copy()
    lines.append("| target | method | horizon | n_windows | MSE | RMSE | MAE | R2 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in key.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.method} | {row.horizon} | {int(row.n_windows)} | "
            f"{float(row.mse):.8f} | {float(row.rmse):.6f} | {float(row.mae):.6f} | {float(row.r2):.6f} |"
        )
    lines.append("")
    lines.append("## 4. 防泄漏检查")
    lines.append("| check_item | pass_flag |")
    lines.append("|---|---:|")
    for row in checks_df.itertuples(index=False):
        lines.append(f"| {row.check_item} | {int(row.pass_flag)} |")
    lines.append("")
    lines.append("## 5. 图表")
    lines.append(f"- 最佳 epoch：**{best_epoch}**")
    lines.append("![loss_curve](./loss_curve.png)")
    lines.append("")
    lines.append("![valid_scatter_horizons](./valid_scatter_horizons.png)")
    lines.append("")
    lines.append("![valid_retention_scatter_horizons](./valid_retention_scatter_horizons.png)")
    lines.append("")
    lines.append("## 6. 说明")
    lines.append("- 模型一次性直接输出未来全窗口，不使用 autoregressive decoder。")
    lines.append("- 目标窗口只用 retention 标签，不把未来 dQ/dV 特征喂给模型。")
    return "\n".join(lines)


def prepare_datasets(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str], MultiStepDqdvDataset, MultiStepDqdvDataset]:
    """Load feature/label data and build train/valid multistep datasets."""

    feature_cols = get_feature_columns(str(args.feature_pack))
    split_df = base_lstm.load_split_map(args.train_split_path, args.valid_split_path)
    feature_df = dqdv_retention.load_dqdv_main_feature_table(args.dqdv_path)
    label_df = dqdv_retention.load_retention_labels(
        life_path=args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )
    merged = dqdv_retention.merge_feature_label_split(feature_df=feature_df, label_df=label_df, split_df=split_df)
    merged = add_cycle_log_scaled(merged, scale=float(args.cycle_log_scale))
    merged = coerce_feature_columns(merged, feature_cols)
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after dQdV/retention/split join.")

    seq_map = build_sequences(merged=merged, feature_cols=feature_cols)
    train_seq_map, valid_seq_map = split_sequences(seq_map)
    train_dataset = MultiStepDqdvDataset(
        sequences=train_seq_map,
        min_history=int(args.min_history),
        horizon_steps=int(args.horizon_steps),
        max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
        seed=int(args.seed),
    )
    valid_dataset = MultiStepDqdvDataset(
        sequences=valid_seq_map,
        min_history=int(args.min_history),
        horizon_steps=int(args.horizon_steps),
        max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
        seed=int(args.seed) + 1,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Train or valid multistep windows are empty. Check min_history/horizon_steps.")
    return merged, feature_cols, train_dataset, valid_dataset


def evaluate_methods(
    model: nn.Module,
    train_dataset: MultiStepDqdvDataset,
    valid_dataset: MultiStepDqdvDataset,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    selected_horizons: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model and baselines on train/valid datasets."""

    metric_rows: List[MetricRow] = []
    valid_pred_long = pd.DataFrame()
    for set_type, dataset, loader in [
        ("train", train_dataset, train_loader),
        ("valid", valid_dataset, valid_loader),
    ]:
        _, model_pred_ret, idxs = predict_loader(model=model, loader=loader, device=device)
        true_ret, _, q_ref = dataset.target_arrays(idxs)
        method_preds = {
            "dqdv_multistep_lstm": model_pred_ret,
            "persistence": dataset.baseline_arrays(idxs, "persistence"),
            "linear_last10": dataset.baseline_arrays(idxs, "linear_last10"),
        }
        metas = [dataset.metas[int(i)] for i in idxs.tolist()]
        for method, pred_ret in method_preds.items():
            metric_rows.extend(
                build_metric_rows(
                    target="retention",
                    method=method,
                    set_type=set_type,
                    y_true=true_ret,
                    y_pred=pred_ret,
                    metas=metas,
                    selected_horizons=selected_horizons,
                )
            )
            metric_rows.extend(
                build_metric_rows(
                    target="q_discharge",
                    method=method,
                    set_type=set_type,
                    y_true=true_ret * q_ref,
                    y_pred=pred_ret * q_ref,
                    metas=metas,
                    selected_horizons=selected_horizons,
                )
            )
        if set_type == "valid":
            valid_pred_long = build_predictions_long(
                dataset=dataset,
                idxs=idxs,
                model_pred_ret=method_preds["dqdv_multistep_lstm"],
                persistence_pred_ret=method_preds["persistence"],
                linear_pred_ret=method_preds["linear_last10"],
            )
    metrics_df = pd.DataFrame([row.__dict__ for row in metric_rows])
    return metrics_df, valid_pred_long


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the complete multistep dQdV training pipeline."""

    if args.smoke_test:
        args.epochs = min(int(args.epochs), 3)
        args.patience = min(int(args.patience), 2)
        args.max_train_windows = int(args.max_train_windows) if int(args.max_train_windows) > 0 else 512
        args.max_valid_windows = int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else 256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_lstm.ensure_matplotlib_backend()
    set_seed(int(args.seed))
    device = base_lstm.resolve_device(str(args.device))

    merged, feature_cols, train_dataset, valid_dataset = prepare_datasets(args)
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
    )
    valid_loader = build_dataloader(
        dataset=valid_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
    )

    model = LSTMMultiHorizonRegressor(
        input_size=len(feature_cols),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        horizon_steps=int(args.horizon_steps),
    ).to(device)
    if float(args.short_bias_penalty) < 0.0:
        raise ValueError("--short-bias-penalty must be >= 0")
    if float(args.short_underprediction_penalty) < 0.0:
        raise ValueError("--short-underprediction-penalty must be >= 0")
    horizon_loss_weights = build_horizon_loss_weights(
        horizon_steps=int(args.horizon_steps),
        short_horizon_count=int(args.short_horizon_count),
        short_horizon_weight=float(args.short_horizon_weight),
    )
    criterion = ShortHorizonAwareMSELoss(
        horizon_weights=torch.from_numpy(horizon_loss_weights).to(device),
        short_horizon_count=int(args.short_horizon_count),
        short_bias_penalty=float(args.short_bias_penalty),
        short_underprediction_penalty=float(args.short_underprediction_penalty),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    out_metrics = args.output_dir / "train_valid_metrics_by_horizon.csv"
    out_preds = args.output_dir / "valid_predictions_long.csv"
    out_loss = args.output_dir / "loss_curve.csv"
    out_loss_png = args.output_dir / "loss_curve.png"
    out_scatter = args.output_dir / "valid_scatter_horizons.png"
    out_retention_scatter = args.output_dir / "valid_retention_scatter_horizons.png"
    out_checks = args.output_dir / "dataset_checks.csv"
    out_config = args.output_dir / "run_config.json"
    out_report = args.output_dir / "lstm_dqdv_multistep_report.md"
    best_ckpt_path = args.output_dir / str(args.best_state_file)
    latest_state_path = args.output_dir / str(args.latest_state_file)
    epoch_log_path = args.output_dir / str(args.epoch_log_file)
    status_path = args.output_dir / str(args.status_file)

    signature_payload = build_resume_signature_payload(args=args, device=device, feature_cols=feature_cols)
    args_signature = build_resume_signature(signature_payload)
    start_epoch = 1
    best_epoch = 0
    best_valid_loss = float("inf")
    no_improve_count = 0
    loss_rows: List[Dict[str, float | int]] = []
    epoch_log_columns = [
        "timestamp",
        "epoch",
        "train_loss",
        "valid_loss",
        "is_best_epoch",
        "best_valid_loss",
        "no_improve_count",
    ]

    if bool(args.resume_interrupted) and latest_state_path.exists():
        latest_state = torch.load(latest_state_path, map_location=device)
        if str(latest_state.get("args_signature", "")) != args_signature:
            raise RuntimeError(
                "Resume signature mismatch. Disable resume with --no-resume-interrupted or use a new output dir."
            )
        model.load_state_dict(latest_state["model_state_dict"])
        optimizer.load_state_dict(latest_state["optimizer_state_dict"])
        start_epoch = int(latest_state["epoch"]) + 1
        best_epoch = int(latest_state.get("best_epoch", 0))
        best_valid_loss = float(latest_state.get("best_valid_loss", float("inf")))
        no_improve_count = int(latest_state.get("no_improve_count", 0))
        stored_rows = latest_state.get("loss_rows", [])
        if isinstance(stored_rows, list):
            loss_rows = [dict(row) for row in stored_rows]
        print(
            f"Resumed from epoch {start_epoch - 1}: best_epoch={best_epoch}, "
            f"best_valid={best_valid_loss:.8f}",
            flush=True,
        )
    elif not bool(args.resume_interrupted):
        print("Resume disabled by --no-resume-interrupted, start from epoch 1.", flush=True)

    if start_epoch <= 1:
        pd.DataFrame(columns=epoch_log_columns).to_csv(epoch_log_path, index=False, encoding="utf-8")

    base_lstm.write_training_status(
        path=status_path,
        status={
            "current_epoch": int(max(start_epoch - 1, 0)),
            "total_epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_valid_loss": None if not math.isfinite(best_valid_loss) else float(best_valid_loss),
            "no_improve_count": int(no_improve_count),
            "finished": False,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args_signature": args_signature,
            "signature_payload": signature_payload,
        },
    )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
        valid_loss = eval_loss(model=model, loader=valid_loader, criterion=criterion, device=device)
        improved = (best_valid_loss - valid_loss) > float(args.min_delta)
        if improved:
            best_valid_loss = float(valid_loss)
            best_epoch = int(epoch)
            no_improve_count = 0
            base_lstm.atomic_torch_save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": int(len(feature_cols)),
                        "hidden_size": int(args.hidden_size),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                        "horizon_steps": int(args.horizon_steps),
                        "feature_pack": str(args.feature_pack),
                        "feature_columns": list(feature_cols),
                        "loss_config": {
                            "short_horizon_count": int(args.short_horizon_count),
                            "short_horizon_weight": float(args.short_horizon_weight),
                            "short_bias_penalty": float(args.short_bias_penalty),
                            "short_underprediction_penalty": float(args.short_underprediction_penalty),
                            "horizon_loss_weights": [float(v) for v in horizon_loss_weights.tolist()],
                        },
                    },
                    "best_valid_loss": float(best_valid_loss),
                    "best_epoch": int(best_epoch),
                },
                path=best_ckpt_path,
            )
        else:
            no_improve_count += 1

        loss_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "is_best_epoch": int(improved),
            }
        )
        pd.DataFrame(loss_rows).to_csv(out_loss, index=False, encoding="utf-8")
        dqdv_retention.append_epoch_progress(
            path=epoch_log_path,
            row={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "is_best_epoch": int(improved),
                "best_valid_loss": float(best_valid_loss),
                "no_improve_count": int(no_improve_count),
            },
        )
        base_lstm.atomic_torch_save(
            {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid_loss),
                "no_improve_count": int(no_improve_count),
                "args_signature": args_signature,
                "signature_payload": signature_payload,
                "loss_rows": list(loss_rows),
            },
            path=latest_state_path,
        )
        base_lstm.write_training_status(
            path=status_path,
            status={
                "current_epoch": int(epoch),
                "total_epochs": int(args.epochs),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid_loss),
                "no_improve_count": int(no_improve_count),
                "finished": False,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "args_signature": args_signature,
                "signature_payload": signature_payload,
            },
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.8f} | valid_loss={valid_loss:.8f} | "
            f"best_valid={best_valid_loss:.8f} | no_improve={no_improve_count}",
            flush=True,
        )
        if no_improve_count >= int(args.patience):
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).", flush=True)
            break

    if not best_ckpt_path.exists():
        raise RuntimeError("Best checkpoint was not saved.")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_pred_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
    )
    selected_horizons = [1, 5, 10, 20, int(args.horizon_steps)]
    metrics_df, valid_pred_long = evaluate_methods(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_loader=train_pred_loader,
        valid_loader=valid_loader,
        device=device,
        selected_horizons=selected_horizons,
    )
    checks_df = build_dataset_checks(
        merged=merged,
        feature_cols=feature_cols,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args,
    )
    loss_df = pd.DataFrame(loss_rows)

    run_config = {
        "script": str(SCRIPT_PATH),
        "python_executable": os.path.realpath(sys.executable),
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "args_signature": args_signature,
        "signature_payload": signature_payload,
        "feature_columns": list(feature_cols),
        "loss_config": {
            "short_horizon_count": int(args.short_horizon_count),
            "short_horizon_weight": float(args.short_horizon_weight),
            "short_bias_penalty": float(args.short_bias_penalty),
            "short_underprediction_penalty": float(args.short_underprediction_penalty),
            "horizon_loss_weights": [float(v) for v in horizon_loss_weights.tolist()],
        },
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "train_windows": int(len(train_dataset)),
        "valid_windows": int(len(valid_dataset)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8")
    valid_pred_long.to_csv(out_preds, index=False, encoding="utf-8")
    loss_df.to_csv(out_loss, index=False, encoding="utf-8")
    checks_df.to_csv(out_checks, index=False, encoding="utf-8")
    out_config.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_loss_plot(loss_df=loss_df, out_path=out_loss_png)
    save_horizon_scatter_plot(
        pred_long=valid_pred_long,
        out_path=out_scatter,
        horizons=[h for h in [1, 10, int(args.horizon_steps)] if h <= int(args.horizon_steps)],
        target="q_discharge",
    )
    save_horizon_scatter_plot(
        pred_long=valid_pred_long,
        out_path=out_retention_scatter,
        horizons=[h for h in [1, 10, int(args.horizon_steps)] if h <= int(args.horizon_steps)],
        target="retention",
    )
    out_report.write_text(
        build_report(
            args=args,
            device=device,
            feature_cols=feature_cols,
            merged_rows=int(len(merged)),
            train_rows=int(len(train_dataset)),
            valid_rows=int(len(valid_dataset)),
            best_epoch=int(best_epoch),
            metrics_df=metrics_df,
            checks_df=checks_df,
        ),
        encoding="utf-8",
    )
    base_lstm.write_training_status(
        path=status_path,
        status={
            "current_epoch": int(loss_df["epoch"].max()) if not loss_df.empty else int(max(start_epoch - 1, 0)),
            "total_epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_valid_loss": float(best_valid_loss),
            "no_improve_count": int(no_improve_count),
            "finished": True,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args_signature": args_signature,
            "signature_payload": signature_payload,
        },
    )
    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_preds}")
    print(f"Saved: {out_loss}")
    print(f"Saved: {out_checks}")
    print(f"Saved: {out_report}")
    return {
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "output_dir": str(args.output_dir),
    }


def main() -> None:
    """CLI entrypoint."""

    run_training(parse_args())


if __name__ == "__main__":
    main()
