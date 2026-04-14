from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"


@dataclass
class SequenceData:
    """Container of one cell-level sequence."""

    policy: str
    cell_code: str
    set_type: str
    cycles: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass
class WindowMeta:
    """Metadata for one sliding window sample."""

    policy: str
    cell_code: str
    cycles: int
    set_type: str


@dataclass
class Metrics:
    """Regression metrics container."""

    set_type: str
    n_windows: int
    mse: float
    rmse: float
    mae: float
    r2: float


class SlidingWindowDataset(Dataset):
    """Window dataset backed by per-cell sequences."""

    def __init__(
        self,
        sequences: Mapping[Tuple[str, str], SequenceData],
        window_size: int,
        max_windows: int | None,
        seed: int,
    ) -> None:
        self._sequences = sequences
        self._window_size = window_size
        self._refs: List[Tuple[Tuple[str, str], int]] = []
        self._metas: List[WindowMeta] = []
        self._lengths: List[int] = []
        self._build_refs()
        if max_windows is not None and max_windows > 0 and len(self._refs) > max_windows:
            self._downsample(max_windows=max_windows, seed=seed)

    def __len__(self) -> int:
        """Return number of windows."""

        return len(self._refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return one window sample."""

        key, end_idx = self._refs[index]
        seq = self._sequences[key]
        start_idx = end_idx - self._window_size + 1
        x_window = seq.x[start_idx : end_idx + 1]
        y_target = seq.y[end_idx]
        return (
            torch.from_numpy(x_window),
            torch.tensor(y_target, dtype=torch.float32),
            index,
            int(self._window_size),
        )

    @property
    def metas(self) -> Sequence[WindowMeta]:
        """Get metadata list for all windows."""

        return self._metas

    @property
    def lengths(self) -> Sequence[int]:
        """Get sequence lengths of all samples."""

        return self._lengths

    def _build_refs(self) -> None:
        """Build all valid window references."""

        refs: List[Tuple[Tuple[str, str], int]] = []
        metas: List[WindowMeta] = []
        lengths: List[int] = []
        for key, seq in self._sequences.items():
            length = int(seq.y.shape[0])
            if length < self._window_size:
                continue
            for end_idx in range(self._window_size - 1, length):
                refs.append((key, end_idx))
                metas.append(
                    WindowMeta(
                        policy=seq.policy,
                        cell_code=seq.cell_code,
                        cycles=int(seq.cycles[end_idx]),
                        set_type=seq.set_type,
                    )
                )
                lengths.append(int(self._window_size))
        self._refs = refs
        self._metas = metas
        self._lengths = lengths

    def _downsample(self, max_windows: int, seed: int) -> None:
        """Randomly sample a subset of windows."""

        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self._refs), size=max_windows, replace=False))
        self._refs = [self._refs[int(i)] for i in keep]
        self._metas = [self._metas[int(i)] for i in keep]
        self._lengths = [self._lengths[int(i)] for i in keep]


class PrefixHistoryDataset(Dataset):
    """Full-history prefix dataset: sample t uses x[0:t+1]."""

    def __init__(
        self,
        sequences: Mapping[Tuple[str, str], SequenceData],
        max_windows: int | None,
        seed: int,
    ) -> None:
        self._sequences = sequences
        self._refs: List[Tuple[Tuple[str, str], int]] = []
        self._metas: List[WindowMeta] = []
        self._lengths: List[int] = []
        self._build_refs()
        if max_windows is not None and max_windows > 0 and len(self._refs) > max_windows:
            self._downsample(max_windows=max_windows, seed=seed)

    def __len__(self) -> int:
        """Return number of prefix samples."""

        return len(self._refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return one prefix sample."""

        key, end_idx = self._refs[index]
        seq = self._sequences[key]
        seq_len = int(end_idx + 1)
        x_prefix = seq.x[:seq_len]
        y_target = seq.y[end_idx]
        return (
            torch.from_numpy(x_prefix),
            torch.tensor(y_target, dtype=torch.float32),
            index,
            seq_len,
        )

    @property
    def metas(self) -> Sequence[WindowMeta]:
        """Get metadata list for all prefix samples."""

        return self._metas

    @property
    def lengths(self) -> Sequence[int]:
        """Get sequence lengths of all prefix samples."""

        return self._lengths

    def _build_refs(self) -> None:
        """Build all prefix references."""

        refs: List[Tuple[Tuple[str, str], int]] = []
        metas: List[WindowMeta] = []
        lengths: List[int] = []
        for key, seq in self._sequences.items():
            n = int(seq.y.shape[0])
            for end_idx in range(n):
                refs.append((key, end_idx))
                metas.append(
                    WindowMeta(
                        policy=seq.policy,
                        cell_code=seq.cell_code,
                        cycles=int(seq.cycles[end_idx]),
                        set_type=seq.set_type,
                    )
                )
                lengths.append(int(end_idx + 1))
        self._refs = refs
        self._metas = metas
        self._lengths = lengths

    def _downsample(self, max_windows: int, seed: int) -> None:
        """Randomly sample a subset of prefix samples."""

        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self._refs), size=max_windows, replace=False))
        self._refs = [self._refs[int(i)] for i in keep]
        self._metas = [self._metas[int(i)] for i in keep]
        self._lengths = [self._lengths[int(i)] for i in keep]


def collate_sequence_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and keep descending length order."""

    sorted_batch = sorted(batch, key=lambda item: item[3], reverse=True)
    xs = [item[0] for item in sorted_batch]
    ys = torch.stack([item[1] for item in sorted_batch], dim=0)
    idxs = torch.tensor([int(item[2]) for item in sorted_batch], dtype=torch.long)
    lengths = torch.tensor([int(item[3]) for item in sorted_batch], dtype=torch.long)
    padded_x = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return padded_x, ys, idxs, lengths


class LSTMRegressor(nn.Module):
    """Simple LSTM sequence regressor."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
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
            nn.Linear(proj_hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with variable-length packed sequence."""

        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(
        description="Train LSTM on charge delta_ah interval features to fit q_discharge."
    )
    parser.add_argument(
        "--charge-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "charge_interval_features.csv",
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
        default=REPO_ROOT / "outputs" / "analysis" / "lstm_charge_delta_ah_q_discharge",
    )
    parser.add_argument(
        "--sequence-mode",
        type=str,
        choices=["fixed_window", "prefix_full"],
        default="fixed_window",
        help="fixed_window: fixed-length sliding window; prefix_full: sample t uses full history 1..t.",
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260407)
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
        help="Disable epoch-level resume and start training from epoch 1.",
    )
    parser.add_argument("--latest-state-file", type=str, default="latest_lstm_state.pt")
    parser.add_argument("--epoch-log-file", type=str, default="epoch_progress.csv")
    parser.add_argument("--status-file", type=str, default="training_status.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set global random seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device."""

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_matplotlib_backend() -> None:
    """Force Agg backend for headless plotting."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to file atomically."""

    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding=encoding)
    os.replace(tmp_path, path)


def atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    """Save torch object atomically."""

    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def build_resume_signature_payload(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Build signature payload used by resume consistency checks."""

    return {
        "sequence_mode": str(args.sequence_mode),
        "device": str(device),
        "max_train_windows": int(args.max_train_windows),
        "max_valid_windows": int(args.max_valid_windows),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }


def build_resume_signature(payload: Mapping[str, Any]) -> str:
    """Build deterministic hash string for resume payload."""

    canonical = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_training_status(path: Path, status: Mapping[str, Any]) -> None:
    """Write current training status snapshot atomically."""

    atomic_write_text(path, json.dumps(dict(status), ensure_ascii=False, indent=2), encoding="utf-8")


def append_epoch_progress(path: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch progress row to CSV."""

    epoch_df = pd.DataFrame([dict(row)])
    if not path.exists():
        epoch_df.to_csv(path, index=False, encoding="utf-8")
    else:
        epoch_df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")


def parse_range_low(range_label: str) -> float:
    """Parse low bound from range label."""

    try:
        return float(str(range_label).strip()[1:].split(",")[0])
    except Exception:
        return float("inf")


def load_split_map(train_split_path: Path, valid_split_path: Path) -> pd.DataFrame:
    """Load split map with set_type."""

    train = pd.read_csv(train_split_path, encoding=ENCODING, usecols=["policy", "cell_code"]).copy()
    valid = pd.read_csv(valid_split_path, encoding=ENCODING, usecols=["policy", "cell_code"]).copy()
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    split = pd.concat([train, valid], ignore_index=True)
    split["policy"] = split["policy"].astype(str)
    split["cell_code"] = split["cell_code"].astype(str)
    split = split.drop_duplicates(["policy", "cell_code"], keep="first")
    return split


def build_cycle_feature_table(charge_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Build cycle-level wide feature table with masks."""

    charge = pd.read_csv(
        charge_path,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"],
    )
    charge["policy"] = charge["policy"].astype(str)
    charge["cell_code"] = charge["cell_code"].astype(str)
    charge["cycles"] = pd.to_numeric(charge["cycles"], errors="coerce")
    charge["range_count"] = pd.to_numeric(charge["range_count"], errors="coerce")
    charge["delta_ah"] = pd.to_numeric(charge["delta_ah"], errors="coerce")
    charge = charge.dropna(subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]).copy()
    charge["cycles"] = charge["cycles"].astype(int)
    charge["range_count"] = charge["range_count"].astype(int)
    charge = charge.loc[charge["range_count"] == 1].copy()

    agg = (
        charge.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(delta_ah_sum=("delta_ah", "sum"))
        .copy()
    )
    range_order = sorted(agg["range"].dropna().unique().tolist(), key=parse_range_low)
    if len(range_order) != 12:
        raise RuntimeError(f"Expected 12 voltage ranges, but got {len(range_order)}.")

    wide = (
        agg.pivot_table(
            index=["policy", "cell_code", "cycles"],
            columns="range",
            values="delta_ah_sum",
            aggfunc="mean",
        )
        .reset_index()
        .copy()
    )
    for col in range_order:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[["policy", "cell_code", "cycles", *range_order]].copy()

    value_cols: List[str] = []
    mask_cols: List[str] = []
    for col in range_order:
        safe_name = (
            str(col)
            .replace("[", "")
            .replace("]", "")
            .replace(")", "")
            .replace(",", "_")
            .replace(".", "p")
            .replace("-", "m")
        )
        value_col = f"delta_ah_{safe_name}"
        mask_col = f"mask_{safe_name}"
        wide[mask_col] = (~wide[col].isna()).astype(np.float32)
        wide[value_col] = wide[col].fillna(0.0).astype(np.float32)
        value_cols.append(value_col)
        mask_cols.append(mask_col)

    keep_cols = ["policy", "cell_code", "cycles", *value_cols, *mask_cols]
    return wide[keep_cols].copy(), range_order


def load_life_labels(life_path: Path, q_min: float, q_max: float) -> pd.DataFrame:
    """Load q_discharge labels with range filtering."""

    life = pd.read_csv(
        life_path,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    life["policy"] = life["policy"].astype(str)
    life["cell_code"] = life["cell_code"].astype(str)
    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life["cycles"] = life["cycles"].astype(int)
    life = life.loc[(life["q_discharge"] >= q_min) & (life["q_discharge"] <= q_max)].copy()
    life = life.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")
    return life


def merge_dataset(feature_df: pd.DataFrame, label_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Merge feature, label, and split tables."""

    merged = label_df.merge(feature_df, on=["policy", "cell_code", "cycles"], how="inner")
    merged = merged.merge(split_df, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    return merged


def get_value_mask_cols(feature_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer value and mask columns from prefixes."""

    value_cols = sorted([c for c in feature_df.columns if c.startswith("delta_ah_")])
    mask_cols = sorted([c for c in feature_df.columns if c.startswith("mask_")])
    if len(value_cols) != 12 or len(mask_cols) != 12:
        raise RuntimeError(
            f"Expected 12 value + 12 mask columns, got {len(value_cols)} + {len(mask_cols)}."
        )
    return value_cols, mask_cols


def build_sequences(
    merged: pd.DataFrame,
    value_cols: Sequence[str],
    mask_cols: Sequence[str],
) -> Dict[Tuple[str, str], SequenceData]:
    """Build per-cell sequence objects."""

    seq_map: Dict[Tuple[str, str], SequenceData] = {}
    for (policy, cell_code), part in merged.groupby(["policy", "cell_code"], sort=False):
        part = part.sort_values("cycles", kind="mergesort").copy()
        x_val = part[list(value_cols)].to_numpy(dtype=np.float32)
        x_mask = part[list(mask_cols)].to_numpy(dtype=np.float32)
        x = np.concatenate([x_val, x_mask], axis=1)
        y = part["q_discharge"].to_numpy(dtype=np.float32)
        cycles = part["cycles"].to_numpy(dtype=np.int32)
        set_types = part["set_type"].dropna().unique().tolist()
        if len(set_types) != 1:
            raise RuntimeError(f"Split leakage detected for key {(policy, cell_code)}.")
        seq_map[(str(policy), str(cell_code))] = SequenceData(
            policy=str(policy),
            cell_code=str(cell_code),
            set_type=str(set_types[0]),
            cycles=cycles,
            x=x,
            y=y,
        )
    return seq_map


def split_sequence_dict(
    seq_map: Mapping[Tuple[str, str], SequenceData],
) -> Tuple[Dict[Tuple[str, str], SequenceData], Dict[Tuple[str, str], SequenceData]]:
    """Split sequence dict into train and valid maps."""

    train_map: Dict[Tuple[str, str], SequenceData] = {}
    valid_map: Dict[Tuple[str, str], SequenceData] = {}
    for key, seq in seq_map.items():
        if seq.set_type == "train":
            train_map[key] = seq
        elif seq.set_type == "valid":
            valid_map[key] = seq
        else:
            raise RuntimeError(f"Unknown set_type: {seq.set_type}")
    return train_map, valid_map


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    collate_fn: Optional[
        Callable[
            [Sequence[Tuple[torch.Tensor, torch.Tensor, int, int]]],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ] = None,
) -> DataLoader:
    """Create DataLoader with suitable pin_memory."""

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, set_type: str) -> Metrics:
    """Compute regression metrics."""

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(set_type=set_type, n_windows=int(len(y_true)), mse=mse, rmse=rmse, mae=mae, r2=r2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch."""

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
    """Evaluate mean loss on loader."""

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
    """Predict over one loader."""

    model.eval()
    ys_true: List[np.ndarray] = []
    ys_pred: List[np.ndarray] = []
    idxs: List[np.ndarray] = []
    for x_batch, y_batch, idx_batch, lengths in loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch, lengths).detach().cpu().numpy()
        ys_pred.append(pred.astype(np.float32))
        ys_true.append(y_batch.numpy().astype(np.float32))
        idxs.append(idx_batch.numpy().astype(np.int64))
    return (
        np.concatenate(ys_true, axis=0),
        np.concatenate(ys_pred, axis=0),
        np.concatenate(idxs, axis=0),
    )


def save_loss_plot(loss_df: pd.DataFrame, out_path: Path) -> None:
    """Save train/valid loss curve."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss", color="#0ea5e9")
    ax.plot(loss_df["epoch"], loss_df["valid_loss"], label="valid_loss", color="#f97316")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Train/Valid Loss Curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def save_scatter_plot(valid_pred_df: pd.DataFrame, metrics: Metrics, out_path: Path) -> None:
    """Save valid true-vs-pred scatter plot."""

    import matplotlib.pyplot as plt

    y_true = valid_pred_df["q_discharge"].to_numpy(dtype=float)
    y_pred = valid_pred_df["pred_q_discharge"].to_numpy(dtype=float)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_true, y_pred, s=9, alpha=0.35, color="#0ea5e9")
    ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", linewidth=1.4)
    ax.set_xlabel("True q_discharge (Ah)")
    ax.set_ylabel("Pred q_discharge (Ah)")
    ax.set_title(f"Valid Scatter | R2={metrics.r2:.4f} | RMSE={metrics.rmse:.5f}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def build_dataset_checks(
    merged: pd.DataFrame,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    value_cols: Sequence[str],
    mask_cols: Sequence[str],
    sequence_mode: str,
    window_size: int,
) -> pd.DataFrame:
    """Build dataset consistency checks."""

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
    overlap = len(train_keys.intersection(valid_keys))

    value_mat = merged[list(value_cols)].to_numpy(dtype=np.float32)
    mask_mat = merged[list(mask_cols)].to_numpy(dtype=np.float32)
    checks = [
        ("check_range_feature_dim_12", int(len(value_cols) == 12)),
        ("check_mask_feature_dim_12", int(len(mask_cols) == 12)),
        ("check_split_overlap_zero", int(overlap == 0)),
        ("check_mask_values_binary", int(np.isin(mask_mat, np.array([0.0, 1.0], dtype=np.float32)).all())),
        ("check_zero_fill_with_mask", int(np.all((mask_mat == 1.0) | (value_mat == 0.0)))),
    ]
    if sequence_mode == "fixed_window":
        checks.extend(
            [
                (
                    "check_window_size_train",
                    int(len(train_dataset) == 0 or int(train_dataset[0][0].shape[0]) == window_size),
                ),
                (
                    "check_window_size_valid",
                    int(len(valid_dataset) == 0 or int(valid_dataset[0][0].shape[0]) == window_size),
                ),
            ]
        )
    else:
        train_lengths = np.array(getattr(train_dataset, "lengths", []), dtype=int)
        valid_lengths = np.array(getattr(valid_dataset, "lengths", []), dtype=int)
        checks.extend(
            [
                ("check_prefix_min_len_train_ge_1", int(train_lengths.size == 0 or int(train_lengths.min()) >= 1)),
                ("check_prefix_min_len_valid_ge_1", int(valid_lengths.size == 0 or int(valid_lengths.min()) >= 1)),
                ("check_prefix_var_len_train", int(train_lengths.size <= 1 or np.unique(train_lengths).size > 1)),
                ("check_prefix_var_len_valid", int(valid_lengths.size <= 1 or np.unique(valid_lengths).size > 1)),
            ]
        )
    return pd.DataFrame(checks, columns=["check_item", "pass_flag"])


def build_report(
    args: argparse.Namespace,
    device: torch.device,
    range_order: Sequence[str],
    train_metrics: Metrics,
    valid_metrics: Metrics,
    merged_rows: int,
    train_window_rows: int,
    valid_window_rows: int,
    best_epoch: int,
) -> str:
    """Build markdown experiment report in Chinese."""

    lines: List[str] = []
    lines.append("# LSTM 训练报告：charge delta_ah 拟合 q_discharge")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 设备：`{device.type}`")
    lines.append(f"- 序列模式：`{args.sequence_mode}`")
    if args.sequence_mode == "fixed_window":
        lines.append(f"- 窗口长度：`{args.window_size}`")
    else:
        lines.append("- 全历史前缀口径：样本第 `t` 条使用 `1..t` 全部历史序列。")
    lines.append("- 每个时间步输入维度：`24`（`12维 delta_ah + 12维 mask`）")
    lines.append(f"- 标签过滤范围：`{args.q_min} <= q_discharge <= {args.q_max}`")
    lines.append("")
    lines.append("## 2. 数据概览")
    lines.append(f"- 合并后 cycle 级样本数：**{merged_rows:,}**")
    lines.append(f"- 训练样本数：**{train_window_rows:,}**")
    lines.append(f"- 验证样本数：**{valid_window_rows:,}**")
    lines.append("- 电压区间：")
    for rng in range_order:
        lines.append(f"  - `{rng}`")
    lines.append("")
    lines.append("## 3. 指标结果")
    lines.append("| set_type | n_samples | MSE | RMSE | MAE | R2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in [train_metrics, valid_metrics]:
        lines.append(
            f"| {m.set_type} | {m.n_windows} | {m.mse:.8f} | {m.rmse:.6f} | {m.mae:.6f} | {m.r2:.6f} |"
        )
    lines.append("")
    lines.append("## 4. 关键图表")
    lines.append(f"- 按验证集损失选出的最佳轮次：**{best_epoch}**")
    lines.append("![loss_curve](./loss_curve.png)")
    lines.append("")
    lines.append("![valid_scatter](./valid_scatter.png)")
    lines.append("")
    lines.append("## 5. 说明")
    lines.append("- 本次训练仅使用充电电压区间 `delta_ah` 特征。")
    lines.append("- 缺失区间采用“零填充 + 显式 mask 通道”处理。")
    return "\n".join(lines)


def main() -> None:
    """Run full training pipeline."""

    args = parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 3)
        args.max_train_windows = args.max_train_windows if args.max_train_windows > 0 else 8192
        args.max_valid_windows = args.max_valid_windows if args.max_valid_windows > 0 else 4096
        args.patience = min(args.patience, 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_matplotlib_backend()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    split_df = load_split_map(args.train_split_path, args.valid_split_path)
    feature_df, range_order = build_cycle_feature_table(args.charge_path)
    value_cols, mask_cols = get_value_mask_cols(feature_df)
    label_df = load_life_labels(args.life_path, q_min=float(args.q_min), q_max=float(args.q_max))
    merged = merge_dataset(feature_df=feature_df, label_df=label_df, split_df=split_df)
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after feature/label/split join.")

    seq_map = build_sequences(merged=merged, value_cols=value_cols, mask_cols=mask_cols)
    train_seq_map, valid_seq_map = split_sequence_dict(seq_map)
    if args.sequence_mode == "fixed_window":
        train_dataset = SlidingWindowDataset(
            sequences=train_seq_map,
            window_size=int(args.window_size),
            max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
            seed=int(args.seed),
        )
        valid_dataset = SlidingWindowDataset(
            sequences=valid_seq_map,
            window_size=int(args.window_size),
            max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
            seed=int(args.seed) + 1,
        )
    else:
        train_dataset = PrefixHistoryDataset(
            sequences=train_seq_map,
            max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
            seed=int(args.seed),
        )
        valid_dataset = PrefixHistoryDataset(
            sequences=valid_seq_map,
            max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
            seed=int(args.seed) + 1,
        )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Train or valid samples are empty. Check sequence_mode and data.")

    collate_fn = collate_sequence_batch

    train_shuffle = bool(args.sequence_mode == "fixed_window")
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=train_shuffle,
        device=device,
        collate_fn=collate_fn,
    )
    valid_loader = build_dataloader(
        dataset=valid_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
        collate_fn=collate_fn,
    )

    model = LSTMRegressor(
        input_size=24,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    out_metrics = args.output_dir / "train_valid_metrics.csv"
    out_preds = args.output_dir / "valid_predictions.csv"
    out_loss = args.output_dir / "loss_curve.csv"
    out_loss_png = args.output_dir / "loss_curve.png"
    out_scatter_png = args.output_dir / "valid_scatter.png"
    out_checks = args.output_dir / "dataset_checks.csv"
    out_config = args.output_dir / "run_config.json"
    out_report = args.output_dir / "lstm_charge_delta_ah_report.md"
    best_valid_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0
    start_epoch = 1
    loss_rows: List[dict] = []
    best_ckpt_path = args.output_dir / "best_lstm_model.pt"
    latest_state_path = args.output_dir / str(args.latest_state_file)
    epoch_log_path = args.output_dir / str(args.epoch_log_file)
    status_path = args.output_dir / str(args.status_file)
    signature_payload = build_resume_signature_payload(args=args, device=device)
    args_signature = build_resume_signature(payload=signature_payload)
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
        saved_signature = str(latest_state.get("args_signature", ""))
        if saved_signature != args_signature:
            raise RuntimeError(
                "Resume signature mismatch. Disable resume with --no-resume-interrupted "
                "or use a separate output directory."
            )
        model.load_state_dict(latest_state["model_state_dict"])
        optimizer.load_state_dict(latest_state["optimizer_state_dict"])
        start_epoch = int(latest_state["epoch"]) + 1
        best_epoch = int(latest_state.get("best_epoch", 0))
        best_valid_loss = float(latest_state.get("best_valid_loss", float("inf")))
        no_improve_count = int(latest_state.get("no_improve_count", 0))
        stored_rows = latest_state.get("loss_rows", [])
        if isinstance(stored_rows, list):
            loss_rows = [dict(item) for item in stored_rows]
        print(
            f"Resumed from epoch {start_epoch - 1}: best_epoch={best_epoch}, "
            f"best_valid={best_valid_loss:.8f}",
            flush=True,
        )
    elif not bool(args.resume_interrupted):
        print("Resume disabled by --no-resume-interrupted, start from epoch 1.", flush=True)

    if start_epoch <= 1:
        pd.DataFrame(columns=epoch_log_columns).to_csv(epoch_log_path, index=False, encoding="utf-8")
        loss_rows = []
        best_valid_loss = float("inf")
        best_epoch = 0
        no_improve_count = 0

    write_training_status(
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

    if start_epoch > int(args.epochs):
        print(
            f"Latest state epoch={start_epoch - 1} already reached target epochs={args.epochs}, "
            "skip additional training.",
            flush=True,
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = eval_loss(model, valid_loader, criterion, device)
        improved = (best_valid_loss - valid_loss) > float(args.min_delta)
        if improved:
            best_valid_loss = float(valid_loss)
            best_epoch = int(epoch)
            no_improve_count = 0
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": 24,
                        "hidden_size": int(args.hidden_size),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                        "sequence_mode": str(args.sequence_mode),
                    },
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
        append_epoch_progress(
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
        atomic_torch_save(
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
        write_training_status(
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
        collate_fn=collate_fn,
    )
    y_train_true, y_train_pred, _ = predict_loader(model=model, loader=train_pred_loader, device=device)
    y_valid_true, y_valid_pred, valid_idx = predict_loader(model=model, loader=valid_loader, device=device)

    train_metrics = calc_metrics(y_true=y_train_true, y_pred=y_train_pred, set_type="train")
    valid_metrics = calc_metrics(y_true=y_valid_true, y_pred=y_valid_pred, set_type="valid")

    loss_df = pd.DataFrame(loss_rows)
    metrics_df = pd.DataFrame([asdict(train_metrics), asdict(valid_metrics)])
    valid_metas = [valid_dataset.metas[int(i)] for i in valid_idx.tolist()]
    valid_pred_df = pd.DataFrame(
        {
            "policy": [m.policy for m in valid_metas],
            "cell_code": [m.cell_code for m in valid_metas],
            "cycles": [m.cycles for m in valid_metas],
            "q_discharge": y_valid_true.astype(float),
            "pred_q_discharge": y_valid_pred.astype(float),
        }
    )
    valid_pred_df["residual"] = valid_pred_df["q_discharge"] - valid_pred_df["pred_q_discharge"]
    valid_pred_df = valid_pred_df.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(
        drop=True
    )

    checks_df = build_dataset_checks(
        merged=merged,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        value_cols=value_cols,
        mask_cols=mask_cols,
        sequence_mode=str(args.sequence_mode),
        window_size=int(args.window_size),
    )
    run_config = {
        "script": str(SCRIPT_PATH),
        "python_executable": os.path.realpath(os.sys.executable),
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resume_start_epoch": int(start_epoch),
        "args_signature": args_signature,
        "signature_payload": signature_payload,
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "range_order": list(range_order),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_text = build_report(
        args=args,
        device=device,
        range_order=range_order,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        merged_rows=int(len(merged)),
        train_window_rows=int(len(train_dataset)),
        valid_window_rows=int(len(valid_dataset)),
        best_epoch=int(best_epoch),
    )

    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8")
    valid_pred_df.to_csv(out_preds, index=False, encoding="utf-8")
    loss_df.to_csv(out_loss, index=False, encoding="utf-8")
    checks_df.to_csv(out_checks, index=False, encoding="utf-8")
    out_config.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_loss_plot(loss_df=loss_df, out_path=out_loss_png)
    save_scatter_plot(valid_pred_df=valid_pred_df, metrics=valid_metrics, out_path=out_scatter_png)
    out_report.write_text(report_text, encoding="utf-8")
    write_training_status(
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
            "valid_metrics": asdict(valid_metrics),
            "train_metrics": asdict(train_metrics),
        },
    )

    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_preds}")
    print(f"Saved: {out_loss}")
    print(f"Saved: {out_loss_png}")
    print(f"Saved: {out_scatter_png}")
    print(f"Saved: {best_ckpt_path}")
    print(f"Saved: {out_checks}")
    print(f"Saved: {out_config}")
    print(f"Saved: {out_report}")
    print(
        f"Sequence mode={args.sequence_mode} | Train/Valid samples: {len(train_dataset)}/{len(valid_dataset)} | "
        f"Valid R2={valid_metrics.r2:.6f} | Valid RMSE={valid_metrics.rmse:.6f}"
    )


if __name__ == "__main__":
    main()
