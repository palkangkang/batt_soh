import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"
N_CROSS_BINS = 60
POLICY_COLS = ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]
INPUT_FEATURE_PACKS = ["all", "charge_crossbin_discharge_capacity_stats"]
INPUT_SCALING_MODES = ["none", "standard", "standard_log1p"]
BRIDGE_FEATURE_MODES = ["dqdv_only", "dqdv_context", "both"]
LSTM_HEAD_TYPES = ["flat", "horizon_decoder"]
DQDV_TARGET_PACKS: Dict[str, List[str]] = {
    "compact2_area_height": [
        "main_peak_area",
        "main_peak_height_dqdv",
    ],
    "compact3_area_height_voltage": [
        "main_peak_area",
        "main_peak_height_dqdv",
        "main_peak_voltage_v",
    ],
    "compact_peak_shape_height_no_width": [
        "main_peak_area",
        "main_peak_skewness",
        "main_peak_voltage_v",
        "main_peak_height_dqdv",
    ],
    "compact_peak_shape_height": [
        "main_peak_area",
        "main_peak_skewness",
        "main_peak_voltage_v",
        "main_peak_width_v",
        "main_peak_height_dqdv",
    ],
}
CONTEXT_COLS = [*POLICY_COLS, "cycles", "cycle_index_norm"]
DISCHARGE_SUMMARY_STAT_COLS = [
    "discharge_cycle_total_delta_ah",
    "discharge_cycle_total_duration_s",
    "discharge_cycle_active_range_count",
    "discharge_cycle_avg_temp_mean",
    "discharge_cum_total_delta_ah",
    "discharge_cum_total_duration_s",
    "discharge_cum_active_range_count",
]


@dataclass
class Metrics:
    """Regression metrics for one target slice."""

    model_name: str
    set_type: str
    target: str
    horizon_step: int
    n_rows: int
    mse: float
    rmse: float
    mae: float
    r2: float


@dataclass
class SequenceData:
    """One policy + cell_code sequence with cycle-aligned features and labels."""

    policy: str
    cell_code: str
    set_type: str
    cycles: np.ndarray
    x: np.ndarray
    dqdv: np.ndarray
    retention: np.ndarray
    q_ref: np.ndarray
    q_discharge: np.ndarray
    context: np.ndarray


@dataclass
class PreparedData:
    """Prepared cycle/window data shared by single runs and tuning trials."""

    merged: pd.DataFrame
    feature_cols: List[str]
    target_cols: List[str]
    charge_stats: Dict[str, int]
    discharge_stats: Dict[str, int]
    train_dataset: "MultiStepWindowDataset"
    valid_dataset: "MultiStepWindowDataset"
    x_train_flat: Optional[np.ndarray]
    y_train_dqdv: np.ndarray
    x_valid_flat: Optional[np.ndarray]
    y_valid_dqdv: np.ndarray
    bridge_models: Dict[str, Pipeline]
    input_transform_info: Dict[str, object]


@dataclass
class TrialConfig:
    """One LSTM tuning trial configuration."""

    trial_id: int
    hidden_size: int
    learning_rate: float
    num_layers: int
    dropout: float


@dataclass
class WindowMeta:
    """Metadata for one history-to-future window sample."""

    policy: str
    cell_code: str
    set_type: str
    input_start_cycle: int
    input_end_cycle: int
    target_cycles: Tuple[int, ...]
    target_retentions: Tuple[float, ...]
    target_q_refs: Tuple[float, ...]
    target_q_discharge: Tuple[float, ...]
    target_contexts: Tuple[Tuple[float, ...], ...]


class MultiStepWindowDataset(Dataset):
    """Fixed-history, multi-step future dQdV dataset."""

    def __init__(
        self,
        sequences: Mapping[Tuple[str, str], SequenceData],
        history_len: int,
        horizon: int,
        max_windows: Optional[int],
        seed: int,
    ) -> None:
        self.sequences = sequences
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.refs: List[Tuple[Tuple[str, str], int]] = []
        self.metas: List[WindowMeta] = []
        self._build_refs()
        if max_windows is not None and int(max_windows) > 0 and len(self.refs) > int(max_windows):
            self._downsample(max_windows=int(max_windows), seed=int(seed))

    def __len__(self) -> int:
        """Return the number of valid windows."""

        return len(self.refs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return one fixed-length history window and flattened future dQdV labels."""

        key, end_idx = self.refs[index]
        seq = self.sequences[key]
        start_idx = end_idx - self.history_len + 1
        target_start = end_idx + 1
        target_end = target_start + self.horizon
        x_window = seq.x[start_idx : end_idx + 1]
        y_future = seq.dqdv[target_start:target_end].reshape(-1)
        return (
            torch.from_numpy(x_window),
            torch.from_numpy(y_future.astype(np.float32)),
            int(index),
            int(self.history_len),
        )

    def _build_refs(self) -> None:
        """Build valid window references and metadata."""

        for key, seq in self.sequences.items():
            n_rows = int(seq.cycles.shape[0])
            first_end = self.history_len - 1
            last_end = n_rows - self.horizon - 1
            if last_end < first_end:
                continue
            for end_idx in range(first_end, last_end + 1):
                target_start = end_idx + 1
                target_end = target_start + self.horizon
                target_contexts = tuple(
                    tuple(float(v) for v in row) for row in seq.context[target_start:target_end]
                )
                self.refs.append((key, end_idx))
                self.metas.append(
                    WindowMeta(
                        policy=seq.policy,
                        cell_code=seq.cell_code,
                        set_type=seq.set_type,
                        input_start_cycle=int(seq.cycles[end_idx - self.history_len + 1]),
                        input_end_cycle=int(seq.cycles[end_idx]),
                        target_cycles=tuple(int(v) for v in seq.cycles[target_start:target_end]),
                        target_retentions=tuple(float(v) for v in seq.retention[target_start:target_end]),
                        target_q_refs=tuple(float(v) for v in seq.q_ref[target_start:target_end]),
                        target_q_discharge=tuple(float(v) for v in seq.q_discharge[target_start:target_end]),
                        target_contexts=target_contexts,
                    )
                )

    def _downsample(self, max_windows: int, seed: int) -> None:
        """Downsample windows reproducibly while preserving sorted order."""

        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self.refs), size=int(max_windows), replace=False))
        self.refs = [self.refs[int(i)] for i in keep]
        self.metas = [self.metas[int(i)] for i in keep]


class RetentionWindowDataset(Dataset):
    """Window dataset sharing interval histories but using future retention labels."""

    def __init__(self, base_dataset: MultiStepWindowDataset) -> None:
        self.base_dataset = base_dataset
        self.sequences = base_dataset.sequences
        self.refs = base_dataset.refs
        self.metas = base_dataset.metas
        self.history_len = int(base_dataset.history_len)
        self.horizon = int(base_dataset.horizon)

    def __len__(self) -> int:
        """Return the number of windows."""

        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return one fixed-length history window and flattened future retention labels."""

        key, end_idx = self.refs[index]
        seq = self.sequences[key]
        start_idx = int(end_idx) - self.history_len + 1
        target_start = int(end_idx) + 1
        target_end = target_start + self.horizon
        x_window = seq.x[start_idx : int(end_idx) + 1]
        y_future = seq.retention[target_start:target_end].reshape(-1)
        return (
            torch.from_numpy(x_window),
            torch.from_numpy(y_future.astype(np.float32)),
            int(index),
            int(self.history_len),
        )


class MultiOutputLSTMRegressor(nn.Module):
    """LSTM regressor that emits a flattened multi-horizon target vector."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        proj_hidden = max(32, int(hidden_size) // 2)
        self.head = nn.Sequential(
            nn.Linear(int(hidden_size), proj_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(proj_hidden, int(output_size)),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Run the LSTM and project the last hidden state."""

        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1])


class HorizonAwareLSTMRegressor(nn.Module):
    """LSTM encoder with a shared horizon-step decoder head."""

    def __init__(
        self,
        input_size: int,
        horizon: int,
        target_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.target_dim = int(target_dim)
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        step_embed_dim = max(8, min(32, int(hidden_size) // 4))
        self.step_embedding = nn.Embedding(int(horizon), step_embed_dim)
        proj_hidden = max(32, int(hidden_size) // 2)
        self.head = nn.Sequential(
            nn.Linear(int(hidden_size) + step_embed_dim, proj_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(proj_hidden, int(target_dim)),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Run the encoder once and decode all forecast offsets with a shared head."""

        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (h_n, _) = self.lstm(packed)
        encoded = h_n[-1]
        batch_size = int(encoded.shape[0])
        step_ids = torch.arange(self.horizon, device=encoded.device, dtype=torch.long)
        step_emb = self.step_embedding(step_ids).unsqueeze(0).expand(batch_size, -1, -1)
        encoded_rep = encoded.unsqueeze(1).expand(-1, self.horizon, -1)
        decoded = self.head(torch.cat([encoded_rep, step_emb], dim=-1))
        return decoded.reshape(batch_size, self.horizon * self.target_dim)


def make_lstm_regressor(
    args: argparse.Namespace,
    input_size: int,
    output_size: int,
) -> nn.Module:
    """Create the configured LSTM regressor head."""

    head_type = str(getattr(args, "lstm_head", "flat"))
    if head_type == "flat":
        return MultiOutputLSTMRegressor(
            input_size=int(input_size),
            output_size=int(output_size),
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        )
    if head_type == "horizon_decoder":
        horizon = int(args.horizon)
        if int(output_size) % horizon != 0:
            raise ValueError(f"output_size={output_size} is not divisible by horizon={horizon}.")
        return HorizonAwareLSTMRegressor(
            input_size=int(input_size),
            horizon=horizon,
            target_dim=int(output_size) // horizon,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        )
    raise ValueError(f"Unsupported lstm_head: {head_type}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train interval-statistics -> dQdV -> retention prediction pipeline."
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["single", "tune", "full-refresh", "tune-and-full"],
        default="single",
        help="single: one run; tune: LSTM grid; full-refresh: train from best config; tune-and-full: both stages.",
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "interval_to_dqdv_retention_h30_k1",
    )
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--model-family", type=str, choices=["both", "rf", "lstm"], default="both")
    parser.add_argument("--skip-direct-baseline", dest="skip_direct_baseline", action="store_true")
    parser.add_argument("--no-skip-direct-baseline", dest="skip_direct_baseline", action="store_false")
    parser.set_defaults(skip_direct_baseline=False)
    parser.add_argument(
        "--target-pack",
        type=str,
        choices=sorted(DQDV_TARGET_PACKS),
        default="compact_peak_shape_height_no_width",
    )
    parser.add_argument(
        "--input-feature-pack",
        type=str,
        choices=sorted(INPUT_FEATURE_PACKS),
        default="all",
        help="Input feature subset for interval-to-dQdV models. Use all for the legacy 251-dim feature set.",
    )
    parser.add_argument(
        "--input-scaling",
        type=str,
        choices=sorted(INPUT_SCALING_MODES),
        default="none",
        help="Input feature scaling fitted on training cycles only.",
    )
    parser.add_argument(
        "--bridge-feature-mode",
        type=str,
        choices=sorted(BRIDGE_FEATURE_MODES),
        default="dqdv_only",
        help="Bridge inputs: dQdV only for deployable metrics, dQdV+context for legacy upper-reference, or both.",
    )
    parser.add_argument(
        "--lstm-head",
        type=str,
        choices=sorted(LSTM_HEAD_TYPES),
        default="flat",
        help="LSTM output head. horizon_decoder shares a head across forecast offsets.",
    )
    parser.add_argument("--direct-retention-baseline", dest="direct_retention_baseline", action="store_true")
    parser.add_argument("--no-direct-retention-baseline", dest="direct_retention_baseline", action="store_false")
    parser.set_defaults(direct_retention_baseline=False)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--rf-n-estimators", type=int, default=180)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260429)
    parser.add_argument("--smoke-train-cells", type=int, default=12)
    parser.add_argument("--smoke-valid-cells", type=int, default=6)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--hidden-sizes", type=str, default="64,128")
    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4")
    parser.add_argument("--num-layers-list", type=str, default="1,2")
    parser.add_argument("--dropout-list", type=str, default="0.1,0.2")
    parser.add_argument("--tune-max-train-windows", type=int, default=20000)
    parser.add_argument("--tune-max-valid-windows", type=int, default=8000)
    parser.add_argument("--partial-results-file", type=str, default="grid_search_results.partial.csv")
    parser.add_argument("--grid-results-file", type=str, default="grid_search_results.csv")
    parser.add_argument("--best-config-file", type=str, default="best_grid_config.json")
    parser.add_argument("--runtime-status-file", type=str, default="grid_tuning_runtime_status.json")
    parser.add_argument("--best-config-path", type=Path, default=None)
    parser.add_argument("--full-refresh-output-dir", type=Path, default=None)
    parser.add_argument("--full-refresh-epochs", type=int, default=80)
    parser.add_argument("--full-refresh-patience", type=int, default=20)
    parser.add_argument("--full-refresh-batch-size", type=int, default=256)
    parser.add_argument("--resume-from-partial", dest="resume_from_partial", action="store_true")
    parser.add_argument("--no-resume-from-partial", dest="resume_from_partial", action="store_false")
    parser.set_defaults(resume_from_partial=True)
    parser.add_argument("--resume-existing", dest="resume_existing", action="store_true")
    parser.add_argument("--no-resume-existing", dest="resume_existing", action="store_false")
    parser.set_defaults(resume_existing=True)
    parser.add_argument("--resume-interrupted", dest="resume_interrupted", action="store_true")
    parser.add_argument("--no-resume-interrupted", dest="resume_interrupted", action="store_false")
    parser.set_defaults(resume_interrupted=True)
    parser.add_argument("--save-partial-results", dest="save_partial_results", action="store_true")
    parser.add_argument("--no-save-partial-results", dest="save_partial_results", action="store_false")
    parser.set_defaults(save_partial_results=True)
    parser.add_argument("--save-trial-latest-state", dest="save_trial_latest_state", action="store_true")
    parser.add_argument("--no-save-trial-latest-state", dest="save_trial_latest_state", action="store_false")
    parser.set_defaults(save_trial_latest_state=True)
    parser.add_argument("--save-trial-epoch-log", dest="save_trial_epoch_log", action="store_true")
    parser.add_argument("--no-save-trial-epoch-log", dest="save_trial_epoch_log", action="store_false")
    parser.set_defaults(save_trial_epoch_log=True)
    parser.add_argument("--run-full-refresh", dest="run_full_refresh", action="store_true")
    parser.add_argument("--skip-full-refresh", dest="run_full_refresh", action="store_false")
    parser.set_defaults(run_full_refresh=False)
    parser.add_argument("--full-refresh-resume-interrupted", dest="full_refresh_resume_interrupted", action="store_true")
    parser.add_argument("--no-full-refresh-resume-interrupted", dest="full_refresh_resume_interrupted", action="store_false")
    parser.set_defaults(full_refresh_resume_interrupted=True)
    return parser.parse_args()


def ensure_matplotlib_config() -> List[str]:
    """Configure matplotlib for headless output and return selected fonts."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    candidates = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = [font for font in candidates if font in installed] or ["DejaVu Sans"]
    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    return selected


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch random seeds."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested torch device."""

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_checkpoint(path: Path) -> Mapping[str, object]:
    """Load a checkpoint produced by this script across PyTorch versions."""

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint at {path} is not a mapping.")
    return loaded


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    """Deduplicate strings while keeping their first-seen order."""

    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_int_list(raw: str) -> List[int]:
    """Parse a comma-separated integer list."""

    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty integer list: {raw!r}")
    return [int(item) for item in values]


def parse_float_list(raw: str) -> List[float]:
    """Parse a comma-separated float list."""

    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty float list: {raw!r}")
    return [float(item) for item in values]


def copy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Make a shallow argparse namespace copy."""

    return argparse.Namespace(**vars(args).copy())


def prepare_trial_grid(args: argparse.Namespace) -> List[TrialConfig]:
    """Build the LSTM hyper-parameter trial grid."""

    trials: List[TrialConfig] = []
    trial_id = 0
    for hidden_size in parse_int_list(args.hidden_sizes):
        for learning_rate in parse_float_list(args.learning_rates):
            for num_layers in parse_int_list(args.num_layers_list):
                for dropout in parse_float_list(args.dropout_list):
                    trial_id += 1
                    trials.append(
                        TrialConfig(
                            trial_id=trial_id,
                            hidden_size=int(hidden_size),
                            learning_rate=float(learning_rate),
                            num_layers=int(num_layers),
                            dropout=float(dropout),
                        )
                    )
    if bool(args.smoke_test):
        return trials[: min(2, len(trials))]
    return trials


def key_token(policy: object, cell_code: object) -> str:
    """Build a stable composite key token."""

    return f"{str(policy)}\x1f{str(cell_code)}"


def build_allowed_key_tokens(split_map: pd.DataFrame, smoke_train: int, smoke_valid: int) -> Optional[set[str]]:
    """Build a smoke-test subset of train and valid cell keys."""

    if int(smoke_train) <= 0 and int(smoke_valid) <= 0:
        return None
    train_part = split_map.loc[split_map["set_type"] == "train"].head(max(0, int(smoke_train)))
    valid_part = split_map.loc[split_map["set_type"] == "valid"].head(max(0, int(smoke_valid)))
    keys = pd.concat([train_part, valid_part], ignore_index=True)
    return {key_token(row.policy, row.cell_code) for row in keys.itertuples(index=False)}


def load_split(train_path: Path, valid_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/valid policy-cell split tables."""

    train = pd.read_csv(train_path, encoding=ENCODING)
    valid = pd.read_csv(valid_path, encoding=ENCODING)
    cols = ["policy", "cell_code", *POLICY_COLS]
    train = train[cols].copy()
    valid = valid[cols].copy()
    for frame in [train, valid]:
        frame["policy"] = frame["policy"].astype(str)
        frame["cell_code"] = frame["cell_code"].astype(str)
        for col in POLICY_COLS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    split_map = pd.concat([train, valid], ignore_index=True).drop_duplicates(
        ["policy", "cell_code"],
        keep="first",
    )
    train_keys = set(train["policy"] + "\x1f" + train["cell_code"])
    valid_keys = set(valid["policy"] + "\x1f" + valid["cell_code"])
    overlap = len(train_keys.intersection(valid_keys))
    if overlap > 0:
        raise RuntimeError(f"Split leakage detected: {overlap} policy-cell keys overlap.")
    return train, valid, split_map


def filter_allowed_keys(frame: pd.DataFrame, allowed_key_tokens: Optional[set[str]]) -> pd.DataFrame:
    """Filter a frame to allowed policy-cell tokens when a smoke subset is requested."""

    if allowed_key_tokens is None:
        return frame
    tokens = frame["policy"].astype(str) + "\x1f" + frame["cell_code"].astype(str)
    return frame.loc[tokens.isin(allowed_key_tokens)].copy()


def load_charge_feature_table(
    timeseries_path: Path,
    allowed_key_tokens: Optional[set[str]],
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Build cycle-level charge cumulative and increment features from 60 cross-bin rows."""

    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "cross_bin",
        "cycle_charge_time_h",
        "cumulative_charge_time_h",
        "is_abnormal_cell",
    ]
    idx_cols = ["policy", "cell_code", "cycles"]
    cum_cols = [f"charge_cross_bin_cum_{i:02d}_h" for i in range(1, N_CROSS_BINS + 1)]
    inc_cols = [f"charge_cross_bin_inc_{i:02d}_h" for i in range(1, N_CROSS_BINS + 1)]
    part_features: List[pd.DataFrame] = []
    part_counts: List[pd.DataFrame] = []
    rows_after_filter = 0

    reader = pd.read_csv(timeseries_path, usecols=usecols, encoding=ENCODING, chunksize=80000)
    for chunk in reader:
        chunk["policy"] = chunk["policy"].astype(str)
        chunk["cell_code"] = chunk["cell_code"].astype(str)
        chunk = filter_allowed_keys(chunk, allowed_key_tokens)
        if chunk.empty:
            continue
        for col in ["cycles", "cross_bin", "cycle_charge_time_h", "cumulative_charge_time_h", "is_abnormal_cell"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=usecols).copy()
        chunk["cycles"] = chunk["cycles"].astype(int)
        chunk["cross_bin"] = chunk["cross_bin"].astype(int)
        chunk["is_abnormal_cell"] = chunk["is_abnormal_cell"].astype(int)
        chunk = chunk[(chunk["cross_bin"] >= 1) & (chunk["cross_bin"] <= N_CROSS_BINS)].copy()
        if chunk.empty:
            continue
        rows_after_filter += int(len(chunk))
        grouped = (
            chunk.groupby([*idx_cols, "cross_bin"], as_index=False)
            .agg(
                cycle_charge_time_h=("cycle_charge_time_h", "sum"),
                cumulative_charge_time_h=("cumulative_charge_time_h", "max"),
                is_abnormal_cell=("is_abnormal_cell", "max"),
            )
        )
        part_counts.append(grouped[[*idx_cols, "cross_bin"]].drop_duplicates())
        abnormal = grouped.groupby(idx_cols, as_index=False)["is_abnormal_cell"].max()
        cum = (
            grouped.pivot_table(
                index=idx_cols,
                columns="cross_bin",
                values="cumulative_charge_time_h",
                aggfunc="max",
                fill_value=0.0,
            )
            .reindex(columns=list(range(1, N_CROSS_BINS + 1)), fill_value=0.0)
            .reset_index()
            .rename(columns={i: cum_cols[i - 1] for i in range(1, N_CROSS_BINS + 1)})
        )
        inc = (
            grouped.pivot_table(
                index=idx_cols,
                columns="cross_bin",
                values="cycle_charge_time_h",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(columns=list(range(1, N_CROSS_BINS + 1)), fill_value=0.0)
            .reset_index()
            .rename(columns={i: inc_cols[i - 1] for i in range(1, N_CROSS_BINS + 1)})
        )
        part_features.append(cum.merge(inc, on=idx_cols, how="inner").merge(abnormal, on=idx_cols, how="left"))

    if not part_features:
        raise RuntimeError("No valid charge feature rows were loaded.")

    feat = pd.concat(part_features, ignore_index=True)
    agg_map = {**{col: "max" for col in cum_cols}, **{col: "sum" for col in inc_cols}, "is_abnormal_cell": "max"}
    feat = feat.groupby(idx_cols, as_index=False).agg(agg_map)
    feat = feat.rename(columns={"is_abnormal_cell": "charge_is_abnormal_cell"})
    cnt = pd.concat(part_counts, ignore_index=True).drop_duplicates()
    cross_counts = cnt.groupby(idx_cols, as_index=False)["cross_bin"].nunique()
    incomplete = int((cross_counts["cross_bin"] < N_CROSS_BINS).sum())

    stat_cols = [
        "charge_cycle_total_h",
        "charge_cycle_active_bin_count",
        "charge_cycle_max_bin_h",
        "charge_cycle_mean_nonzero_bin_h",
        "charge_cum_total_h",
        "charge_cum_active_bin_count",
    ]
    feat[stat_cols[0]] = feat[inc_cols].sum(axis=1)
    feat[stat_cols[1]] = (feat[inc_cols] > 0.0).sum(axis=1)
    feat[stat_cols[2]] = feat[inc_cols].max(axis=1)
    feat[stat_cols[3]] = feat[inc_cols].replace(0.0, np.nan).mean(axis=1).fillna(0.0)
    feat[stat_cols[4]] = feat[cum_cols].sum(axis=1)
    feat[stat_cols[5]] = (feat[cum_cols] > 0.0).sum(axis=1)

    feature_cols = [*cum_cols, *inc_cols, *stat_cols, "charge_is_abnormal_cell"]
    stats = {
        "charge_rows_after_filter": int(rows_after_filter),
        "charge_cycle_rows": int(len(feat)),
        "charge_incomplete_cross_bin_cycle_rows": incomplete,
        "charge_cross_bin_feature_dim": int(N_CROSS_BINS),
    }
    return feat[["policy", "cell_code", "cycles", *feature_cols]].copy(), stats, feature_cols


def parse_range_start(range_label: str) -> float:
    """Parse the first numeric voltage value from a range label."""

    values = re.findall(r"-?\d+(?:\.\d+)?", str(range_label))
    if not values:
        return float("nan")
    return float(values[0])


def sanitize_range_label(range_label: str) -> str:
    """Convert a voltage range label into a safe feature suffix."""

    values = re.findall(r"-?\d+(?:\.\d+)?", str(range_label))
    if len(values) >= 2:
        left = values[0].replace(".", "p")
        right = values[1].replace(".", "p")
        return f"{left}_to_{right}"
    return re.sub(r"[^0-9A-Za-z]+", "_", str(range_label)).strip("_").lower()


def load_discharge_feature_table(
    discharge_path: Path,
    allowed_key_tokens: Optional[set[str]],
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Build cycle-level discharge increment, cumulative, and mask features."""

    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "range",
        "delta_ah",
        "charge_duration_s",
        "avg_temper",
        "range_count",
    ]
    df = pd.read_csv(discharge_path, encoding=ENCODING, usecols=usecols)
    df["policy"] = df["policy"].astype(str)
    df["cell_code"] = df["cell_code"].astype(str)
    for col in ["cycles", "delta_ah", "charge_duration_s", "avg_temper", "range_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["policy", "cell_code", "cycles", "range"]).copy()
    df["cycles"] = df["cycles"].astype(int)
    df["range"] = df["range"].astype(str)
    if df.empty:
        raise RuntimeError("No valid discharge interval rows were loaded.")

    range_order = sorted(df["range"].dropna().unique().tolist(), key=parse_range_start, reverse=True)
    suffix_map = {rng: sanitize_range_label(str(rng)) for rng in range_order}
    df = filter_allowed_keys(df, allowed_key_tokens)
    if df.empty:
        raise RuntimeError("No valid discharge interval rows were loaded after key filtering.")
    grouped = (
        df.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(
            discharge_delta_ah=("delta_ah", "sum"),
            discharge_duration_s=("charge_duration_s", "sum"),
            discharge_avg_temper=("avg_temper", "mean"),
            discharge_range_count=("range_count", "sum"),
        )
    )
    idx_cols = ["policy", "cell_code", "cycles"]
    wide = grouped[idx_cols].drop_duplicates().copy()
    feature_cols: List[str] = []

    pivot_specs = [
        ("discharge_delta_ah", "discharge_inc_delta_ah", "sum"),
        ("discharge_duration_s", "discharge_inc_duration_s", "sum"),
        ("discharge_avg_temper", "discharge_avg_temp", "mean"),
        ("discharge_range_count", "discharge_range_count", "sum"),
    ]
    for value_col, prefix, aggfunc in pivot_specs:
        pivot = (
            grouped.pivot_table(
                index=idx_cols,
                columns="range",
                values=value_col,
                aggfunc=aggfunc,
                fill_value=np.nan,
            )
            .reindex(columns=range_order)
            .reset_index()
        )
        rename = {rng: f"{prefix}_{suffix_map[str(rng)]}" for rng in range_order}
        pivot = pivot.rename(columns=rename)
        cols = [rename[rng] for rng in range_order]
        if prefix == "discharge_avg_temp":
            mask_cols = [f"discharge_mask_{suffix_map[str(rng)]}" for rng in range_order]
            for col, mask_col in zip(cols, mask_cols):
                pivot[mask_col] = (~pivot[col].isna()).astype(np.float32)
            pivot[cols] = pivot[cols].fillna(0.0)
            feature_cols.extend([*cols, *mask_cols])
            wide = wide.merge(pivot[[*idx_cols, *cols, *mask_cols]], on=idx_cols, how="left")
        else:
            pivot[cols] = pivot[cols].fillna(0.0)
            feature_cols.extend(cols)
            wide = wide.merge(pivot[[*idx_cols, *cols]], on=idx_cols, how="left")

    inc_delta_cols = [f"discharge_inc_delta_ah_{suffix_map[str(rng)]}" for rng in range_order]
    inc_duration_cols = [f"discharge_inc_duration_s_{suffix_map[str(rng)]}" for rng in range_order]
    cum_delta_cols = [col.replace("discharge_inc_", "discharge_cum_") for col in inc_delta_cols]
    cum_duration_cols = [col.replace("discharge_inc_", "discharge_cum_") for col in inc_duration_cols]
    wide = wide.sort_values(idx_cols, kind="mergesort")
    wide[cum_delta_cols] = wide.groupby(["policy", "cell_code"], sort=False)[inc_delta_cols].cumsum()
    wide[cum_duration_cols] = wide.groupby(["policy", "cell_code"], sort=False)[inc_duration_cols].cumsum()
    feature_cols.extend([*cum_delta_cols, *cum_duration_cols])

    stat_cols = [
        "discharge_cycle_total_delta_ah",
        "discharge_cycle_total_duration_s",
        "discharge_cycle_active_range_count",
        "discharge_cycle_avg_temp_mean",
        "discharge_cum_total_delta_ah",
        "discharge_cum_total_duration_s",
        "discharge_cum_active_range_count",
    ]
    temp_cols = [f"discharge_avg_temp_{suffix_map[str(rng)]}" for rng in range_order]
    mask_cols = [f"discharge_mask_{suffix_map[str(rng)]}" for rng in range_order]
    wide[stat_cols[0]] = wide[inc_delta_cols].sum(axis=1)
    wide[stat_cols[1]] = wide[inc_duration_cols].sum(axis=1)
    wide[stat_cols[2]] = (wide[inc_delta_cols] > 0.0).sum(axis=1)
    temp_sum = (wide[temp_cols] * wide[mask_cols].to_numpy(dtype=float)).sum(axis=1)
    temp_count = wide[mask_cols].sum(axis=1).replace(0, np.nan)
    wide[stat_cols[3]] = (temp_sum / temp_count).fillna(0.0)
    wide[stat_cols[4]] = wide[cum_delta_cols].sum(axis=1)
    wide[stat_cols[5]] = wide[cum_duration_cols].sum(axis=1)
    wide[stat_cols[6]] = (wide[cum_delta_cols] > 0.0).sum(axis=1)
    feature_cols.extend(stat_cols)
    feature_cols = dedupe_keep_order(feature_cols)
    wide[feature_cols] = wide[feature_cols].fillna(0.0)
    stats = {
        "discharge_interval_rows_after_filter": int(len(df)),
        "discharge_cycle_rows": int(len(wide)),
        "discharge_range_count": int(len(range_order)),
    }
    return wide[[*idx_cols, *feature_cols]].copy(), stats, feature_cols


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean encodings."""

    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["1", "true", "t", "yes", "y"])


def load_dqdv_table(path: Path, target_cols: Sequence[str]) -> pd.DataFrame:
    """Load valid cycle-level dQdV target features."""

    usecols = ["policy", "cell_code", "cycles", "is_valid_curve", *target_cols]
    df = pd.read_csv(path, encoding=ENCODING, usecols=usecols)
    df["policy"] = df["policy"].astype(str)
    df["cell_code"] = df["cell_code"].astype(str)
    df["cycles"] = pd.to_numeric(df["cycles"], errors="coerce")
    df["is_valid_curve"] = normalize_bool_series(df["is_valid_curve"])
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["policy", "cell_code", "cycles", *target_cols]).copy()
    df["cycles"] = df["cycles"].astype(int)
    df = df.loc[df["is_valid_curve"]].copy()
    df = df.drop(columns=["is_valid_curve"])
    return df.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").drop_duplicates(
        ["policy", "cell_code", "cycles"],
        keep="last",
    )


def load_retention_labels(
    life_path: Path,
    q_min: float,
    q_max: float,
    q_ref_cycles: int,
    retention_min: float,
    retention_max: float,
) -> pd.DataFrame:
    """Load q_discharge labels and derive retention by early-cycle q_ref median."""

    life = pd.read_csv(life_path, encoding=ENCODING, usecols=["policy", "cell_code", "cycles", "q_discharge"])
    life["policy"] = life["policy"].astype(str)
    life["cell_code"] = life["cell_code"].astype(str)
    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    life["cycles"] = life["cycles"].astype(int)
    life = life.loc[life["q_discharge"] > 0.0].copy()
    life = life.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")
    abs_filtered = life.loc[(life["q_discharge"] >= float(q_min)) & (life["q_discharge"] <= float(q_max))].copy()
    if abs_filtered.empty:
        raise RuntimeError("No rows remain after absolute q_discharge filtering.")
    early = abs_filtered.groupby(["policy", "cell_code"], sort=False).head(int(q_ref_cycles))
    q_ref = (
        early.groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .median()
        .rename(columns={"q_discharge": "q_ref"})
    )
    q_ref = q_ref.loc[q_ref["q_ref"] > 0.0].copy()
    labeled = abs_filtered.merge(q_ref, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    labeled["retention"] = labeled["q_discharge"] / labeled["q_ref"]
    labeled = labeled.loc[
        (labeled["retention"] >= float(retention_min)) & (labeled["retention"] <= float(retention_max))
    ].copy()
    if labeled.empty:
        raise RuntimeError("No rows remain after retention filtering.")
    return labeled[["policy", "cell_code", "cycles", "q_discharge", "q_ref", "retention"]].copy()


def add_cycle_index_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized per-cell cycle index."""

    out = df.copy()
    c_min = out.groupby(["policy", "cell_code"])["cycles"].transform("min")
    c_max = out.groupby(["policy", "cell_code"])["cycles"].transform("max")
    denom = (c_max - c_min).replace(0, 1)
    out["cycle_index_norm"] = ((out["cycles"] - c_min) / denom).astype(np.float32)
    return out


def select_input_feature_columns(feature_cols: Sequence[str], input_feature_pack: str) -> List[str]:
    """Select the interval input feature columns used by the sequence model."""

    pack = str(input_feature_pack)
    if pack == "all":
        return list(feature_cols)
    if pack != "charge_crossbin_discharge_capacity_stats":
        raise ValueError(f"Unsupported input_feature_pack: {pack}")
    selected = [
        col
        for col in feature_cols
        if col.startswith("charge_cross_bin_cum_")
        or col.startswith("charge_cross_bin_inc_")
        or col.startswith("discharge_inc_delta_ah_")
        or col.startswith("discharge_cum_delta_ah_")
        or col in DISCHARGE_SUMMARY_STAT_COLS
    ]
    if not selected:
        raise RuntimeError(f"Input feature pack {pack} selected zero columns.")
    return selected


def select_log1p_feature_columns(feature_cols: Sequence[str]) -> List[str]:
    """Select positive cumulative exposure features for optional log1p compression."""

    prefixes = (
        "charge_cross_bin_cum_",
        "discharge_cum_delta_ah_",
        "discharge_cum_duration_s_",
    )
    exact = {
        "discharge_cum_total_delta_ah",
        "discharge_cum_total_duration_s",
        "discharge_cum_active_range_count",
    }
    return [col for col in feature_cols if col.startswith(prefixes) or col in exact]


def fit_apply_input_transform(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    input_scaling: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply train-only input transforms and return transform metadata."""

    mode = str(input_scaling)
    if mode not in INPUT_SCALING_MODES:
        raise ValueError(f"Unsupported input_scaling: {mode}")
    out = merged.copy()
    info: Dict[str, object] = {
        "input_scaling": mode,
        "log1p_feature_columns": [],
        "scaled_feature_count": int(len(feature_cols)),
    }
    if not feature_cols or mode == "none":
        return out, info
    log_cols = select_log1p_feature_columns(feature_cols) if mode == "standard_log1p" else []
    for col in log_cols:
        out[col] = np.log1p(np.clip(pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, None))
    train_mask = out["set_type"].astype(str) == "train"
    if not bool(train_mask.any()):
        raise RuntimeError("Cannot fit input scaler because no training rows are present.")
    train_values = out.loc[train_mask, list(feature_cols)].to_numpy(dtype=np.float32)
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    values = out[list(feature_cols)].to_numpy(dtype=np.float32)
    out.loc[:, list(feature_cols)] = ((values - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
    info.update(
        {
            "log1p_feature_columns": list(log_cols),
            "train_mean_summary": {
                "min": float(np.min(mean)) if mean.size else 0.0,
                "max": float(np.max(mean)) if mean.size else 0.0,
            },
            "train_std_summary": {
                "min": float(np.min(std)) if std.size else 0.0,
                "max": float(np.max(std)) if std.size else 0.0,
            },
        }
    )
    return out, info


def build_merged_cycle_table(
    split_map: pd.DataFrame,
    charge_df: pd.DataFrame,
    charge_cols: Sequence[str],
    discharge_df: pd.DataFrame,
    discharge_cols: Sequence[str],
    dqdv_df: pd.DataFrame,
    label_df: pd.DataFrame,
    target_cols: Sequence[str],
    input_feature_pack: str,
    input_scaling: str,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    """Merge cycle-level interval features, dQdV targets, retention labels, and split metadata."""

    interval_df = charge_df.merge(discharge_df, on=["policy", "cell_code", "cycles"], how="outer")
    interval_cols = [*charge_cols, *discharge_cols]
    for col in interval_cols:
        interval_df[col] = pd.to_numeric(interval_df[col], errors="coerce").fillna(0.0)
    merged = (
        label_df.merge(dqdv_df, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(split_map[["policy", "cell_code", *POLICY_COLS, "set_type"]], on=["policy", "cell_code"], how="inner")
        .merge(interval_df, on=["policy", "cell_code", "cycles"], how="left")
    )
    for col in interval_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    for col in [*POLICY_COLS, "q_discharge", "q_ref", "retention", *target_cols]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=[*POLICY_COLS, "q_discharge", "q_ref", "retention", *target_cols]).copy()
    merged = add_cycle_index_norm(merged)
    all_feature_cols = dedupe_keep_order([*interval_cols, *POLICY_COLS, "cycles", "cycle_index_norm"])
    feature_cols = select_input_feature_columns(all_feature_cols, input_feature_pack)
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(np.float32)
    for col in target_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype(np.float32)
    merged["retention"] = merged["retention"].astype(np.float32)
    merged["q_ref"] = merged["q_ref"].astype(np.float32)
    merged["q_discharge"] = merged["q_discharge"].astype(np.float32)
    merged, transform_info = fit_apply_input_transform(merged, feature_cols, input_scaling)
    return (
        merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True),
        feature_cols,
        transform_info,
    )


def build_sequences(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
) -> Dict[Tuple[str, str], SequenceData]:
    """Build per-cell sequence objects for window generation."""

    seq_map: Dict[Tuple[str, str], SequenceData] = {}
    for (policy, cell_code), part in merged.groupby(["policy", "cell_code"], sort=False):
        work = part.sort_values("cycles", kind="mergesort").copy()
        set_types = work["set_type"].dropna().unique().tolist()
        if len(set_types) != 1:
            raise RuntimeError(f"Split leakage detected in sequence {(policy, cell_code)}.")
        seq_map[(str(policy), str(cell_code))] = SequenceData(
            policy=str(policy),
            cell_code=str(cell_code),
            set_type=str(set_types[0]),
            cycles=work["cycles"].to_numpy(dtype=np.int32),
            x=work[list(feature_cols)].to_numpy(dtype=np.float32),
            dqdv=work[list(target_cols)].to_numpy(dtype=np.float32),
            retention=work["retention"].to_numpy(dtype=np.float32),
            q_ref=work["q_ref"].to_numpy(dtype=np.float32),
            q_discharge=work["q_discharge"].to_numpy(dtype=np.float32),
            context=work[CONTEXT_COLS].to_numpy(dtype=np.float32),
        )
    return seq_map


def split_sequence_map(
    seq_map: Mapping[Tuple[str, str], SequenceData],
) -> Tuple[Dict[Tuple[str, str], SequenceData], Dict[Tuple[str, str], SequenceData]]:
    """Split sequence map into train and valid dictionaries."""

    train_map: Dict[Tuple[str, str], SequenceData] = {}
    valid_map: Dict[Tuple[str, str], SequenceData] = {}
    for key, seq in seq_map.items():
        if seq.set_type == "train":
            train_map[key] = seq
        elif seq.set_type == "valid":
            valid_map[key] = seq
        else:
            raise RuntimeError(f"Unknown set_type={seq.set_type!r}.")
    return train_map, valid_map


def collate_window_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate fixed-length windows in descending length order."""

    sorted_batch = sorted(batch, key=lambda item: item[3], reverse=True)
    xs = [item[0] for item in sorted_batch]
    ys = torch.stack([item[1] for item in sorted_batch], dim=0)
    idxs = torch.tensor([int(item[2]) for item in sorted_batch], dtype=torch.long)
    lengths = torch.tensor([int(item[3]) for item in sorted_batch], dtype=torch.long)
    return pad_sequence(xs, batch_first=True, padding_value=0.0), ys, idxs, lengths


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    """Create a DataLoader for window batches."""

    return DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_window_batch,
    )


def dataset_to_numpy(dataset: MultiStepWindowDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a window dataset to flattened NumPy arrays for tree models."""

    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    for idx in range(len(dataset)):
        x, y, _, _ = dataset[idx]
        x_rows.append(x.numpy().reshape(-1).astype(np.float32))
        y_rows.append(y.numpy().astype(np.float32))
    return np.vstack(x_rows), np.vstack(y_rows)


def dqdv_targets_from_dataset(dataset: MultiStepWindowDataset) -> np.ndarray:
    """Build flattened multi-horizon dQdV targets without materializing input windows."""

    rows: List[np.ndarray] = []
    for key, end_idx in dataset.refs:
        seq = dataset.sequences[key]
        target_start = int(end_idx) + 1
        target_end = target_start + int(dataset.horizon)
        rows.append(seq.dqdv[target_start:target_end].reshape(-1).astype(np.float32))
    return np.vstack(rows)


def retention_targets_from_dataset(dataset: MultiStepWindowDataset) -> np.ndarray:
    """Build flattened multi-horizon retention targets from dataset metadata."""

    rows = [np.array(meta.target_retentions, dtype=np.float32) for meta in dataset.metas]
    return np.vstack(rows)


def sklearn_target_for_fit(y: np.ndarray) -> np.ndarray:
    """Use 1D targets for single-output sklearn models to avoid shape warnings."""

    if y.ndim == 2 and int(y.shape[1]) == 1:
        return y.reshape(-1)
    return y


def ensure_2d_prediction(pred: np.ndarray, horizon: int) -> np.ndarray:
    """Convert sklearn single-output predictions back to a 2D horizon matrix."""

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 1:
        return pred.reshape(-1, int(horizon))
    return pred.astype(np.float32)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, set_type: str, target: str, horizon_step: int) -> Metrics:
    """Compute robust regression metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    if y_true.size < 2 or float(np.nanstd(y_true)) <= 0.0:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
    return Metrics(
        model_name=str(model_name),
        set_type=str(set_type),
        target=str(target),
        horizon_step=int(horizon_step),
        n_rows=int(y_true.size),
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
    )


def build_dqdv_metrics(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    model_name: str,
    set_type: str,
    horizon: int,
    target_cols: Sequence[str],
) -> List[Metrics]:
    """Build per-horizon, per-dQdV-target metrics."""

    target_count = len(target_cols)
    y_true = y_true_flat.reshape(-1, int(horizon), target_count)
    y_pred = y_pred_flat.reshape(-1, int(horizon), target_count)
    rows: List[Metrics] = []
    for h_idx in range(int(horizon)):
        for t_idx, target in enumerate(target_cols):
            rows.append(
                calc_metrics(
                    y_true=y_true[:, h_idx, t_idx],
                    y_pred=y_pred[:, h_idx, t_idx],
                    model_name=model_name,
                    set_type=set_type,
                    target=str(target),
                    horizon_step=h_idx + 1,
                )
            )
    return rows


def make_rf_model(args: argparse.Namespace, seed: int) -> Pipeline:
    """Create a RandomForest pipeline with median imputation."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=int(args.rf_n_estimators),
                    max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
                    min_samples_leaf=int(args.rf_min_samples_leaf),
                    max_features=0.45,
                    random_state=int(seed),
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_target_scaler(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean/std target scaler for neural network training."""

    mean = y_train.mean(axis=0).astype(np.float32)
    std = y_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    """Train one LSTM epoch on standardized dQdV targets."""

    model.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch, _, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        scaled_y = (y_batch - target_mean) / target_std
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch, lengths)
        loss = criterion(pred, scaled_y)
        loss.backward()
        optimizer.step()
        batch_size = int(y_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


@torch.no_grad()
def eval_lstm_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    """Evaluate standardized dQdV loss for one loader."""

    model.eval()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch, _, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        scaled_y = (y_batch - target_mean) / target_std
        pred = model(x_batch, lengths)
        loss = criterion(pred, scaled_y)
        batch_size = int(y_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


@torch.no_grad()
def predict_lstm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean_np: np.ndarray,
    target_std_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict natural-scale dQdV targets from a trained LSTM."""

    model.eval()
    ys_true: List[np.ndarray] = []
    ys_pred: List[np.ndarray] = []
    idxs: List[np.ndarray] = []
    target_mean = target_mean_np.reshape(1, -1)
    target_std = target_std_np.reshape(1, -1)
    for x_batch, y_batch, idx_batch, lengths in loader:
        x_batch = x_batch.to(device)
        pred_scaled = model(x_batch, lengths).detach().cpu().numpy().astype(np.float32)
        pred = pred_scaled * target_std + target_mean
        ys_pred.append(pred.astype(np.float32))
        ys_true.append(y_batch.numpy().astype(np.float32))
        idxs.append(idx_batch.numpy().astype(np.int64))
    return np.concatenate(ys_true, axis=0), np.concatenate(ys_pred, axis=0), np.concatenate(idxs, axis=0)


def train_lstm_model(
    args: argparse.Namespace,
    train_dataset: MultiStepWindowDataset,
    valid_dataset: MultiStepWindowDataset,
    y_train_flat: np.ndarray,
    device: torch.device,
    output_size: int,
    feature_dim: int,
    checkpoint_dir: Optional[Path] = None,
    resume_interrupted: bool = False,
    save_latest_state: bool = False,
    save_epoch_log: bool = False,
    status_path: Optional[Path] = None,
    run_signature: str = "",
) -> Tuple[nn.Module, pd.DataFrame, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the LSTM dQdV model and return natural-scale predictions."""

    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, True, device)
    valid_loader = build_dataloader(valid_dataset, args.batch_size, args.num_workers, False, device)
    target_mean_np, target_std_np = fit_target_scaler(y_train_flat)
    target_mean = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
    target_std = torch.tensor(target_std_np, dtype=torch.float32, device=device)
    model = make_lstm_regressor(
        args=args,
        input_size=int(feature_dim),
        output_size=int(output_size),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.MSELoss()
    best_ckpt_path: Optional[Path] = None
    latest_state_path: Optional[Path] = None
    epoch_log_path: Optional[Path] = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_dir / "best.pt"
        latest_state_path = checkpoint_dir / "latest.pt"
        epoch_log_path = checkpoint_dir / "epoch_log.csv"

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_valid = float("inf")
    no_improve = 0
    loss_rows: List[Dict[str, float]] = []
    start_epoch = 1
    if bool(resume_interrupted) and latest_state_path is not None and latest_state_path.exists():
        latest_state = load_torch_checkpoint(latest_state_path)
        saved_signature = str(latest_state.get("run_signature", ""))
        if saved_signature and run_signature and saved_signature != run_signature:
            raise RuntimeError(
                f"LSTM latest checkpoint signature mismatch at {latest_state_path}. "
                "Use a fresh output directory or disable resume."
            )
        model.load_state_dict(latest_state["model_state"])
        optimizer.load_state_dict(latest_state["optimizer_state"])
        start_epoch = int(latest_state.get("epoch", 0)) + 1
        best_epoch = int(latest_state.get("best_epoch", 0))
        best_valid = float(latest_state.get("best_valid_loss", float("inf")))
        no_improve = int(latest_state.get("no_improve", 0))
        if "best_model_state" in latest_state:
            best_state = latest_state["best_model_state"]
        elif best_ckpt_path is not None and best_ckpt_path.exists():
            best_ckpt = load_torch_checkpoint(best_ckpt_path)
            best_state = best_ckpt.get("model_state", best_state)
        print(
            f"[lstm] resumed from epoch={start_epoch - 1}, best_epoch={best_epoch}, best_valid={best_valid:.6f}",
            flush=True,
        )
    elif epoch_log_path is not None and save_epoch_log:
        pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "is_best_epoch", "best_valid_loss"]).to_csv(
            epoch_log_path,
            index=False,
            encoding="utf-8-sig",
        )

    if start_epoch > int(args.epochs):
        print(f"[lstm] latest checkpoint already reached epochs={args.epochs}; evaluating best state.", flush=True)

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, target_mean, target_std)
        valid_loss = eval_lstm_loss(model, valid_loader, criterion, device, target_mean, target_std)
        loss_rows.append({"epoch": float(epoch), "train_loss": float(train_loss), "valid_loss": float(valid_loss)})
        if (best_valid - valid_loss) > float(args.min_delta):
            best_valid = float(valid_loss)
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
            if best_ckpt_path is not None:
                torch.save(
                    {
                        "epoch": int(epoch),
                        "best_epoch": int(best_epoch),
                        "best_valid_loss": float(best_valid),
                        "model_state": best_state,
                        "target_mean": target_mean_np.tolist(),
                        "target_std": target_std_np.tolist(),
                        "run_signature": str(run_signature),
                    },
                    best_ckpt_path,
                )
        else:
            no_improve += 1
        if epoch_log_path is not None and save_epoch_log:
            pd.DataFrame(
                [
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_loss),
                        "valid_loss": float(valid_loss),
                        "is_best_epoch": int(best_epoch == epoch),
                        "best_valid_loss": float(best_valid),
                    }
                ]
            ).to_csv(epoch_log_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        if latest_state_path is not None and save_latest_state:
            torch.save(
                {
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_valid_loss": float(best_valid),
                    "no_improve": int(no_improve),
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "best_model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "target_mean": target_mean_np.tolist(),
                    "target_std": target_std_np.tolist(),
                    "run_signature": str(run_signature),
                },
                latest_state_path,
            )
        if status_path is not None:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "epoch": int(epoch),
                        "target_epochs": int(args.epochs),
                        "best_epoch": int(best_epoch),
                        "best_valid_loss": float(best_valid),
                        "no_improve": int(no_improve),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(
            f"[lstm] epoch={epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}",
            flush=True,
        )
        if no_improve >= int(args.patience):
            print(f"[lstm] early stopping at epoch={epoch}", flush=True)
            break
    if best_ckpt_path is not None and not best_ckpt_path.exists():
        torch.save(
            {
                "epoch": int(best_epoch),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid),
                "model_state": best_state,
                "target_mean": target_mean_np.tolist(),
                "target_std": target_std_np.tolist(),
                "run_signature": str(run_signature),
            },
            best_ckpt_path,
        )
    model.load_state_dict(best_state)
    train_eval_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, False, device)
    y_train_true, y_train_pred, _ = predict_lstm(model, train_eval_loader, device, target_mean_np, target_std_np)
    y_valid_true, y_valid_pred, _ = predict_lstm(model, valid_loader, device, target_mean_np, target_std_np)
    return model, pd.DataFrame(loss_rows), best_epoch, y_train_true, y_train_pred, y_valid_true, y_valid_pred


def build_dqdv_prediction_frame(
    dataset: MultiStepWindowDataset,
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    model_name: str,
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Build a wide prediction table for future dQdV targets."""

    target_count = len(target_cols)
    y_true = y_true_flat.reshape(-1, dataset.horizon, target_count)
    y_pred = y_pred_flat.reshape(-1, dataset.horizon, target_count)
    rows: List[Dict[str, object]] = []
    for sample_idx, meta in enumerate(dataset.metas):
        for h_idx in range(dataset.horizon):
            row: Dict[str, object] = {
                "model_name": model_name,
                "policy": meta.policy,
                "cell_code": meta.cell_code,
                "set_type": meta.set_type,
                "input_start_cycle": meta.input_start_cycle,
                "input_end_cycle": meta.input_end_cycle,
                "horizon_step": h_idx + 1,
                "target_cycle": meta.target_cycles[h_idx],
            }
            for target_idx, col in enumerate(target_cols):
                row[f"true_{col}"] = float(y_true[sample_idx, h_idx, target_idx])
                row[f"pred_{col}"] = float(y_pred[sample_idx, h_idx, target_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def build_bridge_feature_frame(
    dataset: MultiStepWindowDataset,
    dqdv_flat: np.ndarray,
    target_cols: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    """Build a bridge-model feature frame from true or predicted dQdV vectors."""

    target_count = len(target_cols)
    dqdv = dqdv_flat.reshape(-1, dataset.horizon, target_count)
    rows: List[Dict[str, object]] = []
    for sample_idx, meta in enumerate(dataset.metas):
        for h_idx in range(dataset.horizon):
            row: Dict[str, object] = {
                "source_name": source_name,
                "policy": meta.policy,
                "cell_code": meta.cell_code,
                "set_type": meta.set_type,
                "input_start_cycle": meta.input_start_cycle,
                "input_end_cycle": meta.input_end_cycle,
                "horizon_step": h_idx + 1,
                "target_cycle": meta.target_cycles[h_idx],
                "retention_true": meta.target_retentions[h_idx],
                "q_ref": meta.target_q_refs[h_idx],
                "q_discharge_true": meta.target_q_discharge[h_idx],
            }
            for context_col, value in zip(CONTEXT_COLS, meta.target_contexts[h_idx]):
                row[context_col] = float(value)
            for target_idx, col in enumerate(target_cols):
                row[col] = float(dqdv[sample_idx, h_idx, target_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def bridge_mode_keys(bridge_feature_mode: str) -> List[str]:
    """Return bridge model keys implied by the CLI bridge feature mode."""

    mode = str(bridge_feature_mode)
    if mode == "dqdv_only":
        return ["dqdv_only"]
    if mode == "dqdv_context":
        return ["dqdv_context"]
    if mode == "both":
        return ["dqdv_only", "dqdv_context"]
    raise ValueError(f"Unsupported bridge_feature_mode: {mode}")


def bridge_feature_columns(target_cols: Sequence[str], bridge_key: str) -> List[str]:
    """Return model input columns for one bridge variant."""

    if bridge_key == "dqdv_only":
        return list(target_cols)
    if bridge_key == "dqdv_context":
        return [*target_cols, *CONTEXT_COLS]
    raise ValueError(f"Unsupported bridge key: {bridge_key}")


def bridge_model_name(source_name: str, bridge_key: str) -> str:
    """Build a retention bridge metric model name."""

    if bridge_key == "dqdv_only":
        return f"{source_name}_only_bridge"
    if bridge_key == "dqdv_context":
        if source_name == "true_dqdv":
            return "true_dqdv_bridge"
        return f"{source_name}_bridge"
    raise ValueError(f"Unsupported bridge key: {bridge_key}")


def predicted_dqdv_source_name(model_name: str) -> str:
    """Map a dQdV model metric name to a stable predicted-dQdV source name."""

    if model_name == "lstm_dqdv":
        return "predicted_dqdv_lstm"
    if model_name == "rf_dqdv":
        return "predicted_dqdv_rf"
    return f"predicted_{model_name}"


def primary_predicted_bridge_name(model_name: str, bridge_feature_mode: str) -> str:
    """Return the tuning-selection bridge metric name for predicted dQdV."""

    source_name = predicted_dqdv_source_name(model_name)
    bridge_key = "dqdv_only" if str(bridge_feature_mode) in ["dqdv_only", "both"] else "dqdv_context"
    return bridge_model_name(source_name, bridge_key)


def train_bridge_model(
    args: argparse.Namespace,
    merged: pd.DataFrame,
    target_cols: Sequence[str],
    bridge_key: str,
) -> Pipeline:
    """Train one true-dQdV-to-retention bridge model on cycle-level training rows."""

    train_df = merged.loc[merged["set_type"] == "train"].copy()
    bridge_cols = bridge_feature_columns(target_cols, bridge_key)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=max(40, int(args.rf_n_estimators) // 2),
                    max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
                    min_samples_leaf=int(args.rf_min_samples_leaf),
                    max_features=0.7,
                    random_state=int(args.random_seed) + 4000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_df[bridge_cols].to_numpy(dtype=np.float32), train_df["retention"].to_numpy(dtype=np.float32))
    return model


def train_bridge_models(
    args: argparse.Namespace,
    merged: pd.DataFrame,
    target_cols: Sequence[str],
) -> Dict[str, Pipeline]:
    """Train all bridge variants requested by bridge_feature_mode."""

    return {
        key: train_bridge_model(args, merged, target_cols, key)
        for key in bridge_mode_keys(str(args.bridge_feature_mode))
    }


def predict_bridge_frame(
    bridge_model: Pipeline,
    frame: pd.DataFrame,
    target_cols: Sequence[str],
    model_name: str,
    bridge_key: str,
) -> pd.DataFrame:
    """Predict retention for one bridge input frame."""

    out = frame.copy()
    bridge_cols = bridge_feature_columns(target_cols, bridge_key)
    pred = bridge_model.predict(out[bridge_cols].to_numpy(dtype=np.float32))
    out["model_name"] = model_name
    out["pred_retention"] = pred.astype(float)
    out["pred_q_discharge"] = out["pred_retention"].to_numpy(dtype=float) * out["q_ref"].to_numpy(dtype=float)
    out["residual_retention"] = out["retention_true"].to_numpy(dtype=float) - out["pred_retention"].to_numpy(dtype=float)
    return out


def build_direct_retention_predictions(
    dataset: MultiStepWindowDataset,
    pred_flat: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Build direct interval-to-retention prediction rows."""

    pred = pred_flat.reshape(-1, dataset.horizon)
    rows: List[Dict[str, object]] = []
    for sample_idx, meta in enumerate(dataset.metas):
        for h_idx in range(dataset.horizon):
            pred_retention = float(pred[sample_idx, h_idx])
            q_ref = float(meta.target_q_refs[h_idx])
            rows.append(
                {
                    "source_name": model_name,
                    "model_name": model_name,
                    "policy": meta.policy,
                    "cell_code": meta.cell_code,
                    "set_type": meta.set_type,
                    "input_start_cycle": meta.input_start_cycle,
                    "input_end_cycle": meta.input_end_cycle,
                    "horizon_step": h_idx + 1,
                    "target_cycle": meta.target_cycles[h_idx],
                    "retention_true": meta.target_retentions[h_idx],
                    "pred_retention": pred_retention,
                    "q_ref": q_ref,
                    "q_discharge_true": meta.target_q_discharge[h_idx],
                    "pred_q_discharge": pred_retention * q_ref,
                    "residual_retention": float(meta.target_retentions[h_idx]) - pred_retention,
                }
            )
    return pd.DataFrame(rows)


def build_dqdv_prediction_frame_range(
    dataset: MultiStepWindowDataset,
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    model_name: str,
    target_cols: Sequence[str],
    start_sample: int,
    end_sample: int,
) -> pd.DataFrame:
    """Build one chunk of the wide validation dQdV prediction table."""

    target_count = len(target_cols)
    y_true = y_true_flat.reshape(-1, dataset.horizon, target_count)
    y_pred = y_pred_flat.reshape(-1, dataset.horizon, target_count)
    rows: List[Dict[str, object]] = []
    for sample_idx in range(int(start_sample), int(end_sample)):
        meta = dataset.metas[sample_idx]
        for h_idx in range(dataset.horizon):
            row: Dict[str, object] = {
                "model_name": model_name,
                "policy": meta.policy,
                "cell_code": meta.cell_code,
                "set_type": meta.set_type,
                "input_start_cycle": meta.input_start_cycle,
                "input_end_cycle": meta.input_end_cycle,
                "horizon_step": h_idx + 1,
                "target_cycle": meta.target_cycles[h_idx],
            }
            for target_idx, col in enumerate(target_cols):
                row[f"true_{col}"] = float(y_true[sample_idx, h_idx, target_idx])
                row[f"pred_{col}"] = float(y_pred[sample_idx, h_idx, target_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def write_dqdv_prediction_csv_stream(
    out_path: Path,
    dataset: MultiStepWindowDataset,
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    model_name: str,
    target_cols: Sequence[str],
    append: bool,
    chunk_samples: int = 1000,
) -> None:
    """Write validation dQdV predictions in chunks to avoid large DataFrames."""

    mode = "a" if append else "w"
    write_header = not append
    for start in range(0, len(dataset), int(chunk_samples)):
        end = min(start + int(chunk_samples), len(dataset))
        part = build_dqdv_prediction_frame_range(
            dataset=dataset,
            y_true_flat=y_true_flat,
            y_pred_flat=y_pred_flat,
            model_name=model_name,
            target_cols=target_cols,
            start_sample=start,
            end_sample=end,
        )
        part.to_csv(out_path, mode=mode, header=write_header, index=False, encoding=ENCODING)
        mode = "a"
        write_header = False


def build_bridge_feature_frame_range(
    dataset: MultiStepWindowDataset,
    dqdv_flat: np.ndarray,
    target_cols: Sequence[str],
    source_name: str,
    start_sample: int,
    end_sample: int,
) -> pd.DataFrame:
    """Build one chunk of bridge-model features from true or predicted dQdV."""

    target_count = len(target_cols)
    dqdv = dqdv_flat.reshape(-1, dataset.horizon, target_count)
    rows: List[Dict[str, object]] = []
    for sample_idx in range(int(start_sample), int(end_sample)):
        meta = dataset.metas[sample_idx]
        for h_idx in range(dataset.horizon):
            row: Dict[str, object] = {
                "source_name": source_name,
                "policy": meta.policy,
                "cell_code": meta.cell_code,
                "set_type": meta.set_type,
                "input_start_cycle": meta.input_start_cycle,
                "input_end_cycle": meta.input_end_cycle,
                "horizon_step": h_idx + 1,
                "target_cycle": meta.target_cycles[h_idx],
                "retention_true": meta.target_retentions[h_idx],
                "q_ref": meta.target_q_refs[h_idx],
                "q_discharge_true": meta.target_q_discharge[h_idx],
            }
            for context_col, value in zip(CONTEXT_COLS, meta.target_contexts[h_idx]):
                row[context_col] = float(value)
            for target_idx, col in enumerate(target_cols):
                row[col] = float(dqdv[sample_idx, h_idx, target_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def update_retention_metric_states(states: Dict[Tuple[str, str, int], Dict[str, float]], pred_df: pd.DataFrame) -> None:
    """Update streaming retention metric accumulators from a prediction chunk."""

    for (model_name, set_type, horizon_step), part in pred_df.groupby(["model_name", "set_type", "horizon_step"], sort=False):
        y_true = part["retention_true"].to_numpy(dtype=float)
        y_pred = part["pred_retention"].to_numpy(dtype=float)
        err = y_pred - y_true
        key = (str(model_name), str(set_type), int(horizon_step))
        state = states.setdefault(
            key,
            {
                "n": 0.0,
                "sum_y": 0.0,
                "sum_y2": 0.0,
                "sse": 0.0,
                "sae": 0.0,
            },
        )
        state["n"] += float(y_true.size)
        state["sum_y"] += float(np.sum(y_true))
        state["sum_y2"] += float(np.sum(y_true * y_true))
        state["sse"] += float(np.sum(err * err))
        state["sae"] += float(np.sum(np.abs(err)))


def retention_metric_states_to_frame(states: Mapping[Tuple[str, str, int], Mapping[str, float]]) -> pd.DataFrame:
    """Convert streaming retention metric accumulators to the metrics table."""

    rows: List[Metrics] = []
    for (model_name, set_type, horizon_step), state in sorted(states.items(), key=lambda item: item[0]):
        n_rows = int(state["n"])
        mse = float(state["sse"] / max(n_rows, 1))
        rmse = float(math.sqrt(mse))
        mae = float(state["sae"] / max(n_rows, 1))
        sst = float(state["sum_y2"] - (state["sum_y"] * state["sum_y"] / max(n_rows, 1)))
        r2 = float("nan") if n_rows < 2 or sst <= 0.0 else float(1.0 - state["sse"] / sst)
        rows.append(
            Metrics(
                model_name=str(model_name),
                set_type=str(set_type),
                target="retention",
                horizon_step=int(horizon_step),
                n_rows=n_rows,
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2=r2,
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def append_scatter_sample(
    sample_frames: List[pd.DataFrame],
    sample_counts: Dict[str, int],
    pred_df: pd.DataFrame,
    max_rows_per_model: int = 20000,
) -> None:
    """Keep a bounded validation sample for scatter plots."""

    valid = pred_df.loc[pred_df["set_type"] == "valid"]
    if valid.empty:
        return
    for model_name, part in valid.groupby("model_name", sort=False):
        current = int(sample_counts.get(str(model_name), 0))
        remaining = int(max_rows_per_model) - current
        if remaining <= 0:
            continue
        keep = part if len(part) <= remaining else part.sample(n=remaining, random_state=20260508 + current)
        sample_frames.append(keep.copy())
        sample_counts[str(model_name)] = current + int(len(keep))


def stream_bridge_predictions(
    bridge_model: Pipeline,
    dataset: MultiStepWindowDataset,
    dqdv_flat: np.ndarray,
    target_cols: Sequence[str],
    source_name: str,
    model_name: str,
    bridge_key: str,
    metric_states: Dict[Tuple[str, str, int], Dict[str, float]],
    valid_csv_path: Optional[Path],
    append_valid_csv: bool,
    scatter_frames: List[pd.DataFrame],
    scatter_counts: Dict[str, int],
    chunk_samples: int = 2000,
) -> bool:
    """Predict retention in chunks, updating metrics and optionally writing valid rows."""

    wrote_valid = bool(append_valid_csv)
    mode = "a" if append_valid_csv else "w"
    write_header = not append_valid_csv
    for start in range(0, len(dataset), int(chunk_samples)):
        end = min(start + int(chunk_samples), len(dataset))
        feature_frame = build_bridge_feature_frame_range(
            dataset=dataset,
            dqdv_flat=dqdv_flat,
            target_cols=target_cols,
            source_name=source_name,
            start_sample=start,
            end_sample=end,
        )
        pred_frame = predict_bridge_frame(bridge_model, feature_frame, target_cols, model_name, bridge_key)
        update_retention_metric_states(metric_states, pred_frame)
        append_scatter_sample(scatter_frames, scatter_counts, pred_frame)
        if valid_csv_path is not None and str(dataset.metas[0].set_type) == "valid":
            pred_frame.to_csv(valid_csv_path, mode=mode, header=write_header, index=False, encoding=ENCODING)
            mode = "a"
            write_header = False
            wrote_valid = True
    return wrote_valid


def build_direct_retention_predictions_range(
    dataset: MultiStepWindowDataset,
    pred_flat: np.ndarray,
    model_name: str,
    start_sample: int,
    end_sample: int,
) -> pd.DataFrame:
    """Build one chunk of direct interval-to-retention prediction rows."""

    pred = pred_flat.reshape(-1, dataset.horizon)
    rows: List[Dict[str, object]] = []
    for sample_idx in range(int(start_sample), int(end_sample)):
        meta = dataset.metas[sample_idx]
        for h_idx in range(dataset.horizon):
            pred_retention = float(pred[sample_idx, h_idx])
            q_ref = float(meta.target_q_refs[h_idx])
            rows.append(
                {
                    "source_name": model_name,
                    "model_name": model_name,
                    "policy": meta.policy,
                    "cell_code": meta.cell_code,
                    "set_type": meta.set_type,
                    "input_start_cycle": meta.input_start_cycle,
                    "input_end_cycle": meta.input_end_cycle,
                    "horizon_step": h_idx + 1,
                    "target_cycle": meta.target_cycles[h_idx],
                    "retention_true": meta.target_retentions[h_idx],
                    "pred_retention": pred_retention,
                    "q_ref": q_ref,
                    "q_discharge_true": meta.target_q_discharge[h_idx],
                    "pred_q_discharge": pred_retention * q_ref,
                    "residual_retention": float(meta.target_retentions[h_idx]) - pred_retention,
                }
            )
    return pd.DataFrame(rows)


def stream_direct_retention_predictions(
    dataset: MultiStepWindowDataset,
    pred_flat: np.ndarray,
    model_name: str,
    metric_states: Dict[Tuple[str, str, int], Dict[str, float]],
    valid_csv_path: Optional[Path],
    append_valid_csv: bool,
    scatter_frames: List[pd.DataFrame],
    scatter_counts: Dict[str, int],
    chunk_samples: int = 2000,
) -> bool:
    """Stream direct interval-to-retention predictions for metrics and valid output."""

    wrote_valid = bool(append_valid_csv)
    mode = "a" if append_valid_csv else "w"
    write_header = not append_valid_csv
    for start in range(0, len(dataset), int(chunk_samples)):
        end = min(start + int(chunk_samples), len(dataset))
        pred_frame = build_direct_retention_predictions_range(dataset, pred_flat, model_name, start, end)
        update_retention_metric_states(metric_states, pred_frame)
        append_scatter_sample(scatter_frames, scatter_counts, pred_frame)
        if valid_csv_path is not None and str(dataset.metas[0].set_type) == "valid":
            pred_frame.to_csv(valid_csv_path, mode=mode, header=write_header, index=False, encoding=ENCODING)
            mode = "a"
            write_header = False
            wrote_valid = True
    return wrote_valid


def build_retention_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute retention metrics by model, set, and horizon."""

    rows: List[Metrics] = []
    for (model_name, set_type, horizon_step), part in pred_df.groupby(["model_name", "set_type", "horizon_step"], sort=True):
        rows.append(
            calc_metrics(
                y_true=part["retention_true"].to_numpy(dtype=float),
                y_pred=part["pred_retention"].to_numpy(dtype=float),
                model_name=str(model_name),
                set_type=str(set_type),
                target="retention",
                horizon_step=int(horizon_step),
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def save_loss_plot(loss_df: pd.DataFrame, out_path: Path) -> None:
    """Save LSTM train/valid loss curve."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss", color="#2563eb")
    ax.plot(loss_df["epoch"], loss_df["valid_loss"], label="valid_loss", color="#dc2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standardized dQdV MSE")
    ax.set_title("LSTM dQdV Train/Valid Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def save_retention_scatter(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Save validation retention scatter for all retention prediction sources."""

    import matplotlib.pyplot as plt

    valid = pred_df.loc[pred_df["set_type"] == "valid"].copy()
    models = valid["model_name"].drop_duplicates().tolist()
    if not models:
        return
    n_cols = min(3, len(models))
    n_rows = int(math.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.8 * n_rows), squeeze=False)
    y_all = valid["retention_true"].to_numpy(dtype=float)
    p_all = valid["pred_retention"].to_numpy(dtype=float)
    lo = float(min(np.nanmin(y_all), np.nanmin(p_all)))
    hi = float(max(np.nanmax(y_all), np.nanmax(p_all)))
    for idx, model_name in enumerate(models):
        ax = axes[idx // n_cols][idx % n_cols]
        part = valid.loc[valid["model_name"] == model_name]
        ax.scatter(part["retention_true"], part["pred_retention"], s=8, alpha=0.35)
        ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", linewidth=1.2)
        first_h = int(part["horizon_step"].min())
        m = metrics_df.loc[
            (metrics_df["model_name"] == model_name)
            & (metrics_df["set_type"] == "valid")
            & (metrics_df["horizon_step"] == first_h)
        ]
        title = str(model_name)
        if not m.empty:
            title = f"{model_name}\nH{first_h} R2={float(m.iloc[0]['r2']):.4f} RMSE={float(m.iloc[0]['rmse']):.5f}"
        ax.set_title(title)
        ax.set_xlabel("True retention")
        ax.set_ylabel("Predicted retention")
        ax.grid(True, linestyle="--", alpha=0.3)
    for idx in range(len(models), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def build_dataset_checks(
    merged: pd.DataFrame,
    train_dataset: MultiStepWindowDataset,
    valid_dataset: MultiStepWindowDataset,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    charge_stats: Mapping[str, int],
    discharge_stats: Mapping[str, int],
) -> pd.DataFrame:
    """Build dataset diagnostics and pass/fail checks."""

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
    x_mat = merged[list(feature_cols)].to_numpy(dtype=np.float32)
    y_mat = merged[list(target_cols)].to_numpy(dtype=np.float32)
    target_after_input = True
    for dataset in [train_dataset, valid_dataset]:
        for meta in dataset.metas[: min(1000, len(dataset.metas))]:
            target_after_input = target_after_input and all(int(c) > int(meta.input_end_cycle) for c in meta.target_cycles)
    checks = [
        ("merged_cycle_rows", int(len(merged)), 1),
        ("train_window_rows", int(len(train_dataset)), int(len(train_dataset) > 0)),
        ("valid_window_rows", int(len(valid_dataset)), int(len(valid_dataset) > 0)),
        ("input_feature_dim", int(len(feature_cols)), int(len(feature_cols) > 0)),
        ("dqdv_target_dim", int(len(target_cols)), int(len(target_cols) > 0)),
        ("check_split_overlap_zero", int(len(train_keys.intersection(valid_keys))), int(len(train_keys.intersection(valid_keys)) == 0)),
        ("check_input_features_finite", int(np.isfinite(x_mat).all()), int(np.isfinite(x_mat).all())),
        ("check_dqdv_targets_finite", int(np.isfinite(y_mat).all()), int(np.isfinite(y_mat).all())),
        ("check_retention_range", int(((merged["retention"] >= 0.3) & (merged["retention"] <= 1.1)).all()), int(((merged["retention"] >= 0.3) & (merged["retention"] <= 1.1)).all())),
        ("check_future_targets_after_input", int(target_after_input), int(target_after_input)),
        ("charge_cross_bin_feature_dim", int(charge_stats.get("charge_cross_bin_feature_dim", 0)), int(charge_stats.get("charge_cross_bin_feature_dim", 0) == 60)),
        ("discharge_range_count", int(discharge_stats.get("discharge_range_count", 0)), int(discharge_stats.get("discharge_range_count", 0) > 0)),
    ]
    return pd.DataFrame(checks, columns=["check_item", "value", "pass_flag"])


def best_valid_retention_row(metrics_df: pd.DataFrame, model_name: str) -> Optional[pd.Series]:
    """Return the first-horizon valid retention metric row for a model."""

    rows = metrics_df.loc[
        (metrics_df["model_name"] == model_name)
        & (metrics_df["set_type"] == "valid")
        & (metrics_df["horizon_step"] == 1)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_report(
    args: argparse.Namespace,
    device: torch.device,
    target_cols: Sequence[str],
    feature_cols: Sequence[str],
    dataset_checks: pd.DataFrame,
    dqdv_metrics: pd.DataFrame,
    retention_metrics: pd.DataFrame,
    best_epoch: Optional[int],
) -> str:
    """Build a Chinese markdown report for the full pipeline."""

    lines: List[str] = []
    lines.append("# 区间累计工况 -> dQdV -> 容量保持率预测报告")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 设备：`{device.type}`")
    lines.append(f"- 模型范围：`{args.model_family}`")
    if bool(getattr(args, "skip_direct_baseline", False)):
        lines.append("- 直接区间特征 baseline：`skipped by --skip-direct-baseline`")
    else:
        lines.append("- 直接区间特征 baseline：`enabled`")
    lines.append(f"- history_len：`{args.history_len}`")
    lines.append(f"- horizon：`{args.horizon}`")
    lines.append(f"- dQdV目标包：`{args.target_pack}`")
    lines.append(f"- 输入特征包：`{args.input_feature_pack}`")
    lines.append(f"- 输入标准化：`{args.input_scaling}`")
    lines.append(f"- retention bridge 特征模式：`{args.bridge_feature_mode}`")
    lines.append(f"- LSTM输出头：`{args.lstm_head}`")
    lines.append(f"- direct LSTM retention baseline：`{bool(getattr(args, 'direct_retention_baseline', False))}`")
    lines.append(f"- retention定义：`q_discharge / q_ref`，`q_ref` 为前 `{args.q_ref_cycles}` 个有效cycle中位数")
    lines.append(f"- retention过滤：`{args.retention_min} <= retention <= {args.retention_max}`")
    if best_epoch is not None:
        lines.append(f"- LSTM最佳epoch：`{best_epoch}`")
    lines.append("")
    lines.append("## 2. 数据与特征")
    merged_rows = dataset_checks.loc[dataset_checks["check_item"] == "merged_cycle_rows", "value"].iloc[0]
    train_windows = dataset_checks.loc[dataset_checks["check_item"] == "train_window_rows", "value"].iloc[0]
    valid_windows = dataset_checks.loc[dataset_checks["check_item"] == "valid_window_rows", "value"].iloc[0]
    lines.append(f"- 合并后cycle级样本数：**{int(merged_rows):,}**")
    lines.append(f"- 训练窗口数：**{int(train_windows):,}**")
    lines.append(f"- 验证窗口数：**{int(valid_windows):,}**")
    lines.append(f"- 每个时间步输入维度：**{len(feature_cols):,}**")
    lines.append("- dQdV预测目标：")
    for col in target_cols:
        lines.append(f"  - `{col}`")
    lines.append("")
    lines.append("## 3. dQdV预测指标")
    lines.append("| model | set | horizon | target | MSE | RMSE | MAE | R2 |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|")
    for row in dqdv_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.set_type} | {int(row.horizon_step)} | {row.target} | "
            f"{float(row.mse):.8f} | {float(row.rmse):.6f} | {float(row.mae):.6f} | {float(row.r2):.6f} |"
        )
    lines.append("")
    lines.append("## 4. retention桥接指标")
    lines.append("| model | set | horizon | MSE | RMSE | MAE | R2 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in retention_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.set_type} | {int(row.horizon_step)} | "
            f"{float(row.mse):.8f} | {float(row.rmse):.6f} | {float(row.mae):.6f} | {float(row.r2):.6f} |"
        )
    lines.append("")
    lines.append("## 5. 结论")
    direct = best_valid_retention_row(retention_metrics, "direct_interval_baseline")
    direct_lstm = best_valid_retention_row(retention_metrics, "direct_lstm_retention_baseline")
    true_bridge = best_valid_retention_row(retention_metrics, "true_dqdv_only_bridge")
    if true_bridge is None:
        true_bridge = best_valid_retention_row(retention_metrics, "true_dqdv_bridge")
    predicted_rows = retention_metrics.loc[
        retention_metrics["model_name"].astype(str).str.startswith("predicted_dqdv_")
        & (retention_metrics["set_type"] == "valid")
        & (retention_metrics["horizon_step"] == 1)
    ].copy()
    if direct is not None:
        lines.append(f"- 直接区间特征 -> retention 基线 H1 valid R2：**{float(direct['r2']):.6f}**。")
    elif bool(getattr(args, "skip_direct_baseline", False)):
        lines.append("- 直接区间特征 -> retention 基线：已按 `--skip-direct-baseline` 跳过。")
    if direct_lstm is not None:
        lines.append(f"- 直接 LSTM 工况 -> retention 基线 H1 valid R2：**{float(direct_lstm['r2']):.6f}**。")
    if true_bridge is not None:
        lines.append(f"- 真实 dQdV -> retention 参考 H1 valid R2：**{float(true_bridge['r2']):.6f}**。")
    if not predicted_rows.empty:
        best_pred = predicted_rows.sort_values("r2", ascending=False, kind="mergesort").iloc[0]
        lines.append(
            f"- 最优预测 dQdV 链路为 `{best_pred['model_name']}`，H1 valid R2：**{float(best_pred['r2']):.6f}**。"
        )
        if direct is not None:
            gap = float(best_pred["r2"]) - float(direct["r2"])
            lines.append(f"- 预测 dQdV 链路相对直接基线的R2差值：**{gap:.6f}**。")
    lines.append("")
    lines.append("## 6. 图表")
    if best_epoch is not None:
        lines.append("![loss_curve](./loss_curve.png)")
        lines.append("")
    lines.append("![valid_scatter_retention](./valid_scatter_retention.png)")
    return "\n".join(lines)


def prepare_pipeline_data(args: argparse.Namespace) -> PreparedData:
    """Load data, build windows, and fit the true-dQdV bridge model once."""

    target_cols = DQDV_TARGET_PACKS[str(args.target_pack)]
    _, _, split_map = load_split(args.train_split_path, args.valid_split_path)
    allowed_keys = build_allowed_key_tokens(split_map, args.smoke_train_cells, args.smoke_valid_cells) if args.smoke_test else None

    print("Loading charge features...", flush=True)
    charge_df, charge_stats, charge_cols = load_charge_feature_table(args.charge_timeseries_path, allowed_keys)
    print("Loading discharge features...", flush=True)
    discharge_df, discharge_stats, discharge_cols = load_discharge_feature_table(args.discharge_interval_path, allowed_keys)
    print("Loading dQdV targets and retention labels...", flush=True)
    dqdv_df = load_dqdv_table(args.dqdv_path, target_cols)
    label_df = load_retention_labels(
        args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )
    dqdv_df = filter_allowed_keys(dqdv_df, allowed_keys)
    label_df = filter_allowed_keys(label_df, allowed_keys)
    split_use = filter_allowed_keys(split_map, allowed_keys)
    merged, feature_cols, input_transform_info = build_merged_cycle_table(
        split_map=split_use,
        charge_df=charge_df,
        charge_cols=charge_cols,
        discharge_df=discharge_df,
        discharge_cols=discharge_cols,
        dqdv_df=dqdv_df,
        label_df=label_df,
        target_cols=target_cols,
        input_feature_pack=str(args.input_feature_pack),
        input_scaling=str(args.input_scaling),
    )
    if merged.empty:
        raise RuntimeError("Merged dataset is empty.")
    seq_map = build_sequences(merged, feature_cols, target_cols)
    train_map, valid_map = split_sequence_map(seq_map)
    train_dataset = MultiStepWindowDataset(
        train_map,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
        seed=int(args.random_seed),
    )
    valid_dataset = MultiStepWindowDataset(
        valid_map,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
        seed=int(args.random_seed) + 1,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Train or valid windows are empty. Reduce history_len/horizon or check data coverage.")
    needs_flat_arrays = str(args.model_family) in ["both", "rf"] or (
        str(args.run_mode) == "single" and not bool(getattr(args, "skip_direct_baseline", False))
    )
    if needs_flat_arrays:
        print("Materializing flattened interval arrays...", flush=True)
        x_train_flat, y_train_dqdv = dataset_to_numpy(train_dataset)
        x_valid_flat, y_valid_dqdv = dataset_to_numpy(valid_dataset)
    else:
        print("Materializing dQdV target arrays...", flush=True)
        x_train_flat = None
        x_valid_flat = None
        y_train_dqdv = dqdv_targets_from_dataset(train_dataset)
        y_valid_dqdv = dqdv_targets_from_dataset(valid_dataset)
    bridge_models = train_bridge_models(args, merged, target_cols)
    return PreparedData(
        merged=merged,
        feature_cols=list(feature_cols),
        target_cols=list(target_cols),
        charge_stats=dict(charge_stats),
        discharge_stats=dict(discharge_stats),
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        x_train_flat=x_train_flat,
        y_train_dqdv=y_train_dqdv,
        x_valid_flat=x_valid_flat,
        y_valid_dqdv=y_valid_dqdv,
        bridge_models=bridge_models,
        input_transform_info=dict(input_transform_info),
    )


def lstm_run_signature(
    args: argparse.Namespace,
    cfg: Optional[TrialConfig],
    feature_dim: int,
    output_size: int,
    target_name: str = "dqdv",
) -> str:
    """Build a compact signature for checkpoint compatibility checks."""

    payload = {
        "target_name": str(target_name),
        "history_len": int(args.history_len),
        "horizon": int(args.horizon),
        "target_pack": str(args.target_pack),
        "input_feature_pack": str(args.input_feature_pack),
        "input_scaling": str(args.input_scaling),
        "lstm_head": str(args.lstm_head),
        "hidden_size": int(args.hidden_size),
        "learning_rate": float(args.learning_rate),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "feature_dim": int(feature_dim),
        "output_size": int(output_size),
        "trial_id": None if cfg is None else int(cfg.trial_id),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def evaluate_lstm_predictions(
    prepared: PreparedData,
    train_pred: np.ndarray,
    valid_pred: np.ndarray,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate LSTM dQdV predictions and retention bridge metrics."""

    dqdv_metrics = pd.DataFrame(
        [
            asdict(row)
            for row in [
                *build_dqdv_metrics(
                    prepared.y_train_dqdv,
                    train_pred,
                    model_name,
                    "train",
                    prepared.train_dataset.horizon,
                    prepared.target_cols,
                ),
                *build_dqdv_metrics(
                    prepared.y_valid_dqdv,
                    valid_pred,
                    model_name,
                    "valid",
                    prepared.valid_dataset.horizon,
                    prepared.target_cols,
                ),
            ]
        ]
    )
    retention_metric_states: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    scatter_frames: List[pd.DataFrame] = []
    scatter_counts: Dict[str, int] = {}
    source_name = predicted_dqdv_source_name(model_name)
    for bridge_key, bridge_model in prepared.bridge_models.items():
        metric_model_name = bridge_model_name(source_name, bridge_key)
        stream_bridge_predictions(
            bridge_model=bridge_model,
            dataset=prepared.train_dataset,
            dqdv_flat=train_pred,
            target_cols=prepared.target_cols,
            source_name=source_name,
            model_name=metric_model_name,
            bridge_key=bridge_key,
            metric_states=retention_metric_states,
            valid_csv_path=None,
            append_valid_csv=False,
            scatter_frames=scatter_frames,
            scatter_counts=scatter_counts,
        )
        stream_bridge_predictions(
            bridge_model=bridge_model,
            dataset=prepared.valid_dataset,
            dqdv_flat=valid_pred,
            target_cols=prepared.target_cols,
            source_name=source_name,
            model_name=metric_model_name,
            bridge_key=bridge_key,
            metric_states=retention_metric_states,
            valid_csv_path=None,
            append_valid_csv=False,
            scatter_frames=scatter_frames,
            scatter_counts=scatter_counts,
        )
    retention_metrics = retention_metric_states_to_frame(retention_metric_states)
    return dqdv_metrics, retention_metrics


def select_lstm_score(
    retention_metrics: pd.DataFrame,
    dqdv_metrics: pd.DataFrame,
    model_name: str,
    bridge_feature_mode: str,
) -> Dict[str, float]:
    """Select the fixed tuning score from retention and dQdV metrics."""

    bridge_name = primary_predicted_bridge_name(model_name, bridge_feature_mode)
    h1_ret = retention_metrics.loc[
        (retention_metrics["model_name"] == bridge_name)
        & (retention_metrics["set_type"] == "valid")
        & (retention_metrics["horizon_step"] == 1)
    ]
    if h1_ret.empty:
        raise RuntimeError(f"Missing H1 valid retention metric for {bridge_name}.")
    h1_dqdv = dqdv_metrics.loc[
        (dqdv_metrics["model_name"] == model_name)
        & (dqdv_metrics["set_type"] == "valid")
        & (dqdv_metrics["horizon_step"] == 1)
    ]
    if h1_dqdv.empty:
        raise RuntimeError(f"Missing H1 valid dQdV metrics for {model_name}.")
    ret_row = h1_ret.iloc[0]
    return {
        "retention_selection_model": bridge_name,
        "retention_valid_r2": float(ret_row["r2"]),
        "retention_valid_mse": float(ret_row["mse"]),
        "retention_valid_rmse": float(ret_row["rmse"]),
        "retention_valid_mae": float(ret_row["mae"]),
        "dqdv_valid_mean_r2": float(h1_dqdv["r2"].mean()),
        "dqdv_valid_mean_mse": float(h1_dqdv["mse"].mean()),
        "dqdv_valid_mean_rmse": float(h1_dqdv["rmse"].mean()),
        "dqdv_valid_mean_mae": float(h1_dqdv["mae"].mean()),
    }


def format_trial_metric_summary(row: Mapping[str, object]) -> str:
    """Format retention and dQdV trial metrics for real-time logging."""

    def metric_value(key: str) -> float:
        """Read a numeric metric from a trial row."""

        return float(row.get(key, float("nan")))

    return (
        "retention(valid H1): "
        f"R2={metric_value('retention_valid_r2'):.6f} "
        f"MSE={metric_value('retention_valid_mse'):.8f} "
        f"RMSE={metric_value('retention_valid_rmse'):.6f} "
        f"MAE={metric_value('retention_valid_mae'):.6f} | "
        "dQdV(valid H1 mean): "
        f"R2={metric_value('dqdv_valid_mean_r2'):.6f} "
        f"MSE={metric_value('dqdv_valid_mean_mse'):.8f} "
        f"RMSE={metric_value('dqdv_valid_mean_rmse'):.6f} "
        f"MAE={metric_value('dqdv_valid_mean_mae'):.6f}"
    )


def apply_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Clamp runtime parameters for a fast smoke test."""

    if not bool(args.smoke_test):
        return args
    args.epochs = min(int(args.epochs), 3)
    args.patience = min(int(args.patience), 2)
    args.rf_n_estimators = min(int(args.rf_n_estimators), 30)
    args.max_train_windows = int(args.max_train_windows) if int(args.max_train_windows) > 0 else 1024
    args.max_valid_windows = int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else 512
    return args


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    """Run the full interval-to-dQdV-to-retention pipeline."""

    args = apply_smoke_defaults(args)
    set_seed(int(args.random_seed))
    ensure_matplotlib_config()
    device = resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_cols = DQDV_TARGET_PACKS[str(args.target_pack)]
    _, _, split_map = load_split(args.train_split_path, args.valid_split_path)
    allowed_keys = build_allowed_key_tokens(split_map, args.smoke_train_cells, args.smoke_valid_cells) if args.smoke_test else None

    print("Loading charge features...", flush=True)
    charge_df, charge_stats, charge_cols = load_charge_feature_table(args.charge_timeseries_path, allowed_keys)
    print("Loading discharge features...", flush=True)
    discharge_df, discharge_stats, discharge_cols = load_discharge_feature_table(args.discharge_interval_path, allowed_keys)
    print("Loading dQdV targets and retention labels...", flush=True)
    dqdv_df = load_dqdv_table(args.dqdv_path, target_cols)
    label_df = load_retention_labels(
        args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )
    dqdv_df = filter_allowed_keys(dqdv_df, allowed_keys)
    label_df = filter_allowed_keys(label_df, allowed_keys)
    split_use = filter_allowed_keys(split_map, allowed_keys)
    merged, feature_cols, input_transform_info = build_merged_cycle_table(
        split_map=split_use,
        charge_df=charge_df,
        charge_cols=charge_cols,
        discharge_df=discharge_df,
        discharge_cols=discharge_cols,
        dqdv_df=dqdv_df,
        label_df=label_df,
        target_cols=target_cols,
        input_feature_pack=str(args.input_feature_pack),
        input_scaling=str(args.input_scaling),
    )
    if merged.empty:
        raise RuntimeError("Merged dataset is empty.")
    seq_map = build_sequences(merged, feature_cols, target_cols)
    train_map, valid_map = split_sequence_map(seq_map)
    train_dataset = MultiStepWindowDataset(
        train_map,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
        seed=int(args.random_seed),
    )
    valid_dataset = MultiStepWindowDataset(
        valid_map,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
        seed=int(args.random_seed) + 1,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Train or valid windows are empty. Reduce history_len/horizon or check data coverage.")

    needs_rf_arrays = str(args.model_family) in ["both", "rf"]
    needs_direct_baseline = not bool(getattr(args, "skip_direct_baseline", False))
    if needs_rf_arrays or needs_direct_baseline:
        print("Materializing flattened interval arrays...", flush=True)
        x_train_flat, y_train_dqdv = dataset_to_numpy(train_dataset)
        x_valid_flat, y_valid_dqdv = dataset_to_numpy(valid_dataset)
    else:
        print("Materializing dQdV target arrays...", flush=True)
        x_train_flat = None
        x_valid_flat = None
        y_train_dqdv = dqdv_targets_from_dataset(train_dataset)
        y_valid_dqdv = dqdv_targets_from_dataset(valid_dataset)
    dqdv_metric_rows: List[Metrics] = []
    dqdv_valid_outputs: List[Tuple[str, MultiStepWindowDataset, np.ndarray, np.ndarray]] = []
    predicted_dqdv_for_bridge: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    best_epoch: Optional[int] = None
    loss_df = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss"])

    if str(args.model_family) in ["both", "rf"]:
        print("Training RF dQdV model...", flush=True)
        rf_model = make_rf_model(args, seed=int(args.random_seed) + 1000)
        rf_model.fit(x_train_flat, y_train_dqdv)
        rf_train_pred = rf_model.predict(x_train_flat).astype(np.float32)
        rf_valid_pred = rf_model.predict(x_valid_flat).astype(np.float32)
        dqdv_metric_rows.extend(build_dqdv_metrics(y_train_dqdv, rf_train_pred, "rf_dqdv", "train", args.horizon, target_cols))
        dqdv_metric_rows.extend(build_dqdv_metrics(y_valid_dqdv, rf_valid_pred, "rf_dqdv", "valid", args.horizon, target_cols))
        dqdv_valid_outputs.append(("rf_dqdv", valid_dataset, y_valid_dqdv, rf_valid_pred))
        predicted_dqdv_for_bridge["predicted_dqdv_rf"] = (rf_train_pred, rf_valid_pred)

    if str(args.model_family) in ["both", "lstm"]:
        print("Training LSTM dQdV model...", flush=True)
        (
            _lstm_model,
            loss_df,
            best_epoch,
            lstm_train_true,
            lstm_train_pred,
            lstm_valid_true,
            lstm_valid_pred,
        ) = train_lstm_model(
            args=args,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            y_train_flat=y_train_dqdv,
            device=device,
            output_size=int(y_train_dqdv.shape[1]),
            feature_dim=int(len(feature_cols)),
            checkpoint_dir=args.output_dir / "checkpoints" / "lstm_single",
            resume_interrupted=bool(args.resume_interrupted),
            save_latest_state=True,
            save_epoch_log=True,
            status_path=args.output_dir / "checkpoints" / "lstm_single" / "runtime_status.json",
            run_signature=lstm_run_signature(args, None, int(len(feature_cols)), int(y_train_dqdv.shape[1])),
        )
        dqdv_metric_rows.extend(build_dqdv_metrics(lstm_train_true, lstm_train_pred, "lstm_dqdv", "train", args.horizon, target_cols))
        dqdv_metric_rows.extend(build_dqdv_metrics(lstm_valid_true, lstm_valid_pred, "lstm_dqdv", "valid", args.horizon, target_cols))
        dqdv_valid_outputs.append(("lstm_dqdv", valid_dataset, lstm_valid_true, lstm_valid_pred))
        predicted_dqdv_for_bridge["predicted_dqdv_lstm"] = (lstm_train_pred, lstm_valid_pred)

    out_checks = args.output_dir / "dataset_checks.csv"
    out_dqdv_metrics = args.output_dir / "train_valid_metrics_dqdv.csv"
    out_dqdv_pred = args.output_dir / "valid_dqdv_predictions.csv"
    out_ret_metrics = args.output_dir / "retention_bridge_metrics.csv"
    out_ret_pred = args.output_dir / "valid_retention_predictions.csv"
    out_loss = args.output_dir / "loss_curve.csv"
    out_loss_png = args.output_dir / "loss_curve.png"
    out_direct_loss = args.output_dir / "direct_retention_loss_curve.csv"
    out_direct_loss_png = args.output_dir / "direct_retention_loss_curve.png"
    out_scatter_png = args.output_dir / "valid_scatter_retention.png"
    out_config = args.output_dir / "run_config.json"
    out_report = args.output_dir / "interval_to_dqdv_retention_report.md"
    for stale_path in [out_dqdv_pred, out_ret_pred]:
        if stale_path.exists():
            stale_path.unlink()

    print(f"Training retention bridge ({args.bridge_feature_mode})...", flush=True)
    bridge_models = train_bridge_models(args, merged, target_cols)
    retention_metric_states: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    retention_scatter_frames: List[pd.DataFrame] = []
    retention_scatter_counts: Dict[str, int] = {}
    ret_valid_written = False
    for bridge_key, bridge_model in bridge_models.items():
        metric_model_name = bridge_model_name("true_dqdv", bridge_key)
        stream_bridge_predictions(
            bridge_model=bridge_model,
            dataset=train_dataset,
            dqdv_flat=y_train_dqdv,
            target_cols=target_cols,
            source_name="true_dqdv",
            model_name=metric_model_name,
            bridge_key=bridge_key,
            metric_states=retention_metric_states,
            valid_csv_path=None,
            append_valid_csv=False,
            scatter_frames=retention_scatter_frames,
            scatter_counts=retention_scatter_counts,
        )
        ret_valid_written = stream_bridge_predictions(
            bridge_model=bridge_model,
            dataset=valid_dataset,
            dqdv_flat=y_valid_dqdv,
            target_cols=target_cols,
            source_name="true_dqdv",
            model_name=metric_model_name,
            bridge_key=bridge_key,
            metric_states=retention_metric_states,
            valid_csv_path=out_ret_pred,
            append_valid_csv=ret_valid_written,
            scatter_frames=retention_scatter_frames,
            scatter_counts=retention_scatter_counts,
        )
    for model_name, (train_pred, valid_pred) in predicted_dqdv_for_bridge.items():
        for bridge_key, bridge_model in bridge_models.items():
            metric_model_name = bridge_model_name(model_name, bridge_key)
            stream_bridge_predictions(
                bridge_model=bridge_model,
                dataset=train_dataset,
                dqdv_flat=train_pred,
                target_cols=target_cols,
                source_name=model_name,
                model_name=metric_model_name,
                bridge_key=bridge_key,
                metric_states=retention_metric_states,
                valid_csv_path=None,
                append_valid_csv=False,
                scatter_frames=retention_scatter_frames,
                scatter_counts=retention_scatter_counts,
            )
            ret_valid_written = stream_bridge_predictions(
                bridge_model=bridge_model,
                dataset=valid_dataset,
                dqdv_flat=valid_pred,
                target_cols=target_cols,
                source_name=model_name,
                model_name=metric_model_name,
                bridge_key=bridge_key,
                metric_states=retention_metric_states,
                valid_csv_path=out_ret_pred,
                append_valid_csv=ret_valid_written,
                scatter_frames=retention_scatter_frames,
                scatter_counts=retention_scatter_counts,
            )

    y_train_ret: Optional[np.ndarray] = None
    y_valid_ret: Optional[np.ndarray] = None
    if bool(getattr(args, "direct_retention_baseline", False)):
        print("Training direct LSTM retention baseline...", flush=True)
        y_train_ret = retention_targets_from_dataset(train_dataset)
        y_valid_ret = retention_targets_from_dataset(valid_dataset)
        direct_train_dataset = RetentionWindowDataset(train_dataset)
        direct_valid_dataset = RetentionWindowDataset(valid_dataset)
        (
            _direct_lstm_model,
            direct_loss_df,
            _direct_best_epoch,
            _direct_train_true,
            direct_train_pred,
            _direct_valid_true,
            direct_valid_pred,
        ) = train_lstm_model(
            args=args,
            train_dataset=direct_train_dataset,
            valid_dataset=direct_valid_dataset,
            y_train_flat=y_train_ret,
            device=device,
            output_size=int(y_train_ret.shape[1]),
            feature_dim=int(len(feature_cols)),
            checkpoint_dir=args.output_dir / "checkpoints" / "lstm_direct_retention",
            resume_interrupted=bool(args.resume_interrupted),
            save_latest_state=True,
            save_epoch_log=True,
            status_path=args.output_dir / "checkpoints" / "lstm_direct_retention" / "runtime_status.json",
            run_signature=lstm_run_signature(
                args,
                None,
                int(len(feature_cols)),
                int(y_train_ret.shape[1]),
                target_name="direct_retention",
            ),
        )
        direct_loss_df.to_csv(out_direct_loss, index=False, encoding="utf-8-sig")
        if not direct_loss_df.empty:
            save_loss_plot(direct_loss_df, out_direct_loss_png)
        stream_direct_retention_predictions(
            dataset=train_dataset,
            pred_flat=direct_train_pred,
            model_name="direct_lstm_retention_baseline",
            metric_states=retention_metric_states,
            valid_csv_path=None,
            append_valid_csv=False,
            scatter_frames=retention_scatter_frames,
            scatter_counts=retention_scatter_counts,
        )
        ret_valid_written = stream_direct_retention_predictions(
            dataset=valid_dataset,
            pred_flat=direct_valid_pred,
            model_name="direct_lstm_retention_baseline",
            metric_states=retention_metric_states,
            valid_csv_path=out_ret_pred,
            append_valid_csv=ret_valid_written,
            scatter_frames=retention_scatter_frames,
            scatter_counts=retention_scatter_counts,
        )

    if needs_direct_baseline:
        if x_train_flat is None or x_valid_flat is None:
            raise RuntimeError("Direct interval baseline requires flattened interval arrays.")
        if y_train_ret is None:
            y_train_ret = retention_targets_from_dataset(train_dataset)
        if y_valid_ret is None:
            y_valid_ret = retention_targets_from_dataset(valid_dataset)
        direct_model = make_rf_model(args, seed=int(args.random_seed) + 3000)
        direct_model.fit(x_train_flat, sklearn_target_for_fit(y_train_ret))
        direct_train_pred = ensure_2d_prediction(direct_model.predict(x_train_flat), int(args.horizon))
        direct_valid_pred = ensure_2d_prediction(direct_model.predict(x_valid_flat), int(args.horizon))
        stream_direct_retention_predictions(
            dataset=train_dataset,
            pred_flat=direct_train_pred,
            model_name="direct_interval_baseline",
            metric_states=retention_metric_states,
            valid_csv_path=None,
            append_valid_csv=False,
            scatter_frames=retention_scatter_frames,
            scatter_counts=retention_scatter_counts,
        )
        ret_valid_written = stream_direct_retention_predictions(
            dataset=valid_dataset,
            pred_flat=direct_valid_pred,
            model_name="direct_interval_baseline",
            metric_states=retention_metric_states,
            valid_csv_path=out_ret_pred,
            append_valid_csv=ret_valid_written,
            scatter_frames=retention_scatter_frames,
            scatter_counts=retention_scatter_counts,
        )
    else:
        print("Skipping direct interval baseline (--skip-direct-baseline).", flush=True)

    dqdv_metrics_df = pd.DataFrame([asdict(row) for row in dqdv_metric_rows])
    retention_metrics_df = retention_metric_states_to_frame(retention_metric_states)
    retention_scatter_df = pd.concat(retention_scatter_frames, ignore_index=True) if retention_scatter_frames else pd.DataFrame()
    checks_df = build_dataset_checks(
        merged=merged,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        feature_cols=feature_cols,
        target_cols=target_cols,
        charge_stats=charge_stats,
        discharge_stats=discharge_stats,
    )

    checks_df.to_csv(out_checks, index=False, encoding="utf-8-sig")
    dqdv_metrics_df.to_csv(out_dqdv_metrics, index=False, encoding="utf-8-sig")
    dqdv_append = False
    for model_name, dataset, y_true_flat, y_pred_flat in dqdv_valid_outputs:
        write_dqdv_prediction_csv_stream(
            out_path=out_dqdv_pred,
            dataset=dataset,
            y_true_flat=y_true_flat,
            y_pred_flat=y_pred_flat,
            model_name=model_name,
            target_cols=target_cols,
            append=dqdv_append,
        )
        dqdv_append = True
    if not dqdv_append:
        pd.DataFrame().to_csv(out_dqdv_pred, index=False, encoding="utf-8-sig")
    retention_metrics_df.to_csv(out_ret_metrics, index=False, encoding="utf-8-sig")
    if not ret_valid_written:
        pd.DataFrame().to_csv(out_ret_pred, index=False, encoding="utf-8-sig")
    loss_df.to_csv(out_loss, index=False, encoding="utf-8-sig")
    if not loss_df.empty:
        save_loss_plot(loss_df, out_loss_png)
    save_retention_scatter(retention_scatter_df, retention_metrics_df, out_scatter_png)
    config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "target_columns": list(target_cols),
        "input_feature_columns": list(feature_cols),
        "input_transform_info": dict(input_transform_info),
        "charge_stats": dict(charge_stats),
        "discharge_stats": dict(discharge_stats),
        "device": str(device),
        "best_lstm_epoch": best_epoch,
    }
    out_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(
        args=args,
        device=device,
        target_cols=target_cols,
        feature_cols=feature_cols,
        dataset_checks=checks_df,
        dqdv_metrics=dqdv_metrics_df,
        retention_metrics=retention_metrics_df,
        best_epoch=best_epoch,
    )
    out_report.write_text(report, encoding="utf-8")
    print(f"Saved outputs to: {args.output_dir}", flush=True)
    best_valid = retention_metrics_df.loc[
        (retention_metrics_df["set_type"] == "valid")
        & (retention_metrics_df["target"] == "retention")
        & (retention_metrics_df["horizon_step"] == 1)
    ].sort_values("r2", ascending=False, kind="mergesort")
    if not best_valid.empty:
        row = best_valid.iloc[0]
        print(
            f"Best valid H1 retention model: {row['model_name']} | R2={float(row['r2']):.6f} | RMSE={float(row['rmse']):.6f}",
            flush=True,
        )
    return {
        "output_dir": str(args.output_dir),
        "dqdv_metrics_path": str(out_dqdv_metrics),
        "retention_metrics_path": str(out_ret_metrics),
        "report_path": str(out_report),
    }


def load_partial_rows(path: Path) -> Dict[int, Dict[str, object]]:
    """Load partial tuning rows keyed by trial_id."""

    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding=ENCODING)
    rows: Dict[int, Dict[str, object]] = {}
    for row in df.to_dict(orient="records"):
        rows[int(row["trial_id"])] = row
    return rows


def save_grid_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Save grid rows sorted by trial_id."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df = df.sort_values("trial_id", kind="mergesort").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def train_one_tuning_trial(
    args: argparse.Namespace,
    prepared: PreparedData,
    cfg: TrialConfig,
    device: torch.device,
    status_path: Path,
) -> Dict[str, object]:
    """Train and evaluate one resumable LSTM tuning trial."""

    trial_args = copy_args(args)
    trial_args.hidden_size = int(cfg.hidden_size)
    trial_args.learning_rate = float(cfg.learning_rate)
    trial_args.num_layers = int(cfg.num_layers)
    trial_args.dropout = float(cfg.dropout)
    trial_dir = args.output_dir / "checkpoints" / f"trial_{cfg.trial_id:03d}"
    signature = lstm_run_signature(
        trial_args,
        cfg,
        feature_dim=len(prepared.feature_cols),
        output_size=int(prepared.y_train_dqdv.shape[1]),
    )
    (
        _model,
        loss_df,
        best_epoch,
        _train_true,
        train_pred,
        _valid_true,
        valid_pred,
    ) = train_lstm_model(
        args=trial_args,
        train_dataset=prepared.train_dataset,
        valid_dataset=prepared.valid_dataset,
        y_train_flat=prepared.y_train_dqdv,
        device=device,
        output_size=int(prepared.y_train_dqdv.shape[1]),
        feature_dim=len(prepared.feature_cols),
        checkpoint_dir=trial_dir,
        resume_interrupted=bool(args.resume_interrupted),
        save_latest_state=bool(args.save_trial_latest_state),
        save_epoch_log=bool(args.save_trial_epoch_log),
        status_path=status_path,
        run_signature=signature,
    )
    dqdv_metrics, retention_metrics = evaluate_lstm_predictions(
        prepared=prepared,
        train_pred=train_pred,
        valid_pred=valid_pred,
        model_name="lstm_dqdv",
    )
    dqdv_metrics.to_csv(trial_dir / "train_valid_metrics_dqdv.csv", index=False, encoding="utf-8-sig")
    retention_metrics.to_csv(trial_dir / "retention_bridge_metrics.csv", index=False, encoding="utf-8-sig")
    loss_df.to_csv(trial_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")
    score = select_lstm_score(
        retention_metrics,
        dqdv_metrics,
        "lstm_dqdv",
        bridge_feature_mode=str(args.bridge_feature_mode),
    )
    best_valid_loss = float(loss_df["valid_loss"].min()) if not loss_df.empty else float("nan")
    row: Dict[str, object] = {
        "trial_id": int(cfg.trial_id),
        "hidden_size": int(cfg.hidden_size),
        "learning_rate": float(cfg.learning_rate),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "best_epoch": int(best_epoch),
        "best_valid_loss": best_valid_loss,
        "checkpoint_dir": str(trial_dir),
        "best_checkpoint_path": str(trial_dir / "best.pt"),
        **score,
    }
    (trial_dir / "trial_summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def ensure_grid_metric_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure grid result rows contain all displayed metric columns."""

    required_cols = [
        "retention_valid_r2",
        "retention_valid_mse",
        "retention_valid_rmse",
        "retention_valid_mae",
        "dqdv_valid_mean_r2",
        "dqdv_valid_mean_mse",
        "dqdv_valid_mean_rmse",
        "dqdv_valid_mean_mae",
    ]
    out = results_df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def finalize_grid_outputs(args: argparse.Namespace, rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Write final grid outputs and return the best configuration."""

    results_df = ensure_grid_metric_columns(pd.DataFrame(list(rows)))
    if results_df.empty:
        raise RuntimeError("No tuning rows to finalize.")
    results_df = results_df.sort_values(
        ["retention_valid_r2", "dqdv_valid_mean_r2", "dqdv_valid_mean_rmse", "trial_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = results_df.iloc[0].to_dict()
    out_results = args.output_dir / str(args.grid_results_file)
    out_best = args.output_dir / str(args.best_config_file)
    out_report = args.output_dir / "grid_tuning_report.md"
    results_df.to_csv(out_results, index=False, encoding="utf-8-sig")
    best_payload = {
        "selection_rule": (
            "max primary predicted dQdV bridge H1 valid retention R2 "
            "(dQdV-only when bridge_feature_mode is dqdv_only/both); "
            "tie by mean dQdV H1 valid R2 then mean RMSE"
        ),
        "input_feature_pack": str(args.input_feature_pack),
        "input_scaling": str(args.input_scaling),
        "target_pack": str(args.target_pack),
        "bridge_feature_mode": str(args.bridge_feature_mode),
        "lstm_head": str(args.lstm_head),
        "history_len": int(args.history_len),
        "horizon": int(args.horizon),
        "trial_id": int(best["trial_id"]),
        "hidden_size": int(best["hidden_size"]),
        "learning_rate": float(best["learning_rate"]),
        "num_layers": int(best["num_layers"]),
        "dropout": float(best["dropout"]),
        "best_epoch": int(best["best_epoch"]),
        "retention_valid_r2": float(best["retention_valid_r2"]),
        "retention_valid_mse": float(best["retention_valid_mse"]),
        "retention_valid_rmse": float(best["retention_valid_rmse"]),
        "retention_valid_mae": float(best["retention_valid_mae"]),
        "dqdv_valid_mean_r2": float(best["dqdv_valid_mean_r2"]),
        "dqdv_valid_mean_mse": float(best["dqdv_valid_mean_mse"]),
        "dqdv_valid_mean_rmse": float(best["dqdv_valid_mean_rmse"]),
        "dqdv_valid_mean_mae": float(best["dqdv_valid_mean_mae"]),
        "best_checkpoint_path": str(best["best_checkpoint_path"]),
    }
    out_best.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# interval -> dQdV -> retention LSTM tuning report",
        "",
        "## Summary",
        f"- Trials: **{len(results_df)}**",
        f"- Best trial: **{best_payload['trial_id']}**",
        f"- Best H1 valid retention R2: **{best_payload['retention_valid_r2']:.6f}**",
        f"- Best H1 valid retention RMSE: **{best_payload['retention_valid_rmse']:.6f}**",
        f"- Best H1 valid retention MSE/MAE: **{best_payload['retention_valid_mse']:.8f} / {best_payload['retention_valid_mae']:.6f}**",
        "",
        "## Best Config",
        "```json",
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Results",
        "| trial | hidden | lr | layers | dropout | best_epoch | retention R2 | retention MSE | retention RMSE | retention MAE | dQdV mean R2 | dQdV mean MSE | dQdV mean RMSE | dQdV mean MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results_df.itertuples(index=False):
        lines.append(
            f"| {int(row.trial_id)} | {int(row.hidden_size)} | {float(row.learning_rate):.6g} | "
            f"{int(row.num_layers)} | {float(row.dropout):.2f} | {int(row.best_epoch)} | "
            f"{float(row.retention_valid_r2):.6f} | {float(row.retention_valid_mse):.8f} | "
            f"{float(row.retention_valid_rmse):.6f} | {float(row.retention_valid_mae):.6f} | "
            f"{float(row.dqdv_valid_mean_r2):.6f} | {float(row.dqdv_valid_mean_mse):.8f} | "
            f"{float(row.dqdv_valid_mean_rmse):.6f} | {float(row.dqdv_valid_mean_mae):.6f} |"
        )
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved tuning results: {out_results}", flush=True)
    print(f"Saved best config: {out_best}", flush=True)
    return best_payload


def run_tuning(args: argparse.Namespace) -> Dict[str, object]:
    """Run resumable LSTM tuning and optionally full refresh."""

    args = apply_smoke_defaults(args)
    set_seed(int(args.random_seed))
    ensure_matplotlib_config()
    device = resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_args = copy_args(args)
    data_args.model_family = "lstm"
    data_args.max_train_windows = int(args.tune_max_train_windows)
    data_args.max_valid_windows = int(args.tune_max_valid_windows)
    if bool(args.smoke_test):
        data_args.max_train_windows = min(int(data_args.max_train_windows), 1024)
        data_args.max_valid_windows = min(int(data_args.max_valid_windows), 512)
    prepared = prepare_pipeline_data(data_args)
    partial_path = args.output_dir / str(args.partial_results_file)
    runtime_status_path = args.output_dir / str(args.runtime_status_file)
    partial_rows = load_partial_rows(partial_path) if bool(args.resume_from_partial) else {}
    trial_rows: List[Dict[str, object]] = []
    trials = prepare_trial_grid(args)
    for idx, cfg in enumerate(trials, start=1):
        if bool(args.resume_from_partial) and int(cfg.trial_id) in partial_rows:
            print(f"[Trial {cfg.trial_id}/{len(trials)}] skip from partial CSV", flush=True)
            row = partial_rows[int(cfg.trial_id)]
            trial_rows.append(row)
            print(f"[Trial {cfg.trial_id}] cached {format_trial_metric_summary(row)}", flush=True)
            continue
        trial_summary = args.output_dir / "checkpoints" / f"trial_{cfg.trial_id:03d}" / "trial_summary.json"
        if bool(args.resume_existing) and trial_summary.exists():
            print(f"[Trial {cfg.trial_id}/{len(trials)}] backfill from trial_summary.json", flush=True)
            row = json.loads(trial_summary.read_text(encoding="utf-8"))
            trial_rows.append(row)
            print(f"[Trial {cfg.trial_id}] cached {format_trial_metric_summary(row)}", flush=True)
            if bool(args.save_partial_results):
                save_grid_rows(trial_rows, partial_path)
            continue
        print(
            f"[Trial {cfg.trial_id}/{len(trials)}] train hidden={cfg.hidden_size} lr={cfg.learning_rate} "
            f"layers={cfg.num_layers} dropout={cfg.dropout}",
            flush=True,
        )
        runtime_status_path.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "stage": "trial",
                    "current_trial_id": int(cfg.trial_id),
                    "trial_index": int(idx),
                    "total_trials": int(len(trials)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        row = train_one_tuning_trial(args, prepared, cfg, device, runtime_status_path)
        trial_rows.append(row)
        if bool(args.save_partial_results):
            save_grid_rows(trial_rows, partial_path)
        print(f"[Trial {cfg.trial_id}] {format_trial_metric_summary(row)}", flush=True)
    best_payload = finalize_grid_outputs(args, trial_rows)
    runtime_status_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "stage": "completed",
                "total_trials": int(len(trial_rows)),
                "best_trial_id": int(best_payload["trial_id"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if str(args.run_mode) == "tune-and-full" or bool(args.run_full_refresh):
        return run_full_refresh(args)
    return {
        "output_dir": str(args.output_dir),
        "best_config_path": str(args.output_dir / str(args.best_config_file)),
    }


def resolve_best_config_path(args: argparse.Namespace) -> Path:
    """Resolve best-grid-config path for full refresh."""

    if args.best_config_path is not None:
        return args.best_config_path
    return args.output_dir / str(args.best_config_file)


def run_full_refresh(args: argparse.Namespace) -> Dict[str, object]:
    """Run full-size training from a best tuning config."""

    best_path = resolve_best_config_path(args)
    if not best_path.exists():
        raise FileNotFoundError(f"best config not found: {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    full_args = copy_args(args)
    full_args.run_mode = "single"
    full_args.model_family = str(args.model_family)
    full_args.input_feature_pack = str(best.get("input_feature_pack", full_args.input_feature_pack))
    full_args.input_scaling = str(best.get("input_scaling", full_args.input_scaling))
    full_args.target_pack = str(best.get("target_pack", full_args.target_pack))
    full_args.bridge_feature_mode = str(best.get("bridge_feature_mode", full_args.bridge_feature_mode))
    full_args.lstm_head = str(best.get("lstm_head", full_args.lstm_head))
    full_args.hidden_size = int(best["hidden_size"])
    full_args.learning_rate = float(best["learning_rate"])
    full_args.num_layers = int(best["num_layers"])
    full_args.dropout = float(best["dropout"])
    full_args.epochs = min(int(full_args.epochs), 3) if bool(args.smoke_test) else int(args.full_refresh_epochs)
    full_args.patience = min(int(full_args.patience), 2) if bool(args.smoke_test) else int(args.full_refresh_patience)
    full_args.batch_size = int(args.full_refresh_batch_size)
    full_args.max_train_windows = 0
    full_args.max_valid_windows = 0
    full_args.resume_interrupted = bool(args.full_refresh_resume_interrupted)
    explicit_full_refresh_output_dir = args.full_refresh_output_dir is not None
    if explicit_full_refresh_output_dir:
        full_args.output_dir = args.full_refresh_output_dir
    else:
        full_args.output_dir = args.output_dir / "full_refresh"
    if bool(args.smoke_test) and not explicit_full_refresh_output_dir:
        full_args.output_dir = Path(str(full_args.output_dir) + "_smoke") if not str(full_args.output_dir).endswith("_smoke") else full_args.output_dir
    print(f"Running full refresh from {best_path} -> {full_args.output_dir}", flush=True)
    return run_pipeline(full_args)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    if str(args.run_mode) == "single":
        run_pipeline(args)
    elif str(args.run_mode) == "tune":
        run_tuning(args)
    elif str(args.run_mode) == "full-refresh":
        run_full_refresh(args)
    elif str(args.run_mode) == "tune-and-full":
        args.run_full_refresh = True
        run_tuning(args)
    else:
        raise RuntimeError(f"Unknown run_mode={args.run_mode!r}.")


if __name__ == "__main__":
    main()
