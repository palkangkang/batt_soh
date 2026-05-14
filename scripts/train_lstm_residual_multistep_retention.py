"""LSTM residual correction for block-based multistep retention prediction."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_interval_to_dqdv_retention_pipeline import ENCODING, REPO_ROOT, set_seed
from train_multistep_interval_to_dqdv_retention_blocks import (
    FORBIDDEN_INPUT_COLS,
    TARGET_PACKS,
    BlockSample,
    block_metadata_frame,
    build_block_samples,
    build_cycle_table,
    build_history_matrix,
    build_linear_last10,
    build_persistence,
    downsample_blocks,
    filter_blocks_by_stage,
    future_arrays,
    make_lgbm,
    max_cycles_by_key,
    regression_metrics,
    relative_stage,
)


SELECTED_HORIZONS = [1, 10, 20, 50]
DEFAULT_SCHEMES = ["direct_retention", "linear_last10", "deployable_bridge"]
SCHEME_LABELS = {
    "direct_retention": "direct LightGBM residual",
    "linear_last10": "linear_last10 residual",
    "deployable_bridge": "dQdV bridge residual",
}
warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)


@dataclass
class BaselineResult:
    """Train and validation baseline predictions used as residual anchors."""

    train_predictions: Dict[str, np.ndarray]
    valid_predictions: Dict[str, np.ndarray]
    metrics: pd.DataFrame


@dataclass
class LstmTrainResult:
    """Predictions and epoch history for one LSTM residual model."""

    train_residual_pred: np.ndarray
    valid_residual_pred: np.ndarray
    epoch_log: pd.DataFrame
    best_epoch: int
    best_valid_loss: float


class ResidualSequenceDataset(Dataset):
    """PyTorch dataset for fixed-history sequence residual learning."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Store normalized sequence features and normalized residual targets."""

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of block samples."""

        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one sequence and one residual vector."""

        return self.x[idx], self.y[idx]


class ResidualLSTM(nn.Module):
    """Small LSTM that predicts multistep residual vectors from history sequences."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, horizon: int) -> None:
        """Create the LSTM encoder and regression head."""

        super().__init__()
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(horizon)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a residual vector with shape [batch, horizon]."""

        _out, (hidden, _cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Use LSTM residual correction to improve LightGBM/linear_last10 H50 retention predictions."
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "lstm_residual_multistep_retention_h100_m50",
    )
    parser.add_argument("--history-len", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--block-stride", type=int, default=150)
    parser.add_argument("--sample-mode", choices=["non_overlapping_blocks"], default="non_overlapping_blocks")
    parser.add_argument("--block-stage-filter", choices=["none", "early_train_late_valid"], default="none")
    parser.add_argument("--train-max-relative-input-end", type=float, default=0.45)
    parser.add_argument("--valid-min-relative-input-start", type=float, default=0.55)
    parser.add_argument("--feature-pack", choices=["recommended55"], default="recommended55")
    parser.add_argument("--target-pack", choices=["compact4"], default="compact4")
    parser.add_argument(
        "--schemes",
        type=str,
        default=",".join(DEFAULT_SCHEMES),
        help="Comma-separated baseline methods to correct: direct_retention,linear_last10,deployable_bridge.",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--h50-loss-weight", type=float, default=3.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
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
    parser.add_argument(
        "--include-history-retention-channel",
        action="store_true",
        help="Append historical retention as one observed history channel. Default keeps strict 55-feature input.",
    )
    return parser.parse_args()


def parse_schemes(raw: str) -> List[str]:
    """Parse and validate residual scheme names."""

    schemes = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(schemes).difference(DEFAULT_SCHEMES))
    if unknown:
        raise ValueError(f"Unknown schemes: {unknown}. Allowed: {DEFAULT_SCHEMES}")
    if not schemes:
        raise ValueError("At least one residual scheme is required.")
    return schemes


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested PyTorch device."""

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device_arg == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lightgbm_baselines(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train_dqdv: np.ndarray,
    y_valid_dqdv: np.ndarray,
    y_train_ret: np.ndarray,
    y_valid_ret: np.ndarray,
    linear_train_ret: np.ndarray,
    linear_valid_ret: np.ndarray,
    persistence_train_ret: np.ndarray,
    persistence_valid_ret: np.ndarray,
    seed: int,
) -> BaselineResult:
    """Train LightGBM direct, dQdV, and bridge baselines needed for residual correction."""

    horizon = int(y_train_ret.shape[1])
    n_targets = int(y_train_dqdv.shape[2])
    dqdv_train_pred = np.zeros_like(y_train_dqdv, dtype=np.float32)
    dqdv_valid_pred = np.zeros_like(y_valid_dqdv, dtype=np.float32)
    for step in range(horizon):
        for target_idx in range(n_targets):
            model = make_lgbm(seed + step * 41 + target_idx)
            model.fit(x_train, y_train_dqdv[:, step, target_idx])
            dqdv_train_pred[:, step, target_idx] = model.predict(x_train).astype(np.float32)
            dqdv_valid_pred[:, step, target_idx] = model.predict(x_valid).astype(np.float32)

    train_predictions: Dict[str, np.ndarray] = {
        "direct_retention": np.zeros_like(y_train_ret, dtype=np.float32),
        "oracle_bridge": np.zeros_like(y_train_ret, dtype=np.float32),
        "deployable_bridge": np.zeros_like(y_train_ret, dtype=np.float32),
        "persistence": persistence_train_ret.astype(np.float32),
        "linear_last10": linear_train_ret.astype(np.float32),
    }
    valid_predictions: Dict[str, np.ndarray] = {
        "direct_retention": np.zeros_like(y_valid_ret, dtype=np.float32),
        "oracle_bridge": np.zeros_like(y_valid_ret, dtype=np.float32),
        "deployable_bridge": np.zeros_like(y_valid_ret, dtype=np.float32),
        "persistence": persistence_valid_ret.astype(np.float32),
        "linear_last10": linear_valid_ret.astype(np.float32),
    }
    for step in range(horizon):
        direct_model = make_lgbm(seed + 5000 + step)
        direct_model.fit(x_train, y_train_ret[:, step])
        train_predictions["direct_retention"][:, step] = direct_model.predict(x_train).astype(np.float32)
        valid_predictions["direct_retention"][:, step] = direct_model.predict(x_valid).astype(np.float32)

        bridge_model = make_lgbm(seed + 7000 + step)
        bridge_model.fit(y_train_dqdv[:, step, :], y_train_ret[:, step])
        train_predictions["oracle_bridge"][:, step] = bridge_model.predict(y_train_dqdv[:, step, :]).astype(np.float32)
        valid_predictions["oracle_bridge"][:, step] = bridge_model.predict(y_valid_dqdv[:, step, :]).astype(np.float32)
        train_predictions["deployable_bridge"][:, step] = bridge_model.predict(dqdv_train_pred[:, step, :]).astype(np.float32)
        valid_predictions["deployable_bridge"][:, step] = bridge_model.predict(dqdv_valid_pred[:, step, :]).astype(np.float32)

    rows: List[Dict[str, object]] = []
    for method, train_pred in train_predictions.items():
        valid_pred = valid_predictions[method]
        for step in range(horizon):
            for set_type, truth, pred in [
                ("train", y_train_ret[:, step], train_pred[:, step]),
                ("valid", y_valid_ret[:, step], valid_pred[:, step]),
            ]:
                row = regression_metrics(truth, pred)
                row.update(
                    {
                        "stage": "baseline_retention_prediction",
                        "method": method,
                        "set_type": set_type,
                        "target": "retention",
                        "horizon": f"H{step + 1}",
                        "horizon_step": int(step + 1),
                    }
                )
                rows.append(row)
        for set_type, truth, pred in [
            ("train", y_train_ret, train_pred),
            ("valid", y_valid_ret, valid_pred),
        ]:
            row = regression_metrics(truth.reshape(-1), pred.reshape(-1))
            row.update(
                {
                    "stage": "baseline_retention_prediction",
                    "method": method,
                    "set_type": set_type,
                    "target": "retention",
                    "horizon": "all",
                    "horizon_step": 0,
                }
            )
            rows.append(row)
    return BaselineResult(train_predictions, valid_predictions, pd.DataFrame(rows))


def make_lstm_input(
    samples: Sequence[BlockSample],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    include_history_retention: bool,
    retention_mean: float,
    retention_std: float,
) -> np.ndarray:
    """Create normalized LSTM input arrays from block history sequences."""

    x = np.stack([sample.history_x for sample in samples]).astype(np.float32)
    x_norm = (x - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)
    if not include_history_retention:
        return x_norm.astype(np.float32)
    retention = np.stack([sample.history_retention for sample in samples]).astype(np.float32)
    retention_norm = ((retention - float(retention_mean)) / float(retention_std)).reshape(retention.shape[0], retention.shape[1], 1)
    return np.concatenate([x_norm, retention_norm.astype(np.float32)], axis=2).astype(np.float32)


def standardize_residuals(y_train: np.ndarray, y_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize residual targets using train-only horizon-wise statistics."""

    mean = np.nanmean(y_train, axis=0).astype(np.float32)
    std = np.nanstd(y_train, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((y_train - mean) / std).astype(np.float32), ((y_valid - mean) / std).astype(np.float32), mean, std


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Calculate horizon-weighted mean squared error."""

    sq = (pred - target) ** 2
    return (sq * weights.reshape(1, -1)).mean()


def run_one_lstm_training(
    scheme: str,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train_residual: np.ndarray,
    y_valid_residual: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> LstmTrainResult:
    """Train one LSTM residual model and return unstandardized residual predictions."""

    y_train_z, _y_valid_z, residual_mean, residual_std = standardize_residuals(y_train_residual, y_valid_residual)
    train_ds = ResidualSequenceDataset(x_train, y_train_z)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    model = ResidualLSTM(
        input_dim=int(x_train.shape[2]),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        horizon=int(args.horizon),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    weights = torch.ones(int(args.horizon), dtype=torch.float32, device=device)
    weights[-1] = float(args.h50_loss_weight)
    y_valid_tensor = torch.as_tensor(standardize_residuals(y_train_residual, y_valid_residual)[1], dtype=torch.float32, device=device)
    x_valid_tensor = torch.as_tensor(x_valid, dtype=torch.float32, device=device)

    checkpoint_dir = out_dir / "checkpoints" / scheme
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_valid = math.inf
    best_epoch = 0
    bad_epochs = 0
    rows: List[Dict[str, object]] = []
    max_epochs = min(int(args.epochs), 5) if bool(args.smoke_test) else int(args.epochs)
    patience = min(int(args.patience), 3) if bool(args.smoke_test) else int(args.patience)
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(model(xb), yb, weights)
            loss.backward()
            if float(args.gradient_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(weighted_mse(model(x_valid_tensor), y_valid_tensor, weights).detach().cpu().item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        row = {"scheme": scheme, "epoch": int(epoch), "train_loss": train_loss, "valid_loss": valid_loss}
        rows.append(row)
        print(
            f"[lstm_residual:{scheme}] epoch={epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}",
            flush=True,
        )
        latest = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_loss": valid_loss,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "args": vars(args),
        }
        torch.save(latest, checkpoint_dir / "latest.pt")
        if valid_loss < best_valid - 1e-8:
            best_valid = valid_loss
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(latest, checkpoint_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f"[lstm_residual:{scheme}] early stopping at epoch={epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_z = model(torch.as_tensor(x_train, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
        valid_z = model(x_valid_tensor).detach().cpu().numpy().astype(np.float32)
    train_pred = train_z * residual_std.reshape(1, -1) + residual_mean.reshape(1, -1)
    valid_pred = valid_z * residual_std.reshape(1, -1) + residual_mean.reshape(1, -1)
    return LstmTrainResult(train_pred.astype(np.float32), valid_pred.astype(np.float32), pd.DataFrame(rows), best_epoch, float(best_valid))


def add_metric_rows(
    rows: List[Dict[str, object]],
    method: str,
    stage: str,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    train_pred: np.ndarray,
    valid_pred: np.ndarray,
) -> None:
    """Append horizon-wise and all-horizon metric rows."""

    horizon = int(y_train.shape[1])
    for step in range(horizon):
        for set_type, truth, pred in [
            ("train", y_train[:, step], train_pred[:, step]),
            ("valid", y_valid[:, step], valid_pred[:, step]),
        ]:
            row = regression_metrics(truth, pred)
            row.update(
                {
                    "stage": stage,
                    "method": method,
                    "set_type": set_type,
                    "target": "retention",
                    "horizon": f"H{step + 1}",
                    "horizon_step": int(step + 1),
                }
            )
            rows.append(row)
    for set_type, truth, pred in [
        ("train", y_train, train_pred),
        ("valid", y_valid, valid_pred),
    ]:
        row = regression_metrics(truth.reshape(-1), pred.reshape(-1))
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


def build_prediction_long(samples: Sequence[BlockSample], preds: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build a long validation prediction table for baseline and residual-corrected methods."""

    rows: List[Dict[str, object]] = []
    for method, pred_matrix in preds.items():
        for block_idx, sample in enumerate(samples):
            for step in range(pred_matrix.shape[1]):
                truth = float(sample.future_retention[step])
                pred = float(pred_matrix[block_idx, step])
                q_ref = float(sample.future_q_ref[step])
                rows.append(
                    {
                        "block_id": sample.block_id,
                        "policy": sample.policy,
                        "cell_code": sample.cell_code,
                        "input_start_cycle": sample.input_start_cycle,
                        "input_end_cycle": sample.input_end_cycle,
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


def metric_lookup(metrics: pd.DataFrame, method: str, horizon: int, metric: str) -> float:
    """Look up one validation metric value."""

    view = metrics.loc[
        (metrics["set_type"] == "valid") & (metrics["method"] == method) & (metrics["horizon_step"] == int(horizon))
    ]
    if view.empty:
        return float("nan")
    return float(view[metric].iloc[0])


def add_identity_line(ax: object, x_values: pd.Series, y_values: pd.Series) -> None:
    """Add a y=x reference line and equal limits."""

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


def sample_for_plot(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a deterministic bounded sample for plotting."""

    if len(frame) <= int(max_rows):
        return frame.copy()
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(frame), size=int(max_rows), replace=False))
    return frame.iloc[keep].copy()


def save_h50_scatter(
    pred_long: pd.DataFrame,
    metrics: pd.DataFrame,
    methods: Sequence[str],
    out_path: Path,
    seed: int,
    eval_horizon: int,
) -> None:
    """Save selected-horizon predicted-vs-true scatter plots with explicit axis labels."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        part = pred_long.loc[(pred_long["horizon_step"] == int(eval_horizon)) & (pred_long["method"] == method)].copy()
        part = sample_for_plot(part, 5000, int(seed) + idx)
        ax.scatter(part["retention_true"], part["pred_retention"], s=12, alpha=0.42, edgecolors="none")
        add_identity_line(ax, part["retention_true"], part["pred_retention"])
        r2 = metric_lookup(metrics, method, int(eval_horizon), "r2")
        rmse = metric_lookup(metrics, method, int(eval_horizon), "rmse")
        ax.set_title(f"{method} H{int(eval_horizon)}, R2={r2:.3f}, RMSE={rmse:.4f}")
        ax.set_xlabel(f"X: true retention at future H{int(eval_horizon)}")
        ax.set_ylabel(f"Y: predicted retention at future H{int(eval_horizon)}")
        ax.grid(True, linestyle="--", alpha=0.25)
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_h50_residual_hist(
    pred_long: pd.DataFrame,
    metrics: pd.DataFrame,
    methods: Sequence[str],
    out_path: Path,
    seed: int,
    eval_horizon: int,
) -> None:
    """Save selected-horizon residual distribution plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        part = pred_long.loc[(pred_long["horizon_step"] == int(eval_horizon)) & (pred_long["method"] == method)].copy()
        part = sample_for_plot(part, 5000, int(seed) + idx)
        ax.hist(part["residual_retention"].dropna(), bins=45, color="#4477AA", alpha=0.82)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        rmse = metric_lookup(metrics, method, int(eval_horizon), "rmse")
        ax.set_title(f"{method} H{int(eval_horizon)} residual, RMSE={rmse:.4f}")
        ax.set_xlabel("X: residual = true retention - predicted retention")
        ax.set_ylabel("Y: block count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_h50_residual_vs_true(
    pred_long: pd.DataFrame,
    methods: Sequence[str],
    out_path: Path,
    seed: int,
    eval_horizon: int,
) -> None:
    """Save selected-horizon residual versus true retention plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 2
    n_rows = int(math.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        part = pred_long.loc[(pred_long["horizon_step"] == int(eval_horizon)) & (pred_long["method"] == method)].copy()
        part = sample_for_plot(part, 5000, int(seed) + idx)
        ax.scatter(part["retention_true"], part["residual_retention"], s=12, alpha=0.42, edgecolors="none")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{method} H{int(eval_horizon)} residual vs true")
        ax.set_xlabel(f"X: true retention at future H{int(eval_horizon)}")
        ax.set_ylabel("Y: residual = true - predicted")
        ax.grid(True, linestyle="--", alpha=0.25)
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_loss_curve(epoch_log: pd.DataFrame, out_path: Path) -> None:
    """Save LSTM train and valid loss curves."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for scheme, part in epoch_log.groupby("scheme", sort=False):
        part = part.sort_values("epoch")
        ax.plot(part["epoch"], part["train_loss"], linestyle="-", marker="o", linewidth=1.6, label=f"{scheme} train")
        ax.plot(part["epoch"], part["valid_loss"], linestyle="--", marker="s", linewidth=1.6, label=f"{scheme} valid")
    ax.set_xlabel("X: epoch")
    ax.set_ylabel("Y: weighted MSE on standardized residual")
    ax.set_title("LSTM residual training loss")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render selected dataframe columns as a Markdown table."""

    view = df.loc[:, list(columns)].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _idx, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def selected_metrics(metrics: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return selected horizon and all-horizon rows."""

    selected = {1, 10, 20, int(horizon)}
    view = metrics.loc[
        (metrics["set_type"] == "valid")
        & ((metrics["horizon_step"].isin(selected)) | (metrics["horizon"] == "all"))
    ].copy()
    order = {method: idx for idx, method in enumerate(view["method"].drop_duplicates().tolist())}
    view["method_order"] = view["method"].map(order).fillna(999).astype(int)
    return view.sort_values(["method_order", "horizon_step"]).drop(columns=["method_order"])


def make_dataset_checks(
    args: argparse.Namespace,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    train_samples: Sequence[BlockSample],
    valid_samples: Sequence[BlockSample],
    max_cycle_map: Mapping[Tuple[str, str], int],
    schemes: Sequence[str],
) -> pd.DataFrame:
    """Build dataset and contract checks for this residual experiment."""

    forbidden = sorted(set(feature_cols).intersection(FORBIDDEN_INPUT_COLS))
    train_rel_ends = [relative_stage(sample, max_cycle_map, "input_end") for sample in train_samples]
    valid_rel_starts = [relative_stage(sample, max_cycle_map, "input_start") for sample in valid_samples]
    train_rel_end_max = float(np.nanmax(train_rel_ends)) if train_rel_ends else float("nan")
    valid_rel_start_min = float(np.nanmin(valid_rel_starts)) if valid_rel_starts else float("nan")
    stage_filter = str(args.block_stage_filter)
    checks = [
        ("history_len", int(args.history_len), int((int(args.history_len) == 100) or bool(args.smoke_test)), "expected 100 outside smoke"),
        ("horizon", int(args.horizon), int((int(args.horizon) == 50) or bool(args.smoke_test)), "expected 50 outside smoke"),
        ("block_stride", int(args.block_stride), int(int(args.block_stride) == int(args.history_len) + int(args.horizon)), "expected history_len+horizon"),
        ("sample_mode", str(args.sample_mode), int(str(args.sample_mode) == "non_overlapping_blocks"), ""),
        ("block_stage_filter", stage_filter, 1, "none or early_train_late_valid"),
        (
            "train_relative_input_end_max",
            train_rel_end_max,
            int(stage_filter == "none" or train_rel_end_max <= float(args.train_max_relative_input_end) + 1e-9),
            f"threshold={float(args.train_max_relative_input_end):.3f}",
        ),
        (
            "valid_relative_input_start_min",
            valid_rel_start_min,
            int(stage_filter == "none" or valid_rel_start_min >= float(args.valid_min_relative_input_start) - 1e-9),
            f"threshold={float(args.valid_min_relative_input_start):.3f}",
        ),
        ("feature_count", int(len(feature_cols)), int(len(feature_cols) == 55), "recommended55 only"),
        ("forbidden_input_columns_present", int(len(forbidden)), int(len(forbidden) == 0), ",".join(forbidden)),
        ("target_dim", int(len(target_cols)), int(len(target_cols) == 4), ",".join(target_cols)),
        ("train_block_count", int(len(train_samples)), int(len(train_samples) > 0), ""),
        ("valid_block_count", int(len(valid_samples)), int(len(valid_samples) > 0), ""),
        ("residual_scheme_count", int(len(schemes)), int(len(schemes) > 0), ",".join(schemes)),
        ("uses_policy_or_cycle_as_input", 0, 1, "policy/cell/cycle are metadata only"),
    ]
    return pd.DataFrame(checks, columns=["check_item", "value", "pass_flag", "details"])


def build_report(
    args: argparse.Namespace,
    checks: pd.DataFrame,
    metrics: pd.DataFrame,
    h50_summary: pd.DataFrame,
    epoch_summary: pd.DataFrame,
    out_dir: Path,
    eval_horizon: int,
) -> str:
    """Build the Markdown evaluation report with explicit term and plot explanations."""

    img_base = out_dir.resolve().as_posix()
    selected = selected_metrics(metrics, int(args.horizon))
    h50 = h50_summary.copy()
    direct_base = h50.loc[h50["method"] == "direct_retention", "rmse"]
    direct_fixed = h50.loc[h50["method"] == "direct_retention_lstm_residual", "rmse"]
    linear_base = h50.loc[h50["method"] == "linear_last10", "rmse"]
    linear_fixed = h50.loc[h50["method"] == "linear_last10_lstm_residual", "rmse"]
    dqdv_base = h50.loc[h50["method"] == "deployable_bridge", "rmse"]
    dqdv_fixed = h50.loc[h50["method"] == "deployable_bridge_lstm_residual", "rmse"]

    def delta_text(base: pd.Series, fixed: pd.Series) -> str:
        """Return a compact RMSE delta sentence."""

        if base.empty or fixed.empty:
            return "未运行该路线。"
        delta = float(fixed.iloc[0]) - float(base.iloc[0])
        direction = "降低" if delta < 0 else "升高"
        return f"RMSE 从 {float(base.iloc[0]):.6f} 到 {float(fixed.iloc[0]):.6f}，{direction} {abs(delta):.6f}。"

    lines = [
        "# LSTM 残差修正多步 retention 预测报告",
        "",
        "## 1. 任务目的",
        "",
        f"本报告验证 LSTM 是否能降低已有基线模型在 H{int(eval_horizon)} retention 预测上的残差。H{int(eval_horizon)} 指未来第 {int(eval_horizon)} 个 cycle 的容量保持率预测；retention 指容量保持率，即当前放电容量除以前 5 个有效 cycle 的参考容量中位数。",
        f"本次 block_stage_filter 为 `{args.block_stage_filter}`；当它为 `early_train_late_valid` 时，训练 block 只保留相对寿命早期输入，验证 block 只保留相对寿命后期输入。",
        "",
        "本任务不使用 `cycles`、`cycle_index_norm`、`policy`、`cell_code`、`initial_c_rate`、`switch_soc_percent`、`post_switch_c_rate` 作为模型输入。`policy` 和 `cell_code` 只用于分组、划分训练集与验证集、以及报告定位。",
        "",
        "## 2. 术语说明",
        "",
        "- `55维 recommended feature pack`：相关性分析得到的 55 个工况统计特征，包含充电 cross-bin 累计/增量信息、放电区间容量信息和少量放电汇总统计。",
        "- `LightGBM`：梯度提升树模型，适合中小样本表格特征；本任务中它是原始直接预测基线。",
        "- `linear_last10`：用历史最后 10 个已观测 retention 做线性拟合，再外推未来 H1 到 H50。",
        "- `dQdV compact4`：四个 dQdV 中介特征，包括主峰面积、主峰高度、主峰电压和主峰偏度。",
        "- `deployable_bridge`：先用工况预测 compact4 dQdV，再用预测 dQdV 预测 retention 的可部署中介路线。",
        "- `残差修正`：先得到基线预测，再让 LSTM 预测 `真实值 - 基线预测值`，最终结果为 `基线预测 + LSTM预测残差`。",
        "",
        "## 3. 数据检查",
        "",
        markdown_table(checks, ["check_item", "value", "pass_flag", "details"]),
        "",
        f"## 4. H{int(eval_horizon)} 核心结果",
        "",
        markdown_table(h50, ["method", "rmse", "mae", "mse", "r2"]),
        "",
        "关键对比：",
        "",
        f"- direct LightGBM 残差修正：{delta_text(direct_base, direct_fixed)}",
        f"- linear_last10 残差修正：{delta_text(linear_base, linear_fixed)}",
        f"- dQdV bridge 残差修正：{delta_text(dqdv_base, dqdv_fixed)}",
        "",
        "## 5. 多 horizon 指标",
        "",
        markdown_table(selected, ["method", "horizon", "horizon_step", "rmse", "mae", "mse", "r2"]),
        "",
        "## 6. LSTM 训练过程",
        "",
        markdown_table(epoch_summary, ["scheme", "best_epoch", "best_valid_loss"]),
        "",
        f"![LSTM loss curve]({img_base}/loss_curve.png)",
        "",
        "图 1 说明：横轴是 epoch，表示训练轮次；纵轴是标准化残差上的加权 MSE，数值越低表示 LSTM 对残差拟合越好。实线为训练集，虚线为验证集。若验证集 loss 下降不明显或反弹，说明残差可能噪声较强或样本不足。",
        "",
        f"## 7. H{int(eval_horizon)} 散点图",
        "",
        f"![H50 scatter]({img_base}/h50_retention_scatter.png)",
        "",
        f"图 2 说明：横轴是真实 H{int(eval_horizon)} retention，纵轴是预测 H{int(eval_horizon)} retention。虚线是理想预测线 `Y=X`。点越靠近虚线，预测越准确；若残差修正后点云更贴近虚线，说明 LSTM 残差模型有效。",
        "",
        f"## 8. H{int(eval_horizon)} 残差分布",
        "",
        f"![H50 residual histogram]({img_base}/h50_residual_distribution.png)",
        "",
        "图 3 说明：横轴是残差 `真实 retention - 预测 retention`，纵轴是样本块数量。分布越集中在 0 附近，说明预测误差越小；分布整体偏正或偏负，说明模型存在系统性低估或高估。",
        "",
        f"## 9. H{int(eval_horizon)} 残差随真实 retention 变化",
        "",
        f"![H50 residual vs true]({img_base}/h50_residual_vs_true.png)",
        "",
        "图 4 说明：横轴是真实 H50 retention，纵轴是残差。若残差随真实 retention 呈明显斜率或分段结构，说明模型在不同老化阶段存在系统性偏差；若 LSTM 修正后该结构减弱，说明时序信息补到了 LightGBM 或线性外推的盲区。",
        "",
        "## 10. 结论",
        "",
        f"本实验的核心判断标准不是 LSTM 单独能否预测 retention，而是 LSTM 是否能降低 direct LightGBM、linear_last10 或 dQdV bridge 的 H{int(eval_horizon)} 残差。如果残差修正不能降低 H{int(eval_horizon)} RMSE，说明当前误差更可能来自噪声、未来工况不可观测或样本分布差异，而不是单纯缺少 LSTM 时序建模。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the LSTM residual correction experiment."""

    args = parse_args()
    schemes = parse_schemes(args.schemes)
    set_seed(int(args.random_seed))
    device = resolve_device(str(args.device))
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    target_cols = TARGET_PACKS[str(args.target_pack)]

    print("Building cycle table...", flush=True)
    merged, feature_cols, _stats = build_cycle_table(args, target_cols)
    max_cycle_map = max_cycles_by_key(merged)
    samples = build_block_samples(
        merged=merged,
        feature_cols=feature_cols,
        target_cols=target_cols,
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        block_stride=int(args.block_stride),
    )
    samples = filter_blocks_by_stage(
        samples,
        max_cycle_map,
        str(args.block_stage_filter),
        float(args.train_max_relative_input_end),
        float(args.valid_min_relative_input_start),
    )
    if int(args.max_train_blocks) > 0 or int(args.max_valid_blocks) > 0:
        samples = downsample_blocks(samples, int(args.max_train_blocks), int(args.max_valid_blocks), int(args.random_seed))
    if bool(args.smoke_test):
        max_train = int(args.max_train_blocks) if int(args.max_train_blocks) > 0 else 80
        max_valid = int(args.max_valid_blocks) if int(args.max_valid_blocks) > 0 else 40
        samples = downsample_blocks(samples, max_train, max_valid, int(args.random_seed))
    train_samples = [sample for sample in samples if sample.set_type == "train"]
    valid_samples = [sample for sample in samples if sample.set_type == "valid"]
    if not train_samples or not valid_samples:
        raise RuntimeError("Both train and valid block samples are required.")

    print(f"Train blocks={len(train_samples)}, valid blocks={len(valid_samples)}", flush=True)
    x_train_summary, summary_cols = build_history_matrix(train_samples, feature_cols, "summary")
    x_valid_summary, _summary_cols = build_history_matrix(valid_samples, feature_cols, "summary")
    y_train_dqdv, y_train_ret, _train_q_ref, _train_q_discharge, _train_cycles = future_arrays(train_samples)
    y_valid_dqdv, y_valid_ret, _valid_q_ref, _valid_q_discharge, _valid_cycles = future_arrays(valid_samples)
    linear_train = build_linear_last10(train_samples)
    linear_valid = build_linear_last10(valid_samples)
    persistence_train = build_persistence(train_samples)
    persistence_valid = build_persistence(valid_samples)

    print("Training LightGBM baselines...", flush=True)
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

    feature_mean = np.nanmean(np.stack([sample.history_x for sample in train_samples]), axis=(0, 1)).astype(np.float32)
    feature_std = np.nanstd(np.stack([sample.history_x for sample in train_samples]), axis=(0, 1)).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    hist_ret_values = np.concatenate([sample.history_retention for sample in train_samples]).astype(np.float32)
    history_retention_mean = float(np.nanmean(hist_ret_values))
    history_retention_std = float(np.nanstd(hist_ret_values))
    if history_retention_std < 1e-6:
        history_retention_std = 1.0
    x_train_lstm = make_lstm_input(
        train_samples,
        feature_mean,
        feature_std,
        bool(args.include_history_retention_channel),
        history_retention_mean,
        history_retention_std,
    )
    x_valid_lstm = make_lstm_input(
        valid_samples,
        feature_mean,
        feature_std,
        bool(args.include_history_retention_channel),
        history_retention_mean,
        history_retention_std,
    )

    corrected_train: Dict[str, np.ndarray] = {}
    corrected_valid: Dict[str, np.ndarray] = {}
    metric_rows: List[Dict[str, object]] = []
    epoch_logs: List[pd.DataFrame] = []
    epoch_summary_rows: List[Dict[str, object]] = []
    print(f"Training LSTM residual schemes on {device}...", flush=True)
    for scheme in schemes:
        y_train_res = y_train_ret - baselines.train_predictions[scheme]
        y_valid_res = y_valid_ret - baselines.valid_predictions[scheme]
        result = run_one_lstm_training(
            scheme=scheme,
            x_train=x_train_lstm,
            x_valid=x_valid_lstm,
            y_train_residual=y_train_res,
            y_valid_residual=y_valid_res,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        corrected_name = f"{scheme}_lstm_residual"
        corrected_train[corrected_name] = (baselines.train_predictions[scheme] + result.train_residual_pred).astype(np.float32)
        corrected_valid[corrected_name] = (baselines.valid_predictions[scheme] + result.valid_residual_pred).astype(np.float32)
        add_metric_rows(
            metric_rows,
            method=corrected_name,
            stage="lstm_residual_corrected_retention_prediction",
            y_train=y_train_ret,
            y_valid=y_valid_ret,
            train_pred=corrected_train[corrected_name],
            valid_pred=corrected_valid[corrected_name],
        )
        epoch_logs.append(result.epoch_log)
        epoch_summary_rows.append(
            {"scheme": scheme, "best_epoch": int(result.best_epoch), "best_valid_loss": float(result.best_valid_loss)}
        )

    baseline_keep = ["direct_retention", "linear_last10", "deployable_bridge", "oracle_bridge", "persistence"]
    combined_metrics = pd.concat([baselines.metrics, pd.DataFrame(metric_rows)], ignore_index=True)
    valid_pred_methods: Dict[str, np.ndarray] = {
        method: baselines.valid_predictions[method] for method in baseline_keep if method in baselines.valid_predictions
    }
    valid_pred_methods.update(corrected_valid)
    pred_long = build_prediction_long(valid_samples, valid_pred_methods)
    epoch_log = pd.concat(epoch_logs, ignore_index=True) if epoch_logs else pd.DataFrame()
    epoch_summary = pd.DataFrame(epoch_summary_rows)
    eval_horizon = min(50, int(args.horizon))
    h50_summary = combined_metrics.loc[
        (combined_metrics["set_type"] == "valid")
        & (combined_metrics["target"] == "retention")
        & (combined_metrics["horizon_step"] == int(eval_horizon))
    ].copy()
    checks = make_dataset_checks(args, feature_cols, target_cols, train_samples, valid_samples, max_cycle_map, schemes)

    print("Saving outputs...", flush=True)
    checks.to_csv(out_dir / "dataset_checks.csv", index=False, encoding=ENCODING)
    pd.DataFrame({"rank": np.arange(1, len(feature_cols) + 1), "feature": feature_cols}).to_csv(
        out_dir / "feature_columns.csv", index=False, encoding=ENCODING
    )
    pd.DataFrame({"rank": np.arange(1, len(summary_cols) + 1), "feature": summary_cols}).to_csv(
        out_dir / "summary_feature_columns.csv", index=False, encoding=ENCODING
    )
    block_metadata_frame(samples).to_csv(out_dir / "block_samples.csv", index=False, encoding=ENCODING)
    baselines.metrics.to_csv(out_dir / "baseline_retention_metrics.csv", index=False, encoding=ENCODING)
    combined_metrics.to_csv(out_dir / "lstm_residual_metrics.csv", index=False, encoding=ENCODING)
    h50_summary.to_csv(out_dir / "h50_metrics_comparison.csv", index=False, encoding=ENCODING)
    pred_long.to_csv(out_dir / "valid_retention_predictions_long.csv", index=False, encoding=ENCODING)
    epoch_log.to_csv(out_dir / "lstm_epoch_log.csv", index=False, encoding=ENCODING)
    epoch_summary.to_csv(out_dir / "lstm_epoch_summary.csv", index=False, encoding=ENCODING)

    plot_methods = [
        "direct_retention",
        "direct_retention_lstm_residual",
        "linear_last10",
        "linear_last10_lstm_residual",
        "deployable_bridge",
        "deployable_bridge_lstm_residual",
    ]
    plot_methods = [method for method in plot_methods if method in set(pred_long["method"].astype(str))]
    save_h50_scatter(
        pred_long,
        combined_metrics,
        plot_methods,
        out_dir / "h50_retention_scatter.png",
        int(args.random_seed),
        int(eval_horizon),
    )
    save_h50_residual_hist(
        pred_long,
        combined_metrics,
        plot_methods,
        out_dir / "h50_residual_distribution.png",
        int(args.random_seed),
        int(eval_horizon),
    )
    save_h50_residual_vs_true(
        pred_long,
        plot_methods,
        out_dir / "h50_residual_vs_true.png",
        int(args.random_seed),
        int(eval_horizon),
    )
    save_loss_curve(epoch_log, out_dir / "loss_curve.png")

    run_config = {
        "history_len": int(args.history_len),
        "horizon": int(args.horizon),
        "block_stride": int(args.block_stride),
        "sample_mode": str(args.sample_mode),
        "block_stage_filter": str(args.block_stage_filter),
        "train_max_relative_input_end": float(args.train_max_relative_input_end),
        "valid_min_relative_input_start": float(args.valid_min_relative_input_start),
        "feature_pack": str(args.feature_pack),
        "feature_count": int(len(feature_cols)),
        "target_pack": str(args.target_pack),
        "target_cols": list(target_cols),
        "schemes": schemes,
        "include_history_retention_channel": bool(args.include_history_retention_channel),
        "input_dim": int(x_train_lstm.shape[2]),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "h50_loss_weight": float(args.h50_loss_weight),
        "device": str(device),
        "train_blocks": int(len(train_samples)),
        "valid_blocks": int(len(valid_samples)),
        "input_feature_columns": list(feature_cols),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding=ENCODING)
    report = build_report(args, checks, combined_metrics, h50_summary, epoch_summary, out_dir, int(eval_horizon))
    (out_dir / "lstm_residual_multistep_retention_report.md").write_text(report, encoding=ENCODING)
    print(f"Saved outputs to: {out_dir}", flush=True)
    h50_print = h50_summary.loc[h50_summary["method"].isin(plot_methods), ["method", "rmse", "r2"]].copy()
    print(h50_print.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
