from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
sys.path.append(str(REPO_ROOT))

import scripts.train_lstm_charge_delta_ah as lstm_mod


@dataclass(frozen=True)
class TrialConfig:
    """One grid-search trial configuration."""

    trial_id: int
    sequence_mode: str
    window_size: Optional[int]
    hidden_size: int
    learning_rate: float
    num_layers: int
    dropout: float


def parse_args() -> argparse.Namespace:
    """Parse CLI args for grid tuning."""

    parser = argparse.ArgumentParser(description="Grid tuning for LSTM charge delta_ah model.")
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
        default=REPO_ROOT / "outputs" / "analysis" / "lstm_charge_delta_ah_prefix_full_grid_cpu",
    )
    parser.add_argument(
        "--sequence-mode",
        type=str,
        choices=["fixed_window", "prefix_full"],
        default="prefix_full",
        help="fixed_window: search includes window_size; prefix_full: full-history prefix sequence.",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="20,30,60",
        help="Only used when sequence_mode=fixed_window.",
    )
    parser.add_argument("--hidden-sizes", type=str, default="64,128,192")
    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4,2e-4")
    parser.add_argument("--num-layers-list", type=str, default="1,2")
    parser.add_argument("--dropout-list", type=str, default="0.1,0.2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument(
        "--resume-existing",
        dest="resume_existing",
        action="store_true",
        default=True,
        help="Resume by evaluating existing checkpoints (default: enabled).",
    )
    parser.add_argument(
        "--no-resume-existing",
        dest="resume_existing",
        action="store_false",
        help="Disable resume from existing checkpoints.",
    )
    parser.add_argument(
        "--resume-interrupted",
        dest="resume_interrupted",
        action="store_true",
        default=True,
        help="Resume interrupted trial from trial_latest state file.",
    )
    parser.add_argument(
        "--no-resume-interrupted",
        dest="resume_interrupted",
        action="store_false",
        help="Disable interrupted-trial resume and restart trial from epoch 1.",
    )
    parser.add_argument(
        "--resume-from-partial",
        dest="resume_from_partial",
        action="store_true",
        default=True,
        help="Skip trials already recorded in partial csv (default: enabled).",
    )
    parser.add_argument(
        "--no-resume-from-partial",
        dest="resume_from_partial",
        action="store_false",
        help="Ignore partial csv and rerun all trials unless checkpoint resume matches.",
    )
    parser.add_argument(
        "--save-partial-results",
        dest="save_partial_results",
        action="store_true",
        default=True,
        help="Write grid_search_results.partial.csv after each trial (default: enabled).",
    )
    parser.add_argument(
        "--no-save-partial-results",
        dest="save_partial_results",
        action="store_false",
        help="Disable partial csv writing.",
    )
    parser.add_argument(
        "--save-trial-latest-state",
        dest="save_trial_latest_state",
        action="store_true",
        default=True,
        help="Save per-trial latest epoch state for interrupted resume (default: enabled).",
    )
    parser.add_argument(
        "--no-save-trial-latest-state",
        dest="save_trial_latest_state",
        action="store_false",
        help="Disable per-trial latest state writing and interrupted-trial resume source.",
    )
    parser.add_argument(
        "--save-trial-epoch-log",
        dest="save_trial_epoch_log",
        action="store_true",
        default=True,
        help="Save per-trial epoch csv log (default: enabled).",
    )
    parser.add_argument(
        "--no-save-trial-epoch-log",
        dest="save_trial_epoch_log",
        action="store_false",
        help="Disable per-trial epoch csv logging.",
    )
    parser.add_argument(
        "--save-runtime-status",
        dest="save_runtime_status",
        action="store_true",
        default=True,
        help="Save grid runtime status json every epoch (default: enabled).",
    )
    parser.add_argument(
        "--no-save-runtime-status",
        dest="save_runtime_status",
        action="store_false",
        help="Disable runtime status json writing.",
    )
    parser.add_argument(
        "--partial-results-file",
        type=str,
        default="grid_search_results.partial.csv",
        help="Partial grid result file name under output-dir.",
    )
    parser.add_argument(
        "--runtime-status-file",
        type=str,
        default="grid_tuning_runtime_status.json",
        help="Runtime status json file name under output-dir/checkpoints.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny smoke tuning for quick validation.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    """Parse comma-separated int list."""

    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    """Parse comma-separated float list."""

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_sequence_maps(
    args: argparse.Namespace,
) -> Tuple[Dict[Tuple[str, str], lstm_mod.SequenceData], Dict[Tuple[str, str], lstm_mod.SequenceData]]:
    """Load data once and return train/valid sequence maps."""

    split_df = lstm_mod.load_split_map(args.train_split_path, args.valid_split_path)
    feature_df, _ = lstm_mod.build_cycle_feature_table(args.charge_path)
    value_cols, mask_cols = lstm_mod.get_value_mask_cols(feature_df)
    label_df = lstm_mod.load_life_labels(args.life_path, q_min=args.q_min, q_max=args.q_max)
    merged = lstm_mod.merge_dataset(feature_df=feature_df, label_df=label_df, split_df=split_df)
    seq_map = lstm_mod.build_sequences(merged=merged, value_cols=value_cols, mask_cols=mask_cols)
    return lstm_mod.split_sequence_dict(seq_map)


def _prepare_trial_grid(args: argparse.Namespace) -> List[TrialConfig]:
    """Build trial search grid according to sequence mode."""

    hidden_sizes = parse_int_list(args.hidden_sizes)
    lrs = parse_float_list(args.learning_rates)
    num_layers_list = parse_int_list(args.num_layers_list)
    dropout_list = parse_float_list(args.dropout_list)
    windows: List[Optional[int]]
    if args.sequence_mode == "fixed_window":
        windows = [int(x) for x in parse_int_list(args.window_sizes)]
    else:
        windows = [None]

    trials: List[TrialConfig] = []
    trial_id = 0
    for window_size in windows:
        for hidden_size in hidden_sizes:
            for lr in lrs:
                for num_layers in num_layers_list:
                    for dropout in dropout_list:
                        trial_id += 1
                        trials.append(
                            TrialConfig(
                                trial_id=trial_id,
                                sequence_mode=args.sequence_mode,
                                window_size=window_size,
                                hidden_size=hidden_size,
                                learning_rate=lr,
                                num_layers=num_layers,
                                dropout=dropout,
                            )
                        )
    return trials


def _build_datasets(
    cfg: TrialConfig,
    train_seq_map: Dict[Tuple[str, str], lstm_mod.SequenceData],
    valid_seq_map: Dict[Tuple[str, str], lstm_mod.SequenceData],
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build train/valid datasets for one trial."""

    if cfg.sequence_mode == "fixed_window":
        if cfg.window_size is None:
            raise ValueError("window_size must be provided when sequence_mode='fixed_window'.")
        train_dataset: torch.utils.data.Dataset = lstm_mod.SlidingWindowDataset(
            sequences=train_seq_map,
            window_size=int(cfg.window_size),
            max_windows=args.max_train_windows if args.max_train_windows > 0 else None,
            seed=args.seed + cfg.trial_id,
        )
        valid_dataset: torch.utils.data.Dataset = lstm_mod.SlidingWindowDataset(
            sequences=valid_seq_map,
            window_size=int(cfg.window_size),
            max_windows=args.max_valid_windows if args.max_valid_windows > 0 else None,
            seed=args.seed + cfg.trial_id + 1,
        )
        return train_dataset, valid_dataset

    train_dataset = lstm_mod.PrefixHistoryDataset(
        sequences=train_seq_map,
        max_windows=args.max_train_windows if args.max_train_windows > 0 else None,
        seed=args.seed + cfg.trial_id,
    )
    valid_dataset = lstm_mod.PrefixHistoryDataset(
        sequences=valid_seq_map,
        max_windows=args.max_valid_windows if args.max_valid_windows > 0 else None,
        seed=args.seed + cfg.trial_id + 1,
    )
    return train_dataset, valid_dataset


def _build_loaders(
    cfg: TrialConfig,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and valid data loaders."""

    collate_fn = lstm_mod.collate_sequence_batch
    train_loader = lstm_mod.build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=bool(cfg.sequence_mode == "fixed_window"),
        device=device,
        collate_fn=collate_fn,
    )
    valid_loader = lstm_mod.build_dataloader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def _make_model(cfg: TrialConfig, device: torch.device) -> nn.Module:
    """Create model by trial config."""

    return lstm_mod.LSTMRegressor(
        input_size=24,
        hidden_size=int(cfg.hidden_size),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
    ).to(device)


def _atomic_torch_save(obj: Mapping[str, Any], path: Path) -> None:
    """Save torch object atomically."""

    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(dict(obj), tmp_path)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text file atomically."""

    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _append_trial_epoch_log(path: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch record for a trial."""

    row_df = pd.DataFrame([dict(row)])
    if not path.exists():
        row_df.to_csv(path, index=False, encoding="utf-8")
    else:
        row_df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")


def _build_trial_signature_payload(cfg: TrialConfig, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Build resume signature payload for one trial."""

    return {
        "trial_id": int(cfg.trial_id),
        "sequence_mode": str(cfg.sequence_mode),
        "window_size": None if cfg.window_size is None else int(cfg.window_size),
        "hidden_size": int(cfg.hidden_size),
        "learning_rate": float(cfg.learning_rate),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_train_windows": int(args.max_train_windows),
        "max_valid_windows": int(args.max_valid_windows),
        "seed": int(args.seed),
    }


def _build_trial_signature(payload: Mapping[str, Any]) -> str:
    """Hash payload into deterministic signature."""

    canonical = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_trial_paths(ckpt_dir: Path, trial_id: int) -> Tuple[Path, Path, Path]:
    """Return (best_ckpt, latest_state, epoch_log) paths for one trial."""

    best_ckpt = ckpt_dir / f"trial_{trial_id:03d}.pt"
    latest_state = ckpt_dir / f"trial_{trial_id:03d}_latest.pt"
    epoch_log = ckpt_dir / f"trial_{trial_id:03d}_epoch_log.csv"
    return best_ckpt, latest_state, epoch_log


def _build_result_row(
    cfg: TrialConfig,
    ckpt_path: Path,
    best_epoch: int,
    best_valid_loss: float,
    valid_metrics: lstm_mod.Metrics,
    train_samples: int,
    valid_samples: int,
) -> Dict[str, float | int | str]:
    """Build one result row for CSV."""

    window_size_out = np.nan if cfg.window_size is None else int(cfg.window_size)
    return {
        "trial_id": int(cfg.trial_id),
        "sequence_mode": str(cfg.sequence_mode),
        "window_size": window_size_out,
        "hidden_size": int(cfg.hidden_size),
        "learning_rate": float(cfg.learning_rate),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "valid_mse": float(valid_metrics.mse),
        "valid_rmse": float(valid_metrics.rmse),
        "valid_mae": float(valid_metrics.mae),
        "valid_r2": float(valid_metrics.r2),
        "train_samples": int(train_samples),
        "valid_samples": int(valid_samples),
        "checkpoint_path": str(ckpt_path),
    }


def train_one_trial(
    cfg: TrialConfig,
    train_seq_map: Dict[Tuple[str, str], lstm_mod.SequenceData],
    valid_seq_map: Dict[Tuple[str, str], lstm_mod.SequenceData],
    args: argparse.Namespace,
    device: torch.device,
    ckpt_dir: Path,
) -> Dict[str, float | int | str]:
    """Run one trial and return metrics row."""

    train_dataset, valid_dataset = _build_datasets(
        cfg=cfg,
        train_seq_map=train_seq_map,
        valid_seq_map=valid_seq_map,
        args=args,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError(
            f"Trial {cfg.trial_id}: empty dataset for sequence_mode={cfg.sequence_mode}, "
            f"window_size={cfg.window_size}"
        )

    train_loader, valid_loader = _build_loaders(
        cfg=cfg,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args,
        device=device,
    )
    model = _make_model(cfg=cfg, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    signature_payload = _build_trial_signature_payload(cfg=cfg, args=args, device=device)
    args_signature = _build_trial_signature(payload=signature_payload)
    ckpt_path, latest_state_path, epoch_log_path = _build_trial_paths(ckpt_dir=ckpt_dir, trial_id=cfg.trial_id)
    runtime_status_path = ckpt_dir / str(args.runtime_status_file)
    resume_allowed = bool(args.resume_interrupted and args.save_trial_latest_state)

    best_valid_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    start_epoch = 1

    if bool(args.resume_interrupted) and not bool(args.save_trial_latest_state):
        print(
            f"Trial {cfg.trial_id}: save_trial_latest_state disabled, resume_interrupted ignored.",
            flush=True,
        )

    if resume_allowed and latest_state_path.exists():
        latest_state = torch.load(latest_state_path, map_location=device)
        saved_sig = str(latest_state.get("args_signature", ""))
        if saved_sig != args_signature:
            raise RuntimeError(
                f"Trial {cfg.trial_id}: latest state signature mismatch. "
                "Use --no-resume-interrupted or a fresh output directory."
            )
        model.load_state_dict(latest_state["model_state_dict"])
        optimizer.load_state_dict(latest_state["optimizer_state_dict"])
        start_epoch = int(latest_state["epoch"]) + 1
        best_epoch = int(latest_state.get("best_epoch", 0))
        best_valid_loss = float(latest_state.get("best_valid_loss", float("inf")))
        no_improve = int(latest_state.get("no_improve", 0))
        print(
            f"Trial {cfg.trial_id}: resumed from epoch {start_epoch - 1}, "
            f"best_epoch={best_epoch}, best_valid={best_valid_loss:.8f}",
            flush=True,
        )
    elif not bool(args.resume_interrupted):
        print(f"Trial {cfg.trial_id}: resume_interrupted disabled, start from epoch 1.", flush=True)

    if start_epoch <= 1 and bool(args.save_trial_epoch_log):
        pd.DataFrame(
            columns=[
                "timestamp",
                "trial_id",
                "epoch",
                "train_loss",
                "valid_loss",
                "is_best_epoch",
                "best_valid_loss",
                "no_improve",
            ]
        ).to_csv(epoch_log_path, index=False, encoding="utf-8")
        best_valid_loss = float("inf")
        best_epoch = 0
        no_improve = 0

    if start_epoch > int(args.epochs):
        print(
            f"Trial {cfg.trial_id}: latest epoch already >= target epochs ({args.epochs}), skip extra epochs.",
            flush=True,
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        _ = lstm_mod.train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = lstm_mod.eval_loss(model, valid_loader, criterion, device)
        improved = (best_valid_loss - valid_loss) > float(args.min_delta)
        if improved:
            best_valid_loss = float(valid_loss)
            best_epoch = int(epoch)
            no_improve = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "sequence_mode": cfg.sequence_mode,
                    "window_size": cfg.window_size,
                    "hidden_size": cfg.hidden_size,
                    "learning_rate": cfg.learning_rate,
                    "num_layers": cfg.num_layers,
                    "dropout": cfg.dropout,
                    "best_epoch": best_epoch,
                    "best_valid_loss": best_valid_loss,
                },
                path=ckpt_path,
            )
        else:
            no_improve += 1

        if bool(args.save_trial_epoch_log):
            _append_trial_epoch_log(
                path=epoch_log_path,
                row={
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "trial_id": int(cfg.trial_id),
                    "epoch": int(epoch),
                    "train_loss": np.nan,
                    "valid_loss": float(valid_loss),
                    "is_best_epoch": int(improved),
                    "best_valid_loss": float(best_valid_loss),
                    "no_improve": int(no_improve),
                },
            )
        if bool(args.save_trial_latest_state):
            _atomic_torch_save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": int(best_epoch),
                    "best_valid_loss": float(best_valid_loss),
                    "no_improve": int(no_improve),
                    "args_signature": args_signature,
                    "signature_payload": signature_payload,
                },
                path=latest_state_path,
            )
        if bool(args.save_runtime_status):
            _atomic_write_text(
                path=runtime_status_path,
                content=json.dumps(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "current_trial_id": int(cfg.trial_id),
                        "current_epoch": int(epoch),
                        "target_epochs": int(args.epochs),
                        "best_epoch": int(best_epoch),
                        "best_valid_loss": float(best_valid_loss),
                        "no_improve": int(no_improve),
                        "args_signature": args_signature,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        print(
            f"Trial {cfg.trial_id} | epoch={epoch:03d} | valid_loss={valid_loss:.8f} | "
            f"best_valid={best_valid_loss:.8f} | no_improve={no_improve}",
            flush=True,
        )
        if no_improve >= int(args.patience):
            break

    if not ckpt_path.exists():
        raise RuntimeError(f"Trial {cfg.trial_id}: checkpoint not saved.")
    return eval_existing_checkpoint(
        cfg=cfg,
        train_seq_map=train_seq_map,
        valid_seq_map=valid_seq_map,
        args=args,
        device=device,
        ckpt_path=ckpt_path,
    )


def eval_existing_checkpoint(
    cfg: TrialConfig,
    train_seq_map: Dict[Tuple[str, str], lstm_mod.SequenceData],
    valid_seq_map: Dict[Tuple[str, str], lstm_mod.SequenceData],
    args: argparse.Namespace,
    device: torch.device,
    ckpt_path: Path,
) -> Dict[str, float | int | str]:
    """Evaluate one existing checkpoint and build result row."""

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_dataset, valid_dataset = _build_datasets(
        cfg=cfg,
        train_seq_map=train_seq_map,
        valid_seq_map=valid_seq_map,
        args=args,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError(
            f"Trial {cfg.trial_id}: empty dataset for sequence_mode={cfg.sequence_mode}, "
            f"window_size={cfg.window_size}"
        )

    _, valid_loader = _build_loaders(
        cfg=cfg,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args,
        device=device,
    )
    model = _make_model(cfg=cfg, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    y_true, y_pred, _ = lstm_mod.predict_loader(model=model, loader=valid_loader, device=device)
    valid_metrics = lstm_mod.calc_metrics(y_true=y_true, y_pred=y_pred, set_type="valid")

    best_epoch = int(ckpt.get("best_epoch", 0))
    best_valid_loss = float(ckpt.get("best_valid_loss", np.nan))
    return _build_result_row(
        cfg=cfg,
        ckpt_path=ckpt_path,
        best_epoch=best_epoch,
        best_valid_loss=best_valid_loss,
        valid_metrics=valid_metrics,
        train_samples=int(len(train_dataset)),
        valid_samples=int(len(valid_dataset)),
    )


def _fmt_window_for_table(window_size: float) -> str:
    """Format window value for markdown table."""

    if pd.isna(window_size):
        return "-"
    return str(int(window_size))


def _best_window_value(best_row: pd.Series) -> Optional[int]:
    """Return best window size for JSON output."""

    if pd.isna(best_row["window_size"]):
        return None
    return int(best_row["window_size"])


def save_tuning_plot(results_df: pd.DataFrame, sequence_mode: str, out_png: Path) -> None:
    """Save scatter plot of grid trials."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    if sequence_mode == "fixed_window":
        marker_size = 36 + 1.3 * results_df["window_size"].fillna(0).to_numpy(dtype=float)
    else:
        marker_size = np.full(shape=len(results_df), fill_value=75.0, dtype=float)

    scatter = ax.scatter(
        results_df["valid_rmse"].to_numpy(dtype=float),
        results_df["valid_r2"].to_numpy(dtype=float),
        s=marker_size,
        c=results_df["hidden_size"].to_numpy(dtype=float),
        cmap="viridis",
        alpha=0.82,
    )
    for row in results_df.itertuples(index=False):
        ax.text(float(row.valid_rmse), float(row.valid_r2), f"T{int(row.trial_id)}", fontsize=8)
    ax.set_xlabel("Valid RMSE")
    ax.set_ylabel("Valid R2")
    ax.set_title("LSTM Grid Tuning Scatter (color=hidden_size, label=trial_id)")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("hidden_size")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def build_report(args: argparse.Namespace, results_df: pd.DataFrame, best_row: pd.Series) -> str:
    """Build Chinese markdown tuning report."""

    lines: List[str] = []
    lines.append("# LSTM 网格调参报告（delta_ah 口径）")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 设备：`{args.device}`")
    lines.append(f"- 序列模式：`{args.sequence_mode}`")
    lines.append(f"- checkpoint 回填：`{args.resume_existing}`")
    lines.append(f"- partial 续跑：`{args.resume_from_partial}`")
    lines.append(f"- trial 中断恢复：`{args.resume_interrupted}`")
    lines.append(f"- 写 partial 结果：`{args.save_partial_results}`")
    lines.append(f"- 写 trial latest：`{args.save_trial_latest_state}`")
    lines.append(f"- 写 trial epoch 日志：`{args.save_trial_epoch_log}`")
    lines.append(f"- 写 runtime 状态：`{args.save_runtime_status}`")
    if args.sequence_mode == "fixed_window":
        lines.append(
            f"- 搜索空间：window_size={args.window_sizes}，hidden_size={args.hidden_sizes}，"
            f"lr={args.learning_rates}，num_layers={args.num_layers_list}，dropout={args.dropout_list}"
        )
    else:
        lines.append(
            f"- 搜索空间：hidden_size={args.hidden_sizes}，lr={args.learning_rates}，"
            f"num_layers={args.num_layers_list}，dropout={args.dropout_list}"
        )
        lines.append("- 全历史前缀定义：样本 `t` 使用 `1..t` 全部历史序列。")
    lines.append(
        f"- 训练参数：epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}"
    )
    lines.append("")
    lines.append("## 2. 全部试验结果（按 Valid R2 降序）")
    lines.append(
        "| trial_id | seq_mode | window | hidden | lr | layers | dropout | best_epoch | valid_rmse | valid_mae | valid_r2 |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results_df.itertuples(index=False):
        lines.append(
            f"| {int(row.trial_id)} | {row.sequence_mode} | {_fmt_window_for_table(float(row.window_size))} | "
            f"{int(row.hidden_size)} | {float(row.learning_rate):.6g} | {int(row.num_layers)} | "
            f"{float(row.dropout):.2f} | {int(row.best_epoch)} | {float(row.valid_rmse):.6f} | "
            f"{float(row.valid_mae):.6f} | {float(row.valid_r2):.6f} |"
        )
    lines.append("")
    lines.append("## 3. 最优配置")
    lines.append(f"- trial_id：**{int(best_row['trial_id'])}**")
    best_window = "-" if pd.isna(best_row["window_size"]) else str(int(best_row["window_size"]))
    lines.append(
        f"- 参数：`sequence_mode={best_row['sequence_mode']}`, `window_size={best_window}`, "
        f"`hidden_size={int(best_row['hidden_size'])}`, `learning_rate={float(best_row['learning_rate']):.6g}`, "
        f"`num_layers={int(best_row['num_layers'])}`, `dropout={float(best_row['dropout']):.2f}`"
    )
    lines.append(
        f"- 指标：`valid_rmse={float(best_row['valid_rmse']):.6f}`, "
        f"`valid_mae={float(best_row['valid_mae']):.6f}`, `valid_r2={float(best_row['valid_r2']):.6f}`"
    )
    lines.append("")
    lines.append("## 4. 图表")
    lines.append("![grid_tuning_scatter](./grid_tuning_scatter.png)")
    lines.append("")
    lines.append("## 5. 结论")
    lines.append("- 建议后续正式训练优先使用本报告最优超参数。")
    lines.append("- 若需继续提升，可在最优配置附近细化学习率和隐藏层宽度。")
    return "\n".join(lines)


def _rows_to_df(rows: Sequence[Dict[str, float | int | str]]) -> pd.DataFrame:
    """Convert rows to DataFrame sorted by trial_id."""

    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(list(rows))
    return df.sort_values(["trial_id"], kind="mergesort").reset_index(drop=True)


def save_partial(rows: Sequence[Dict[str, float | int | str]], out_path: Path) -> None:
    """Save partial grid results after each trial."""

    partial_df = _rows_to_df(rows)
    if partial_df.empty:
        return
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    partial_df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, out_path)


def _load_partial_rows(partial_path: Path) -> Dict[int, Dict[str, float | int | str]]:
    """Load existing partial rows keyed by trial_id."""

    if not partial_path.exists():
        return {}
    try:
        partial_df = pd.read_csv(partial_path)
    except Exception as exc:
        print(f"Warning: failed to read partial file {partial_path}: {exc}", flush=True)
        return {}
    if partial_df.empty:
        return {}
    rows_by_id: Dict[int, Dict[str, float | int | str]] = {}
    for row in partial_df.to_dict(orient="records"):
        trial_id = int(row["trial_id"])
        rows_by_id[trial_id] = row
    return rows_by_id


def finalize_outputs(args: argparse.Namespace, rows: Sequence[Dict[str, float | int | str]]) -> None:
    """Save final outputs and print best trial info."""

    results_df = pd.DataFrame(list(rows))
    results_df = results_df.sort_values(
        ["valid_r2", "valid_rmse", "valid_mae"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_row = results_df.iloc[0]

    out_results = args.output_dir / "grid_search_results.csv"
    out_best = args.output_dir / "best_grid_config.json"
    out_plot = args.output_dir / "grid_tuning_scatter.png"
    out_report = args.output_dir / "lstm_grid_tuning_report.md"

    results_df.to_csv(out_results, index=False, encoding="utf-8")
    out_best.write_text(
        json.dumps(
            {
                "trial_id": int(best_row["trial_id"]),
                "sequence_mode": str(best_row["sequence_mode"]),
                "window_size": _best_window_value(best_row=best_row),
                "hidden_size": int(best_row["hidden_size"]),
                "learning_rate": float(best_row["learning_rate"]),
                "num_layers": int(best_row["num_layers"]),
                "dropout": float(best_row["dropout"]),
                "valid_rmse": float(best_row["valid_rmse"]),
                "valid_mae": float(best_row["valid_mae"]),
                "valid_r2": float(best_row["valid_r2"]),
                "best_epoch": int(best_row["best_epoch"]),
                "checkpoint_path": str(best_row["checkpoint_path"]),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_tuning_plot(results_df=results_df, sequence_mode=args.sequence_mode, out_png=out_plot)
    out_report.write_text(build_report(args=args, results_df=results_df, best_row=best_row), encoding="utf-8")

    best_window_str = "None" if pd.isna(best_row["window_size"]) else str(int(best_row["window_size"]))
    print(f"Saved: {out_results}")
    print(f"Saved: {out_best}")
    print(f"Saved: {out_plot}")
    print(f"Saved: {out_report}")
    print(
        "Best:",
        f"trial={int(best_row['trial_id'])}, seq_mode={best_row['sequence_mode']}, window={best_window_str},",
        f"hidden={int(best_row['hidden_size'])}, lr={float(best_row['learning_rate']):.6g},",
        f"layers={int(best_row['num_layers'])}, dropout={float(best_row['dropout']):.2f},",
        f"valid_r2={float(best_row['valid_r2']):.6f}",
    )


def main() -> None:
    """Run grid tuning with checkpoint resume and partial results."""

    args = parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
        args.max_train_windows = args.max_train_windows if args.max_train_windows > 0 else 4096
        args.max_valid_windows = args.max_valid_windows if args.max_valid_windows > 0 else 2048

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / str(args.partial_results_file)

    lstm_mod.ensure_matplotlib_backend()
    lstm_mod.set_seed(int(args.seed))
    device = lstm_mod.resolve_device(args.device)
    train_seq_map, valid_seq_map = load_sequence_maps(args=args)

    trials = _prepare_trial_grid(args=args)
    total_trials = len(trials)
    if bool(args.resume_from_partial):
        partial_rows = _load_partial_rows(partial_path=partial_path)
        print(f"Loaded partial rows: {len(partial_rows)}")
    else:
        partial_rows = {}
        print("Partial resume disabled by --no-resume-from-partial.")
    if not bool(args.save_partial_results):
        print("Partial result writing disabled by --no-save-partial-results.")

    trial_rows: List[Dict[str, float | int | str]] = []
    for cfg in trials:
        if bool(args.resume_from_partial) and cfg.trial_id in partial_rows:
            row = partial_rows[cfg.trial_id]
            trial_rows.append(row)
            print(f"[Trial {cfg.trial_id}/{total_trials}] skip (from partial csv)")
            continue

        ckpt_path = ckpt_dir / f"trial_{cfg.trial_id:03d}.pt"
        if args.resume_existing and ckpt_path.exists():
            print(
                f"[Trial {cfg.trial_id}/{total_trials}] backfill from checkpoint | "
                f"hidden={cfg.hidden_size}, lr={cfg.learning_rate}, layers={cfg.num_layers}, dropout={cfg.dropout}"
            )
            row = eval_existing_checkpoint(
                cfg=cfg,
                train_seq_map=train_seq_map,
                valid_seq_map=valid_seq_map,
                args=args,
                device=device,
                ckpt_path=ckpt_path,
            )
        else:
            print(
                f"[Trial {cfg.trial_id}/{total_trials}] train | hidden={cfg.hidden_size}, "
                f"lr={cfg.learning_rate}, layers={cfg.num_layers}, dropout={cfg.dropout}"
            )
            row = train_one_trial(
                cfg=cfg,
                train_seq_map=train_seq_map,
                valid_seq_map=valid_seq_map,
                args=args,
                device=device,
                ckpt_dir=ckpt_dir,
            )

        trial_rows.append(row)
        if bool(args.save_partial_results):
            save_partial(rows=trial_rows, out_path=partial_path)
        print(
            f"[Trial {cfg.trial_id}] valid_r2={float(row['valid_r2']):.6f}, "
            f"valid_rmse={float(row['valid_rmse']):.6f}, best_epoch={int(row['best_epoch'])}"
        )

    if len(trial_rows) != total_trials:
        raise RuntimeError(f"Incomplete grid rows: got {len(trial_rows)}, expected {total_trials}.")
    finalize_outputs(args=args, rows=trial_rows)


if __name__ == "__main__":
    main()

