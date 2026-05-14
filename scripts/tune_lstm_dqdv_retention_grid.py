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

import scripts.train_lstm_charge_delta_ah as base_lstm
import scripts.train_lstm_dqdv_retention as train_mod


@dataclass(frozen=True)
class TrialConfig:
    """One grid-search trial configuration."""

    trial_id: int
    hidden_size: int
    learning_rate: float
    num_layers: int
    dropout: float


def parse_args() -> argparse.Namespace:
    """Parse CLI args for dQdV-retention grid tuning and full refresh."""

    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 subset grid tuning for dQdV retention LSTM, then stage-2 full refresh "
            "from best_grid_config.json."
        )
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
        default=REPO_ROOT / "outputs" / "analysis" / "lstm_dqdv_retention_grid_cpu",
    )
    parser.add_argument(
        "--feature-pack",
        type=str,
        choices=sorted(train_mod.FEATURE_PACK_COLUMNS),
        default="main_peak_temp_cycle",
        help=(
            "Feature pack passed to the dQdV retention trainer. Use compact_peak_shape for "
            "main_peak_area, main_peak_skewness, main_peak_voltage_v, main_peak_width_v; "
            "use compact_peak_shape_height to add main_peak_height_dqdv without cycle index."
        ),
    )
    parser.add_argument("--hidden-sizes", type=str, default="64,128,192")
    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4")
    parser.add_argument("--num-layers-list", type=str, default="1,2")
    parser.add_argument("--dropout-list", type=str, default="0.1,0.2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--max-train-windows", type=int, default=20000)
    parser.add_argument("--max-valid-windows", type=int, default=8000)
    parser.add_argument("--checkpoint-snapshot-interval", type=int, default=10)
    parser.add_argument(
        "--resume-existing",
        dest="resume_existing",
        action="store_true",
        default=True,
        help="Backfill by evaluating existing trial best checkpoints.",
    )
    parser.add_argument(
        "--no-resume-existing",
        dest="resume_existing",
        action="store_false",
        help="Disable backfill from existing trial best checkpoints.",
    )
    parser.add_argument(
        "--resume-interrupted",
        dest="resume_interrupted",
        action="store_true",
        default=True,
        help="Resume trial training from latest state.",
    )
    parser.add_argument(
        "--no-resume-interrupted",
        dest="resume_interrupted",
        action="store_false",
        help="Disable trial interrupted resume.",
    )
    parser.add_argument(
        "--resume-from-partial",
        dest="resume_from_partial",
        action="store_true",
        default=True,
        help="Skip trials already stored in partial CSV.",
    )
    parser.add_argument(
        "--no-resume-from-partial",
        dest="resume_from_partial",
        action="store_false",
        help="Ignore partial CSV and rerun all trials.",
    )
    parser.add_argument(
        "--save-partial-results",
        dest="save_partial_results",
        action="store_true",
        default=True,
        help="Write partial trial results after each trial.",
    )
    parser.add_argument(
        "--no-save-partial-results",
        dest="save_partial_results",
        action="store_false",
        help="Disable partial result CSV writes.",
    )
    parser.add_argument(
        "--partial-results-file",
        type=str,
        default="grid_search_results.partial.csv",
    )
    parser.add_argument(
        "--runtime-status-file",
        type=str,
        default="grid_tuning_runtime_status.json",
    )
    parser.add_argument(
        "--run-full-refresh",
        dest="run_full_refresh",
        action="store_true",
        default=True,
        help="Run stage-2 full refresh from best_grid_config.json.",
    )
    parser.add_argument(
        "--skip-full-refresh",
        dest="run_full_refresh",
        action="store_false",
        help="Skip stage-2 full refresh training.",
    )
    parser.add_argument("--full-refresh-output-dir", type=Path, default=None)
    parser.add_argument("--full-refresh-epochs", type=int, default=80)
    parser.add_argument("--full-refresh-patience", type=int, default=12)
    parser.add_argument("--full-refresh-min-delta", type=float, default=1e-4)
    parser.add_argument("--full-refresh-batch-size", type=int, default=256)
    parser.add_argument("--full-refresh-snapshot-interval", type=int, default=10)
    parser.add_argument(
        "--full-refresh-resume-interrupted",
        dest="full_refresh_resume_interrupted",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-full-refresh-resume-interrupted",
        dest="full_refresh_resume_interrupted",
        action="store_false",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    """Parse comma-separated integer list."""

    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    """Parse comma-separated float list."""

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def prepare_trial_grid(args: argparse.Namespace) -> List[TrialConfig]:
    """Prepare Cartesian-product grid over required 4 hyper-parameters."""

    hidden_sizes = parse_int_list(args.hidden_sizes)
    learning_rates = parse_float_list(args.learning_rates)
    num_layers_list = parse_int_list(args.num_layers_list)
    dropout_list = parse_float_list(args.dropout_list)

    trials: List[TrialConfig] = []
    trial_id = 0
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for num_layers in num_layers_list:
                for dropout in dropout_list:
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
    return trials


def load_sequence_maps(
    args: argparse.Namespace,
) -> Tuple[
    Dict[Tuple[str, str], base_lstm.SequenceData],
    Dict[Tuple[str, str], base_lstm.SequenceData],
    Dict[Tuple[str, str, int], Dict[str, float]],
]:
    """Load data once and build train/valid sequence maps + label lookup."""

    split_df = base_lstm.load_split_map(args.train_split_path, args.valid_split_path)
    feature_df = train_mod.load_dqdv_main_feature_table(args.dqdv_path)
    label_df = train_mod.load_retention_labels(
        life_path=args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )
    merged = train_mod.merge_feature_label_split(feature_df=feature_df, label_df=label_df, split_df=split_df)
    merged = train_mod.add_cycle_index_norm(merged)
    feature_cols = train_mod.get_model_feature_columns(str(args.feature_pack))
    merged = train_mod.coerce_feature_columns(merged, feature_cols)
    seq_map = train_mod.build_sequences(merged=merged, feature_cols=feature_cols)
    train_seq_map, valid_seq_map = base_lstm.split_sequence_dict(seq_map)
    label_lookup = train_mod.build_label_lookup(merged)
    return train_seq_map, valid_seq_map, label_lookup


def build_datasets(
    cfg: TrialConfig,
    train_seq_map: Dict[Tuple[str, str], base_lstm.SequenceData],
    valid_seq_map: Dict[Tuple[str, str], base_lstm.SequenceData],
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build trial train/valid prefix datasets with subset caps."""

    train_dataset: torch.utils.data.Dataset = base_lstm.PrefixHistoryDataset(
        sequences=train_seq_map,
        max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
        seed=int(args.seed) + int(cfg.trial_id),
    )
    valid_dataset: torch.utils.data.Dataset = base_lstm.PrefixHistoryDataset(
        sequences=valid_seq_map,
        max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
        seed=int(args.seed) + int(cfg.trial_id) + 1,
    )
    return train_dataset, valid_dataset


def build_loaders(
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build trial dataloaders."""

    train_loader = base_lstm.build_dataloader(
        dataset=train_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
        collate_fn=base_lstm.collate_sequence_batch,
    )
    valid_loader = base_lstm.build_dataloader(
        dataset=valid_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
        collate_fn=base_lstm.collate_sequence_batch,
    )
    return train_loader, valid_loader


def make_model(cfg: TrialConfig, args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Build one LSTM model instance for current trial."""

    feature_cols = train_mod.get_model_feature_columns(str(args.feature_pack))
    return base_lstm.LSTMRegressor(
        input_size=len(feature_cols),
        hidden_size=int(cfg.hidden_size),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
    ).to(device)


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically."""

    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def append_trial_epoch_log(path: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch record to per-trial epoch log."""

    row_df = pd.DataFrame([dict(row)])
    if not path.exists():
        row_df.to_csv(path, index=False, encoding="utf-8")
    else:
        row_df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")


def build_trial_signature_payload(cfg: TrialConfig, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Build deterministic resume payload for one trial."""

    feature_cols = train_mod.get_model_feature_columns(str(args.feature_pack))
    return {
        "trial_id": int(cfg.trial_id),
        "feature_pack": str(args.feature_pack),
        "feature_columns": list(feature_cols),
        "input_size": int(len(feature_cols)),
        "device": str(device),
        "hidden_size": int(cfg.hidden_size),
        "learning_rate": float(cfg.learning_rate),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "batch_size": int(args.batch_size),
        "max_train_windows": int(args.max_train_windows),
        "max_valid_windows": int(args.max_valid_windows),
        "q_min": float(args.q_min),
        "q_max": float(args.q_max),
        "q_ref_cycles": int(args.q_ref_cycles),
        "retention_min": float(args.retention_min),
        "retention_max": float(args.retention_max),
        "checkpoint_snapshot_interval": int(args.checkpoint_snapshot_interval),
        "seed": int(args.seed),
    }


def build_trial_signature(payload: Mapping[str, Any]) -> str:
    """Hash trial payload into deterministic signature."""

    canonical = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def infer_checkpoint_input_size(checkpoint: Mapping[str, Any]) -> Optional[int]:
    """Infer saved model input size from checkpoint metadata or LSTM weight shape."""

    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping) and "input_size" in model_config:
        return int(model_config["input_size"])
    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, Mapping):
        weight = state_dict.get("lstm.weight_ih_l0")
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[1])
    return None


def build_trial_paths(ckpt_root: Path, trial_id: int) -> Tuple[Path, Path, Path, Path]:
    """Build per-trial output paths: (best, latest, epoch_log, snapshot_dir)."""

    trial_dir = ckpt_root / f"trial_{int(trial_id):03d}"
    best_path = trial_dir / "best.pt"
    latest_path = trial_dir / "latest.pt"
    epoch_log_path = trial_dir / "epoch_log.csv"
    snapshot_dir = trial_dir / "snapshots"
    return best_path, latest_path, epoch_log_path, snapshot_dir


def save_snapshot_if_needed(
    snapshot_dir: Path,
    epoch: int,
    interval: int,
    payload: Mapping[str, Any],
) -> Optional[Path]:
    """Save periodic trial snapshot if interval condition is met."""

    if int(interval) <= 0:
        return None
    if int(epoch) % int(interval) != 0:
        return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out_path = snapshot_dir / f"epoch_{int(epoch):04d}.pt"
    base_lstm.atomic_torch_save(dict(payload), path=out_path)
    return out_path


def build_result_row(
    cfg: TrialConfig,
    ckpt_path: Path,
    best_epoch: int,
    best_valid_loss: float,
    valid_metrics_ret: base_lstm.Metrics,
    valid_metrics_q: base_lstm.Metrics,
    train_samples: int,
    valid_samples: int,
    args: argparse.Namespace,
) -> Dict[str, float | int | str]:
    """Build one result row for grid result CSV."""

    feature_cols = train_mod.get_model_feature_columns(str(args.feature_pack))
    return {
        "trial_id": int(cfg.trial_id),
        "feature_pack": str(args.feature_pack),
        "input_size": int(len(feature_cols)),
        "feature_columns": ",".join(feature_cols),
        "hidden_size": int(cfg.hidden_size),
        "learning_rate": float(cfg.learning_rate),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "valid_retention_mse": float(valid_metrics_ret.mse),
        "valid_retention_rmse": float(valid_metrics_ret.rmse),
        "valid_retention_mae": float(valid_metrics_ret.mae),
        "valid_retention_r2": float(valid_metrics_ret.r2),
        "valid_q_mse": float(valid_metrics_q.mse),
        "valid_q_rmse": float(valid_metrics_q.rmse),
        "valid_q_mae": float(valid_metrics_q.mae),
        "valid_q_r2": float(valid_metrics_q.r2),
        "train_samples": int(train_samples),
        "valid_samples": int(valid_samples),
        "checkpoint_path": str(ckpt_path),
    }


def eval_existing_checkpoint(
    cfg: TrialConfig,
    train_seq_map: Dict[Tuple[str, str], base_lstm.SequenceData],
    valid_seq_map: Dict[Tuple[str, str], base_lstm.SequenceData],
    label_lookup: Mapping[Tuple[str, str, int], Mapping[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    ckpt_path: Path,
) -> Dict[str, float | int | str]:
    """Evaluate an existing trial best checkpoint and return result row."""

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_dataset, valid_dataset = build_datasets(
        cfg=cfg,
        train_seq_map=train_seq_map,
        valid_seq_map=valid_seq_map,
        args=args,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError(f"Trial {cfg.trial_id}: empty dataset.")

    _, valid_loader = build_loaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args,
        device=device,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    expected_input_size = len(train_mod.get_model_feature_columns(str(args.feature_pack)))
    ckpt_input_size = infer_checkpoint_input_size(ckpt)
    if ckpt_input_size is not None and ckpt_input_size != expected_input_size:
        raise RuntimeError(
            f"Trial {cfg.trial_id}: checkpoint input_size={ckpt_input_size} does not match "
            f"feature_pack={args.feature_pack!r} input_size={expected_input_size}. "
            "Use a separate output directory or disable --resume-existing."
        )

    model = make_model(cfg=cfg, args=args, device=device)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true_ret, y_pred_ret, valid_idx = base_lstm.predict_loader(model=model, loader=valid_loader, device=device)
    valid_metrics_ret = base_lstm.calc_metrics(y_true=y_true_ret, y_pred=y_pred_ret, set_type="valid")

    valid_metas = [valid_dataset.metas[int(i)] for i in valid_idx.tolist()]
    valid_q_true, valid_q_ref = train_mod.build_q_arrays_from_metas(valid_metas, label_lookup=label_lookup)
    valid_q_pred = y_pred_ret.astype(np.float32) * valid_q_ref.astype(np.float32)
    valid_metrics_q = base_lstm.calc_metrics(y_true=valid_q_true, y_pred=valid_q_pred, set_type="valid")

    return build_result_row(
        cfg=cfg,
        ckpt_path=ckpt_path,
        best_epoch=int(ckpt.get("best_epoch", 0)),
        best_valid_loss=float(ckpt.get("best_valid_loss", np.nan)),
        valid_metrics_ret=valid_metrics_ret,
        valid_metrics_q=valid_metrics_q,
        train_samples=int(len(train_dataset)),
        valid_samples=int(len(valid_dataset)),
        args=args,
    )


def train_one_trial(
    cfg: TrialConfig,
    train_seq_map: Dict[Tuple[str, str], base_lstm.SequenceData],
    valid_seq_map: Dict[Tuple[str, str], base_lstm.SequenceData],
    label_lookup: Mapping[Tuple[str, str, int], Mapping[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    ckpt_root: Path,
) -> Dict[str, float | int | str]:
    """Train one trial with low-cost checkpoint policy and return result row."""

    train_dataset, valid_dataset = build_datasets(
        cfg=cfg,
        train_seq_map=train_seq_map,
        valid_seq_map=valid_seq_map,
        args=args,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError(f"Trial {cfg.trial_id}: empty dataset.")

    train_loader, valid_loader = build_loaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args,
        device=device,
    )
    model = make_model(cfg=cfg, args=args, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_path, latest_path, epoch_log_path, snapshot_dir = build_trial_paths(ckpt_root=ckpt_root, trial_id=cfg.trial_id)
    runtime_status_path = ckpt_root / str(args.runtime_status_file)
    signature_payload = build_trial_signature_payload(cfg=cfg, args=args, device=device)
    args_signature = build_trial_signature(payload=signature_payload)

    best_valid_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    start_epoch = 1

    if bool(args.resume_interrupted) and latest_path.exists():
        latest_state = torch.load(latest_path, map_location=device)
        saved_signature = str(latest_state.get("args_signature", ""))
        if saved_signature != args_signature:
            raise RuntimeError(
                f"Trial {cfg.trial_id}: latest state signature mismatch. "
                "Use --no-resume-interrupted or a new output dir."
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

    if start_epoch <= 1:
        epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
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

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_loss = base_lstm.train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = base_lstm.eval_loss(model, valid_loader, criterion, device)
        improved = (best_valid_loss - valid_loss) > float(args.min_delta)
        if improved:
            best_valid_loss = float(valid_loss)
            best_epoch = int(epoch)
            no_improve = 0
            base_lstm.atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": len(train_mod.get_model_feature_columns(str(args.feature_pack))),
                        "hidden_size": int(cfg.hidden_size),
                        "num_layers": int(cfg.num_layers),
                        "dropout": float(cfg.dropout),
                        "feature_pack": str(args.feature_pack),
                        "feature_columns": train_mod.get_model_feature_columns(str(args.feature_pack)),
                        "sequence_mode": "prefix_full",
                    },
                    "hidden_size": int(cfg.hidden_size),
                    "learning_rate": float(cfg.learning_rate),
                    "num_layers": int(cfg.num_layers),
                    "dropout": float(cfg.dropout),
                    "best_epoch": int(best_epoch),
                    "best_valid_loss": float(best_valid_loss),
                },
                path=best_path,
            )
        else:
            no_improve += 1

        append_trial_epoch_log(
            path=epoch_log_path,
            row={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trial_id": int(cfg.trial_id),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "is_best_epoch": int(improved),
                "best_valid_loss": float(best_valid_loss),
                "no_improve": int(no_improve),
            },
        )

        latest_payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": int(best_epoch),
            "best_valid_loss": float(best_valid_loss),
            "no_improve": int(no_improve),
            "args_signature": args_signature,
            "signature_payload": signature_payload,
        }
        base_lstm.atomic_torch_save(latest_payload, path=latest_path)
        save_snapshot_if_needed(
            snapshot_dir=snapshot_dir,
            epoch=int(epoch),
            interval=int(args.checkpoint_snapshot_interval),
            payload=latest_payload,
        )
        atomic_write_text(
            path=runtime_status_path,
            content=json.dumps(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "current_trial_id": int(cfg.trial_id),
                    "current_epoch": int(epoch),
                    "target_epochs": int(args.epochs),
                    "feature_pack": str(args.feature_pack),
                    "input_size": len(train_mod.get_model_feature_columns(str(args.feature_pack))),
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
            f"Trial {cfg.trial_id} | epoch={epoch:03d} | train_loss={train_loss:.8f} | "
            f"valid_loss={valid_loss:.8f} | best_valid={best_valid_loss:.8f} | no_improve={no_improve}",
            flush=True,
        )
        if no_improve >= int(args.patience):
            print(f"Trial {cfg.trial_id}: early stop at epoch {epoch}.", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError(f"Trial {cfg.trial_id}: best checkpoint not saved.")

    return eval_existing_checkpoint(
        cfg=cfg,
        train_seq_map=train_seq_map,
        valid_seq_map=valid_seq_map,
        label_lookup=label_lookup,
        args=args,
        device=device,
        ckpt_path=best_path,
    )


def load_partial_rows(path: Path) -> Dict[int, Dict[str, float | int | str]]:
    """Load partial trial rows from csv, keyed by trial_id."""

    if not path.exists():
        return {}
    try:
        partial_df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to read partial file {path}: {exc}", flush=True)
        return {}
    if partial_df.empty:
        return {}
    rows: Dict[int, Dict[str, float | int | str]] = {}
    for row in partial_df.to_dict(orient="records"):
        rows[int(row["trial_id"])] = row
    return rows


def rows_to_df(rows: Sequence[Dict[str, float | int | str]]) -> pd.DataFrame:
    """Convert row list to sorted DataFrame."""

    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(list(rows))
    return df.sort_values(["trial_id"], kind="mergesort").reset_index(drop=True)


def save_partial(rows: Sequence[Dict[str, float | int | str]], out_path: Path) -> None:
    """Save partial rows atomically."""

    partial_df = rows_to_df(rows)
    if partial_df.empty:
        return
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    partial_df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, out_path)


def save_tuning_plot(results_df: pd.DataFrame, out_png: Path) -> None:
    """Save retention-RMSE/R2 scatter plot for trial comparison."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    marker_size = 48 + 0.2 * results_df["hidden_size"].to_numpy(dtype=float)
    scatter = ax.scatter(
        results_df["valid_retention_rmse"].to_numpy(dtype=float),
        results_df["valid_retention_r2"].to_numpy(dtype=float),
        s=marker_size,
        c=results_df["hidden_size"].to_numpy(dtype=float),
        cmap="viridis",
        alpha=0.82,
    )
    for row in results_df.itertuples(index=False):
        ax.text(float(row.valid_retention_rmse), float(row.valid_retention_r2), f"T{int(row.trial_id)}", fontsize=8)
    ax.set_xlabel("Valid Retention RMSE")
    ax.set_ylabel("Valid Retention R2")
    ax.set_title("dQdV-Retention Grid Tuning Scatter (color=hidden_size)")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("hidden_size")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def build_report(args: argparse.Namespace, results_df: pd.DataFrame, best_row: pd.Series) -> str:
    """Build markdown report for stage-1 grid tuning."""

    lines: List[str] = []
    lines.append("# dQdV Retention LSTM 网格调参报告（阶段1：子集寻优）")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 设备：`{args.device}`")
    lines.append(f"- 特征包：`{args.feature_pack}`（{train_mod.describe_feature_pack(str(args.feature_pack))}）")
    lines.append(f"- 输入维度：`{len(train_mod.get_model_feature_columns(str(args.feature_pack)))}`")
    lines.append(f"- 搜索空间：hidden={args.hidden_sizes}, lr={args.learning_rates}, layers={args.num_layers_list}, dropout={args.dropout_list}")
    lines.append(f"- 每 trial 训练：epochs={args.epochs}, patience={args.patience}")
    lines.append(f"- 子集窗口上限：train={args.max_train_windows}, valid={args.max_valid_windows}")
    lines.append("")
    lines.append("## 2. 全部试验结果（按 retention R2 降序）")
    lines.append("| trial_id | hidden | lr | layers | dropout | best_epoch | ret_rmse | ret_r2 | q_rmse | q_r2 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results_df.itertuples(index=False):
        lines.append(
            f"| {int(row.trial_id)} | {int(row.hidden_size)} | {float(row.learning_rate):.6g} | "
            f"{int(row.num_layers)} | {float(row.dropout):.2f} | {int(row.best_epoch)} | "
            f"{float(row.valid_retention_rmse):.6f} | {float(row.valid_retention_r2):.6f} | "
            f"{float(row.valid_q_rmse):.6f} | {float(row.valid_q_r2):.6f} |"
        )
    lines.append("")
    lines.append("## 3. 最优配置")
    lines.append(f"- trial_id：**{int(best_row['trial_id'])}**")
    lines.append(
        f"- 参数：`hidden_size={int(best_row['hidden_size'])}`, "
        f"`learning_rate={float(best_row['learning_rate']):.6g}`, "
        f"`num_layers={int(best_row['num_layers'])}`, `dropout={float(best_row['dropout']):.2f}`"
    )
    lines.append(
        f"- retention：`rmse={float(best_row['valid_retention_rmse']):.6f}`, "
        f"`r2={float(best_row['valid_retention_r2']):.6f}`"
    )
    lines.append("")
    lines.append("## 4. 图表")
    lines.append("![grid_tuning_scatter](./grid_tuning_scatter.png)")
    return "\n".join(lines)


def finalize_stage1_outputs(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, float | int | str]],
) -> Tuple[pd.DataFrame, Path]:
    """Save stage-1 outputs and return sorted result dataframe + best json path."""

    results_df = pd.DataFrame(list(rows))
    results_df = results_df.sort_values(
        ["valid_retention_r2", "valid_retention_rmse", "valid_retention_mae"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_row = results_df.iloc[0]

    out_results = args.output_dir / "grid_search_results.csv"
    out_best = args.output_dir / "best_grid_config.json"
    out_plot = args.output_dir / "grid_tuning_scatter.png"
    out_report = args.output_dir / "lstm_dqdv_retention_grid_report.md"

    results_df.to_csv(out_results, index=False, encoding="utf-8")
    out_best.write_text(
        json.dumps(
            {
                "trial_id": int(best_row["trial_id"]),
                "feature_pack": str(args.feature_pack),
                "feature_pack_description": train_mod.describe_feature_pack(str(args.feature_pack)),
                "feature_columns": train_mod.get_model_feature_columns(str(args.feature_pack)),
                "input_size": int(len(train_mod.get_model_feature_columns(str(args.feature_pack)))),
                "hidden_size": int(best_row["hidden_size"]),
                "learning_rate": float(best_row["learning_rate"]),
                "num_layers": int(best_row["num_layers"]),
                "dropout": float(best_row["dropout"]),
                "best_epoch": int(best_row["best_epoch"]),
                "best_valid_loss": float(best_row["best_valid_loss"]),
                "valid_retention_rmse": float(best_row["valid_retention_rmse"]),
                "valid_retention_r2": float(best_row["valid_retention_r2"]),
                "valid_q_rmse": float(best_row["valid_q_rmse"]),
                "valid_q_r2": float(best_row["valid_q_r2"]),
                "checkpoint_path": str(best_row["checkpoint_path"]),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_tuning_plot(results_df=results_df, out_png=out_plot)
    out_report.write_text(build_report(args=args, results_df=results_df, best_row=best_row), encoding="utf-8")

    print(f"Saved: {out_results}")
    print(f"Saved: {out_best}")
    print(f"Saved: {out_plot}")
    print(f"Saved: {out_report}")
    return results_df, out_best


def run_full_refresh_from_best(args: argparse.Namespace, best_config_path: Path) -> Dict[str, Any]:
    """Run stage-2 full refresh training from best_grid_config.json."""

    if not best_config_path.exists():
        raise FileNotFoundError(f"Best config json not found: {best_config_path}")
    best_cfg = json.loads(best_config_path.read_text(encoding="utf-8"))

    full_output_dir = args.full_refresh_output_dir
    if full_output_dir is None:
        full_output_dir = args.output_dir / "full_refresh"

    train_args = argparse.Namespace(
        dqdv_path=args.dqdv_path,
        life_path=args.life_path,
        train_split_path=args.train_split_path,
        valid_split_path=args.valid_split_path,
        output_dir=full_output_dir,
        feature_pack=str(args.feature_pack),
        sequence_mode="prefix_full",
        batch_size=int(args.full_refresh_batch_size),
        epochs=int(args.full_refresh_epochs),
        learning_rate=float(best_cfg["learning_rate"]),
        weight_decay=float(args.weight_decay),
        hidden_size=int(best_cfg["hidden_size"]),
        num_layers=int(best_cfg["num_layers"]),
        dropout=float(best_cfg["dropout"]),
        patience=int(args.full_refresh_patience),
        min_delta=float(args.full_refresh_min_delta),
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
        device=str(args.device),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        max_train_windows=int(args.max_train_windows) if bool(args.smoke_test) else 0,
        max_valid_windows=int(args.max_valid_windows) if bool(args.smoke_test) else 0,
        checkpoint_snapshot_interval=int(args.full_refresh_snapshot_interval),
        resume_interrupted=bool(args.full_refresh_resume_interrupted),
        best_state_file="best.pt",
        latest_state_file="latest.pt",
        epoch_log_file="epoch_log.csv",
        status_file="runtime_status.json",
        smoke_test=bool(args.smoke_test),
    )

    print("[Stage-2] Run full refresh training from best_grid_config.json", flush=True)
    summary = train_mod.run_training(args=train_args)
    return summary


def main() -> None:
    """Run stage-1 tuning and optional stage-2 full refresh."""

    args = parse_args()
    if args.smoke_test:
        args.epochs = min(int(args.epochs), 3)
        args.patience = min(int(args.patience), 2)
        args.max_train_windows = int(args.max_train_windows) if int(args.max_train_windows) > 0 else 2048
        args.max_valid_windows = int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else 1024
        args.full_refresh_epochs = min(int(args.full_refresh_epochs), 3)
        args.full_refresh_patience = min(int(args.full_refresh_patience), 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = args.output_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / str(args.partial_results_file)

    base_lstm.ensure_matplotlib_backend()
    base_lstm.set_seed(int(args.seed))
    device = base_lstm.resolve_device(args.device)

    trials = prepare_trial_grid(args=args)
    total_trials = len(trials)
    if total_trials != 24:
        print(f"Warning: current trial count is {total_trials}, not 24.", flush=True)

    train_seq_map, valid_seq_map, label_lookup = load_sequence_maps(args=args)

    partial_rows = load_partial_rows(partial_path) if bool(args.resume_from_partial) else {}
    if bool(args.resume_from_partial):
        print(f"Loaded partial rows: {len(partial_rows)}")
    else:
        print("Partial resume disabled by --no-resume-from-partial.")

    trial_rows: List[Dict[str, float | int | str]] = []
    for cfg in trials:
        if bool(args.resume_from_partial) and int(cfg.trial_id) in partial_rows:
            trial_rows.append(partial_rows[int(cfg.trial_id)])
            print(f"[Trial {cfg.trial_id}/{total_trials}] skip (from partial csv)")
            continue

        best_path, _, _, _ = build_trial_paths(ckpt_root=ckpt_root, trial_id=cfg.trial_id)
        if bool(args.resume_existing) and best_path.exists():
            print(
                f"[Trial {cfg.trial_id}/{total_trials}] backfill from best checkpoint "
                f"| hidden={cfg.hidden_size}, lr={cfg.learning_rate}, layers={cfg.num_layers}, dropout={cfg.dropout}",
                flush=True,
            )
            row = eval_existing_checkpoint(
                cfg=cfg,
                train_seq_map=train_seq_map,
                valid_seq_map=valid_seq_map,
                label_lookup=label_lookup,
                args=args,
                device=device,
                ckpt_path=best_path,
            )
        else:
            print(
                f"[Trial {cfg.trial_id}/{total_trials}] train "
                f"| hidden={cfg.hidden_size}, lr={cfg.learning_rate}, layers={cfg.num_layers}, dropout={cfg.dropout}",
                flush=True,
            )
            row = train_one_trial(
                cfg=cfg,
                train_seq_map=train_seq_map,
                valid_seq_map=valid_seq_map,
                label_lookup=label_lookup,
                args=args,
                device=device,
                ckpt_root=ckpt_root,
            )

        trial_rows.append(row)
        if bool(args.save_partial_results):
            save_partial(rows=trial_rows, out_path=partial_path)
        print(
            f"[Trial {cfg.trial_id}] valid_retention_r2={float(row['valid_retention_r2']):.6f}, "
            f"valid_retention_rmse={float(row['valid_retention_rmse']):.6f}, best_epoch={int(row['best_epoch'])}",
            flush=True,
        )

    if len(trial_rows) != total_trials:
        raise RuntimeError(f"Incomplete grid rows: got {len(trial_rows)}, expected {total_trials}.")

    _, best_json_path = finalize_stage1_outputs(args=args, rows=trial_rows)

    if bool(args.run_full_refresh):
        full_summary = run_full_refresh_from_best(args=args, best_config_path=best_json_path)
        print(f"[Stage-2] Full refresh done: {json.dumps(full_summary, ensure_ascii=False)}")
    else:
        print("[Stage-2] skipped by --skip-full-refresh")


if __name__ == "__main__":
    main()
