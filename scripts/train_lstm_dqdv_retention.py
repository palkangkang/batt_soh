from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"
sys.path.append(str(REPO_ROOT))

import scripts.train_lstm_charge_delta_ah as base_lstm

MAIN_PEAK_FEATURE_COLUMNS: List[str] = [
    "main_peak_voltage_v",
    "main_peak_width_v",
    "main_peak_height_dqdv",
    "main_peak_area",
    "main_peak_prominence",
    "main_peak_skewness",
    "main_peak_temp_max_c",
    "main_peak_temp_min_c",
    "main_peak_temp_avg_c",
]
MODEL_FEATURE_COLUMNS: List[str] = [*MAIN_PEAK_FEATURE_COLUMNS, "cycle_index_norm"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dQdV-retention LSTM training."""

    parser = argparse.ArgumentParser(
        description=(
            "Train LSTM on discharge dQdV main-peak features to fit capacity retention "
            "under policy+cell_code sequence split."
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
        default=REPO_ROOT / "outputs" / "analysis" / "lstm_dqdv_retention_full_refresh",
    )
    parser.add_argument(
        "--feature-pack",
        type=str,
        choices=["main_peak_temp_cycle"],
        default="main_peak_temp_cycle",
        help="Fixed feature pack: main peak + main-peak temperature + normalized cycle index.",
    )
    parser.add_argument(
        "--sequence-mode",
        type=str,
        choices=["prefix_full"],
        default="prefix_full",
        help="Fixed to prefix_full by design.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--checkpoint-snapshot-interval", type=int, default=10)
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
        help="Disable epoch-level resume and restart from epoch 1.",
    )
    parser.add_argument("--best-state-file", type=str, default="best.pt")
    parser.add_argument("--latest-state-file", type=str, default="latest.pt")
    parser.add_argument("--epoch-log-file", type=str, default="epoch_log.csv")
    parser.add_argument("--status-file", type=str, default="runtime_status.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Normalize truth-like strings and bools into strict bool series."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["1", "true", "yes", "y", "t"])


def load_dqdv_main_feature_table(path: Path) -> pd.DataFrame:
    """Load fixed dQdV feature pack and keep only valid curves."""

    use_cols = [
        "policy",
        "cell_code",
        "cycles",
        "is_valid_curve",
        *MAIN_PEAK_FEATURE_COLUMNS,
    ]
    dqdv = pd.read_csv(path, encoding=ENCODING, usecols=use_cols)
    dqdv["policy"] = dqdv["policy"].astype(str)
    dqdv["cell_code"] = dqdv["cell_code"].astype(str)
    dqdv["cycles"] = pd.to_numeric(dqdv["cycles"], errors="coerce")
    dqdv["is_valid_curve"] = normalize_bool_series(dqdv["is_valid_curve"])
    for col in MAIN_PEAK_FEATURE_COLUMNS:
        dqdv[col] = pd.to_numeric(dqdv[col], errors="coerce")

    dqdv = dqdv.dropna(subset=["policy", "cell_code", "cycles"]).copy()
    dqdv["cycles"] = dqdv["cycles"].astype(int)
    dqdv = dqdv.loc[dqdv["is_valid_curve"]].copy()
    dqdv = dqdv.drop(columns=["is_valid_curve"])
    dqdv = dqdv.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")
    dqdv = dqdv.drop_duplicates(["policy", "cell_code", "cycles"], keep="last").reset_index(drop=True)
    return dqdv


def load_retention_labels(
    life_path: Path,
    q_min: float,
    q_max: float,
    q_ref_cycles: int,
    retention_min: float,
    retention_max: float,
) -> pd.DataFrame:
    """Load labels and build retention target with fixed two-stage filtering."""

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
    life = life.loc[life["q_discharge"] > 0].copy()
    life = life.sort_values(["policy", "cell_code", "cycles"], kind="mergesort")

    abs_filtered = life.loc[(life["q_discharge"] >= q_min) & (life["q_discharge"] <= q_max)].copy()
    if abs_filtered.empty:
        raise RuntimeError("No rows remain after absolute q_discharge filtering.")

    early_cycles = abs_filtered.groupby(["policy", "cell_code"], sort=False).head(int(q_ref_cycles))
    q_ref = (
        early_cycles.groupby(["policy", "cell_code"], as_index=False)["q_discharge"]
        .median()
        .rename(columns={"q_discharge": "q_ref"})
    )
    q_ref = q_ref.loc[q_ref["q_ref"] > 0].copy()
    if q_ref.empty:
        raise RuntimeError("No valid q_ref generated from early cycles.")

    labeled = abs_filtered.merge(q_ref, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    labeled["retention"] = labeled["q_discharge"] / labeled["q_ref"]
    labeled = labeled.loc[
        (labeled["retention"] >= float(retention_min)) & (labeled["retention"] <= float(retention_max))
    ].copy()
    if labeled.empty:
        raise RuntimeError("No rows remain after retention filtering.")

    return labeled[["policy", "cell_code", "cycles", "q_discharge", "q_ref", "retention"]].copy()


def merge_feature_label_split(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge dQdV features, retention labels, and train/valid split."""

    merged = label_df.merge(feature_df, on=["policy", "cell_code", "cycles"], how="inner")
    merged = merged.merge(split_df, on=["policy", "cell_code"], how="inner", validate="many_to_one")
    merged = merged.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(drop=True)
    return merged


def add_cycle_index_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-cell normalized cycle index in [0, 1]."""

    out = df.copy()
    group_keys = ["policy", "cell_code"]
    c_min = out.groupby(group_keys)["cycles"].transform("min")
    c_max = out.groupby(group_keys)["cycles"].transform("max")
    denom = (c_max - c_min).replace(0, 1)
    out["cycle_index_norm"] = ((out["cycles"] - c_min) / denom).astype(np.float32)
    return out


def coerce_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Coerce model feature columns to float32 and fill NaN with zeros."""

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
) -> Dict[Tuple[str, str], base_lstm.SequenceData]:
    """Build per-cell sequential samples for prefix_full dataset."""

    seq_map: Dict[Tuple[str, str], base_lstm.SequenceData] = {}
    for (policy, cell_code), part in merged.groupby(["policy", "cell_code"], sort=False):
        part = part.sort_values("cycles", kind="mergesort").copy()
        x = part[list(feature_cols)].to_numpy(dtype=np.float32)
        y = part["retention"].to_numpy(dtype=np.float32)
        cycles = part["cycles"].to_numpy(dtype=np.int32)
        set_types = part["set_type"].dropna().unique().tolist()
        if len(set_types) != 1:
            raise RuntimeError(f"Split leakage detected for key {(policy, cell_code)}.")
        seq_map[(str(policy), str(cell_code))] = base_lstm.SequenceData(
            policy=str(policy),
            cell_code=str(cell_code),
            set_type=str(set_types[0]),
            cycles=cycles,
            x=x,
            y=y,
        )
    return seq_map


def build_label_lookup(merged: pd.DataFrame) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """Build fast lookup of label metadata by (policy, cell_code, cycles)."""

    lookup: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for row in merged.itertuples(index=False):
        key = (str(row.policy), str(row.cell_code), int(row.cycles))
        lookup[key] = {
            "q_discharge": float(row.q_discharge),
            "q_ref": float(row.q_ref),
            "retention": float(row.retention),
        }
    return lookup


def build_resume_signature_payload(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Build deterministic payload to validate resume compatibility."""

    return {
        "feature_pack": str(args.feature_pack),
        "sequence_mode": str(args.sequence_mode),
        "device": str(device),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
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


def build_resume_signature(payload: Mapping[str, Any]) -> str:
    """Hash payload into deterministic resume signature."""

    canonical = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_epoch_progress(path: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch row into epoch log CSV."""

    epoch_df = pd.DataFrame([dict(row)])
    if not path.exists():
        epoch_df.to_csv(path, index=False, encoding="utf-8")
    else:
        epoch_df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")


def save_snapshot_if_needed(
    snapshot_dir: Path,
    epoch: int,
    interval: int,
    payload: Mapping[str, Any],
) -> Optional[Path]:
    """Save periodic checkpoint snapshot when interval condition is met."""

    if interval <= 0:
        return None
    if int(epoch) % int(interval) != 0:
        return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out_path = snapshot_dir / f"epoch_{int(epoch):04d}.pt"
    base_lstm.atomic_torch_save(dict(payload), path=out_path)
    return out_path


def build_dataset_checks(
    merged: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Build high-level consistency checks for dQdV-retention dataset."""

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
    feature_mat = merged[list(feature_cols)].to_numpy(dtype=np.float32)
    checks = [
        ("check_feature_dim_10", int(len(feature_cols) == 10)),
        ("check_split_overlap_zero", int(overlap == 0)),
        ("check_feature_nan_free", int(np.isfinite(feature_mat).all())),
        ("check_retention_positive", int((merged["retention"] > 0).all())),
    ]
    return pd.DataFrame(checks, columns=["check_item", "pass_flag"])


def build_q_arrays_from_metas(
    metas: Sequence[base_lstm.WindowMeta],
    label_lookup: Mapping[Tuple[str, str, int], Mapping[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build true q_discharge and q_ref arrays aligned to dataset metas."""

    q_true: List[float] = []
    q_ref: List[float] = []
    for meta in metas:
        key = (str(meta.policy), str(meta.cell_code), int(meta.cycles))
        payload = label_lookup.get(key)
        if payload is None:
            raise RuntimeError(f"Missing label lookup for key={key}.")
        q_true.append(float(payload["q_discharge"]))
        q_ref.append(float(payload["q_ref"]))
    return np.asarray(q_true, dtype=np.float32), np.asarray(q_ref, dtype=np.float32)

def build_report(
    args: argparse.Namespace,
    device: torch.device,
    merged_rows: int,
    train_rows: int,
    valid_rows: int,
    best_epoch: int,
    metrics_df: pd.DataFrame,
) -> str:
    """Build markdown report for dQdV retention training."""

    lines: List[str] = []
    lines.append("# LSTM 训练报告：dQdV 主峰特征拟合容量保持率")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 设备：`{device.type}`")
    lines.append(f"- 序列模式：`{args.sequence_mode}`")
    lines.append(f"- 特征包：`{args.feature_pack}`")
    lines.append(f"- q 绝对过滤：`{args.q_min} <= q_discharge <= {args.q_max}`")
    lines.append(
        "- retention 过滤："
        f"`{args.retention_min} <= retention <= {args.retention_max}`，"
        f"`q_ref`=前 `{args.q_ref_cycles}` 个有效循环中位数"
    )
    lines.append(f"- checkpoint 快照间隔：每 `{args.checkpoint_snapshot_interval}` 轮")
    lines.append("")
    lines.append("## 2. 数据概览")
    lines.append(f"- 合并后 cycle 级样本数：**{merged_rows:,}**")
    lines.append(f"- 训练样本数：**{train_rows:,}**")
    lines.append(f"- 验证样本数：**{valid_rows:,}**")
    lines.append("- 每个时间步输入维度：`10`（主峰9维 + cycle_index_norm）")
    lines.append("")
    lines.append("## 3. 指标结果")
    lines.append("| target | set_type | n_samples | MSE | RMSE | MAE | R2 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in metrics_df.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.set_type} | {int(row.n_windows)} | "
            f"{float(row.mse):.8f} | {float(row.rmse):.6f} | {float(row.mae):.6f} | {float(row.r2):.6f} |"
        )
    lines.append("")
    lines.append("## 4. 图表")
    lines.append(f"- 最佳 epoch：**{best_epoch}**")
    lines.append("![loss_curve](./loss_curve.png)")
    lines.append("")
    lines.append("![valid_scatter](./valid_scatter.png)")
    return "\n".join(lines)


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Run full training pipeline and return summary fields."""

    if args.smoke_test:
        args.epochs = min(int(args.epochs), 3)
        args.patience = min(int(args.patience), 2)
        args.max_train_windows = int(args.max_train_windows) if int(args.max_train_windows) > 0 else 2048
        args.max_valid_windows = int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else 1024
        args.checkpoint_snapshot_interval = min(int(args.checkpoint_snapshot_interval), 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_lstm.ensure_matplotlib_backend()
    base_lstm.set_seed(int(args.seed))
    device = base_lstm.resolve_device(args.device)

    if str(args.sequence_mode) != "prefix_full":
        raise RuntimeError("This script only supports --sequence-mode prefix_full.")

    split_df = base_lstm.load_split_map(args.train_split_path, args.valid_split_path)
    feature_df = load_dqdv_main_feature_table(args.dqdv_path)
    label_df = load_retention_labels(
        life_path=args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
    )
    merged = merge_feature_label_split(feature_df=feature_df, label_df=label_df, split_df=split_df)
    merged = add_cycle_index_norm(merged)
    merged = coerce_feature_columns(merged, MODEL_FEATURE_COLUMNS)
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after dQdV/retention/split join.")

    seq_map = build_sequences(merged=merged, feature_cols=MODEL_FEATURE_COLUMNS)
    train_seq_map, valid_seq_map = base_lstm.split_sequence_dict(seq_map)
    train_dataset = base_lstm.PrefixHistoryDataset(
        sequences=train_seq_map,
        max_windows=int(args.max_train_windows) if int(args.max_train_windows) > 0 else None,
        seed=int(args.seed),
    )
    valid_dataset = base_lstm.PrefixHistoryDataset(
        sequences=valid_seq_map,
        max_windows=int(args.max_valid_windows) if int(args.max_valid_windows) > 0 else None,
        seed=int(args.seed) + 1,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Train or valid samples are empty after dataset construction.")

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

    model = base_lstm.LSTMRegressor(
        input_size=len(MODEL_FEATURE_COLUMNS),
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

    best_ckpt_path = args.output_dir / str(args.best_state_file)
    latest_state_path = args.output_dir / str(args.latest_state_file)
    epoch_log_path = args.output_dir / str(args.epoch_log_file)
    status_path = args.output_dir / str(args.status_file)
    snapshot_dir = args.output_dir / "snapshots"
    out_metrics = args.output_dir / "train_valid_metrics.csv"
    out_preds = args.output_dir / "valid_predictions.csv"
    out_loss = args.output_dir / "loss_curve.csv"
    out_loss_png = args.output_dir / "loss_curve.png"
    out_scatter_png = args.output_dir / "valid_scatter.png"
    out_checks = args.output_dir / "dataset_checks.csv"
    out_config = args.output_dir / "run_config.json"
    out_report = args.output_dir / "lstm_dqdv_retention_report.md"

    signature_payload = build_resume_signature_payload(args=args, device=device)
    args_signature = build_resume_signature(payload=signature_payload)
    start_epoch = 1
    best_valid_loss = float("inf")
    best_epoch = 0
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

    if start_epoch > int(args.epochs):
        print(
            f"Latest state epoch={start_epoch - 1} already reached target epochs={args.epochs}, "
            "skip additional training.",
            flush=True,
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_loss = base_lstm.train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = base_lstm.eval_loss(model, valid_loader, criterion, device)
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
                        "input_size": len(MODEL_FEATURE_COLUMNS),
                        "hidden_size": int(args.hidden_size),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                        "sequence_mode": "prefix_full",
                        "feature_pack": str(args.feature_pack),
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

        latest_payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": int(best_epoch),
            "best_valid_loss": float(best_valid_loss),
            "no_improve_count": int(no_improve_count),
            "args_signature": args_signature,
            "signature_payload": signature_payload,
            "loss_rows": list(loss_rows),
        }
        base_lstm.atomic_torch_save(latest_payload, path=latest_state_path)
        save_snapshot_if_needed(
            snapshot_dir=snapshot_dir,
            epoch=int(epoch),
            interval=int(args.checkpoint_snapshot_interval),
            payload=latest_payload,
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

    train_pred_loader = base_lstm.build_dataloader(
        dataset=train_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        device=device,
        collate_fn=base_lstm.collate_sequence_batch,
    )
    y_train_true_ret, y_train_pred_ret, train_idx = base_lstm.predict_loader(
        model=model,
        loader=train_pred_loader,
        device=device,
    )
    y_valid_true_ret, y_valid_pred_ret, valid_idx = base_lstm.predict_loader(
        model=model,
        loader=valid_loader,
        device=device,
    )
    train_metrics_ret = base_lstm.calc_metrics(y_true=y_train_true_ret, y_pred=y_train_pred_ret, set_type="train")
    valid_metrics_ret = base_lstm.calc_metrics(y_true=y_valid_true_ret, y_pred=y_valid_pred_ret, set_type="valid")

    label_lookup = build_label_lookup(merged)
    train_metas = [train_dataset.metas[int(i)] for i in train_idx.tolist()]
    valid_metas = [valid_dataset.metas[int(i)] for i in valid_idx.tolist()]
    train_q_true, train_q_ref = build_q_arrays_from_metas(train_metas, label_lookup=label_lookup)
    valid_q_true, valid_q_ref = build_q_arrays_from_metas(valid_metas, label_lookup=label_lookup)
    train_q_pred = y_train_pred_ret.astype(np.float32) * train_q_ref.astype(np.float32)
    valid_q_pred = y_valid_pred_ret.astype(np.float32) * valid_q_ref.astype(np.float32)
    train_metrics_q = base_lstm.calc_metrics(y_true=train_q_true, y_pred=train_q_pred, set_type="train")
    valid_metrics_q = base_lstm.calc_metrics(y_true=valid_q_true, y_pred=valid_q_pred, set_type="valid")

    metrics_df = pd.DataFrame(
        [
            {"target": "retention", **asdict(train_metrics_ret)},
            {"target": "retention", **asdict(valid_metrics_ret)},
            {"target": "q_discharge", **asdict(train_metrics_q)},
            {"target": "q_discharge", **asdict(valid_metrics_q)},
        ]
    )

    valid_pred_df = pd.DataFrame(
        {
            "policy": [m.policy for m in valid_metas],
            "cell_code": [m.cell_code for m in valid_metas],
            "cycles": [m.cycles for m in valid_metas],
            "q_discharge": valid_q_true.astype(float),
            "q_ref": valid_q_ref.astype(float),
            "retention_true": y_valid_true_ret.astype(float),
            "pred_retention": y_valid_pred_ret.astype(float),
        }
    )
    valid_pred_df["pred_q_discharge"] = valid_pred_df["pred_retention"] * valid_pred_df["q_ref"]
    valid_pred_df["residual_retention"] = valid_pred_df["retention_true"] - valid_pred_df["pred_retention"]
    valid_pred_df["residual_q_discharge"] = valid_pred_df["q_discharge"] - valid_pred_df["pred_q_discharge"]
    valid_pred_df = valid_pred_df.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").reset_index(
        drop=True
    )

    loss_df = pd.DataFrame(loss_rows)
    checks_df = build_dataset_checks(merged=merged, feature_cols=MODEL_FEATURE_COLUMNS)
    run_config = {
        "script": str(SCRIPT_PATH),
        "python_executable": os.path.realpath(os.sys.executable),
        "device": str(device),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resume_start_epoch": int(start_epoch),
        "args_signature": args_signature,
        "signature_payload": signature_payload,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_text = build_report(
        args=args,
        device=device,
        merged_rows=int(len(merged)),
        train_rows=int(len(train_dataset)),
        valid_rows=int(len(valid_dataset)),
        best_epoch=int(best_epoch),
        metrics_df=metrics_df,
    )

    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8")
    valid_pred_df.to_csv(out_preds, index=False, encoding="utf-8")
    loss_df.to_csv(out_loss, index=False, encoding="utf-8")
    checks_df.to_csv(out_checks, index=False, encoding="utf-8")
    out_config.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    base_lstm.save_loss_plot(loss_df=loss_df, out_path=out_loss_png)
    base_lstm.save_scatter_plot(valid_pred_df=valid_pred_df, metrics=valid_metrics_q, out_path=out_scatter_png)
    out_report.write_text(report_text, encoding="utf-8")
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
            "valid_metrics_retention": asdict(valid_metrics_ret),
            "valid_metrics_q_discharge": asdict(valid_metrics_q),
            "train_metrics_retention": asdict(train_metrics_ret),
            "train_metrics_q_discharge": asdict(train_metrics_q),
        },
    )

    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_preds}")
    print(f"Saved: {out_loss}")
    print(f"Saved: {out_loss_png}")
    print(f"Saved: {out_scatter_png}")
    print(f"Saved: {best_ckpt_path}")
    print(f"Saved: {latest_state_path}")
    print(f"Saved: {out_checks}")
    print(f"Saved: {out_config}")
    print(f"Saved: {out_report}")
    print(
        "Done:",
        f"Train/Valid samples={len(train_dataset)}/{len(valid_dataset)} | "
        f"Valid retention R2={valid_metrics_ret.r2:.6f} | "
        f"Valid q_discharge R2={valid_metrics_q.r2:.6f}",
    )

    return {
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "valid_retention_r2": float(valid_metrics_ret.r2),
        "valid_q_discharge_r2": float(valid_metrics_q.r2),
        "output_dir": str(args.output_dir),
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_state_path),
    }


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    run_training(args=args)


if __name__ == "__main__":
    main()
