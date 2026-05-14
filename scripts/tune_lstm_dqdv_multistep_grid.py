from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
sys.path.append(str(REPO_ROOT))

import scripts.train_lstm_dqdv_multistep as train_mod


@dataclass(frozen=True)
class TrialConfig:
    """One multistep grid-search trial."""

    trial_id: int
    hidden_size: int
    learning_rate: float
    num_layers: int
    dropout: float


def parse_args() -> argparse.Namespace:
    """Parse CLI args for multistep dQ/dV grid tuning."""

    parser = argparse.ArgumentParser(
        description="Grid tuning and full-refresh training for dQ/dV multistep retention prediction."
    )
    parser.add_argument("--dqdv-path", type=Path, default=REPO_ROOT / "data" / "processed" / "discharge_dqdv_peak_features_skill_full.csv")
    parser.add_argument("--life-path", type=Path, default=REPO_ROOT / "data" / "processed" / "life_performance.csv")
    parser.add_argument("--train-split-path", type=Path, default=REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv")
    parser.add_argument("--valid-split-path", type=Path, default=REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis" / "lstm_dqdv_multistep_h50_grid")
    parser.add_argument(
        "--feature-pack",
        type=str,
        choices=sorted(train_mod.FEATURE_PACK_COLUMNS),
        default="compact_peak_shape_height",
    )
    parser.add_argument("--horizon-steps", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=30)
    parser.add_argument("--cycle-log-scale", type=float, default=3000.0)
    parser.add_argument("--hidden-sizes", type=str, default="64,128,192")
    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4")
    parser.add_argument("--num-layers-list", type=str, default="1,2")
    parser.add_argument("--dropout-list", type=str, default="0.1,0.2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--short-horizon-count", type=int, default=10)
    parser.add_argument("--short-horizon-weight", type=float, default=3.0)
    parser.add_argument("--short-bias-penalty", type=float, default=0.25)
    parser.add_argument("--short-underprediction-penalty", type=float, default=0.0)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--q-ref-cycles", type=int, default=5)
    parser.add_argument("--retention-min", type=float, default=0.3)
    parser.add_argument("--retention-max", type=float, default=1.1)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--max-train-windows", type=int, default=20000)
    parser.add_argument("--max-valid-windows", type=int, default=8000)
    parser.add_argument("--resume-existing", dest="resume_existing", action="store_true", default=True)
    parser.add_argument("--no-resume-existing", dest="resume_existing", action="store_false")
    parser.add_argument("--resume-interrupted", dest="resume_interrupted", action="store_true", default=True)
    parser.add_argument("--no-resume-interrupted", dest="resume_interrupted", action="store_false")
    parser.add_argument("--resume-from-partial", dest="resume_from_partial", action="store_true", default=True)
    parser.add_argument("--no-resume-from-partial", dest="resume_from_partial", action="store_false")
    parser.add_argument("--save-partial-results", dest="save_partial_results", action="store_true", default=True)
    parser.add_argument("--no-save-partial-results", dest="save_partial_results", action="store_false")
    parser.add_argument("--partial-results-file", type=str, default="grid_search_results.partial.csv")
    parser.add_argument("--runtime-status-file", type=str, default="grid_tuning_runtime_status.json")
    parser.add_argument("--run-full-refresh", dest="run_full_refresh", action="store_true", default=True)
    parser.add_argument("--skip-full-refresh", dest="run_full_refresh", action="store_false")
    parser.add_argument("--full-refresh-output-dir", type=Path, default=None)
    parser.add_argument("--full-refresh-epochs", type=int, default=100)
    parser.add_argument("--full-refresh-patience", type=int, default=15)
    parser.add_argument("--full-refresh-min-delta", type=float, default=1e-5)
    parser.add_argument("--full-refresh-batch-size", type=int, default=256)
    parser.add_argument("--full-refresh-resume-interrupted", dest="full_refresh_resume_interrupted", action="store_true", default=True)
    parser.add_argument("--no-full-refresh-resume-interrupted", dest="full_refresh_resume_interrupted", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated integer list."""

    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    """Parse a comma-separated float list."""

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def prepare_trial_grid(args: argparse.Namespace) -> List[TrialConfig]:
    """Create Cartesian-product trial configs."""

    trials: List[TrialConfig] = []
    trial_id = 0
    for hidden_size in parse_int_list(args.hidden_sizes):
        for learning_rate in parse_float_list(args.learning_rates):
            for num_layers in parse_int_list(args.num_layers_list):
                for dropout in parse_float_list(args.dropout_list):
                    trial_id += 1
                    trials.append(TrialConfig(trial_id, hidden_size, learning_rate, num_layers, dropout))
    return trials


def atomic_write_text(path: Path, content: str) -> None:
    """Write a text file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_status(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist grid runtime status."""

    status = dict(payload)
    status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_text(path, json.dumps(status, ensure_ascii=False, indent=2))


def rows_to_df(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert trial rows to a stable sorted DataFrame."""

    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows]).sort_values("trial_id").reset_index(drop=True)


def load_partial_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load partial trial results keyed by trial_id."""

    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return {int(row["trial_id"]): dict(row) for row in df.to_dict(orient="records")}


def save_partial(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Save partial trial rows atomically."""

    df = rows_to_df(rows)
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def build_train_args(args: argparse.Namespace, cfg: TrialConfig, output_dir: Path, full_refresh: bool = False) -> argparse.Namespace:
    """Build an argparse namespace accepted by train_lstm_dqdv_multistep.run_training."""

    if full_refresh:
        epochs = int(args.full_refresh_epochs)
        patience = int(args.full_refresh_patience)
        min_delta = float(args.full_refresh_min_delta)
        batch_size = int(args.full_refresh_batch_size)
        max_train_windows = 0
        max_valid_windows = 0
        resume_interrupted = bool(args.full_refresh_resume_interrupted)
        seed = int(args.seed)
    else:
        epochs = int(args.epochs)
        patience = int(args.patience)
        min_delta = float(args.min_delta)
        batch_size = int(args.batch_size)
        max_train_windows = int(args.max_train_windows)
        max_valid_windows = int(args.max_valid_windows)
        resume_interrupted = bool(args.resume_interrupted)
        seed = int(args.seed) + int(cfg.trial_id)

    return argparse.Namespace(
        dqdv_path=args.dqdv_path,
        life_path=args.life_path,
        train_split_path=args.train_split_path,
        valid_split_path=args.valid_split_path,
        output_dir=output_dir,
        feature_pack=str(args.feature_pack),
        horizon_steps=int(args.horizon_steps),
        min_history=int(args.min_history),
        cycle_log_scale=float(args.cycle_log_scale),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_size=int(cfg.hidden_size),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        short_horizon_count=int(args.short_horizon_count),
        short_horizon_weight=float(args.short_horizon_weight),
        short_bias_penalty=float(args.short_bias_penalty),
        short_underprediction_penalty=float(args.short_underprediction_penalty),
        patience=patience,
        min_delta=min_delta,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
        q_ref_cycles=int(args.q_ref_cycles),
        retention_min=float(args.retention_min),
        retention_max=float(args.retention_max),
        device=str(args.device),
        num_workers=int(args.num_workers),
        seed=seed,
        max_train_windows=max_train_windows,
        max_valid_windows=max_valid_windows,
        resume_interrupted=resume_interrupted,
        best_state_file="best.pt",
        latest_state_file="latest.pt",
        epoch_log_file="epoch_log.csv",
        status_file="runtime_status.json",
        smoke_test=bool(args.smoke_test),
    )


def metric_value(metrics_df: pd.DataFrame, target: str, horizon: str | int, metric: str) -> float:
    """Read one valid weighted dQ/dV-LSTM metric from the metrics table."""

    mask = (
        metrics_df["target"].astype(str).eq(str(target))
        & metrics_df["method"].astype(str).eq("dqdv_multistep_lstm")
        & metrics_df["set_type"].astype(str).eq("valid")
        & metrics_df["aggregation"].astype(str).eq("weighted")
        & metrics_df["horizon"].astype(str).eq(str(horizon))
    )
    subset = metrics_df.loc[mask]
    if subset.empty or metric not in subset.columns:
        return float("nan")
    return float(subset.iloc[0][metric])


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object or return an empty dict."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_checks_pass(trial_dir: Path) -> int:
    """Return 1 when all dataset checks passed."""

    path = trial_dir / "dataset_checks.csv"
    if not path.exists():
        return 0
    checks = pd.read_csv(path)
    if checks.empty or "pass_flag" not in checks.columns:
        return 0
    return int((checks["pass_flag"].astype(int) == 1).all())


def existing_trial_matches(trial_dir: Path, args: argparse.Namespace, cfg: TrialConfig) -> bool:
    """Check whether existing trial outputs correspond to the requested config."""

    run_config = load_json(trial_dir / "run_config.json")
    saved_args = run_config.get("args", {}) if isinstance(run_config, Mapping) else {}
    if not isinstance(saved_args, Mapping):
        return False
    expected: Dict[str, Any] = {
        "feature_pack": str(args.feature_pack),
        "horizon_steps": int(args.horizon_steps),
        "min_history": int(args.min_history),
        "hidden_size": int(cfg.hidden_size),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "learning_rate": float(cfg.learning_rate),
        "short_horizon_count": int(args.short_horizon_count),
        "short_horizon_weight": float(args.short_horizon_weight),
        "short_bias_penalty": float(args.short_bias_penalty),
        "short_underprediction_penalty": float(args.short_underprediction_penalty),
        "max_train_windows": int(args.max_train_windows),
        "max_valid_windows": int(args.max_valid_windows),
    }
    for key, value in expected.items():
        if key not in saved_args:
            return False
        saved = saved_args[key]
        if isinstance(value, float):
            if not np.isclose(float(saved), value):
                return False
        elif str(saved) != str(value):
            return False
    return True


def build_result_row(cfg: TrialConfig, trial_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Build one grid result row from a trial output directory."""

    metrics_path = trial_dir / "train_valid_metrics_by_horizon.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    run_config = load_json(trial_dir / "run_config.json")
    feature_cols = train_mod.get_feature_columns(str(args.feature_pack))
    row: Dict[str, Any] = {
        "trial_id": int(cfg.trial_id),
        "feature_pack": str(args.feature_pack),
        "horizon_steps": int(args.horizon_steps),
        "min_history": int(args.min_history),
        "input_size": int(len(feature_cols)),
        "feature_columns": ",".join(feature_cols),
        "hidden_size": int(cfg.hidden_size),
        "learning_rate": float(cfg.learning_rate),
        "num_layers": int(cfg.num_layers),
        "dropout": float(cfg.dropout),
        "short_horizon_count": int(args.short_horizon_count),
        "short_horizon_weight": float(args.short_horizon_weight),
        "short_bias_penalty": float(args.short_bias_penalty),
        "short_underprediction_penalty": float(args.short_underprediction_penalty),
        "best_epoch": int(run_config.get("best_epoch", 0)),
        "best_valid_loss": float(run_config.get("best_valid_loss", np.nan)),
        "train_windows": int(run_config.get("train_windows", 0)),
        "valid_windows": int(run_config.get("valid_windows", 0)),
        "dataset_checks_all_pass": dataset_checks_pass(trial_dir),
        "output_dir": str(trial_dir),
        "checkpoint_path": str(trial_dir / "best.pt"),
    }
    for target in ["retention", "q_discharge"]:
        prefix = "valid_q_discharge" if target == "q_discharge" else "valid_retention"
        for metric in ["mse", "rmse", "mae", "r2"]:
            row[f"{prefix}_{metric}"] = metric_value(metrics_df, target, "all", metric)
        for horizon in [1, 5, 10, 20, int(args.horizon_steps)]:
            row[f"{prefix}_h{horizon}_rmse"] = metric_value(metrics_df, target, horizon, "rmse")
            row[f"{prefix}_h{horizon}_r2"] = metric_value(metrics_df, target, horizon, "r2")
    return row


def run_or_backfill_trial(cfg: TrialConfig, args: argparse.Namespace, ckpt_root: Path) -> Dict[str, Any]:
    """Run one trial or backfill its row from matching existing outputs."""

    trial_dir = ckpt_root / f"trial_{int(cfg.trial_id):03d}"
    metrics_path = trial_dir / "train_valid_metrics_by_horizon.csv"
    if bool(args.resume_existing) and metrics_path.exists() and existing_trial_matches(trial_dir, args, cfg):
        print(f"[Trial {cfg.trial_id}] backfill from existing metrics: {trial_dir}", flush=True)
        return build_result_row(cfg=cfg, trial_dir=trial_dir, args=args)

    print(
        f"[Trial {cfg.trial_id}] train hidden={cfg.hidden_size}, lr={cfg.learning_rate}, "
        f"layers={cfg.num_layers}, dropout={cfg.dropout}",
        flush=True,
    )
    train_args = build_train_args(args=args, cfg=cfg, output_dir=trial_dir, full_refresh=False)
    train_mod.run_training(train_args)
    return build_result_row(cfg=cfg, trial_dir=trial_dir, args=args)


def best_result_row(results_df: pd.DataFrame) -> pd.Series:
    """Select the best trial by valid weighted retention all-window RMSE."""

    ranked = results_df.copy()
    ranked["valid_retention_rmse"] = pd.to_numeric(ranked["valid_retention_rmse"], errors="coerce")
    ranked["valid_retention_r2"] = pd.to_numeric(ranked["valid_retention_r2"], errors="coerce")
    ranked = ranked.sort_values(["valid_retention_rmse", "valid_retention_r2"], ascending=[True, False])
    if ranked.empty or not np.isfinite(float(ranked.iloc[0]["valid_retention_rmse"])):
        raise RuntimeError("No finite valid_retention_rmse found for best-trial selection.")
    return ranked.iloc[0]


def save_tuning_plot(results_df: pd.DataFrame, out_path: Path) -> None:
    """Save a compact trial comparison plot."""

    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = results_df.sort_values("trial_id")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(plot_df["trial_id"], plot_df["valid_retention_rmse"], marker="o", label="retention RMSE")
    ax1.set_xlabel("trial_id")
    ax1.set_ylabel("valid retention RMSE")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(plot_df["trial_id"], plot_df["valid_retention_r2"], marker="s", color="tab:green", label="retention R2")
    ax2.set_ylabel("valid retention R2")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional dependencies."""

    if df.empty:
        return "(empty)"
    cols = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values: List[str] = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                value_text = "nan" if not np.isfinite(value) else f"{value:.8g}"
            else:
                value_text = str(value)
            values.append(value_text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def build_report(results_df: pd.DataFrame, best_row: Mapping[str, Any], args: argparse.Namespace) -> str:
    """Build a Markdown report for the grid run."""

    top = results_df.sort_values("valid_retention_rmse").head(10)
    cols = [
        "trial_id",
        "hidden_size",
        "learning_rate",
        "num_layers",
        "dropout",
        "short_horizon_count",
        "short_horizon_weight",
        "short_bias_penalty",
        "short_underprediction_penalty",
        "valid_retention_rmse",
        "valid_retention_r2",
        "valid_q_discharge_rmse",
        "dataset_checks_all_pass",
    ]
    lines = [
        "# dQ/dV Multistep Grid Tuning Report",
        "",
        f"- feature_pack: `{args.feature_pack}`",
        f"- horizon_steps: `{args.horizon_steps}`",
        f"- min_history: `{args.min_history}`",
        (
            f"- short horizon loss: first `{args.short_horizon_count}` steps weight "
            f"`{args.short_horizon_weight}`, bias penalty `{args.short_bias_penalty}`, "
            f"underprediction penalty `{args.short_underprediction_penalty}`"
        ),
        "- selection: valid weighted retention all-window RMSE (lower is better)",
        "",
        "## Best Trial",
        "",
        f"- trial_id: `{int(best_row['trial_id'])}`",
        f"- hidden_size: `{int(best_row['hidden_size'])}`",
        f"- learning_rate: `{float(best_row['learning_rate'])}`",
        f"- num_layers: `{int(best_row['num_layers'])}`",
        f"- dropout: `{float(best_row['dropout'])}`",
        f"- short_horizon_count: `{int(best_row['short_horizon_count'])}`",
        f"- short_horizon_weight: `{float(best_row['short_horizon_weight'])}`",
        f"- short_bias_penalty: `{float(best_row['short_bias_penalty'])}`",
        f"- short_underprediction_penalty: `{float(best_row['short_underprediction_penalty'])}`",
        f"- valid_retention_rmse: `{float(best_row['valid_retention_rmse']):.8f}`",
        f"- valid_retention_r2: `{float(best_row['valid_retention_r2']):.8f}`",
        "",
        "## Top Trials",
        "",
        dataframe_to_markdown(top[cols]),
        "",
    ]
    return "\n".join(lines)


def save_grid_outputs(results_df: pd.DataFrame, args: argparse.Namespace) -> Path:
    """Save complete grid outputs and return best-config path."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_results = args.output_dir / "grid_search_results.csv"
    out_best = args.output_dir / "best_grid_config.json"
    out_plot = args.output_dir / "grid_tuning_scatter.png"
    out_report = args.output_dir / "lstm_dqdv_multistep_grid_report.md"
    best_row = best_result_row(results_df)
    results_df.to_csv(out_results, index=False, encoding="utf-8")
    save_tuning_plot(results_df, out_plot)
    out_report.write_text(build_report(results_df, best_row, args), encoding="utf-8")
    best_payload = {
        "selection_metric": "valid_retention_rmse",
        "selection_scope": {
            "target": "retention",
            "method": "dqdv_multistep_lstm",
            "set_type": "valid",
            "aggregation": "weighted",
            "horizon": "all",
        },
        "trial_id": int(best_row["trial_id"]),
        "feature_pack": str(best_row["feature_pack"]),
        "horizon_steps": int(best_row["horizon_steps"]),
        "min_history": int(best_row["min_history"]),
        "hidden_size": int(best_row["hidden_size"]),
        "learning_rate": float(best_row["learning_rate"]),
        "num_layers": int(best_row["num_layers"]),
        "dropout": float(best_row["dropout"]),
        "short_horizon_count": int(best_row["short_horizon_count"]),
        "short_horizon_weight": float(best_row["short_horizon_weight"]),
        "short_bias_penalty": float(best_row["short_bias_penalty"]),
        "short_underprediction_penalty": float(best_row["short_underprediction_penalty"]),
        "valid_retention_rmse": float(best_row["valid_retention_rmse"]),
        "valid_retention_r2": float(best_row["valid_retention_r2"]),
        "valid_q_discharge_rmse": float(best_row["valid_q_discharge_rmse"]),
        "checkpoint_path": str(best_row["checkpoint_path"]),
        "output_dir": str(best_row["output_dir"]),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_text(out_best, json.dumps(best_payload, ensure_ascii=False, indent=2))
    print(f"Saved: {out_results}", flush=True)
    print(f"Saved: {out_best}", flush=True)
    print(f"Saved: {out_plot}", flush=True)
    print(f"Saved: {out_report}", flush=True)
    return out_best


def run_full_refresh_from_best(args: argparse.Namespace, best_config_path: Path) -> Dict[str, Any]:
    """Run full-window training from best_grid_config.json."""

    best_cfg = json.loads(best_config_path.read_text(encoding="utf-8"))
    cfg = TrialConfig(
        trial_id=int(best_cfg["trial_id"]),
        hidden_size=int(best_cfg["hidden_size"]),
        learning_rate=float(best_cfg["learning_rate"]),
        num_layers=int(best_cfg["num_layers"]),
        dropout=float(best_cfg["dropout"]),
    )
    full_dir = args.full_refresh_output_dir or (args.output_dir / "full_refresh")
    print(f"[Stage-2] full refresh -> {full_dir}", flush=True)
    train_args = build_train_args(args=args, cfg=cfg, output_dir=full_dir, full_refresh=True)
    return train_mod.run_training(train_args)


def main() -> None:
    """Run grid tuning and optional best-parameter full refresh."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_mod.base_lstm.ensure_matplotlib_backend()
    status_path = args.output_dir / str(args.runtime_status_file)
    partial_path = args.output_dir / str(args.partial_results_file)
    ckpt_root = args.output_dir / "checkpoints"
    trials = prepare_trial_grid(args)
    partial_rows = load_partial_rows(partial_path) if bool(args.resume_from_partial) else {}
    rows: List[Dict[str, Any]] = []

    write_status(status_path, {"stage": "grid", "finished": False, "total_trials": len(trials), "completed_trials": len(partial_rows)})
    for cfg in trials:
        if bool(args.resume_from_partial) and int(cfg.trial_id) in partial_rows:
            print(f"[Trial {cfg.trial_id}/{len(trials)}] skip from partial CSV", flush=True)
            rows.append(partial_rows[int(cfg.trial_id)])
            continue
        write_status(status_path, {"stage": "grid", "finished": False, "current_trial_id": int(cfg.trial_id), "total_trials": len(trials), "completed_trials": len(rows)})
        row = run_or_backfill_trial(cfg=cfg, args=args, ckpt_root=ckpt_root)
        rows.append(row)
        if bool(args.save_partial_results):
            save_partial(rows, partial_path)
        print(f"[Trial {cfg.trial_id}] valid_retention_rmse={float(row['valid_retention_rmse']):.8f}", flush=True)

    if len(rows) != len(trials):
        raise RuntimeError(f"Incomplete grid rows: got {len(rows)}, expected {len(trials)}.")
    results_df = rows_to_df(rows)
    best_path = save_grid_outputs(results_df, args)
    full_summary: Optional[Dict[str, Any]] = None
    if bool(args.run_full_refresh):
        full_summary = run_full_refresh_from_best(args=args, best_config_path=best_path)
    write_status(
        status_path,
        {
            "stage": "done",
            "finished": True,
            "total_trials": len(trials),
            "completed_trials": len(rows),
            "best_config_path": str(best_path),
            "full_refresh_summary": full_summary,
        },
    )


if __name__ == "__main__":
    main()


