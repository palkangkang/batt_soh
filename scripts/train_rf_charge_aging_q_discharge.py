from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
ENCODING = "utf-8-sig"
N_CROSS_BINS = 60
POLICY_COLS = ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]
FEATURE_PACKS = ["F0_60cum_plus_policy", "F1_F0_plus_cycles", "F2_F1_plus_60inc_plus_stats"]


@dataclass
class Metrics:
    """Container for regression metrics."""

    model_name: str
    set_type: str
    n_rows: int
    mae: float
    rmse: float
    mse: float
    r2: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="RF tuning with F0/F1/F2 feature packs.")
    parser.add_argument("--timeseries-path", type=Path, default=REPO_ROOT / "data" / "processed" / "charge_aging_path_timeseries.csv")
    parser.add_argument("--life-path", type=Path, default=REPO_ROOT / "data" / "processed" / "life_performance.csv")
    parser.add_argument("--train-split-path", type=Path, default=REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv")
    parser.add_argument("--valid-split-path", type=Path, default=REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "analysis" / "rf_charge_aging_q_discharge")
    parser.add_argument("--label-missing-strategy", type=str, choices=["drop", "zero"], default="drop")
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--target-r2", type=float, default=0.9)
    parser.add_argument("--stage-a-trials", type=int, default=40)
    parser.add_argument("--stage-b-trials", type=int, default=20)
    parser.add_argument("--tune-max-rows", type=int, default=20000)
    parser.add_argument("--tune-valid-group-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=20260401)
    return parser.parse_args()


def ensure_matplotlib_config() -> List[str]:
    """Configure matplotlib backend and font fallback list."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams  # noqa: WPS433

    candidates = ["Noto Sans CJK SC", "DejaVu Sans"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = [font for font in candidates if font in installed] or ["DejaVu Sans"]
    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    return selected


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    """Deduplicate while preserving order."""

    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_split(train_path: Path, valid_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load split tables and make split_map."""

    train = pd.read_csv(train_path, encoding=ENCODING)
    valid = pd.read_csv(valid_path, encoding=ENCODING)
    cols = ["policy", "cell_code", *POLICY_COLS]
    train = train[cols].copy()
    valid = valid[cols].copy()
    for df in [train, valid]:
        df["policy"] = df["policy"].astype(str)
        df["cell_code"] = df["cell_code"].astype(str)
        for col in POLICY_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    split_map = pd.concat([train, valid], ignore_index=True).drop_duplicates(["policy", "cell_code"], keep="first")

    overlap = len(set(train["policy"] + "||" + train["cell_code"]).intersection(set(valid["policy"] + "||" + valid["cell_code"])))
    if overlap > 0:
        raise RuntimeError(f"split leakage detected: {overlap}")
    return train, valid, split_map


def load_feature_table(timeseries_path: Path) -> Tuple[pd.DataFrame, Dict[str, int], List[str], List[str], List[str]]:
    """Build cycle-level feature table from long 60-bin rows."""

    usecols = [
        "policy",
        "cell_code",
        "cycles",
        "cross_bin",
        "cycle_charge_time_h",
        "cumulative_charge_time_h",
        "is_abnormal_cell",
    ]
    idx_cols = ["policy", "cell_code", "cycles", "is_abnormal_cell"]
    cum_cols = [f"cross_bin_cum_{i:02d}_h" for i in range(1, 61)]
    inc_cols = [f"cross_bin_inc_{i:02d}_h" for i in range(1, 61)]

    part_features: List[pd.DataFrame] = []
    part_counts: List[pd.DataFrame] = []
    total_rows_after_clean = 0

    reader = pd.read_csv(
        timeseries_path,
        usecols=usecols,
        encoding="utf-8",
        chunksize=30000,
        engine="python",
    )
    for chunk in reader:
        chunk["policy"] = chunk["policy"].astype(str)
        chunk["cell_code"] = chunk["cell_code"].astype(str)
        for col in ["cycles", "cross_bin", "cycle_charge_time_h", "cumulative_charge_time_h", "is_abnormal_cell"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=usecols).copy()
        chunk["cycles"] = chunk["cycles"].astype(int)
        chunk["cross_bin"] = chunk["cross_bin"].astype(int)
        chunk["is_abnormal_cell"] = chunk["is_abnormal_cell"].astype(int)
        chunk = chunk[(chunk["cross_bin"] >= 1) & (chunk["cross_bin"] <= N_CROSS_BINS)].copy()
        if chunk.empty:
            continue
        total_rows_after_clean += int(len(chunk))

        chunk = (
            chunk.groupby(["policy", "cell_code", "cycles", "cross_bin"], as_index=False)
            .agg(
                cycle_charge_time_h=("cycle_charge_time_h", "sum"),
                cumulative_charge_time_h=("cumulative_charge_time_h", "max"),
                is_abnormal_cell=("is_abnormal_cell", "max"),
            )
        )
        part_counts.append(
            chunk.groupby(["policy", "cell_code", "cycles"], as_index=False)["cross_bin"]
            .nunique()
            .rename(columns={"cross_bin": "cross_bin_count"})
        )

        cum = (
            chunk.pivot_table(
                index=idx_cols,
                columns="cross_bin",
                values="cumulative_charge_time_h",
                aggfunc="max",
                fill_value=0.0,
            )
            .reindex(columns=list(range(1, 61)), fill_value=0.0)
            .reset_index()
            .rename(columns={i: cum_cols[i - 1] for i in range(1, 61)})
        )
        inc = (
            chunk.pivot_table(
                index=idx_cols,
                columns="cross_bin",
                values="cycle_charge_time_h",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(columns=list(range(1, 61)), fill_value=0.0)
            .reset_index()
            .rename(columns={i: inc_cols[i - 1] for i in range(1, 61)})
        )
        part_features.append(cum.merge(inc, on=idx_cols, how="inner", validate="one_to_one"))

    if not part_features:
        raise RuntimeError("No valid rows found in timeseries file.")

    feat = pd.concat(part_features, ignore_index=True)
    agg_map = {**{col: "max" for col in cum_cols}, **{col: "sum" for col in inc_cols}}
    feat = feat.groupby(idx_cols, as_index=False).agg(agg_map)

    cnt = pd.concat(part_counts, ignore_index=True)
    cnt = (
        cnt.groupby(["policy", "cell_code", "cycles"], as_index=False)["cross_bin_count"]
        .sum()
    )
    incomplete = int((cnt["cross_bin_count"] < N_CROSS_BINS).sum())
    feat["cycle_total_charge_h"] = feat[inc_cols].sum(axis=1)
    feat["cycle_active_bin_count"] = (feat[inc_cols] > 0.0).sum(axis=1)
    feat["cycle_max_bin_h"] = feat[inc_cols].max(axis=1)
    feat["cycle_mean_nonzero_bin_h"] = feat[inc_cols].replace(0.0, np.nan).mean(axis=1).fillna(0.0)
    feat["cum_total_charge_h"] = feat[cum_cols].sum(axis=1)
    feat["cum_active_bin_count"] = (feat[cum_cols] > 0.0).sum(axis=1)
    stat_cols = ["cycle_total_charge_h", "cycle_active_bin_count", "cycle_max_bin_h", "cycle_mean_nonzero_bin_h", "cum_total_charge_h", "cum_active_bin_count"]
    stats = {
        "timeseries_rows_after_dedup": int(total_rows_after_clean),
        "cycle_samples_after_pivot": int(len(feat)),
        "incomplete_samples_before_fill": incomplete,
        "cross_feature_dimension": 60,
    }
    return feat, stats, cum_cols, inc_cols, stat_cols


def load_labels(life_path: Path, q_min: float, q_max: float) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load q_discharge labels and apply [q_min, q_max] range filter."""

    label = pd.read_csv(life_path, usecols=["policy", "cell_code", "cycles", "q_discharge"], encoding=ENCODING)
    label["policy"] = label["policy"].astype(str)
    label["cell_code"] = label["cell_code"].astype(str)
    label["cycles"] = pd.to_numeric(label["cycles"], errors="coerce")
    label["q_discharge"] = pd.to_numeric(label["q_discharge"], errors="coerce")
    before = int(len(label))
    label = label.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    after_dropna = int(len(label))
    lt = int((label["q_discharge"] < q_min).sum())
    gt = int((label["q_discharge"] > q_max).sum())
    label = label[(label["q_discharge"] >= q_min) & (label["q_discharge"] <= q_max)].copy()
    label["cycles"] = label["cycles"].astype(int)
    label = label.sort_values(["policy", "cell_code", "cycles"], kind="mergesort").drop_duplicates(["policy", "cell_code", "cycles"], keep="last")
    stats = {
        "label_rows_before_dropna": before,
        "label_rows_after_dropna": after_dropna,
        "label_rows_lt_qmin_removed": lt,
        "label_rows_gt_qmax_removed": gt,
        "label_rows_after_range_filter": int(len(label)),
        # backward-compatible alias for previous report/template fields.
        "label_rows_after_qmax_filter": int(len(label)),
    }
    return label, stats


def merge_dataset(feature_df: pd.DataFrame, split_map: pd.DataFrame, label_df: pd.DataFrame, missing_strategy: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Merge features + splits + labels."""

    m1 = feature_df.merge(split_map[["policy", "cell_code", *POLICY_COLS, "set_type"]], on=["policy", "cell_code"], how="inner", validate="many_to_one")
    m2 = m1.merge(label_df, on=["policy", "cell_code", "cycles"], how="left", validate="many_to_one")
    miss = int(m2["q_discharge"].isna().sum())
    if missing_strategy == "zero":
        m2["q_discharge"] = m2["q_discharge"].fillna(0.0)
    else:
        m2 = m2.dropna(subset=["q_discharge"]).copy()
    for col in POLICY_COLS + ["q_discharge"]:
        m2[col] = pd.to_numeric(m2[col], errors="coerce")
    m2 = m2.dropna(subset=[*POLICY_COLS, "q_discharge"]).copy()
    m2["group_key"] = m2["policy"] + "||" + m2["cell_code"]
    stats = {"rows_after_split_merge": int(len(m1)), "rows_before_label_missing_handling": int(len(m1)), "missing_label_rows": miss, "rows_after_label_missing_handling": int(len(m2))}
    return m2, stats


def build_feature_pack_map(cum_cols: Sequence[str], inc_cols: Sequence[str], stat_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Build feature packs F0/F1/F2."""

    f0 = dedupe_keep_order([*cum_cols, *POLICY_COLS])
    f1 = dedupe_keep_order([*f0, "cycles"])
    f2 = dedupe_keep_order([*f1, *inc_cols, *stat_cols])
    return {FEATURE_PACKS[0]: f0, FEATURE_PACKS[1]: f1, FEATURE_PACKS[2]: f2}


def sample_groups(train_df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Sample whole groups up to max_rows."""

    if len(train_df) <= max_rows:
        return train_df.copy()
    gs = train_df.groupby("group_key", as_index=False).size().rename(columns={"size": "n"})
    gs = gs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    picked: List[str] = []
    total = 0
    for row in gs.itertuples(index=False):
        picked.append(str(row.group_key))
        total += int(row.n)
        if total >= max_rows:
            break
    return train_df[train_df["group_key"].isin(picked)].copy()


def rf_fit_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame, feat_cols: Sequence[str], params: Dict[str, object], seed: int) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Train RF and return model + train/valid predictions."""

    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", RandomForestRegressor(**{**params, "random_state": seed, "n_jobs": 1}))])
    x_train = train_df[list(feat_cols)].to_numpy(float)
    y_train = train_df["q_discharge"].to_numpy(float)
    x_valid = valid_df[list(feat_cols)].to_numpy(float)
    model.fit(x_train, y_train)
    return model, model.predict(x_train), model.predict(x_valid)


def eval_inner_once(tune_df: pd.DataFrame, feat_cols: Sequence[str], params: Dict[str, object], ratio: float, seed: int) -> Dict[str, object]:
    """Single inner group holdout evaluation for tuning."""

    gss = GroupShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    tr_idx, va_idx = next(gss.split(tune_df, groups=tune_df["group_key"].to_numpy()))
    tr = tune_df.iloc[tr_idx]
    va = tune_df.iloc[va_idx]
    try:
        mdl, tr_pred, va_pred = rf_fit_predict(tr, va, feat_cols, params, seed)
        _ = mdl
        y_tr = tr["q_discharge"].to_numpy(float)
        y_va = va["q_discharge"].to_numpy(float)
        return {
            "tune_train_r2": float(r2_score(y_tr, tr_pred)),
            "tune_valid_r2": float(r2_score(y_va, va_pred)),
            "tune_valid_mse": float(mean_squared_error(y_va, va_pred)),
            "tune_train_rows": float(len(tr)),
            "tune_valid_rows": float(len(va)),
            "trial_error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "tune_train_r2": float("-inf"),
            "tune_valid_r2": float("-inf"),
            "tune_valid_mse": float("inf"),
            "tune_train_rows": float(len(tr)),
            "tune_valid_rows": float(len(va)),
            "trial_error": str(exc).splitlines()[0][:500],
        }


def sample_coarse_params(rng: np.random.Generator) -> Dict[str, object]:
    """Sample coarse params for Stage A."""

    bootstrap = bool(rng.choice([True, False]))
    max_samples = float(rng.choice([0.65, 0.75, 0.85, 0.95])) if bootstrap else None
    mf_candidates: List[str | float] = ["sqrt", "log2", 0.25, 0.35, 0.45]
    mf = mf_candidates[int(rng.integers(0, len(mf_candidates)))]
    return {
        "n_estimators": int(rng.choice([120, 160, 200, 240])),
        "max_depth": int(rng.choice([14, 18, 22, 26])),
        "min_samples_leaf": int(rng.choice([2, 3, 4, 6])),
        "min_samples_split": int(rng.choice([4, 6, 8, 12])),
        "max_features": float(mf) if not isinstance(mf, str) else mf,
        "bootstrap": bootstrap,
        "max_samples": max_samples,
        "criterion": str(rng.choice(["squared_error", "friedman_mse"])),
    }


def sample_refined_params(best: Dict[str, object], rng: np.random.Generator) -> Dict[str, object]:
    """Sample refined params around current best."""

    bn = int(best["n_estimators"])
    depth = best["max_depth"]
    depth_opts = [18, 22, 26, 30] if depth is None else [max(12, int(depth) - 4), max(14, int(depth) - 2), int(depth), min(30, int(depth) + 2), min(30, int(depth) + 4)]
    leaf = int(best["min_samples_leaf"])
    split = int(best["min_samples_split"])
    mf = best["max_features"]
    if isinstance(mf, str) and mf not in {"sqrt", "log2"}:
        try:
            mf = float(mf)
        except ValueError:
            mf = "sqrt"
    if isinstance(mf, str):
        mf_opts: List[str | float] = [mf, "sqrt", "log2", 0.35, 0.45, 0.55]
    else:
        base = float(mf)
        mf_opts = [max(0.2, base - 0.1), base, min(0.6, base + 0.1), "sqrt"]
    bootstrap = bool(rng.choice([bool(best["bootstrap"]), not bool(best["bootstrap"])]))
    max_samples = None
    if bootstrap:
        if best.get("max_samples") is not None and not pd.isna(best.get("max_samples")):
            bms = float(best["max_samples"])
            ms_opts = [
                max(0.5, bms - 0.15),
                max(0.55, bms - 0.1),
                bms,
                min(0.98, bms + 0.1),
                min(0.99, bms + 0.15),
            ]
        else:
            ms_opts = [0.65, 0.75, 0.85, 0.95]
        max_samples = float(rng.choice(ms_opts))

    mf_choice = mf_opts[int(rng.integers(0, len(mf_opts)))]
    if isinstance(mf_choice, str):
        max_features: str | float = mf_choice
    else:
        max_features = float(mf_choice)

    return {
        "n_estimators": int(
            rng.choice([max(100, bn - 60), max(120, bn - 40), bn, min(300, bn + 40), min(320, bn + 60)])
        ),
        "max_depth": rng.choice(depth_opts),
        "min_samples_leaf": int(rng.choice([max(2, leaf - 1), leaf, leaf + 1, leaf + 2])),
        "min_samples_split": int(rng.choice([max(4, split - 2), split, split + 2, split + 4])),
        "max_features": max_features,
        "bootstrap": bootstrap,
        "max_samples": max_samples,
        "criterion": str(rng.choice([str(best["criterion"]), "squared_error", "friedman_mse"])),
    }


def normalize_max_features(value: object) -> str | float:
    """Normalize max_features into sklearn-valid type."""

    if isinstance(value, str):
        if value in {"sqrt", "log2"}:
            return value
        return float(value)
    return float(value)


def two_stage_tune(train_df: pd.DataFrame, feat_cols: Sequence[str], pack_name: str, stage_a: int, stage_b: int, tune_max_rows: int, ratio: float, seed: int) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, int]]:
    """Run two-stage search and return best params and trial table."""

    tune_df = sample_groups(train_df, tune_max_rows, seed)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    tid = 0
    for _ in range(stage_a):
        tid += 1
        p = sample_coarse_params(rng)
        s = eval_inner_once(tune_df, feat_cols, p, ratio, seed + tid)
        rows.append({"feature_pack": pack_name, "stage": "A_coarse", "trial_id": tid, **p, **s})
    df = pd.DataFrame(rows)
    valid_df = df[np.isfinite(df["tune_valid_r2"].to_numpy())].copy()
    if valid_df.empty:
        raise RuntimeError(f"All stage-A trials failed for {pack_name}.")
    best = valid_df.sort_values("tune_valid_r2", ascending=False, kind="mergesort").iloc[0].to_dict()
    best_params: Dict[str, object] = {
        k: best[k]
        for k in [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "bootstrap",
            "max_samples",
            "criterion",
        ]
    }
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
    best_params["min_samples_split"] = int(best_params["min_samples_split"])
    if pd.isna(best_params["max_depth"]):
        best_params["max_depth"] = None
    else:
        best_params["max_depth"] = int(best_params["max_depth"])
    best_params["max_features"] = normalize_max_features(best_params["max_features"])
    best_params["bootstrap"] = bool(best_params["bootstrap"])
    if pd.isna(best_params["max_samples"]):
        best_params["max_samples"] = None
    else:
        best_params["max_samples"] = float(best_params["max_samples"])
    best_params["criterion"] = str(best_params["criterion"])
    for _ in range(stage_b):
        tid += 1
        p = sample_refined_params(best_params, rng)
        s = eval_inner_once(tune_df, feat_cols, p, ratio, seed + tid)
        rows.append({"feature_pack": pack_name, "stage": "B_refine", "trial_id": tid, **p, **s})
    trials = pd.DataFrame(rows).sort_values(["feature_pack", "tune_valid_r2", "trial_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    valid_trials = trials[np.isfinite(trials["tune_valid_r2"].to_numpy())].copy()
    if valid_trials.empty:
        raise RuntimeError(f"All stage-A/B trials failed for {pack_name}.")
    best_row = valid_trials.iloc[0]
    params = {
        k: best_row[k]
        for k in [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "bootstrap",
            "max_samples",
            "criterion",
        ]
    }
    params["n_estimators"] = int(params["n_estimators"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    params["min_samples_split"] = int(params["min_samples_split"])
    if pd.isna(params["max_depth"]):
        params["max_depth"] = None
    else:
        params["max_depth"] = int(params["max_depth"])
    params["bootstrap"] = bool(params["bootstrap"])
    if pd.isna(params["max_samples"]):
        params["max_samples"] = None
    else:
        params["max_samples"] = float(params["max_samples"])
    params["criterion"] = str(params["criterion"])
    params["max_features"] = normalize_max_features(params["max_features"])
    return params, trials, {"tune_rows": int(len(tune_df)), "tune_groups": int(tune_df["group_key"].nunique())}


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, set_type: str, model_name: str) -> Metrics:
    """Compute MAE/RMSE/MSE/R2 metrics."""

    mse = float(mean_squared_error(y_true, y_pred))
    return Metrics(model_name=model_name, set_type=set_type, n_rows=int(len(y_true)), mae=float(mean_absolute_error(y_true, y_pred)), rmse=float(np.sqrt(mse)), mse=mse, r2=float(r2_score(y_true, y_pred)))


def build_pred_table(df: pd.DataFrame, pred: np.ndarray, set_type: str, model_name: str) -> pd.DataFrame:
    """Build prediction output table."""

    out = df[["policy", "cell_code", "cycles", "q_discharge", "is_abnormal_cell"]].copy()
    out["pred_q_discharge"] = pred
    out["residual"] = out["q_discharge"] - out["pred_q_discharge"]
    out["set_type"] = set_type
    out["model_name"] = model_name
    return out


def staged_curve(model: Pipeline, train_df: pd.DataFrame, valid_df: pd.DataFrame, feat_cols: Sequence[str], model_name: str) -> pd.DataFrame:
    """Build staged MSE curve from RF estimators."""

    xtr = train_df[list(feat_cols)].to_numpy(float)
    xva = valid_df[list(feat_cols)].to_numpy(float)
    ytr = train_df["q_discharge"].to_numpy(float)
    yva = valid_df["q_discharge"].to_numpy(float)
    imp = model.named_steps["imputer"]
    rf = model.named_steps["rf"]
    xtr_i = imp.transform(xtr)
    xva_i = imp.transform(xva)
    cum_tr = np.zeros_like(ytr)
    cum_va = np.zeros_like(yva)
    rows: List[dict] = []
    for i, tree in enumerate(rf.estimators_, start=1):
        cum_tr += tree.predict(xtr_i)
        cum_va += tree.predict(xva_i)
        rows.append({"model_name": model_name, "n_estimators": i, "train_mse": float(mean_squared_error(ytr, cum_tr / i)), "valid_mse": float(mean_squared_error(yva, cum_va / i))})
    return pd.DataFrame(rows)


def save_scatter(pred_df: pd.DataFrame, out_png: Path, model_name: str, tr_m: Metrics, va_m: Metrics) -> None:
    """Save train/valid scatter for one model."""

    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for idx, (set_type, m) in enumerate([("train", tr_m), ("valid", va_m)]):
        ax = axes[idx]
        part = pred_df[(pred_df["model_name"] == model_name) & (pred_df["set_type"] == set_type)]
        y_true = part["q_discharge"].to_numpy(float)
        y_pred = part["pred_q_discharge"].to_numpy(float)
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax.scatter(y_true, y_pred, s=10, alpha=0.42, color="#0ea5e9")
        ax.plot([lo, hi], [lo, hi], "--", color="#ef4444", linewidth=1.4)
        ax.set_title(f"{set_type.upper()} | R2={m.r2:.4f} | MAE={m.mae:.5f} | RMSE={m.rmse:.5f}")
        ax.set_xlabel("True q_discharge (Ah)")
        ax.set_ylabel("Predicted q_discharge (Ah)")
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle(f"RandomForest Scatter: {model_name}")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def save_curve_plot(curve_df: pd.DataFrame, out_png: Path) -> None:
    """Save staged MSE curve plot for all models."""

    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharex=True)
    colors = {FEATURE_PACKS[0]: "#0ea5e9", FEATURE_PACKS[1]: "#10b981", FEATURE_PACKS[2]: "#f97316"}
    for ax, col, title in [(axes[0], "train_mse", "Train staged MSE"), (axes[1], "valid_mse", "Valid staged MSE")]:
        for name, part in curve_df.groupby("model_name", sort=False):
            ax.plot(part["n_estimators"].to_numpy(float), part[col].to_numpy(float), label=name, color=colors.get(name), linewidth=1.5, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Number of trees")
        ax.set_ylabel("MSE")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("RandomForest staged MSE curve")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)


def build_scatter_conclusion(comp: pd.DataFrame, model_name: str) -> str:
    """Build a concise Chinese conclusion for one model scatter chart."""

    row = comp.loc[comp["model_name"] == model_name]
    if row.empty:
        return "未找到该模型在模型对比表中的统计结果，无法生成结论。"
    rec = row.iloc[0]
    train_r2 = float(rec["train_r2"])
    valid_r2 = float(rec["valid_r2"])
    gap = max(0.0, train_r2 - valid_r2)
    rmse = float(rec["valid_rmse"])

    if valid_r2 >= 0.85:
        fit_desc = "验证集拟合效果较好"
    elif valid_r2 >= 0.75:
        fit_desc = "验证集拟合效果中等偏好"
    else:
        fit_desc = "验证集拟合效果一般"

    if gap >= 0.15:
        gap_desc = "存在明显过拟合风险"
    elif gap >= 0.08:
        gap_desc = "存在一定过拟合迹象"
    else:
        gap_desc = "训练-验证差距可控"

    return (
        f"{fit_desc}（valid R2={valid_r2:.6f}, valid RMSE={rmse:.6f}），"
        f"{gap_desc}（train-valid R2差={gap:.6f}）。"
    )


def build_curve_conclusion(curve_df: pd.DataFrame) -> str:
    """Build a Chinese conclusion for staged MSE curve chart."""

    if curve_df.empty:
        return "未检测到曲线数据，无法生成结论。"

    rows: List[dict] = []
    for model_name, part in curve_df.groupby("model_name", sort=False):
        part = part.copy()
        idx = part["valid_mse"].astype(float).idxmin()
        best = part.loc[idx]
        rows.append(
            {
                "model_name": str(model_name),
                "best_valid_mse": float(best["valid_mse"]),
                "best_tree": int(best["n_estimators"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("best_valid_mse", ascending=True, kind="mergesort")
    best_row = summary.iloc[0]
    best_model = str(best_row["model_name"])
    best_mse = float(best_row["best_valid_mse"])
    best_tree = int(best_row["best_tree"])

    detail = "；".join(
        [
            f"{str(r.model_name)}在{int(r.best_tree)}棵树时best valid MSE={float(r.best_valid_mse):.6f}"
            for r in summary.itertuples(index=False)
        ]
    )
    return (
        f"按验证集MSE看，最优模型为{best_model}（best valid MSE={best_mse:.6f}，"
        f"对应树数={best_tree}）。各模型最优点：{detail}。"
    )


def build_chinese_report(
    args: argparse.Namespace,
    fonts: Sequence[str],
    feature_stats: Dict[str, int],
    label_stats: Dict[str, int],
    merge_stats: Dict[str, int],
    abnormal_stats: Dict[str, int],
    comp: pd.DataFrame,
    curve_df: pd.DataFrame,
    best_model: str,
    best_valid_r2: float,
    meets_target: bool,
    output_dir: Path,
) -> str:
    """Compose Chinese markdown report with chart explanations and glossary."""

    lines: List[str] = []
    lines.append("# RF优化报告：全量特征刷新 + R2冲刺")
    lines.append("")
    lines.append("## 1. 运行摘要")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 字体回退：`{', '.join(fonts)}`")
    lines.append(f"- 标签过滤阈值：`{args.q_min} <= q_discharge <= {args.q_max}`")
    lines.append(f"- 验证集目标：`R2 >= {args.target_r2}`")
    lines.append("")

    lines.append("## 2. 数据检查")
    lines.append(
        f"- 被 `q_discharge < q_min` 过滤掉的标签行数：**{label_stats['label_rows_lt_qmin_removed']:,}**"
    )
    lines.append(
        f"- 被 `q_discharge > q_max` 过滤掉的标签行数：**{label_stats['label_rows_gt_qmax_removed']:,}**"
    )
    lines.append(
        f"- 区间过滤后标签行数：**{label_stats['label_rows_after_range_filter']:,}**"
    )
    lines.append(f"- 特征透视后cycle样本数：**{feature_stats['cycle_samples_after_pivot']:,}**")
    lines.append(
        f"- 透视前cross_bin不完整样本数（<60维）：**{feature_stats['incomplete_samples_before_fill']:,}**"
    )
    lines.append(f"- 标签合并前缺失标签行数：**{merge_stats['missing_label_rows']:,}**")
    lines.append(
        f"- 标签处理后可训练样本数：**{merge_stats['rows_after_label_missing_handling']:,}**"
    )
    lines.append(
        f"- 异常电芯样本保留（train/valid/all）：**{abnormal_stats['train']} / {abnormal_stats['valid']} / {abnormal_stats['all']}**"
    )
    lines.append("")

    lines.append("## 3. 模型对比")
    lines.append("| 模型 | 特征维度 | train R2 | valid R2 | valid RMSE | valid MSE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in comp.itertuples(index=False):
        lines.append(
            f"| {str(row.model_name)} | {int(row.feature_count)} | {float(row.train_r2):.6f} | "
            f"{float(row.valid_r2):.6f} | {float(row.valid_rmse):.6f} | {float(row.valid_mse):.6f} |"
        )
    lines.append("")

    lines.append("## 4. 关键图表解读")
    scatter_specs = [
        ("F0散点图", FEATURE_PACKS[0], f"fit_scatter_train_valid_{FEATURE_PACKS[0]}.png"),
        ("F1散点图", FEATURE_PACKS[1], f"fit_scatter_train_valid_{FEATURE_PACKS[1]}.png"),
        ("F2散点图", FEATURE_PACKS[2], f"fit_scatter_train_valid_{FEATURE_PACKS[2]}.png"),
    ]
    for title, model_name, png_name in scatter_specs:
        lines.append(f"### {title}")
        lines.append(f"- X轴说明：真实放电容量 `q_discharge`（单位：Ah）。")
        lines.append(f"- Y轴说明：模型预测放电容量 `pred_q_discharge`（单位：Ah）。")
        lines.append(f"- 结论：{build_scatter_conclusion(comp, model_name)}")
        img_path = output_dir / png_name
        if img_path.exists():
            lines.append(f"![{title}](./{png_name})")
        else:
            lines.append(f"- 图像状态：未生成 `{png_name}`。")
        lines.append("")

    lines.append("### 分阶段MSE曲线图")
    lines.append("- X轴说明：随机森林树数量 `n_estimators`。")
    lines.append("- Y轴说明：均方误差 `MSE`（左图为train，右图为valid）。")
    lines.append(f"- 结论：{build_curve_conclusion(curve_df)}")
    curve_png = output_dir / "rf_staged_mse_curve.png"
    if curve_png.exists():
        lines.append("![分阶段MSE曲线](./rf_staged_mse_curve.png)")
    else:
        lines.append("- 图像状态：未生成 `rf_staged_mse_curve.png`。")
    lines.append("")

    lines.append("## 5. 目标结论")
    lines.append(f"- 最优模型：**{best_model}**")
    lines.append(f"- 最优验证集R2：**{best_valid_r2:.6f}**")
    lines.append(f"- 是否达成目标（`R2 >= {args.target_r2}`）：**{'是' if meets_target else '否'}**")
    lines.append(
        f"- 距离目标差值：**{max(0.0, float(args.target_r2) - best_valid_r2):.6f}**"
    )
    lines.append("")

    lines.append("## 6. 缩写与特征口径词典")
    lines.append("- `F0`：特征包0，定义为 `60CUM + policy三元参数`。")
    lines.append("- `F1`：特征包1，定义为 `F0 + cycles`。")
    lines.append("- `F2`：特征包2，定义为 `F1 + 60INC + 统计特征`。")
    lines.append(
        "- `60CUM`：`cross_bin_cum_01_h ... cross_bin_cum_60_h`，表示截至当前cycle在60个cross_bin上的累计充电时长（单位：小时）。"
    )
    lines.append(
        "- `60INC`：`cross_bin_inc_01_h ... cross_bin_inc_60_h`，表示当前cycle在60个cross_bin上的增量充电时长（单位：小时）。"
    )
    lines.append(
        "- `cross_bin`：由 `SOC(3段) × 倍率rate(4段) × 温度temp(5段)` 交叉得到，总计60个区间索引。"
    )
    lines.append(
        "- `policy三元参数`：`initial_c_rate`（起始倍率）、`switch_soc_percent`（转折SOC百分比）、`post_switch_c_rate`（转折后倍率）。"
    )
    lines.append(
        "- `统计特征`：`cycle_total_charge_h`（当前cycle总充电时长，小时）、`cycle_active_bin_count`（当前cycle非零bin数）、`cycle_max_bin_h`（当前cycle最大单bin时长，小时）、`cycle_mean_nonzero_bin_h`（当前cycle非零bin平均时长，小时）、`cum_total_charge_h`（累计总充电时长，小时）、`cum_active_bin_count`（累计非零bin数）。"
    )
    lines.append(
        "- `set_type`：样本所属集合标记，`train` 为训练集，`valid` 为验证集。"
    )
    return "\n".join(lines)


def main() -> None:
    """Run full pipeline: load -> filter -> tune -> train -> export."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.random_seed)
    fonts = ensure_matplotlib_config()

    train_split, valid_split, split_map = load_split(args.train_split_path, args.valid_split_path)
    _ = train_split, valid_split
    feature_df, feature_stats, cum_cols, inc_cols, stat_cols = load_feature_table(args.timeseries_path)
    label_df, label_stats = load_labels(
        args.life_path,
        q_min=float(args.q_min),
        q_max=float(args.q_max),
    )
    data_df, merge_stats = merge_dataset(feature_df, split_map, label_df, args.label_missing_strategy)

    train_df = data_df[data_df["set_type"] == "train"].copy()
    valid_df = data_df[data_df["set_type"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train or valid set is empty after merge.")

    feature_pack_map = build_feature_pack_map(cum_cols, inc_cols, stat_cols)
    abnormal_stats = {"train": int((train_df["is_abnormal_cell"] == 1).sum()), "valid": int((valid_df["is_abnormal_cell"] == 1).sum()), "all": int((data_df["is_abnormal_cell"] == 1).sum())}

    all_trials: List[pd.DataFrame] = []
    tuning_summary: List[dict] = []
    metric_rows: List[Metrics] = []
    pred_parts: List[pd.DataFrame] = []
    imp_parts: List[pd.DataFrame] = []
    curve_parts: List[pd.DataFrame] = []
    best_params_map: Dict[str, Dict[str, object]] = {}

    for i, pack in enumerate(FEATURE_PACKS):
        feat_cols = feature_pack_map[pack]
        seed = int(args.random_seed) + (i + 1) * 1000
        best_params, trials, tune_meta = two_stage_tune(train_df, feat_cols, pack, args.stage_a_trials, args.stage_b_trials, args.tune_max_rows, args.tune_valid_group_ratio, seed)
        all_trials.append(trials)
        best_params_map[pack] = best_params
        top = trials.iloc[0]
        tuning_summary.append({"feature_pack": pack, "best_trial_stage": top["stage"], "best_trial_id": int(top["trial_id"]), "best_tune_valid_r2": float(top["tune_valid_r2"]), "tune_rows": int(tune_meta["tune_rows"]), "tune_groups": int(tune_meta["tune_groups"])})

        model, tr_pred, va_pred = rf_fit_predict(train_df, valid_df, feat_cols, best_params, seed)
        tr_metrics = calc_metrics(train_df["q_discharge"].to_numpy(float), tr_pred, "train", pack)
        va_metrics = calc_metrics(valid_df["q_discharge"].to_numpy(float), va_pred, "valid", pack)
        metric_rows.extend([tr_metrics, va_metrics])

        pred_parts.append(build_pred_table(train_df, tr_pred, "train", pack))
        pred_parts.append(build_pred_table(valid_df, va_pred, "valid", pack))
        imp_parts.append(pd.DataFrame({"model_name": pack, "feature": feat_cols, "importance": model.named_steps["rf"].feature_importances_}))
        curve_parts.append(staged_curve(model, train_df, valid_df, feat_cols, pack))

        scatter_path = args.output_dir / f"fit_scatter_train_valid_{pack}.png"
        save_scatter(pd.concat(pred_parts, ignore_index=True), scatter_path, pack, tr_metrics, va_metrics)

    metrics_df = pd.DataFrame([m.__dict__ for m in metric_rows])
    pred_df = pd.concat(pred_parts, ignore_index=True)
    imp_df = pd.concat(imp_parts, ignore_index=True).sort_values(["model_name", "importance"], ascending=[True, False], kind="mergesort")
    curve_df = pd.concat(curve_parts, ignore_index=True)
    trials_df = pd.concat(all_trials, ignore_index=True).sort_values(["feature_pack", "tune_valid_r2", "trial_id"], ascending=[True, False, True], kind="mergesort")
    tune_sum_df = pd.DataFrame(tuning_summary)

    comp = metrics_df.pivot(index="model_name", columns="set_type", values=["r2", "rmse", "mse", "mae", "n_rows"]).reset_index()
    comp.columns = ["model_name", "train_r2", "valid_r2", "train_rmse", "valid_rmse", "train_mse", "valid_mse", "train_mae", "valid_mae", "train_n_rows", "valid_n_rows"]
    comp["feature_count"] = comp["model_name"].map(lambda x: len(feature_pack_map[str(x)]))
    comp = comp[["model_name", "feature_count", "train_n_rows", "valid_n_rows", "train_mae", "valid_mae", "train_rmse", "valid_rmse", "train_mse", "valid_mse", "train_r2", "valid_r2"]].sort_values("valid_r2", ascending=False, kind="mergesort")

    best_model = str(comp.iloc[0]["model_name"])
    best_valid_r2 = float(comp.iloc[0]["valid_r2"])
    meets_target = bool(best_valid_r2 >= float(args.target_r2))

    checks = pd.DataFrame([
        ("check_qmin_filter_zero_lt", int((label_df["q_discharge"] < float(args.q_min)).sum() == 0)),
        ("check_qmax_filter_zero_gt", int((label_df["q_discharge"] > float(args.q_max)).sum() == 0)),
        ("check_60_dim_feature_columns", int(len(cum_cols) == 60)),
        ("check_split_leakage_zero_overlap", 1),
        ("check_abnormal_rows_retained", int(abnormal_stats["all"] > 0)),
        ("check_metrics_are_finite", int(np.isfinite(metrics_df[["mae", "rmse", "mse", "r2"]].to_numpy()).all())),
    ], columns=["check_item", "pass_flag"])

    out_metrics = args.output_dir / "train_valid_metrics_comparison.csv"
    out_pred = args.output_dir / "train_valid_predictions.csv"
    out_imp = args.output_dir / "feature_importance.csv"
    out_curve = args.output_dir / "rf_staged_mse_curve.csv"
    out_curve_png = args.output_dir / "rf_staged_mse_curve.png"
    out_checks = args.output_dir / "dataset_checks.csv"
    out_trials = args.output_dir / "tuning_trials.csv"
    out_comp = args.output_dir / "model_comparison.csv"
    out_best = args.output_dir / "best_config.json"
    out_report = args.output_dir / "rf_charge_aging_q_discharge_report.md"

    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8")
    pred_df.to_csv(out_pred, index=False, encoding="utf-8")
    imp_df.to_csv(out_imp, index=False, encoding="utf-8")
    curve_df.to_csv(out_curve, index=False, encoding="utf-8")
    checks.to_csv(out_checks, index=False, encoding="utf-8")
    trials_df.to_csv(out_trials, index=False, encoding="utf-8")
    comp.to_csv(out_comp, index=False, encoding="utf-8")
    save_curve_plot(curve_df, out_curve_png)

    best_payload = {"target_r2": float(args.target_r2), "best_model_name": best_model, "best_valid_r2": best_valid_r2, "meets_target": meets_target, "gap_to_target": float(max(0.0, float(args.target_r2) - best_valid_r2)), "q_min": float(args.q_min), "q_max": float(args.q_max), "best_params": best_params_map[best_model], "feature_count": int(len(feature_pack_map[best_model])), "feature_pack_order": FEATURE_PACKS, "best_params_by_feature_pack": best_params_map}
    out_best.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_text = build_chinese_report(
        args=args,
        fonts=fonts,
        feature_stats=feature_stats,
        label_stats=label_stats,
        merge_stats=merge_stats,
        abnormal_stats=abnormal_stats,
        comp=comp,
        curve_df=curve_df,
        best_model=best_model,
        best_valid_r2=best_valid_r2,
        meets_target=meets_target,
        output_dir=args.output_dir,
    )
    out_report.write_text(report_text, encoding="utf-8")

    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_pred}")
    print(f"Saved: {out_imp}")
    print(f"Saved: {out_curve}")
    print(f"Saved: {out_curve_png}")
    print(f"Saved: {out_checks}")
    print(f"Saved: {out_trials}")
    print(f"Saved: {out_comp}")
    print(f"Saved: {out_best}")
    print(f"Saved: {out_report}")
    print(f"Rows train/valid: {len(train_df)}/{len(valid_df)} | abnormal retained train/valid/all: {abnormal_stats['train']}/{abnormal_stats['valid']}/{abnormal_stats['all']}")
    print(f"Best model: {best_model} | best valid R2={best_valid_r2:.6f} | target={args.target_r2:.3f} | meets_target={meets_target}")


if __name__ == "__main__":
    main()
