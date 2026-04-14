from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from random import Random
import sys
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd


# =========================
# Config (edit here first)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "dqdv_polyfit_experiment"

RANDOM_SEED = 20260410
TOTAL_SAMPLE_COUNT = 5

FIXED_POLICY = "4_8C-80PER_4_8C"
FIXED_CELL_CODE = "465027"
FIXED_CYCLE = 10

DEGREE_MIN = 7
DEGREE_MAX = 9

HIGH_SENS_QUANTILE = 0.80
MIN_HIGH_SENS_POINTS = 5
SENSITIVE_VOLTAGE_LOW = 2.5
SENSITIVE_VOLTAGE_HIGH = 3.4

SCORE_WEIGHT_RMSE_HIGH = 0.60
SCORE_WEIGHT_ONE_MINUS_R2 = 0.25
SCORE_WEIGHT_RMSE_Q = 0.15

HR_VOLTAGE_STEP = 0.002
MIN_HIGHRES_POINTS = 50
SIGN_CONSISTENCY_THRESHOLD = 0.95

PROMOTE_MIN_VALID_SAMPLES = 4
PROMOTE_MIN_MEDIAN_R2 = 0.996
PROMOTE_MIN_MEDIAN_SIGN_CONSISTENCY = 0.98

SAMPLE_SELECTION_PATH = OUTPUT_DIR / "sample_selection.csv"
PER_DEGREE_METRICS_PATH = OUTPUT_DIR / "per_degree_metrics.csv"
BEST_DEGREE_SUMMARY_PATH = OUTPUT_DIR / "best_degree_summary.csv"
HIGHRES_CURVES_PATH = OUTPUT_DIR / "highres_dqdv_curves.csv"
FIT_REPORT_PATH = OUTPUT_DIR / "fit_report.md"


def load_extract_module() -> Any:
    """Load the existing dQ/dV extractor module for shared preprocessing logic."""
    module_path = REPO_ROOT / "scripts" / "extract_discharge_dqdv_peak_features.py"
    spec = importlib.util.spec_from_file_location("extract_dqdv_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extraction module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_sample_id(policy: str, cell_code: str, cycles: int) -> str:
    """Create a stable sample id for output files."""
    return f"{policy}|{cell_code}|{cycles}"


def norm_col(values: pd.Series) -> pd.Series:
    """Min-max normalize a metric column with safe zero-range handling."""
    valid = values.replace([np.inf, -np.inf], np.nan)
    lo = valid.min(skipna=True)
    hi = valid.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (valid - lo) / (hi - lo)


def build_desc_voltage_grid(v_high: float, v_low: float, step: float) -> np.ndarray:
    """Build a descending high-resolution voltage grid with fixed step."""
    if step <= 0:
        raise ValueError("HR_VOLTAGE_STEP must be > 0")
    if v_high <= v_low:
        return np.array([], dtype=float)
    count = int(math.floor((v_high - v_low) / step)) + 1
    grid = v_high - np.arange(count, dtype=float) * step
    if grid[-1] > v_low + 1e-12:
        grid = np.append(grid, v_low)
    return grid


def gather_candidates(mod: Any, file_paths: Sequence[Path]) -> Tuple[List[dict], Dict[str, pd.DataFrame]]:
    """
    Gather candidate policy+cell+cycle samples from all raw files.

    Returns:
        candidates: candidate rows with file metadata.
        frame_cache: loaded discharge frame cache by file path string.
    """
    candidates: List[dict] = []
    frame_cache: Dict[str, pd.DataFrame] = {}

    for file_path in file_paths:
        frame = mod.load_discharge_frame(file_path)
        key = str(file_path.resolve())
        frame_cache[key] = frame
        if frame.empty:
            continue

        uniq = frame[["policy", "cell_code", "cycles"]].drop_duplicates().copy()
        uniq["file_path"] = key
        uniq["cell_code"] = uniq["cell_code"].astype(str)
        uniq["cycles"] = uniq["cycles"].astype(int)
        for row in uniq.itertuples(index=False):
            candidates.append(
                {
                    "policy": row.policy,
                    "cell_code": row.cell_code,
                    "cycles": int(row.cycles),
                    "file_path": row.file_path,
                }
            )
    return candidates, frame_cache


def is_valid_candidate(mod: Any, frame: pd.DataFrame, policy: str, cell_code: str, cycles: int) -> bool:
    """Check whether a candidate can produce valid baseline dQ/dV points."""
    group = frame.loc[
        (frame["policy"] == policy)
        & (frame["cell_code"].astype(str) == cell_code)
        & (frame["cycles"] == cycles)
    ]
    if group.empty:
        return False
    v_mid, dqdv_ref, _, _ = mod.build_dqdv_series(group)
    return len(v_mid) > 0 and len(dqdv_ref) > 0


def select_samples(mod: Any, candidates: Sequence[dict], frame_cache: Dict[str, pd.DataFrame]) -> List[dict]:
    """
    Select five reproducible valid samples:
    - always include the fixed sample
    - randomly draw remaining valid samples with fixed seed.
    """
    fixed = None
    for row in candidates:
        if (
            row["policy"] == FIXED_POLICY
            and str(row["cell_code"]) == FIXED_CELL_CODE
            and int(row["cycles"]) == FIXED_CYCLE
        ):
            frame = frame_cache[row["file_path"]]
            if is_valid_candidate(mod, frame, row["policy"], str(row["cell_code"]), int(row["cycles"])):
                fixed = row
                break
    if fixed is None:
        raise ValueError("Fixed sample not found or invalid in candidate pool.")

    selected: List[dict] = [fixed]
    seen_ids = {make_sample_id(fixed["policy"], str(fixed["cell_code"]), int(fixed["cycles"]))}

    others = []
    for row in candidates:
        sample_id = make_sample_id(row["policy"], str(row["cell_code"]), int(row["cycles"]))
        if sample_id in seen_ids:
            continue
        others.append(row)

    rng = Random(RANDOM_SEED)
    rng.shuffle(others)

    for row in others:
        if len(selected) >= TOTAL_SAMPLE_COUNT:
            break
        frame = frame_cache[row["file_path"]]
        if not is_valid_candidate(mod, frame, row["policy"], str(row["cell_code"]), int(row["cycles"])):
            continue
        sample_id = make_sample_id(row["policy"], str(row["cell_code"]), int(row["cycles"]))
        if sample_id in seen_ids:
            continue
        selected.append(row)
        seen_ids.add(sample_id)

    if len(selected) < TOTAL_SAMPLE_COUNT:
        raise RuntimeError(
            f"Could not find enough valid samples. expected={TOTAL_SAMPLE_COUNT}, got={len(selected)}"
        )
    return selected


def ensure_high_sensitivity_mask(v_mid_ref: np.ndarray) -> np.ndarray:
    """Build high-sensitivity mask by fixed sensitive voltage window [2.5V, 3.4V]."""
    mask = (v_mid_ref >= SENSITIVE_VOLTAGE_LOW) & (v_mid_ref <= SENSITIVE_VOLTAGE_HIGH)
    if int(mask.sum()) < MIN_HIGH_SENS_POINTS:
        # Keep experiment robust when a cycle has too few points in the sensitive window.
        return np.ones_like(v_mid_ref, dtype=bool)
    return mask


def fit_one_degree(v: np.ndarray, q: np.ndarray, v_eval: np.ndarray, degree: int) -> Tuple[Any, float, float, bool]:
    """Fit one polynomial degree and return fitted poly, r2, rmse_q, rankwarning flag."""
    rank_warning = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        poly = np.polynomial.Polynomial.fit(v, q, deg=degree)
    for item in caught:
        if "poorly conditioned" in str(item.message).lower():
            rank_warning = True
            break

    q_pred = poly(v)
    ss_res = float(np.sum((q - q_pred) ** 2))
    ss_tot = float(np.sum((q - np.mean(q)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    rmse_q = float(np.sqrt(np.mean((q - q_pred) ** 2)))

    _ = poly.deriv()(v_eval)
    return poly, r2, rmse_q, rank_warning


def evaluate_sample(mod: Any, frame: pd.DataFrame, sample_row: dict) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Evaluate polynomial degrees on one sample and return metrics, best summary, and highres curve rows."""
    policy = sample_row["policy"]
    cell_code = str(sample_row["cell_code"])
    cycles = int(sample_row["cycles"])
    sample_id = make_sample_id(policy, cell_code, cycles)

    group = frame.loc[
        (frame["policy"] == policy)
        & (frame["cell_code"].astype(str) == cell_code)
        & (frame["cycles"] == cycles)
    ].copy()
    if group.empty:
        raise ValueError(f"Group not found for sample_id={sample_id}")

    v_mid_ref, dqdv_ref_raw, n_window, n_dqdv = mod.build_dqdv_series(group)
    if len(v_mid_ref) == 0 or len(dqdv_ref_raw) == 0:
        raise ValueError(f"Baseline dQ/dV is empty for sample_id={sample_id}")

    # Requirement: for discharge, evaluate dQ/dV by absolute value.
    dqdv_ref = np.abs(dqdv_ref_raw)

    hs_mask = ensure_high_sensitivity_mask(v_mid_ref)
    hs_points = int(hs_mask.sum())

    v = group["V"].to_numpy(dtype=float)
    q = group["ah_dischg"].to_numpy(dtype=float)

    degree_rows: List[dict] = []
    poly_map: Dict[int, Any] = {}
    for degree in range(DEGREE_MIN, DEGREE_MAX + 1):
        poly, r2, rmse_q, rank_warning = fit_one_degree(v, q, v_mid_ref, degree)
        dqdv_fit = np.abs(poly.deriv()(v_mid_ref))
        rmse_high = float(np.sqrt(np.mean((dqdv_fit[hs_mask] - dqdv_ref[hs_mask]) ** 2)))
        rmse_all = float(np.sqrt(np.mean((dqdv_fit - dqdv_ref) ** 2)))

        degree_rows.append(
            {
                "sample_id": sample_id,
                "policy": policy,
                "cell_code": cell_code,
                "cycles": cycles,
                "degree": degree,
                "r2_q": r2,
                "rmse_q": rmse_q,
                "rmse_dqdv_high": rmse_high,
                "rmse_dqdv_all": rmse_all,
                "high_sens_threshold_abs_dqdv": np.nan,
                "high_sens_points": hs_points,
                "sensitive_voltage_low": SENSITIVE_VOLTAGE_LOW,
                "sensitive_voltage_high": SENSITIVE_VOLTAGE_HIGH,
                "rank_warning": int(rank_warning),
                "n_points_window": int(n_window),
                "n_points_dqdv_ref": int(n_dqdv),
            }
        )
        poly_map[degree] = poly

    metric_df = pd.DataFrame(degree_rows)
    metric_df["one_minus_r2"] = 1.0 - metric_df["r2_q"]
    metric_df["norm_rmse_high"] = norm_col(metric_df["rmse_dqdv_high"])
    metric_df["norm_one_minus_r2"] = norm_col(metric_df["one_minus_r2"])
    metric_df["norm_rmse_q"] = norm_col(metric_df["rmse_q"])
    metric_df["score"] = (
        SCORE_WEIGHT_RMSE_HIGH * metric_df["norm_rmse_high"]
        + SCORE_WEIGHT_ONE_MINUS_R2 * metric_df["norm_one_minus_r2"]
        + SCORE_WEIGHT_RMSE_Q * metric_df["norm_rmse_q"]
    )

    best_df = metric_df.sort_values(
        by=["score", "rmse_dqdv_high", "degree"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = best_df.iloc[0]
    best_degree = int(best["degree"])
    metric_df["is_best_degree"] = (metric_df["degree"] == best_degree).astype(int)

    poly_best = poly_map[best_degree]
    v_high = float(np.max(v))
    v_low = float(np.min(v))
    v_hr = build_desc_voltage_grid(v_high=v_high, v_low=v_low, step=HR_VOLTAGE_STEP)
    q_hr = poly_best(v_hr)
    dqdv_hr = np.abs(np.gradient(q_hr, v_hr))

    finite_mask = np.isfinite(v_hr) & np.isfinite(q_hr) & np.isfinite(dqdv_hr)
    non_finite_count = int((~finite_mask).sum())
    sign_consistency = float(np.mean(dqdv_hr[finite_mask] >= -1e-12)) if finite_mask.any() else float("nan")
    n_highres_points = int(len(v_hr))

    passes_physical = bool(
        n_highres_points >= MIN_HIGHRES_POINTS
        and non_finite_count == 0
        and np.isfinite(sign_consistency)
        and sign_consistency >= SIGN_CONSISTENCY_THRESHOLD
    )

    summary_row = {
        "sample_id": sample_id,
        "policy": policy,
        "cell_code": cell_code,
        "cycles": cycles,
        "file_path": str(sample_row["file_path"]),
        "best_degree": best_degree,
        "r2_q_best": float(best["r2_q"]),
        "rmse_q_best": float(best["rmse_q"]),
        "rmse_dqdv_high_best": float(best["rmse_dqdv_high"]),
        "rmse_dqdv_all_best": float(best["rmse_dqdv_all"]),
        "score_best": float(best["score"]),
        "high_sens_threshold_abs_dqdv": float(best["high_sens_threshold_abs_dqdv"]),
        "high_sens_points": int(best["high_sens_points"]),
        "sensitive_voltage_low": float(best["sensitive_voltage_low"]),
        "sensitive_voltage_high": float(best["sensitive_voltage_high"]),
        "rank_warning_best": int(best["rank_warning"]),
        "n_points_window": int(best["n_points_window"]),
        "n_points_dqdv_ref": int(best["n_points_dqdv_ref"]),
        "n_highres_points": n_highres_points,
        "non_finite_count": non_finite_count,
        "sign_consistency": sign_consistency,
        "passes_physical_checks": int(passes_physical),
    }

    curve_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "policy": policy,
            "cell_code": cell_code,
            "cycles": cycles,
            "best_degree": best_degree,
            "voltage_hr": v_hr,
            "capacity_fit": q_hr,
            "dqdv_hr": dqdv_hr,
        }
    )
    return metric_df, summary_row, curve_df


def build_promotion_decision(summary_df: pd.DataFrame) -> dict:
    """Build promotion gate decision based on five-sample experiment thresholds."""
    valid_mask = summary_df["passes_physical_checks"] == 1
    valid_df = summary_df.loc[valid_mask].copy()

    valid_samples = int(len(valid_df))
    median_r2 = float(valid_df["r2_q_best"].median()) if valid_samples > 0 else float("nan")
    median_sign = float(valid_df["sign_consistency"].median()) if valid_samples > 0 else float("nan")

    promote = bool(
        valid_samples >= PROMOTE_MIN_VALID_SAMPLES
        and np.isfinite(median_r2)
        and np.isfinite(median_sign)
        and median_r2 >= PROMOTE_MIN_MEDIAN_R2
        and median_sign >= PROMOTE_MIN_MEDIAN_SIGN_CONSISTENCY
    )
    return {
        "valid_samples": valid_samples,
        "median_r2_q_best_valid": median_r2,
        "median_sign_consistency_valid": median_sign,
        "promote_to_full": int(promote),
    }


def write_report(
    sample_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    decision: dict,
) -> None:
    """Write markdown report summarizing experiment setup and recommendation."""
    lines: List[str] = []
    lines.append("# dQ/dV Polynomial HR Experiment Report")
    lines.append("")
    lines.append("## 1. Experiment Setup")
    lines.append("")
    lines.append(f"- sample_count: **{TOTAL_SAMPLE_COUNT}**")
    lines.append(
        f"- fixed_sample: `{FIXED_POLICY} | {FIXED_CELL_CODE} | {FIXED_CYCLE}`"
    )
    lines.append(f"- random_seed: `{RANDOM_SEED}`")
    lines.append(f"- degree_range: `{DEGREE_MIN}~{DEGREE_MAX}`")
    lines.append(
        f"- high_sensitivity_voltage_window: `{SENSITIVE_VOLTAGE_LOW:.1f}V~{SENSITIVE_VOLTAGE_HIGH:.1f}V`"
    )
    lines.append("- discharge_dqdv_transform: `abs(dQ/dV)`")
    lines.append(f"- hr_voltage_step: `{HR_VOLTAGE_STEP} V`")
    lines.append("")
    lines.append("## 2. Degree Selection Score")
    lines.append("")
    lines.append(
        "- `score = 0.60*norm(RMSE_high) + 0.25*norm(1-R2_q) + 0.15*norm(RMSE_q)`"
    )
    lines.append("- tie-break: lower `RMSE_high`, then lower `degree`.")
    lines.append("")
    lines.append("## 3. Promotion Gates")
    lines.append("")
    lines.append(f"- valid_samples: `{decision['valid_samples']}` (threshold `{PROMOTE_MIN_VALID_SAMPLES}`)")
    lines.append(
        f"- median(R2_q_best): `{decision['median_r2_q_best_valid']:.6f}` "
        f"(threshold `{PROMOTE_MIN_MEDIAN_R2}`)"
    )
    lines.append(
        f"- median(nonnegative_consistency): `{decision['median_sign_consistency_valid']:.6f}` "
        f"(threshold `{PROMOTE_MIN_MEDIAN_SIGN_CONSISTENCY}`)"
    )
    lines.append(f"- promote_to_full: **{bool(decision['promote_to_full'])}**")
    lines.append("")
    lines.append("## 4. Sample Result Snapshot")
    lines.append("")

    show_cols = [
        "sample_id",
        "best_degree",
        "r2_q_best",
        "rmse_q_best",
        "rmse_dqdv_high_best",
        "sign_consistency",
        "passes_physical_checks",
    ]
    table_df = summary_df[show_cols].copy()
    table_df["r2_q_best"] = table_df["r2_q_best"].map(lambda x: f"{float(x):.6f}")
    table_df["rmse_q_best"] = table_df["rmse_q_best"].map(lambda x: f"{float(x):.6f}")
    table_df["rmse_dqdv_high_best"] = table_df["rmse_dqdv_high_best"].map(lambda x: f"{float(x):.6f}")
    table_df["sign_consistency"] = table_df["sign_consistency"].map(lambda x: f"{float(x):.6f}")
    header = "| " + " | ".join(show_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(show_cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for row in table_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    lines.append("## 5. Output Files")
    lines.append("")
    lines.append(f"- `{SAMPLE_SELECTION_PATH}`")
    lines.append(f"- `{PER_DEGREE_METRICS_PATH}`")
    lines.append(f"- `{BEST_DEGREE_SUMMARY_PATH}`")
    lines.append(f"- `{HIGHRES_CURVES_PATH}`")
    lines.append(f"- `{FIT_REPORT_PATH}`")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run 5-sample polynomial high-resolution dQ/dV experiment."""
    mod = load_extract_module()
    files = mod.find_target_files()
    candidates, frame_cache = gather_candidates(mod, files)
    selected = select_samples(mod, candidates, frame_cache)

    sample_rows: List[dict] = []
    degree_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []
    curve_frames: List[pd.DataFrame] = []

    for idx, sample in enumerate(selected, start=1):
        sample_id = make_sample_id(sample["policy"], str(sample["cell_code"]), int(sample["cycles"]))
        sample_rows.append(
            {
                "sample_order": idx,
                "sample_id": sample_id,
                "policy": sample["policy"],
                "cell_code": str(sample["cell_code"]),
                "cycles": int(sample["cycles"]),
                "file_path": str(sample["file_path"]),
                "is_fixed_sample": int(
                    sample["policy"] == FIXED_POLICY
                    and str(sample["cell_code"]) == FIXED_CELL_CODE
                    and int(sample["cycles"]) == FIXED_CYCLE
                ),
            }
        )

        frame = frame_cache[str(sample["file_path"])]
        metric_df, summary_row, curve_df = evaluate_sample(mod, frame, sample)
        degree_frames.append(metric_df)
        summary_rows.append(summary_row)
        curve_frames.append(curve_df)

    sample_df = pd.DataFrame(sample_rows).sort_values("sample_order").reset_index(drop=True)
    per_degree_df = pd.concat(degree_frames, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("sample_id").reset_index(drop=True)
    curves_df = pd.concat(curve_frames, axis=0, ignore_index=True)

    decision = build_promotion_decision(summary_df)
    summary_df["promote_to_full"] = int(decision["promote_to_full"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(SAMPLE_SELECTION_PATH, index=False, encoding="utf-8")
    per_degree_df.to_csv(PER_DEGREE_METRICS_PATH, index=False, encoding="utf-8")
    summary_df.to_csv(BEST_DEGREE_SUMMARY_PATH, index=False, encoding="utf-8")
    curves_df.to_csv(HIGHRES_CURVES_PATH, index=False, encoding="utf-8")
    write_report(sample_df, summary_df, decision)

    print(f"Saved: {SAMPLE_SELECTION_PATH}")
    print(f"Saved: {PER_DEGREE_METRICS_PATH}")
    print(f"Saved: {BEST_DEGREE_SUMMARY_PATH}")
    print(f"Saved: {HIGHRES_CURVES_PATH}")
    print(f"Saved: {FIT_REPORT_PATH}")
    print(
        "Promotion decision | "
        f"valid_samples={decision['valid_samples']} | "
        f"median_r2={decision['median_r2_q_best_valid']:.6f} | "
        f"median_sign={decision['median_sign_consistency_valid']:.6f} | "
        f"promote_to_full={bool(decision['promote_to_full'])}"
    )


if __name__ == "__main__":
    main()
