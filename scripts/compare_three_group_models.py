from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# =========================
# Config (edit here first)
# =========================
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

LIFE_PERFORMANCE_PATH = REPO_ROOT / "data" / "processed" / "life_performance.csv"
CHARGE_FEATURE_PATH = REPO_ROOT / "data" / "processed" / "charge_interval_features.csv"
TRAIN_SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "train_policy_cell_samples.csv"
VALID_SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "valid_policy_cell_samples.csv"

OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "charge_feature_q_discharge_corr"
    / "model_group_comparison"
)

ENCODING = "utf-8-sig"
RANDOM_SEED = 20260317
FIRST_OCCURRENCE_RANGE_COUNT = 1

PYTHON_ENV_HOME = Path(r"C:\Users\pal\pyenv\ds_env")
MPL_CONFIG_DIR = REPO_ROOT / "outputs" / ".mplconfig"

MODEL_PARAMS = {
    "n_estimators": 220,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}

STAGE_DEFS: list[tuple[str, str, float, float]] = [
    ("early", "早期(0%-33%)", 0.0, 1.0 / 3.0),
    ("middle", "中期(33%-67%)", 1.0 / 3.0, 2.0 / 3.0),
    ("late", "晚期(67%-100%)", 2.0 / 3.0, 1.0000001),
]


def parse_range_low(range_label: str) -> float:
    try:
        body = range_label.strip()[1:]
        return float(body.split(",")[0])
    except Exception:
        return float("inf")


def sanitize_range_label(range_label: str) -> str:
    # [3.0,3.1) -> 3p0_3p1
    out = (
        range_label.replace("[", "")
        .replace("]", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
    )
    return out


def ensure_matplotlib_config() -> List[str]:
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
    import matplotlib.pyplot as plt  # noqa: WPS433
    from matplotlib import font_manager, rcParams  # noqa: WPS433

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    selected = [f for f in candidates if f in installed]
    if not selected:
        selected = ["DejaVu Sans"]

    rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 220
    _ = plt  # keep import explicit
    return selected


def load_split_sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_SAMPLE_PATH, encoding=ENCODING)
    valid = pd.read_csv(VALID_SAMPLE_PATH, encoding=ENCODING)

    required = {
        "policy",
        "cell_code",
        "initial_c_rate",
        "switch_soc_percent",
        "post_switch_c_rate",
    }
    for name, df in [("train", train), ("valid", valid)]:
        missing = required.difference(set(df.columns))
        if missing:
            raise KeyError(f"Missing required columns in {name} split: {sorted(missing)}")

    keep_cols = [
        "policy",
        "cell_code",
        "initial_c_rate",
        "switch_soc_percent",
        "post_switch_c_rate",
        "max_cycles",
        "policy_note",
    ]
    train = train[keep_cols].copy()
    valid = valid[keep_cols].copy()
    train["set_type"] = "train"
    valid["set_type"] = "valid"
    return train, valid


def build_cycle_level_dataset(train_split: pd.DataFrame, valid_split: pd.DataFrame) -> pd.DataFrame:
    life = pd.read_csv(
        LIFE_PERFORMANCE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "q_discharge"],
    )
    charge = pd.read_csv(
        CHARGE_FEATURE_PATH,
        encoding=ENCODING,
        usecols=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"],
    )

    life["cycles"] = pd.to_numeric(life["cycles"], errors="coerce")
    life["q_discharge"] = pd.to_numeric(life["q_discharge"], errors="coerce")
    charge["cycles"] = pd.to_numeric(charge["cycles"], errors="coerce")
    charge["delta_ah"] = pd.to_numeric(charge["delta_ah"], errors="coerce")
    charge["range_count"] = pd.to_numeric(charge["range_count"], errors="coerce")

    life = life.dropna(subset=["policy", "cell_code", "cycles", "q_discharge"]).copy()
    charge = charge.dropna(subset=["policy", "cell_code", "cycles", "range", "delta_ah", "range_count"]).copy()
    life["cycles"] = life["cycles"].astype(int)
    charge["cycles"] = charge["cycles"].astype(int)
    charge["range_count"] = charge["range_count"].astype(int)

    charge = charge.loc[charge["range_count"] == FIRST_OCCURRENCE_RANGE_COUNT].copy()

    # If same cycle-range appears multiple times due to data anomalies, sum to one value.
    charge_agg = (
        charge.groupby(["policy", "cell_code", "cycles", "range"], as_index=False)
        .agg(delta_ah_sum=("delta_ah", "sum"))
    )

    range_order = sorted(charge_agg["range"].dropna().unique().tolist(), key=parse_range_low)
    charge_agg["feature_name"] = charge_agg["range"].map(
        lambda r: f"charge_delta_ah_{sanitize_range_label(str(r))}"
    )
    charge_wide = (
        charge_agg.pivot_table(
            index=["policy", "cell_code", "cycles"],
            columns="feature_name",
            values="delta_ah_sum",
            aggfunc="mean",
        )
        .reset_index()
    )
    feature_order = [f"charge_delta_ah_{sanitize_range_label(str(r))}" for r in range_order]
    charge_feature_cols = [c for c in feature_order if c in charge_wide.columns]
    charge_wide = charge_wide[["policy", "cell_code", "cycles"] + charge_feature_cols]

    split_map = pd.concat([train_split, valid_split], axis=0, ignore_index=True)
    split_map = split_map.drop_duplicates(subset=["policy", "cell_code"], keep="first")
    split_map["initial_c_rate"] = pd.to_numeric(split_map["initial_c_rate"], errors="coerce")
    split_map["switch_soc_percent"] = pd.to_numeric(split_map["switch_soc_percent"], errors="coerce")
    split_map["post_switch_c_rate"] = pd.to_numeric(split_map["post_switch_c_rate"], errors="coerce")

    dataset = (
        life.merge(charge_wide, on=["policy", "cell_code", "cycles"], how="inner")
        .merge(
            split_map[
                [
                    "policy",
                    "cell_code",
                    "initial_c_rate",
                    "switch_soc_percent",
                    "post_switch_c_rate",
                    "max_cycles",
                    "set_type",
                ]
            ],
            on=["policy", "cell_code"],
            how="inner",
            validate="many_to_one",
        )
        .copy()
    )
    dataset["max_cycles"] = pd.to_numeric(dataset["max_cycles"], errors="coerce")
    dataset = dataset.dropna(subset=["max_cycles"]).copy()
    dataset["cycle_ratio"] = dataset["cycles"] / dataset["max_cycles"]
    dataset["cycle_ratio"] = dataset["cycle_ratio"].clip(lower=0.0, upper=1.0000001)
    dataset["range_feature_count_non_null"] = dataset[charge_feature_cols].notna().sum(axis=1)
    return dataset


def assign_lifecycle_stage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lifecycle_stage"] = "unknown"
    out["lifecycle_stage_label"] = "未知"
    for stage_key, stage_label, lo, hi in STAGE_DEFS:
        mask = (out["cycle_ratio"] >= lo) & (out["cycle_ratio"] < hi)
        out.loc[mask, "lifecycle_stage"] = stage_key
        out.loc[mask, "lifecycle_stage_label"] = stage_label
    return out


@dataclass
class EvalResult:
    model_name: str
    feature_group: str
    train_rows: int
    valid_rows: int
    n_features: int
    mae: float
    rmse: float
    r2: float
    mape: float


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    safe_den = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_den)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train_and_eval(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_group: str,
) -> tuple[EvalResult, np.ndarray]:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(**MODEL_PARAMS)),
        ]
    )

    X_train = train_df[list(feature_cols)].to_numpy(dtype=float)
    y_train = train_df["q_discharge"].to_numpy(dtype=float)
    X_valid = valid_df[list(feature_cols)].to_numpy(dtype=float)
    y_valid = valid_df["q_discharge"].to_numpy(dtype=float)

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    m = calc_metrics(y_valid, pred)

    result = EvalResult(
        model_name="random_forest",
        feature_group=feature_group,
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        n_features=len(feature_cols),
        mae=m["mae"],
        rmse=m["rmse"],
        r2=m["r2"],
        mape=m["mape"],
    )
    return result, pred


def calc_policy_level_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for policy, part in df.groupby("policy", sort=False):
        y_true = part["q_discharge"].to_numpy(dtype=float)
        y_pred = part[pred_col].to_numpy(dtype=float)
        if len(part) < 3:
            continue
        m = calc_metrics(y_true, y_pred)
        rows.append(
            {
                "policy": policy,
                "n_rows": len(part),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "mape": m["mape"],
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)


def build_uplift_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    idx = {row["feature_group"]: row for _, row in metrics_df.iterrows()}
    comb = idx["combined_charge_plus_policy"]
    out_rows: list[dict] = []
    for base_name in ["policy_only", "charge_only"]:
        base = idx[base_name]
        out_rows.append(
            {
                "compare_to": base_name,
                "delta_mae": comb["mae"] - base["mae"],
                "mae_improve_pct": (base["mae"] - comb["mae"]) / max(base["mae"], 1e-12),
                "delta_rmse": comb["rmse"] - base["rmse"],
                "rmse_improve_pct": (base["rmse"] - comb["rmse"]) / max(base["rmse"], 1e-12),
                "delta_r2": comb["r2"] - base["r2"],
                "delta_mape": comb["mape"] - base["mape"],
                "mape_improve_pct": (base["mape"] - comb["mape"]) / max(base["mape"], 1e-12),
            }
        )
    return pd.DataFrame(out_rows)


def calc_stage_metrics_from_predictions(
    pred_df: pd.DataFrame,
    train_stage_counts: Dict[str, int],
) -> pd.DataFrame:
    model_cols = {
        "charge_only": "pred_charge_only",
        "policy_only": "pred_policy_only",
        "combined_charge_plus_policy": "pred_combined_charge_plus_policy",
    }
    stage_label_map = {k: v for k, v, _, _ in STAGE_DEFS}

    rows: list[dict] = []
    for stage_key, stage_label, _, _ in STAGE_DEFS:
        part = pred_df.loc[pred_df["lifecycle_stage"] == stage_key].copy()
        if part.empty:
            continue
        y_true = part["q_discharge"].to_numpy(dtype=float)
        for feature_group, pred_col in model_cols.items():
            y_pred = part[pred_col].to_numpy(dtype=float)
            m = calc_metrics(y_true, y_pred)
            rows.append(
                {
                    "stage_key": stage_key,
                    "stage_label": stage_label_map.get(stage_key, stage_label),
                    "feature_group": feature_group,
                    "n_train_stage": int(train_stage_counts.get(stage_key, 0)),
                    "n_valid_stage": int(len(part)),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "mape": m["mape"],
                }
            )
    return pd.DataFrame(rows)


def build_stage_uplift_table(stage_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if stage_metrics_df.empty:
        return pd.DataFrame(rows)

    for stage_key, stage_label, _, _ in STAGE_DEFS:
        part = stage_metrics_df.loc[stage_metrics_df["stage_key"] == stage_key]
        if part.empty:
            continue
        idx = {row["feature_group"]: row for _, row in part.iterrows()}
        if "combined_charge_plus_policy" not in idx:
            continue
        comb = idx["combined_charge_plus_policy"]
        for base_name in ["policy_only", "charge_only"]:
            if base_name not in idx:
                continue
            base = idx[base_name]
            rows.append(
                {
                    "stage_key": stage_key,
                    "stage_label": stage_label,
                    "compare_to": base_name,
                    "n_valid_stage": int(comb["n_valid_stage"]),
                    "delta_mae": comb["mae"] - base["mae"],
                    "mae_improve_pct": (base["mae"] - comb["mae"]) / max(base["mae"], 1e-12),
                    "delta_rmse": comb["rmse"] - base["rmse"],
                    "rmse_improve_pct": (base["rmse"] - comb["rmse"]) / max(base["rmse"], 1e-12),
                    "delta_r2": comb["r2"] - base["r2"],
                    "delta_mape": comb["mape"] - base["mape"],
                    "mape_improve_pct": (base["mape"] - comb["mape"]) / max(base["mape"], 1e-12),
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    metrics_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    out_metrics_png: Path,
    out_uplift_png: Path,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    order = ["charge_only", "policy_only", "combined_charge_plus_policy"]
    label_map = {
        "charge_only": "仅充电特征",
        "policy_only": "仅policy三元参数",
        "combined_charge_plus_policy": "充电+policy三元参数",
    }
    mdf = metrics_df.set_index("feature_group").loc[order].reset_index()
    labels = [label_map[x] for x in mdf["feature_group"]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    axes[0].bar(x, mdf["r2"], color=["#60a5fa", "#fbbf24", "#34d399"])
    axes[0].set_title("R²（越大越好）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, mdf["mae"], color=["#60a5fa", "#fbbf24", "#34d399"])
    axes[1].set_title("MAE（越小越好）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    axes[2].bar(x, mdf["rmse"], color=["#60a5fa", "#fbbf24", "#34d399"])
    axes[2].set_title("RMSE（越小越好）")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("三组模型验证集指标对比（RandomForest）")
    fig.tight_layout()
    fig.savefig(out_metrics_png, format="png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9.2, 4.8))
    labels2 = ["vs policy_only", "vs charge_only"]
    y1 = uplift_df["mae_improve_pct"].to_numpy(dtype=float) * 100.0
    y2 = uplift_df["rmse_improve_pct"].to_numpy(dtype=float) * 100.0
    y3 = uplift_df["delta_r2"].to_numpy(dtype=float)

    xx = np.arange(len(labels2))
    w = 0.24
    ax2.bar(xx - w, y1, width=w, label="MAE改善(%)", color="#22c55e")
    ax2.bar(xx, y2, width=w, label="RMSE改善(%)", color="#0ea5e9")
    ax2.bar(xx + w, y3, width=w, label="R²提升(绝对值)", color="#f97316")
    ax2.axhline(0, color="#64748b", linewidth=1)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels2)
    ax2.set_title("组合模型相对基线模型的增益")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_uplift_png, format="png")
    plt.close(fig2)


def save_stage_plots(
    stage_metrics_df: pd.DataFrame,
    stage_uplift_df: pd.DataFrame,
    out_stage_metrics_png: Path,
    out_stage_uplift_png: Path,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    if stage_metrics_df.empty:
        return

    label_map = {
        "charge_only": "仅充电特征",
        "policy_only": "仅policy三元参数",
        "combined_charge_plus_policy": "充电+policy三元参数",
    }
    stage_order = [k for k, _, _, _ in STAGE_DEFS]
    stage_label_map = {k: v for k, v, _, _ in STAGE_DEFS}
    model_order = ["charge_only", "policy_only", "combined_charge_plus_policy"]
    colors = {"charge_only": "#60a5fa", "policy_only": "#fbbf24", "combined_charge_plus_policy": "#34d399"}

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    metric_names = [("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE")]
    xx = np.arange(len(stage_order))
    w = 0.22
    for ax, (metric, metric_label) in zip(axes, metric_names):
        for i, model_name in enumerate(model_order):
            vals = []
            for sk in stage_order:
                part = stage_metrics_df[
                    (stage_metrics_df["stage_key"] == sk) & (stage_metrics_df["feature_group"] == model_name)
                ]
                vals.append(float(part.iloc[0][metric]) if not part.empty else np.nan)
            ax.bar(xx + (i - 1) * w, vals, width=w, label=label_map[model_name], color=colors[model_name])
        ax.set_xticks(xx)
        ax.set_xticklabels([stage_label_map[k] for k in stage_order], rotation=8)
        ax.set_title(f"{metric_label} 分段对比")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if metric == "r2":
            ax.axhline(0, color="#64748b", linewidth=1)

    axes[2].legend(loc="best")
    fig.suptitle("按寿命阶段分段的三组模型表现")
    fig.tight_layout()
    fig.savefig(out_stage_metrics_png, format="png")
    plt.close(fig)

    if stage_uplift_df.empty:
        return

    fig2, ax2 = plt.subplots(figsize=(10.6, 5.0))
    pairs = [("early", "早期"), ("middle", "中期"), ("late", "晚期")]
    xx2 = np.arange(len(pairs))
    w2 = 0.18
    for i, base_name in enumerate(["policy_only", "charge_only"]):
        sub = stage_uplift_df.loc[stage_uplift_df["compare_to"] == base_name].copy()
        vals = []
        for sk, _ in pairs:
            p = sub.loc[sub["stage_key"] == sk]
            vals.append(float(p.iloc[0]["delta_r2"]) if not p.empty else np.nan)
        ax2.bar(xx2 + (i - 0.5) * w2, vals, width=w2, label=f"R²提升 vs {base_name}")

    ax2.set_xticks(xx2)
    ax2.set_xticklabels([x[1] for x in pairs])
    ax2.set_title("分段增益：组合模型相对基线模型的 R² 提升")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.axhline(0, color="#64748b", linewidth=1)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_stage_uplift_png, format="png")
    plt.close(fig2)


def render_report(
    report_path: Path,
    font_list: List[str],
    dataset: pd.DataFrame,
    metrics_df: pd.DataFrame,
    uplift_df: pd.DataFrame,
    policy_metrics_map: Dict[str, pd.DataFrame],
    charge_feature_cols: Sequence[str],
) -> None:
    label_map = {
        "charge_only": "仅充电特征",
        "policy_only": "仅policy三元参数",
        "combined_charge_plus_policy": "充电+policy三元参数",
    }
    m = metrics_df.set_index("feature_group")
    comb = m.loc["combined_charge_plus_policy"]

    best_r2_name = m["r2"].idxmax()
    best_mae_name = m["mae"].idxmin()

    lines: list[str] = []
    lines.append("# 三组模型对比与增益报告")
    lines.append("")
    lines.append("## 1. 任务目标")
    lines.append("- 比较三组特征方案对放电容量 `q_discharge` 的预测效果。")
    lines.append("- 在当前 `policy + cell_code` 训练/验证划分下，评估“充电特征 + policy三元参数”是否带来增益。")
    lines.append("")
    lines.append("## 2. 数据与口径")
    lines.append(f"- 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Python 环境入口目录：`{PYTHON_ENV_HOME}`（`pipenv run python`）")
    lines.append(f"- Python 解释器：`{os.path.realpath(os.sys.executable)}`")
    lines.append(f"- 中文字体回退链（已检测）：`{', '.join(font_list)}`")
    lines.append(f"- 首次出现口径：仅使用 `range_count == {FIRST_OCCURRENCE_RANGE_COUNT}` 的充电区间特征。")
    lines.append(f"- 样本总行数（cycle级）：**{len(dataset):,}**")
    lines.append(
        f"- 训练/验证行数：**{int((dataset['set_type'] == 'train').sum()):,} / {int((dataset['set_type'] == 'valid').sum()):,}**"
    )
    lines.append(
        f"- 充电特征维度：**{len(charge_feature_cols)}**（区间特征）；policy特征维度：**3**。"
    )
    lines.append("")
    lines.append("## 3. 模型设置")
    lines.append("- 模型算法：`RandomForestRegressor`（同一超参数，保证三组结果可比）。")
    lines.append(f"- 关键参数：`{MODEL_PARAMS}`")
    lines.append("- 评估指标：`R²`、`MAE`、`RMSE`、`MAPE`（验证集）。")
    lines.append("")
    lines.append("## 4. 三组模型结果")
    lines.append("| 方案 | R² | MAE | RMSE | MAPE |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {label_map[row['feature_group']]} | {row['r2']:.5f} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.4%} |"
        )
    lines.append("")
    lines.append(
        f"- 最优 R² 方案：**{label_map[best_r2_name]}**；最优 MAE 方案：**{label_map[best_mae_name]}**。"
    )
    lines.append("")
    lines.append("## 5. 增益分析（组合模型）")
    lines.append("| 对比基线 | MAE改善(%) | RMSE改善(%) | R²提升(绝对值) | MAPE改善(%) |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in uplift_df.iterrows():
        lines.append(
            f"| {row['compare_to']} | {row['mae_improve_pct']:.2%} | {row['rmse_improve_pct']:.2%} | {row['delta_r2']:.5f} | {row['mape_improve_pct']:.2%} |"
        )
    lines.append("")

    comb_policy = policy_metrics_map["combined_charge_plus_policy"]
    policy_only = policy_metrics_map["policy_only"]
    common_policy = comb_policy.merge(
        policy_only, on="policy", suffixes=("_combined", "_policy_only"), how="inner"
    )
    if not common_policy.empty:
        common_policy["delta_r2"] = common_policy["r2_combined"] - common_policy["r2_policy_only"]
        improve_ratio = float(np.mean(common_policy["delta_r2"] > 0))
        lines.append("## 6. 按Policy分层的增益情况")
        lines.append(
            f"- 在验证集中，组合模型相对 `policy_only` 的 R² 改善 policy 占比：**{improve_ratio:.2%}**。"
        )
        lines.append(
            f"- R²提升中位数：**{common_policy['delta_r2'].median():.5f}**，"
            f"分位数 Q25/Q75：**{common_policy['delta_r2'].quantile(0.25):.5f} / {common_policy['delta_r2'].quantile(0.75):.5f}**。"
        )
        lines.append("")

    lines.append("## 7. 图表")
    lines.append("![三组模型指标对比](./model_metrics_comparison.png)")
    lines.append("")
    lines.append("![组合模型增益](./model_uplift_comparison.png)")
    lines.append("")
    lines.append("## 8. 结论与建议")
    lines.append(
        f"- 组合特征模型（充电+policy）验证集 R²={comb['r2']:.5f}，说明将 `policy` 三元参数与充电特征联合建模具备实际价值。"
    )
    lines.append("- 建议在后续训练中保留该组合，并继续做时间外推验证与按 policy 稳健性监控。")
    lines.append("- 当前报告是对比与增益分析，不替换既有相关性报告。")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def render_lifecycle_report(
    report_path: Path,
    stage_metrics_df: pd.DataFrame,
    stage_uplift_df: pd.DataFrame,
) -> None:
    label_map = {
        "charge_only": "仅充电特征",
        "policy_only": "仅policy三元参数",
        "combined_charge_plus_policy": "充电+policy三元参数",
    }
    stage_label_map = {k: v for k, v, _, _ in STAGE_DEFS}
    stage_order = [k for k, _, _, _ in STAGE_DEFS]

    lines: list[str] = []
    lines.append("# 按寿命阶段分段的三组模型增益报告")
    lines.append("")
    lines.append("## 1. 评估口径")
    lines.append("- 阶段定义基于循环进度：`cycle_ratio = cycles / max_cycles`。")
    lines.append("- 早期：0%-33%；中期：33%-67%；晚期：67%-100%。")
    lines.append("- 方法：三组模型先做全局训练，再在验证集按寿命阶段分段评估。")
    lines.append("")
    lines.append("## 2. 分段模型指标")
    lines.append("| 阶段 | 方案 | n_train | n_valid | R² | MAE | RMSE | MAPE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for stage_key in stage_order:
        part = stage_metrics_df.loc[stage_metrics_df["stage_key"] == stage_key].copy()
        if part.empty:
            continue
        part = part.set_index("feature_group").reindex(["charge_only", "policy_only", "combined_charge_plus_policy"]).reset_index()
        for _, r in part.iterrows():
            lines.append(
                f"| {stage_label_map.get(stage_key, stage_key)} | {label_map[r['feature_group']]} | "
                f"{int(r['n_train_stage'])} | {int(r['n_valid_stage'])} | {r['r2']:.5f} | "
                f"{r['mae']:.6f} | {r['rmse']:.6f} | {r['mape']:.4%} |"
            )
    lines.append("")
    lines.append("## 3. 分段增益（组合模型相对基线）")
    lines.append("| 阶段 | 对比基线 | MAE改善(%) | RMSE改善(%) | R²提升 | MAPE改善(%) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for stage_key in stage_order:
        part = stage_uplift_df.loc[stage_uplift_df["stage_key"] == stage_key]
        if part.empty:
            continue
        for _, r in part.iterrows():
            lines.append(
                f"| {stage_label_map.get(stage_key, stage_key)} | {r['compare_to']} | "
                f"{r['mae_improve_pct']:.2%} | {r['rmse_improve_pct']:.2%} | "
                f"{r['delta_r2']:.5f} | {r['mape_improve_pct']:.2%} |"
            )
    lines.append("")
    lines.append("## 4. 图表")
    lines.append("![分段模型指标](./lifecycle_stage_model_metrics.png)")
    lines.append("")
    lines.append("![分段增益图](./lifecycle_stage_uplift.png)")
    lines.append("")
    lines.append("## 5. 解读建议")
    lines.append("- 优先关注组合模型在晚期阶段是否仍显著优于基线模型。")
    lines.append("- 若某阶段增益明显下降，建议针对该阶段补充特征或单独建模。")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font_list = ensure_matplotlib_config()

    train_split, valid_split = load_split_sample_tables()
    dataset = build_cycle_level_dataset(train_split, valid_split)
    dataset = assign_lifecycle_stage(dataset)
    if dataset.empty:
        raise RuntimeError("Cycle-level dataset is empty after merges.")

    train_df = dataset.loc[dataset["set_type"] == "train"].copy()
    valid_df = dataset.loc[dataset["set_type"] == "valid"].copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train/valid dataset is empty after split mapping.")

    charge_feature_cols = sorted([c for c in dataset.columns if c.startswith("charge_delta_ah_")])
    policy_feature_cols = ["initial_c_rate", "switch_soc_percent", "post_switch_c_rate"]
    combined_cols = charge_feature_cols + policy_feature_cols

    result_rows: list[EvalResult] = []
    pred_df = valid_df[
        ["policy", "cell_code", "cycles", "cycle_ratio", "lifecycle_stage", "lifecycle_stage_label", "q_discharge"]
    ].copy()
    policy_metrics_map: dict[str, pd.DataFrame] = {}

    for group_name, cols in [
        ("charge_only", charge_feature_cols),
        ("policy_only", policy_feature_cols),
        ("combined_charge_plus_policy", combined_cols),
    ]:
        result, pred = train_and_eval(train_df, valid_df, cols, group_name)
        result_rows.append(result)
        pred_col = f"pred_{group_name}"
        pred_df[pred_col] = pred
        pm = calc_policy_level_metrics(valid_df.assign(**{pred_col: pred}), pred_col)
        pm["feature_group"] = group_name
        policy_metrics_map[group_name] = pm

    metrics_df = pd.DataFrame([r.__dict__ for r in result_rows]).sort_values("feature_group")
    uplift_df = build_uplift_table(metrics_df)
    policy_metrics_df = pd.concat(policy_metrics_map.values(), axis=0, ignore_index=True)

    train_stage_counts = train_df["lifecycle_stage"].value_counts().to_dict()
    stage_metrics_df = calc_stage_metrics_from_predictions(pred_df, train_stage_counts)
    stage_uplift_df = build_stage_uplift_table(stage_metrics_df)

    out_metrics_csv = OUTPUT_DIR / "model_group_metrics.csv"
    out_uplift_csv = OUTPUT_DIR / "model_uplift_summary.csv"
    out_pred_csv = OUTPUT_DIR / "valid_predictions_by_model.csv"
    out_policy_csv = OUTPUT_DIR / "policy_level_metrics_by_model.csv"
    out_metrics_png = OUTPUT_DIR / "model_metrics_comparison.png"
    out_uplift_png = OUTPUT_DIR / "model_uplift_comparison.png"
    out_report_md = OUTPUT_DIR / "report_three_group_model_comparison.md"
    out_stage_metrics_csv = OUTPUT_DIR / "lifecycle_stage_metrics_by_model.csv"
    out_stage_uplift_csv = OUTPUT_DIR / "lifecycle_stage_uplift_summary.csv"
    out_stage_metrics_png = OUTPUT_DIR / "lifecycle_stage_model_metrics.png"
    out_stage_uplift_png = OUTPUT_DIR / "lifecycle_stage_uplift.png"
    out_stage_report_md = OUTPUT_DIR / "report_three_group_model_comparison_by_lifecycle.md"

    metrics_df.to_csv(out_metrics_csv, index=False, encoding="utf-8")
    uplift_df.to_csv(out_uplift_csv, index=False, encoding="utf-8")
    pred_df.to_csv(out_pred_csv, index=False, encoding="utf-8")
    policy_metrics_df.to_csv(out_policy_csv, index=False, encoding="utf-8")
    stage_metrics_df.to_csv(out_stage_metrics_csv, index=False, encoding="utf-8")
    stage_uplift_df.to_csv(out_stage_uplift_csv, index=False, encoding="utf-8")
    save_plots(metrics_df, uplift_df, out_metrics_png, out_uplift_png)
    save_stage_plots(stage_metrics_df, stage_uplift_df, out_stage_metrics_png, out_stage_uplift_png)
    render_report(
        report_path=out_report_md,
        font_list=font_list,
        dataset=dataset,
        metrics_df=metrics_df,
        uplift_df=uplift_df,
        policy_metrics_map=policy_metrics_map,
        charge_feature_cols=charge_feature_cols,
    )
    render_lifecycle_report(
        report_path=out_stage_report_md,
        stage_metrics_df=stage_metrics_df,
        stage_uplift_df=stage_uplift_df,
    )

    print(f"Saved: {out_metrics_csv}")
    print(f"Saved: {out_uplift_csv}")
    print(f"Saved: {out_pred_csv}")
    print(f"Saved: {out_policy_csv}")
    print(f"Saved: {out_metrics_png}")
    print(f"Saved: {out_uplift_png}")
    print(f"Saved: {out_report_md}")
    print(f"Saved: {out_stage_metrics_csv}")
    print(f"Saved: {out_stage_uplift_csv}")
    print(f"Saved: {out_stage_metrics_png}")
    print(f"Saved: {out_stage_uplift_png}")
    print(f"Saved: {out_stage_report_md}")
    print(
        f"Rows train/valid: {len(train_df)}/{len(valid_df)} | "
        f"features charge/policy/combined: {len(charge_feature_cols)}/{len(policy_feature_cols)}/{len(combined_cols)}"
    )


if __name__ == "__main__":
    main()
