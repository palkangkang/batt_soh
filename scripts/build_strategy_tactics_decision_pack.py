from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_STRATEGIC_DIR = REPO_ROOT / "outputs" / "analysis" / "causal_rate_temp_mediation"
DEFAULT_TACTICAL_DIR = REPO_ROOT / "outputs" / "analysis" / "charge_bin_substitution_causal"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "strategy_tactics_closed_loop"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for strategy-tactics fusion workflow."""

    parser = argparse.ArgumentParser(
        description="Build two-layer strategy/tactics decision pack from causal outputs."
    )
    parser.add_argument("--strategic-dir", type=Path, default=DEFAULT_STRATEGIC_DIR)
    parser.add_argument("--tactical-dir", type=Path, default=DEFAULT_TACTICAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-experiment-arms", type=int, default=3)
    parser.add_argument("--q-min", type=float, default=0.3)
    parser.add_argument("--q-max", type=float, default=1.3)
    parser.add_argument("--report-style", type=str, default="paper_zh_layman")
    parser.add_argument("--appendix-level", type=str, default="full")
    parser.add_argument("--encoding", type=str, default="utf-8")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Ensure output directory exists."""

    path.mkdir(parents=True, exist_ok=True)


def ensure_matplotlib_config() -> None:
    """Configure matplotlib backend and cache directory."""

    mpl_dir = REPO_ROOT / "outputs" / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib  # noqa: WPS433

    matplotlib.use("Agg")


def parse_interval_label(text: str) -> Tuple[float, float]:
    """Parse interval label to numeric bounds."""

    value = str(text).strip()
    if value == "":
        return float("nan"), float("nan")
    if value.startswith("<"):
        upper = float(re.findall(r"[-+]?\d*\.?\d+", value)[0])
        return 0.0, upper
    if value.startswith(">="):
        lower = float(re.findall(r"[-+]?\d*\.?\d+", value)[0])
        return lower, float("inf")
    if value.startswith("[") and "," in value:
        nums = re.findall(r"[-+]?\d*\.?\d+", value)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    if "-" in value and value.endswith("C"):
        core = value[:-1]
        parts = core.split("-")
        if len(parts) >= 2:
            left = parts[0].strip()
            right = parts[1].strip()
            if left != "" and right != "":
                return float(left), float(right)
    nums = re.findall(r"[-+]?\d*\.?\d+", value)
    if len(nums) == 1:
        x = float(nums[0])
        return x, x
    return float("nan"), float("nan")


def safe_float(value: object) -> float:
    """Convert to float with NaN fallback."""

    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def safe_int(value: object, default: int = 0) -> int:
    """Convert to int with fallback when value is not finite."""

    number = safe_float(value)
    if not np.isfinite(number):
        return int(default)
    return int(number)


def interval_midpoint(lower: float, upper: float) -> float:
    """Compute interval midpoint with infinity handling."""

    if np.isfinite(lower) and np.isfinite(upper):
        return float((lower + upper) / 2.0)
    if np.isfinite(lower) and (not np.isfinite(upper)):
        return float(lower + 0.6)
    if (not np.isfinite(lower)) and np.isfinite(upper):
        return float(max(0.0, upper - 0.6))
    return float("nan")


def infer_evidence_tier(
    q_value: float,
    ci_low: float,
    ci_high: float,
    support_var: float,
) -> str:
    """Infer evidence tier A/B/C."""

    if not np.isfinite(q_value) or not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "C"
    ci_width = ci_high - ci_low
    ci_cross_zero = bool(ci_low <= 0.0 <= ci_high)
    if (q_value <= 0.05) and (not ci_cross_zero) and (support_var >= 1e-3) and (ci_width <= 3e-4):
        return "A"
    if (q_value <= 0.10) and (not ci_cross_zero) and (support_var >= 1e-4) and (ci_width <= 8e-4):
        return "B"
    return "C"


def infer_strategic_risk(te_r: float, ci_low: float) -> str:
    """Map strategic risk from TE estimate and lower CI."""

    if not np.isfinite(te_r) or not np.isfinite(ci_low):
        return "未知"
    if (ci_low > 0.0) and (te_r >= 0.012):
        return "高"
    if (ci_low > 0.0) and (te_r >= 0.006):
        return "中"
    if ci_low > 0.0:
        return "中低"
    return "低"


def infer_strategic_evidence(te_r: float, ci_low: float, ci_high: float) -> str:
    """Infer strategic evidence tier."""

    if not np.isfinite(te_r) or not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "C"
    width = ci_high - ci_low
    if (ci_low > 0.0) and (te_r >= 0.01) and (width <= 0.01):
        return "A"
    if ci_low > 0.0:
        return "B"
    return "C"


def infer_strategic_action(risk_level: str) -> str:
    """Map risk level to strategic action."""

    mapping = {
        "高": "收紧倍率上限并降低该段暴露",
        "中": "保守控制并持续监测",
        "中低": "可小幅优化并保留监测",
        "低": "可维持现状",
        "未知": "证据不足，先补数",
    }
    return mapping.get(risk_level, "证据不足，先补数")


def build_strategic_layer_decisions(
    global_df: pd.DataFrame,
    rate_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """Build strategic decision table and suggested C-rate upper cap."""

    _ = global_df
    part = rate_df[
        (rate_df["scenario"] == "baseline_tplus1_temp0_70")
        & (rate_df["treatment_mode"] == "window_mean")
    ].copy()
    if part.empty:
        raise ValueError("No strategic baseline rate-bin rows found.")
    if "rate_bin_order" in part.columns:
        part = part.sort_values("rate_bin_order", kind="mergesort")
    else:
        part = part.sort_values("rate_bin_label", kind="mergesort")

    lows: List[float] = []
    uppers: List[float] = []
    risk_levels: List[str] = []
    evidence_tiers: List[str] = []
    actions: List[str] = []
    for row in part.itertuples(index=False):
        lower, upper = parse_interval_label(str(row.rate_bin_label))
        te_r = safe_float(getattr(row, "te_r"))
        ci_low = safe_float(getattr(row, "te_r_ci_low"))
        ci_high = safe_float(getattr(row, "te_r_ci_high"))
        risk = infer_strategic_risk(te_r=te_r, ci_low=ci_low)
        tier = infer_strategic_evidence(te_r=te_r, ci_low=ci_low, ci_high=ci_high)
        lows.append(lower)
        uppers.append(upper)
        risk_levels.append(risk)
        evidence_tiers.append(tier)
        actions.append(infer_strategic_action(risk))

    part["rate_lower_c"] = lows
    part["rate_upper_c"] = uppers
    part["risk_level"] = risk_levels
    part["evidence_tier"] = evidence_tiers
    part["recommended_action"] = actions

    high_rows = part[part["risk_level"] == "高"].copy()
    if not high_rows.empty:
        strategic_limit = float(
            pd.to_numeric(high_rows["rate_lower_c"], errors="coerce").dropna().min()
        )
    else:
        strategic_limit = float("nan")
    part["recommended_c_rate_upper"] = strategic_limit

    keep_cols = [
        "rate_bin",
        "rate_bin_label",
        "rate_lower_c",
        "rate_upper_c",
        "te_r",
        "te_r_ci_low",
        "te_r_ci_high",
        "nie_share",
        "n_rows",
        "n_clusters",
        "risk_level",
        "evidence_tier",
        "recommended_action",
        "recommended_c_rate_upper",
    ]
    return part[keep_cols].reset_index(drop=True), strategic_limit


def map_rate_to_strategic_risk(
    rate_mid_c: float,
    strategic_df: pd.DataFrame,
) -> Tuple[str, str]:
    """Map tactical bin rate midpoint to strategic risk/evidence tiers."""

    if not np.isfinite(rate_mid_c):
        return "未知", "C"
    for row in strategic_df.itertuples(index=False):
        lo = safe_float(getattr(row, "rate_lower_c"))
        hi = safe_float(getattr(row, "rate_upper_c"))
        if np.isfinite(lo) and np.isfinite(hi):
            if (rate_mid_c >= lo) and (rate_mid_c < hi):
                return str(row.risk_level), str(row.evidence_tier)
        elif np.isfinite(lo) and (not np.isfinite(hi)):
            if rate_mid_c >= lo:
                return str(row.risk_level), str(row.evidence_tier)
    return "未知", "C"


def infer_tactical_action(q_value: float, ci_low: float, ci_high: float) -> str:
    """Infer tactical action from significance and confidence interval."""

    if np.isfinite(q_value) and (q_value <= 0.10) and np.isfinite(ci_low) and (ci_low > 0.0):
        return "增加份额"
    if np.isfinite(q_value) and (q_value <= 0.10) and np.isfinite(ci_high) and (ci_high < 0.0):
        return "降低份额"
    return "观察验证"


def infer_strategy_compatibility(
    action: str,
    rate_lower: float,
    rate_upper: float,
    strategic_limit: float,
) -> str:
    """Check compatibility between tactical action and strategic cap."""

    if (not np.isfinite(strategic_limit)) or (action != "增加份额"):
        return "兼容"
    if np.isfinite(rate_lower) and (rate_lower >= strategic_limit):
        return "冲突"
    if np.isfinite(rate_upper) and (rate_upper > strategic_limit):
        return "部分冲突"
    return "兼容"


def build_tactical_layer_decisions(
    tactical_causal_df: pd.DataFrame,
    top_df: pd.DataFrame,
    strategic_df: pd.DataFrame,
    strategic_limit: float,
    top_k: int,
) -> pd.DataFrame:
    """Build tactical decision table for top-k bins."""

    top = top_df.sort_values("rank_combined", kind="mergesort").head(int(top_k)).copy()
    merged = top.merge(
        tactical_causal_df[
            [
                "cross_bin",
                "effect_per_1pp_ah",
                "effect_per_5pp_ah",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
                "var_treatment",
            ]
        ],
        on="cross_bin",
        how="left",
        validate="one_to_one",
    )

    rate_lows: List[float] = []
    rate_uppers: List[float] = []
    rate_mids: List[float] = []
    actions: List[str] = []
    tiers: List[str] = []
    supports: List[str] = []
    compatibilities: List[str] = []
    strat_risks: List[str] = []
    strat_tiers: List[str] = []
    classes: List[str] = []
    priorities: List[float] = []
    for row in merged.itertuples(index=False):
        lo, hi = parse_interval_label(str(row.rate_label))
        mid = interval_midpoint(lo, hi)
        q_value = safe_float(row.q_value)
        ci_low = safe_float(row.ci_low)
        ci_high = safe_float(row.ci_high)
        support_var = safe_float(row.var_treatment)
        effect_1pp = safe_float(row.effect_per_1pp_ah)

        action = infer_tactical_action(q_value=q_value, ci_low=ci_low, ci_high=ci_high)
        tier = infer_evidence_tier(
            q_value=q_value, ci_low=ci_low, ci_high=ci_high, support_var=support_var
        )
        if support_var >= 1e-3:
            support = "高"
        elif support_var >= 1e-4:
            support = "中"
        else:
            support = "低"
        compatibility = infer_strategy_compatibility(
            action=action, rate_lower=lo, rate_upper=hi, strategic_limit=strategic_limit
        )
        strat_risk, strat_tier = map_rate_to_strategic_risk(rate_mid_c=mid, strategic_df=strategic_df)

        if (action == "增加份额") and (compatibility in {"冲突", "部分冲突"}):
            decision_class = "禁止外推"
        elif (action in {"增加份额", "降低份额"}) and (tier in {"A", "B"}) and (compatibility == "兼容"):
            decision_class = "可上线"
        else:
            decision_class = "待验证"

        tier_score = {"A": 3.0, "B": 2.0, "C": 1.0}.get(tier, 1.0)
        action_score = 2.0 if action in {"增加份额", "降低份额"} else 1.0
        compatibility_score = {"兼容": 2.0, "部分冲突": 1.0, "冲突": 0.0}.get(compatibility, 1.0)
        class_penalty = {"可上线": 0.0, "待验证": 20.0, "禁止外推": 40.0}.get(decision_class, 20.0)
        priority = float(
            1000.0
            - class_penalty
            + tier_score * 10.0
            + action_score * 4.0
            + compatibility_score * 3.0
            + abs(effect_1pp) * 1.0e6
        )

        rate_lows.append(lo)
        rate_uppers.append(hi)
        rate_mids.append(mid)
        actions.append(action)
        tiers.append(tier)
        supports.append(support)
        compatibilities.append(compatibility)
        strat_risks.append(strat_risk)
        strat_tiers.append(strat_tier)
        classes.append(decision_class)
        priorities.append(priority)

    merged["rate_lower_c"] = rate_lows
    merged["rate_upper_c"] = rate_uppers
    merged["rate_mid_c"] = rate_mids
    merged["tactical_action"] = actions
    merged["evidence_tier"] = tiers
    merged["support_domain"] = supports
    merged["strategy_compatibility"] = compatibilities
    merged["strategic_risk_level"] = strat_risks
    merged["strategic_evidence_tier"] = strat_tiers
    merged["decision_class"] = classes
    merged["priority_score"] = priorities
    merged = merged.sort_values("priority_score", ascending=False, kind="mergesort").reset_index(drop=True)
    merged["priority_rank"] = np.arange(1, len(merged) + 1)
    keep_cols = [
        "priority_rank",
        "cross_bin",
        "cross_label",
        "soc_label",
        "rate_label",
        "temp_label",
        "tactical_action",
        "effect_per_1pp_ah",
        "effect_per_5pp_ah",
        "ci_low",
        "ci_high",
        "q_value",
        "evidence_tier",
        "support_domain",
        "strategy_compatibility",
        "strategic_risk_level",
        "decision_class",
        "priority_score",
        "rate_lower_c",
        "rate_upper_c",
        "rate_mid_c",
    ]
    return merged[keep_cols]


def build_strategy_tactics_matrix(
    tactical_df: pd.DataFrame,
    strategic_limit: float,
) -> pd.DataFrame:
    """Build final strategy-tactics matrix."""

    matrix = tactical_df.copy()
    final_actions: List[str] = []
    expected_gain: List[float] = []
    notes: List[str] = []
    for row in matrix.itertuples(index=False):
        action = str(row.tactical_action)
        cls = str(row.decision_class)
        effect_5pp = safe_float(row.effect_per_5pp_ah)
        if cls == "禁止外推":
            final_actions.append("冻结该区间加时（仅保留观测）")
            expected_gain.append(0.0)
            notes.append(f"与战略上限 {strategic_limit:.2f}C 冲突或证据不足")
        elif action == "增加份额":
            final_actions.append("执行 +5pp 替代（在战略约束内）")
            expected_gain.append(effect_5pp)
            notes.append("需满足温度/电流安全边界")
        elif action == "降低份额":
            final_actions.append("执行 -5pp 回撤并转移至其余池")
            expected_gain.append(-abs(effect_5pp))
            notes.append("用于抑制潜在衰减风险")
        else:
            final_actions.append("保持现状并继续验证")
            expected_gain.append(0.0)
            notes.append("等待更多证据后升级动作")
    matrix["final_action"] = final_actions
    matrix["expected_delta_q_next_5pp_ah"] = expected_gain
    matrix["constraint_note"] = notes
    keep_cols = [
        "priority_rank",
        "cross_bin",
        "cross_label",
        "tactical_action",
        "decision_class",
        "final_action",
        "expected_delta_q_next_5pp_ah",
        "strategy_compatibility",
        "strategic_risk_level",
        "evidence_tier",
        "support_domain",
        "constraint_note",
        "soc_label",
        "rate_label",
        "temp_label",
    ]
    return matrix[keep_cols]


def select_experiment_arms(matrix_df: pd.DataFrame, max_arms: int) -> pd.DataFrame:
    """Select top intervention arms for controlled experiment."""

    eligible = matrix_df[
        (matrix_df["decision_class"] != "禁止外推")
        & (matrix_df["tactical_action"].isin(["增加份额", "降低份额"]))
    ].copy()
    if eligible.empty:
        return eligible
    class_order = {"可上线": 0, "待验证": 1}
    tier_order = {"A": 0, "B": 1, "C": 2}
    eligible["class_order"] = eligible["decision_class"].map(class_order).fillna(2)
    eligible["tier_order"] = eligible["evidence_tier"].map(tier_order).fillna(3)
    eligible["effect_abs"] = pd.to_numeric(
        eligible["expected_delta_q_next_5pp_ah"], errors="coerce"
    ).abs()
    eligible = eligible.sort_values(
        ["class_order", "tier_order", "effect_abs", "priority_rank"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    keep_cols = [c for c in matrix_df.columns if c in eligible.columns]
    return eligible.head(int(max_arms)).reset_index(drop=True)[keep_cols]


def write_closed_loop_protocol(
    args: argparse.Namespace,
    strategic_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    strategic_global_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Write full closed-loop experiment protocol and return selected arms."""

    strategic_limit = safe_float(
        pd.to_numeric(strategic_df["recommended_c_rate_upper"], errors="coerce").dropna().iloc[0]
        if not strategic_df.empty
        else float("nan")
    )
    strategic_limit_text = f"{strategic_limit:.2f}C" if np.isfinite(strategic_limit) else "不可判定"
    global_row = strategic_global_df[
        (strategic_global_df["scenario"] == "baseline_tplus1_temp0_70")
        & (strategic_global_df["treatment_mode"] == "window_mean")
    ].copy()
    g = global_row.iloc[0] if not global_row.empty else None
    n_clusters = int(safe_float(getattr(g, "n_clusters", np.nan))) if g is not None else 0
    n_rows = int(safe_float(getattr(g, "n_rows", np.nan))) if g is not None else 0

    arms = select_experiment_arms(matrix_df=matrix_df, max_arms=int(args.max_experiment_arms))
    protocol_arms = arms.copy()
    if not protocol_arms.empty:
        protocol_arms["planned_adjustment"] = np.where(
            protocol_arms["tactical_action"] == "增加份额", "+5pp", "-5pp"
        )
        protocol_arms["target_effect_ah"] = pd.to_numeric(
            protocol_arms["expected_delta_q_next_5pp_ah"], errors="coerce"
        )

    lines: List[str] = []
    lines.append("# 双层闭环受控实验方案（Top3+对照）")
    lines.append("")
    lines.append("## 0. 方案摘要")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"- 统一标签区间：`{args.q_min:.1f} <= q_discharge <= {args.q_max:.1f}`，战略层与战术层同口径。"
    )
    lines.append(f"- 历史样本规模：`n_rows={n_rows}`，`n_clusters={n_clusters}`。")
    if np.isfinite(strategic_limit):
        lines.append(f"- 战略层建议倍率上限：`{strategic_limit:.2f}C`。")
    else:
        lines.append("- 战略层建议倍率上限：不可判定，按保守策略执行。")
    lines.append(f"- 试验结构：`1个对照组 + {int(args.max_experiment_arms)}个干预臂`。")
    lines.append("")
    lines.append("## 1. 研究假设")
    lines.append("- H0（零假设）：在战略约束内实施区间配时替代后，`q_discharge_{t+1}` 的均值与对照组无差异。")
    lines.append("- H1（研究假设）：至少一个干预臂可显著改善 `q_discharge_{t+1}`，且不提升安全风险。")
    lines.append("")
    lines.append("## 2. 随机化与分层设计")
    lines.append("- 随机化单元：`policy + cell_code`（整电芯级随机化，避免同电芯跨组污染）。")
    lines.append("- 分层变量：policy三元参数、`is_abnormal_cell`、基线容量分位。")
    lines.append("- 分组结构：对照组维持当前策略；干预组在同一战略约束内执行指定区间替代。")
    lines.append("")
    lines.append("## 3. 干预实施规则（Top3 + 对照）")
    lines.append("- Phase A（战略层）：先统一执行倍率上限策略，验证风险净改善。")
    if np.isfinite(strategic_limit):
        lines.append(f"- Phase A执行阈值：最大充电倍率 `< {strategic_limit:.2f}C`。")
    else:
        lines.append("- Phase A执行阈值：采用当前工程保守上限。")
    lines.append("- Phase B（战术层）：在Phase A约束内实施 Top3 区间 `±5pp` 替代。")
    lines.append("")
    lines.append("Top3 干预臂清单：")
    if protocol_arms.empty:
        lines.append("- 当前无可执行干预臂，建议补充样本后重评。")
        lines.append("")
    else:
        show_cols = [
            "priority_rank",
            "cross_bin",
            "cross_label",
            "soc_label",
            "rate_label",
            "temp_label",
            "tactical_action",
            "planned_adjustment",
            "target_effect_ah",
            "evidence_tier",
            "decision_class",
        ]
        lines.append(df_to_markdown(protocol_arms[show_cols]))
        lines.append("")
    lines.append("## 4. 终点定义与统计分析计划")
    lines.append("- 主要终点：`q_discharge_{t+1}`（下一循环放电容量，单位Ah）。")
    lines.append("- 次要终点：30-cycle 衰减斜率、`dt_s>3600` 事件率、温度越界率。")
    lines.append("- 分析口径：ITT（意向治疗）为主，PP（依方案）为辅。")
    lines.append("- 统计方法：组间均值差 + 聚类稳健标准误（聚类单元 `policy+cell`）。")
    lines.append("- 显著性控制：主终点按 `p<0.05`，多臂比较同步报告 FDR 修正后的 `q`。")
    lines.append("")
    lines.append("## 5. 样本量与观察窗建议")
    lines.append("- 建议每臂最少 `20` 个电芯簇（clusters），优先保证分层后每层>=5个簇。")
    lines.append("- 建议观察窗口：最短 `30 cycles`，推荐 `60 cycles` 以降低短期波动。")
    lines.append("- 若阶段中期效果绝对值 < `0.0002Ah/5pp` 且 CI 长期跨0，进入“延长观察或停臂复盘”。")
    lines.append("")
    lines.append("## 6. 停止规则与安全阈值")
    lines.append("- 连续3个 cycle 出现容量显著恶化且伴随安全告警（温度/电流）时，立即停臂。")
    lines.append("- 任一组温度保护或电流保护触发率较对照组上升超过 `20%`，暂停该组并回滚。")
    lines.append("- 如发生策略冲突（超出倍率上限），该批次数据标记为协议违背并纳入PP剔除。")
    lines.append("")
    lines.append("## 7. 上线闸门（Go/No-Go）")
    lines.append("- Go：主要终点改善显著（q<0.10）且安全指标不劣于对照组。")
    lines.append("- Hold：效果方向正确但CI跨0，继续扩样验证。")
    lines.append("- No-Go：效果为负或出现明确安全恶化。")
    lines.append("")
    lines.append("## 8. 实施检查清单")
    lines.append("- 每cycle记录字段：`policy`、`cell_code`、`cycles`、`q_discharge`、`cross_bin_inc_01..60_h`、`is_abnormal_cell`。")
    lines.append("- 每日执行核查：分组样本数、协议违背率、温度/电流越界率、缺失值比例。")
    lines.append("- 每周复盘：ITT/PP差异、各干预臂效应方向稳定性、是否触发停臂规则。")
    lines.append("")
    lines.append("## 9. 证据来源")
    lines.append("- 战略层：`mediation_effect_global.csv`、`mediation_effect_by_rate_bin_fixed_a_window_mean.csv`。")
    lines.append("- 战术层：`causal_substitution_effects.csv`、`strategy_tactics_decision_matrix.csv`。")
    out_path.write_text("\n".join(lines), encoding=args.encoding)
    return arms


def save_strategy_risk_tactical_effect_map(
    strategic_df: pd.DataFrame,
    tactical_df: pd.DataFrame,
    out_png: Path,
) -> None:
    """Save strategy risk versus tactical effect scatter map."""

    ensure_matplotlib_config()
    import matplotlib.pyplot as plt  # noqa: WPS433

    if tactical_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.4))
        ax.text(0.5, 0.5, "No tactical rows.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return
    risk_color = {"高": "#dc2626", "中": "#f59e0b", "中低": "#facc15", "低": "#22c55e", "未知": "#94a3b8"}
    class_marker = {"可上线": "o", "待验证": "s", "禁止外推": "X"}
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 6.4))
    for row in tactical_df.itertuples(index=False):
        x = safe_float(row.rate_mid_c)
        y = safe_float(row.effect_per_1pp_ah)
        ax.scatter(
            x,
            y,
            c=risk_color.get(str(row.strategic_risk_level), "#94a3b8"),
            marker=class_marker.get(str(row.decision_class), "o"),
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.text(x, y, f"b{int(row.cross_bin):02d}", fontsize=8, ha="left", va="bottom")
    limit = safe_float(
        pd.to_numeric(strategic_df["recommended_c_rate_upper"], errors="coerce").dropna().iloc[0]
        if not strategic_df.empty
        else float("nan")
    )
    if np.isfinite(limit):
        ax.axvline(limit, color="#111827", linestyle="--", linewidth=1.2, label=f"Cap {limit:.2f}C")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.set_xlabel("Rate midpoint (C)")
    ax.set_ylabel("Effect on q_next (Ah per +1pp)")
    ax.set_title("Strategy Risk vs Tactical Effect Map")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_two_layer_decision_heatmap(
    matrix_df: pd.DataFrame,
    out_png: Path,
) -> None:
    """Save two-layer decision heatmap."""

    ensure_matplotlib_config()
    import matplotlib.pyplot as plt  # noqa: WPS433

    if matrix_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.4))
        ax.text(0.5, 0.5, "No matrix rows.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return
    work = matrix_df.copy().sort_values("priority_rank", ascending=True, kind="mergesort")
    tier_score_map = {"A": 3.0, "B": 2.0, "C": 1.0}
    compat_score_map = {"兼容": 2.0, "部分冲突": 1.0, "冲突": 0.0}
    decision_score_map = {"可上线": 2.0, "待验证": 1.0, "禁止外推": 0.0}
    work["evidence_score"] = work["evidence_tier"].map(tier_score_map).fillna(1.0)
    work["compatibility_score"] = work["strategy_compatibility"].map(compat_score_map).fillna(1.0)
    work["decision_score"] = work["decision_class"].map(decision_score_map).fillna(1.0)
    mat = work[["evidence_score", "compatibility_score", "decision_score"]].to_numpy(dtype=float)
    fig_h = max(5.2, 0.34 * len(work) + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(8.8, fig_h))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=3.0)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Evidence", "Compatibility", "Decision"])
    labels = [f"b{int(r.cross_bin):02d} ({str(r.decision_class)})" for r in work.itertuples(index=False)]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Two-layer Decision Matrix Heatmap")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#0f172a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without optional dependencies."""

    if df is None or df.empty:
        return "_无数据_"
    safe_df = df.copy()
    safe_df = safe_df.replace({np.nan: ""})
    cols = [str(c) for c in safe_df.columns]
    lines: List[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in safe_df.itertuples(index=False):
        vals: List[str] = []
        for item in row:
            if isinstance(item, (float, np.floating)):
                vals.append(f"{float(item):.6f}")
            else:
                vals.append(str(item))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def relative_markdown_path(base_file: Path, target_file: Path) -> str:
    """Return markdown-friendly relative path from one file to another."""

    return Path(os.path.relpath(target_file, start=base_file.parent)).as_posix()


def format_signed(value: float, digits: int = 6) -> str:
    """Format float with explicit sign."""

    if not np.isfinite(value):
        return "nan"
    return f"{value:+.{digits}f}"


def ci_crosses_zero(ci_low: float, ci_high: float) -> bool:
    """Return whether confidence interval crosses zero."""

    if (not np.isfinite(ci_low)) or (not np.isfinite(ci_high)):
        return True
    return bool(ci_low <= 0.0 <= ci_high)


def add_figure_block(
    lines: List[str],
    title: str,
    report_path: Path,
    figure_path: Path,
    x_desc: str,
    y_desc: str,
    conclusion: str,
    evidence_sources: str,
) -> None:
    """Append one figure block with axis explanations and conclusion."""

    lines.append(f"### {title}")
    if figure_path.exists():
        rel = relative_markdown_path(report_path, figure_path)
        lines.append(f"![{title}]({rel})")
    else:
        lines.append(f"_图像缺失：{figure_path.name}_")
    lines.append(f"- X轴说明：{x_desc}")
    lines.append(f"- Y轴说明：{y_desc}")
    lines.append(f"- 结论：{conclusion}")
    lines.append(f"- 证据来源：{evidence_sources}")
    lines.append("")


def pick_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Pick columns that actually exist in DataFrame."""

    return [col for col in cols if col in df.columns]


def load_optional_csv(path: Path, encoding: str) -> pd.DataFrame:
    """Load CSV if exists, else return empty DataFrame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding=encoding)


def write_integrated_report(
    args: argparse.Namespace,
    strategic_df: pd.DataFrame,
    tactical_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    strategic_global_df: pd.DataFrame,
    strategic_diag_df: pd.DataFrame,
    tactical_causal_df: pd.DataFrame,
    tactical_sens_df: pd.DataFrame,
    screening_df: pd.DataFrame,
    selected_arms_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write paper-style integrated Chinese report for two-layer decisions."""

    global_row = strategic_global_df[
        (strategic_global_df["scenario"] == "baseline_tplus1_temp0_70")
        & (strategic_global_df["treatment_mode"] == "window_mean")
    ].copy()
    g = global_row.iloc[0] if not global_row.empty else None

    if {"scenario", "treatment_mode"}.issubset(strategic_diag_df.columns):
        diag_row = strategic_diag_df[
            (strategic_diag_df["scenario"] == "baseline_tplus1_temp0_70")
            & (strategic_diag_df["treatment_mode"] == "window_mean")
        ].copy()
    else:
        diag_row = pd.DataFrame()
    d = diag_row.iloc[0] if not diag_row.empty else None

    strategic_limit = safe_float(
        pd.to_numeric(strategic_df["recommended_c_rate_upper"], errors="coerce").dropna().iloc[0]
        if not strategic_df.empty
        else float("nan")
    )
    strategic_limit_text = f"{strategic_limit:.2f}C" if np.isfinite(strategic_limit) else "不可判定"
    class_counts = matrix_df["decision_class"].value_counts(dropna=False).to_dict()
    total_decisions = int(len(matrix_df))
    online_count = int(class_counts.get("可上线", 0))
    valid_count = int(class_counts.get("待验证", 0))
    block_count = int(class_counts.get("禁止外推", 0))
    online_ratio = (100.0 * online_count / total_decisions) if total_decisions > 0 else float("nan")
    conflict_count = int((matrix_df["strategy_compatibility"].isin(["冲突", "部分冲突"])).sum())

    causal_main = tactical_causal_df.copy()
    if not causal_main.empty:
        causal_main["ci_cross_zero"] = causal_main.apply(
            lambda row: ci_crosses_zero(
                safe_float(row.get("ci_low", np.nan)),
                safe_float(row.get("ci_high", np.nan)),
            ),
            axis=1,
        )
        sig_bins = causal_main[
            (pd.to_numeric(causal_main["q_value"], errors="coerce") <= 0.10)
            & (~causal_main["ci_cross_zero"])
        ].copy()
    else:
        sig_bins = pd.DataFrame()

    sens_consistency = float("nan")
    if (not causal_main.empty) and (not tactical_sens_df.empty):
        merge = causal_main[["cross_bin", "effect_per_1pp_ah"]].merge(
            tactical_sens_df[["cross_bin", "effect_per_1pp_ah"]],
            on="cross_bin",
            how="inner",
            suffixes=("_main", "_sens"),
        )
        if not merge.empty:
            sign_main = np.sign(pd.to_numeric(merge["effect_per_1pp_ah_main"], errors="coerce"))
            sign_sens = np.sign(pd.to_numeric(merge["effect_per_1pp_ah_sens"], errors="coerce"))
            sens_consistency = float((sign_main == sign_sens).mean())

    top1_screen = screening_df.sort_values("combined_score", ascending=False).head(1).copy()
    top20_screen = screening_df.sort_values("combined_score", ascending=False).head(20).copy()
    top20_gap = float("nan")
    if len(top20_screen) >= 20:
        top20_gap = safe_float(top20_screen["combined_score"].iloc[0]) - safe_float(
            top20_screen["combined_score"].iloc[-1]
        )

    best_heat_row = top1_screen.iloc[0] if not top1_screen.empty else None
    best_decision = matrix_df.sort_values("priority_rank", ascending=True).head(1).copy()
    best_decision_row = best_decision.iloc[0] if not best_decision.empty else None

    figure_path_decompose = args.strategic_dir / "fig_path_decomposition_global.png"
    figure_contribution = args.strategic_dir / "fig_contribution_share_global.png"
    figure_top20 = args.tactical_dir / "screening_top20_bar.png"
    figure_heatmap = args.tactical_dir / "screening_heatmap_soc_panels.png"
    figure_forest = args.tactical_dir / "effect_forest_plot.png"
    figure_matrix = args.output_dir / "two_layer_decision_matrix_heatmap.png"
    figure_map = args.output_dir / "strategy_risk_tactical_effect_map.png"

    lines: List[str] = []
    lines.append("# 双层决策闭环综合报告（论文式）")
    lines.append("")
    lines.append("## 摘要")
    lines.append(f"- 报告风格：`{args.report_style}`；附录级别：`{args.appendix_level}`。")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。")
    lines.append(f"- Python解释器：`{sys.executable}`。")
    lines.append(f"- 统一标签区间：`{args.q_min:.1f} <= q_discharge <= {args.q_max:.1f}`。")
    if g is not None:
        lines.append(
            "- 关键结论1（战略层）："
            f"window_mean 全局效应 `TE={safe_float(g['te_r']):.6f}`，"
            f"`NDE={safe_float(g['nde_r']):.6f}`，`NIE={safe_float(g['nie_r']):.6f}`，"
            f"建议倍率上限 `{strategic_limit_text}`。"
            "（证据来源：`mediation_effect_global.csv`、`mediation_effect_by_rate_bin_fixed_a_window_mean.csv`）"
        )
    if not selected_arms_df.empty:
        arm_rows = selected_arms_df.sort_values("priority_rank", kind="mergesort").head(3)
        arm_text = "；".join(
            [
                (
                    f"bin{int(row.cross_bin):02d}({str(row.cross_label)}) "
                    f"{'+5pp' if str(row.tactical_action) == '增加份额' else '-5pp'} "
                    f"预期 {format_signed(safe_float(row.expected_delta_q_next_5pp_ah))}Ah"
                )
                for row in arm_rows.itertuples(index=False)
            ]
        )
        lines.append(
            f"- 关键结论2（战术层）：Top3 可执行干预臂为 {arm_text}。"
            "（证据来源：`strategy_tactics_decision_matrix.csv`）"
        )
    lines.append(
        f"- 关键结论3（决策层）：可上线 `{online_count}` 项、待验证 `{valid_count}` 项、禁止外推 `{block_count}` 项，"
        f"可上线占比 `{online_ratio:.1f}%`，战略冲突项 `{conflict_count}`。"
        "（证据来源：`strategy_tactics_decision_matrix.csv`）"
    )
    lines.append("")

    lines.append("## 1. 业务问题与决策目标（面向非技术读者）")
    lines.append("- 战略层问题：整体充电倍率策略是否会显著增加容量衰减风险？")
    lines.append("- 战术层问题：在既定倍率上限下，60个区间中“把1个百分点充电时间替代到某区间”对下一循环容量有何影响？")
    lines.append("- 决策目标：形成“先控风险（战略）再做配时（战术）”的可执行闭环，并通过受控实验验证。")
    lines.append("")

    lines.append("## 2. 方法与理论基础")
    lines.append("- 战略层采用中介分解：`TE = NDE + NIE`。")
    lines.append("- 战术层采用替代效应估计：`+1pp` 表示将总充电时间中1个百分点从其余区间转移到目标区间。")
    lines.append("- 显著性解释：`p` 是单次检验显著性，`q(FDR)` 是多重比较后显著性，更保守。")
    lines.append("- 当95%CI跨0时，结论应解释为“当前证据不足”，而不是“确定无效”。")
    lines.append("")

    lines.append("## 3. 数据与质量控制")
    if d is not None:
        lines.append(
            f"- 标签过滤后样本：`{safe_int(d.get('label_rows_after_range_filter', np.nan)):,}`；"
            f"剔除 `<q_min`：`{safe_int(d.get('label_rows_lt_qmin_removed', np.nan)):,}`，"
            f"剔除 `>q_max`：`{safe_int(d.get('label_rows_gt_qmax_removed', np.nan)):,}`。"
            "（证据来源：`mediation_dataset_diagnostics.csv`）"
        )
        lines.append(
            f"- 战略层可用样本：`n_rows={safe_int(d.get('rows_final', np.nan)):,}`，"
            f"`n_clusters={safe_int(d.get('n_clusters', np.nan)):,}`。"
            "（证据来源：`mediation_dataset_diagnostics.csv`）"
        )
    if not tactical_causal_df.empty:
        lines.append(
            f"- 战术层Top10主分析样本范围：`n_rows={int(pd.to_numeric(tactical_causal_df['n_rows'], errors='coerce').max()):,}`，"
            f"`n_groups={int(pd.to_numeric(tactical_causal_df['n_groups'], errors='coerce').max()):,}`。"
            "（证据来源：`causal_substitution_effects.csv`）"
        )
    if np.isfinite(sens_consistency):
        lines.append(
            f"- 主分析与敏感性方向一致率：`{sens_consistency * 100:.1f}%`。"
            "（证据来源：`causal_substitution_effects.csv`、`causal_sensitivity_abnormal_excluded.csv`）"
        )
    lines.append("")

    lines.append("## 4. 核心结果")
    lines.append("### 4.1 战略层结果（倍率风险）")
    strategic_cols = pick_existing_columns(
        strategic_df,
        [
            "rate_bin_label",
            "te_r",
            "te_r_ci_low",
            "te_r_ci_high",
            "nie_share",
            "risk_level",
            "evidence_tier",
            "recommended_action",
            "recommended_c_rate_upper",
        ],
    )
    lines.append(df_to_markdown(strategic_df[strategic_cols]))
    lines.append("")

    lines.append("### 4.2 战术层结果（60区间替代效应）")
    tactical_cols = pick_existing_columns(
        tactical_df,
        [
            "priority_rank",
            "cross_bin",
            "cross_label",
            "soc_label",
            "rate_label",
            "temp_label",
            "effect_per_1pp_ah",
            "ci_low",
            "ci_high",
            "q_value",
            "tactical_action",
            "decision_class",
        ],
    )
    lines.append(df_to_markdown(tactical_df[tactical_cols]))
    lines.append("")
    lines.append(
        f"- 统计显著（q<=0.10 且CI不跨0）的区间数量：`{len(sig_bins)}` / `{len(causal_main)}`。"
        "（证据来源：`causal_substitution_effects.csv`）"
    )
    lines.append("")

    lines.append("### 4.3 关键图表解读")
    if g is not None:
        add_figure_block(
            lines=lines,
            title="图1 战略层路径分解图",
            report_path=out_path,
            figure_path=figure_path_decompose,
            x_desc="因果路径分量（TE/NDE/NIE）。",
            y_desc="对容量变化的效应值（Ah尺度）。",
            conclusion=(
                f"全局效应中 TE={safe_float(g['te_r']):.6f}，且 TE≈NDE+NIE；"
                f"其中NIE={safe_float(g['nie_r']):.6f}，表明温度中介占比非零。"
            ),
            evidence_sources="`mediation_effect_global.csv`",
        )
        add_figure_block(
            lines=lines,
            title="图2 战略层贡献占比图",
            report_path=out_path,
            figure_path=figure_contribution,
            x_desc="路径分量类别（直接效应/中介效应）。",
            y_desc="效应贡献比例（%）。",
            conclusion=(
                f"NIE占比约 `{safe_float(g['nie_share']) * 100:.2f}%`，说明倍率影响中有一部分通过温度路径传递。"
            ),
            evidence_sources="`mediation_effect_global.csv`",
        )
    top_row_text = "无"
    if not top1_screen.empty:
        row = top1_screen.iloc[0]
        top_row_text = (
            f"Top1为 bin{int(safe_float(row['cross_bin'])):02d}({str(row['cross_label'])})，"
            f"综合分数 `{safe_float(row['combined_score']):.6f}`"
        )
    add_figure_block(
        lines=lines,
        title="图3 Top20筛选得分条形图",
        report_path=out_path,
        figure_path=figure_top20,
        x_desc="综合筛选得分（相关性与重要性归一化后平均）。",
        y_desc="候选区间（bin标签）。",
        conclusion=(
            f"{top_row_text}；Top1与Top20末位分差 "
            f"`{top20_gap:.6f}`。"
            if np.isfinite(top20_gap)
            else f"{top_row_text}；Top20样本不足，分差未计算。"
        ),
        evidence_sources="`screening_scores.csv`",
    )
    heat_text = "无可用筛选数据。"
    if best_heat_row is not None:
        heat_text = (
            f"最高热区在 `{best_heat_row['soc_label']}` × `{best_heat_row['rate_label']}` × `{best_heat_row['temp_label']}`，"
            f"对应 bin{int(safe_float(best_heat_row['cross_bin'])):02d}({str(best_heat_row['cross_label'])})。"
        )
    add_figure_block(
        lines=lines,
        title="图4 60区间筛选热力图（SOC分面）",
        report_path=out_path,
        figure_path=figure_heatmap,
        x_desc="温度分段（temp_bin）。",
        y_desc="倍率分段（rate_bin），不同SOC分面显示。",
        conclusion=heat_text,
        evidence_sources="`screening_scores.csv`",
    )
    pos_count = int((pd.to_numeric(causal_main.get("effect_per_1pp_ah", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    neg_count = int((pd.to_numeric(causal_main.get("effect_per_1pp_ah", pd.Series(dtype=float)), errors="coerce") < 0).sum())
    add_figure_block(
        lines=lines,
        title="图5 替代效应森林图（Top10）",
        report_path=out_path,
        figure_path=figure_forest,
        x_desc="将1pp充电时间替代到目标区间后的容量变化（Ah/1pp）。",
        y_desc="Top10区间标识（cross_bin与标签）。",
        conclusion=(
            f"正向区间 `{pos_count}` 个、负向区间 `{neg_count}` 个；"
            f"显著且CI不跨0区间 `{len(sig_bins)}` 个。"
        ),
        evidence_sources="`causal_substitution_effects.csv`",
    )
    matrix_text = "无融合决策结果。"
    if best_decision_row is not None:
        matrix_text = (
            f"最高优先级为 bin{int(safe_float(best_decision_row['cross_bin'])):02d}"
            f"({str(best_decision_row['cross_label'])})，动作 `{str(best_decision_row['final_action'])}`。"
        )
    add_figure_block(
        lines=lines,
        title="图6 双层决策热力图",
        report_path=out_path,
        figure_path=figure_matrix,
        x_desc="证据强度/战略兼容性/执行分类评分维度。",
        y_desc="候选区间及其决策类别。",
        conclusion=matrix_text,
        evidence_sources="`strategy_tactics_decision_matrix.csv`",
    )
    add_figure_block(
        lines=lines,
        title="图7 战略风险-战术效应映射",
        report_path=out_path,
        figure_path=figure_map,
        x_desc="区间中点倍率（C）。",
        y_desc="单位替代效应（Ah/1pp）。",
        conclusion=f"战略冲突项 `{conflict_count}` 个，说明当前Top10战术动作与战略层约束总体一致。",
        evidence_sources="`tactical_layer_decisions.csv`、`strategy_tactics_decision_matrix.csv`",
    )

    lines.append("## 5. 双层决策与执行清单")
    decision_cols = pick_existing_columns(
        matrix_df,
        [
            "priority_rank",
            "cross_bin",
            "cross_label",
            "decision_class",
            "final_action",
            "expected_delta_q_next_5pp_ah",
            "strategy_compatibility",
            "evidence_tier",
            "constraint_note",
        ],
    )
    lines.append(df_to_markdown(matrix_df[decision_cols]))
    lines.append("")
    lines.append("### Top3 干预臂")
    if selected_arms_df.empty:
        lines.append("- 当前无可直接执行干预臂。")
    else:
        show_cols = pick_existing_columns(
            selected_arms_df,
            [
                "priority_rank",
                "cross_bin",
                "cross_label",
                "tactical_action",
                "decision_class",
                "expected_delta_q_next_5pp_ah",
                "soc_label",
                "rate_label",
                "temp_label",
            ],
        )
        lines.append(df_to_markdown(selected_arms_df[show_cols]))
    lines.append("- 受控实验实施细则见 `closed_loop_experiment_protocol.md`。")
    lines.append("")

    lines.append("## 6. 局限性与外推边界")
    lines.append("- 观测因果估计依赖“给定控制变量后无未观测混杂”的强假设。")
    lines.append("- CI跨0表示当前证据不足，需扩样或受控实验，不应解释为“确定无效应”。")
    lines.append("- 本报告建议仅在样本支持域内执行，不建议把+5pp策略外推到未覆盖区间。")
    lines.append("- 最终上线结论以受控实验结果为准，观测分析仅用于排序与优先级。")
    lines.append("")

    lines.append("## 7. 结论与行动建议")
    lines.append("1. 战略先行：先按战略层上限控制倍率暴露，再执行战术区间配时。")
    lines.append("2. 战术聚焦：优先执行Top3中证据等级高且战略兼容的区间替代。")
    lines.append("3. 闭环验证：采用ITT/PP双口径推进受控实验，达到Go门槛后再规模化。")
    lines.append("")

    appendix_level = str(args.appendix_level).strip().lower()
    include_appendix = appendix_level not in {"none", "off", "0"}
    if include_appendix:
        lines.append("## 附录A：关键公式与识别假设")
        lines.append("```text")
        lines.append("战略层：TE = NDE + NIE")
        lines.append("战术层（残差化DML）：")
        lines.append("Y~ = Y - m_y(W)")
        lines.append("T~ = T - m_t(W)")
        lines.append("theta = Cov(Y~, T~) / Var(T~)")
        lines.append("effect_per_1pp = 0.01 * theta")
        lines.append("effect_per_5pp = 0.05 * theta")
        lines.append("```")
        lines.append("- 不确定性：按 `policy+cell` 聚类 bootstrap 置信区间。")
        lines.append("- 多重比较：Benjamini-Hochberg FDR，报告 `q` 值。")
        lines.append("")

        lines.append("## 附录B：术语词典（非技术读者）")
        glossary: List[Tuple[str, str]] = [
            ("TE", "总效应：倍率变化对容量变化的总体影响。"),
            ("NDE", "自然直接效应：不通过温度路径的直接影响。"),
            ("NIE", "自然间接效应：通过温度中介传递的影响。"),
            ("DML", "双重机器学习：先用模型消化混杂，再估计处理效应。"),
            ("CI", "置信区间：效应可能范围；跨0通常表示证据不足。"),
            ("p 值", "单次检验显著性概率指标。"),
            ("q 值(FDR)", "多重比较修正后的显著性指标，比p更严格。"),
            ("pp", "百分点（percentage point），如+5pp表示份额增加5个百分点。"),
            ("支持域", "数据中真实出现过、可被可靠估计的操作范围。"),
            ("外推", "把结论用于数据未覆盖区域，风险较高。"),
        ]
        glossary_df = pd.DataFrame(glossary, columns=["术语", "解释"])
        lines.append(df_to_markdown(glossary_df))
        lines.append("")

        if appendix_level in {"full", "all", "2"}:
            lines.append("## 附录C：CI跨0区间解释模板")
            cross_zero_df = causal_main[causal_main["ci_cross_zero"]].copy()
            if cross_zero_df.empty:
                lines.append("- 当前Top10中无CI跨0区间。")
            else:
                explain_cols = pick_existing_columns(
                    cross_zero_df,
                    ["cross_bin", "cross_label", "effect_per_1pp_ah", "ci_low", "ci_high", "q_value"],
                )
                lines.append(df_to_markdown(cross_zero_df[explain_cols]))
                lines.append("- 统一解释：该区间为“证据不足”，建议进入观察集并优先纳入受控实验。")
            lines.append("")
            lines.append("## 附录D：证据来源映射")
            evidence_map = pd.DataFrame(
                [
                    ["战略全局效应", "mediation_effect_global.csv", "TE/NDE/NIE与总体风险结论"],
                    ["战略分段风险", "mediation_effect_by_rate_bin_fixed_a_window_mean.csv", "倍率分段风险与上限建议"],
                    ["数据质量诊断", "mediation_dataset_diagnostics.csv", "标签过滤与样本规模核验"],
                    ["战术主分析", "causal_substitution_effects.csv", "Top10替代效应、CI、p、q"],
                    ["战术敏感性", "causal_sensitivity_abnormal_excluded.csv", "剔除异常电芯后的方向一致性"],
                    ["全量筛选", "screening_scores.csv", "60区间筛选得分与热区定位"],
                    ["融合决策", "strategy_tactics_decision_matrix.csv", "执行分类与动作优先级"],
                ],
                columns=["结论模块", "来源文件", "用途"],
            )
            lines.append(df_to_markdown(evidence_map))
            lines.append("")

    out_path.write_text("\n".join(lines), encoding=args.encoding)


def main() -> int:
    """Run two-layer decision pack build process."""

    args = parse_args()
    ensure_dir(args.output_dir)

    strategic_global = pd.read_csv(args.strategic_dir / "mediation_effect_global.csv", encoding=args.encoding)
    strategic_rate = pd.read_csv(
        args.strategic_dir / "mediation_effect_by_rate_bin_fixed_a_window_mean.csv",
        encoding=args.encoding,
    )
    strategic_diag = load_optional_csv(
        args.strategic_dir / "mediation_dataset_diagnostics.csv",
        encoding=args.encoding,
    )
    tactical_causal = pd.read_csv(args.tactical_dir / "causal_substitution_effects.csv", encoding=args.encoding)
    tactical_top = pd.read_csv(args.tactical_dir / "top10_bins.csv", encoding=args.encoding)
    tactical_sens = load_optional_csv(
        args.tactical_dir / "causal_sensitivity_abnormal_excluded.csv",
        encoding=args.encoding,
    )
    screening_scores = load_optional_csv(
        args.tactical_dir / "screening_scores.csv",
        encoding=args.encoding,
    )

    strategic_decisions, strategic_limit = build_strategic_layer_decisions(
        global_df=strategic_global,
        rate_df=strategic_rate,
    )
    tactical_decisions = build_tactical_layer_decisions(
        tactical_causal_df=tactical_causal,
        top_df=tactical_top,
        strategic_df=strategic_decisions,
        strategic_limit=strategic_limit,
        top_k=int(args.top_k),
    )
    matrix_df = build_strategy_tactics_matrix(
        tactical_df=tactical_decisions,
        strategic_limit=strategic_limit,
    )

    out_strategic = args.output_dir / "strategic_layer_decisions.csv"
    out_tactical = args.output_dir / "tactical_layer_decisions.csv"
    out_matrix = args.output_dir / "strategy_tactics_decision_matrix.csv"
    out_map = args.output_dir / "strategy_risk_tactical_effect_map.png"
    out_heat = args.output_dir / "two_layer_decision_matrix_heatmap.png"
    out_protocol = args.output_dir / "closed_loop_experiment_protocol.md"
    out_report = args.output_dir / "strategy_tactics_integrated_report.md"

    strategic_decisions.to_csv(out_strategic, index=False, encoding=args.encoding)
    tactical_decisions.to_csv(out_tactical, index=False, encoding=args.encoding)
    matrix_df.to_csv(out_matrix, index=False, encoding=args.encoding)
    save_strategy_risk_tactical_effect_map(strategic_decisions, tactical_decisions, out_map)
    save_two_layer_decision_heatmap(matrix_df, out_heat)
    selected_arms = write_closed_loop_protocol(
        args=args,
        strategic_df=strategic_decisions,
        matrix_df=matrix_df,
        strategic_global_df=strategic_global,
        out_path=out_protocol,
    )
    write_integrated_report(
        args=args,
        strategic_df=strategic_decisions,
        tactical_df=tactical_decisions,
        matrix_df=matrix_df,
        strategic_global_df=strategic_global,
        strategic_diag_df=strategic_diag,
        tactical_causal_df=tactical_causal,
        tactical_sens_df=tactical_sens,
        screening_df=screening_scores,
        selected_arms_df=selected_arms,
        out_path=out_report,
    )

    print(f"Saved: {out_strategic}")
    print(f"Saved: {out_tactical}")
    print(f"Saved: {out_matrix}")
    print(f"Saved: {out_map}")
    print(f"Saved: {out_heat}")
    print(f"Saved: {out_protocol}")
    print(f"Saved: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
