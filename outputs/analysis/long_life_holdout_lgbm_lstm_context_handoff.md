# long_life_holdout LightGBM/LSTM context handoff

## Purpose

This file is a compact handoff for continuing the long `batt_soh` LightGBM/LSTM evaluation thread without relying on remote context compaction.

## Compact failure diagnosis

- Codex thread: `019e1a41-bb6e-75f3-a6e4-158524a94b09`
- Workspace: `C:/Users/pal/projects/batt_soh`
- Session JSONL: `C:/Users/pal/.codex/sessions/2026/05/12/rollout-2026-05-12T11-36-15-019e1a41-bb6e-75f3-a6e4-158524a94b09.jsonl`
- SQLite log source: `C:/Users/pal/.codex/logs_2.sqlite`
- Key failure:

```text
remote compaction failed
compact_error=stream disconnected before completion:
error sending request for url (https://chatgpt.com/backend-api/codex/responses/compact)
```

The failure happened in the Codex remote `/responses/compact` request path, not in the battery training scripts.

## Network and proxy findings

- The failed compact request used `proxy(http://127.0.0.1:7897/)`.
- `Test-NetConnection 127.0.0.1 -Port 7897` succeeded.
- `Test-NetConnection 127.0.0.1 -Port 9` failed.
- The current shell inherited dead process-scoped proxy variables pointing to `http://127.0.0.1:9`.
- `curl.exe -I --proxy http://127.0.0.1:7897 https://chatgpt.com --max-time 20 -v` established the proxy tunnel, then failed at Windows SChannel with `SEC_E_NO_CREDENTIALS`.

Recommended startup hygiene before retrying compaction:

```powershell
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY, Env:http_proxy, Env:https_proxy, Env:all_proxy -ErrorAction SilentlyContinue
$env:NO_PROXY = "localhost,127.0.0.1,::1"
$env:no_proxy = "localhost,127.0.0.1,::1"
```

If Clash Verge is the intended route, restart Codex from an environment that consistently uses:

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:ALL_PROXY = "http://127.0.0.1:7897"
```

## Main experiment artifacts

- H100/M50 comparison report: `C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h100_m50_comparison.md`
- H100/M50 figures: `C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h100_m50_figures`
- H50/M100 comparison report: `C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_comparison.md`
- H50/M100 figures: `C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_lstm_blocks_h50_m100_figures`

Key run directories:

- `outputs/analysis/long_life_holdout_lgbm_blocks_h100_m50`
- `outputs/analysis/long_life_holdout_lstm_blocks_h100_m50`
- `outputs/analysis/long_life_holdout_lgbm_history_retention_blocks_h100_m50`
- `outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h100_m50`
- `outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h100_m50`
- `outputs/analysis/long_life_holdout_lgbm_history_retention_blocks_h50_m100`
- `outputs/analysis/long_life_holdout_lstm_blocks_h50_m100`
- `outputs/analysis/long_life_holdout_lgbm_last_retention_only_blocks_h50_m100`
- `outputs/analysis/long_life_holdout_lstm_last_retention_only_blocks_h50_m100`

## Current route-level conclusions

For H100/M50:

- `linear_last10` remains the strongest low-cost trend baseline.
- Pure operational LSTM (`100x55`) is stronger than LightGBM direct under the same long-life holdout split.
- In last-retention-only ablation, LSTM delta (`1x1`) beats LightGBM last-only at H50 and ALL, suggesting the monotonic delta structure uses the last-retention starting point better for future-curve extrapolation.

For H50/M100:

- `LightGBM + history retention summary` is the strongest endpoint H100 route.
- Pure operational LSTM (`50x55`) remains stronger than LightGBM direct.
- In last-retention-only ablation, LSTM delta (`1x1`) is much stronger than LightGBM last-only at H100 and ALL.

## Suggested new-thread prompt

```text
继续 C:/Users/pal/projects/batt_soh 中 long_life_holdout LightGBM/LSTM 工况统计 -> retention 预测评估工作。先读取并遵守 AGENTS.md，再读取以下交接文件：

C:/Users/pal/projects/batt_soh/outputs/analysis/long_life_holdout_lgbm_lstm_context_handoff.md

不要重跑训练，除非我明确要求。优先基于现有 H100/M50 与 H50/M100 报告、run_config、dataset_checks 和 CSV 指标继续分析或整理结论。
```
