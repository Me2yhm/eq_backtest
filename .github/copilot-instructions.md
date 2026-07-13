# Agent Instructions — eq-backtest Validation

## Project Identity

This is a **backtest validation project**. We have an independent 15-minute-frequency
A-share quantitative backtest engine. An external team also has their own implementation.
Both run with the **same predictions** and **same strategy parameters**. The goal is to
make our backtest positions **exactly match** the external positions so that all
performance metrics (annualized return, Sharpe, Calmar, max drawdown) converge.

## Current State

- Position overlap with external: **~96%** (was ~85%, improved through several rounds)
- Remaining ~4% gap is documented in `tmp/holding_diff_worklog.md`
- The residual is dominated by: rank-function differences (54.5%), bilateral carry/frozen
  state persistence, and target selection residuals

## Data Sources

| Path | Content |
|------|---------|
| `/ext/trq/predictions.parquet` | Shared predictions (identical for both sides) |
| `/ext/trq/weights.parquet` | External actual position output |
| `/ext/eq_data/15min_bar_full_left.parquet` | Local 15m market data |
| `/ext/eq_data/daily_with_limit.pqt` | Local daily constraint data |
| `/ext/eq_data/ret_csi_1000.csv` | CSI 1000 benchmark returns |
| `output_long_4400/` | Local backtest output |

## Immutable Rules

These must NOT be changed without explicit discussion and approval:

1. **Market-cap proxy**: Use `log_size` / `size_rank`, never external `market_cap`
2. **T+1 freeze release**: Calendar-day-based (not session-based); session-based was
   tried and reverted — it causes incorrect same-day intraday turnover
3. **Signal timing**: `trade_on_next_bar=True` — previous bar's signal, current bar's execution
4. **Benchmark**: CSI 1000 (000852)
5. **Cost**: One-way 4.5bp (`COST_PER_TURNOVER = 0.00045`)
6. **Core code files**: Only modify `portfolio_15min.py`, `data_loader_15min.py`, `config.py`
   for position-alignment work. Do NOT touch metrics, plotting, or external data.

## Standard Work Cycle

For every change aimed at closing the position gap:

1. **Hypothesize** — read current residual attribution (`attribute_residual_overlap.py`
   output or `drill_*.py` output), form a specific hypothesis about which semantic
   mismatch causes the gap. Write it in `tmp/holding_diff_worklog.md` first.

2. **Implement** — make ONE atomic semantic change at a time. Only touch
   `portfolio_15min.py`, `data_loader_15min.py`, or `config.py`.

3. **Run backtest** — `python run.py`

4. **Compare** — `python tmp/compare_positions.py` (mandatory). Optionally run
   drill-down scripts if the residual distribution changed.

5. **Record** — append a record to `tmp/holding_diff_worklog.md` using the template:
   status, files changed, reason, code snippet, validation command, contribution
   (Δ overlap, Δ diff, qualitative conclusion).

6. **Decide** — keep if overlap ↑ / diff ↓; revert otherwise. **If the change caused
   divergence (overlap ↓ / diff ↑), revert the code and do NOT record it in
   `holding_diff_worklog.md`. Instead, record it in `tmp/divergence_log.md`
   with the hypothesis, what was changed, how much worse it got, and the conclusion.**

## Key Diagnostic Scripts (in `tmp/`)

| Script | Purpose |
|--------|---------|
| `compare_positions.py` | End-to-end position comparison (run after every change) |
| `attribute_residual_overlap.py` | Classify residual mismatch into target/execution/carry |
| `ab_active_holding_changes.py` | A/B test: measure marginal contribution of each active change |
| `inspect_actual_mismatch.py` | Export per-row actual mismatch for first 2 days |
| `inspect_actual_mismatch_pool.py` | Join mismatch rows back to local pool rows |
| `drill_out_of_pool.py` | Drill-down on size-pool mismatch names |
| `drill_carry.py` | Analyze bilateral carry/frozen state overlap |

## Reference Documents

- `tmp/external_logic_doc.md` — External backtest logic in detail (portfolio + backtest steps)
- `tmp/actual.md` — External `actual` per-bar update logic
- `tmp/holding_diff_worklog.md` — **Chronological log of ALL successful changes, their status, and impact**
- `tmp/divergence_log.md` — **Hypothesized reasons for remaining gap + failed experiments (reverted)**
- `.github/workflow.md` — Full workflow specification (this file's companion)

## External Backtest Semantics (Quick Reference)

```
Ideal Target:
  Universe = _valid_universe ∩ top 4400 by market_cap
  Eligible = (Universe ∩ finite_prediction) ∪ (hold_mask ∩ _valid_listed)
  Rank     = sort Eligible by prediction desc
  Sell     = hold_mask ∧ (rank > 1400) ∧ _valid_sell
  Buy      = rank ≤ 800 ∧ _valid_buy, fill up to 800 slots
  Weight   = equal weight

Actual Progressive Execution (per bar):
  1. Sell: sort (target_weight asc, reduction desc), capped by _valid_sell + T+1 frozen
  2. Buy:  sort (deficit desc), top-up existing first, then open new
  3. T+1:  newly bought → frozen until next calendar day
```

## Communication Style

- **Every response MUST start with the word "echo" on its own line.**
- **使用中文进行交流。** 所有回复、解释、代码注释均使用中文。
- Be concise. Do not explain things the user already knows.
- When referencing a file, use backticks: `portfolio_15min.py`
- When suggesting a change, state the hypothesis first, then the code change, then
  the exact validation command to run.
- Never output a code block with file changes — use the edit tools instead.

## Dev Workflow Conventions

本项目的开发流程遵循多阶段工作流：**communicate → plan → implement → judge → (plan → implement → judge)… → summary → over**

```
communicate → plan → implement → judge ⇄ plan → summary → over
                        ↑                    │
                        └────────────────────┘
                   (循环直到需求完成或质量达标)
```

### 模型使用策略

| 阶段 | 模型 | 需 Human 确认 |
|------|------|:---:|
| **communicate** | `GLM-5.2 Coder (customendpoint)` | ✅ |
| **plan** | `GLM-5.2 Coder (customendpoint)` | ✅ |
| **implement** | `DeepSeek V4 Pro (deepseek)` | ❌ 自动 |
| **judge** | `GLM-5.2 Coder (customendpoint)` | ❌ 自动 |
| **summary** | `DeepSeek V4 Pro (deepseek)` | ❌ 自动 |

> **注意**: `model` 字段格式必须为 `"<Display Name> (<vendor>)"`，其中 Display Name 是模型选择器中显示的**大写带空格**名称（如 `"GPT-4o"`、`"Claude Sonnet 4.5"`、`"DeepSeek V4 Pro"`），不是内部 model ID（如 `gpt-4o`、`claude-sonnet-4.5`、`deepseek-v4-pro`）。
> 如需使用 GLM，需先安装 `vicanent.gcmp` 或 `smallmain.vscode-unify-chat-provider` 扩展。

### 文档目录

所有工作流产物统一存放在 communicate 阶段确认的文档目录下（默认 `docs/workflow/`）：

| 产物 | 路径 |
|------|------|
| 需求文档 | `{doc_dir}/requirements.md` |
| 计划文档 | `{doc_dir}/plans/*.md` |
| 验收报告 | `{doc_dir}/reports/*.md` |
| 评审日志 | `{doc_dir}/judge-logs/*.md` |
| 总结报告 | `{doc_dir}/summary.md` |

### 流转规则

| 阶段 | 模型 | 需 Human 确认 | 产出 |
|------|------|:---:|------|
| **communicate** | `GLM-5.2 Coder (customendpoint)` | ✅ | `requirements.md` |
| **plan** | `GLM-5.2 Coder (customendpoint)` | ✅ | `plans/*.md` |
| **implement** | `DeepSeek V4 Pro (deepseek)` | ❌ 自动 | `reports/*.md` + git commits |
| **judge** | `GLM-5.2 Coder (customendpoint)` | ❌ 自动 | `judge-logs/*.md` |
| **summary** | `DeepSeek V4 Pro (deepseek)` | ❌ 自动 | `summary.md` |

### Git 提交规范

implement 阶段每完成一个 checkpoint 做一次 git commit，格式：

```
<type>(<scope>): <简短描述>

完成需求: <FR-01, FR-02>
Checkpoint: <名称>
```

### 与回测验证工作流的关系

- 上述 Dev Workflow 适用于**新功能开发、复杂重构、多轮迭代**等场景。
- 日常的回测对齐工作（修改 `portfolio_15min.py` 等核心文件、运行 `compare_positions.py`）
  仍遵循上方的 **Standard Work Cycle**（Hypothesize → Implement → Run → Compare → Record → Decide），
  不启动完整 Dev Workflow。
- 当回测对齐涉及大规模重构或架构变更时，应启动 Dev Workflow 进行管理。
