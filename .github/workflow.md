# Backtest Validation Workflow

> 本地回测 vs 外部回测的 validation 完整工作流。
> 目标：在相同的预测值和参数下，让本地持仓与外部持仓完全对齐，使业绩指标一致。

最后更新：2026-07-03

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [项目结构](#2-项目结构)
3. [外部回测逻辑速查](#3-外部回测逻辑速查)
4. [标准工作循环](#4-标准工作循环)
5. [当前 Residual 结构](#5-当前-residual-结构)
6. [不可变规则](#6-不可变规则)
7. [诊断脚本详解](#7-诊断脚本详解)
8. [快速参考命令](#8-快速参考命令)
9. [文档分工](#9-文档分工)

---

## 1. 背景与目标

### 1.1 任务定位

我们独立实现了一套 15 分钟频率的 A 股量化回测系统。外部团队也有独立实现。双方使用：

| 资源 | 是否共享 | 路径 |
|------|----------|------|
| 预测值 | ✅ 同一份 | `/ext/trq/predictions.parquet` |
| 策略参数 | ✅ 相同 | pool_size=4400, port_size=800, thresh_out=1400, 等权, T+1 |
| 行情数据 | ❌ 各自维护 | 本地: `/ext/eq_data/`, 外部: 自有一份 |

### 1.2 当前状态

- 持仓 overlap（相对外部）：**~96.35%**（从 ~85% 逐步收敛）
- `mean_diff`（每 bar 平均持仓差异）：**~55.7**
- `median_diff`：**~58**
- 全区间 `full_match` 的 bar 很少，但差异分布已大幅收窄

### 1.3 最终目标

- 持仓 overlap → 尽可能接近 100%
- 年化收益率、夏普、卡玛、最大回撤等指标与外部一致
- 对无法消除的残差给出清晰归因

---

## 2. 项目结构

### 2.1 本地回测代码

```
eq-backtest/
├── config.py                 # 所有可调参数
├── run.py                    # 回测入口
├── data_loader_15min.py      # 15m 数据加载 → BacktestDataset15Min
├── portfolio_15min.py        # 15m 调仓引擎（target + actual）
├── metrics.py                # 业绩指标
├── plotting.py               # 画图
├── data_loader.py            # 日频版本（本任务不使用）
├── portfolio.py              # 日频版本（本任务不使用）
├── output_long_4400/         # 输出目录
│   ├── positions_800.csv     # 每 bar 持仓明细
│   ├── portfolio_pnl_800.csv # PnL 与收益分解
│   └── plots_long/           # 图表
├── tmp/                      # 诊断脚本与参考文档
└── .github/                  # Agent 指令与工作流
    ├── copilot-instructions.md
    └── workflow.md           # 本文件
```

### 2.2 核心模块职责

| 模块 | 输入 | 输出 | 关键语义 |
|------|------|------|----------|
| `data_loader_15min.build_pool_15min()` | 行情 + 预测 + 日频约束 | `BacktestDataset15Min` | `can_open_base`, `can_open`, `can_close`, `tradable`, `size_rank` |
| `portfolio_15min._simulate_portfolio_core_15min()` | pool + 参数 | 持仓记录、收益、换手 | target（理想层）、actual（执行层）、T+1 冻结 |
| `portfolio_15min._materialize_positions_15min()` | 模拟输出 | `positions_800.csv` | 每 bar 每票的 weight、pred_rank、vwap_ret 等 |
| `run.compute_returns()` | 持仓收益 | 业绩指标 | 日频聚合、超额收益、交易成本扣除 |

### 2.3 数据清单

| 路径 | 内容 | 关键列 |
|------|------|--------|
| `/ext/trq/predictions.parquet` | 共享预测值 | `trade_date`, `stock_code`, `prediction` |
| `/ext/trq/weights.parquet` | 外部 actual 持仓 | `trade_date`, `stock_code`, `weight`, `weight_ideal` |
| `/ext/trq/eligible_full.csv` | 外部每个 bar 的 eligible（~17GB） | `trade_date`, `stock_code`, `pred_rank`, `_valid_buy`, `_valid_sell`, `held_ideal_entry` |
| `/ext/eq_data/15min_bar_full_left.parquet` | 本地 15m 行情 | `datetime`, `symbol`, `vwap_ret`, `vwap15`, `turnover` |
| `/ext/eq_data/daily_with_limit.pqt` | 本地日频约束 | `date`, `symbol`, `limit_up_price`, `limit_down_price`, `listed_Satisfied`, `is_ST`, `normal_days` |
| `/ext/eq_data/ret_csi_1000.csv` | 中证 1000 日收益 | `date`, `ret` |

### 2.4 诊断脚本清单

所有脚本位于 `tmp/`，详见 [第 7 节](#7-诊断脚本详解)。

### 2.5 参考文档

| 文件 | 内容 |
|------|------|
| `tmp/external_logic_doc.md` | 外部回测逻辑的详细说明：portfolio 步骤（universe → eligible → rank → ideal weights → progressive rebalance）和 backtest 步骤（逐 bar 模拟、涨跌停约束、收益计算） |
| `tmp/actual.md` | 外部 `actual` 逐 bar 更新逻辑的补充说明：卖出优先级、买入匹配规则、`held_ideal_entry` 语义、日内更新节奏 |
| `tmp/holding_diff_worklog.md` | **持仓差异工作台账**：所有有效改动的状态、代码、验证结果、边际贡献 |
| `tmp/divergence_log.md` | **原因推测与发散记录**：对当前残留差异的原因推测 + 导致发散的失败实验 |

---

## 3. 外部回测逻辑速查

以下是从 `external_logic_doc.md` 和 `actual.md` 中提炼的核心逻辑。本地 `portfolio_15min.py` 需要对这些语义。

### 3.1 Ideal Target（理想层）

```
for each bar:
  # Universe
  universe_mask = _valid_universe ∧ finite(market_cap) ∧ top 4400 by market_cap desc

  # Eligible
  eligible_mask = (universe_mask ∧ finite_prediction) ∨ (hold_mask ∧ _valid_listed)

  # Rank in eligible
  rank_pos = sort eligible by prediction desc (1-based)

  # Sell: hold_mask ∧ rank > 1400 ∧ _valid_sell
  # Buy:  rank ≤ 800 ∧ _valid_buy, fill up to 800 slots
  # Weight: equal = 1 / n_holdings
```

### 3.2 Actual Progressive Execution（执行层）

```
state: current_weights, frozen_weights, cash_weight = 1.0

for each bar:
  # ── Per-calendar-day T+1 release ──
  if new calendar day: frozen_weights = 0

  # ── Sell phase ──
  sellable = max(0, current_weights - frozen_weights) masked by _valid_sell
  sell_order = sort(current > target, key=(target_weight asc, reduction desc, stock_idx asc))
  for each in sell_order:
    sell = min(reduction, sellable[i], remaining_sell_budget)
    execute sell

  # ── Buy phase ──
  deficit = max(0, target - current_after_sell)
  buy_order = sort(deficit desc, stock_idx asc)
  1) top-up existing positions first (current > 0)
  2) then open new positions (current == 0)
  for each in buy_order:
    buy = min(deficit, cash, capacity)
    execute buy
    if new position: frozen_weights += buy
```

### 3.3 关键标志位

| 外部标志 | 含义 | 本地对应 |
|----------|------|----------|
| `_valid_listed` | 上市 ≥1d, 未退市, 未停牌, 非 ST | `can_open_base` 中的 `listed_Satisfied` + 非 ST |
| `_valid_trade` | 上市 ≥1d, 未退市, 未停牌, amount > 0 | `tradable` |
| `_valid_universe` | 上市 ≥20d, 近 10d 无停牌, 非 ST | 日频过滤条件 |
| `_valid_buy` | `_valid_trade` ∧ 不在涨停板 | `can_open`（`can_trade_buy` ∧ `can_open_base`） |
| `_valid_sell` | `_valid_trade` ∧ 不在跌停板 | `can_close`（`can_trade_sell`） |
| `held_ideal_entry` | bar 开始时 ideal 已持有 | 不是 actual 执行指令，仅用于 eligible 计算 |

---

## 4. 标准工作循环

每次尝试缩小持仓差异时，严格按以下 6 步执行。

### Step 1: 形成假设

基于当前归因数据，提出一个**具体的**差异来源假设。

**信息来源**：
- `tmp/attribute_residual_overlap.py` 的最新输出
- `tmp/drill_out_of_pool.py` / `tmp/drill_carry.py` 的分析结论
- `tmp/compare_positions.py` 的 by-time 分布（看哪些时段差异大）

**假设应明确**：
- 差异发生在 target 层还是 actual 执行层？
- 具体是哪个语义点不一致？（rank 排序方式、eligible 定义、买卖优先级排序、T+1 冻结时机、sell_budget、…）
- 预期影响的方向和大致量级

**先记录再动手**：在 `tmp/holding_diff_worklog.md` 中追加一条记录，写清楚假设和预期。

### Step 2: 实施修改

- **范围**：仅限 `portfolio_15min.py`、`data_loader_15min.py`、`config.py`
- **原子性**：一次只改**一个**语义点
- **可回退**：修改前确保工作区干净（`git stash` 或 commit）
- 如果修改需要新的诊断脚本辅助验证，放入 `tmp/` 并在 worklog 中登记

### Step 3: 运行回测

```bash
cd /home/yhm/eq-backtest && /home/yhm/eq-backtest/.venv/bin/python run.py
```

确认：
- 回测正常完成（无异常退出）
- `output_long_4400/positions_800.csv` 已更新
- `output_long_4400/portfolio_pnl_800.csv` 已更新

### Step 4: 运行比较

**每次必做**：

```bash
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/compare_positions.py
```

关注指标：
- `mean_overlap`：相对 external 的持仓重叠率（目标：上升）
- `mean_diff`：每 bar 平均差异数（目标：下降）
- `first_bar_mean_diff` vs `rest_mean_diff`：首 bar 是否有异常
- `full_match` 和 `diff <= 20` 的 bar 数

**当 residual 分布发生变化时按需运行**：

```bash
# 逐行 mismatch 导出 + 回连 pool（看具体哪些票、什么原因）
python tmp/inspect_actual_mismatch.py
python tmp/inspect_actual_mismatch_pool.py

# 全区间 residual 归因（看 target/execution/carry 三类占比变化）
python tmp/attribute_residual_overlap.py

# A/B 归因（如果一次改了多个语义点，测单项边际贡献）
python tmp/ab_active_holding_changes.py
```

### Step 5: 记录结果

在 `tmp/holding_diff_worklog.md` 中追加一条记录，使用以下模板：

```markdown
### N. <简短标题>

- 状态：`active` | `reverted` | `diagnostic-only`
- 文件：`<path>`
- 原因：<一句话>
- 代码：
  ```python
  <最小代码片段>
  ```
- 验证命令：
  ```bash
  <command>
  ```
- 贡献：
  - mean_overlap: X% → Y% (Δ = +Z pp)
  - mean_diff: A → B (Δ = -C)
  - <定性结论>
```

### Step 6: 决策

| 结果 | 行动 |
|------|------|
| overlap ↑ / diff ↓ 明显（>0.5pp） | **保留**，状态 `active`，记录到 `holding_diff_worklog.md` |
| 变化微小（<0.1pp） | 视情况保留或回退，注明原因 |
| overlap ↓ / diff ↑（发散） | **回退代码**，**不要**记入 `holding_diff_worklog.md`，改为记入 `tmp/divergence_log.md`（假设、修改、变化量、结论） |

---

## 5. 当前 Residual 结构

来自 `attribute_residual_overlap.py` + `drill_*.py` 的综合分析。

### 5.1 `only_ext`（外部有，本地没有）：~720K bar-name

| 类别 | 占比 | 根因 |
|------|------|------|
| Size pool mismatch | 54.5% | 外部用复合 `pred_rank` 做 top-4400，本地用纯 `size_rank` 过滤。一批外部 pred_rank≈385 的票在本地 size_rank 被挤到 4400–4500 边界外 |
| External carry | 32.3% | 外部 actual 持但 external ideal 不持。其中 95.8% 本地也 carry 同票。External-only 部分是因为本地 target 没选中 |
| Local execution lag | 10.4% | 本地 target 想持，但 local actual 还没买到位 |
| Target misc | 2.8% | 池内但未进 local target 或被 can_open 挡住 |

### 5.2 `only_local`（本地有，外部没有）：~720K bar-name

| 类别 | 占比 | 根因 |
|------|------|------|
| Local frozen carry | 37.4% | T+1 日内冻结：当天买入的仓位当天不能卖。99.5% 是全仓冻结 |
| External execution lag | 32.6% | External ideal 想持但 external actual 还没买到位 |
| Local target mismatch | 29.8% | 本地 target 想持但 external ideal 不想持 |

### 5.3 缩小差异的优先级

1. **排名函数对齐** — 影响最大（~54.5% 的 only_ext），需要理解外部 `pred_rank` 的构造方式
2. **T+1 冻结语义确认** — 确认外部是否有对应机制；如有差异需决定是否对齐
3. **Target 选股残差** — remaining 细部调整

---

## 6. 不可变规则

以下规则已经过验证和讨论，**除非明确重新决定，否则不可修改**：

### 6.1 框架层面

| 规则 | 说明 |
|------|------|
| 市值代理 | 使用 `log_size` / `size_rank`，不引入外部 `market_cap` |
| T+1 释放 | 按自然日（calendar day），不是交易时段（session）。按 session 释放已被验证会导致错误的同日盘中换仓，已回退 |
| 信号时序 | `trade_on_next_bar=True`：用前一 bar 的信号在当前 bar 执行 |
| 基准 | 中证 1000（000852） |
| 成本 | 单边 4.5bp（`COST_PER_TURNOVER = 0.00045`），不扣除滑点 |
| 排除期 | 2024-01-01 至 2024-03-31 超额收益置零 |

### 6.2 代码修改范围

- ✅ 可修改：`portfolio_15min.py`, `data_loader_15min.py`, `config.py`
- ❌ 不可修改：`metrics.py`, `plotting.py`, `data_loader.py`, `portfolio.py`
- ❌ 不可修改：外部数据（`/ext/trq/`, `/ext/eq_data/`）

### 6.3 工作流规则

1. **原子性**：每次只改一个语义点
2. **可回退**：用 git 管理改动，确保随时可回退
3. **先记录再动手**：在 worklog 中先写假设，再改代码
4. **改完必验证**：每次修改后跑 `run.py` + `compare_positions.py`
5. **新脚本入 tmp/**：所有临时诊断脚本放入 `tmp/`，并在 worklog 中登记

---

## 7. 诊断脚本详解

### 7.1 `compare_positions.py` — 端到端持仓比较

**用途**：本地 `positions_800.csv` vs 外部 `weights.parquet` 的 bar 级对齐比较。

**输入**：
- `/ext/trq/weights.parquet`
- `output_long_4400/positions_800.csv`

**输出**（控制台）：
- 对齐后的总 bar 数、丢弃数
- `mean_diff`, `median_diff`, `mean_overlap`
- `full_match` / `diff <= 20` 的 bar 数
- `first_bar_mean_diff` vs `rest_mean_diff`（首 bar 诊断）
- 按 time-of-day 分桶统计
- 前 32 个对齐 bar 和后 16 个 bar 的逐行明细

**命令**：
```bash
python tmp/compare_positions.py
```

### 7.2 `attribute_residual_overlap.py` — Residual 归因

**用途**：把当前 baseline 的 actual mismatch 按 target mismatch、execution gap、carry 三类拆开，并细分到 size pool、冻结状态、可买卖约束。

**输入**：
- `/ext/trq/weights.parquet`
- `output_long_4400/positions_800.csv`
- `tmp/eligible.csv`（由 `compare_local_eligible_window.py` 生成）

**输出**：
- `tmp/residual_overlap_attribution.csv`
- 控制台打印分桶统计

**命令**：
```bash
python tmp/attribute_residual_overlap.py
```

### 7.3 `ab_active_holding_changes.py` — A/B 归因

**用途**：在内存中直接跑当前引擎逻辑，逐项关掉单个改动，测量每项改动的边际贡献。

**输出**：
- `tmp/ab_active_holding_changes.csv`
- 控制台打印 baseline 和每个变体的 `mean_overlap`, `mean_diff`

**命令**：
```bash
python tmp/ab_active_holding_changes.py
```

### 7.4 `inspect_actual_mismatch.py` — 逐行 Mismatch 导出

**用途**：导出前两天 `only_ext` / `only_local` 的 actual 持仓逐行差异，包含外部 `_valid_buy`, `_valid_sell`, `held_ideal_target` 和本地 `pred_rank`, `can_open`, `tradable`。

**输出**：`tmp/actual_mismatch_detail.csv`

**命令**：
```bash
python tmp/inspect_actual_mismatch.py
```

### 7.5 `inspect_actual_mismatch_pool.py` — Mismatch 回连 Pool

**用途**：把 `actual_mismatch_detail.csv` 的逐行记录回连到本地 15 分钟 pool 行上，检查对应的 `size_rank`, `can_open_base`, `can_trade_buy/sell`, `turnover`, 涨跌停标记。

**输出**：`tmp/actual_mismatch_with_pool.csv`

**命令**：
```bash
python tmp/inspect_actual_mismatch_pool.py
```

### 7.6 `compare_local_eligible_window.py` — Eligible 对比

**用途**：把本地回测逻辑导出成和外部同结构的 eligible 样本，与外部 `eligible_full.csv` 逐列对齐。

**两种模式**：
- 默认：两天窗口，落全量 row-level 中间文件
- `--summary-only`：全区间，只输出 bar 级摘要统计

**输出**：
- `tmp/local_eligible_window.csv`
- `tmp/local_vs_external_eligible_window.csv`

**命令**：
```bash
# 窗口模式
python tmp/compare_local_eligible_window.py

# 全区间摘要模式
python tmp/compare_local_eligible_window.py --summary-only
```

### 7.7 `drill_out_of_pool.py` — Size Pool Drill-down

**用途**：对 external 有但本地没有的 size-pool 外名字做逐名分析，查看外部 `pred_rank`、本地 `size_rank`、`_valid_buy` 等分布。

**输出**：
- `tmp/drill_out_of_pool/out_of_pool_details.csv`
- `tmp/drill_out_of_pool/out_of_pool_by_symbol.csv`

**命令**：
```bash
python tmp/drill_out_of_pool.py
```

### 7.8 `drill_carry.py` — Carry/Frozen 分析

**用途**：分析双边 carry 的重合度、external-only carry 的根因、local frozen carry 的冻结状态分布。

**输出**：控制台统计摘要

**命令**：
```bash
python tmp/drill_carry.py
```

### 7.9 `compare_weights.py` — 权重对比

**用途**：对双方共享持仓的首 bar 做权重比较（外部 weight vs 本地等权）。

**输出**：控制台统计（均值、分布、偏离比例）

**命令**：
```bash
python tmp/compare_weights.py
```

---

## 8. 快速参考命令

```bash
# === 核心循环（每次改动后必跑） ===

# 1. 运行完整回测
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/run.py

# 2. 端到端持仓比较
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/compare_positions.py

# === 深入诊断（按需） ===

# 3. 全区间 residual 归因
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/attribute_residual_overlap.py

# 4. A/B 归因（测单项贡献）
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/ab_active_holding_changes.py

# 5. 逐行 mismatch 导出 + 回连 pool
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/inspect_actual_mismatch.py
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/inspect_actual_mismatch_pool.py

# 6. Eligible 对比
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/compare_local_eligible_window.py --summary-only

# 7. Size pool drill-down
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/drill_out_of_pool.py

# 8. Carry/frozen 分析
/home/yhm/eq-backtest/.venv/bin/python /home/yhm/eq-backtest/tmp/drill_carry.py

# === 快速 sanity check ===
/home/yhm/eq-backtest/.venv/bin/python -c "
import polars as pl
from datetime import date
pos = pl.read_csv('output_long_4400/positions_800.csv', try_parse_dates=True)
ext = pl.read_parquet('/ext/trq/weights.parquet')
for d in [date(2019,1,2), date(2019,1,3), date(2019,1,4)]:
    local_syms = set(pos.filter((pl.col('date').dt.date()==pl.lit(d))&(pl.col('date').dt.hour()==9)&(pl.col('date').dt.minute()==31))['symbol'].to_list())
    ext_syms = set(ext.filter(pl.col('trade_date').cast(pl.Date)==pl.lit(d))['stock_code'].unique().to_list())
    overlap = local_syms & ext_syms
    print(f'{d}: overlap={len(overlap)}/{len(local_syms)} = {100*len(overlap)/max(1,len(local_syms)):.1f}%')
"
```

---

## 9. 文档分工

| 文件 | 角色 | 更新频率 |
|------|------|----------|
| `.github/copilot-instructions.md` | Agent 行为指令（自动加载） | 低（规则变更时） |
| `.github/workflow.md`（本文件） | 完整工作流规范 | 低（流程变更时） |
| `tmp/holding_diff_worklog.md` | 改动台账（状态、代码、贡献） | **高**（每次有效改动后立即更新） |
| `tmp/divergence_log.md` | 原因推测 + 发散实验记录（已回退的改动） | **中**（每次发散实验后更新） |
| `tmp/external_logic_doc.md` | 外部逻辑参考 | 低（外部逻辑变更时） |
| `tmp/actual.md` | 外部 actual 逻辑补充 | 低 |
