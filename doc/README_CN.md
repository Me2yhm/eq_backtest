# eq-backtest

基于因子预测的 A 股量化回测引擎。支持 **日频 / 15 分钟 / 5 分钟** 三种频率，
多组合规模并行回测，提供盘口级约束（涨跌停、ST、上市时长、停牌）与份额级
（share-based）的持仓模拟，并自带 pool 预处理磁盘缓存与结果可视化。

---

## 1. 项目设计

### 1.1 整体流程

```
config.yml ──覆盖──▶ config.py（默认值）──▶ run.py
                                                  │
             ┌────────────────────────────────────┤
             ▼                                    ▼
      data_loader.build_pool()          portfolio.generate_portfolio()
   （行情 + 预测 + 日频约束 → pool）      （Numba JIT 核心模拟，逐 bar 调仓）
             │                                    │
             └────────────┬───────────────────────┘
                          ▼
                 evaluate_portfolio_result()
          （收益率聚合、超额收益、绩效指标、PNL 帧）
                          │
                          ▼
             输出 CSV / Parquet / 图表 到 OUTPUT_DIR
```

四个核心模块职责清晰：

| 模块 | 职责 |
|---|---|
| `config.py` | 全部可调参数。默认值定义于此，由 `config.yml` 深度合并覆盖 |
| `data_loader.py` | 频率无关的数据加载：行情、预测、日频约束归一化 → 构建统一 `pool`，带缓存 |
| `portfolio.py` | 频率无关的组合模拟引擎（Numba JIT 核心 + 持仓物化），含理想目标层与实际渐进成交层 |
| `run.py` | 入口：加载 pool → 逐组合规模跑模拟 → 收益率/指标计算 → 写盘 + 绘图 |
| `metrics.py` | 绩效指标（年化收益、Sharpe、最大回撤、Calmar）与持仓周期统计 |
| `plotting.py` | 可视化（累积超额、仓位分布、持仓天数、指标表） |

> **频率无关设计**：不同频率只改变"数据加载"与"收益率日频聚合"两步，
> 组合模拟核心 `_simulate_portfolio_core` 对三种频率共用同一套 Numba 代码。

### 1.2 设计要点

- **分层持仓**：`target`（理想/排名带层）与 `held`（实际成交层）分离。
  实际持仓可因 T+1 冻结与新建仓并存而**暂时超过** `port_size`。
- **信号与成交解耦**：默认 `trade_on_next_bar=True`（daily 内置），
  用上一 bar 的信号排序决定本 bar 的执行；15min/5min 亦可配置。
- **份额级记账**：以股数 `current_shares` 与盯市价格核算 PnL，
  交易成本按换手率从组合净值中扣除，而非简单权重减法。
- **约束分层**：日频基准约束（ST、上市时长、停牌天数）广播到日内各 bar；
  日内涨跌停由 `limit_up_price / limit_down_price` 与成交 VWAP 比对得出。

---

## 2. 环境安装与运行

### 2.1 安装依赖

```bash
pip install uv          # 若尚未安装
uv sync                 # 根据 pyproject.toml 安装依赖（pandas/polars/numba/matplotlib 等）
```

### 2.2 运行回测

```bash
.venv/bin/python run.py        # Linux
.venv\Scripts\python.exe run.py  # Windows
```

运行结束后，结果写入 `config.py` 中 `OUTPUT_DIR` 指定目录，
目录名由 `long/short + pool_size + frequency` 拼接，例如：

- `output_long_4400_daily/`
- `output_long_4400_15min/`
- `output_long_4400_5min/`

> 修改 `config.yml` 后直接重新运行即可；pool 缓存会在数据或配置变化时自动失效重建。

### 2.3 调试模式

在 `config.yml` 中开启 `debug: true` 并指定 `debug_symbol` / `debug_datetime`，
终端会打印该股票在目标 bar 的开/平仓逐项判定原因（如未进排序、不满足 size_cut、
T+1 冻结、现金不足等），用于定位"为什么没买/没卖"。

---

## 3. 配置说明（config.yml）

### 3.1 配置加载机制

所有配置项都有默认值（定义在 `config.py` 的 `_DEFAULTS`）。
`config.yml`（git-ignored，不提交）只写**需要覆盖的项**，加载时对默认值做
**深度合并**：缺失键沿用默认，字典键递归合并。路径型键自动转换为 `Path`。

### 3.2 时间区间

| 键 | 默认值 | 说明 |
|---|---|---|
| `start` | `"2019-01-02"` | 回测起始日期 |
| `end` | `"2026-01-01"` | 回测结束日期（含） |

### 3.3 频率与数据路径

`frequency` 取值必须在 `freq_config` 的键中，各频率的行情、预测目录、
horizon、成交价列可独立配置：

| 键（freq_config 子表） | 默认值 | 说明 |
|---|---|---|
| `market_data` | 各频率独立 | 行情 parquet 路径 |
| `preds_dir` | 各频率独立 | 预测文件目录 |
| `horizons` | 各频率独立 | 参与合并的预测文件名片段（后缀匹配） |
| `market_columns.execution_vwap` | `vwap30` / `vwap15` / `vwap5` | 成交（执行）VWAP 列 |
| `market_columns.bar_close` | `close_ex` / `close` | bar 收盘价列 |
| `market_columns.prev_close` | `null`（自动推导） | 前收列（daily 可选） |
| `trade_on_next_bar` | `false`（daily 默认 `true`） | 该频率是否下一 bar 成交 |

默认配置示例（`config.py`）：

```yaml
frequency: "5min"
freq_config:
  daily:
    market_data: "/ext/eq_data/daily_with_limit_prevcap.pqt"
    preds_dir: "data/preds_size"
    horizons: ["3d", "5d", "10d"]
    market_columns: {execution_vwap: "vwap30", bar_close: "close_ex", prev_close: null}
    trade_on_next_bar: true
  "15min":
    market_data: "/ext/eq_data/15min_bar_full_left_close.parquet"
    preds_dir: "/ext/trq"
    horizons: ["predictions"]
    market_columns: {execution_vwap: "vwap15", bar_close: "close"}
  "5min":
    market_data: "/ext/eq_data/5min_bar_full_left_close.parquet"
    preds_dir: "/tmp/eq_preds/output_mse/output_bs16/predictions"
    horizons: [""]
    market_columns: {execution_vwap: "vwap5", bar_close: "close"}
```

> `horizons: [""]` 表示匹配目录下任意单文件（5min 生产者常用固定文件名）。

### 3.4 基准

| 键 | 默认值 | 说明 |
|---|---|---|
| `market_cache_dir` | `data/market_cache` | 本地市场缓存目录，每个基准一个规范化 CSV |
| `benchmark_symbol` | `"000852"` | 本次回测使用的缓存基准标识；用于输出文件命名 |

### 3.5 股票池与策略参数

| 键 | 默认值 | 说明 |
|---|---|---|
| `universe` | `null` | 开仓股票池（如 `["000852.XSHG"]`，指数成分限制）；`null` 为全市场 |
| `allow_st_open` | `false` | 是否允许对 ST 股票开仓 |
| `pool_size` | `4400` | 候选池大小（按日频 `size_rank ≤ pool_size` 纳入） |
| `port_sizes` | `[800]` | 实际持仓数量列表，支持多组并行回测 |
| `thresh_out_buffer` | `600` | 退出缓冲：持仓滑出 `port_size + buffer` 排名才平仓 |
| `close_on_size_drop` | `false` | 持仓跌出 size 池是否强制平仓 |
| `is_short` | `false` | `true` 做空排名靠后股票，否则做多排名靠前股票 |
| `strict_first_bar_top_n` | `false` | 首执行 bar 只允许严格 top-N 信号开仓 |
| `trade_on_next_bar` | `false` | 通用默认（各频率可用 `freq_config` 覆盖） |
| `weight_mode` | `"equal"` | `equal` / `rank_linear` / `rank_square` |
| `max_weight_multiple` | `2.0` | 非等权模式下单票权重相对等权的上限倍数 |
| `nosuspend_days` | `10` | 可开仓所需的最少连续正常交易日（停牌过滤） |

### 3.6 聚合与成本

| 键 | 默认值 | 说明 |
|---|---|---|
| `agg_mode` | `"simple"` | 日内收益率聚合成日频：`simple`=求和，`compound`=复利 |
| `compounding` | `false` | 年化方式：`false`=算术年化（`mean×242`），`true`=几何年化 |
| `cost_per_turnover` | `0.00045` | 单边交易成本（4.5bp，按换手单位扣减） |
| `portfolio_initial_value` | `1e8` | 组合初始资金 |
| `exclude_period` | `["2024-01-01","2024-03-31"]` | 仅从最终评价指标序列剔除该区间；行情/持仓时间轴保持完整。`null` 关闭 |

### 3.7 预测合并与缓存

| 键 | 默认值 | 说明 |
|---|---|---|
| `prediction_merge_mode` | `"concat_disjoint"` | `concat_disjoint`=不重叠日期拼接；`mean`=重叠日期按 `(datetime,symbol)` 求均值 |
| `use_pool_cache` | `true` | 是否启用 pool 磁盘缓存 |
| `pool_cache_dir` | `data/.cache` | 缓存目录 |

缓存 key 为行情/预测文件路径、大小、修改时间与关键配置参数的 SHA256；
数据或配置变化后自动失效。手动删除 `data/.cache/` 可强制重建。

### 3.8 输出

| 键 | 默认值 | 说明 |
|---|---|---|
| `record_target_weights` | `false` | 是否记录理想目标权重（非 daily 自动开启） |

---

## 4. 数据结构与数据加载

### 4.1 各频率数据源与 schema

#### 日频（daily）

行情 `/ext/eq_data/daily_with_limit_prevcap.pqt`：

```
symbol, date, turnover, log_size, size_rank, industry, index, ret,
is_limit_up, is_limit_down, limit_up_price, limit_down_price,
close_ex, vwap30, vwap30ori, listed_Satisfied, is_ST, normal_days
```

预测：`trade_date, stock_code, prediction`（long 型），或含 `pred_t3/pred_t5/pred_t10`
等多列（加载时归一化到 `pred`）。

日频收益路径为 **前收 → 当日 VWAP30 → 当日 close_ex** 两段式
（对应 `market_columns` 的 `prev_close → execution_vwap → bar_close`）。

#### 15 分钟（15min）

行情 `/ext/eq_data/15min_bar_full_left_close.parquet`：

```
symbol, datetime, open, close, turnover, vwap15, vwap_ret
```

#### 5 分钟（5min）

行情 `/ext/eq_data/5min_bar_full_left_close.parquet`：

```
symbol, datetime, open, close, turnover, vwap5, vwap_ret
```

### 4.2 日频约束广播（日内频率专用）

日内频率缺少 ST / 上市时长 / 停牌 / 涨跌停价等日频约束，因此从
`freq_config["daily"]["market_data"]` 加载 `load_daily_flags`，与日内 bar 按
`(date, symbol)` 左连接广播：

- `can_open_base`：ST、上市时长、停牌天数、universe 综合可开仓条件
- `is_limit_up / is_limit_down`：由 `execution_price` 与 `limit_up_price /
  limit_down_price` 比对（容差 0.0005）得出
- `can_trade_buy / can_trade_sell`：`turnover > 0` 且非单边涨跌停
- `can_open`：买/卖双向可交易 且 通过日频基准条件

### 4.3 预测归一化与合并

所有预测文件统一归一化为 long 型 `[datetime, symbol, pred]`，支持三种源格式：

1. `[datetime, symbol, pred]` 或 `[datetime, symbol, prediction]`
2. `[trade_date, stock_code, prediction]`
3. 宽表 pandas parquet（`__index_level_0__` 为日期索引，自动 `unpivot`）

多个文件按 `horizons` 后缀过滤后合并：

- `concat_disjoint`：校验各文件日期区间不重叠后纵向拼接
- `mean`：重叠文件按 `(datetime, symbol)` 分组求 `pred` 均值

### 4.4 统一 pool 结构与编码

加载后市场行与预测按 `(datetime, symbol)` 左连接，形成统一池，并编码成
`BacktestDataset`（numpy 数组）供 Numba 核心消费：

| 字段 | 说明 |
|---|---|
| `bars` / `bar_offsets` | 所有 bar 时间戳与行偏移（`bars[bar_idx:bar_idx+1]` 区间内为该 bar 全部股票行） |
| `symbols` / `row_symbol_ids` | 符号表与每行的 symbol id |
| `pred` | 预测值（非有限值不参与排序） |
| `size_rank` / `log_size` | 日频市值代理排名 / 市值代理（size_rank 仅在**可开仓域**内重排） |
| `execution_vwap` / `bar_close` / `prev_close` | 成交价 / bar 收盘价 / 前收 |
| `vwap_ret` | bar 内 VWAP 收益（用于持仓物化展示） |
| `can_open` / `can_trade_buy` / `can_trade_sell` / `can_open_base` | 各层可交易/可开仓标记 |
| `bar_session_index` | 日内 bar 所属会话（午盘前后）索引 |

> **size_rank 语义**：先在剔除 ST / 未上市 / 正常天数不足的**可开仓域**内对
> `log_size` 降序排名，不合格行置 `size_rank=999999` 移出 size 池。这与外部
> 基准口径一致，避免不合格票挤占排名位次。

### 4.5 缓存

`build_pool` 以行情/预测文件签名（路径、大小、mtime）与关键配置的 SHA256 作为
缓存 key。命中则跳过加载直接返回 `BacktestDataset`。

---

## 5. 回测逻辑（portfolio.py）

组合模拟是**逐 bar 的渐进成交**过程，每 bar 顺序执行以下步骤：

### 5.1 信号与排序

- 每 bar 先确定信号 bar：`trade_on_next_bar` 时取上一 bar，否则取当前 bar。
- 信号 bar 内对 `pred` 有限的行按预测值排序（做多降序 / 做空升序），
  得到排序行与每只股票的 `rank_by_symbol`。
- **Ideal 排名**：`rank` 只在「size 池内 或 已在理想持仓」中递增
  （对应外部口径：可开池 且 持仓内），其余股票 rank=0。

### 5.2 理想目标层（target）

1. **保留旧目标**：已持有的理想目标若排名仍 ≤ `thresh_out`（= `port_size +
   buffer`）且当日通过日频基准校验，则继续保留。
2. **补充新目标**：沿信号排序顺序，`signal_rank ≤ port_size` 且通过
   `can_open_base`（ST/上市/停牌/universe）的股票进入理想目标，填满
   `port_size` 为止。理想层**忽略**日内涨跌停、零成交等执行限制。
3. **目标权重**：`equal` 为等权 `1/port_size`；`rank_linear` / `rank_square`
   按 `(thresh_out - rank + 1)/thresh_out` 打分并归一化，受 `max_weight_multiple`
   上限约束。

### 5.3 实际卖出层

- 卖出候选 = 实际持仓权重 `> 目标权重` 的股票。
- 排序键：`target 标记（有目标优先卖 10 - 需减仓量）`，即 **目标内低权重优先、减仓量大优先**。
- 逐笔执行：需满足 `can_close`（做多时 = 可卖）且扣除 T+1 冻结权重后仍有可卖量。
- 卖出受现金预算限制（卖出释放资金最多 1.0 单位权重）。

### 5.4 实际买入层

- 买入候选 = 理想目标中 `目标权重 > 当前权重`（有缺口）且 `can_open`（双向可交易）
  的股票。
- 排序键：`pred 降序`（预测越高越优先补仓）。
- 逐笔执行：**先补已有持仓**（保持连续性），现金耗尽即止；新开仓立即进入
  **T+1 冻结**（`frozen_weights`）。

### 5.5 T+1 冻结

- 当日新买入的权重被冻结，当日起不可卖出。
- 冻结在**下一自然日**（按 `bar_day_index` 变化）的第一个 bar 释放。
- 因此实际持仓可能因"冻结旧仓 + 新建仓"并存而超过 `port_size`。

### 5.6 份额级记账（share-based）

每 bar 记账顺序：

1. **Step 1（close→vwap）**：上期持仓按 `shares × (execution_vwap − prev_close)` 计 PnL，更新组合净值。
2. **Step 2（盯市权重）**：调仓前按 `shares × execution_vwap / 净值` 计算盯市权重 `marktomarket_weights`。
3. **Step 3（成本 + shares 同步）**：有信号时按
   `cost = 净值 × 2 × turnover × cost_per_turnover` 扣成本；随后把权重同步为股数
   `shares = weight × 净值 / 价格`（`trade_on_next_bar` 时用信号 bar 收盘价换算，否则用执行 VWAP）。
4. **Step 4（vwap→close）**：本期持仓按 `shares × (bar_close − execution_vwap)` 计 PnL，再更新净值。
5. **Step 5（记录收益）**：`bar_ret = (Step1 PnL + Step4 PnL − 交易成本) / 期初净值`。

**换手率**：仅在**有信号**的 bar 计算
`turnover = 0.5 × Σ|effective_target − marktomarket_weights|`；
非信号 bar 的换手率为 0（价格漂移不计入）。`effective_target` 在不可交易
（VWAP 无效）时退化为盯市权重，避免虚假换手。

---

## 6. 收益率计算逻辑

### 6.1 bar 收益 → 日收益

- **daily**：bar 收益即日收益。
- **15min / 5min**：按自然日聚合，
  `agg_mode="simple"` 时 `Σ bar_ret`，`"compound"` 时 `Π(1+bar_ret)−1`。

### 6.2 超额收益

$$excess = \begin{cases} portfolio - benchmark, & \text{做多} \\ benchmark - portfolio, & \text{做空} \end{cases}$$

基准收益按日期对齐到组合收益索引（日内频率按自然日广播）。交易成本已在
份额模拟器中扣除，超额收益不再重复扣费。

### 6.3 绩效指标（metrics.py）

| 指标 | 公式 / 口径 |
|---|---|
| `Ann. Return` | 算术年化：`mean(daily) × 242`；`compounding=true` 时几何年化 `(∏(1+r))^(1/years) − 1` |
| `Volatility` | 日收益标准差 × √242 |
| `Sharpe` | `(Ann.Return − risk_free) / Volatility` |
| `Max Drawdown` | 外部口径：`wealth = 1 + cumsum(r)`，`drawdown = wealth − expanding_max(wealth)`，取 `quantile(0.001)` |
| `Calmar` | `Ann.Return / |Max Drawdown|` |
| `Ann. Turnover` | `mean(日换手率) × 242` |

> `exclude_period` 在完整回测结束后，仅从**评价指标序列**中剔除该区间；
> 行情、持仓、调仓时间轴保持完整。

---

## 7. 产出结果

回测结果写入 `OUTPUT_DIR`（如 `output_long_4400_daily/`），对每个 `port_size`
生成一组文件：

### 7.1 汇总文件

| 文件 | 内容 |
|---|---|
| `metrics_{pool}_{bm}.csv` | 每个组合规模一行的绩效指标（Ann. Return / Volatility / Sharpe / Max Drawdown / Calmar / Ann. Turnover） |
| `returns_{pool}_{bm}.csv` | 日频超额收益率（每列一个组合规模） |
| `cumrets_{pool}_{bm}.csv` | 日频**累积**超额收益率（超额收益累加） |
| `portfolio_pnl_{size}.csv` | 逐日 PnL 帧：`all_pl`（组合累计）、`alpha_pl`（超额累计）、`benchmark`（基准累计）、`daily_strategy` / `daily_benchmark` / `daily_alpha`（当日值）、`daily_tto`（当日单边换手）、`close_count`（当日平仓数） |

### 7.2 持仓文件

| 文件 | 内容 |
|---|---|
| `positions_{size}.csv` | 逐 bar 实际持仓明细（index = `date × symbol`），列包括：`turnover, log_size, size_rank, industry, index, ret, weight_actual, is_limit_up, is_limit_down, listed_Satisfied, is_ST, normal_days, tradable, can_open, pred, pred_rank, vwap_ret` |
| `target_weights_{size}.csv` / `.parquet` | 理想目标权重快照（`date, symbol, weight_target`），非 daily 频率自动记录 |

### 7.3 图表（`plots_long/` 或 `plots_short/`）

| 图 | 内容 |
|---|---|
| `portfolio_results.png` | 各组合规模累积超额收益曲线 + 每日持仓数 / 平仓数 |
| `metrics_table.png` | 绩效指标表（含累计超额收益曲线） |
| `holding_days_p{size}.png` | 持仓天数随时间变化（平均 / 中位 / 已平仓平均） |
| `position_size_heatmap_p{size}.png` | 持仓数量在 size_rank 区间的热图 |

---

## 8. 注意事项

- **核心文件**：回测对齐工作只修改 `portfolio.py`、`data_loader.py`、`config.py`；
  不要改动 `metrics.py` / `plotting.py` / 外部数据。
- **市值代理**：使用 `log_size` / `size_rank`，不使用外部 `market_cap`。
- **T+1 冻结**：按自然日释放（非会话），避免同日内错误换手。
- **成本口径**：单边 4.5bp，成本按换手率从净值扣减。
- **基准**：CSI 1000（000852）。
- `config.yml` 为 git-ignored 的个人覆盖文件，默认值见 `config.py`。
