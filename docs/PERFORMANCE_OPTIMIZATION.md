# 性能优化说明

## 总结

## 2026-04-14 两阶段 PnL 性能回归修复

在引入两阶段收益口径后，`generate_portfolio` 的实现一度回退到 Python/pandas 后处理路径，导致热缓存命中场景从约 `4s` 上升到约 `22s`。

本次修复将两阶段收益与交易成本计算完全并回 Numba 主循环，避免了逐日字典构建与 Python 层 symbol 迭代。

### 修复前后（热缓存命中）

| 阶段 | 修复前 | 修复后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| `generate_portfolio` | 19.72s | 1.36s | 18.36s | 14.50x |
| `build_pool` | 0.80s | 0.89s | -0.09s | 0.90x |
| `compute_returns` | 0.00s | 0.01s | -0.01s | 0.44x |
| 三段合计 | 20.53s | 2.25s | 18.28s | 9.11x |

### 关键实现变更

- `BacktestDataset` 直接暴露 `close_ex`、`vwap30`、`vwap30ori` 的 NumPy 数组，避免从 `pool_frame` 转 pandas 再查价。
- `_simulate_portfolio_core` 内部直接计算两阶段日收益：
	- 隔夜段：`close_ex(t-1) -> vwap30(t)`
	- 日内段：`vwap30(t) -> close_ex(t)`
- 交易成本口径在同一核心循环中计算：`abs(Δshares) * vwap30ori(t)`，并按 `prev_total_mv` 归一化成 `daily_tto`。
- 移除了 Python 侧 `_compute_two_stage_portfolio_series`，`generate_portfolio` 直接消费 Numba 输出的 `portfolio_returns` 与 `turnover`。

### 数值一致性验证

以 `output_long_4400/portfolio_pnl_900.csv` 为基准，优化后结果与修复前两阶段实现保持一致（误差仅为浮点噪声）：

| 列 | MAE | Max Abs |
| --- | ---: | ---: |
| `daily_strategy` | 2.41e-09 | 4.99e-09 |
| `daily_tto` | 2.41e-09 | 4.99e-09 |
| `close_count` | 0.0 | 0.0 |

本轮优化将同一台机器、同一份数据上的完整回测耗时从约 `64.32s` 降低到了约 `7.68s`；在进一步加入预处理后的 pool cache 并命中热缓存后，完整回测耗时进一步下降到了约 `4.76s`。

| 指标 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 完整回测耗时（无 pool cache） | 64.32s | 7.68s | 56.64s | 8.38x |
| 完整回测耗时（热缓存命中） | 64.32s | 4.76s | 59.56s | 13.51x |

在完成性能优化并修正语义偏差后，最终验证得到的指标保持不变：

| 指标 | 数值 |
| --- | ---: |
| Ann. Return | 0.180933 |
| Volatility | 0.058914 |
| Sharpe | 3.071123 |
| Max Drawdown | -0.068785 |
| Calmar | 2.630419 |
| Ann. Turnover | 98.578803 |

需要说明的是，下面每一项加速数据都来自重构过程中的阶段性 profiling。它们可以解释时间主要花在什么地方，但并不是严格可相加的，因为不同优化之间会相互影响。

## 为什么在这些地方使用 Numba

这次重构中，Numba 主要被放在了最核心、最稳定、最适合数组计算的调仓主循环里，而不是铺满整个项目。

相关代码：

- `portfolio.py:64` `_simulate_portfolio_core`

我的选择原则是：只有当一段逻辑同时满足“调用次数很多”“数据结构规则稳定”“可以被压平成 NumPy 数组”“没有太多 Python 对象操作”这几个条件时，Numba 才值得用。

具体来说，组合调仓核心非常适合 Numba，原因有几点：

- 它是全项目最热的路径。原始 profiling 显示，绝大部分时间都耗在逐日调仓逻辑上。
- 它本质上是一个状态机。每天都在重复执行“更新当日行索引、计算排名、判断是否平仓、补足开仓、累计收益”的固定流程，这类逻辑非常适合编译成原生循环。
- 它可以被自然表示成数组。持仓状态、symbol 映射、收益、可交易标记、size rank 等信息都能压缩成定长数组，不需要复杂对象。
- 它对 pandas 的需求很低。真正需要 pandas 的只是最终输出持仓表给后续 CSV、绘图和统计使用，而不是调仓核心本身。

相反，我刻意没有在以下地方用 Numba：

- 数据加载部分。这里大量是 parquet 读取、join、排序、列运算，更适合交给 Polars 这类列式引擎，而不是自己写 Numba。
- 绘图和报表部分。这里属于 I/O 和可视化，不是数值密集型循环，Numba 不会带来明显收益。
- 边界层格式转换部分，例如 `_materialize_positions`。这里更多是把内部数组状态恢复成业务可读的表结构，重点是兼容性，不是极限算力。

所以，Numba 在这次重构中的角色不是“全项目加速器”，而是“专门负责最热、最纯计算的状态推进引擎”。这种用法收益最高，维护成本也最低。

## 为什么在这些地方使用 Polars

Polars 主要被用在数据加载、列变换、连接、导出这几个典型的表计算环节，而没有强行替代所有 pandas 代码。

相关代码：

- `data_loader.py:110` `_load_predictions_wide_mean`
- `data_loader.py:252` `build_pool`
- `run.py:60` `_write_positions_csv`

我的选择原则是：凡是以“大表列运算、parquet/csv 读写、join、sort、unpivot、批量列转换”为主的步骤，优先使用 Polars；凡是已经和现有分析/绘图代码深度耦合、且不在热点上的部分，则继续保留 pandas。

具体来说，Polars 适合这次重构中的几个关键位置：

- Prediction 加载。prediction parquet 原本是宽表，Polars 在读取 parquet、执行 `unpivot`、列过滤、排序这类列式操作上明显更高效。
- 市场数据预处理。像 `tradable`、`can_open` 这种派生列，本质上是批量列表达式，非常适合用 Polars 一次性向量化生成。
- Pool 构建。大表 join、列保序、编码前准备，本来就是 Polars 的强项。
- 持仓 CSV 导出。大表写盘是 Polars 的优势场景之一，这一点实测收益非常明显。

相反，我没有把所有 pandas 代码都替换掉，原因也很明确：

- `metrics.py` 和 `plotting.py` 这类模块本来就以 pandas / matplotlib 为中心，直接替换的收益不高，但改动面会很大。
- 回测结果最终还是需要以 pandas DataFrame / Series 的形式喂给现有统计和绘图接口，因此在边界层保留 pandas 可以减少大量兼容性工作。
- 项目的目标是“在热点处提速”，不是“为了统一技术栈而重写所有模块”。

因此，Polars 在这次重构中的定位是“高效的数据工程层”：负责重型表操作、列变换和 I/O；而 pandas 则保留在结果消费层，继续服务于统计分析和可视化。

## 具体做了哪些优化

### 1. 用 Numba 状态机替换 pandas 的逐日调仓循环

相关代码：

- `portfolio.py:31` `_build_day_orders`
- `portfolio.py:64` `_simulate_portfolio_core`
- `portfolio.py:306` `generate_portfolio`

具体改动：

- 旧实现是在 Python 层按天循环，并在循环里反复使用 pandas 做排名、刷新持仓、平仓和开仓。
- 新实现先把股票池编码成紧凑的 NumPy 数组，再在 Numba JIT 编译后的核心函数里推进整个组合状态。
- `_build_day_orders` 会预先计算每天可交易股票的顺序，并按多空方向缓存，避免在调仓主循环里重复做昂贵的 pandas 排序和排名。
- 模拟器在扫描状态的同时直接产出 `portfolio_returns`、`turnover` 和 `held_counts`，而不是在结束后再从持仓表反推。
- pandas 只在 `_materialize_positions` 这个边界层重新引入，因此现有的绘图和 CSV 输出接口仍然兼容。

为什么会更快：

- 最热路径中不再有 Python 对象调度开销，也不再依赖 pandas 的索引对齐。
- 状态存储在连续数组中，例如 `held`、`current_row`、`last_row`、`rank_by_symbol`，更新成本远低于 DataFrame 切片。
- 每天的排序结果只构建一次，后续直接复用。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 组合生成阶段 | 49.137s | 2.627s | 46.510s | 18.70x |

这是本次优化中单项贡献最大的部分。

### 2. 先在宽表上对 prediction parquet 求均值，再只做一次 unpivot

相关代码：

- `data_loader.py:86` `PredictionAlignmentError`
- `data_loader.py:90` `_load_predictions_long_form`
- `data_loader.py:110` `_load_predictions_wide_mean`

具体改动：

- 原始实现会逐个读取 prediction parquet，先分别 `unpivot` 成长表，再把所有 horizon 拼接起来，最后才按 `(date, symbol)` 做平均。
- 新的快路径先保持宽表矩阵形式，用 NumPy 直接对对齐后的 horizon 矩阵求均值，最后只在末尾执行一次 `unpivot`。
- 如果不同 horizon 文件在列集合或日期索引上不完全一致，会自动回退到原来的长表聚合逻辑，保证稳健性。

为什么会更快：

- 避免提前生成和聚合一个更大的中间长表。
- 对齐后的稠密矩阵做 NumPy 运算，比长表 concat + groupby 便宜得多。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| Prediction 加载 | 3.687s | 0.917s | 2.770s | 4.02x |

### 3. 去掉 `build_pool` 过程中的重复全量排序

相关代码：

- `data_loader.py:252` `build_pool`
- `data_loader.py:227` `_encode_dataset`

具体改动：

- `load_market_data()` 输出本身已经按 `(date, symbol)` 排好序。
- 和 prediction 的左连接会保留左表顺序，因此 `build_pool()` 中额外的全表排序其实是多余的。
- `_encode_dataset()` 也简化为直接消费已经排好序的 pool，而不是再次排序。

为什么会更快：

- 对几百万行数据做排序本身就是重操作，而且之前重复发生了不止一次。
- 去掉这些排序后，CPU 开销和临时内存分配都明显下降。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| `build_pool` 整体 | 5.945s | 2.764s | 3.181s | 2.15x |

### 4. 将大体量持仓 CSV 输出从 pandas 改为 Polars

相关代码：

- `run.py:60` `_write_positions_csv`

具体改动：

- 原来的做法是直接对大型 pandas DataFrame 调用 `positions.to_csv(...)`。
- 现在改成先把最终持仓快照转换为 Polars DataFrame，再用 `write_csv()` 输出。

为什么会更快：

- 对当前这种大表写盘场景，Polars 的 CSV writer 比 pandas `to_csv` 快得多。
- 在组合核心加速之后，CSV 导出一度成了新的最大瓶颈。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 持仓 CSV 导出 | 12.238s | 0.285s | 11.953s | 42.94x |

仅这一项就把端到端耗时直接减少了约 12 秒。

### 5. 缓存预处理后的 pool

相关代码：

- `config.py:15` `POOL_CACHE_DIR`
- `config.py:16` `USE_POOL_CACHE`
- `data_loader.py:37` `POOL_CACHE_VERSION`
- `data_loader.py:282` `_pool_cache_path`
- `data_loader.py:302` `_read_cached_pool`
- `data_loader.py:313` `build_pool`
- `run.py:92` `use_cache=cfg.USE_POOL_CACHE`

具体改动：

- 在 `build_pool()` 这一层增加了预处理后 pool 的磁盘缓存。
- 缓存内容不是原始行情，也不是单独的 prediction，而是已经完成以下步骤后的结果：
	- prediction 聚合
	- 市场数据 `tradable` / `can_open` 派生列计算
	- market 与 prediction 的 join
	- `row_idx` / `symbol_id` 编码
- cache key 会根据以下信息自动生成：
	- 市场数据文件的路径、大小、修改时间
	- 所有 prediction 文件的路径、大小、修改时间
	- `START`
	- `UNIVERSE`
	- `ALLOW_ST_OPEN`
	- `POOL_CACHE_VERSION`
- 只要这些输入没有变化，后续重复回测就可以直接命中 cache，跳过整个预处理阶段。

为什么会更快：

- `build_pool` 本质上是重复回测时最适合缓存的阶段，因为它只依赖静态输入文件和少量配置参数。
- 对研究和调参场景来说，很多次回测都是在同一批数据上重复运行，此时每次重新做 prediction 加载、join、编码都属于重复劳动。
- 把 join 后并编码好的 pool 直接落盘，可以显著降低重复运行时的启动成本。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| `build_pool`（当前代码，无缓存 vs 热缓存） | 3.157s | 0.662s | 2.495s | 4.77x |
| `build_pool`（冷缓存首次构建 vs 热缓存命中） | 4.535s | 0.699s | 3.836s | 6.49x |
| 完整回测（启用热缓存后） | 7.68s | 4.76s | 2.92s | 1.61x |

这一步的意义在于：它不会改变单次冷启动的理论上限太多，但会显著优化“同一份数据反复回测”的实际体验。

### 6. 简化 `compute_returns`，直接消费模拟器输出的序列

相关代码：

- `run.py:29` `compute_returns`

具体改动：

- 旧实现会先从 `positions.groupby("date")` 重建组合日收益，再构建一个稠密的 date-symbol 权重矩阵，然后对该矩阵做 diff 来计算换手。
- 现在模拟器已经直接输出 `portfolio_returns` 和 `turnover`，所以 `compute_returns()` 只需要对齐 benchmark，并应用交易成本和排除区间逻辑。

为什么会更快：

- 去掉了对整个持仓表的 dense pivot 和 diff。
- 不再重复构造模拟器本来就已经精确知道的信息。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 收益计算 | 0.309s | 0.003s | 0.306s | 103.00x |

这一项绝对值节省不算最大，但实现更简洁，而且与参考口径完全等价。

### 7. 将持有期统计改写为一次索引扫描

相关代码：

- `metrics.py:52` `holding_period_stats`

具体改动：

- 旧实现按日期循环，并反复执行 `positions.loc[date]` 切片。
- 新实现直接扫描已经排好序的 MultiIndex，用 NumPy 找出日期边界，再按天构造 symbol 集合并计算开仓/平仓统计。

为什么会更快：

- 在大型 MultiIndex 上反复做 pandas 切片成本很高。
- 直接扫描索引数组只需单次遍历，临时对象也更少。

量化收益：

| 阶段 | 优化前 | 优化后 | 节省时间 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 持有期统计 | 2.050s | 0.652s | 1.398s | 3.14x |

## 为保证正确性而做的修正

最快的实现只有在语义不变的前提下才有价值。重构过程中暴露了两个正确性问题，最终都修正后才接受结果。

### 1. 换手定义

- 朴素地用 `n_closed + n_opened` 计算换手，会在“同一天先卖出又重新买回同一只股票”时重复计数。
- 本项目中正确的换手定义应当等价于最终组合权重变化对应的换手。
- 当前模拟器输出已经与 positions-based 的参考路径完全一致。

### 2. Benchmark 对齐语义

- 在 benchmark reindex 后使用 `fillna(0.0)`，会改变旧版本在 benchmark 缺失日期上的行为。
- 最终实现保留了原始语义，即对齐时保持这些值为 `NaN`。

### 验证结果

修正之后：

- direct return 路径相对 positions-based reference 的最大绝对误差：`1.3877787807814457e-16`
- direct turnover 路径相对 positions-based reference 的最大绝对误差：`2.220446049250313e-16`

从工程角度看，优化后的路径已经与原始实现数值一致。

## 当前最终阶段耗时拆解

当前测得的各阶段耗时如下（热缓存命中场景）：

| 阶段 | 耗时 |
| --- | ---: |
| `build_pool_warm_cache` | 0.610s |
| `generate_portfolio_with_heatmap` | 2.283s |
| `positions_to_csv` | 0.238s |
| `compute_returns` | 0.003s |
| `portfolio_metrics` | 0.000s |
| `holding_period_stats` | 0.546s |
| `plot_holding_periods` | 0.349s |
| `save_summary_csvs` | 0.007s |
| `plot_portfolio_results` | 0.596s |
| `plot_metrics_table` | 0.115s |
| Total | 4.74s |

目前剩余的主要热点是：

1. `generate_portfolio`
2. 主模拟结束后的绘图与分析步骤
3. 冷缓存场景下的 `build_pool`

如果还要继续压缩运行时间，下一步最合理的方向是：继续完善 pool cache 策略（例如手动强制重建、分层缓存），以及在研究型运行中把较重的绘图/报表步骤做成可选开关。