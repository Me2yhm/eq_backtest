# eq-backtest

基于因子预测的 A 股日频回测框架，支持多组合规模并行回测、缓存加速与性能指标可视化。

---

## 快速开始

### 环境安装

```bash
pip install uv          # 若尚未安装
uv sync                 # 根据 pyproject.toml 安装依赖
```

### 运行回测

```bash
.venv\Scripts\python.exe run.py # Windows
.venv/bin/python run.py # Linux
```

回测结果（指标 CSV、收益率 CSV、持仓 CSV、图表）会写入 `config.py` 中 `OUTPUT_DIR` 指定的目录，默认为 `output_long_4400/`。

---

## 项目结构

```
eq-backtest/
├── run.py              # 回测入口，串联所有步骤
├── config.py           # 全部可调参数（唯一配置文件）
├── data_loader.py      # 数据加载、预处理、pool 构建与缓存
├── portfolio.py        # 组合模拟（Numba JIT 核心 + 持仓物化）
├── metrics.py          # 绩效指标计算、持仓周期统计
├── plotting.py         # 结果可视化（累积收益、仓位、指标表）
├── data/
│   ├── daily.pqt               # 日频行情（symbol × date，long-form）
│   ├── preds_size/             # 因子预测文件（宽表 parquet，date × symbol）
│   │   ├── *_3d.parquet
│   │   ├── *_5d.parquet
│   │   └── *_10d.parquet
│   ├── bm_open/
│   │   └── ret_csi_1000.csv    # 基准日收益率
│   └── .cache/                 # pool 预处理缓存（自动管理，可删除重建）
└── output_long_4400/           # 回测输出（由 OUTPUT_DIR 决定）
    ├── metrics_*.csv
    ├── cumrets_*.csv
    ├── returns_*.csv
    ├── positions_*.csv
    └── plots_long/
```

---

## 回测参数配置

所有参数集中在 `config.py`，修改该文件后直接重新运行 `run.py` 即可。

### 时间与数据路径

| 参数 | 默认值 | 说明 |
|---|---|---|
| `START` | `"2019-01-01"` | 回测起始日期 |
| `DATA_PATH` | `data/daily.pqt` | 日频行情文件路径 |
| `PREDS_DIR` | `data/preds_size` | 因子预测文件目录 |
| `BM_PATH` | `data/bm_open/ret_csi_1000.csv` | 基准收益率文件路径 |
| `BM_NAME` | `"csi_1000"` | 基准名称（用于输出文件命名） |

### 策略参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `HORIZONS` | `["3d","5d","10d"]` | 参与平均的预测期限后缀 |
| `UNIVERSE` | `["000300.XSHG","000905.XSHG","000852.XSHG"]` | 开仓股票池（指数成分要求） |
| `IS_SHORT` | `False` | `True`：做空排名靠后的股票；`False`：做多排名靠前的股票 |
| `ALLOW_ST_OPEN` | `True` | 是否允许对 ST 股票开仓 |
| `POOL_SIZE` | `4400` | 候选池大小（按预测排名取前 N） |
| `PORT_SIZES` | `[900]` | 实际持仓数量列表，支持多组并行回测 |
| `THRESH_OUT_BUFFER` | `500` | 退出缓冲（持仓滑出排名 `port_size + buffer` 才平仓） |
| `TRADE_ON_NEXT_DAY` | `True` | `True`：T 日信号在 T+1 日执行，首日只生成信号不建仓；`False`：同日信号、同日持仓 |
| `STRICT_FIRST_DAY_TOP_N` | `False` | `True`：首日只允许从严格 top N 信号窗口开仓；`False`：首日继续向后扫描直到尽量补满持仓 |
| `COST_PER_TURNOVER` | `0.00045` | 单边交易成本（每换手单位扣减） |
| `EXCLUDE_PERIOD` | `("2024-01-01","2024-03-31")` | 超额收益归零的异常区间，设为 `None` 关闭 |

### Pool 缓存

| 参数 | 默认值 | 说明 |
|---|---|---|
| `USE_POOL_CACHE` | `True` | 是否启用 pool 预处理磁盘缓存 |
| `POOL_CACHE_DIR` | `data/.cache` | 缓存存储目录 |

缓存以行情文件与预测文件的路径、大小、修改时间及关键配置参数为 key（SHA256），数据或参数变更后自动失效。手动删除 `data/.cache/` 可强制重建。

---

## 输出文件说明

| 文件 | 内容 |
|---|---|
| `metrics_<pool>_<bm>.csv` | 各组合规模的绩效指标汇总 |
| `cumrets_<pool>_<bm>.csv` | 日频累积超额收益率 |
| `returns_<pool>_<bm>.csv` | 日频超额收益率 |
| `positions_<size>.csv` | 每日持仓明细（date × symbol） |
| `plots_long/` | 累积收益、仓位热图、持仓周期分布等图表 |
