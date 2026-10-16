# Optimizer 会话寿命加固与合并准备验收

**报告日期：2026-10-16（Asia/Shanghai）。** 专项长空闲与压力证据生成于 2026-10-15；本次恢复后复用原记录，完成 CI、字段说明、最终三版本回归及交付复核。

**结论：代码可提交审阅。回测 GitHub 集成测试仍因已知私有依赖授权缺口未执行，不具备“全部远端门禁通过”的结论。两个 PR 均待审，未合并或删除分支。**

机器记录：[OPTIMIZER_SESSION_HARDENING_RESULTS_20261016.json](OPTIMIZER_SESSION_HARDENING_RESULTS_20261016.json)。其中保留源文件 SHA256、日志摘要与哈希、版本、时序、逐会话消息数量和资源采样。原始运行数据留在 `/tmp`，不提交 Parquet、图表、缓存或完整运行日志。

## 版本与边界

| 项目 | 基线 | 本轮交付 |
| --- | --- | --- |
| optimizer | `f80f5b7bdd2841ecadf064688b842ab66f5031d2` | `3ca073335b40d54e7396d258c42470dec96a4c42`；`fix/shm-arena-contract-hardening`；[PR #2 → master](https://github.com/Me2yhm/optimizer/pull/2) |
| eq_backtest | `f2b072a0c45004e6c54c8e2b33fd884f68e72c01` | 代码验收提交 `58483b1e026c93dbcf711ea52990ad96bf2d5df0`；`fix/optimizer-session-lifecycle`；[PR #1 → 5min-old](https://github.com/Me2yhm/eq_backtest/pull/1)；随后仅追加本验收报告 |

所有远端操作使用 `github`。optimizer/master 保持 `988a1381f0030d93870c36cacddf71efc1815d3f`，回测/5min-old 保持基线，未向 main 合入其它变化。`pyproject.toml`、`uv.lock` 和三版本实际 `site-packages/direct_url.json` 均指向 optimizer 的完整 `3ca0733…` SHA。

本轮提交顺序：optimizer 红灯 `d2e414f` → 服务寿命修复 `3ca0733`；回测红灯 `e734ad8` → 固定依赖 `c8d65d0` → 逐组合会话 `5a09cd9` → 慢路径工具 `453313e` → CI `fc09130` → 字段说明/测试 `58483b1`。慢路径执行时的 `run.py` 与最终版本 AST 相同，仅签名排版有变化。工具最后补了服务退出码断言，并对保存的两个真实默认服务案例重新执行全部生命周期断言通过，没有重跑 31 秒或 2,000 日案例。

原 R1–R4 报告 [OPTIMIZER_REWORK_VERIFICATION_20261015.md](OPTIMIZER_REWORK_VERIFICATION_20261015.md) 及其机器记录保持原样。

## S0–S3：基线、红灯与修复

基线完整回归：optimizer 56 项、回测 50 项，均通过、无 skip。环境基线及日志哈希在机器记录中；本地 Linux 内核 `4.18.0-348.7.1.el8_5.x86_64`，uv `0.11.3`，普通进程 `RLIMIT_NOFILE=262144/262144`。专项服务与客户端均实际设置 soft limit 256。

红灯来自真实 Unix socket：

- optimizer：测试专用 socket 包装把旧固定 30 秒读超时缩至 150ms，等待 450ms 后，HELLO 后 HEALTH、持有 Arena 的 release、下一次单请求/批量请求共四条路径失败。日志 `/tmp/optimizer-session-red.log`。
- 回测：慢加载时已连接，四组合及报表共用同一会话，E1/E3 两项失败。有效日志 `/tmp/eq-session-red-uds.log`。先前 `/tmp/eq-session-red.log` 的 sandbox PermissionError 不计入缺陷复现。
- CI：修前事件矩阵对 main push 与 main 目标 PR 均失败，修后两项契约测试通过；缺令牌测试直接运行工作流的安装脚本，确认在安装前非零退出。

服务端现在区分两种等待：HELLO 前保留默认 30 秒的绝对期限，非法预握手消息不会延长期限；成功 HELLO 后使用阻塞读，不再因业务空闲关闭合法会话。EOF、进程退出、协议错误及 shutdown 继续负责结束与回收；shutdown 唤醒阻塞读取。客户端 `request_timeout_ms` 和模型 `timeout_ms` 未取消。无心跳、重连、重试、fallback 或新增权重复制。

回测先完成基准与股票池加载，再逐 `port_size` 创建 client。正常顺序为连接 → capacity 校验 → prepare → simulation → release 输出/lease → close → CSV/评估/绘图。容量及服务可用性错误改为在加载之后出现，仍记录 `failed / complete=false`。原模拟/请求错误在 client context 退出也失败时保留身份与 traceback，cleanup 独立记录；仅 cleanup 失败时整次运行失败。

专项快速测试：optimizer 8 项、回测生命周期 7 项通过。覆盖预握手期限、空闲后新请求、异常退出、shutdown、旧 lease 失效、真实请求超时、容量/工厂配置错误和主异常优先级；既有 disabled、日频/5min/15min、R1–R4 测试包含于最终完整回归。

## S4：真实慢路径、故障与数值回归

真实默认服务命令：

```bash
.venv/bin/python -m optimizer.transport.shm.service --socket <dedicated-run>/ipc/service.sock
MPLCONFIGDIR=/tmp/eq-session-mpl .venv/bin/python scripts/verify_optimizer_session_lifetime.py \
  --work-dir /tmp/eq-session-default31-20261015 --real-default --hold-seconds 31
```

脚本自行启动独立真实服务，实际调用 `run.run()`；包装器仅记录或延迟真实加载、模拟与绘图。重新执行必须使用新的工作目录。原始报告 `/tmp/eq-session-default31-20261015/comparison.json`。

慢加载 0.3 秒，加载结束时刻 0.317475 秒，首次连接完成 0.323570 秒；加载与全部报表/绘图采样点均无活跃会话。首个组合在 finalized Arena 上无控制消息等待 **31.001130 秒**，等待前后双方各有一个输出映射，结束时仍可直接读取只读权重。

下表时间相对该 worker 首条事件，单位秒；组合归属按独立 session 与 prepare 对应：

| port_size | session_id 前缀 | connected | released | closed | report_begin |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `402cfd7c` | 0.323570 | 31.648428 | 31.650076 | 31.650857 |
| 2 | `893c1feb` | 32.380111 | 32.427590 | 32.429299 | 32.430166 |
| 4 | `569cdb21` | 33.185587 | 33.235608 | 33.237409 | 33.238314 |
| 8 | `f9281eec` | 33.963798 | 34.012969 | 34.014738 | 34.015601 |

四个 session_id 不同，前一组合关闭后才进入下一组合。每组合真实发送 8 次 OPTIMIZE_INTO_ARENA、各一次 BEGIN/FINALIZE/CONSUMED/RELEASE；不存在向下一个会话释放旧 buffer 的情况。每个 plan 一个 lease，输出 `copy_bytes=0`。

12 日合成数据、容量 1/2/4/8：归一化审计中的随机身份、传输与耗时字段后，**20 个文件、44 个业务事件、81 行 decision_targets 一致**。覆盖 M<N、NO_SIGNAL、LIQUIDATE、涨跌停约束。权重、日收益、现金权重最大绝对误差均为 **0**；收益/持仓/目标/现金与换手统计均比较。

最后一个组合模拟后终止服务：运行非零退出，`failed / complete=false`，无 metrics 成功汇总，四个独立会话且无第五次连接。错误为 `OptimizerClientError: failed to release 1 optimizer lease(s)`；cleanup 保留底层 `optimizer control connection failed; session was abandoned`，`retryable=false`。该故障发生于 CONSUMED，随后放弃会话，不发送 RELEASE 或重放请求。此例没有更早模拟错误，故 release 错误是主错误；另有真实请求超时加 context 退出失败测试验证主异常优先级。

资源为每 5ms 采样的观测峰值，非连续监测上界：

| 案例 | 客户端 / 服务端 FD 峰值 | 输出 mmap 峰值 | 活客户端终值 optimizer FD/mmap | 活服务端终值 optimizer FD/mmap |
| --- | --- | --- | --- | --- |
| 31 秒、多组合成功 | 13 / 9（基线 6 / 6） | 1 / 1 | 0 / 0 | 0 / 0 |
| 最终组合断服务 | 14 / 9 | 1 / 1 | 0 / 0 | 服务已退出，终值不可采样 |
| 2,000 有效调仓日 | 12 / 9 | 1 / 1 | 0 / 0 | 0 / 0 |

故障服务退出后 `/proc/<pid>/fd` 返回 PermissionDenied，机器记录保留该事实；不能把它写作“测得服务终值为零”。两条慢路径服务退出码均为 0，socket 均移除；健康案例的服务端终值是在服务仍存活时实测。

2,000 日命令与原始记录：

```bash
.venv/bin/python scripts/verify_optimizer_pressure.py long-cli \
  --work-dir /tmp/eq-session-pressure-20261015 --output /tmp/eq-session-pressure-20261015.json
```

实际 2,043 个行情日，2,000 TARGET、21 NO_SIGNAL、21 LIQUIDATE，7,937 行 decision_targets；8 个数值文件及事件一致，权重/收益/现金误差均为 0。双方输出映射字节峰值 65,536（逻辑有效数据 63,496），FD 增量低于 16，最后无 optimizer FD/mmap 或 socket 残留。

## S5–S6：CI 与字段语义

工作流 push 与 pull_request 过滤器均为 `[main, 5min-old]`，PR 按目标 base 分支匹配，保留 workflow_dispatch、`contents: read`、固定 uv/Python 版本与 frozen 安装。事件矩阵明确检查 main/5min-old 正例及 feature/other 分支反例；缺令牌安装脚本必须失败，不能静默跳过而变绿。

- optimizer [本轮实际 GitHub CI](https://github.com/Me2yhm/optimizer/actions/runs/38726994944)：提交 `3ca0733…`，实际 Python **3.10.22 / 3.11.17 / 3.14.8**，各 64 项、wheel/sdist 构建和源码树外安装验证均通过。
- 回测 [实际 PR 触发记录](https://github.com/Me2yhm/eq_backtest/actions/runs/38836658164)：PR #1 目标 `5min-old`、代码提交 `58483b1…`；私有依赖安装明确失败：`CI dependency authorization missing … integration has not run.` 完整测试/CLI 步骤为 skipped，**集成未执行**。用户已接受本轮记录此限制，未创建或扩大凭据权限。
- main PR/push 的配置及事件匹配已验证；没有制造 main 提交或整分支合并来触发实际事件。main 的实际调度仍待后续获授权的变更。

README 已说明：`positions.pred` 是执行行原始预测，可包含执行日新值；`trade_on_next_bar=true` 时买入顺序使用 `signal_row_by_symbol` 对应 `signal_bar_idx` 的预测。例：信号日 A=4、B=3，执行日 A=-100、B=100，现金不足时仍优先买 A，A 的导出 pred 为 -100。新增一个测试覆盖 local/external 两条路径以及 `decision_as_of` / `planned_execution_at` 对齐；不改列名。

## S7：最终回归、安装与复核

| 本地实际 Python | optimizer | eq_backtest | frozen 安装 optimizer SHA |
| --- | --- | --- | --- |
| 3.10.21 | 64 项，14.684s | 60 项，58.559s | `3ca073335b40d54e7396d258c42470dec96a4c42` |
| 3.11.9 | 64 项，14.799s | 60 项，59.823s | 同上 |
| 3.14.3 | 64 项，14.641s | 60 项，52.811s | 同上 |

全部无 skip，Linux SHM 真实执行。回测使用新 NUMBA_CACHE_DIR 且并行运行，时间仅为功能回归耗时，包含冷缓存编译，不作生产性能比较。回测 NumPy 为 2.2.6/2.4.4/2.4.4，Numba 0.65.0、Polars 1.39.3；完整包版本、解释器路径、实际安装位置和来源见 JSON。

复现方式（用实际解释器，避免 `.python-version` 把矩阵重新指向 3.14）：

```bash
# optimizer 仓库；分别使用 3.10、3.11、3.14 环境的解释器
<python> -m unittest discover -s tests -v
uv build --out-dir /tmp/<new-dist-dir>

# 回测仓库；每个 Python 使用独立环境
UV_PROJECT_ENVIRONMENT=/tmp/<new-env> uv sync --frozen --python <python>
RUN_OPTIMIZER_IPC_TESTS=1 /tmp/<new-env>/bin/python -m unittest discover -s tests -v
```

wheel/sdist 的全部 **17 个包内 Python 模块**与最终源码逐字节一致，制品 SHA256 已记录；三版本均从 `/tmp` 使用 `python -I` 验证安装后的 Core、in-process、单请求 SHM、批量只读共享输出及 service 入口。wheel 的独立依赖安装与项目 frozen 环境分别记录，不混作同一组依赖版本。另用原样安装于 `988a138…` 的旧客户端连接 `3ca0733…` 新服务，单请求、只读零复制、CONSUMED/RELEASE、重复 release 和 HEALTH 均通过。

compileall、Ruff `E9,F63,F7,F82,F401`、两个仓库基线到最终版本及工作区的 `git diff --check` 均通过。差异检查未发现无关业务修改、凭据或运行产物。独立代理复核生产代码、CI 与字段说明，无阻断问题。一个非阻断观察：150ms 握手测试含 90ms sleep，在极高调度延迟下可能敏感；本轮本地及实际 CI 六组解释器均通过，未为此变更服务代码或依赖 pin。

## 后续合并条件

本轮 S0–S7 的代码、测试、文档与待审 PR 已交付；GitHub 私有依赖授权限制继续保留。下一步需要用户正式授权后先处理 optimizer 合并；若 squash，必须将回测重新固定到合并产生的 master SHA、更新锁文件、重新 frozen 安装并执行关键回归。之后再决定是否合并回测 PR。不能据本次结果整体合并含其它变化的 5min-old 到 main。

挂起但未断开 socket 的客户端仍可能占用服务资源，需要主动关闭及运维监测；本轮没有引入心跳或租约续期。验收使用合成数据，不代表生产数据性能验收。
