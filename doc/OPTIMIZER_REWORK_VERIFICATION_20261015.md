# Optimizer Review 返工验收记录

日期：2026-10-15（Asia/Shanghai）。**R1–R4 修复及本地验收完成；R5 的 optimizer CI 通过，回测 CI 保留明确的授权限制。** [optimizer PR #2](https://github.com/Me2yhm/optimizer/pull/2) 已建立，独立代码审查无阻塞项，等待人工审查与合并授权。`eq_backtest` 保留 `5min-old`，未合并到 `main`。用户已选择本轮先记录回测 CI 授权限制。

本记录及[机器结果](OPTIMIZER_REWORK_RESULTS_20261015.json)仅使用本轮新实验。初次验收记录保留为历史，不用于证明本轮新场景通过。

## 基线、分支与安装来源

| 仓库 | 起始 HEAD | 本轮位置及验收运行时 HEAD |
| --- | --- | --- |
| optimizer | `988a1381f0030d93870c36cacddf71efc1815d3f` | `fix/shm-arena-contract-hardening`，`f80f5b7bdd2841ecadf064688b842ab66f5031d2` |
| eq_backtest | `4d131d6073187bdee3f69e94e0303f3af02b2e05` | `5min-old`，`62557430afb9003a53c6a3185d7b0746c7341843`；之后仅添加本报告和 README |

开始时工作区干净；fetch/push 均使用 `github`。新分支从最新 `github/master` 创建，旧 feature 分支未继续开发。`pyproject.toml`、`uv.lock` 的 requested revision 与解析 SHA 完全相同，均为 `f80f5b7bdd2841ecadf064688b842ab66f5031d2`。

所有最终回测环境均从 SSH Git 安装到 `site-packages`；机器结果包含每个环境的 `direct_url.json` 和导入路径。中途本地 checkout 安装仅用于开发，已被 Git 安装替换。`371bc69` 的本地全量测试与短期/容量原始结果保留：该提交与最终 SHA 的全部 optimizer 模块、包声明、测试和安装检查脚本逐字节相同，末次提交只修正 CI workflow；最终 SHA 又执行了完整 frozen 回归与 2,000 日压力测试。

主机 AMD EPYC 7T83，128 逻辑 CPU、约 251.5 GiB 内存；Linux `4.18.0-348.7.1.el8_5.x86_64`，uv `0.11.3`。主环境 Python 3.14.3，回测 NumPy 2.4.4、Numba 0.65.0、Polars 1.39.3。Python 3.10 使用 pandas 2.3.3，其余使用 pandas 3.0.1。AF_UNIX、SCM_RIGHTS、memfd 和 libc fallback 均实际执行。

## 阶段 A：修复前证据

基线 optimizer 26 项、回测启用 IPC 35 项均通过。新增专项先运行原实现并保留红灯：

- **R1**：现金只够买一个新目标，T 日目标始终为 A/B。仅颠倒 T+1 预测，原版实际持仓由 A/C 变成 B/C，收益由 +5% 变为 −5%；外部权重和本地目标两条路径均复现。日志 `/tmp/eq-rework-r1-red.log`，before/after 数值已嵌入机器结果。
- **R2**：客户端和独立服务均限制 `RLIMIT_NOFILE=256`，高频采集两端；原始逐日保留租约每输出增加约 2 FD、1 mmap。N=1,000/2,000 均在已完成 124 输出后失败，不能称其已经完成长周期模拟。
- **R3**：目标构造和模拟 finally 两处均能用释放异常覆盖原异常；原版长 CLI 也实际记录了 `failed to release 123 optimizer leases`，遮盖请求故障。日志 `/tmp/eq-rework-r3-red.log`。
- **R4**：Core、in-process、SHM 的四个政策布尔字段共 108 个非 bool 子场景未按要求拒绝，合法 bool 通过。日志 `/tmp/optimizer-rework-r4-red.log`。

| 原版请求日数 N | 实际完成数 | 客户端 / 服务端 FD 峰值 | 客户端 / 服务端输出 mmap 峰值 |
| --- | --- | --- | --- |
| 10 | 10 | 27 / 27 | 10 / 10 |
| 100 | 100 | 207 / 207 | 100 / 100 |
| 1000 | 124 | 255 / 255 | 124 / 124 |
| 2000 | 124 | 255 / 255 | 124 / 124 |

原版每次连接基线约 6 FD；失败后收尾 FD 回到基线、输出映射归零，说明本轮解决的是运行中资源峰值与容量界限，不能用退出时无泄漏替代峰值验证。

## 修复与专项验收

**R1**：多头现金竞争排序改用 `signal_row_by_symbol` 对应的信号行预测，执行行提供行情与可交易性。七项因果性测试覆盖反转、单个/全部 NaN、缺预测、同分、执行涨跌停、外部权重、5min/15min next-bar 与 same-bar、short 路径。修后执行日预测变动不再改变已锁定目标的实际买入、权重和收益。目标与实际持仓、T+1、share accounting 及成本口径保留。

**R2**：[Arena API 与状态机](https://github.com/Me2yhm/optimizer/blob/f80f5b7bdd2841ecadf064688b842ab66f5031d2/docs/OUTPUT_ARENA.md) 在服务端实现前提交。两个客户端提供 `optimize_many()`；in-process 通过 Core `out=` 写一块 NumPy 输出，SHM 显式协商 `output_arena/1` 后逐请求写同一个 memfd。只有全部区间和请求成功，FINALIZE 才发布一个 FD。Adapter 只持有一个批量租约与每日只读切片；Numba 消费后再统一 CONSUMED/RELEASE。

Schema 1.1 请求和原 `shm/1` 单请求语义保留，Arena 响应使用独立的可选扩展 schema。保留的 `988a138` 安装客户端已在仓库外直接连接最终服务，单请求计算、CONSUMED/RELEASE 及重复 release 均通过，未升级客户端。测试另覆盖单标的/5,000 标的、M/N 预算、空批次、非 list 输入、重复 ID、对齐/重叠/越界、假布尔 generation、错误 session/epoch、未完成/错误 finalize、模型故障、超时、客户端退出、服务关闭、重复释放及复用 FD。部分输出不发布、不重试、不切换 backend；transport 到 NumPy/Numba 消费边界不复制数值权重，审计标量及报表物化另计。有效 NumPy/Numba 视图在模拟期间保持映射，活导出引用阻止提前关闭。

**R3**：主请求/模拟异常保留原类型、对象、信息和 traceback，释放错误独立写 `cleanup_errors` / `cleanup_status`。无主错误时释放失败使 run 失败。真实 SHM 四格矩阵覆盖成功/失败主路径与成功/失败清理；另覆盖已发布单请求输出后的第二请求真超时、旧租约释放失败、实际 Numba 抛错，以及最终 DataFrame 和 audit 物化失败。Review 补修了 try 外物化绕过清理和 cleanup-only 重复诊断两处遗漏，均有红转绿证据。仍有导出视图时保留明确重试入口，远端会话已放弃时不重复发送 RELEASE。

**R4**：四个政策布尔字段使用严格 bool 校验；0/1、浮点、字符串、空值和容器均以 `INVALID_TYPE`、准确字段路径拒绝，两个 backend 映射为 `invalid_request`，合法输入数学结果一致。

**跨版本补修**：Python 3.10/3.11 上，`locals()` 快照会额外持有共享视图，导致坏数值输出的原校验错误被 BufferError 替代；现用显式初始化/清空引用并测试 NaN 与错误预算在单请求、批量两条路径均无泄漏。Python 3.10 的报表 join 发现 ns/us 精度不一致，只在报表边界对齐时间单位，专测输出一致，不改变交易时间线。

## 最终测试、安装与 CI

| 本地 Python | 回测 NumPy / pandas | optimizer 完整测试 | 回测完整测试（IPC 启用） | wheel 仓库外安装 |
| --- | --- | --- | --- | --- |
| 3.10.21 | 2.2.6 / 2.3.3 | 56 | 50 | 通过 |
| 3.11.9 | 2.4.4 / 3.0.1 | 56 | 50 | 通过 |
| 3.14.3 | 2.4.4 / 3.0.1 | 56 | 50 | 通过 |

所有单元测试无 skip；每个回测完整套件均真实执行短期双模式与 2,000 有效调仓日低 FD 测试。各版本 fresh/frozen 安装结果、依赖版本和日志路径在机器结果中。最终 wheel/sdist 构建通过，两个 archive 内 17 个 optimizer Python 模块与 checkout 逐字节一致，SHA256 已保存。安装检查从 `/tmp` 用 `python -I` 运行，覆盖 Core、双后端单请求和批量、只读切片、同 buffer ID、offset、copy=0、控制台入口与服务退出。

[最终 optimizer CI](https://github.com/Me2yhm/optimizer/actions/runs/38693488208) 三个 Linux job 全绿：**真实 Python 3.10.22、3.11.17、3.14.8**，各 56 项及 wheel 检查通过。先前 workflow 被 `.python-version` 重选解释器的问题已修复，现直接用 matrix 环境并断言实际版本，之前的绿色结果不作为跨版本证据。

[回测 CI 实际失败记录](https://github.com/Me2yhm/eq_backtest/actions/runs/38692763019) 显示缺少私有 optimizer 的 `OPTIMIZER_READ_TOKEN`，测试步骤未执行。用户明确选择本轮先记录限制。Workflow 要求仅对 optimizer 有 Contents read 权限的 token，临时配置权限 0600、限定 repo、finally 删除；未写入任何私钥/token。此项不能标为 GitHub 回测测试已通过，本地三版本结果独立有效。

compileall、`git diff --check`、所改 Python 文件的 Ruff correctness/unused-import 检查通过。仓库没有把整体格式检查新增为验收前提。

## 最终真实 CLI 与资源峰值

短期场景 12 日、4 标的、容量 1/2/4/8：20 个输出文件、44 个事件、81 行决策一致；权重、日收益、现金差异均为 0。固定种子 20261015，包含 M<N、NO_SIGNAL、LIQUIDATE、涨停阻碍与目标/持仓分离。保留目录 `/tmp/eq-rework-short-371bc69`；最终 f80f5b7 的全套回归再次执行同样场景。

长周期最终 SHA 场景：2,043 市场日、**2,000 有效 TARGET 调仓日**，另有 21 NO_SIGNAL、21 LIQUIDATE，7,937 行决策。双后端真实 run.py，8 个 CSV/Parquet 和逐日业务审计比较通过，标的/日期/事件严格一致，最大权重、日收益、现金误差均为 0，优于 1e-10/1e-8 门槛。涨停阻买和跌停阻卖断言实际触发。

| 后端 | 客户端 / 服务端 FD 峰值 | 输出 mmap 峰值 | RSS 峰值 KiB（客户端 / 服务端） |
| --- | --- | --- | --- |
| inprocess | 11 / 6 | 0 / 0 | 567972 / 37472 |
| shared_memory | 12 / 9 | 1 / 1 | 569248 / 38596 |

两端 `RLIMIT_NOFILE=256`，每 5ms 采样，FD 相对基线增量均小于 16；单个组合只有一个输出 memfd、双方各一个映射。客户端最终 FD 与 in-process 相同；两端最终 optimizer FD/mmap 为 0，服务正常退出、无 socket 残留。共享区逻辑输出 63,496 字节，页面映射计 65,536 字节。RSS 包含解释器、Numba、Polars 和报表对象；只主张 FD/mmap 数量有界，不主张完整回测历史内存恒定。

容量负测允许 64 字节，实际需求 63,496 字节：`complete=false`，错误明确包含 required/allowed，cleanup 完成，无 metrics、无共享输出映射、无 socket 残留。目录 `/tmp/eq-rework-capacity-371bc69`。原始失败产物没有伪装成成功结果。

## 性能复测

相同机器、回测 Python 3.14.3 / NumPy 2.4.4，每后端每规模先一次预热，再计 30 次。为保持与旧报告可比，测量原单请求 API，call 包含响应/映射/数值验证/CONSUMED/RELEASE，排除请求构造；Arena 的长周期行为由上面的真实 CLI 另行验证。单位 ms，列内为 P50 / P95。

| 后端 | 资产数 | wait | model | call | 首次 call（排除于统计） |
| --- | --- | --- | --- | --- | --- |
| inprocess | 1 | 0.0354 / 0.0440 | 0.0130 / 0.0155 | 0.1402 / 0.1664 | 16.5844 |
| inprocess | 800 | 0.1345 / 0.1449 | 0.0152 / 0.0172 | 0.2585 / 0.2730 | 0.3103 |
| inprocess | 5000 | 0.5383 / 0.5645 | 0.0173 / 0.0192 | 0.7101 / 0.7388 | 0.9010 |
| shared_memory | 1 | 0.2325 / 0.2572 | 0.0225 / 0.0274 | 0.5709 / 0.6325 | 1.0698 |
| shared_memory | 800 | 0.5892 / 0.5989 | 0.0260 / 0.0267 | 0.9668 / 0.9791 | 1.0239 |
| shared_memory | 5000 | 2.6136 / 2.8023 | 0.0378 / 0.0402 | 3.1943 / 3.3953 | 3.8611 |

5,000 资产的旧 call P95：in-process 0.7434ms、SHM 3.2552ms；本轮分别 0.7388ms、3.3953ms。完整六组旧值与新值均保留在机器结果，不将小幅波动解释成优化收益。copy_bytes 全为 0、FD 增量 0。

新进程 import 99.690ms，独立服务启动到 socket 可用 120.134ms；连接建立 in-process 0.025ms、SHM 1.644ms。首次 call 包含验证脚本首次 NumPy testing 初始化等开销，不能等同于纯模型耗时。OS 页缓存没有清空，optimizer 基准不含 Numba 首次编译；长 CLI 使用 warm Numba 缓存，不作为生产性能承诺。

## 复现、审查与交付边界

在对应仓库根目录执行，`--work-dir` 必须为新目录：

```bash
# optimizer：当前 hardening 分支
uv sync --frozen
.venv/bin/python -m unittest discover -s tests -v
uv build

# eq_backtest：5min-old
uv sync --frozen
RUN_OPTIMIZER_IPC_TESTS=1 uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/verify_optimizer_modes.py --work-dir /tmp/eq-short-new
uv run --frozen python scripts/verify_optimizer_pressure.py long-cli --work-dir /tmp/eq-long-new --nofile 256
# 预期退出1，检查 failed / complete=false 及 required/allowed
uv run --frozen python scripts/verify_optimizer_pressure.py long-cli --work-dir /tmp/eq-cap-new --max-shared-bytes 64
git diff --check
```

多版本手动运行必须用对应虚拟环境的绝对解释器路径，或给 uv 的每次 run 显式指定 `--python`，避免项目 `.python-version` 重新选择解释器。原始基线 curve、短期与长周期 comparison、run status、calls 和日志均在上述 /tmp 新目录；关键数据完整汇总进本仓库机器结果，不提交合成 parquet、runs、缓存和图表。

独立代码审查覆盖两个仓库完整变更，并增量复核到 optimizer `f80f5b7bdd2841ecadf064688b842ab66f5031d2`、回测 `62557430afb9003a53c6a3185d7b0746c7341843`；发现的问题已修，无剩余阻塞缺陷。GitHub 人工审批尚待用户，PR 未合并；不会自行合并回测到 main。若后续 squash 合并 optimizer，须将回测依赖重新固定到新 master SHA，再验证 frozen 安装与相关回归。

保留边界：本轮只验证合成数据，未跑生产市场数据；一个 Arena 必须容纳全部输出，不支持超容量分块续跑；旧单请求 API 的逐结果租约成本保留，长周期消费者应使用批量 API。回退时须一起回退 adapter 和 Git 依赖，单独把依赖改回 988a138 会缺少批量接口。回测 GitHub CI 私有依赖授权是本轮用户接受的唯一环境覆盖缺口。
