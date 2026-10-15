# Optimizer 双模式验收记录

日期：2026-10-15（Asia/Shanghai）。结论：**验收完成，已合并**。两个仓库的 Linux 回归、真实 CLI 双模式对比、资源异常测试和独立安装均通过。用户于 13:14 手动将 [optimizer PR #1](https://github.com/Me2yhm/optimizer/pull/1) squash 合并到 `master`；`eq_backtest/5min-old` 保留原分支。

## 代码与环境

执行前两个工作区均无未提交修改。远端通过 `git fetch github` 核对；本次提交和推送只使用 `github`。

| 仓库 | 分支 | 执行基线 | 验收范围 |
| --- | --- | --- | --- |
| optimizer | feature/dual-mode-architecture | b205edde54b45d415b84de815e34ee6f81bae03b | 相对 master 的完整双模式变更及新增异常修复 |
| eq_backtest | 5min-old | 6cd5f4d2d986efa0023da2b57efaf0133ff6bff1 | 优化器接入相关代码与本次修改 |

`eq_backtest` 的基线比 handoff 多出锁文件、ruff 开发依赖和导入整理三个提交，均已保留。初次验收的运行时依赖固定为 `optimizer@2cc03e2c60b04013b871757863b95c9c0b674c8c`，后续安装验证脚本和文档提交没有改变运行时代码。合并后，依赖和锁文件固定到主线提交 `988a1381f0030d93870c36cacddf71efc1815d3f`。

合并提交与已验收的特性分支 HEAD `9b2cf6a14f33efb0cc754f1dd179b890d93e3420` 的完整 Git tree 均为 `13d7f0a740a4cbc61bac010257253ed51c475faf`，文件内容完全一致。由于 squash 合并不保留旧特性提交作为主线祖先，消费者改为固定主线 SHA，以便不依赖特性分支历史。原始测试和性能数据仍对应本记录中的已验收代码。

测试机器为 AMD EPYC 7T83，64 核、128 逻辑 CPU，约 251.5 GiB 内存；Linux `4.18.0-348.7.1.el8_5.x86_64`、小端、uv `0.12.10`。主环境 Python `3.14.3`；optimizer 环境 NumPy `2.5.2`，回测环境 NumPy `2.4.4`、Numba `0.65.0`、Polars `1.39.3`、pandas `3.0.1`。Python 3.11 验证使用 `3.11.9` 和 NumPy `2.4.6`。AF_UNIX、SOCK_SEQPACKET、SCM_RIGHTS 和 memfd 均实际执行；当前 Python 3.14 没有 `os.memfd_create` 时，使用已有 libc 包装。

## 回归结果

| 检查 | 结果 |
| --- | --- |
| optimizer 原始修复提交回归 | 7 项通过，包含 handoff 中两处 SHM 失败 |
| optimizer 最终完整回归，Python 3.14 | 26 项通过，无 skip |
| optimizer 最终完整回归，Python 3.11 | 26 项通过，无 skip |
| eq_backtest 完整回归，启用 IPC | 35 项通过，无 skip |
| eq_backtest IPC 接入专项 | 11 项通过，服务来自安装包，无兄弟源码目录依赖 |
| wheel 和 sdist 构建 | 通过；全部 optimizer 模块与源码逐文件一致 |
| 仓库外 wheel 安装，Python 3.11 | Core、in-process、独立 SHM、控制台入口通过 |
| 初次全新环境 uv sync --frozen，Python 3.14 | 通过；SSH Git 安装固定到 2cc03e2 |
| 全新环境的仓库外安装验证 | 两种后端均通过，导入路径为 site-packages |
| 合并后固定 988a138 的完整回归，启用 IPC | 35 项通过，无 skip，包含真实 CLI 双模式对比 |
| 合并后全新 frozen 环境和仓库外安装验证 | 通过；固定主线 988a138，两种后端均可调用 |
| compileall 和 git diff --check | 通过 |

重跑命令分别在对应仓库根目录执行：

```bash
# optimizer
uv sync
uv run python -m unittest discover -s tests -v
uv build

# eq_backtest
uv sync --frozen
RUN_OPTIMIZER_IPC_TESTS=1 uv run python -m unittest discover -s tests -v
uv run python scripts/verify_optimizer_modes.py --work-dir /tmp/eq-dual-verification-new
```

独立安装检查使用 optimizer 的 `scripts/verify_installed_package.py`，以 `python -I` 从 `/tmp` 运行，确认导入不会落回源码。根目录旧入口在源码环境验证；wheel 的支持入口为 `optimizer`、`optimizer.transport.shm` 和 `optimizer-service`。

## 双模式真实回测对比

固定种子为 `20261015`，12 个交易日、4 个标的，持仓容量分别为 1、2、4、8。市场、预测、基准均为脚本新建的合成数据；未读取或修改真实行情。每个后端运行真实 `run.py`，使用独立运行目录。SHM 服务从已安装的包启动。

| 比较项 | 结果 |
| --- | --- |
| CSV 和 Parquet 文件 | 20 个文件通过逐列比较 |
| optimizer_calls.jsonl 业务事件 | 44 个，业务字段完全一致 |
| decision_targets.parquet | 81 行，标的、顺序、时间、预算一致 |
| 权重最大绝对差 | 0；门槛为 1e-10 |
| 日收益最大绝对差 | 0；门槛为 1e-8 |
| 现金权重最大绝对差 | 0；按 1 减持仓实际权重之和比较 |
| 收益、持仓、换手、清仓次数、最终指标 | 通过比较 |
| 成功状态 | 两种后端均 success，complete 为 true |
| 失败状态 | 同进程容量不足、SHM 无服务均失败，complete 为 false，无伪造成功指标 |

覆盖 `M=N`、`M<N`、单标的、多标的、相同预测分数、无信号、显式空目标清仓、涨停导致 target 与 held 分离。单标的部分投资日只持有一个标的，目标权重仍为 `1/N`。选择来自决策日信息，执行发生于下一交易日；已有因果性测试另行验证改变执行日开仓资格不会改变前一日选择。

随机 request_id 按组合和决策时点建立一一对应，不能直接跨运行比较 UUID。config_signature 的定义包含 protocol_version，脚本先按每个后端重算 SHA256 校验，再从业务等价比较中剔除。会话、epoch、缓冲区 ID、延迟及字节计数分别检查真实含义，未要求字节级相等。

最终留存目录为 `/tmp/eq-dual-acceptance-20261015-final`，包含两个成功回测、两个失败回测、全部配置与日志及 `comparison.json`。可重跑脚本为 [verify_optimizer_modes.py](../scripts/verify_optimizer_modes.py)，机器可读摘要见 [OPTIMIZER_DUAL_MODE_RESULTS_20261015.json](OPTIMIZER_DUAL_MODE_RESULTS_20261015.json)。

## 缺陷与资源安全修复

1. 锁文件仍指向 SHM 修复前的 `5ffb23e`，普通 `uv lock` 不更新已有 Git 分支解析。真实安装包回测复现了双响应问题。先显式升级，再将最终依赖和锁文件固定至 `2cc03e2`。
2. 测试直接调用当前解释器缺少的 `os.memfd_create`，导致响应线程异常后阻塞。测试改用包已有的 libc 兼容包装，并设置有限 socket 超时。
3. `np.ndarray(buffer=mmap)` 未可靠保留导出缓冲区引用，额外视图存在时仍可关闭映射。客户端改用 `np.frombuffer`，活跃 NumPy/Numba 视图会阻止关闭，保留零拷贝读取。
4. 远端释放失败后再次 release 可能关闭复用的 FD。租约在本地资源已关闭的失败路径进入终态，关闭会话；重复调用不再重放释放。活跃视图造成的本地关闭失败允许显式重试。
5. 异常展开丢弃租约可能遗留原始 FD。接收句柄由 Python 文件对象持有，销毁所有权对象时会关闭句柄；mmap 生命周期仍由导出视图维持。
6. 畸形 universe、控制 type、shape 或 buffer_id 曾导致会话线程退出。现在返回单个契约/控制错误；布尔 generation 和 shape 不能伪装成整数。
7. 客户端在超时、断连、错误身份、异常 ACK 时关闭会话。旧租约不能发送到重连后的会话；CONSUMED 和 RELEASE 确认必须匹配缓冲区 ID 与 generation。
8. 服务的发送动作移出模型异常处理，避免发送成功后再生成第二个错误响应。关闭服务会唤醒现有连接，清理缓冲区并移除 socket；结束的线程不无限保留。
9. 公共已解析请求会重新经过 Core 校验，不能绕过新的 max_assets、唯一标的、预算或超时限制。Core 保持不依赖 socket、memfd、会话和传输所有权。
10. 回测清理先移除 Numba 列表引用，再释放输出；仍在使用的租约保留以便显式清理。第二次请求注入失败时，首次输出正常释放、原错误保留、FD 无增长。

测试涵盖合法和异常 HELLO/HEALTH/CAPABILITIES/ALLOC/INPUT_READY/OPTIMIZE/CONSUMED/RELEASE、重复请求 ID、错误会话/epoch、能力政策缺失、模型异常、处理超时、客户端等待超时、断连、服务关闭、重复释放、活跃视图和异常 ACK。成功与失败响应后均检查无第二个陈旧报文。80 次连续请求释放及 20 次连接循环后 FD 不增长，已释放输出不再出现在进程 mmap 列表；畸形报文携带的 FD 也会关闭。

## 合成性能

每种后端、每种资产规模先预热 1 次，再测 30 次。下表为本机本次测量，单位毫秒；不是生产行情回测性能。该基准不运行 Numba，回测可能发生的首次编译不计入这些数值。

| 后端 | 资产数 | Wait P50 | Wait P95 | Model P50 | Model P95 | Call P50 | Call P95 | 共享字节 | 输出复制字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inprocess | 1 | 0.034 | 0.043 | 0.013 | 0.015 | 0.141 | 0.164 | 0 | 0 |
| inprocess | 800 | 0.136 | 0.142 | 0.015 | 0.017 | 0.264 | 0.271 | 0 | 0 |
| inprocess | 5000 | 0.533 | 0.565 | 0.017 | 0.021 | 0.706 | 0.743 | 0 | 0 |
| shared_memory | 1 | 0.227 | 0.251 | 0.023 | 0.024 | 0.557 | 0.608 | 8 | 0 |
| shared_memory | 800 | 0.580 | 0.591 | 0.025 | 0.026 | 0.939 | 0.951 | 6400 | 0 |
| shared_memory | 5000 | 2.588 | 2.746 | 0.037 | 0.039 | 3.158 | 3.255 | 40000 | 0 |

Wait 为客户端等待核心/控制响应的计时；SHM 映射校验另行记录。Call 包括 optimize、映射、数值校验和 CONSUMED/RELEASE；不包括构造请求。输出复制指标仅覆盖模型直接写入数值输出及消费者视图读取，不包含资产 ID JSON、审计标量物化或执行状态更新。所有规模释放前后客户端 FD 数相同。5,000 资产的 Call P95 在两种模式均满足小于 5,000 ms 的项目目标。

复现实验时先从安装包启动独立服务，再运行：

```bash
uv run python -m optimizer.transport.shm.service --socket /tmp/optimizer-benchmark.sock
# 另一个终端
uv run python scripts/benchmark_optimizer.py --mode both \
  --socket /tmp/optimizer-benchmark.sock --sizes 1 800 5000 --repeats 30
```

## 审查与合并边界

已审查 optimizer 相对 `github/master` 的全部变更；核心、模型注册、契约没有传输实现，服务只持有 IPC 和缓冲区生命周期。兼容转发模块沿用原名，Schema 1.1 和 equal_weight/1 的数学行为保持一致。eq_backtest 只审查优化器相关差异，本次没有改动行情归一化、交易状态机或收益计算；既有历史策略改动不纳入双模式合并范围。

无已知阻塞缺陷。保留的非阻塞边界为：未使用真实生产行情；未验证声明下限 Python 3.10；两仓库均无 GitHub Actions workflow。当前自动化证据覆盖 Linux Python 3.11 和 3.14，后续版本和生产工作负载应按实际部署补验。

optimizer 已由用户通过 PR #1 合并到 `master`，本地 `master` 已快进同步至 `github/master`；保留特性分支。eq_backtest 保持 `5min-old`，固定合并提交而不跟随移动分支。回滚时新建独立回测目录并设置 `optimizer.enabled: false`；若回退提交，应同时还原 pyproject.toml 和 uv.lock，不能在一个运行目录混用两种结果。
