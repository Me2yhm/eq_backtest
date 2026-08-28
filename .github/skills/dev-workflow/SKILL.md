---
name: dev-workflow
description: "启动多阶段开发工作流：communicate→plan→coordinator(implement→judge→summary)。用于新功能开发、复杂重构、需要多轮迭代的任务。使用'开始工作流'、'启动dev-workflow'、'多阶段开发'等触发。"
argument-hint: "描述你要做的任务..."
user-invocable: true
---

# 多阶段开发工作流 (dev-workflow)

## 概述

这是一个带循环和分支的多阶段开发工作流，适用于复杂任务的分步推进。

工作流采用**混合路线**架构：
- **communicate / plan**：独立 agent，通过 handoffs 按钮半自动切换（human 确认点）
- **implement / judge / summary**：subagent，由 coordinator 全自动编排（无需人工切换 agent picker）

## 工作流程

```
communicate → plan → coordinator → (implement → judge) ⇄ plan → summary → over
    ✅           ✅         🤖            🤖      🤖                      🤖
  (人工确认)  (人工确认)  (自动编排)   (自动执行) (自动评审)            (自动总结)
                                   ↑                    │
                                   └────────────────────┘
                              (循环直到需求完成或质量达标)
```

## 各阶段说明

| 阶段 | 模型 | 类型 | 职责 | 产出 | 流转条件 |
|------|------|------|------|------|----------|
| **communicate** | GLM-5.2 | 独立 agent | 沟通需求，挖掘潜在需求，生成需求文档 | `requirements.md` + `.state.json` 初始化 | human 点 handoff 按钮 ✅ |
| **plan** | GLM-5.2 | 独立 agent | 制定实施计划，拆解为 checkpoints | `plans/*.md` | human 点 handoff 按钮 ✅ |
| **coordinator** | DeepSeek（可配置） | 独立 agent | 自动编排 implement→judge→(plan\|summary) | 更新 `.state.json` | 自动 → implement |
| **implement** | DeepSeek | subagent | 按计划逐 checkpoint 执行编码+commit，生成验收报告 | `reports/*.md` + git commits | 自动 → judge |
| **judge** | GLM-5.2 | subagent | 评估验收报告和代码，输出 NEXT_STAGE 标记 | `judge-logs/*.md` + `<!-- NEXT_STAGE: xxx -->` | 自动 → coordinator 解析 |
| **summary** | DeepSeek | subagent | 总结本轮修改，生成总结报告 | `summary.md` | 自动 → over |
| **researcher** | DeepSeek | subagent | 代码探索类体量型任务（定位/摘要/执行），返回结构化报告 | 结构化报告（返回给 communicate/plan） | 被 communicate/plan 调用 |

## 使用方法

1. 在 Chat 中输入 `/dev-workflow` 或在 Agent 选择器中选择 `communicate`
2. 首先进入 **communicate** 阶段，与 Agent 充分沟通需求并确认文档目录
3. 按 Agent 引导逐阶段推进：
   - communicate → plan：human 点 handoff 按钮确认需求
   - plan → coordinator：human 点 handoff 按钮确认计划
   - coordinator → implement → judge → (plan|summary)：**全自动**，无需人工切换
   - 若 judge 分支到 plan（返工/有新需求），coordinator 停止自动循环，由 human 重新进入 plan
   - summary → over：自动结束

## 阶段 Agent 列表

- `communicate` — 需求沟通 Agent（GLM-5.2，独立 agent，可调用 researcher）
- `plan` — 计划制定 Agent（GLM-5.2，独立 agent，可调用 researcher）
- `coordinator` — 工作流编排 Agent（DeepSeek 可配置，独立 agent）
- `implement` — 代码实现 Agent（DeepSeek V4 Pro，subagent）
- `judge` — 评审判断 Agent（GLM-5.2，subagent）
- `summary` — 总结报告 Agent（DeepSeek V4 Pro，subagent）
- `researcher` — 代码探索 Agent（DeepSeek V4 Pro，subagent，供 communicate/plan 调用）

## Researcher Subagent 调用关系

communicate 和 plan 阶段在需要代码探索时，通过 `agent` 工具调用 researcher subagent（DeepSeek），将体量型任务（代码定位、文件摘要、日志分析、命令执行）卸载到低成本模型。researcher 返回结构化报告，GLM 基于报告做决策，保留直读权作为兜底。

```
communicate ──┐
               ├──→ researcher（DeepSeek）──→ 结构化报告 ──→ 返回 GLM
plan ──────────┘
```

**模型策略说明**：researcher 用 DeepSeek V4 Pro（低成本），承担代码探索类体量型任务，优化 communicate/plan 阶段 token 成本（预计节省 40-60%）。GLM 聚焦高价值推理决策，简单任务（单文件 < 100 行、决策关键代码）保留直读权，避免过度调用 subagent 增加延迟。

## 文件保存规则

每轮工作流使用独立子目录（子目录名由 communicate 在第一轮确定，英文 kebab-case），所有文档存放在子目录下。

| 文件 | 规则 |
|------|------|
| `.state.json` | 位于 `docs/workflow/.state.json`（根目录，唯一可覆盖文件），`doc_dir` 字段指向当前子目录 |
| `requirements.md` | 子目录内固定名，不再时间戳备份 |
| `plans/plan-XX-xxx.md` | 子目录内按编号保存，不覆盖 |
| `reports/acceptance-XX.md` | 子目录内按编号保存，不覆盖 |
| `judge-logs/judge-XX.md` | 子目录内按编号保存，不覆盖 |
| `summary.md` | 子目录内固定名，不再时间戳备份 |

> **注意**：历史文档（`docs/workflow/` 根目录下平铺的旧文档）不迁移，只对新工作流生效。
