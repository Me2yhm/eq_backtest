---
name: dev-workflow
description: "启动多阶段开发工作流：communicate→plan→implement→judge→summary。用于新功能开发、复杂重构、需要多轮迭代的任务。使用'开始工作流'、'启动dev-workflow'、'多阶段开发'等触发。"
argument-hint: "描述你要做的任务..."
user-invocable: true
---

# 多阶段开发工作流 (dev-workflow)

## 概述

这是一个带循环和分支的多阶段开发工作流，适用于复杂任务的分步推进。

## 工作流程

```
communicate → plan → implement → judge ⇄ plan → summary → over
                        ↑                    │
                        └────────────────────┘
                   (循环直到需求完成或质量达标)
```

## 各阶段说明

| 阶段 | 模型 | 职责 | 产出 | 流转条件 |
|------|------|------|------|----------|
| **communicate** | GLM-5.2 | 沟通需求，挖掘潜在需求，生成需求文档 | `requirements.md` | human 确认 ✅ |
| **plan** | GLM-5.2 | 制定实施计划，拆解为 checkpoints | `plans/*.md` | human 确认 ✅ |
| **implement** | DeepSeek | 按计划逐 checkpoint 执行编码+commit，生成验收报告 | `reports/*.md` + git commits | 自动 → judge |
| **judge** | GLM-5.2 | 评估验收报告和代码，分支判断下一阶段 | `judge-logs/*.md` | 自动 → plan 或 summary |
| **summary** | DeepSeek | 总结本轮修改，生成总结报告 | `summary.md` | 自动 → over |

## 使用方法

1. 在 Chat 中输入 `/dev-workflow` 或在 Agent 选择器中选择 `communicate`
2. 首先进入 **communicate** 阶段，与 Agent 充分沟通需求并确认文档目录
3. 按 Agent 引导逐阶段推进：
   - communicate → plan（需 human 确认）
   - plan → implement（需 human 确认）
   - implement → judge（自动）
   - judge → plan 或 summary（自动）
   - summary → over（自动）

## 阶段 Agent 列表

- `communicate` — 需求沟通 Agent（GLM-5.2）
- `plan` — 计划制定 Agent（GLM-5.2）
- `implement` — 代码实现 Agent（DeepSeek V4 Pro）
- `judge` — 评审判断 Agent（GLM-5.2）
- `summary` — 总结报告 Agent（DeepSeek V4 Pro）
