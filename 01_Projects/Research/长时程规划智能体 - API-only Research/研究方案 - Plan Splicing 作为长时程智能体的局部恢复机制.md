# 研究方案 - Plan Splicing 作为长时程智能体的局部恢复机制

## 文件来源
- `~/Project/research/refine-logs/FINAL_PROPOSAL.md`

## 问题锚点
长时程 LLM agent 在执行多步任务时，只要中间某一步失败，常见做法就是 **full replanning**：从当前状态重新生成剩余计划。

但这种做法有两个明显问题：
- 成本高
- 容易造成 **semantic drift**，让新的 plan 偏离原本的高层目标

因此更具体的问题变成：

**失败后能不能只修补最小的失效子计划，而不是重写整个剩余计划？**

## 方法核心
这个 proposal 提出的核心方法是：

### Dependency-Scoped Plan Splicing
先把整个 plan 表示成一个带依赖关系的结构图：
- node：subgoal / action
- edge：produces / requires / order-before

当某一步失败时：
1. 检测失败
2. 用 failure analyzer 判断失败类型与受影响区域
3. 顺着 dependency graph 找到最小 invalid subgraph
4. 选择 splice boundary
5. 只对这个局部区域重新生成 subplan
6. 验证 repaired subplan 是否与边界接口兼容
7. 如果验证通过，就接回原来的 plan 继续执行

## 核心设计原则
**Minimal by default, expandable by evidence**

也就是：
- 默认优先修最小区域
- 只有在 contract check 失败时，才扩大 repair region

## 这项方法的价值
它试图同时优化几件事：
- 保留 global intent
- 降低 recovery token cost
- 提高 artifact reuse
- 减少不必要的 plan 改动

## 与已有工作的差异
这个方法最关键的差异点在于三件事的组合：
1. 显式的 dependency graph 表示
2. failure-induced invalid subgraph localization
3. boundary-constrained local repair generation

所以它不是普通的 self-critique，也不是简单 retry，而是一种更结构化的局部恢复机制。

## 我的判断
这已经不是“一个模糊 idea”，而是一个很像论文方法章节雏形的 proposal。

它现在最强的地方在于：
- 问题定义非常清楚
- baseline 清楚
- 评价指标容易设计
- 即便负结果也有研究价值
