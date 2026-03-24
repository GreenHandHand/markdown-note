# Agent Generation 的研究问题

## 文件来源
- `~/Project/research/IDEA_REPORT_AGENT_GENERATION.md`

## 主题概述
这个主题关注的是：

**LLM agents 如何自动生成其他 agent，或自动生成 agent 的组成部分。**

它和 long-horizon planning 不是同一个主题，更像是另一条平行研究线。

## 当前材料总结
这份 idea report 主要提出了 4 个值得优先考虑的方向：
- Static Evaluators for Generated Agents
- Interface-First Agent Composition
- Minimal Specification Thresholds
- Negative Results on Over-Generation in Multi-Agent Design

## 我对这个主题的判断
这个方向更偏向：
- agent design automation
- agent architecture evaluation
- specification-to-agent generation

和之前的 long-horizon planning 主线不同，它更关注：
- agent 是怎么被生成出来的
- 什么样的 specification 足够生成一个有用的 agent
- 复杂 multi-agent architecture 是否真的必要

## 推荐定位
这个主题适合单独维护，不建议混进“长时程规划智能体”目录里。
