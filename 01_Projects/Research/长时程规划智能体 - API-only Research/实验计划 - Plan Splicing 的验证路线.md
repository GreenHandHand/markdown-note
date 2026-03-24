# 实验计划 - Plan Splicing 的验证路线

## 文件来源
- `~/Project/research/refine-logs/EXPERIMENT_PLAN.md`

## 总体目标
这份实验计划的作用，是把 `Plan Splicing` 从方法 proposal 进一步推进到可执行的验证路线。

它的核心目标是验证：

**Dependency-scoped local repair 是否比 full replanning 更高效、更稳定，并且更能保留原始计划语义。**

## 实验结构
整个计划被拆成 4 个 phase：

### Phase 0: Task Suite Construction
先构建适合实验的任务集合，要求：
- 6-12 步任务
- 有明确中间 artifact
- 能自动检测 success / failure
- 可以人工注入 failure

### Phase 1: Pilot
在 10 个任务上先做小规模测试，对比：
- Full Replan
- Plan Splicing

目标是先看有没有初步正信号。

### Phase 2: Main Experiment
扩大到 30-50 个任务，并加入更多 baseline：
- Full Replan
- Retry Only
- Suffix Replan
- Plan Splicing
- （可选）Fixed-Window Repair

### Phase 3: Ablation
如果主实验有效，再测试：
- dependency tracing 是否必要
- splice size 多大最合适
- contract-constrained repair 是否有价值
- failure analyzer 是否重要
- fallback policy 如何设计更稳

### Phase 4: Generalization
最后换一个任务域，验证方法是否具有跨域泛化能力。

## 关键指标
这份计划的一个优点是指标设计很完整，覆盖了：

### 成功率
- Final Success Rate
- Recovery Success Rate

### 效率
- Recovery Token Cost
- Recovery Step Count
- Planning Calls

### 语义保持 / 结构稳定性
- Plan Preservation Rate
- Edit Span Size
- Artifact Reuse Rate

### 定位质量
- Localization F1

## 我的判断
这份实验计划说明当前工作已经进入很实在的阶段：
- 不再只是 idea 层面讨论“值得不值得做”
- 而是已经在明确“怎么验证、跟谁比、看哪些指标”

尤其好的地方在于：
- baseline 很清楚
- phase 划分合理
- 成本控制明确
- 全程 API-only，无需训练资源
