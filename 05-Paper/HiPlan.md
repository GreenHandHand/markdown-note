# HiPlan: Hierarchical Planning for LLM-Based Agents with Adaptive Global-Local Guidance

arXiv-2025

**核心一句话**：这篇论文针对 LLM 代理在长时序复杂任务中容易迷失方向的问题，搞了个分层规划框架，通过里程碑全局指导和步步微调的局部提示，让代理更灵活适应环境变化。

---

## Key Contribution

- We introduce HIPLAN, a novel hierarchical planning framework that tightly integrates global milestone action guides with local step-wise hints, achieving adaptive global-local guidance for agent planning.
- We propose an efficient milestone-level experience reuse strategy that allows agents to draw on prior demonstrations in a way that is both generalizable and actionable.
- We conduct extensive experiments on multiple challenging benchmarks, demonstrating that HIPLAN significantly improves task success rates and robustness compared to strong baselines, confirming its effectiveness across diverse decision-making scenarios.

- 作者推出 HIPLAN 这个框架，本质上是把全局里程碑和局部提示绑在一起，目的是让代理既有大局观，又能实时纠错，这比单纯的高层规划或步步反应更靠谱，因为它解决了高低层脱节的痛点。
- 他们设计了里程碑级别的经验复用策略，不是全盘抄袭过去任务，而是挑中间粒度的片段重用，这样既避免了细节噪声，又保持了通用性，这点挺聪明，能让代理从专家演示中学到精华而不被琐碎拖累。
- 通过在 ALFWorld 和 WebShop 上的实验，证明 HIPLAN 在成功率和鲁棒性上碾压基线，这不光是数据好看，还验证了框架在不同决策场景下的普适性，我觉得这贡献的核心价值在于为 LLM 代理的长时序规划提供了可操作的模板。

---

## Method

### Milestone Library Construction (Offline Phase)

> [!question] 不合理的动机
> 作者建这个库是为了从专家演示中提炼可复用经验，避免代理从零开始规划长任务，但直觉上合理吗？还是有点强行，因为假设专家演示覆盖所有变体不太现实。

作者的直觉是，过去成功的任务轨迹太细太杂，直接重用容易噪声多，而整任务重用又太粗不灵活，所以挑里程碑这个中间层来存经验，就像存路标而不是全地图。

它这么工作：从演示集 D 中分割轨迹成片段 $ζ_k$，每个片段用 LLM 生成描述 $m_k$，然后嵌入成向量存进库 ML，每个条目是 $(v_task, v_milestone, m_k, ζ_k)$。

必须结合公式解释：$ML = ∪ (v^{(i)}_task, v^{(i,k)}_milestone, m^{(i)}_k, ζ^{(i)}_k)$，这里 $v_task$ 是任务τ的嵌入，$v_milestone$ 是里程碑 $m_k$ 的嵌入，$ζ_k$ 是对应轨迹片段，相似度用点积算，物理意义是向量空间中找语义近的经验来复用。

> [!tip] 高光时刻
> 这个中粒度复用是 Aha 点，通常大家要么全轨迹要么单步，这里卡在中间，平衡了泛化和细节。

> [!warning] 异议
> - 作者没解释为什么不选更细的动作级或更粗的任务级，有没有对比实验证明里程碑最好？
> - 反直觉的是，通常检索用全轨迹，为什么他们觉得片段更好？可能是因为长任务噪声大，但缺少证明。
> - 缺失了消融实验来验证库构建的有效性，比如不分割轨迹会怎样。

### Hierarchical Planning and Execution (Execution Phase)

> [!question] 不合理的动机
> 作者加这个是为了结合全局和局部，让代理不偏航，但问题是，如果环境变化太剧烈，预设里程碑会不会变成累赘？

作者的直觉是，全局里程碑像地图，给大方向；局部提示像 GPS，实时纠偏，这样代理不会只顾眼前或迷失整体。

它这么工作：先检索类似任务生成全局导图 $G_τ = [m1, ..., mK]$，然后每步用当前里程碑检索片段生成提示 $h_t$，包括状态上下文、里程碑差距和动作纠错，最后政策 $π$ 用这些输出动作 $a_t$。

必须结合公式解释：$G_τ = LLM(τ, {(τ^{(j)}, ξ^{(j)}, G_τ^{(j)})})$，这里 LLM 用检索的任务和里程碑生成导图；$h_t = LLM(m_k, {(o_s, a_s)}, {(m^*_l, ζ^*_l)})$，各项是当前里程碑、历史观测动作和检索片段，物理意义是桥接观测到目标的差距；$a_t = π(τ, {(o_s, a_s)}, m_k, h_t)$，整合全局局部决策。

![[Figure-2.png]]
*结合图片的一句话分析：注意看图中右边的执行阶段，它对应了检索里程碑生成提示的循环，正好可视化了自适应过程。*

![[Figure-1.png]]
*结合图片的一句话分析：注意看图底部的 HiPLAN，它对应了高低层结合，避免了左上全局僵化和右上局部短视。*

> [!note] 补充说明
> 这里 POMDP 设定是部分可观测，代理只见 o_t 不见全状态，所以指导必须动态。

> [!warning] 异议
> - 作者选内积相似度而不是其他距离，为什么？没解释。
> - 反直觉的是，提示 h_t 包括纠错，但如果 LLM 纠错偏了呢？缺少鲁棒性讨论。
> - 算法 1 有，但没消融证明双层缺一不可。

---

## Experiments

> [!note] 重要指标解释
> 除了成功率，还有 WebShop 的平均奖励，奖励测属性匹配度，为什么用？因为成功率是 0/1 厳苛，奖励能看部分匹配，适合购物场景的模糊目标。

- **定性总结**：实验结果显示 HiPLAN 在长任务上提升明显，证明分层指导确实帮代理避开局部陷阱和噪声。

- **图片占位**：关键结果处保留 ![[Table-1.png]] 和 ![[Table-2.png]]。

- **关键分析**：聚焦于 *Ablation Study* 中揭示的方法有效性来源，比如去掉里程碑或提示，性能掉 11-32%，证明双层协同是关键；去掉里程碑演示也掉，验证了复用价值。

![[Figure-3.png]]
*结合图片的一句话分析：注意看 ablation 图，HiPLAN 比变体高，突出双层和演示的贡献。*

> [!warning] 漏洞
> - 没比更强的基线如多代理系统或强化学习融合的。
> - 数据集 ALFWorld 和 WebShop 挑了家务和购物，是否泛化到更动态环境如游戏？
> - 结果没提失败案例分析，可能是 cherry-picked 成功路径。

---

## Future

作者提及的方向包括扩展到更多基准和多模态环境；作为专家，我觉得值得挖掘的潜在方向是整合强化学习来在线更新里程碑库，避免纯离线依赖；还有处理不确定性更高的开放世界任务，比如用不确定性估计来动态调整检索阈值。
