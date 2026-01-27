# HIPLAN: Hierarchical Planning for LLM Agents with Adaptive Global-Local Guidance

arXiv-2025

**核心一句话**：解决 LLM 代理在长时程复杂任务中「全局方向迷失」和「局部环境不适应」的双重问题，核心是通过「离线构建里程碑库」+「在线动态生成全局里程碑引导 + 局部步骤提示」的分层框架，平衡经验复用的通用性和执行的灵活性。

---

## Key Contribution

- We introduce HIPLAN, a novel hierarchical planning framework that tightly integrates global milestone action guides with local step-wise hints, achieving adaptive global-local guidance for agent planning.
- We propose an efficient milestone-level experience reuse strategy that allows agents to draw on prior demonstrations in a way that is both generalizable and actionable.
- We conduct extensive experiments on multiple challenging benchmarks, demonstrating that HIPLAN significantly improves task success rates and robustness compared to strong baselines, confirming its effectiveness across diverse decision-making scenarios.

- 突破现有方法的二元对立：现有方法要么侧重全局子目标分解（缺乏灵活性），要么侧重局部步骤适应（丢失全局视角），HIPLAN 首次将二者紧密耦合，让全局引导提供「路线图」，局部提示提供「实时路况」，解决长时程任务的核心矛盾。
- 里程碑级复用的粒度创新：拒绝动作级（太依赖具体场景）和任务级（细节冗余）的经验复用，选择中间粒度的里程碑级，既保留结构化知识又具备泛化能力，这是经验复用效率提升的关键。
- 跨场景跨模型的通用性验证：在文本交互环境 ALFWorld 和网页购物环境 WebShop 上均实现显著提升，且适配 Mixtral（稀疏混合专家模型）和 LLaMA（稠密预训练模型），证明框架不依赖特定模型架构或任务场景。

---

## Method

### 核心动机：现有规划方法的两大痛点

> [!question] 方法设计的底层逻辑
> 作者观察到 LLM 代理在长时程任务中面临两个无法调和的问题：1）全局规划方法（如 Plan-and-Solve）缺乏动态适应能力，遇到环境变化容易失败；2）局部步骤方法（如 REACT）容易陷入局部最优，忘记最终目标。这种二元矛盾是否是长时程规划的本质瓶颈？

作者的核心直觉是：**长时程规划需要「战略定力」和「战术灵活」的结合**——全局层面需要明确的阶段划分来避免迷失，局部层面需要基于实时观测的动态调整来纠正偏差。

![[98_Assets/HiPlan.png]]

上图清晰对比了三种方法的差异：左上角全局规划方法因缺乏灵活性失败，右上角局部步骤方法因缺乏全局引导失败，而 HIPLAN 的分层架构（下方）通过里程碑引导 + 步骤提示的组合实现自适应规划。

### 模块一：离线阶段 - 里程碑库构建（Milestone Library Construction）

> [!tip] 设计亮点
> 这一步的核心是「提前提炼结构化经验」，把专家演示转化为可复用的中间粒度知识，避免在线规划时从零开始或被冗余信息干扰。

作者的直觉是：专家轨迹中隐藏着完成任务的关键阶段（里程碑），将这些阶段提取出来并结构化存储，后续任务可通过检索相似里程碑快速获取指导。

#### 核心流程

1. 轨迹分割与里程碑描述：对每个专家轨迹ξ⁽ⁱ⁾，用 LLM（GPT-4o）将其分割为 K⁽ⁱ⁾个连续片段ζₖ⁽ⁱ⁾，每个片段对应一个语义明确的子目标，并用自然语言生成里程碑描述 mₖ⁽ⁱ⁾。
2. 嵌入与存储：将任务指令和里程碑描述分别编码为稠密向量 v_task⁽ⁱ⁾和 v_milestone⁽ⁱᵏ⁾，存储到里程碑库 M_L 中，库中条目格式为：
$$\left(v_{task}^{(i)}, v_{milestone}^{(i,k)}, m_{k}^{(i)}, \zeta_{k}^{(i)}\right)$$
- v_task⁽ⁱ⁾：任务指令的向量表示，用于任务级相似性检索
- v_milestone⁽ⁱᵏ⁾：里程碑的向量表示，用于里程碑级相似性检索
- mₖ⁽ⁱ⁾：里程碑的自然语言描述（子目标）
- ζₖ⁽ⁱ⁾：对应里程碑的轨迹片段（动作 - 观测对序列）

![[98_Assets/HiPlan-1.png]]

上图左侧展示了离线阶段的流程：从专家演示中提取里程碑，构建任务级和里程碑级两层索引的库，为在线执行提供经验支持。

> [!warning] 潜在疑问
> - 作者使用 GPT-4o 进行里程碑分割，这种手动设计的分割逻辑是否具有泛化性？如果换用其他 LLM 进行分割，是否会影响后续规划效果？原文未做相关消融实验。
> - 里程碑的数量 K⁽ⁱ⁾是由 LLM 自动决定还是人工设定？不同任务的里程碑数量差异是否会影响检索精度？

### 模块二：在线阶段 - 分层规划与执行（Hierarchical Planning and Execution）

> [!question] 动态适配的关键
> 如何让全局引导不僵化、局部提示不跑偏？作者的解决方案是「双级检索 + 动态生成」，即通过检索相似任务生成全局引导，检索相似里程碑生成局部提示。

#### 子模块 1：全局引导 - 里程碑动作指南（Milestone Action Guide）

核心作用是为当前任务提供「战略路线图」，明确完成任务的关键阶段顺序。

#### 生成流程

1. 任务检索：给定测试任务τ，将其编码为 v_τ，从里程碑库中检索 M=2 个最相似的任务及其对应的里程碑序列：
$$\left\{\left(\tau^{(j)}, \xi^{(j)}, \mathcal{G}_{\tau^{(j)}}\right)\right\}_{j=1}^{M}=Retrieve\left(v_{\tau}\right)$$
2. 引导生成：用 LLM 结合当前任务τ和检索到的相似任务信息，生成适配当前任务的里程碑序列 G_τ=[m₁, m₂, ..., m_K]：
$$\mathcal{G}_{\tau}=LLM\left(\tau,\left\{\left(\tau^{(j)}, \xi^{(j)}, \mathcal{G}_{\tau(j)}\right)\right\}_{j=1}^{M}\right)$$

> [!note] 细节补充
> 检索后会按轨迹长度重新排序，优先选择更短、更通用的示例，避免引入冗余细节。

#### 子模块 2：局部引导 - 步骤提示（Step-Wise Hints）

核心作用是为每个时间步提供「战术指导」，基于当前观测和里程碑目标纠正偏差、填补差距。

#### 生成流程

1. 里程碑检索：在时间步 t，确定当前应完成的里程碑 m_ψ(t)，将其编码为 v_m_ψ(t)，从里程碑库中检索 P=2 个最相似的里程碑及其轨迹片段：
$$\left\{\left(m_{l}^{*}, \zeta_{l}^{*}\right)\right\}_{l=1}^{P}=Retrieve\left(v_{m_{\psi(t)}}\right)$$
2. 提示生成：结合当前里程碑 mₖ、历史动作 - 观测对{(oₛ, aₛ)}ₛ=1ᵗ和检索到的轨迹片段，生成步骤提示 hₜ：
$$h_{t}=LLM\left(m_{k},\left\{\left(o_{s}, a_{s}\right)\right\}_{s=1}^{t},\left\{\left(m_{l}^{*}, \zeta_{l}^{*}\right)\right\}_{l=1}^{P}\right)$$

hₜ包含三个核心部分（可选 Action Correction）：
- Current State：当前智能体与环境、物体的关系描述
- Milestone Gap：完成当前里程碑仍需执行的操作
- Action Correction：仅当最近动作错误时，指出问题并给出正确方向

> [!tip] 设计巧思
> 步骤提示不仅是简单的动作建议，还包含「状态评估」和「差距分析」，帮助智能体理解「为什么要做」，而不只是「要做什么」，这提升了执行的鲁棒性。

#### 子模块 3：双级引导增强策略（Dual-level Guidance Enhanced Policy）

智能体的动作生成同时融合全局里程碑引导和局部步骤提示，公式如下：
$$a_{t}=\pi \left(\tau ,\left\{ (o_{s},a_{s})\right\} _{s=1}^{t},m_{k},h_{t}\right)$$
- τ：当前任务指令
- {(oₛ, aₛ)}ₛ=1ᵗ：历史动作 - 观测序列
- mₖ：当前里程碑
- hₜ：当前步骤提示

该策略确保智能体在执行动作时，既不偏离全局目标（受 mₖ约束），又能适应实时环境（受 hₜ指导）。

### 模块三：算法流程（Algorithm 1）

算法清晰呈现了 HIPLAN 的完整工作流：离线阶段构建里程碑库，在线阶段先检索生成里程碑动作指南，再在每个时间步检索生成步骤提示，最后融合双级引导生成动作，直至任务完成或达到最大步数。

> [!note] 超参数设定
> 原文设定 M=2（任务级检索数量）、P=2（里程碑级检索数量），作者称这是基于上下文学习的经验值，2-3 个示例能在性能和噪声之间取得平衡。

---

## Experiments

> [!note] 指标解释
> - ALFWorld：采用任务成功率（Success Rate），评估任务是否完全完成，涵盖 6 类子任务。
> - WebShop：采用平均奖励（Average Reward，衡量产品属性匹配度）和成功率（Success Rate，衡量是否完全满足所有要求），更贴合购物场景的实际需求。

### 核心结果

1. ALFWorld benchmark：HIPLAN 在所有子任务中均取得最高成功率，LLaMA-3.3-70B 版本的整体成功率达 94%，较 TRAD 提升 15 个百分点，较 REACT+Reflexion 提升 38 个百分点。尤其在复杂的 PutTwo 任务（需处理多个目标物体）上提升显著（从 18% 提升至 82%），证明分层引导对复杂任务的有效性。
2. WebShop benchmark：HIPLAN 的成功率达 40%（LLaMA），较最佳基线（TRAD）提升 26 个百分点，平均奖励达 0.58，远超其他基线，说明即使未找到完全匹配的产品，也能通过分层引导选择最符合约束的替代方案。
3. 步骤效率：HIPLAN 在 ALFWorld 平均步骤减少 28%，WebShop 减少 37%，证明分层引导不仅提升成功率，还能避免冗余动作，提高规划效率（见图 5）。

### 消融实验分析（关键）

作者通过三个变体验证核心组件的必要性：
- HIPLAN-Direct（无里程碑 + 无步骤提示）：性能最差，证明引导机制的核心作用。
- HIPLAN-Milestone（仅里程碑引导）：性能优于 Direct 但弱于完整 HIPLAN，说明局部步骤提示对纠正偏差、适应环境至关重要。
- HIPLAN-w/o milestone-level demonstrations（无里程碑级轨迹复用）：性能显著下降，验证里程碑级经验复用的价值。

![[98_Assets/HiPlan-2.png]]

上图显示，完整 HIPLAN 在两个数据集、两种模型上均优于所有变体，证明分层引导和里程碑复用的协同效应。

> [!warning] 实验漏洞
> - 基线选择：TRAD 在 WebShop 上的性能异常（Mixtral 版本成功率仅 4%），作者解释为低级别动作演示引入噪声，但未验证是否是 TRAD 的实现细节问题（如检索策略、提示设计），可能存在不公平比较。
> - 数据集局限性：两个数据集均为模拟环境（文本交互 + 网页购物），缺乏真实物理世界的 embodied 任务验证，泛化性存疑。
> - 模型依赖：实验中 LLaMA-3.3-70B 的性能显著优于 Mixtral，作者未分析模型规模和架构对 HIPLAN 效果的影响，是否只有大模型才能发挥分层框架的优势？

### 案例研究（ALFWorld 任务）

以「put two soapbar in garbagecan」任务为例，HIPLAN 生成 8 个连续的里程碑序列，步骤提示动态引导智能体完成物体定位、拾取、移动、放置等操作，还能纠正错误（如未到达垃圾桶就尝试放置）和复用历史信息（如重新访问已知肥皂位置）。

![[98_Assets/HiPlan-3.png]]

该案例直观展示了 HIPLAN 的工作机制：里程碑引导确保全局进度，步骤提示处理局部细节，二者结合实现连贯高效的长时程规划。

---

## Future

### 作者提及的方向

- 扩展到更广泛的任务和领域，评估框架的泛化性和可扩展性。
- 研究步骤提示经验的总结与抽象方法，实现跨任务知识迁移。

### 专家建议的潜在方向

- 优化里程碑分割的自动化方法：当前依赖 GPT-4o，可探索端到端的里程碑提取模型，减少对强 LLM 的依赖。
- 增强部分可观测环境的适应性：原文指出部分失败源于环境部分可观测（如物体状态未明确描述），可引入状态估计模块提升鲁棒性。
- 动态调整检索数量和粒度：当前 M=2、P=2 为固定值，可设计自适应策略，根据任务复杂度和相似性动态调整检索参数。
- 多模态场景适配：将框架扩展到视觉 - 语言交互任务，验证在真实物理世界中的有效性。
- 里程碑库的增量更新：当前为静态库，可设计在线增量学习机制，让框架在持续交互中更新里程碑知识。
