# ParamMem: Augmenting Language Agents with Parametric Reflective Memory

ICML 2026

**核心一句话**：这篇论文想解决反思型 Agent 反复说同一类废话的问题。作者把跨样本的反思模式微调进一个轻量模型，让它在推理时生成新的反思信号，再与 episodic memory 和 cross-sample memory 一起喂给 Agent，从而扩大错误诊断空间。

---

## Key Contribution

- ParamMem is a parametric memory module that internalizes cross-sample reflection patterns.
- ParamAgent integrates parametric memory with episodic memory.
- ParamAgent-plus further unifies episodic memory, cross-sample memory, and parametric memory.
- ParamMem is sample-efficient, supports self-improvement, and enables weak-to-strong transfer.

作者的核心判断很直接：反思型 Agent 的瓶颈不只在于反思是否正确，也在于反思是否足够多样。原文先用 Figure 1 说明 reflective diversity 和任务表现之间有正相关，平均 Pearson correlation 为 0.76。这个图是整篇论文的动机入口。

![[98_Assets/ParamMem.png]]

我注意到这里的贡献点其实有两层。第一层是方法层面的 memory 形态变化：从 prompt diversity 和 retrieval diversity，推进到 parametric diversity。第二层是 Agent 系统层面的组合：ParamMem 本身可以单独插入 Reflexion，也可以和 DoT-bank 这类跨样本检索方法组合。作者的野心在于把反思信号做成一种可训练、可迁移、可采样的模块，而不只是 prompt engineering。

> [!tip] 这篇论文最值得记住的直觉
> 反思失败时，Agent 经常卡在同一个错误解释里。ParamMem 的作用是提供另一组错误诊断假设，让 Agent 有机会看到自己原本不会说出来的可能原因。

---

## Method

### Motivation: 为什么反思需要多样性

作者的直觉是：Agent 的自我反思会陷入重复模式，导致每一轮修改都沿着同一个错误方向继续走。原文指出，已有研究发现 self-reflection 经常生成重复且不准确的内容，这会限制反思机制的收益。DoT 通过 prompt 修改增加反思多样性，DoT-bank 通过跨样本轨迹增加多样性，但检索方法依赖 embedding similarity，容易受到组合泛化能力和 embedding collapse 的限制。

作者的问题可以概括为：

**如何在 episodic memory 和 retrieval memory 之外，再提供一种新的反思多样性来源？**

我注意到这不是一个很强的理论问题，更像一个经验驱动的问题。Figure 1 显示 diversity 和 performance 正相关，但相关性不能证明多样性是性能提升的原因。作者后面用 ablation 和 case study 补这个论证，但因果链条仍然比较软。

> [!warning] 动机中的潜在漏洞
> Figure 1 的横轴是反思文本 embedding 的 pairwise cosine distance，纵轴是任务表现。这里衡量的是文本分布差异，不一定等价于有效的错误诊断差异。两个反思可以语义距离很远，但都没有指出真正 bug。作者默认 diversity 对 reasoning 有益，这个假设需要更细粒度的诊断质量分析。

---

### ParamMem: 把跨样本反思模式写进参数

**作者的直觉是：与其每次从 memory bank 里找相似样本，不如训练一个小模型，让它学会跨样本的反思模式，然后在新样本上生成新的反思。**

ParamMem 的构建流程很简单。作者先构造辅助数据集：

$$
D = {(x_i, r_i^g)}_{i=1}^{n}
$$

其中 $x_i$ 是输入任务，例如编程题或数学题，$r_i^g$ 是由 LLM 根据 task-specific prompt 生成的辅助反思信号。然后作者用 LoRA 微调一个预训练 LLM，得到 parametric module $M_g$。

这里 $r_i^g$ 的含义随任务改变：

- 编程和数学任务中，$r_i^g$ 是全局反思，列出潜在错误、可能 bug、错误实现方式。
- 多跳问答中，$r_i^g$ 不是完整 passage，而是 query 的语义单元和潜在推理子任务，因为直接放入所有支持文本会消耗大量 token。

![[98_Assets/ParamMem-1.png]]

Figure 3 很关键。左边的编程例子中，ParamMem 生成的是可能错误实现和修正提示。右边的 multi-hop QA 例子中，ParamMem 输出 key components 和 reasoning trace。也就是说，ParamMem 并不是固定生成同一种反思文本，它更像一个任务相关的 reasoning hint generator。

> [!note] 我对 ParamMem 的理解
> ParamMem 本质上是一个反思风格的 SFT 模型。它不是存储具体 episode，而是把很多样本中的反思分布压缩进参数。推理时通过 temperature sampling 从这个分布中采样一条新的反思。

> [!warning] 这里有一个容易被忽略的问题
> 作者说 ParamMem 学到 cross-sample regularities，但训练目标仍然是普通的监督微调。它没有显式优化 diversity，也没有显式约束反思覆盖不同错误类型。所谓 diversity 主要来自训练数据分布、LoRA 后模型分布变化、以及温度采样。这个机制有效，但理论解释偏经验。

---

### ParamAgent: 把 ParamMem 接入 Reflexion

**作者的直觉是：原来的 Reflexion 只看自己过去几轮的反思，现在额外给它一条由 ParamMem 生成的全局反思，让 actor 在生成答案时看到更多诊断假设。**

Reflexion 的基本形式是：

$$
y_k \sim p_{\theta}(\cdot \mid x, r_{1:k-1})
$$

这里 $x$ 是任务输入，$r_{1:k-1}$ 是前几轮 self-reflection，$y_k$ 是第 $k$ 轮生成的候选答案。

ParamAgent 加入 ParamMem 后变成：

$$
r_k^g \sim p_{\phi}(\cdot \mid x)
$$

$$
y_k \sim p_{\theta}(\cdot \mid x, r_{1:k-1}, r_k^g)
$$

这里 $r_k^g$ 是从 ParamMem 采样出来的反思信号。作者把它和 episodic memory 中的 self-reflection 拼接后喂给 actor。

![[98_Assets/ParamMem-2.png]]

Figure 2 展示了几个框架的区别：

- Reflexion / DoT：只有 episodic memory。
- DoT-bank：episodic memory + cross-sample memory。
- ParamAgent：episodic memory + parametric memory。
- ParamAgent-plus：episodic memory + cross-sample memory + parametric memory。

我注意到 ParamAgent 的工程实现很干净：它不改环境交互，不改 evaluator，不改 actor 的训练，只是在 prompt 中多塞一条 ParamMem 反思。这个设计的好处是容易接入现有 Agent 框架。代价是 token 会增加，而且质量完全依赖生成的反思是否对当前任务有用。

> [!tip] Aha moment
> ParamMem 没有直接和环境交互，但它会改变 actor 的输出分布。actor 生成新的答案后，环境反馈又会产生新的 self-reflection。这样 ParamMem 间接参与了动态反思循环。

---

### ParamAgent-plus: 三种 memory 的组合

**作者的直觉是：retrieval memory 提供相似题目的经验，ParamMem 提供参数化生成的反思，两者信息来源不同，可以互补。**

ParamAgent-plus 在 ParamAgent 的基础上加入从 memory bank $B$ 中检索到的轨迹 $\tau_{1:j}$：

$$
y_k \sim p_{\theta}(\cdot \mid x, r_{1:k-1}, r_k^g, \tau_{1:j})
$$

其中 $\tau_{1:j}$ 是从成功任务中检索到的相似 reasoning trajectories。作者为了公平，沿用 DoT-bank 的 retrieval 机制。

Algorithm 1 里流程分成两阶段：

1. Phase 1 先运行 ParamAgent。如果任务成功，就把轨迹存入 bank。
2. Phase 2 对失败任务再尝试 ParamAgent-plus，从 bank 中检索相似轨迹后重新求解。

![[98_Assets/ParamMem-3.png|300]]

> [!question] 这里的流程有一点需要追问
> Phase 2 只对失败任务启用 cross-sample memory，这会让 ParamAgent-plus 更像一个 reattempt 机制。作者报告结果时是否严格区分了所有任务一开始就使用三种 memory，还是失败后才启用？Algorithm 1 更像后者。这个差异会影响 token cost 和实际部署逻辑。

---

### Temperature-controlled sampling

**作者的直觉是：第一轮需要稳定一点，后续轮次需要更多探索。**

Algorithm 1 中，作者设置第一轮 $T=0.2$，后续轮次 $T=1.0$。实现细节也说明所有实验固定最大迭代数为 5，ParamAgent 第一轮低温采样，后续高温采样以促进多样性。

这点很符合反思型 Agent 的动态过程。第一轮如果反思太飘，可能会把 actor 带偏。后续轮次已经有环境反馈和 self-reflection，可以接受更多探索。

> [!warning] 缺失的消融
> 作者没有充分解释为什么第一轮用 0.2，后续用 1.0。这里最好有 temperature schedule 的 ablation，例如固定 0.2、固定 0.7、固定 1.0、逐轮升温。否则这个策略看起来合理，但很难判断收益来自 ParamMem 本身还是采样策略。

---

## Experiments

### Setup

作者在三个领域评估：

- Code：HumanEval、MBPP，另外用 LiveCodeBench 做更强评估。
- Math：MATH。
- Multi-hop QA：HotpotQA、2WikiMultiHopQA。

编程任务用 Pass@1，数学和多跳问答用 0-1 accuracy。编程最终评估用 hidden tests。

对比方法包括 Base、Reflexion、Retroformer、DoT、DoT-bank。这里 Retroformer 很关键，因为它同样使用 parametric reflective module，但目标是提高 reflection accuracy，而 ParamMem 的目标是增加 reflection diversity。

> [!note] 重要 baseline
> Retroformer 是这篇论文最直接的对照。它说明参数化反思模块本身并不新，ParamMem 的差异在于把目标从反思准确性转到反思多样性。

---

### Main Results

**作者的核心结论是：ParamMem 在 code、math、QA 三类任务上都能提升 Reflexion-based framework。**

Table 1 中，ParamAgent 在 HumanEval 上将 Llama-3.1-8B 从 Base 的 59.15 提升到 82.93，超过 Reflexion、DoT 和 DoT-bank。ParamAgent-plus 在 MATH 上达到 75.45，超过 DoT-bank 的 73.02。QA 上也有明显提升，例如 2WikiMultiHopQA 上 Llama-3.1-8B 从 Base 的 40.33 提升到 ParamAgent 的 88.67。

![[98_Assets/ParamMem-4.png]]

我注意到不同任务的收益结构并不一样。编程和 QA 中，ParamAgent 单独就很强，说明 parametric reflection 本身有用。数学任务中，ParamAgent-plus 明显更强，作者也承认数学更依赖类似题目和解法模式，因此 cross-sample trajectories 更关键。

> [!warning] token cost 需要认真看
> Table 1 里性能提升通常伴随 prompt token 增加。某些设置下 ParamAgent-plus 的 token 成本非常高，例如 MATH 上 token usage 明显放大。作者在 limitations 中也承认额外反思多样性带来 token 消耗问题。

---

### Diversity Analysis

**作者的直觉是：如果 ParamMem 真有效，它应该生成比普通 self-reflection 更丰富的反思分布。**

作者从两个角度验证 diversity：

1. 静态设置：对 HumanEval 每个任务采样 10 条 ParamMem 反思，计算 embedding 的 pairwise cosine distance。
2. 动态设置：把 ParamMem 接入 Reflexion 后，收集完整反思历史，用 K-means 和 silhouette score 分析语义聚类。

Figure 4a 显示，用 GPT-4o-mini 数据训练的 ParamMem diversity 最高，平均 $D_{mean}=0.351$。用 Llama-3.1-8B 自己生成的数据微调后，也比 frozen Llama-3.1-8B 更高。Figure 4b 显示 ParamAgent 的最优聚类数 $K^*=39$，高于 Reflexion、DoT 和 DoT-bank。Figure 4c 显示 ParamAgent 的 silhouette score 更高，说明它的反思分布既更分散，也有一定语义结构。

![[98_Assets/ParamMem-5.png]]

> [!tip] 这里的实验设计比较有说服力
> 作者没有只看最终分数，而是把反思文本本身拿出来分析。静态 diversity 说明 ParamMem 会生成更分散的反思，动态 clustering 说明这种分散在 Agent 迭代过程中仍然存在。

> [!warning] diversity metric 仍然偏粗
> embedding distance 和 K-means cluster 数只能说明文本分布更散。它不能直接说明这些反思覆盖了真正的错误原因。更强的证据应该是 error category coverage，或者对 bug localization、reasoning subgoal coverage 做人工或自动标注。

---

### Case Study: 多样性为什么有用

**作者的解释是：多样反思扩大了错误诊断的 hypothesis space。**

在 MBPP case study 中，作者观察到 Reflexion 和 DoT 的 self-reflection 有时没有抓住核心错误，甚至会把 Agent 带向错误实现。ParamAgent 通过更丰富的反思信号，增加遇到正确线索的概率。

这句话其实是整篇论文最核心的机制解释：

**多样性不是为了让反思看起来不一样，而是为了让错误解释空间更大。**

> [!question] 这里仍然缺一块
> 作者说 broader diagnostic hypotheses，但实验主要展示最终任务成功率和文本多样性。还缺少一个中间变量：ParamMem 是否真的更常命中正确错误原因。如果没有这个分析，diversity 到 success 的因果链条仍然依赖直觉。

---

### Self-improvement

**作者的直觉是：即使没有更强模型提供数据，基础模型自己生成的反思数据也可以用来训练 ParamMem，从而让 Agent 获得额外 diversity。**

Table 2 中，Llama-3.1-8B 同时作为 agent 和数据生成器。ParamAgent-plus 在 HumanEval 上达到 86.59，在 HotpotQA 上达到 83.33，超过 DoT-bank。作者据此认为 ParamMem 可以在没有 stronger external model 的情况下增强 reasoning。

![[98_Assets/ParamMem-6.png]]

我注意到这个结果对 Agent 研究很有价值，因为它避开了一个常见问题：方法提升是否只是因为背后用了更强的 teacher。这里至少说明，ParamMem 不完全依赖 GPT-4o-mini 这类外部模型。

> [!warning] self-improvement 的表述要谨慎
> 这里的 self-improvement 更接近 self-generated SFT data 后的分布重塑。它并没有证明模型获得了新的知识，也没有使用环境反馈筛选高质量样本。提升可能来自反思格式稳定化、采样分布变化、LoRA 适配任务域等因素。

---

### Iterative Self-teaching

**作者的直觉是：ParamMem 可以反复用自己生成的新反思再训练自己，让反思分布逐步变得更有用。**

作者从 Llama-3.1-8B-Instruct 开始，用 1000 个样本训练 ParamMem。训练后再用当前 ParamMem 为同样输入生成新的反思目标，得到 $D'$，然后继续微调，重复 3 轮。Figure 5 显示 ParamAgent 在 HumanEval 上逐轮提升，而 ParamAgent-plus 提升较小。作者推测 ParamAgent-plus 已经接近 diversity ceiling，因为它有 cross-sample trajectories。

![[98_Assets/ParamMem-7.png]]

> [!question] 这里我会重点追问退化风险
> 反复训练自己生成的数据，很容易产生分布收缩或模板化。作者报告了 3 轮提升，但没有展示更多轮次，也没有展示反思质量是否变得更单一。这里需要长期迭代曲线和 diversity collapse 分析。

---

### Weak-to-strong Transfer

**作者的直觉是：强模型不一定需要更强的反思模型，小模型生成的多样反思也能补充强模型的盲区。**

Table 3 中，agent 使用 Qwen3-Next-80B-A3B-Instruct，ParamMem 分别使用 Llama-3.1-8B 或 Qwen3-Next-30B-A3B-Instruct。结果显示两种 ParamMem 配置都能超过 baseline。代码任务上 30B ParamMem 更强，multi-hop QA 中较小的 Llama-3.1-8B ParamMem 反而优于 30B 版本。

![[Table 3.png]]

> [!tip] 这个结果很有意思
> 它暗示 ParamMem 的价值不完全由模型能力决定。只要它能提供与 actor 不同的反思分布，就可能帮助更强 actor。这里的关键词是互补性，而不是 teacher 能力。

> [!warning] 还缺少分布互补性的直接证据
> 作者展示了性能提升，但没有测量 ParamMem 和 actor 自身反思之间的差异来源。比如小 ParamMem 是否更保守、是否更关注局部 bug、是否更偏向结构化拆解，这些都没有展开。

---

### Sample Efficiency

**作者的直觉是：如果 ParamMem 学的是反思模式，那么不需要特别大的训练集，少量多样样本就足够。**

作者用 K-means 从 GPT-4o-mini 合成数据中选 500 个多样样本进行微调。Table 4 显示，ParamAgent 500 samples 在 HumanEval 上达到 81.71，接近 8000 samples 的 82.93；ParamAgent-plus 500 samples 达到 86.59，甚至超过 ParamAgent 8000 samples。

![[Table 4.png]]

> [!tip] 这个结果对实践很重要
> ParamMem 不需要大规模训练。它更像一个轻量 task adapter，用少量多样反思样本就能提供可用的反思分布。

---

## Critical Thinking

### 1. 核心创新成立，但本质接近 reflection SFT adapter

ParamMem 的形式很简洁：合成反思数据 + LoRA 微调 + 推理时采样。这个方法有效，但从训练机制看，它并没有发明新的 memory learning objective。它的创新主要在于把参数化模块定位为 reflective diversity source，并把它系统性接入 Agent memory 框架。

### 2. Diversity 和 correctness 的关系没有完全讲透

作者强调 diversity，但任务成功最终仍然需要正确反思。过高 diversity 可能产生噪声，甚至误导 actor。论文中 Retroformer 在 MATH 上强于 ParamAgent，作者解释为数学任务中 reflection accuracy 可能更重要。这个观察很关键，因为它说明 diversity 并非单调收益。

> [!warning] 可能存在任务依赖性
> 编程和 QA 中，错误诊断空间比较多样，ParamMem 很有效。数学中，类似题目和准确推导更重要，cross-sample trajectories 的价值更明显。ParamMem 的最佳使用场景可能是错误类型丰富、反馈信号稀疏、self-reflection 容易重复的任务。

### 3. Token cost 是真实问题

ParamMem 的收益来自额外反思输入，因此 token 消耗增加很自然。作者在 conclusion 中承认这是 limitation，并提出未来研究 token-efficient integration。

我认为后续可以考虑把 ParamMem 输出压缩成结构化 fields，例如 error type、suspected cause、repair direction、confidence，而不是完整自然语言段落。

### 4. 缺少反思质量的中间评估

论文证明了：

- ParamMem 提升最终表现。
- ParamMem 提高 embedding diversity。
- case study 显示多样反思可能扩大诊断假设。

但它还没有充分证明：

- ParamMem 更常指出真实错误。
- ParamMem 的反思覆盖更多错误类别。
- ParamMem 的错误反思不会显著误导 actor。
- ParamMem 和 retrieval memory 的互补性具体来自哪里。

这些会决定这个方法能否从 empirical trick 变成更稳定的 Agent memory 机制。

---

## Future

### 作者提到的方向

作者明确提到未来要解决 token cost，探索更高效的 ParamMem 集成方式。

### 我认为值得继续挖的方向

1. **Reflection diversity 的结构化度量**

当前使用 embedding distance 和 clustering。后续可以设计更贴近任务机制的指标，例如 bug category coverage、reasoning subgoal coverage、repair action diversity、false diagnosis rate。

2. **Diversity-quality Pareto frontier**

ParamMem 需要在多样性和准确性之间找平衡。一个自然方向是同时建模 diversity 和 usefulness，让 ParamMem 输出多条候选反思，再由 evaluator 或 critic 选择。

3. **Token-efficient ParamMem**

可以把自然语言反思压缩成结构化 memory token 或 schema，例如：

```markdown
error_type: boundary condition
evidence: failed hidden test likely includes empty list
repair_hint: add explicit empty input branch
```

这比长段反思更适合 Agent 循环。

4. **Actor-aware ParamMem**

当前 ParamMem 只看输入 $x$，没有看 actor 当前失败答案 $y_t$ 和 evaluator feedback。一个更强版本可以生成条件反思：

$$
r_t^g \sim p_\phi(\cdot \mid x, y_t, feedback_t)
$$

这样 ParamMem 会更像一个参数化 debugger，而不只是全局 hint generator。

5. **ParamMem 与 retrieval memory 的互补机制分析**

ParamAgent-plus 有性能收益，但机制还不够清楚。后续可以分析 retrieval trajectory 和 ParamMem reflection 的 overlap。如果两者高度重复，说明 ParamMem 只是提供了另一种表达；如果互补，才说明三种 memory 的组合有真正价值。
