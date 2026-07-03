# Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents

ICML-2026

**核心一句话**：作者想解决长期交互历史中记忆检索太死的问题。传统 Agent 先按 query 检索 top-k，再把结果塞给 LLM 推理；MRAgent 把检索过程改成多步探索，让 LLM 一边推理一边决定下一步该沿着哪些记忆线索继续查。作者的核心判断是：长期记忆更像**重建过程**，需要根据中间证据不断改写检索方向。

---

## Key Contribution

- We propose active memory reconstruction, a new paradigm that integrates memory access into the reasoning process, allowing the agent to dynamically adapt search strategy based on intermediate evidence.
- We introduce a Cue–Tag–Content memory graph in which associative tags mediate retrieval between cues and content, enabling the LLM to identify promising retrieval paths while pruning irrelevant branches.
- We provide a theoretical analysis proving that active retrieval policies are strictly more expressive than passive retrieval.
- Extensive experiments demonstrate that MRAgent outperforms strong baselines with significantly improved token and runtime efficiency.

作者的贡献可以压缩成三层：

- **检索范式变化**：作者把 memory access 从一次性 top-k 检索改成状态依赖的多步决策过程。每一步检索之后，LLM 都可以根据已发现证据调整下一步要查什么。这个设计直接针对长程记忆问答中的多跳问题，比如先发现时间线索，再用时间线索去查另一个人的活动。
- **记忆结构变化**：作者没有直接让 query 连到 content，而是在中间引入 tag。tag 的作用是给记忆边加一层语义路由，先判断某个 cue 下面哪些关联方向值得走，再决定是否读取完整 memory content。这样可以减少直接展开 graph neighbor 带来的噪声。
- **理论包装变化**：作者用 active policy 和 passive policy 的表达能力差异来支撑方法动机。直觉是，passive retrieval 必须提前选完所有节点；active retrieval 可以先看一个节点，再根据返回内容决定下一个节点，所以在类似树形寻路的任务上具备天然优势。

> [!tip] 这篇论文真正值得学习的地方
> 作者抓住了一个很清楚的科研直觉：长期记忆任务的难点通常不在于缺少一个相似文本，而在于**当前 query 没有包含足够检索线索**。如果检索系统只能看原始 query，它就很难找到需要中间推理才能暴露出来的证据。

---

## Method

### 1. Active Memory Access

**直觉**：作者认为记忆检索应该像侦查线索一样推进。先拿到一部分证据，再根据证据决定下一步查哪里。

作者先把外部记忆写成 memory units $V=\{v_1,\dots,v_N\}$。给定 query $x$，memory access 要在 $T$ 步内选择若干记忆单元。传统 passive retrieval 的形式是：

$$
\{v^{(1)}, \dots, v^{(T)}\} = \pi_p(x)
$$

这里 $\pi_p$ 只依赖 query。也就是说，系统在看到任何中间证据之前，就已经决定了要取回哪些 memory units。作者认为这就是 static retrieve-then-reason 的根本限制。

Active reconstruction 写成：

$$
v^{(t)} = \pi_a^{(t)}(x, S^{(t-1)}), \quad S^{(t)} = S^{(t-1)} \cup {v^{(t)}}
$$

这里 $S^{(t-1)}$ 是前面已经积累的证据。新的记忆选择依赖 query，也依赖已检索证据。这个公式的含义很直接：每一步检索都允许被前一步结果影响。

![[98_Assets/Memory Is Reconstructed.png|500]]

Figure 1 的作用是建立核心对比：passive retrieval 把 memory 当成静态数据库，active reconstruction 把 memory 当成可探索空间。我注意到，这个图其实在暗示一个更大的变化：retriever 从一个相似度函数变成了一个带状态的 controller。

> [!question] 这里的动机是否充分
> 作者把现有系统整体归为 passive retrieval，这个判断大体成立，但有一点需要留意：某些 agentic RAG 系统也会多轮查询，只是查询对象通常是外部 corpus，而这里强调的是 agent 自身的长期交互记忆。作者在 related work 中也承认 Search-o1 和 Search-R1 已经把 search 放进 reasoning loop，但它们处理的是单次任务中的外部知识缺口，不是 persistent interaction history。

---

### 2. Why Passive Retrieval Fails

**直觉**：query 里经常没有正确答案所需的全部线索，直接相似度检索只能找到表面相关内容。

作者把 passive retrieval 分成两类。

第一类是 similarity-based retrieval：

$$
\pi_{sim}(x)=TopK({sim(x,v)}_{v\in V}, k)
$$

它的检索结果完全由 query 和 memory unit 的相似度决定。问题是，如果用户问的是多跳问题，真正有用的证据可能和 query 表面词不相似。比如 Figure 2 里的例子，query 提到 Nate 的视频游戏比赛，直接检索会召回很多比赛相关内容，但关键线索其实是 July。

第二类是 graph-based retrieval：

$$
V_{sim}=TopK({sim(x,v)}_{v\in V},k)
$$

$$
\pi_{graph}(x)=V_{sim}\cup Neighbor(V_{sim})
$$

这种方法比纯向量检索多了图邻居扩展，但它仍然使用固定规则。只要相关证据没有连在 top-k seed 的邻域中，系统仍然找不到。固定 N-hop 扩展还会带来大量无关节点。

![[98_Assets/Memory Is Reconstructed-1.png]]

Figure 2 是这篇论文最关键的直觉图。左边的 similarity retrieval 只围绕 video game tournament 找，召回噪声很多。中间的 graph-based retrieval 多走了一些 neighbor，但仍然没有找到 Caroline 的 July 活动。右边的 active reconstruction 先从已检索内容中推理出 July，再把 July 当成新的检索约束去查 Caroline。这里的关键不是图本身，而是**中间证据能不能转化成新 query cue**。

> [!warning] 这里的潜在漏洞
> 作者把失败原因归结为 passive retrieval 无法根据中间证据调整策略。这个判断合理，但 Figure 2 本质上是一个手工构造的解释性例子。它说明 active retrieval 有优势场景，但不能单独证明真实 benchmark 中的大部分错误都来自这个原因。

---

### 3. Cue–Tag–Content Associative Memory

**直觉**：作者引入 tag 是为了让 LLM 在读取完整记忆前，先判断某条关联路径值不值得继续走。

作者把 memory 建成异构图：

$$
M=(C,V,R)
$$

其中：

- $C$ 是 cues，比如实体、属性、关键词。
- $V$ 是 contents，也就是具体记忆内容。
- $R \subseteq C \times G \times V$ 是带 tag 的关系三元组。
- 每个关系 $(c,g,v)$ 表示 cue $c$ 通过 tag $g$ 连接到 content $v$。

tag 的作用是做中间语义桥。作者让 LLM 先选 tag，再根据 cue-tag pair 取 content。这避免了直接从 cue 展开到大量 contents。

核心映射是：

$$
\phi_{c\rightarrow g}(c) \triangleq \{g \mid (c,g,\cdot)\in R\}
$$

$$
\phi_{(c,g)\rightarrow v}(c,g) \triangleq \{v \mid (c,g,v)\in R\}
$$

第一行表示：给定 cue，激活候选 tags。第二行表示：给定 cue 和选中的 tag，再取对应 contents。作者把 associative reasoning 和 content retrieval 拆开，目的是让 LLM 先在便宜的 tag 层做路由，减少读取完整 episodic content 的成本。

![[98_Assets/Memory Is Reconstructed-2.png]]

Figure 4 上半部分展示了从 dialogue 到 associative memory system 的构建流程。dialogue 先经过 element generation，抽取 cues、tags、episodes、semantic memories，再构成图。右侧的 semantic memory 说明作者不只存事件，还存稳定属性。下半部分展示 active reconstruction：query 初始化 cue，LLM 选择 action，memory traversal 产生候选节点，再由 LLM routing 保留有用证据。

> [!tip] 这个设计的精妙点
> tag 是一个很轻的中间层。它既比原始 content 短，适合 LLM 快速判断；又比普通 graph edge 更有语义，适合承担检索路径选择。这个设计把 graph retrieval 中最难的 neighbor explosion 问题转成了语义路由问题。

> [!warning] 这里有一个依赖假设
> tag 的质量很关键。如果 LLM 在 memory construction 阶段抽取的 tag 太泛，路径选择会退化成粗糙分类；如果 tag 太细，系统可能无法跨事件泛化。作者展示了 tag 有用的 ablation，但对 tag 质量、tag 粒度、tag 冲突的系统分析还不够充分。

---

### 4. Multi-Granular Memory Layers

**直觉**：长期记忆不应该只有原始事件。Agent 需要同时访问具体事件、稳定事实和高层主题。

作者把 memory contents 分成三层：

#### Episodic Layer

Episodic memory 存具体事件 $e_i \in V_e$。每个 episode 对应一个特定时间和上下文，通过 cues 和 tags 取回。作者还把 episodic memories 放到统一时间线中，使系统可以在 reconstruction 中加入 temporal constraints。

#### Semantic Layer

Semantic memory 存稳定知识 $s_i \in V_s$，比如个人属性、偏好、一般事实。每个 semantic node 锚定到实体 cue，tag 表示 aspect，比如 personality traits、long-term preferences、factual attributes。这样系统可以直接访问稳定事实，而不必每次从长历史事件中重新归纳。

#### Abstraction Layer

Topic nodes $\tau \in V_\tau$ 总结多个 episodes 的共同模式。Agent 可以先定位 topic，再向下找到相关 episodes。作者把这个叫 top-down transition $\phi_{\tau\rightarrow e}$。

![[98_Assets/Memory Is Reconstructed-2.png]]

Figure 4 里左上角的 episodic memory 和右上角的 semantic memory 是两个互补分支。episodic layer 保留细节，semantic layer 保留稳定抽象，topic layer 负责更粗粒度的入口。我注意到，作者在结构上其实做了一个 memory routing hierarchy：topic 负责粗定位，tag 负责路径选择，content 负责证据落地。

> [!question] 这里原文没有完全说清楚
> semantic memory 和 topic memory 如何避免重复、冲突、过期？作者后面承认当前实现使用相对简单的 construction strategy，没有引入复杂的 updating 或 forgetting mechanism。也就是说，当前方法更擅长检索时动态重建，对长期维护还没有完整解决。

---

### 5. Memory Population via LLM Distillation

**直觉**：作者让 LLM 先把原始对话压成可检索图元素，使后续检索能在结构化空间里进行。

构建 episodic memory 时，作者先对原始 dialogue text $T$ 做重写和分段：

$$
{e_i} \leftarrow R_{LLM}(T)
$$

这里 $R_{LLM}$ 负责 pronoun resolution、temporal normalization、episodic segmentation。换句话说，它把含糊的对话改写成上下文更明确的事件单元。

然后对每个 episode 生成 tag 和 cues：

$$
g_i \leftarrow T_{LLM}(x_i), \quad C_i \leftarrow K_{LLM}(x_i)
$$

$T_{LLM}$ 生成短 tag，总结 episode 的核心语义关系；$K_{LLM}$ 抽取实体、属性和显著描述。每个 cue 通过 tag 连到 episode，形成 Cue–Tag–Episode relations。

semantic memory 的构建公式是：

$$
{(c_i^s,g_i^s,s_i)}\leftarrow S_{LLM}(T)
$$

其中 $s_i$ 是稳定语义内容，$c_i^s$ 是实体级 cue，$g_i^s$ 是 aspect-level tag。

topic memory 通过抽象函数得到：

$$
{\tau_j}\leftarrow A_{LLM}({e_i})
$$

topic node 总结一组 episodes 的共同主题，并连接到对应 episodes。

> [!warning] 构建阶段的成本与误差可能被低估
> 作者强调 MRAgent 的 construction phase 比某些 baseline 更轻，但这里仍然依赖多次 LLM extraction。更关键的是，构建阶段一旦抽错 cue、tag、semantic unit，后续 active reconstruction 可能沿着错误路径越走越远。论文的实验主要验证结果提升，对 construction error 的鲁棒性分析不足。

---

### 6. Reconstructive Memory Framework

**直觉**：作者把检索写成一个显式状态机。每一步都有当前候选节点、已积累证据、可选 traversal actions。

重建状态定义为：

$$
S^{(t)}=(Z^{(t)},H^{(t)})
$$

其中：

- $Z^{(t)}$ 是当前 active set，包含 cues、tags、contents，是下一步 traversal 的候选对象。
- $H^{(t)}$ 是已重建上下文，也就是前面步骤积累的证据。

作者定义一组 traversal actions：

$$
A={\Pi_1,\dots,\Pi_m}
$$

这些 action 来自前面的映射操作，包括 Cue→Tag、Cue,Tag→Content、Content→Cue,Tag。forward traversal 用于从 cue/tag 找 content；reverse traversal 用于从已检索 content 反向激活新的 cue/tag。

forward traversal：

$$
\Pi_{c\rightarrow g}(C^{(t)}) \triangleq \bigcup_{c'\in C^{(t)}} \phi_{c\rightarrow g}(c')
$$

$$
\Pi_{(c,g)\rightarrow v}(C^{(t)},G^{(t)}) \triangleq \bigcup_{c'\in C^{(t)}}\bigcup_{g'\in G^{(t)}} \phi_{(c,g)\rightarrow v}(c',g')
$$

reverse traversal：

$$
\Pi_{v\rightarrow(c,g)}(V^{(t)}) \triangleq {(c',g') \mid \exists v'\in V^{(t)}, (c',g',v')\in R}
$$

我注意到 reverse traversal 是一个很关键的设计。它让系统从某个已取回 content 中反推出新的 cue/tag，相当于把已发现证据变成下一轮检索入口。这正是 Figure 2 中从 tournament event 推出 July 的结构化实现。

---

### 7. Memory Reconstruction Process

**直觉**：MRAgent 的运行过程是 query cue 初始化，然后循环执行三步：LLM 选方向、图上取候选、LLM 做路由和停止判断。

给定 query，MRAgent 先抽取 fine-grained cues，并和存储的 cue set 匹配，得到初始 active set $Z^{(0)}$ 和初始状态：

$$
S^{(0)}=(Z^{(0)},\emptyset)
$$

随后进入 iterative reconstruction loop，包括 LLM reasoning、controlled memory traversal、LLM-guided routing。

#### Step 1：LLM Reasoning and Action Selection

$$
A^{(t)} = f_{select}(x,H^{(t)},Z^{(t)})
$$

$ f_{select}$ 是 LLM-based action-selection function。它根据 query、当前证据和 active set，选择下一步走哪些 traversal actions。这里的目的不是直接回答，而是决定该查什么方向。

#### Step 2：Controlled Memory Traversal

$$
\tilde{Z}^{(t+1)} = \bigcup_{a\in A^{(t)}}\Pi_a(Z^{(t)})
$$

这一阶段执行图上的 traversal operator，产生新候选节点。关键点是，它不是 exhaustive expansion，而是由 LLM 选择 action 后再扩展。

#### Step 3：LLM Routing and State Update

$$
Z^{(t+1)} = f_{route}(x,H^{(t)},\tilde{Z}^{(t+1)})
$$

$$
H^{(t+1)} = H^{(t)} \cup Z^{(t+1)}
$$

$f_{route}$ 用 LLM 判断哪些候选节点真正有用，并剪掉无关分支。更新后，系统再判断当前 evidence 是否足够回答问题。

![[98_Assets/Memory Is Reconstructed-3.png]]

Figure 8 对应的是可执行算法。伪代码中，系统先 `EXTRACTCUES`，再 `ACTIVESETINIT`，然后循环执行 `fselect`、traversal、`froute`、`STOP`，最后用 `ANSWERLLM` 基于重建证据回答。作者还把运行分成 `Navigate` 和 `Answer` 两种模式：证据不足时调用 memory tools，证据足够时生成最终答案。

> [!tip] Aha moment
> MRAgent 的本质很像一个小型工具调用 Agent，只是工具不是 web search 或 calculator，而是 memory graph traversal operators。这个抽象很有价值，因为它把 memory retrieval 变成了可解释的 tool-use trajectory。

> [!warning] 这里可能存在 prompt/controller 依赖
> action selection、routing、stop 都交给 LLM。方法效果很可能依赖 prompt 质量和 backbone 能力。论文报告 Gemini 和 Claude 两个 backbone，说明方法具备一定泛化，但没有深入分析弱模型下 controller 是否稳定。

---

### 8. Theoretical Analysis

**直觉**：如果检索路径本身需要根据前一步内容决定，那么 passive retrieval 无法高效完成任务。

作者定义两个 hypothesis classes：

$$
\mathcal{H}^{LM}_{active}(T)
$$

表示 LLM 可以进行 $T$ 次 adaptive retrieval calls。

$$
\mathcal{H}^{LM}_{passive}(T)
$$

表示 $T$ 次 retrieval calls 必须提前固定，只能作为 query 的函数。主定理是：

$$
\mathcal{H}^{LM}_{passive}(T) \subsetneq \mathcal{H}^{LM}_{active}(T), \quad T\geq 2
$$

作者的意思是，active retrieval 能模拟 passive retrieval，因为它可以选择忽略历史；但 passive retrieval 不能模拟所有 active retrieval，因为它不能根据已返回节点决定下一步。

证明构造是 Binary-Tree Needle-in-a-Haystack。query 只告诉 root node。每个 internal node 的 payload 告诉下一步该走左还是右。真正答案藏在目标 leaf。active retrieval 可以从 root 开始，一步步读 bit，走到目标 leaf；passive retrieval 必须提前猜要取哪些节点，如果预算不是指数级，就很难命中目标 leaf。

active retrieval 可以做到零错误：

$$
opt(\mathcal{H}^{LM}_{active}(d+1);D*{n,d})=0
$$

因为它按路径提示逐层走到目标 leaf。

passive retrieval 的错误下界是：

$$
L(\pi_{\theta}^{pass};D_{n,d})\geq \epsilon_Y\left(1-\frac{T}{2^d}\right)
$$

因为目标 leaf 在 $2^d$ 个叶子中均匀分布，passive policy 只有 $T$ 次提前选择机会，命中目标的概率至多是 $T/2^d$。没命中目标时，它只能根据 label prior 猜。

> [!warning] 理论结果的边界
> 这个理论证明的是表达能力分离，不直接证明实际系统一定更准。构造任务非常适合 active traversal，因为每一步 payload 都明确告诉下一跳。真实记忆图里的线索通常更噪、更含糊，LLM 可能会选错 action 或过早停止。因此理论主要支撑方法动机，不能等价于经验性能保证。

---

## Experiments

### Setup

作者在两个 long-context memory benchmark 上评估：

- LoCoMo：长期对话记忆理解，包含 single-hop、multi-hop、temporal、open-domain 等问题。
- LongMemEval：跨多 session 的长期交互记忆评估，每个样本有很长的历史。

Baselines 包括 RAG、LangMem、A-Mem、MemoryOS、Mem0。作者使用 Gemini-2.5-Flash 和 Claude-Sonnet-4.5 两个 backbone，报告 F1、LLM-Judge score 和 evidence recall。

> [!note] 指标解释
> Evidence Recall 衡量检索过程是否找到了 ground-truth supporting evidence，计算方式是每个问题中已检索证据和真实证据的重叠比例。这个指标比最终答案更贴近本文主张，因为 MRAgent 的核心贡献是 memory reconstruction。

---

### Main Results

![[98_Assets/Memory Is Reconstructed-4.png]]

Table 1 显示，在 LoCoMo 上，MRAgent 在 Gemini backbone 下把 overall LLM-Judge 从最强 baseline Mem0 的 68.31 提升到 84.21；在 Claude backbone 下从 LangMem 的 78.61 提升到 88.32。multi-hop、temporal、open-domain、single-hop 几类问题上整体都有提升。

作者解释提升来自两个因素：

- CTC memory 用 tag 显式编码 associative relations，使系统可以基于语义方向选择路径。
- 多轮 memory access 让检索方向随着 accumulated evidence 调整，逐步形成 coherent reasoning chains。

![[98_Assets/Memory Is Reconstructed-5.png]]

Table 2 显示 LongMemEval 上 MRAgent 的 overall LLM-Judge 为 72.95，高于所有 Gemini-backbone baseline；当使用 Claude 做 retrieval 时，MRAgent* 达到 86.76。

> [!warning] MRAgent* 的比较需要小心
> MRAgent* 使用 Claude for retrieval，而 memories are constructed by Gemini。这个设置可以展示上限，但和 Gemini-backbone baselines 不完全同条件。它更像能力分析，不应作为最公平主结果。

---

### Cost Analysis

![[98_Assets/Memory Is Reconstructed-6.png]]

Table 3 显示 MRAgent 的 token consumption 是 118k，低于 A-Mem 的 632k、MemoryOS 的 273k、LangMem 的 3268k、Mem0 的 245k。runtime 是 586.11 秒，比 Mem0 的 533.29 秒略慢，但明显低于 A-Mem、MemoryOS、LangMem。

作者的解释是：MRAgent 把复杂关系建模推迟到 retrieval stage，在 query-specific reconstruction 中按需查证据。tag 层可以在读取 expensive episodic content 前剪掉无关路径。

我注意到这里有一个重要 trade-off：MRAgent 节省 token，但 runtime 未必最优。它用更多 LLM controller steps 换取更少的上下文输入。对 API 成本敏感的场景，这是优势；对低延迟要求很高的场景，可能需要进一步优化。

---

### Ablation Study

![[98_Assets/Memory Is Reconstructed-7.png]]

Figure 5 比较了三种结构：

- CE：Cue→Episode，直接索引。
- CTE：Cue–Tag–Episode，引入 tag 做 episodic retrieval。
- CTC：Cue–Tag–Content，完整结构。

图中绿色柱子表示 no reasoning，蓝色柱子表示 with reasoning。作者观察到：

- 加入 reasoning 后，各结构表现都提升，说明多步 traversal 是主要收益来源。
- no reasoning 设置下，CE→CTE→CTC 逐步提升，说明 tag 和 richer associative structure 确实改善检索。
- 去掉 semantic memory 后性能下降，说明 episodic memory 和 semantic memory 互补。

> [!tip] Ablation 的有效信息
> Figure 5 不是只证明完整模型最好，它分开验证了两个来源：结构本身有用，active reasoning 也有用。这比单纯去掉一个模块更清楚。

> [!warning] 还缺少的 ablation
> 我会希望看到 tag 粒度的 ablation，比如每个 episode 一个 tag、多 tag、自动 tag、人工规则 tag、随机 tag。当前结果能说明 tag 有帮助，但还不能解释 tag 应该如何设计。

---

### Multi-turn Reasoning Analysis

![[98_Assets/Memory Is Reconstructed-8.png]]

Figure 6(a) 展示 evidence recall 随 reasoning turns 增长的趋势。single-hop 和 temporal 大约三轮内接近高 recall；multi-hop 需要更多轮，recall 随步骤提升超过 30%。Figure 6(b) 对比 Average Turns 和 Max Valid Turns，作者据此认为 LLM 能较好判断什么时候继续搜索、什么时候停止。

Appendix D.6 进一步分析 budget sensitivity。结果显示，提高每轮并行检索预算 $K$ 的收益很快饱和，而增加 reasoning turns $T$ 会稳定提升准确率。作者据此认为 reconstruction depth 不能被 parallel exploration 替代。

> [!tip] 这里的结论很重要
> 多跳记忆任务真正需要的是**顺序组合证据**。一次性多取一些节点只能增加 breadth，无法代替根据前一步证据生成下一步线索的 depth。

---

### Case Study

![[98_Assets/Memory Is Reconstructed-9.png]]

Figure 7 展示一个多 session 查询：用户问 Joanna 的哪些 screenplay 被 production companies 拒绝。MRAgent 从 Joanna 这个 cue 出发，沿着 associative tags 找到 screenplay submission 和 rejection events，再查询 event context、keywords、semantic information 和 temporal information，最后对齐提交和拒绝事件，得出 first 和 third screenplays 被拒绝。

Appendix 里更详细说明了五步过程：第一轮找 submission 和 rejection events；第二轮查 event context 和 keywords；第三、四轮查 Joanna 的 semantic information；第五轮查 temporal information 以验证顺序。

> [!warning] 定性案例的风险
> 这个 case 非常贴合方法设计，能清楚展示 active reconstruction 的优势。但单个案例很可能是 cherry-picked。更强的证据应该是按错误类型统计：哪些问题需要 temporal cue，哪些问题需要 semantic memory，哪些问题需要 topic traversal。

---

## Critical Thinking

### 核心直觉是否成立

我认为作者的核心直觉成立：长期记忆问答经常需要通过中间证据发现新线索。静态 top-k retrieval 对这类任务天然吃亏，因为原始 query 不一定包含答案证据的索引词。

这篇论文的好处是，它没有只说要更好的 graph memory，而是把 retrieval policy 从 stateless 改成 stateful。这个变化比单纯换一个 memory representation 更本质。

### 方法的主要假设

- LLM 能可靠地从中间证据中推断新 cue。
- LLM 能在 tag 层做有效 routing。
- memory construction 阶段能抽出足够好的 cues、tags、semantic units。
- 多轮 traversal 的收益大于 controller 调用带来的延迟和误差传播。

这些假设大部分在实验中得到间接支持，但没有被完全拆开验证。

### 最大短板

> [!warning] 长期运行下的 memory maintenance 没解决
> 作者自己承认，当前 memory graph 静态构建，不做持续更新、合并和遗忘。随着交互累积，graph 会单调增长，存储和检索成本都会上升。

> [!warning] LLM controller 的稳定性仍是黑箱
> fselect、froute、STOP 都由 LLM 实现。论文证明了 active retrieval 更 expressive，但真实系统能不能稳定选对路径，仍取决于 LLM 的判断能力。对于小模型或噪声图，这可能是主要瓶颈。

> [!warning] Benchmark 可能偏向可检索记忆重建
> LoCoMo 和 LongMemEval 都是长期记忆 QA benchmark，天然适合检索式方法。MRAgent 是否能迁移到更开放的 agent planning、web interaction、tool-use history compression，还需要额外验证。

---

## Future

作者提到的未来方向包括：

- adaptive construction：让 memory construction 随使用动态调整。
- lightweight memory maintenance：降低长期增长带来的存储和检索成本。
- robust traversal policies：让多步检索策略更稳，减少深度探索带来的延迟。

我认为更值得继续挖的方向有四个：

1. **Tag learning instead of tag prompting**

当前 tag 由 LLM 抽取，缺少可优化目标。可以考虑让 tag 变成可学习或可反馈修正的 routing abstraction，用 retrieval success 反向改进 tag 粒度。

2. **Memory graph consolidation**

MRAgent 把复杂推理推迟到 retrieval 阶段，但长期系统不能无限增长。下一步应该研究如何把多次 reconstruction trajectory 反写回 memory graph，把常用路径压缩成更稳定的 higher-order memory。

3. **Traversal policy training**

现在 fselect 和 froute 依赖 prompt。可以把 traversal 看成 sequential decision process，用 evidence recall、answer correctness、token cost 作为 reward，训练一个更稳定的 memory navigation policy。

4. **Failure-aware reconstruction**

当前系统主要判断 evidence 是否 sufficient。更进一步可以显式判断失败类型：缺 temporal cue、缺 entity disambiguation、缺 semantic aspect、路径过宽、路径过深。这样 memory reconstruction 才能从经验中改进，而不是每次重新搜索。
