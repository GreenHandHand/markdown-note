# LazyMem：先广泛检索，再有选择地构建高效的长期智能体记忆

Preprint- 审稿中

**核心一句话**：LazyMem 把长期记忆中的有损压缩推迟到查询已经出现之后。系统先尽可能召回原始历史，再让一个轻量级模型根据当前问题决定哪些消息值得保留、保留哪些细节，最终只向答案模型提供极短的查询相关记忆。作者真正抓住的问题是：**未来问题未知时，很难提前判断一段历史中的哪些细节最终有价值**。

---

## Key Contribution

- **Retrieve broadly, construct selectively**：保留原始交互，在查询到来后进行高召回检索，再执行查询条件化的选择与压缩。
- **A lightweight memory-processing model trained with SFT and RL**：使用 4B 模型执行逐消息 Keep、Drop 与压缩，通过 SFT 学习任务格式，再通过 GRPO 优化记忆质量。
- **Joint reward for selection, faithfulness and utility**：奖励同时考虑黄金证据召回、无关上下文剔除、压缩忠实度以及对最终问题的效用。
- **Strong accuracy-context trade-off**：LazyMem-4B 在 LongMemEval 上达到 0.85 LJ，只向答案模型提供平均 213 个 memory token。

我认为这篇论文最值得关注的贡献其实只有一个：

**它重新定义了 Memory Construction 应该发生的时机。**

传统长期记忆通常形成：

$$
\text{History}
\rightarrow
\text{Memory Construction}
\rightarrow
\text{Storage}
\rightarrow
\text{Retrieval}(q)
\rightarrow
\text{Answer}
$$

LazyMem 改成：

$$
\text{History}
\rightarrow
\text{Raw Storage}
\rightarrow
\text{Retrieval}(q)
\rightarrow
\text{Memory Construction}(q)
\rightarrow
\text{Answer}
$$

差异看起来只是调换两个模块，实际改变了记忆构建问题的条件：

$$
m = f(H)
$$

变成

$$
m_q = f(H,q)
$$

前一种方案要求模型在不知道未来需求的情况下决定什么值得记住。后一种方案已经知道查询 $q$，所以可以明确判断什么信息值得进入当前上下文。

> [!tip] 核心 Aha Moment
> 这篇工作的核心价值不在新的 Retriever，也不在 GRPO 本身。
>
> **作者把 Memory Compression 从长期存储问题重新表述成了 Query-conditioned Evidence Construction。**
>
> 一旦采用这个表述，很多长期记忆设计中的困难自然消失了：写入时无需预测未来，不需要提前决定某个细节是否值得保留，也不会因为一次错误摘要永久丢失原始信息。

作者用 AMA-Bench 已有实验进一步支撑这个动机。即使完全排除检索错误，只比较包含目标证据的原始观察与由这些观察构建出的记忆，已有方法仍产生 19.6% 到 41.3% 的性能下降。也就是说，**Memory Construction 本身就是一个显著的信息损失源**。

> [!warning] 这里需要明确论文的边界
> LazyMem 并没有真正解决长期信息的结构化组织、主动遗忘、记忆更新等问题。它选择完整保存原始 History，然后将困难推迟到 Query Time。
>
> 因此它提出的是一种很强的 **Long-context QA Memory Architecture**，其 Memory 更接近查询条件化的信息编辑层。

---

## Method

### Overall Architecture

**直觉**：既然写入时不知道未来会问什么，就先完整保存。问题真正出现以后，再花计算量判断哪些历史内容值得送进答案模型。

![[figure-1.png]]

Figure 1 基本把整篇论文讲完了。上半部分是推理流程，下半部分是 Memory Processing Model 的训练流程。

整个在线过程可以概括为：

$$
H
\xrightarrow{\text{Retrieve}(q)}
R_q
\xrightarrow{\text{Window}}
\{W_j\}_{j=1}^{P}
\xrightarrow{\pi_\theta(q,W_j)}
m_q
\xrightarrow{\text{Answer Model}}
a
$$

其中：

- $H$：完整的原始对话历史
- $q$：当前查询
- $R_q$：高召回候选消息集合
- $W_j$：根据检索结果恢复出来的局部历史窗口
- $\pi_\theta$：Memory Processing Model
- $m_q$：最终提供给答案模型的紧凑记忆

作者把在线过程拆成三个阶段：

1. **Broad Retrieval**
2. **Selective Construction**
3. **Answer Generation**

真正有方法创新的主要是第二阶段。Retriever 本身全部采用现成组件。

---

### Broad Retrieval

**直觉**：这一阶段追求 Recall。多召回一些噪声是可以接受的，因为后面还有专门的 Memory Processing Model 负责过滤。

系统同时运行：

- Dense Retriever
- Sparse BM25 Retriever

然后使用 Reciprocal Rank Fusion 合并两个排序：

$$
R_q^{dense},R_q^{sparse}
\xrightarrow{\text{RRF}}
R_q^{fusion}
$$

之后 Cross-Encoder Reranker 再次打分，保留 Top-$n$ 消息。

正式实验中：

- $n=50$
- Dense embedding：Qwen3-Embedding-8B
- Sparse retrieval：Okapi BM25
- Reranker：BGE-Reranker-v2-M3
- Query answer model：Qwen3-32B

> [!tip] 很重要的模块职责分离
> 作者实际上人为地把 Recall 与 Precision 分开优化：
>
> $$\text{Retriever}\rightarrow\text{high recall}$$
>
> $$\text{Memory Processor}\rightarrow\text{high precision + compression}$$
>
> 这比直接让 Retriever 同时承担相关性判别和上下文预算控制更容易优化。

> [!question] Retriever 真的是次要问题吗？
> LongMemEval 中 Top-50 加邻域扩展后的黄金证据覆盖率达到 0.98，因此 Retriever 几乎已经足够强。
>
> LoCoMo 中覆盖率只有约 0.89，后续错误分析立即显示 Retrieval Miss 成为主要瓶颈。
>
> 因此 LazyMem 的成功依赖一个隐含条件：**Broad Retrieval 首先要能够把关键证据送进 Candidate Pool。**

---

### Historical Window Construction

**直觉**：Retriever 找到的是孤立消息，但对话含义经常依赖附近的消息。模型需要看到局部上下文才能理解代词、事件关系和时间信息。

对于每条被检索出的消息 $d_i$，作者向前、向后分别扩展 $w$ 条消息。

如果两条检索结果距离足够近：

$$
\operatorname{idx}(d_{i+1})-\operatorname{idx}(d_i)\le 2w
$$

就将对应区域合并为同一个连续窗口。

这样做可以理解成：

```text
Retrieval Hit

        ↓
... x x [HIT] x x ...

        ↓ expand

------ local evidence window ------

        ↓ overlap

------ merged evidence region ------
```

合并后，如果窗口仍然太长，则继续拆成长度不超过 $L$ 的重叠窗口。

正式配置：

$$
w=2,\qquad L=8,\qquad s=7
$$

因此相邻完整子窗口会共享一条消息。

> [!tip] 这个设计很实用
> Windowing 同时解决两个问题：
>
> - **context rot**：每次只让小模型处理很短的局部窗口
> - **latency**：不同窗口没有依赖关系，可以直接并行
>
> Figure 1 中 Window 阶段的 Expand、Merge、Split 就是在做这件事。

消融也比较干净：

| $w$ | LongMemEval LJ |    All@50 | LoCoMo LJ |    All@50 |
| --: | -------------: | --------: | --------: | --------: |
|   0 |           0.77 |     0.950 |      0.63 |     0.781 |
|   1 |           0.82 |     0.970 |      0.64 |     0.847 |
|   2 |       **0.85** | **0.980** |      0.68 |     0.889 |
|   3 |       **0.85** | **0.980** |  **0.69** | **0.892** |

在 LongMemEval 中，$w=2$ 后基本饱和。LoCoMo 的问题更多来自检索区域之外，因此继续扩大邻域也只能小幅改善。

---

### Query-conditioned Memory Processing

**直觉**：模型不用重新总结整个 History。它只需要针对当前问题，对每条候选消息回答两个问题：

1. 这条消息要不要保留？
2. 如果保留，其中真正有用的内容是什么？

对于窗口 $W_j$ 中的每条消息 $x$：

$$
\alpha_x\in\{\mathrm{Keep},\mathrm{Drop}\}
$$

如果

$$
\alpha_x=\mathrm{Keep}
$$

模型还需要产生压缩结果：

$$
\tilde{x}=\pi_\theta(x,q,W_j)
$$

如果选择 Drop，则这条消息完全不进入最终 Memory。

因此：

$$
m_q=
\operatorname{Concat}
\left(
\{\tilde{x}\mid\alpha_x=\mathrm{Keep}\}
\right)
$$

最终内容按照原始时间顺序拼接。对于重叠窗口中重复出现的消息，作者执行去重，并优先接受 Keep 决策。

模型实际输出 JSON，每条消息包含：

- `op`
- `compressed_content`
- `reason`

其中 Keep 可以采用三种形式：

- 原文保留
- 提取关键片段
- 简洁压缩

论文的 Prompt 还明确加入一种偏召回规则：当消息可能包含答案证据、用户事实、时间更新或查询相关实体时，即使相关性不确定也优先 Keep。

> [!warning] 这里存在一个很关键的风险
> Keep 并不等于信息安全了。
>
> 模型还会继续生成 $\tilde{x}$，因此真正的信息瓶颈变成：
>
> $$x\rightarrow\tilde{x}$$
>
> 后面的错误分析正好证明这一点。4B 模型很多失败案例都成功 Keep 了正确消息，却在压缩时删掉了完成跨会话推理所需的小细节。

---

## Training the Memory Processor

### Stage 1：SFT

**直觉**：直接对一个 4B 模型进行 RL 很容易先卡在 JSON 格式、Keep/Drop 输出数量不匹配等低级问题。SFT 先让模型学会任务接口和基本行为。

作者从 LongMemEval 训练集抽取 100 个问题，经过 Retrieval 和 Window Construction 后产生：

$$3075
$$

个 Query-Window Pair。

随后使用 DeepSeek-V4-Flash 为每条消息生成：

$$
(\mathrm{Keep/Drop},\tilde{x})
$$

教师如果：

- 丢弃黄金证据
- 输出无法解析
- 违反结构
- 输出决策数量与消息数量不匹配

对应样本会直接被删除。

原始 Teacher Label 极度不平衡：

$$
\mathrm{Keep}=3.92%
$$

$$
\mathrm{Drop}=96.08%
$$

作者在 Window Level 重新采样：

$$
388\ \text{Keep-containing windows} + 388\ \text{all-Drop windows} = 776
$$

个最终 SFT 样本。平均输入 2585 tokens，输出 873 tokens。

> [!tip] 这个 SFT 的目标非常明确
> SFT 主要承担 **interface learning**：
>
> - 输出合法 JSON
> - 每条消息输出一次决策
> - 初步学会 Keep/Drop
> - 初步学会查询相关压缩
>
> 后面的实验确实证明它主要改善的是格式执行能力。

---

### Stage 2：GRPO

**直觉**：Teacher 可以教会模型怎么做，但 Teacher 输出本身并不代表最优记忆。RL 允许模型针对记忆选择和压缩质量继续搜索。

每个 Prompt $p$ 对应：

$$
p=(q,W_j)
$$

策略对同一个 Prompt 采样 $G$ 个输出：

$$
{o^k}_{k=1}^{G}
$$

实验中：

$$
G=8
$$

每个 rollout 获得奖励：

$$
r^k=R(o^k;p)
$$

随后在同一个 Group 内计算相对优势：

$$
\hat A^k=
\frac{
r^k-\operatorname{mean}(\{r^l\}_{l=1}^{G}) }{ \operatorname{std}(\{r^l\}_{l=1}^{G})+\delta }
$$

这里的核心很简单：

**同一个 Query-Window 下，哪个候选 Memory Construction 更好，就增加它的概率。**

对于每个生成 Token：

$$
\rho_t^k=
\frac{
\pi_\theta(o_t^k\mid p,o_{<t}^k)
}{
\pi_{\theta_{\mathrm{old}}}(o_t^k\mid p,o_{<t}^k)
}
$$

采用 PPO clipping：

$$
l_t^k=
\min
\left(
\rho_t^k\hat A^k,
\operatorname{clip}(\rho_t^k,1-\epsilon,1+\epsilon)\hat A^k
\right)
$$

最终目标还包含相对于 SFT Reference Policy $\pi_{\mathrm{ref}}$ 的 KL 约束：

$$
J(\theta) = \mathbb E \left[ \frac{ \sum_k\sum_t l_t^k }{ \sum_k |o^k| } - \beta D_{\mathrm{KL}} (\pi_\theta\Vert\pi_{\mathrm{ref}}) \right]
$$

作者采用 DAPO 风格的 **Group-level Token Normalization**，即整个 group 的 token 一起归一化，防止长 rollout 因逐 rollout averaging 得到较小权重。

训练配置包括：

$$
\epsilon=0.2,\qquad
\beta=0.001
$$

RL 学习率为 $10^{-6}$，训练 3 epochs，最大输入长度 8192，最大生成长度 3072，rollout temperature 0.9。训练使用两张 A800 80GB，完整 RL 运行约 7.5 到 8.5 天。

> [!warning] 从工程成本看并不轻
> 推理阶段的 Memory Processor 只有 4B，但其 RL 训练耗时达到约一周，并需要两张 A800。
>
> 因此论文中的 lightweight 描述主要对应 **inference model size**，训练过程仍属于标准的大模型 RL 工作流。

---

## Reward Design

这一部分是整篇论文中我认为最值得仔细看的地方。

### Overall Reward

**直觉**：好的 Memory 同时需要解决三个问题：

1. 别删掉真正的证据
2. 别保留太多垃圾
3. 压缩后仍然真实、有用

作者定义：

$$
R(o;p) = \mathbb I[o\in V] \left[ (1-\lambda_{\mathrm{qual}})R_{\mathrm{act}} + \lambda_{\mathrm{qual}}R_{\mathrm{qual}} \right]
$$

其中 $V$ 表示结构完全合法的输出集合。

如果 JSON 错了、消息数量不匹配、Keep 没有 compressed content：

$$
R=0
$$

因此整个 Reward 可以看成：

$$
\text{Format Gate}
\times
(
\text{Selection Reward}
+
\text{Compression Reward}
)
$$

---

### Action Reward

**直觉**：漏掉真正需要的证据，比多保留一小段噪声严重得多。

定义：

- $A_g$：黄金消息中被 Keep 的比例
- $A_n$：非黄金消息中被 Drop 的比例

那么：

$$
R_{\mathrm{act}} = \eta A_g+(1-\eta)A_n
$$

其中：

$$
\eta>0.5
$$

正式实验：

$$
\eta=0.915
$$

这基本相当于强烈偏向 Recall：

$$
\mathrm{Cost}(\text{missing evidence})
\gg
\mathrm{Cost}(\text{keeping some noise})
$$

这个选择并非纯经验拍脑袋。

作者做了 Controlled Perturbation：

| Memory Modification | Accuracy |
| ------------------- | -------: |
| 完整黄金证据              |    77.5% |
| 删除一条黄金证据            |    18.0% |
| 删除全部黄金证据            |    10.0% |
| 增加 1 条非黄金消息         |    85.8% |
| 增加 5 条非黄金消息         |    79.2% |
| 增加全部非黄金消息           |    42.5% |

删除一条黄金消息直接造成：

$$
-59.5\ \text{percentage points}
$$

而加入少量非黄金信息没有观察到明显退化。只有大量噪声累积后才产生显著问题。

> [!tip] 很漂亮的 reward justification
> 作者没有简单说 Recall 更重要，而是先测量答案模型对两种错误的敏感性：
>
> $$\text{Evidence Omission}$$
>
> $$\text{Noise Addition}$$
>
> 然后根据这种不对称性设计 Reward。
>
> 这种 **先做 downstream intervention，再设计 reward** 的思路非常值得借鉴。

作者同时明确承认：

$$
\eta=0.915
$$

只是一个偏召回的工作点，并没有证明这是下游成本的最优校准值。

---

### 一个潜在监督漏洞

> [!warning] Non-gold 并不等于 Irrelevant
> 论文自己明确说明：
>
> **非黄金消息仅仅表示 benchmark 没有将其标记为必需证据，它仍可能提供辅助上下文。**
>
> 可 $R_{\mathrm{act}}$ 会奖励 Drop 非黄金消息。
>
> 因此监督信号中天然存在一定 Label Noise。
>
> 作者通过较大的 $\eta$ 降低这种问题的破坏性，但没有从定义上消除它。

这可能解释为什么单纯的 Message-level Keep/Drop Ground Truth 很难完整定义真实 Memory Utility。

---

### Quality Reward

**直觉**：Keep 对了消息还不够。模型可能在压缩过程中歪曲事实，也可能保留一段完全没用的信息。

对于每个被 Keep 的消息：

$$
x\in K
$$

Judge 给出两个三级分数：

$$
f_x\in{0,1,2}
$$

表示 **Faithfulness**；

$$
u_x\in{0,1,2}
$$

表示 **Utility**。

然后：

$$
Q(x) = \mathbb I[f_x>0] \frac{f_x+u_x}{4}
$$

这里最关键的是：

$$
f_x=0
\Rightarrow
Q(x)=0
$$

也就是说，即使一条 hallucinated memory 对答案看起来非常有帮助，只要无法从原消息得到支持，就直接拿零分。

窗口奖励为：

$$
R_{\mathrm{qual}} = \begin{cases} 0,&K=\varnothing\\[4pt] \frac{1}{|K|} \sum_{x\in K}Q(x),&K\neq\varnothing
\end{cases}
$$

> [!tip] Reward 的结构很好理解
> $$R_{\mathrm{act}}$$
> 决定 **保留什么**
>
> $$R_{\mathrm{qual}}$$
> 决定 **怎么保留**
>
> 这两个目标没有揉成一个模糊的 LLM Judge score，因此后续也比较容易诊断。

---

### 一个非常重要的训练信号细节

Quality Judge 在训练时会看到：

- Query
- Source Message
- Gold Answer
- Referenced Reasoning Chain

其中 Gold Answer 和 Referenced Reasoning Chain 只用于 Utility 评价，推理阶段不会出现。

所以 $R_{\mathrm{qual}}$ 本质上使用了非常强的 privileged supervision：

$$
R_{\mathrm{qual}} = f( x,\tilde{x}, q, a^\star, \mathrm{CoT}^\star )
$$

> [!warning] 这可能是方法推广到真实 Agent Memory 时最大的训练障碍
> LongMemEval 能够给出 Gold Evidence、Gold Answer 和 Referenced Reasoning Chain。
>
> 真实长期 Agent 数据通常没有这些信息。
>
> 因此论文证明了 **这种监督信号可以训练出有效 Query-conditioned Memory Processor**，但如何廉价获得这种监督仍然没有解决。

质量奖励 Judge 本身经过了人工校验。100 个随机 rollout 上：

- Faithfulness 与人工标签完全一致率：91%
- Utility：85%

Utility 明显更难评判，这也符合直觉，因为它涉及当前信息对未来推理过程是否有帮助。

---

### Reward Curriculum

**直觉**：模型连合法 JSON 都输出不了时，调用昂贵的 LLM Judge 没什么价值。

训练分为两个阶段。

第一阶段：

$$
\lambda_{\mathrm{qual}}=0
$$

只使用：

$$
\text{Format Gate}+R_{\mathrm{act}}
$$

直到验证集格式合法率达到：

$$
\tau=0.99
$$

第二阶段：

$$
\lambda_{\mathrm{qual}}=0.5
$$

同时优化：

$$
R_{\mathrm{act}}+R_{\mathrm{qual}}
$$

![[figure-1.png]]

Figure 1 下半部分的 Phase 1 和 Phase 2 对应的正是这一 curriculum。

> [!tip] 这是一个很合理的训练工程设计
> 训练早期先优化廉价、确定性的规则信号。
>
> 等模型输出进入可用区域后，再引入昂贵且有噪声的 LLM Judge。
>
> 可以理解为：
>
> $$\text{Learn syntax}\rightarrow\text{Learn selection}\rightarrow\text{Learn semantic quality}$$

---

## Experiments

### Dataset

LongMemEval 一共 500 个问题。

作者由于需要训练 Memory Processor，将它重新划分为：

$$
360/40/100
$$

分别作为：

$$
\text{Train}/\text{Validation}/\text{Test}
$$

并按照问题类型分层抽样。测试集完全不参与模型选择。

LoCoMo 完全不参与 SFT 和 RL，仅用于域外测试，共评估 314 个问题。

> [!warning] 阅读表 1 时一定要记住这个设置
> LazyMem 的 LongMemEval 结果属于 **in-domain trained setting**。
>
> 它并不是在官方完整 500 样本设置上的 training-free evaluation。
>
> 作者已经重新在自己的 100-question Test Split 上跑了所有 Baseline，因此论文内部比较成立；这些数字无法直接拿去与其他论文报告的完整 LongMemEval 数字横向比较。

---

### Main Results

LongMemEval：

| Method          |       LJ |
| --------------- | -------: |
| Oracle Turn     |     0.82 |
| Oracle Session  |     0.84 |
| RAG Top20       |     0.71 |
| RAG Top50       |     0.75 |
| LightMem        |     0.79 |
| StructMem       |     0.82 |
| Mem0            |     0.64 |
| MemoryBank      |     0.68 |
| MemSkill        |     0.57 |
| MemT            |     0.61 |
| NanoMemory      |     0.80 |
| **LazyMem-4B**  | **0.85** |
| **LazyMem-32B** | **0.93** |

最显眼的结果是：

$$
0.93 > 0.84
$$

LazyMem-32B 甚至超过 Oracle Session。

乍看很奇怪，但这里需要仔细理解 Oracle 的定义。

> [!warning] Oracle Session 并不是真正的上界
> Oracle Turn 只提供黄金轮次。
>
> Oracle Session 只提供包含黄金轮次的完整 Session。
>
> 多会话问题可能需要分布在多个 Session 的证据，因此单个 Oracle Session 本身就可能信息不足。
>
> 同时完整 Session 又会携带噪声。
>
> LazyMem 可以从多个 Session 中搜集并重新组合证据，因此超过这个 Oracle 是完全可能的。

所以这里更准确的理解是：

**LazyMem 超过了 benchmark 提供的局部黄金上下文构造方式。**

不能据此得出它超过了完整的完美 Memory。

---

### 一个值得关注的异常：Preference

Single-session Preference：

$$
\mathrm{LazyMem}=0.67
$$

而 StructMem 和 Mem0：

$$
0.83
$$

作者推测这一类推荐问题更适合长期维护的 User Profile。

> [!warning] 这其实暴露了一条很有价值的边界
> LazyMem 非常适合 **fact retrieval / evidence aggregation**。
>
> Preference Memory 往往要求持续形成高层用户表示：
>
> $$H\rightarrow z_{\mathrm{user}}$$
>
> 每次 Query 临时从原始历史拼装证据未必是最优表示。
>
> 这说明 Persistent Memory 与 Query-time Memory Construction 很可能应该共存。

---

### Domain Generalization

LoCoMo 没有用于训练。

结果：

$$
\mathrm{LazyMem\text{-}32B}=0.69
$$

$$
\mathrm{LazyMem\text{-}4B}=0.68
$$

其中 NanoMemory 为 0.68。

这部分实验比 LongMemEval 更能说明其方法的通用性，因为这里没有目标域训练。

不过自动 Judge 在 LoCoMo 上存在明显噪声。人工审核发现：

- 3768 个 Judge label 中 294 个错误
- 总噪声率 7.80%
- 293 个为 False Negative
- 修正后 LazyMem-4B 与 LazyMem-32B 都为 0.75

> [!note] 指标解读
> LoCoMo 的原始 0.68 和 0.69 不应该被过度解读成精确的 1 个百分点差异。
>
> 人工修正后两者实际上并列 0.75。

---

## Efficiency

![[figure-2.png]]

Figure 2 是论文第二个真正重要的实验。

LazyMem-4B：

$$
213\ \text{memory tokens}
$$

对应：

$$
LJ=0.85
$$

RAG Top50：

$$
14628\ \text{tokens},\qquad LJ=0.75
$$

StructMem：

$$
4483\ \text{tokens},\qquad LJ=0.82
$$

因此 LazyMem-4B 相比 RAG Top50 将答案上下文减少：

$$
68.7\times
$$

相比 StructMem：

$$
21.0\times
$$

这其实非常说明问题。

**Memory Processor 的主要价值未必只是提高 QA Accuracy，它显著提高了 Information Density。**

可以粗略写成：

$$
\text{Memory Efficiency}
\approx
\frac{\text{Useful Evidence}}
{\text{Context Tokens}}
$$

LazyMem 正是在优化这个比例。

---

### Latency

这里的数据需要冷静看。

| Method         | Mean latency |
| -------------- | -----------: |
| MemT           |      20.38 s |
| LightMem       |      27.69 s |
| RAG Top50      |      30.24 s |
| StructMem      |      34.55 s |
| **LazyMem-4B** |  **40.86 s** |
| NanoMemory     |      55.35 s |

所以 LazyMem 的 Query-Time 延迟仍然不低。

它真正证明的是：

$$
\text{LazyMem}

>

\text{NanoMemory}
$$

在相同 Query-time Construction 范式下：

$$
55.35s\rightarrow40.86s
$$

同时：

$$
0.80\rightarrow0.85
$$

> [!warning] 图 2 的 latency 结论需要带条件理解
> LazyMem 可以同时发起最多 64 个 Window Request，因此获得了天然的 Query 内并行能力。
>
> NanoMemory 一次处理完整 Recall Pool，没有同样的并行结构。作者明确说明延迟结果包含这一结构优势。
>
> 真实在线服务中，如果同时存在大量用户请求，这种 Window-level Parallelism 会转化为更高的 GPU 并发压力。
>
> 因此单查询 Latency 与系统级 Throughput 仍需要进一步区分。

另外，写入时方法的 Offline Memory Construction Cost 没有计入延迟。

因此表中的 Query-Time Latency 很公平地描述在线体验，但不能代表整个生命周期总计算量。

---

## Ablation Study

### SFT 与 RL 到底分别贡献了什么

这是全文最干净的一组实验。

| Model           | LongMemEval Gate Fmt. | LongMemEval LJ | LoCoMo Gate Fmt. | LoCoMo LJ |
| --------------- | --------------------: | -------------: | ---------------: | --------: |
| Qwen3-4B Prompt |                 0.397 |           0.41 |            0.622 |      0.56 |
| + SFT           |                 0.963 |           0.73 |            0.985 |      0.63 |
| + RL            |             **0.997** |       **0.85** |        **0.997** |  **0.68** |

这里可以非常明确地得到：

### SFT

LongMemEval：

$$
0.41\rightarrow0.73
$$

同时格式正确率：

$$
0.397\rightarrow0.963
$$

所以大量收益来自 **模型终于学会稳定执行 Memory Editing Protocol**。

### RL

随后：

$$
0.73\rightarrow0.85
$$

格式正确率几乎没有变化：

$$
0.963\rightarrow0.997
$$

因此 RL 的提升主要来自：

**Keep/Drop 与 Compression 本身质量提高。**

> [!tip] 这个 Ablation 很有说服力
> 它能够把：
>
> $$\text{format learning}$$
>
> 与
>
> $$\text{memory policy learning}$$
>
> 比较清楚地分离出来。
>
> 如果 RL 后只有格式率提高，整个 RL Story 会弱很多。现在的结果至少支持 Reward 确实在优化语义行为。

---

## Failure Analysis

![[figure-3.png]]

Figure 3 把失败分成三个阶段：

$$
\text{Retrieval Miss}
\rightarrow
\text{Editing Loss}
\rightarrow
\text{QA Reasoning Error}
$$

这个分类非常重要，因为它告诉我们下一步到底应该改哪个模块。

### LongMemEval

Top-50 覆盖率：

$$
\mathrm{All@50}=0.98
$$

人工归因的数据中已经没有 Retrieval Miss。

4B：

$$
15\ \text{errors} = 13\ \text{editing} + 2\ \text{QA}
$$

32B：

$$
7\ \text{errors} = 7\ \text{editing}
$$

这说明 LongMemEval 上当前瓶颈已经很明确：

**Memory Editing，而不是 Retrieval。**

---

### 为什么 4B 在 Multi-session 上掉得最严重

32B：

$$
MS=0.93
$$

4B：

$$
MS=0.67
$$

编辑错误：

$$
2\rightarrow9
$$

关键是，这些错误很多都没有发生 Keep/Drop 错误。

模型正确 Keep 了消息，只是在 $\tilde{x}$ 中删掉了一些局部看来不重要的信息。跨 Session 聚合时，这些信息恰好构成必要操作数。

论文给出的代表性例子非常典型：

- 保留了预批准金额 $350k
- 丢掉售价 $325k
- 最终无法计算 $25k 差值

另一个例子：

- 保留 HelloFresh 40% 折扣
- 丢掉 UberEats 20% 折扣
- 最终无法比较

> [!warning] 这是当前 Reward 最重要的漏洞
> $R_{\mathrm{qual}}$ 是逐消息评价的。
>
> 某条 Compression 单独看来：
>
> - Faithful
> - Useful
>
> 仍然可能缺少完成 **跨消息联合推理** 所需的信息。
>
> 当前 Reward 没有直接检查：
>
> $$\operatorname{Completeness}
> \left(
> \bigcup_{x\in K}\tilde{x}
> \right)
> $$
>
> 也就是最终 Memory Set 是否完整包含了解题所需的全部变量。

这个失败模式与论文的主要结果完全一致，因此我认为它可能比继续扩大模型更值得研究。

---

### Temporal Editing

另一类很典型的 Editing Loss 是时间信息：

- 将 **上周** 错误改写成消息日期
- 将 **昨天** 错误改写成当前消息日期
- 混淆 report time 与 event time

这里暴露了 Compression 的一个普遍问题：

原始表达：

$$
\text{relative temporal relation}
$$

压缩后模型试图将其规范化成：

$$
\text{absolute timestamp}
$$

一旦解析错误，原始信息已经丢失。

> [!question] 为什么一定要改写时间表达？
> 对这一类信息，更安全的 Memory Construction 可能同时保留：
>
> $$\text{original phrase}+\text{timestamp anchor}$$
>
> 当前模型允许自由压缩，因此有机会在 Construction 阶段自行引入错误。

---

## 我对这篇论文的整体判断

> [!tip] 最强的地方
> 论文的核心直觉非常干净：
>
> **未来 Query 未知时，任何有损 Memory Construction 都存在提前丢失未来有用信息的风险。**
>
> 所以原始记录长期保存，Query 出现后再进行任务条件化压缩。
>
> 这个思路直接击中了长期 Memory 中非常根本的信息瓶颈。

同时，论文实验也相当完整：

- Query-time vs Write-time paradigm
- 4B vs 32B
- Context Tokens
- Latency
- Window Radius
- SFT vs RL
- Reward Diagnostics
- Human Error Attribution
- Reward Judge Validation
- Final Judge Noise Audit

很多论文做到 0.85 就结束了，这篇继续追踪 **0.15 到底丢在哪里**，这一点很有价值。

---

> [!warning] 第一个核心局限：Memory 被简化成 Query-time Evidence Editing
> LazyMem 完整保存原始 History。
>
> 它没有回答长期 Agent 中另外几个问题：
>
> - History 无限增长后怎么管理
> - 什么信息应该遗忘
> - 如何形成跨任务复用的抽象知识
> - 用户画像如何持续演化
> - 一段记忆如何被主动修改
>
> 当前框架的主要目标非常明确：
>
> $$\text{Long History}
> +
> \text{Current Query}
> \rightarrow
> \text{Compact Evidence}
> $$

---

> [!warning] 第二个核心局限：训练依赖强监督
> 当前训练直接利用：
>
> - Gold Evidence
> - Gold Answer
> - Referenced Reasoning Chain
>
> 这些信号使 Reward 非常强，但普通 Agent trajectory 通常没有这样的标注。
>
> 所以这篇论文已经解决了：
>
> **有高质量监督时，如何训练一个有效的小型 Memory Constructor。**
>
> 大规模真实 Memory 数据如何生成对应监督，论文还没有给出答案。

---

> [!warning] 第三个核心局限：LongMemEval 属于同域训练
> LongMemEval 500 个问题被拆成 360/40/100。
>
> 因此 0.85 是训练过该 benchmark 分布后的结果。
>
> 好的一面是 LoCoMo 完全零样本，仍然得到较强结果。
>
> 对这篇论文的泛化能力判断，我会更重视 LoCoMo。

---

> [!warning] 第四个核心局限：压缩仍然是不可逆的信息瓶颈
> 作者解决了 Write-time Compression 的永久损失，却在 Query-time 又执行了一次有损 Compression。
>
> Query 已知确实降低了风险，但 Multi-session 结果说明风险依然存在。
>
> 当前最主要的失败已经从：
>
> $$\text{Did I retrieve the evidence?}$$
>
> 转移到：
>
> $$\text{Did I preserve every detail required for joint reasoning?}$$

---

## Future

### 1. 从 Utility Reward 进一步走向 Evidence Completeness Reward

这是论文错误分析最直接指向的方向。

当前：

$$
R_{\mathrm{qual}} = \frac{1}{|K|} \sum_x Q(x)
$$

每条消息分别判断 Faithfulness 和 Utility。

Multi-session 的真实需求更接近：

$$
R_{\mathrm{global}} = f( q, {\tilde{x}_1,\ldots,\tilde{x}_n} )
$$

需要判断整个 Memory Set 是否共同覆盖完成答案所需的变量。

例如问题需要：

$$
a-b
$$

Memory 中只留下 $a$ 时，$a$ 本身完全忠实且有用，但整个 Evidence Set 明显不完整。

> [!tip] 我认为这是论文自身最自然的下一步
> 从：
>
> **message-wise useful compression**
>
> 进一步训练：
>
> **set-wise sufficient evidence construction**

论文在 MS Failure 中已经明确暴露了这一问题。

---

### 2. 为 Temporal Information 设计保真约束

时间错误具有很强的结构性。

可以避免模型随意将：

$$
\text{昨天}
$$

直接重写成某个日期，而是让 Memory 保存：

$$
(\text{event},\text{relative expression},\text{anchor timestamp})
$$

至少从论文的失败案例看，时间压缩值得单独设计保真机制。

---

### 3. Retrieval 与 Construction 应该联合优化

LongMemEval 中 Retriever 已经达到：

$$
0.98
$$

覆盖率，所以论文主要研究 Construction。

LoCoMo 中：

$$
0.89
$$

左右的覆盖率立即变成主要瓶颈，尤其 Multi-hop 问题。

因此更一般的系统最终需要：

$$
q
\rightarrow
\text{Retrieve}
\rightarrow
\text{Construct}
\rightarrow
\text{detect missing evidence}
\rightarrow
\text{Retrieve again}
$$

固定 Top-50 一次性召回会限制进一步提升。

---

### 4. Persistent Profile 与 Lazy Construction 的混合 Memory

SSP 的结果很有启发性：

$$
0.67<0.83
$$

LazyMem 在 Preference Question 上落后 StructMem 和 Mem0。

这提示两类 Memory 可能承担不同功能：

$$
M=
M_{\mathrm{persistent}}
+
M_{\mathrm{query}}
$$

其中：

- $M_{\mathrm{persistent}}$ 保存长期稳定的用户偏好、画像和高层抽象
- $M_{\mathrm{query}}$ 从原始历史中按当前任务动态恢复具体证据

这种混合方式可以保留 LazyMem 对细节的高保真，同时避免每次重新从原始历史推断长期稳定状态。

---

### 5. 减少 Gold Evidence 与 Gold Reasoning Chain 依赖

这是从 benchmark 方法走向真实 Agent Memory 最需要解决的问题。

当前 Reward 已经说明了需要优化什么：

$$
\text{Recall}
+
\text{Faithfulness}
+
\text{Utility}
+
\text{Completeness}
$$

接下来的问题变成：

**这些信号能否从 Agent 自身的 downstream outcome 中获得，而无需人工提供 Gold Evidence？**

当前论文没有给出答案，但其 Reward decomposition 已经提供了相当清楚的问题定义。

---

### 6. Query-time Parallelism 的系统效率

LazyMem 将一个大上下文任务拆成大量独立小窗口：

$$
W_1,W_2,\ldots,W_P
$$

它们可以：

$$
\pi_\theta(W_1),\ldots,\pi_\theta(W_P)
$$

并行运行。

这使单 Query latency 很漂亮，但最大并发达到 64 个 Window Request。

未来真正部署时需要进一步研究：

$$
\text{Latency}
+
\text{Throughput}
+
\text{GPU Memory}
+
\text{Concurrent Users}
$$

之间的整体系统权衡。

---

### 7. 进一步减少自由生成式 Compression

当前最大剩余错误来自 Editing Loss。

这提示一个更根本的问题：

**Memory Construction 是否一定需要自由文本生成？**

当前的：

$$
x\rightarrow\tilde{x}
$$

允许模型重新表述信息，因此天然存在：

- operand omission
- temporal distortion
- entity omission
- unsupported generalization

论文已经通过 Faithfulness Reward 控制这些问题，但依然无法完全消除。

从论文自身的错误证据出发，一个很自然的研究问题是：

$$
\text{How much compression can be extractive?}
$$

以及：

$$
\text{When should the system preserve raw spans?}
$$

这可能成为进一步降低 Editing Loss 的关键。
