# δ-mem: Efficient Online Memory for Large Language Models

arXiv-2026

**核心一句话**：δ-mem 将历史信息持续压缩进一个极小的关联记忆矩阵，并让该矩阵在推理时直接修改 Transformer 的注意力计算，从而在冻结主干模型、删除显式历史文本的情况下继续利用过去的信息。

---

## Key Contribution

- **We propose δ-mem, a memory mechanism that augments a frozen full-attention backbone with a compact online state of associative memory, enabling historical information to be dynamically maintained and directly coupled with the backbone’s attention computation.**
- **We show that an extremely small memory state, implemented as an $8 \times 8$ matrix, can retain useful historical signals through OSAM and help the model recover context-relevant information even after explicit history is removed.**
- **We evaluate δ-mem on multiple memory-heavy and general capability benchmarks with significant gains on MemoryAgentBench and LoCoMo, without full fine-tuning or replacing the backbone architecture.**

作者的核心贡献可以拆成三个层次。

- **存储形式**：历史信息被压缩为固定大小的在线关联记忆状态 $S$，状态大小不会随历史长度增加。
- **使用方式**：记忆不会被还原成文本，也不会作为额外 token 拼接到输入中。记忆读出直接生成注意力模块的修正量。
- **更新方式**：状态使用带遗忘门的 delta rule 在线更新。每次写入主要记录当前状态尚未预测正确的残差信息。

> [!tip] 真正值得关注的设计
> δ-mem 将记忆问题拆成了两个独立问题：
>
> 1. 历史信息如何压缩成持续演化的状态
> 2. 压缩后的状态如何进入 LLM 的内部计算
>
> 很多记忆方法只处理第一个问题，然后通过文本检索或外部网络将结果重新送入模型。δ-mem 的主要价值在于给出了一个较紧密的内部接口：**记忆状态直接生成 attention correction**。

我注意到，论文反复强调 $8 \times 8$ 状态，但这里需要区分两个概念：

- 在线记忆状态本身只有 $64$ 个标量
- 整套 δ-mem 仍然包含每层的投影矩阵和修正矩阵，在 Qwen3-4B 上共有约 $4.87$M 个可训练参数

因此，论文证明的是**运行时状态可以很小**，并未证明整个记忆模块只需要几十个参数。

---

## Method

### Problem Formulation: Memory State and Memory Steering

**直觉**：一个记忆机制既要决定过去的信息存在哪里，也要决定这些信息怎样影响当前推理。

作者用两个维度重新整理现有方法：

- **Memory state**：历史信息以什么形式保存
- **Memory steering**：保存的信息通过什么路径影响主干模型

基于这两个维度，作者将现有方法划分为三类。

#### Textual Memory Mechanisms

文本记忆将历史保存为原始文本、摘要或检索文档，然后重新放入上下文。

优点是接口简单，几乎不需要修改模型结构。主要问题包括：

- 历史长度受上下文窗口约束
- 检索结果可能包含噪声
- 文本摘要会丢失信息
- 主干模型未必能有效利用很长的输入

#### Outside-Channel Memory Mechanisms

外部通道记忆将 hidden state 或其他潜在表示保存在外部模块中，再通过独立 reader、retriever 或 side network 融合回模型。

这类方法可以保留比文本摘要更丰富的表示，但需要额外处理：

- 如何检索
- 如何对齐旧表示与当前表示
- 如何将外部结果融合进主干网络

#### Parametric Memory Mechanisms

参数记忆将信息写入 prefix、adapter、LoRA 或局部模型权重。

这种表示可以高效控制冻结模型，但通常需要执行优化或模型编辑，并且写入后相对静态，不适合连续、逐步变化的交互历史。

> [!question] 分类是否完全成立
> 作者将 Context2LoRA 和 MemGen 归入 parametric memory，并强调其静态性。但部分测试时参数更新方法本身也可以在线变化。
>
> 更准确的区分可能是：
>
> - **persistent parameter update**
> - **instance-conditioned recurrent state**
>
> δ-mem 的关键特征是第二种。它维护的是随实例和历史变化的状态，而投影参数保持固定。

---

### Overall Architecture

**直觉**：先用当前输入查询过去的状态，再让查询结果影响当前注意力，最后把当前信息写入记忆。

δ-mem 在每个位置执行三个步骤：

1. **Read**：从旧状态 $S_{t-1}$ 读取与当前输入相关的记忆
2. **Steer**：用读出向量修正当前 attention
3. **Write**：根据当前信息更新状态，得到 $S_t$

![[figure1.png]]

图 1 中可以看到两条并行路径。

左侧是冻结的 Transformer attention：

$$
x_t \rightarrow q_t^0,k_t,v_t \rightarrow \operatorname{Attn} \rightarrow a_t
$$

右侧是 δ-mem：

$$
x_t
\rightarrow
q_t^m,k_t^m,v_t^m
\rightarrow
S_{t-1}
\rightarrow
r_t
\rightarrow
\Delta q_t,\Delta o_t
$$

记忆读出 $r_t$ 同时作用于 attention 的输入端和输出端：

- $\Delta q_t$ 改变模型当前在显式上下文中寻找什么
- $\Delta o_t$ 向 attention 输出补充历史相关信号

完成当前 attention 计算后，模型再使用 $k_t^m$ 和 $v_t^m$ 更新记忆状态。

> [!tip] Read before Write
> 作者先读取 $S_{t-1}$，再写入当前 token。
>
> 这个顺序避免当前 token 立即从自己刚写入的状态中读取信息，也使 $r_t$ 可以明确解释为**过去历史对当前计算的影响**。

---

### Online State of Associative Memory

**直觉**：将记忆矩阵视为一个小型的在线映射器，输入 memory key，输出与其关联的 memory value。

设在线状态为：

$$
S_t \in \mathbb{R}^{r \times r}
$$

对于当前 memory key $k_t \in \mathbb{R}^r$ 和 memory value $v_t \in \mathbb{R}^r$，作者希望状态满足：

$$
S_t k_t \approx v_t
$$

旧状态对当前 value 的预测为：

$$
\hat v_t = S_{t-1}k_t
$$

其中：

- $k_t$ 表示当前信息在记忆空间中的地址或条件
- $v_t$ 表示希望在该地址下保存的内容
- $S_{t-1}k_t$ 表示旧记忆对当前信息的预测
- $v_t-S_{t-1}k_t$ 表示当前状态缺失或预测错误的部分

作者将状态学习写成在线回归：

$$
\mathcal{L}_t(S)
================

\frac{1}{2}
\left|Sk_t-v_t\right|_2^2
$$

对该损失执行一步梯度下降，可以得到：

$$
S_t
===

S_{t-1}
+
\beta_t
\left(
v_t-S_{t-1}k_t
\right)
k_t^\top
$$

这里的更新是一个外积：

$$
\left(
v_t-S_{t-1}k_t
\right)k_t^\top
$$

它只沿当前 key 的方向修改映射。

当旧状态已经能够正确预测 $v_t$ 时：

$$
v_t-S_{t-1}k_t \approx 0
$$

状态几乎不再更新。重复出现、已经学会的关联不会持续叠加。

> [!tip] Delta rule 的核心
> 普通外积记忆可能直接执行：
>
> $$
> S_t=S_{t-1}+v_tk_t^\top
> $$
>
> 这种写法会不断累积重复信息。
>
> δ-mem 写入的是预测残差：
>
> $$
> v_t-S_{t-1}k_t
> $$
>
> 因此，它具有简单的误差修正能力。

---

### Memory Projection

**直觉**：主干模型的 hidden state 维度太高，作者先将其投影到一个很小的关联记忆空间。

对于 Transformer 某一层的 hidden state：

$$
x_t\in\mathbb{R}^d
$$

作者生成三种 memory representation：

$$
q_t^m
=====

\operatorname{L2Norm}
\left(
\tanh(W_q^m x_t)
\right)
$$

$$
k_t^m
=====

\operatorname{L2Norm}
\left(
\tanh(W_k^m x_t)
\right)
$$

$$
v_t^m
=====

W_v^m x_t
$$

其中：

$$
q_t^m,k_t^m,v_t^m\in\mathbb{R}^r
$$

三者分别承担不同功能：

- $q_t^m$：当前输入用什么方向查询旧记忆
- $k_t^m$：当前信息写到记忆空间的什么方向
- $v_t^m$：当前信息希望保存什么内容

作者对 query 和 key 使用 $\tanh$ 与 L2 normalization，主要为了控制递归状态的数值稳定性。

由于状态不断执行：

$$
S_tk_t^m
$$

如果 $k_t^m$ 的范数随时间漂移，写入幅度和预测幅度也会随之变化。将 key 和 query 归一化后，状态更新更接近方向关联，而较少受到向量尺度影响。

> [!question] 为什么 value 不归一化
> 作者只对 $q_t^m$ 和 $k_t^m$ 归一化，$v_t^m$ 保留线性投影。
>
> 一个合理推测是，value 的幅度需要表达写入内容的强度。但论文没有分析 value scale 对长期稳定性的影响，也没有给出 projection normalization 的消融实验。

---

### Gated Delta Update

**直觉**：固定容量状态持续接收新信息时，需要在保留旧信息与吸收新信息之间进行动态平衡。

作者从当前 hidden state 生成写入门：

$$
\beta_t
=======

\sigma(W_\beta x_t+b)
$$

并定义保留门：

$$
\lambda_t=1-\beta_t
$$

其中：

$$
\beta_t,\lambda_t\in\mathbb{R}^r
$$

完整更新公式为：

$$
S_t
===

\operatorname{Diag}(\lambda_t)S_{t-1}
+
\operatorname{Diag}(\beta_t)
\left(
v_t^m-S_{t-1}k_t^m
\right)
(k_t^m)^\top
$$

展开后：

$$
\begin{aligned}
S_t
={}&
\operatorname{Diag}(\lambda_t)S_{t-1} \
&-
\operatorname{Diag}(\beta_t)
S_{t-1}k_t^m(k_t^m)^\top \
&+
\operatorname{Diag}(\beta_t)
v_t^m(k_t^m)^\top
\end{aligned}
$$

三个部分分别表示：

1. 保留旧状态
2. 移除旧状态在当前 key 方向上的错误预测
3. 将新的 value 写入相同方向

按状态矩阵的第 $i$ 行展开：

$$
s_t^{(i)}
=========

\lambda_{t,i}s_{t-1}^{(i)}
+
\beta_{t,i}
\left(
v_{t,i}^m-s_{t-1}^{(i)}k_t^m
\right)
(k_t^m)^\top
$$

每一行都有独立的 $\beta_{t,i}$，因此不同记忆维度可以表现出不同更新速度。

> [!warning] 保留门和写入门被强耦合
> 作者设置：
>
> $$
> \lambda_t=1-\beta_t
> $$
>
> 这使遗忘强度和写入强度完全绑定。
>
> 当 $\beta_t$ 较大时，模型同时执行强写入和强遗忘。模型无法表达以下状态：
>
> - 保留大量旧信息，同时强力修正当前方向
> - 大量遗忘旧状态，但暂时不写入当前信息
>
> 独立的 retention gate 和 update gate 可能提供更大的控制空间。论文没有比较耦合门与双门结构。

> [!warning] 全局衰减可能伤害无关记忆
> 第一项为：
>
> $$
> \operatorname{Diag}(\lambda_t)S_{t-1}
> $$
>
> 它会对整行状态进行衰减，影响范围超过当前 key 方向。
>
> 因此，一次写入可能同时削弱与当前信息无关的关联。固定容量下，这种衰减有助于防止状态爆炸，但也可能形成隐式、难以控制的遗忘。

---

### Reading from the Online State

**直觉**：当前输入生成一个 memory query，并从固定状态中直接读出历史相关信号。

读取过程为：

$$
r_t
===

S_{t-1}q_t^m
$$

其中：

$$
r_t\in\mathbb{R}^r
$$

标准 attention 需要将当前 query 与显式上下文中的所有 key 比较：

$$
\operatorname{softmax}(q_tK^\top)V
$$

δ-mem 直接计算：

$$
S_{t-1}q_t^m
$$

读取成本只与 $r$ 有关：

$$
O(r^2)
$$

它不会随历史 token 数增加。

这里的 $r_t$ 不是某段历史文本，也不是一个被检索出来的旧 hidden state。它是所有历史写入经过压缩、干涉和遗忘后形成的连续读出。

> [!note] 与 KV cache 的区别
> KV cache 保留每个历史 token 的 key 和 value：
>
> $$
> {(k_1,v_1),\ldots,(k_t,v_t)}
> $$
>
> δ-mem 将这些关联压缩进单一矩阵：
>
> $$
> S_t\in\mathbb{R}^{r\times r}
> $$
>
> KV cache 提供细粒度、可寻址的历史记录，存储成本随序列增长。δ-mem 提供固定成本的近似关联映射，但不同记忆会共享同一状态空间并产生干涉。

---

### Steering Attention through Low-Rank Corrections

**直觉**：记忆读出无需还原为文本，只需要告诉当前 attention 应该怎样调整查询和输出。

作者将 $r_t$ 映射成两个修正量：

$$
\Delta q_t=W_q^\Delta r_t
$$

$$
\Delta o_t=W_o^\Delta r_t
$$

原始 query 为：

$$
q_t^0=W_Qx_t
$$

加入记忆修正后：

$$
\tilde q_t
==========

q_t^0+\frac{\alpha}{r}\Delta q_t
$$

随后使用冻结的 key 和 value 执行 attention：

$$
a_t
===

\operatorname{Attn}
\left(
\tilde q_t,K_{\leq t},V_{\leq t}
\right)
$$

最后在 attention 输出端加入另一项修正：

$$
\tilde y_t
==========

a_t+\frac{\alpha}{r}\Delta o_t
$$

其中 $\alpha/r$ 类似 LoRA 中的缩放项，用于控制低秩分支的幅度。

两种修正承担不同作用。

#### Query-side Correction

$$
\Delta q_t
$$

它改变当前 token 对显式上下文的注意力分布。过去的记忆可以影响当前模型寻找哪些 token。

例如，历史状态可能保存了某个用户偏好。当前问题出现相关 cue 后，$\Delta q_t$ 可以让注意力更偏向与该偏好一致的当前证据。

#### Output-side Correction

$$
\Delta o_t
$$

它直接为 attention 输出增加一个历史条件化的 residual。即使相关历史已经从显式上下文中删除，记忆仍可向当前表示注入信息。

> [!tip] 动态 LoRA 式接口
> $W_q^\Delta$ 和 $W_o^\Delta$ 在训练结束后保持固定，但它们的输入 $r_t$ 来自动态状态 $S_{t-1}$。
>
> 因此，同一套参数会随历史不同生成不同的修正量：
>
> $$
> \Delta q_t
> ==========
>
> W_q^\Delta S_{t-1}q_t^m
> $$
>
> 这可以理解为一种由在线状态条件化的低秩 attention adapter。

> [!question] 低秩的定义需要更谨慎
> 单个时间步的修正量来自 $r$ 维瓶颈，因此映射的有效秩受 $r$ 限制。
>
> 但论文中的 low-rank correction 并不是直接生成完整权重更新：
>
> $$
> \Delta W
> $$
>
> 它生成的是当前 token 的 activation correction：
>
> $$
> \Delta q_t,\Delta o_t
> $$
>
> 因此，它更接近**低维状态驱动的激活修正**。

---

### Writing Granularity

**直觉**：记忆更新的基本单位会决定状态保存的是局部 token 信息、消息级语义，还是多组并行关联。

作者考察三种写入策略。

![[figure1.png]]

#### Token-State Write

每个 token 都更新一次状态：

$$
S_t
===

\operatorname{Update}(S_{t-1},x_t)
$$

优点：

- 保留细粒度变化
- 能记录具体词项和局部事实
- 与自回归生成过程自然对齐

问题：

- 标点、模板词和格式 token 也会触发写入
- 相似表达可能反复覆盖状态
- 长序列中更新次数很高
- 状态容易被短期局部信号影响

实验中，TSW 在 Qwen3-4B 上取得最高综合平均分 $51.66$，并在 HotpotQA 上表现最好。

#### Sequence-State Write

作者将一个消息或语义段内的 hidden state 平均：

$$
\bar x^{(j)}
============

\frac{1}{|M^{(j)}|}
\sum_{t\in M^{(j)}}x_t
$$

随后每个 segment 只更新一次：

$$
S^{(j)}
=======

\operatorname{Update}
\left(
S^{(j-1)},\bar x^{(j)}
\right)
$$

它减少了重复写入，并让状态变化更平滑。

> [!warning] Mean pooling 会删除顺序信息
> 两个包含相同 token、排列顺序不同的消息，可能得到相近的平均表示。
>
> 对时间顺序、否定关系、实体绑定和局部细节敏感的记忆任务中，简单平均可能损失关键信息。

SSW 在 Qwen3-8B 上表现最好。作者推测，大模型已有较强推理能力，message-level 写入可以过滤 token 噪声。

我认为这个解释目前只是一种事后归因。论文没有测量状态噪声，也没有比较 mean pooling、last-token pooling、attention pooling 或专门的 segment encoder。

#### Multi-State Write

作者维护多个并行子状态：

$$
\mathcal{S}_t
=============

\left{
S_t^{(1)},\ldots,S_t^{(N)}
\right}
$$

每个状态独立更新：

$$
S_t^{(i)}
=========

\operatorname{Update}^{(i)}
\left(
S_{t-1}^{(i)},x_t
\right)
$$

各子状态分别读出：

$$
r_t^{(i)}
=========

S_{t-1}^{(i)}q_t^{m,(i)}
$$

最后拼接：

$$
r_t
===

\operatorname{Concat}
\left(
r_t^{(1)},\ldots,r_t^{(N)}
\right)
$$

作者希望不同状态分别承载事实、偏好、任务进度或局部事件，从而降低单一状态内的干涉。

> [!question] 子状态为什么会形成分工
> 论文没有显式 router、state assignment、orthogonality loss 或 diversity objective。
>
> 所有子状态都接收同一个 $x_t$。它们可能通过随机初始化和任务损失形成差异，也可能学习出高度冗余的表示。
>
> 作者将 MSW 的收益解释为减少干涉，但缺少以下证据：
>
> - 子状态之间的相似度
> - 不同子状态保存的信息类型
> - 屏蔽某一子状态后的功能变化
> - 不同 state 数量的容量曲线

MSW 在 LoCoMo 和部分小模型实验中表现较好，这说明增加并行容量确实有用。但目前很难确定收益来自**结构化分工**，还是单纯来自更多参数和更大的总状态。

---

### Training Objective

**直觉**：训练时主动移除历史文本，迫使模型只能通过记忆状态完成回答。

设历史 context 为 $C$，query 为 $Q$，response 为 $Y$。

首先，模型将 $C$ 写入状态：

$$
C\rightarrow S_C
$$

随后，主干模型在预测阶段只接收：

$$
Q,Y_{<j}
$$

历史 $C$ 不再作为显式 backbone input。

训练目标为：

$$
\mathcal{L}_{\mathrm{SFT}}
==========================

*

\sum_{j=1}^{|Y|}
\log
p_{\phi,\theta}
\left(
y_j
\mid
Q,y_{<j},S_C
\right)
$$

其中：

- $\phi$：冻结的 backbone 参数
- $\theta$：δ-mem 的可训练参数
- $S_C$：由历史 context 在线构造的状态

训练数据来自 QASPER 的 $2219$ 个样本，训练一个 epoch。主干模型最大输入长度为 $512$，记忆写入预算为 $8192$ token。

> [!tip] 最关键的训练信号
> 这篇论文最有效的设计可能并非 delta rule 本身，而是**context removal training**。
>
> 作者将历史写入状态后，从主干输入中删除历史。此时模型无法绕过记忆模块直接从上下文复制答案。
>
> 这形成了非常直接的监督：
>
> $$
> \text{过去的信息是否被写入，并能否在查询时影响生成}
> $$
>
> 许多潜在记忆模块训练失败，是因为显式上下文仍然存在，主干模型可以忽略记忆路径。

> [!warning] 训练数据规模和分布仍然有限
> δ-mem 只使用 QASPER 的 $2219$ 个样本训练一个 epoch，却在多种 benchmark 上取得提升。
>
> 这说明接口具有一定迁移能力，同时也引出两个问题：
>
> - QASPER 的长文档问答结构是否为 HotpotQA 和部分 memory benchmark 提供了较强任务先验
> - 模型学到的是通用记忆机制，还是针对问答式信息压缩的表示策略
>
> 论文缺少跨训练数据源和跨任务形式的系统比较。

---

## Mechanism Interpretation

### δ-mem 实际存储了什么

从公式上看，状态为：

$$
S_t
\approx
\sum_i
\text{gated residual}_i
(k_i^m)^\top
$$

它无法被直接解释为一组离散事实。它更接近一个低维条件映射：

$$
q^m
\mapsto
r
$$

因此，δ-mem 保存的是**对未来查询有用的关联响应函数**。

这与文本记忆存在明显差异：

- 文本记忆保存可读内容
- KV cache 保存逐 token 的内部表示
- δ-mem 保存一个压缩后的输入到读出映射

我注意到，这种形式与 fast weight memory、linear attention state 和在线回归记忆之间非常接近。其创新主要集中在：

1. 将小型 delta-rule state 插入冻结的 full-attention LLM
2. 用 state readout 生成 query/output correction
3. 构造删除原始上下文的监督方式
4. 系统比较 token、segment 和 multi-state 写入

---

### 记忆容量来自哪里

表面上，$S\in\mathbb{R}^{8\times8}$ 只有 $64$ 个标量。

但其有效表达能力还依赖：

$$
W_q^m,W_k^m,W_v^m,W_\beta,W_q^\Delta,W_o^\Delta
$$

这些投影在每层中学习：

- 哪些 hidden-state 特征应该成为 memory key
- 哪些特征应该成为 memory value
- 当前查询应该如何访问状态
- 读出结果应该怎样修改 attention

因此，$S$ 更像一个**实例级动态变量**，投影矩阵则定义了其读写协议。

> [!note] 一个更准确的理解
> δ-mem 的知识分布在两个位置：
>
> - 固定参数中保存通用的记忆编码和解码规则
> - 在线状态中保存当前实例的历史条件
>
> 单独查看 $8\times8$ 状态，无法完整解释模型的记忆能力。

---

### 层级状态

论文在全部 Transformer 层插入 δ-mem 时表现最好。

这意味着模型实际维护的并非一个全局 $8\times8$ 状态，而是每个插入层各自维护一个状态：

$$
\left{
S_t^{(1)},S_t^{(2)},\ldots,S_t^{(L)}
\right}
$$

如果 Qwen3-4B 含多个 Transformer 层，那么总动态状态容量约为：

$$
L\times r^2
$$

尽管每层状态很小，总状态仍会随层数线性增长。

> [!warning] $8\times8$ 的宣传口径容易造成误解
> 论文反复强调 only an $8\times8$ state，但 full-layer δ-mem 实际上需要在多个层维护多个 $8\times8$ 状态。
>
> 更严谨的表述应当是：
>
> **Each inserted layer maintains an $8\times8$ online state.**
>
> 这不影响方法的轻量性结论，但会影响对总容量和总计算量的理解。

---

## Experiments

### Experimental Setup

作者使用三个 backbone：

- Qwen3-4B-Instruct
- Qwen3-8B
- SmolLM3-3B

主要 benchmark 包括：

#### General Capability

- IFEval
- GPQA-Diamond
- HotpotQA

#### Memory-heavy Tasks

- LoCoMo
- MemoryAgentBench

MemoryAgentBench 覆盖：

- Accurate Retrieval
- Test-time Learning
- Long Range Understanding
- Selective Forgetting

对比方法包括：

- BM25 RAG
- LLMLingua-2
- MemoryBank
- Context2LoRA
- MemGen
- MLP Memory

默认配置：

$$
r=8,\qquad \alpha=16
$$

δ-mem 默认注入 query 和 output 分支，MSW 使用 $4$ 个状态。

---

### Main Results

在 Qwen3-4B-Instruct 上：

- Frozen backbone 平均分：$46.79$
- δ-mem SSW：$51.44$
- δ-mem TSW：$51.66$
- δ-mem MSW：$50.74$
- Context2LoRA：$44.90$

TSW 相比 backbone 提升：

$$
51.66-46.79=4.87
$$

在 MemoryAgentBench 上：

- Backbone：$29.54$
- 最佳 δ-mem MSW：$38.85$

相对提升约为：

$$
\frac{38.85}{29.54}\approx1.32
$$

在 LoCoMo 上：

- Backbone：$40.79$
- 最佳 δ-mem MSW：$49.12$

在 HotpotQA 上：

- Backbone：$42.35$ EM，$56.00$ F1
- TSW：$49.41$ EM，$63.66$ F1

> [!note] 结果真正说明了什么
> δ-mem 的优势并非在所有子任务上统一出现。
>
> - TSW 更适合需要具体证据和细粒度关联的 HotpotQA
> - MSW 在 LoCoMo 和 MemoryAgentBench 上更强
> - SSW 在更大的 Qwen3-8B 上表现稳定
>
> 这说明写入粒度属于任务相关超参数。论文尚未得到一个能够自动选择粒度的统一机制。

---

### Cross-Backbone Results

在 Qwen3-8B 上：

$$
47.20\rightarrow50.86
$$

最佳策略为 SSW。

在 SmolLM3-3B 上：

$$
26.08\rightarrow36.96
$$

最佳策略为 MSW。

作者认为：

- 大模型自身推理能力较强，SSW 可以过滤 token-level 噪声
- 小模型内部表示能力有限，多状态可以减少信息干涉

> [!question] 小模型上的巨大提升可能包含补偿效应
> SmolLM3-3B 的原始 HotpotQA EM 只有 $1.67$，加入 δ-mem 后提升到最高 $31.61$。
>
> 这个幅度很大，可能说明 δ-mem 确实提供了历史信息，也可能说明训练过程给模型加入了额外的问答适配能力。
>
> 需要增加以下对照：
>
> - 相同 QASPER 数据上的普通 LoRA
> - 无历史输入的 δ-mem
> - 随机或清零状态下的 δ-mem
> - 保留投影模块但禁止 state update
>
> 这些对照可以分离**记忆收益**与**额外任务训练收益**。

---

### Context Recovery

**直觉**：如果状态真的保存了历史信息，那么删除原始历史后，模型仍应恢复部分答案。

作者构造 no-context setting：

- 原始历史完全删除
- 只保留压缩后的在线状态
- 模型使用 query 和状态进行回答

![[figure2.png]]

在 HotpotQA 上：

- No-context EM：$0.08$
- δ-mem EM：$6.48$
- No-context F1：$8.27$
- δ-mem F1：$15.20$

在 LoCoMo 上：

- No-context 平均分：约 $3.49$
- δ-mem 平均分：约 $8.05$

> [!tip] 这是论文中最关键的有效性证据
> 主结果只能说明加入 δ-mem 后任务分数提高。
>
> Context recovery 实验进一步说明：
>
> $$
> S_C
> $$
>
> 确实携带了原始 context 中的部分信息，而非仅仅充当普通 adapter。

> [!warning] 恢复能力的绝对值依然很低
> HotpotQA EM 从 $0.08$ 提升到 $6.48$，相对提升巨大，但模型仍然无法回答绝大多数问题。
>
> 这表明 $8\times8$ 状态主要保留粗粒度、任务相关信号，无法替代完整历史。
>
> 论文中的 recover context-relevant information 应理解为**恢复一部分有用信息**，不应解释为可完整重建历史内容。

---

### Head Ablation

作者比较了在不同 attention 分支注入记忆修正：

- q
- k
- v
- o
- qo
- qkvo 等组合

单分支中，output 最强：

$$
47.05
$$

默认的 qo 达到：

$$
47.97
$$

完整 qkvo 达到：

$$
48.05
$$

qkvo 只比 qo 高：

$$
0.08
$$

作者最终选择 qo，以减少参数和计算开销。

> [!tip] Query 与 Output 是合理的最小接口
> Query correction 可以调整当前检索显式上下文的方式。
>
> Output correction 可以直接补充历史状态。
>
> 两者分别作用于 attention 前后，功能相对互补。

> [!warning] 缺少跨任务一致性分析
> qkvo 在综合平均上最好，但不同任务的最优 head 可能不同。
>
> 论文主要依据两个 benchmark 选择 qo。若部署到代码 Agent、长期规划或个性化对话，最佳注入位置可能变化。

---

### Insertion Depth Ablation

作者比较：

- Front 12 layers
- Middle 12 layers
- Back 12 layers
- All layers

结果为：

- Front：$44.39$
- Middle：$46.66$
- Back：$44.06$
- All layers：$47.97$

作者认为中间层已经形成语义表示，同时仍有足够后续深度传播记忆信号。

> [!note] 中间层的结果符合常见表示层级直觉
> 前层更接近词法和局部模式。
>
> 后层更接近最终预测，修正信号缺少进一步加工空间。
>
> 中间层提供了较好的语义抽象与传播深度。

> [!warning] All Layers 的优势伴随更多参数
> All Layers 同时增加：
>
> - trainable projection 数量
> - 在线状态数量
> - 读写计算次数
>
> 因此，该消融无法完全区分**插入深度优势**与**容量增加优势**。
>
> 更公平的实验应固定总参数量或固定总状态维度。

---

### Inference Efficiency

![[figure3.png]]

作者比较不同 prompt length 和 decoding length 下的：

- decoding throughput
- GPU memory usage

δ-mem 的显存使用接近 Vanilla 和 Context2LoRA。原因是在线状态大小固定，不需要保存随历史增长的外部 memory representation。

其解码速度低于 Vanilla 和 Context2LoRA，因为每个 token、每个插入层都需要执行：

$$
r_t=S_{t-1}q_t^m
$$

以及：

$$
S_t=\operatorname{Update}(S_{t-1},k_t^m,v_t^m)
$$

但 δ-mem 仍明显快于 MemGen。

> [!note] 复杂度分析
> 状态读写本身的复杂度约为：
>
> $$
> O(r^2)
> $$
>
> 但将 hidden state 投影到 memory space，以及将读出投影回 attention space，还需要：
>
> $$
> O(dr)
> $$
>
> 若在 $L$ 层使用，则每个 token 的额外开销约为：
>
> $$
> O(Ldr+Lr^2)
> $$
>
> 它与历史长度无关，但仍会随模型深度和 hidden dimension 增长。

---

### Parameter Overhead

![[figure4.png]]

在 Qwen3-4B 上：

- δ-mem SSW：$4.87$M，约 $0.12%$
- δ-mem TSW：$4.87$M，约 $0.12%$
- δ-mem MSW：$19.47$M，约 $0.48%$
- Context2LoRA：$5.90$M，约 $0.15%$
- MemGen：$46.20$M，约 $1.13%$
- MLP Memory：$3078$M，约 $76.40%$

δ-mem 的参数效率确实较高，尤其相对于大型外部 memory network。

> [!warning] 参数量与状态容量需要分别报告
> $4.87$M 是固定可训练参数。
>
> 每个实例、每个层维护的 $S_t$ 才是动态在线记忆。
>
> 两者承担不同作用：
>
> - 参数定义读写协议
> - 状态保存当前历史
>
> 将 $8\times8$ 状态与其他方法的百万级参数直接并列，容易混淆 runtime memory 与 trainable capacity。

---

## Baseline Analysis

### Textual Memory Baselines

BM25 RAG、LLMLingua-2 和 MemoryBank 的整体表现偏弱，部分任务甚至低于原始 backbone。

作者将其归因于：

- retrieval noise
- textual compression loss
- context budget

这个解释具有合理性，但实验公平性仍需进一步确认。

> [!warning] 部分 baseline 缺少通用任务结果
> 表 1 中部分文本记忆方法在 IFEval 和 GPQA-D 上标记为缺失值。
>
> 但论文仍报告 Final Average。不同方法的 average 是否基于完全相同的任务集合，需要明确说明。
>
> 如果平均分使用的有效项不同，则 $1.15\times$ strongest baseline 的结论可能不够严格。

### Parametric Memory Baselines

Context2LoRA 在 LoCoMo Single 子任务达到 $60.11$，高于多数 δ-mem 变体，但其综合表现不稳定。

MemGen 的 IFEval 和 HotpotQA 明显下降：

- IFEval：$39.37$
- HotpotQA EM：$5.36$

这可能说明当前实现和训练设置不适配 MemGen，也可能说明生成式 latent memory 对小规模训练数据较敏感。

> [!warning] 强 baseline 的复现质量
> 当某个已发表方法出现灾难性性能下降时，需要检查：
>
> - 是否使用原论文推荐的训练方式
> - 是否为每个方法调节学习率和 rank
> - 是否保持相近训练 token 数
> - 是否允许方法使用其原本需要的上下文结构
>
> 统一 backbone 和参数预算很重要，但统一训练配置未必公平。

### Outside-Channel Baseline

MLP Memory 参数量达到 $3.078$B，但平均性能只有 $22.85$。

如此大的性能差距更可能反映 baseline 适配或训练失败，而非单纯说明 outside-channel memory 路线无效。

我会谨慎看待作者从这一结果推出的范式级结论。一个表现异常差的实现不能代表整个外部潜在记忆方向。

---

## Critical Assessment

### 核心优势

#### 1. 记忆路径非常短

$$
x_t
\rightarrow
q_t^m
\rightarrow
S_{t-1}q_t^m
\rightarrow
\Delta q_t,\Delta o_t
$$

从状态读取到影响 attention 只经过线性映射，避免复杂检索器和生成式 memory decoder。

#### 2. 状态成本与历史长度解耦

无论处理多少历史 token：

$$
S_t\in\mathbb{R}^{r\times r}
$$

状态大小保持固定。这对于持续交互和流式推理非常重要。

#### 3. 训练目标迫使模型使用记忆

删除显式 context 是一个简单但有效的约束，可以减少模型绕过 memory module 的情况。

#### 4. Delta rule 提供在线纠错

状态保存当前映射尚未拟合的残差，天然具备一定去重和覆盖能力。

---

### 核心局限

#### 1. 极小状态存在严重信息瓶颈

将数千 token 压入每层 $64$ 个标量，必然产生信息损失。

论文展示了任务分数提升，却没有回答：

- 状态最多能稳定保存多少独立关联
- 关联数量增加时何时崩溃
- 相似 key 如何相互干扰
- 状态经过多少轮更新后开始遗忘

#### 2. 写入完全由当前 hidden state 决定

$$
\beta_t=\sigma(W_\beta x_t+b)
$$

写入门只观察当前 $x_t$，没有显式考虑：

- 当前状态中已经保存了什么
- 信息是否可能在未来有用
- 当前事件是否与长期目标相关
- 写入会覆盖哪些已有记忆

虽然 residual 项间接依赖 $S_{t-1}$，但写入强度 $\beta_t$ 本身缺少完整的 state-aware 决策。

#### 3. 记忆无法显式检索或验证

状态是连续矩阵，模型不能直接回答：

- 当前保存了哪些事实
- 某条记忆来自哪个历史位置
- 两条记忆是否冲突
- 某条记忆应当被删除还是修正

这限制了可解释性、可控遗忘和事实溯源。

#### 4. MSW 缺少真正的路由机制

多个状态同时接收输入，状态分工依赖隐式学习。论文尚未证明它们形成稳定、可解释的 specialization。

#### 5. 长期在线稳定性未被充分验证

实验主要基于 benchmark context 写入。真实 Agent 可能运行数万轮，状态持续受到：

- 模板文本
- 工具输出
- 重复观察
- 错误推理
- 用户修正
- 环境噪声

论文没有展示持续在线数千轮后的状态退化曲线。

#### 6. 缺少主动记忆决策

δ-mem 的写入由每个 token 或 segment 自动触发。模型没有显式决定：

- 是否写入
- 写入哪个状态
- 是否回看原始历史
- 是否覆盖旧记忆
- 是否保留证据位置
- 是否在任务结束后压缩或清理

因此，它更接近一个**连续的隐状态更新器**，距离具备策略性的 Agent memory controller 仍有明显差距。

---

## Aha Moment

> [!tip] 最值得延伸的公式
> δ-mem 的核心可以写成：
>
> $$
> \text{Memory Read}
> ==================
>
> S_{t-1}q_t
> $$
>
> $$
> \text{Model Steering}
> =====================
>
> f(S_{t-1}q_t)
> $$
>
> $$
> \text{Memory Update}
> ====================
>
> S_{t-1}
> +
> \text{prediction error}
> \times
> \text{address}
> $$
>
> 这个结构将记忆抽象为一个可在线学习的函数：
>
> $$
> q\mapsto r
> $$
>
> 它保存的重点并非过去文本的压缩副本，而是**未来查询到有效内部修正之间的映射**。

我认为这是论文最重要的研究直觉。

传统记忆通常先保存内容，再设计检索方式。δ-mem 将内容存储与未来读出耦合成一个在线函数近似问题。只要未来 query 与历史 key 在学习到的记忆空间中对齐，状态就能产生有用修正。

这一思路也暴露了其根本风险：训练阶段无法预知所有未来查询分布。若未来 query 与写入时形成的 key 坐标不匹配，相关信息即使被压进状态，也可能无法正确读出。

---

## Future

### Query-Aware Memory Writing

当前写入发生在历史到达时：

$$
x_t\rightarrow k_t^m,v_t^m
$$

但系统尚不知道未来会提出什么问题。

一个自然方向是允许模型在获得 query 后重新处理部分历史，或者使用 query 对已有状态进行二次重构：

$$
S_C
\rightarrow
\operatorname{Refine}(S_C,Q)
\rightarrow
S_{C,Q}
$$

这可以缓解无查询条件下压缩所有信息所造成的瓶颈。

---

### State-Aware Write Controller

让写入门同时观察当前输入和旧状态：

$$
\beta_t
=======

g(x_t,S_{t-1})
$$

控制器需要判断：

- 当前信息是否已经被保存
- 当前写入是否会覆盖重要关联
- 当前信息应进入短期状态还是长期状态
- 当前信息与任务目标是否相关

可以进一步将写入、保留和遗忘拆成独立决策。

---

### Decoupled Retention and Update Gates

将：

$$
\lambda_t=1-\beta_t
$$

改为：

$$
\lambda_t
=========

\sigma(W_\lambda x_t+b_\lambda)
$$

$$
\beta_t
=======

\sigma(W_\beta x_t+b_\beta)
$$

这样模型可以分别控制：

- 全局状态保留
- 当前残差写入

还可以进一步引入 address-specific forgetting，使遗忘集中发生在当前 key 附近，而非对整行状态统一衰减。

---

### Routed Multi-State Memory

为 MSW 添加显式路由器：

$$
\pi_t
=====

\operatorname{softmax}
\left(
g(x_t,S_{t-1}^{(1:N)})
\right)
$$

更新时：

$$
S_t^{(i)}
=========

\operatorname{Update}
\left(
S_{t-1}^{(i)},x_t,\pi_{t,i}
\right)
$$

路由可以是：

- 稀疏 top-$k$
- 可微 soft routing
- 强化学习决策
- 基于状态新颖度的自动分配

这样才能更明确地实现不同状态之间的分工。

---

### Memory Capacity and Interference Theory

需要系统研究以下变量：

$$
r,\quad N,\quad L,\quad T
$$

分别表示：

- 单状态维度
- 子状态数量
- 插入层数量
- 历史长度

重点测量：

- 可保存关联数量
- key 相似度与干涉程度
- 旧记忆随时间的衰减
- 连续纠错后的稳定性
- 状态容量与任务性能的 scaling law

---

### Revisitable Memory

δ-mem 将历史压缩后，原始信息通常不再可用。

更完整的系统可以同时维护：

- 极小在线状态，用于快速推理
- 稀疏历史索引，用于必要时回看
- 原始证据存储，用于验证和纠错

推理流程可以是：

$$
\text{Fast state read}
\rightarrow
\text{uncertainty detection}
\rightarrow
\text{selective history revisit}
\rightarrow
\text{state correction}
$$

这种结构保留 δ-mem 的常数级快速路径，同时允许模型在信息不足时恢复细节。

---

### Memory Correction and Contradiction Handling

当前 delta rule 会用新 value 修正旧预测，但它无法判断新旧信息属于：

- 同一事实的新版本
- 相互矛盾的来源
- 不同时间点的状态
- 不同实体下的相似描述

可以引入时间、实体或关系条件：

$$
k_t^m
=====

f(x_t,\text{time},\text{entity},\text{relation})
$$

并保存更新来源，使状态具备更明确的版本控制能力。

---

### Memory-Centric Training Tasks

当前训练主要使用长文档问答。未来可以构造更直接的训练信号：

- 写入后延迟查询
- 多次冲突更新
- 选择性遗忘
- 跨阶段任务进度恢复
- 查询分布变化
- 记忆污染与纠错
- 不相关信息持续注入
- 需要主动回看原始历史的问题

这些任务可以区分：

$$
\text{stored}
,\quad
\text{retrievable}
,\quad
\text{usable}
,\quad
\text{correctable}
$$

四种不同的记忆能力。

---

## Final Assessment

δ-mem 给出了一条很清晰的潜在记忆路径：

$$
\text{historical hidden states}
\rightarrow
\text{fixed-size associative state}
\rightarrow
\text{attention correction}
$$

它的工程结构简单，训练目标直接，状态成本不会随历史增长。实验说明，一个极小的在线状态确实能够保留并利用部分历史信号。

这篇论文目前最有价值的结论是：

> [!tip] 核心结论
> LLM 的在线记忆不一定需要保存可检索文本或更新模型权重。一个持续演化的小型关联状态，也可以通过 attention 内部接口影响后续推理。

它尚未解决长期记忆中的核心困难：

- 未知未来查询下的信息选择
- 固定容量中的干涉与遗忘
- 记忆内容的显式验证
- 冲突修正与版本管理
- 主动写入、回看和清理
- 长时间 Agent 运行中的稳定性

因此，我更愿意将 δ-mem 看作一个**高效、可训练的在线潜在状态接口**。它证明了 latent online memory 可以通过极小动态状态发挥作用，也为进一步研究主动记忆、查询后回看和多状态路由提供了非常合适的基础结构。
