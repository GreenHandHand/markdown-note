# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

arXiv-2025 · arXiv:2507.02259v1 

**核心一句话**：MemAgent 将超长文本处理改写为一个查询条件下的顺序记忆更新问题。模型逐块读取文档，用普通文本 token 反复覆盖固定容量的记忆，再根据最终答案的正确性，通过强化学习反向塑造每一步应该保留什么信息，由此让标准 Transformer 在固定窗口内处理百万级输入。

---

## Key Contribution

* We introduce a novel approach that enables LLMs to process arbitrarily long inputs within limited context window under linear time complexity during inference, overcoming a significant bottleneck in long-context processing.
* We design an agent workflow to implement this mechanism and propose an end-to-end training approach using the multi-conversation DAPO algorithm.
* We empirically demonstrate that our RL-trained method allows models to extrapolate to vastly long documents with minimal performance degradation, pushing the boundaries of what is currently achievable in long-context LLM systems.

### 贡献的技术含义

* 作者没有扩展 Transformer 的物理上下文窗口，而是将长文档转换为连续的数据流。每次只输入当前文本块、问题和上一轮记忆，因此总文档长度不再受到单次上下文窗口限制。
* 记忆直接使用自然语言 token 表示。模型架构、注意力实现和 tokenizer 均不需要修改，记忆更新与普通自回归生成共用同一套模型参数。
* 作者提出的 **Multi-Conv DAPO** 解决了一个较为通用的问题：一条 Agent 轨迹由多个彼此独立、无法拼接进同一上下文的对话组成时，如何用最终结果奖励联合优化所有对话。
* 实验表明，使用 8K 上下文窗口、约 32K 长度训练数据进行强化学习后，14B 模型能够处理最长 3.5M token 的问答输入，并在多个 RULER 任务上维持稳定性能。
* 论文中的 memory 更接近**面向当前问题的临时文本状态**。它在一次文档问答过程中持续更新，任务结束后没有跨任务保留，也没有形成可供其他问题复用的文档记忆库。

> [!tip] 最值得保留的核心设计
> MemAgent 的关键价值不只是固定长度文本摘要。更重要的是，作者把**如何写记忆**纳入策略学习，让最终任务奖励直接塑造中间记忆行为，同时允许每一步使用独立上下文。这套训练形式可以迁移到其他长轨迹 Agent workflow。

---

## Method

### Long-context 问题的重新定义

**作者的直觉是：模型无需让所有历史 token 始终参与注意力，只需维护一个能够支撑当前任务的有限状态。**

作者将现有长上下文方法划分为三类：

1. **位置外推与长上下文继续预训练**

   通过修改 RoPE 频率、位置索引或插值方式扩大可接受的序列长度。该路线仍然需要对大量 token 执行 Dense Attention，计算复杂度随长度快速增长，实际可用上下文通常也低于标称窗口。

2. **稀疏注意力与线性序列模型**

   通过限制注意力连接或修改架构降低复杂度。这类方法通常需要特殊 kernel、重新训练模型或预先定义稀疏模式。

3. **上下文压缩与外部记忆**

   将历史内容压缩成 token、向量或外部存储。作者认为额外模块会增加系统集成成本，并改变标准生成流程。

作者据此提出三个目标：

* 文档长度原则上不受限制
* 长度增加时性能保持稳定
* 总计算量关于文档长度近似线性增长

![[MemAgent-Figure1-Length-Extrapolation.png]]

Figure 1 展示了作者想强调的现象。多个标称支持 128K 或 1M 上下文的模型在输入长度增加后迅速退化，MemAgent-14B 的曲线在 7K 到 3.5M 范围内相对平稳。

> [!warning] 目标表述需要收紧
> **任意长度**描述的是算法能够持续运行，不代表系统能够以固定延迟处理无限文本。MemAgent 必须顺序执行每一次记忆更新，文档越长，生成轮数和实际等待时间仍然线性增加。
>
> **无性能下降**也只获得了特定问答和合成检索任务上的经验证据，目前没有理论保证。

---

### Query-conditioned Streaming Memory

**作者的直觉是：先给模型问题，再让模型阅读文档，模型便可以只记录与这个问题相关的内容。**

设问题为 (q)，文档被划分为 (K) 个连续文本块：

[
D=(c_1,c_2,\ldots,c_K)
]

初始记忆为空：

[
m_0=\varnothing
]

读取第 (k) 个文本块后，模型生成新的记忆：

[
m_k \sim \pi_\theta
\left(
\cdot
\mid q,m_{k-1},c_k
\right)
]

全部文本块处理完成后，模型仅根据问题和最终记忆生成答案：

[
\hat y \sim \pi_\theta
\left(
\cdot
\mid q,m_K
\right)
]

因此，一次完整推理可以写成：

```text
问题 q
  ↓
文本块 c1 + 空记忆 m0 → 新记忆 m1
  ↓
文本块 c2 + 记忆 m1 → 新记忆 m2
  ↓
...
  ↓
文本块 cK + 记忆 mK-1 → 最终记忆 mK
  ↓
问题 q + 最终记忆 mK → 答案
```

![[MemAgent-Figure2-Workflow.png]]

Figure 2 的上半部分表示传统长上下文模型一次接收完整文档。下半部分表示 MemAgent 在多个固定窗口中依次处理文本块。每一轮只将上一轮生成的记忆传递给下一轮。

#### Prompt 实现

作者没有实现独立的神经记忆模块。Context Processing 和 Answer Generation 都通过提示词控制同一个 LLM。

![[MemAgent-Table1-Prompt-Template.png]]

上下文处理阶段可以抽象为：

```text
UpdateMemory(
    problem=q,
    previous_memory=m_{k-1},
    current_chunk=c_k
) -> m_k
```

最终回答阶段为：

```text
Answer(
    problem=q,
    memory=m_K
) -> y
```

这意味着 Figure 4 中的 Controller、Read Head 和 Write Head 属于概念建模。实际实现中，同一个自回归模型同时承担阅读、判断和记忆生成。

> [!tip] 工程上的简洁性
> 记忆是普通 token，模型调用方式与标准 Chat Completion 基本一致。任何能够进行指令生成的 Decoder-only LLM 都可以接入这套 workflow，无需增加可训练的外部存储模块。

#### 记忆覆盖

每一轮生成完整的新记忆 (m_k)，旧记忆 (m_{k-1}) 不再保留。覆盖策略带来两个直接结果：

* 记忆预算始终受控，文档长度不会导致上下文持续增长。
* 删除的信息无法恢复，早期一次错误压缩可能永久影响后续回答。

我注意到，这种更新形式本质上是一个带信息瓶颈的递归状态：

[
m_k=f_\theta(q,m_{k-1},c_k)
]

模型必须同时完成四个操作：

1. 判断当前文本块是否相关
2. 保留旧记忆中的有效信息
3. 加入新发现的信息
4. 删除过时、重复或低价值内容

论文通过最终答案奖励隐式学习这些操作，没有分别监督四种行为。

> [!warning] 论文中的 memory 边界
> 这里的记忆高度依赖当前问题 (q)。同一篇文档面对不同问题时，需要重新从头处理。
>
> 因此，MemAgent 没有解决先构建一次记忆、之后回答任意查询的问题，也没有研究跨会话长期记忆、记忆检索、历史修正和多用户共享等场景。

> [!question] 未显现价值的信息如何保留
> 某条事实当前看起来与问题无关，后续文本可能使其成为关键证据。覆盖式记忆只能根据当前状态预测未来价值。
>
> 作者没有专门测试延迟相关性场景，即第一条证据的意义只有在很久以后读到第二条证据时才能识别。

---

### 复杂度分析

**作者的直觉是：只要每一轮的输入和输出长度都有固定上限，处理更多文本块只会线性增加调用次数。**

设：

* 总文档长度为 (N)
* 每个文本块长度为 (C)
* 记忆预算为 (M)
* 问题及模板长度为 (Q)

文本块数量为：

[
K=\left\lceil\frac{N}{C}\right\rceil
]

每轮模型处理的上下文长度约为：

[
L=Q+C+M
]

若 (Q,C,M) 均固定，则总计算量可写为：

[
T(N)
\approx
K\cdot T_{\mathrm{LLM}}(L,M)
+
T_{\mathrm{answer}}(Q+M)
]

因此：

[
T(N)=O(N)
]

这里的线性复杂度建立在固定 (C) 和 (M) 的条件上。单轮内部依旧使用 Dense Attention，其计算量仍然大致与 (L^2) 相关。

![[MemAgent-Figure7-Compute-Complexity.png]]

Figure 7 使用 FLOP 估计器比较一次性 Dense Attention 与 MemAgent。输入增长至百万 token 后，前者的估计计算量呈二次增长，MemAgent 保持近似线性。

> [!warning] FLOP 线性不等于推理高效
> MemAgent 每处理一个文本块，都要自回归生成一段新的记忆。文档块之间存在严格的数据依赖，无法直接并行处理。
>
> 论文没有报告端到端延迟、首 token 时间、吞吐量、KV cache 使用量或真实 GPU 成本。Figure 7 只能证明渐近 FLOP 趋势，无法证明实际系统一定更快。

---

### Training MemAgent with Multi-Conv DAPO

**作者的直觉是：只评价最终答案，再把这个结果归因给之前所有记忆更新，让模型逐渐学会哪些记忆轨迹会产生正确答案。**

#### 为什么普通多轮 RL 不够

常见工具调用 Agent 会将所有 observation 和 action 串联成一段对话：

```text
user → assistant → tool → assistant → tool → assistant
```

MemAgent 的每一次更新都使用独立上下文：

```text
conversation 1: q + c1 + m0 → m1
conversation 2: q + c2 + m1 → m2
conversation 3: q + c3 + m2 → m3
...
conversation K+1: q + mK → answer
```

整条轨迹可能远超模型窗口，无法作为一段连续对话统一计算 attention mask。作者因此将每个 conversation 视为独立优化单元，再用同一个终局奖励关联它们。

![[MemAgent-Figure3-MultiConv-DAPO.png]]

Figure 3 上半部分是普通 GRPO，每个样本产生一个 response。下半部分的每个样本产生多个 context-independent conversations，只有最后一个 conversation 包含答案，但所有 conversation 都参与参数更新。

#### GRPO 基础

对于同一输入采样 (G) 条轨迹，轨迹 (i) 的最终奖励为 (R_i)。普通 GRPO 使用组内标准化优势：

[
\hat A_i
========

\frac{
R_i-\operatorname{mean}(R_1,\ldots,R_G)
}{
\operatorname{std}(R_1,\ldots,R_G)
}
]

token 级 importance ratio 为：

[
\rho_{i,t}(\theta)
==================

\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}
]

随后使用 PPO 风格的 clipped objective，并加入相对 reference model 的 KL 惩罚。

#### Multi-Conv 扩展

假设第 (i) 条轨迹包含 (n_i) 个独立 conversation：

[
o_{i,1},o_{i,2},\ldots,o_{i,n_i}
]

作者根据最后一个 conversation 的答案计算 (R_i)，并将同一个优势分配给该轨迹中的全部 conversation 和 token：

[
\hat A_{i,j,t}
==============

R_i-\operatorname{mean}(R_1,\ldots,R_G)
]

论文跟随 DrGRPO，没有再除以组内奖励标准差。

每个 token 的 clipped 项为：

[
C_{i,j,t}
=========

\min
\left(
\rho_{i,j,t}\hat A_{i,j,t},
\operatorname{clip}
\left(
\rho_{i,j,t},
1-\epsilon_{\mathrm{low}},
1+\epsilon_{\mathrm{high}}
\right)
\hat A_{i,j,t}
\right)
]

整体目标扩展为 group、conversation、token 三个维度：

[
J(\theta)
=========

\mathbb E
\left[
\frac{
\sum_{i=1}^{G}
\sum_{j=1}^{n_i}
\sum_{t=1}^{|o_{i,j}|}
\left(
C_{i,j,t}
---------

\beta D_{\mathrm{KL}}
\right)
}{
\sum_{i=1}^{G}
\sum_{j=1}^{n_i}
|o_{i,j}|
}
\right]
]

与普通 GRPO 相比，核心变化可以概括为：

```text
一个样本
→ 多个独立 conversation
→ 一个最终奖励
→ 奖励广播到全部 conversation
→ 按全部 token 总数归一化
```

> [!tip] Multi-Conv DAPO 的通用价值
> 它允许 Agent 的每一步运行在独立上下文中，同时仍然由一个终局任务奖励联合训练。这对长周期搜索、分阶段规划和多文档处理都有潜在价值。

#### Credit Assignment 问题

所有记忆更新共享相同优势：

[
\hat A_{i,1}
============

# \hat A_{i,2}

# \cdots

\hat A_{i,n_i}
]

这一处理非常简单，但归因较粗糙。

* 最终答案正确时，早期无用或错误的记忆更新也会得到正向强化。
* 最终答案错误时，已经正确保存的中间证据也会受到负向更新。
* 某一步删除关键事实导致失败时，算法无法直接定位是哪一次更新造成的。
* 最终解码器偶然答对时，一条低质量记忆轨迹也可能整体获得正奖励。

> [!question] 为什么没有分步价值估计
> 作者没有尝试比较删除某段记忆前后的答案变化，也没有训练 conversation-level value function。
>
> 更精细的方案可以根据记忆对最终答案概率的边际贡献，为每一次写入分配独立奖励。

#### RL 是否必需

作者提出，记忆由离散 token 构成，覆盖操作无法通过普通反向传播直接获得保留与删除策略，因此需要 RL。

这个判断的准确表述应当是：

**训练数据只有最终答案，没有高质量中间记忆标签。RLVR 可以绕过中间监督，直接利用可验证答案训练记忆策略。**

离散记忆本身并不排除其他训练方式。还可以采用：

* 教师模型生成记忆轨迹，再进行 SFT
* 通过答案梯度训练连续 latent memory
* 对候选记忆进行偏好优化
* 通过搜索获得高奖励记忆，再进行蒸馏
* 使用答案概率构造可微或近似可微的记忆效用

因此，论文证明了 RL 的显著收益，还没有证明 RL 是唯一可行路径。

> [!warning] 符号细节
> Equation 4 写成 (r_i-\operatorname{mean}(R_i))，正文将该量描述为最终奖励 (R_i)。这里大概率存在符号不统一，(r_i) 容易与 importance ratio 混淆。

---

### Reward Modeling

**作者的直觉是：记忆质量很难直接标注，但最终答案是否正确可以通过规则验证。**

#### 等价答案任务

对于多个等价 ground truth：

[
Y={y_1,y_2,\ldots,y_n}
]

奖励定义为：

[
R(\hat y,Y)
===========

\max_{y\in Y}
\mathbb I
\left[
\operatorname{is_equiv}(y,\hat y)
\right]
]

只要预测与任一合法答案等价，奖励为 1。

#### 多值检索任务

对于要求输出所有目标值的任务：

[
R(\hat y,Y)
===========

\frac{
\sum_{y\in Y}
\mathbb I[y\in\hat y]
}{
|Y|
}
]

该奖励衡量 ground-truth recall。

> [!warning] 多值奖励没有惩罚错误输出
> Equation 7 只检查正确值是否出现。模型额外输出错误值时，奖励不会降低。
>
> 这种设计可能鼓励模型保留和输出更多候选信息，降低记忆精度。论文没有报告 precision、F1 或 hallucination rate。

> [!question] is_equiv 的具体实现
> 作者将其描述为规则验证，但没有详细说明字符串归一化、别名、数值格式和语义等价如何处理。奖励边界会直接影响 RL 的学习行为。

---

### Autoregressive Latent Memory View

**作者的直觉是：可以把固定文本记忆视为循环神经网络的隐状态，从而将一条超长自回归序列拆成多次读取和写入。**

![[MemAgent-Figure4-Latent-Memory-Model.png]]

Figure 4 左侧将 MemAgent 表示为 Controller、Read Head、Write Head 和 Memory。右侧给出图模型：

* (c^k) 表示第 (k) 个文本块
* (m^k) 表示读取该文本块后得到的记忆
* 绿色路径表示从记忆读取
* 红色路径表示向记忆写入

标准语言模型对完整序列进行分解：

[
p(x_{1:N})
==========

\prod_{n=1}^{N}
p(x_n\mid x_{1:n-1})
]

作者引入记忆序列后写为：

[
p(x_{1:N})
==========

\sum_{m_{1:K-1}}
\prod_{k=1}^{K}
\underbrace{
p(c^k\mid m^{k-1})
}*{\text{read}}
\underbrace{
p(m^k\mid c^k,m^{k-1})
}*{\text{write}}
]

该表达希望说明，每一步只需要访问固定大小的 (m^{k-1}) 和当前块 (c^k)，Transformer 因而可以被理解为一个状态大小可控的循环模型。

> [!warning] 这部分更接近概念解释
> 实际 workflow 中，文本块 (c^k) 是外部给定的 observation，模型没有生成 (c^k)，训练目标也没有最大化上述边缘似然。真正被优化的是最终答案奖励下的策略目标。
>
> 因此，Equation 8 没有构成 MemAgent 训练算法的概率推导。

我注意到公式还存在几个记号问题：

* 求和范围写作 (m_{1:K-1})，乘积中却包含 (p(m^K\mid c^K,m^{K-1}))。
* Figure 4 还包含最终输出 (c^{K+1})，Equation 8 没有明确写出答案生成项。
* 文中将块内读取展开为 (p(x_i\mid x_{1:i-1},m^{k-1}))，这里仍然出现全部历史 token (x_{1:i-1})。若严格对应固定窗口，应只依赖当前块内前缀和记忆。
* 记忆在实际轨迹中是可读、可编辑的显式文本。它只相对于最终答案监督属于未标注的中间变量。

> [!note] 更准确的形式化
> 对问答任务，可以直接写成有限状态策略：
>
> [
> m_k\sim\pi_\theta(m_k\mid q,m_{k-1},c_k)
> ]
>
> [
> \hat y\sim\pi_\theta(\hat y\mid q,m_K)
> ]
>
> [
> \max_\theta\ \mathbb E[R(\hat y,Y)]
> ]
>
> 这组表达与实际训练流程更一致。

---

## Experiments

### Datasets

作者主要使用 RULER 风格的长文本问答数据。

#### HotpotQA 训练数据

* 从 HotpotQA 构造多跳问答样本。
* 将包含答案的 golden paragraphs 混入大量同分布干扰文档。
* 每个训练样本约包含 200 篇文章，总长度约 28K token。
* 初始处理 80,000 个 HotpotQA training split 样本。
* 使用 Qwen2.5-7B-Base 和 Qwen2.5-7B-Instruct 进行无上下文 Best-of-2 测试。
* 过滤模型不读取文档也能全部答对的问题，约删除 50%。
* 从剩余数据中选择前 32,768 个样本训练。

#### 长度外推测试集

作者从 HotpotQA validation split 构造 128 个问题，并为同一批问题生成不同长度的干扰上下文：

| 文章数量 | 近似 token 长度 |
| ---: | ----------: |
|   50 |          7K |
|  100 |         14K |
|  200 |         28K |
|  400 |         56K |
|  800 |        112K |
| 1600 |        224K |
| 3200 |        448K |
| 6400 |        3.5M |

这种设计控制了问题内容，只改变干扰文本数量，适合测量长度增加引起的性能变化。

> [!warning] 长度外推与任务外推需要区分
> 训练集和主测试集共享 HotpotQA 问题形式与 RULER 合成方式。实验有力证明了**长度外推**，对开放任务、真实书籍和复杂 Agent 历史的泛化证据较弱。

---

### Training Setup

基础模型：

* Qwen2.5-7B-Instruct
* Qwen2.5-14B-Instruct

模型始终限制在 8K context window：

| 部分               | token 预算 |
| ---------------- | -------: |
| Query            |     1024 |
| Current Chunk    |     5000 |
| Previous Memory  |     1024 |
| Generated Output |     1024 |
| Chat Template    |     剩余空间 |

训练样本通常需要 5 到 7 个 conversation 才能完成全文处理。

主要超参数：

* GRPO / Multi-Conv DAPO
* Group size 16
* KL coefficient (10^{-3})
* AdamW
* Learning rate (10^{-6})
* Constant schedule with linear warm-up
* 不使用 entropy loss
* 7B rollout batch size 128
* 14B rollout batch size 256

> [!note] 隐含的训练成本
> 每条样本需要生成多轮记忆，每轮最多生成 1024 token，再乘以 16 条 group rollout。训练的自回归采样成本可能很高，论文没有报告总训练 token、GPU hours 或与长上下文继续训练的成本比较。

---

### Main Results

![[MemAgent-Table2-Main-Results.png]]

主要 HotpotQA 长度外推结果：

| Model               |    7K |  112K |  896K | 1.75M |  3.5M |
| ------------------- | ----: | ----: | ----: | ----: | ----: |
| RL-MemAgent-14B     | 83.59 | 76.56 | 77.34 | 76.56 | 78.12 |
| RL-MemAgent-7B      | 82.03 | 79.69 | 76.56 | 75.78 | 71.09 |
| QwenLong-L1-32B     | 72.66 | 31.25 | 11.72 |   N/A |   N/A |
| Qwen2.5-14B-1M      | 60.16 | 50.00 |  0.00 |   N/A |   N/A |
| DS-Distill-Qwen-32B | 70.31 | 23.44 |  7.03 |   N/A |   N/A |

最有说服力的结果是曲线形状：

* MemAgent-14B 从 7K 到 3.5M 没有出现随长度持续下降的趋势。
* 7B 模型到 896K 仍然稳定，3.5M 时下降更加明显。
* 标称 1M context 的 Qwen2.5-Instruct 在 896K 输入上降至 0。
* 推理模型在上下文增长后退化尤其迅速，表明推理能力无法自动转化为超长文本利用能力。

> [!warning] 摘要中的小于 5% 需要明确口径
> Table 2 中 MemAgent-14B 从 7K 的 83.59 降至 3.5M 的 78.12，下降 5.47 个百分点。
>
> MemAgent-7B 从 82.03 降至 71.09，下降 10.94 个百分点。
>
> 摘要没有说明小于 5% 使用的基准长度、相对变化还是其他统计方式。

> [!warning] 超窗口比较的公平性
> 128K 模型在更长输入上必然需要截断。此时比较主要说明 MemAgent 能够持续流式处理，无法排除一个强 chunking、RAG 或递归摘要 baseline 取得类似结果。
>
> 论文最缺少的 baseline 正是**使用同一基础模型、同一文本块和同一问题，但采用人工 prompt 更新记忆、检索或分层摘要**的方法。

---

### RL Ablation

**作者想验证的是：固定记忆 workflow 提供了长文本处理能力，RL 进一步教会模型稳定地利用这套能力。**

![[MemAgent-Figure5-RL-Ablation.png]]

Figure 5 对比三类模型：

* 原始 Qwen2.5-Instruct
* 使用 MemAgent prompt，但未经 RL 训练
* 经过 RL 训练的 MemAgent

结论较为清晰：

* 原始模型在超过窗口后迅速退化。
* 未训练的 MemAgent 已经能够处理超出模型窗口的文本，说明 workflow 本身非常重要。
* RL-MemAgent 在长输入下更加稳定，说明强化学习改善了记忆选择和更新策略。

我认为这项消融支持两个不同结论：

1. **分块加固定文本记忆本身就是强 baseline**
2. **RL 主要改善长度增长时的记忆稳定性**

论文将重点放在第二点，但第一点同样值得强调。部分附录任务上，32B 的 MemAgent w/o RL 已经接近 RL-MemAgent-14B。

> [!warning] 仍然缺少关键对照
>
> * 没有 memory trajectory SFT baseline
> * 没有 rejection sampling 后蒸馏
> * 没有 DPO 或 preference learning
> * 没有只训练最终回答、不训练记忆更新的对照
> * 没有相同 rollout 预算下的不同 RL 算法比较

因此，当前实验能够说明 RL 有效，无法区分收益来自 outcome optimization、采样搜索、DAPO 细节还是更大的训练计算量。

---

### Out-of-Distribution Tasks

作者进一步测试：

* Single-key NIAH
* Multi-key NIAH
* Multi-value NIAH
* Multi-query NIAH
* Variable Tracking
* Frequent Words Extraction
* SQuAD-based QA

![[MemAgent-Figure6-OOD-Results.png]]

在 10 个 RULER 合成任务的平均结果中：

* RL-MemAgent-14B 从 8K 的 97.45 保持到 512K 的 95.40。
* RL-MemAgent-7B 从 93.03 降至 81.91。
* MemAgent-32B w/o RL 从 99.04 降至 81.51。
* 多个 Long Context Model 和 Reasoning Model 在 128K 后出现明显性能崩溃。

SQuAD QA 上，RL-MemAgent 在 8K 到 256K 范围内也较为稳定，说明模型没有只记住 HotpotQA 的固定实体关系模板。

![[MemAgent-Figure8-Single-Key-NIAH.png]]

![[MemAgent-Figure9-Multi-Key-NIAH.png]]

![[MemAgent-Figure10-Advanced-NIAH.png]]

![[MemAgent-Figure11-FWE-Variable-Tracking.png]]

附录揭示了更细的规律：

* 14B RL 模型在 Multi-query 和 Multi-value NIAH 的 512K 输入上仍接近满分。
* 7B 模型在复杂多值任务上出现明显下降，固定 1024-token 记忆对模型能力仍有要求。
* Frequent Words Extraction 中，MemAgent-32B w/o RL 与 RL-MemAgent-14B 非常接近，说明部分任务依赖明确的可压缩统计量，RL 的增益有限。
* Variable Tracking 对状态更新要求更强，RL-MemAgent-14B 的优势更明显。

> [!note] OOD 的实际范围
> 这些任务在数据来源和表面形式上超出 HotpotQA，但仍属于可规则验证的合成长上下文任务。它们通常具有明确查询和较短目标答案。
>
> 真实长文档中的隐含主题、时间冲突、模糊指代、开放式总结和事实修订尚未被覆盖。

---

### Case Study

**作者想展示模型学会了三种行为：预先保留潜在相关信息、遇到关键证据后立即更新、后续干扰不再破坏已完成的记忆。**

问题要求找到电影 Big Stone Gap 的导演居住在纽约的哪个地区。

![[MemAgent-Case-Study-Memory-Trajectory.png]]

第一轮文本只出现一个位于 New York City 的音乐制作团队 Ghost。模型将其保存在记忆中。

第二轮没有相关信息，模型维持旧记忆。

第三轮同时出现两条关键证据：

* Big Stone Gap 由 Adriana Trigiani 编剧并执导
* Adriana Trigiani 位于 Greenwich Village, New York City

模型将两条事实组合，得到正确答案 Greenwich Village。

作者将第一轮行为解释为对潜在相关内容的提前保存。我看到的另一面是：

* Ghost 与问题中的导演没有关系
* 模型主要因为 New York City 关键词保留了该信息
* 第三轮之后，记忆仍保存 Ghost 和大量无关人物描述
* 1024-token 预算掩盖了记忆选择精度不足的问题

> [!warning] 这个案例同时暴露了记忆冗余
> 模型成功抵抗了后续干扰，但没有主动删除已经确认无关的 Ghost 信息。案例证明了答案可以成功生成，对高精度遗忘和记忆压缩的证明较弱。

---

### Important Baselines

* **Qwen2.5-Instruct**

  基础指令模型，用于判断标准模型在上下文窗口内外的表现。

* **Qwen2.5-Instruct-1M**

  采用长上下文训练和 DCA 外推的 1M context 模型，用于比较标称窗口与实际有效窗口。

* **QwenLong-L1-32B**

  使用强化学习训练的长上下文推理模型，代表直接优化长上下文 reasoning 的路线。

* **DeepSeek-R1-Distill-Qwen**

  代表具有较强推理能力，但没有专门记忆 workflow 的模型。

* **MemAgent w/o RL**

  最重要的消融 baseline，直接衡量分块覆盖式文本记忆自身的能力。

---

### Experimental Weaknesses

> [!warning] 核心实验漏洞
>
> * 主结果只有 128 个 HotpotQA validation 问题，表格数值以约 0.78 为最小变化单位。论文没有报告置信区间或多随机种子结果。
> * 主测试只增加干扰文档数量，问题和答案结构保持不变，容易将长上下文能力收缩为抗干扰检索能力。
> * 数据过滤由 Qwen2.5-7B 系列完成，训练模型也使用 Qwen2.5 backbone，可能形成面向该模型知识边界的数据偏置。
> * 缺少 RAG、BM25、embedding retrieval、递归摘要和 map-reduce QA 等系统级 baseline。
> * 没有测试问题在文档读取完成后才给出的情况。
> * 没有测试同一文档多次查询时的复用成本。
> * 没有 memory size、chunk size、输出长度、覆盖频率和文本顺序消融。
> * 没有直接评价记忆的 recall、precision、压缩率、冲突率和事实保真度。
> * 没有真实 latency、峰值显存和训练成本。
> * Case Study 数量过少，无法排除 cherry-picking。

---

## Overall Assessment

> [!tip] 论文真正成立的结论
> 在问题预先给定、答案可规则验证、文档能够顺序扫描的场景中，固定长度自然语言记忆加终局强化学习可以让标准 LLM 获得非常强的长度外推能力。Multi-Conv DAPO 为独立上下文组成的长 Agent 轨迹提供了简洁的端到端训练方法。

> [!warning] 论文尚未证明的结论
>
> * 固定文本记忆能够覆盖一般长上下文理解
> * 该方法在真实系统中具有更低延迟
> * RL 是训练离散记忆的必要条件
> * 记忆能够在不同查询和不同任务之间复用
> * 模型能够可靠执行主动遗忘、冲突修正和长期知识更新

我认为这篇论文最重要的洞察可以压缩为：

[
\text{Long Context}
\rightarrow
\text{Bounded Recurrent State}
\rightarrow
\text{Outcome-supervised Memory Policy}
]

它将长上下文建模从扩大 attention window 转移到学习状态更新规则。其当前实现仍然是一个面向查询的文本压缩 Agent，距离通用、持久、可检索和可修正的记忆系统还有明显距离。

---

## Future

论文没有单独讨论局限与未来方向。基于当前方法，以下方向最值得继续推进。

### Query-independent Memory

当前记忆始终依赖问题：

[
m_k=f(q,m_{k-1},c_k)
]

更强的目标是先构建文档记忆：

[
m_k=f(m_{k-1},c_k)
]

之后针对任意问题进行读取：

[
\hat y=g(q,m_K)
]

这里会出现更困难的信息选择问题。写入阶段不知道未来查询，记忆必须保留可支持多种任务的结构。

### Fine-grained Credit Assignment

为每次记忆更新估计独立价值：

[
\Delta_k
========

V(q,m_k)-V(q,m_{k-1})
]

或者通过删除、替换某段记忆后观察答案概率变化，估计该记忆的边际贡献。这样可以减少终局奖励广播造成的错误归因。

### Memory Correction and Forgetting

当前覆盖操作没有显式区分新增、保留、修改和删除。可以将更新建模为结构化操作：

```text
KEEP
ADD
REVISE
DELETE
MERGE
```

训练信号需要衡量事实是否过期、是否冲突以及删除后能否恢复。

### Hierarchical and Addressable Memory

单个 1024-token 面板容易形成信息瓶颈。可以建立：

* 短期工作记忆
* 长期归档记忆
* 可检索的记忆单元
* 任务相关的动态读取视图

模型每一步只访问少量相关记忆，从而保留更多历史信息，同时控制上下文长度。

### Hybrid Token and Latent Memory

自然语言记忆具有可读性，但每轮都需要生成大量 token。可以组合：

* 文本记忆用于可解释事实
* KV 或 hidden-state 记忆用于高带宽状态
* 文本索引用于地址
* latent value 用于快速读取

核心问题是如何让不同表示之间保持稳定语义。

### Parallel or Tree-structured Processing

顺序覆盖导致推理延迟随文档长度增长。可以让多个子 Agent 并行处理不同文档块，再通过层级合并器压缩：

[
m^{(l+1)}_i
===========

\operatorname{Merge}
\left(
m^{(l)}*{2i},
m^{(l)}*{2i+1}
\right)
]

这会将顺序深度从 (O(K)) 降至 (O(\log K))，但跨块依赖与全局一致性会更难处理。

### Stronger Memory Evaluation

除了最终准确率，还应直接测量：

* Evidence Recall
* Memory Precision
* Compression Ratio
* Contradiction Rate
* Stale Information Rate
* Recovery after Incorrect Update
* Robustness to Adversarial Distractors
* Reusability across Queries
* Memory Drift over Long Horizons

这些指标能够区分模型是形成了稳定记忆，还是依靠较大的文本预算暂时保留大量候选信息。

### Real-world Long-horizon Tasks

下一步需要从 RULER 扩展到：

* 整本书级别的多问题问答
* 长时间 Agent 交互历史
* 大型代码仓库理解
* 多日研究与搜索轨迹
* 持续变化的事实数据库
* 包含冲突、修订和时间关系的文档流

这些场景能够真正检验固定容量记忆是否具备持续学习、错误修正和长期稳定性。
