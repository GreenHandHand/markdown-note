# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents

ICLR 2026

**核心一句话**：MemGen 将历史经验写入独立的 Memory Weaver，再由 Memory Trigger 判断推理过程中的关键节点，动态生成一小段 latent tokens 注入冻结的 LLM，使记忆能够随着当前推理状态被重新构造和按需读出。

---

## Key Contribution

- A dynamic and generative memory framework that interleaves memory synthesis with the token-level reasoning process.
- A reinforcement learning-trained memory trigger that determines when latent memory should be invoked.
- A memory weaver that reconstructs experiential knowledge into context-dependent, machine-native latent tokens while keeping the reasoner frozen.
- Extensive evaluation across multiple domains, including web search, embodied action, mathematical reasoning, scientific reasoning, and code generation.
- A post-hoc intervention study suggesting that different latent memory clusters support planning, procedural execution, and working-memory-like functions.

- **贡献一的核心在于改变记忆的读出时机。** 传统 Agent Memory 通常在任务开始时检索一次，或者在每个环境交互步骤检索一次。MemGen 将记忆调用进一步细化到生成过程内部，让模型写到某个句子边界时决定是否补充记忆。
- **贡献二是将经验学习与基础模型参数解耦。** Reasoner 始终冻结，训练信号只更新 Weaver 和 Trigger。这样可以降低直接微调 Reasoner 导致通用能力退化的风险。
- **贡献三是引入查询条件化的 latent readout。** Weaver 不输出可读文本，而是根据当前 hidden states 构造固定长度的连续向量。相同历史经验面对不同推理状态，可以产生不同的记忆表示。
- **论文真正提出的是两层记忆结构。** 长期经验存放在 Weaver 的 LoRA 参数中，推理时产生的 latent tokens 是面向当前上下文的短时读出。论文将两层统一称为 latent memory，但分析方法时最好将长期存储与动态载体分开理解。
- **论文对 human-like memory hierarchy 的论证具有探索性。** 作者确实进行了干预实验，但 planning、procedural、working memory 等标签来自事后聚类和错误类型映射，还不足以说明模型形成了与人类认知机制同构的记忆系统。

---

## Method

### 1. 从静态检索到推理过程中的动态记忆

**直觉**：模型在整个任务中对记忆的需求并不均匀，因此记忆系统应当观察当前推理状态，在真正需要帮助的位置生成针对性的记忆。

作者首先将 Agent Memory 分成三种范式：

1. **Parametric Memory**

   历史经验通过 SFT、GRPO、DPO 等训练方式写入 Agent Policy 的参数。

   优点是经验与模型推理结合紧密，缺点是修改核心模型参数可能造成灾难性遗忘。

2. **Retrieval-based Memory**

   历史轨迹、经验总结或技能被存放在外部数据库中，推理时通过检索加入 prompt。

   这种方式保留了原模型，但性能依赖检索、排序、格式化和上下文组织。检索结果通常以文本块的形式粗粒度地加入上下文。

3. **Latent Memory in MemGen**

   历史经验先被 Weaver 吸收到辅助参数中。推理过程中，Weaver 根据当前 hidden states 生成 latent tokens，并直接插入 Reasoner 的计算上下文。

![[Figure 1.png]]

Figure 1 想强调两个维度：

- **存储位置**：经验可以存放在核心模型参数、外部数据库或 Weaver 参数中。
- **读出形式**：经验可以通过文本检索，也可以被重构成机器原生的 latent tokens。

> [!tip] 关键立意
> MemGen 最值得关注的地方是将记忆读出建模成一个条件生成过程：
>
> [
> \text{当前推理状态}
> \rightarrow
> \text{重新构造记忆}
> \rightarrow
> \text{继续推理}
> ]
>
> 记忆不再对应固定文本片段，而是当前认知状态下的一次动态投影。

作者用统一函数描述不同记忆系统：

[
m_t=f_{\mathcal M}(s_t,\mathcal H,m_{<t})
]

其中：

- (s_t) 是当前环境状态。
- (\mathcal H) 是历史经验集合。
- (m_{<t}) 是之前已经生成或调用的记忆。
- (f_{\mathcal M}) 决定如何构造当前记忆 (m_t)。

对于任务级记忆，(f_{\mathcal M}) 只在任务开始时调用。对于 step-level memory，它在每次 Agent 与环境交互时调用。MemGen 将调用粒度推进到 token generation 内部。

> [!warning] 范式划分存在一定包装成分
> 从长期存储角度看，MemGen 仍然将经验写入了 Weaver 的参数。它规避的是对核心 Reasoner 的直接修改，而非完全摆脱参数化记忆。
>
> 更准确的结构描述是：
>
> [
> \text{辅助参数化长期记忆}
> \text{动态 latent readout}
> ]
>
> 因此，论文与传统 parametric memory 的主要区别在于参数隔离和动态读出位置。

---

### 2. MemGen 总体流程

**直觉**：Reasoner 正常生成文本；Trigger 持续观察生成状态；遇到需要经验支持的位置时，Weaver 临时生成 latent memory，随后 Reasoner 带着这段记忆继续生成。

![[Figure 2.png]]

Figure 2 将系统分成三层：

- **Reasoner (\pi_\theta)**：冻结的基础 LLM，负责显式推理和动作生成。
- **Trigger (\mathcal T)**：判断当前是否需要调用记忆。
- **Weaver (\mathcal W)**：根据当前 hidden states 生成固定长度的 latent memory。

设 Agent 在环境状态 (s_t) 下生成一个动作：

[
a_t=(z_{t,1},z_{t,2},\ldots,z_{t,L_t})
]

在生成第 (j) 个 token 之前，Reasoner 已经产生 hidden-state sequence：

[
H_{t,<j}
========

(h_{t,1},h_{t,2},\ldots,h_{t,j-1})
]

Trigger 根据这些 hidden states 计算调用概率：

[
p_j
===

\sigma\left(
T_{\text{trigger}}(h_{t,1},\ldots,h_{t,j-1})
\right)
]

随后采样二元决策：

[
d_j\sim\operatorname{Bernoulli}(p_j)
]

其中：

- (d_j=0) 表示 `SKIP`。
- (d_j=1) 表示 `INVOKE`。

当 Trigger 选择 `SKIP` 时，Reasoner 按照常规方式继续生成：

[
z_{t,j}
\sim
\pi_\theta(\cdot\mid s_t,z_{t,<j})
]

当 Trigger 选择 `INVOKE` 时，生成过程暂时停止，Weaver 接收相同的 hidden states：

[
M_t
===

# W_{\text{weaver}}(H_{t,<j})

[m_{t,1},m_{t,2},\ldots,m_{t,K}]
]

其中：

[
M_t\in\mathbb R^{K\times d_{\text{model}}}
]

- (K) 是 latent memory token 的数量。
- (d_{\text{model}}) 与 Reasoner 的 hidden dimension 一致。
- 每个 (m_{t,k}) 可以被理解为一个无法直接阅读的 soft token。

生成的 (M_t) 被加入 Reasoner 当前的计算上下文：

[
z_{t,j}
\sim
\pi_\theta(
\cdot\mid
s_t,z_{t,<j},M_t
)
]

此后，Reasoner 继续生成文本，Trigger 继续监控，整个过程可以在单次推理中重复发生。

> [!tip] Aha Moment
> Weaver 生成的不是任务开始前就确定的经验摘要。它接收的是当前推理轨迹形成的 hidden states，因此记忆内容会随着推理进度变化。
>
> 这相当于把记忆读出写成：
>
> [
> M_t=g(\text{经验参数},\text{当前思考状态})
> ]
>
> 当前问题、已经尝试过的步骤和局部失败都可以影响最终生成的记忆。

> [!question] Latent token 的生成机制缺少关键细节
> 论文将 Weaver 定义为：
>
> [
> W:\mathbb R^{j\times D}\rightarrow\mathbb R^{K\times D}
> ]
>
> 并说明它由挂载在基础 LLM 上的 LoRA 实现，但没有完整说明：
>
> - (K) 个输出位置由什么输入 token 或 learnable queries 初始化。
> - Weaver 是一次前向传播并行产生 (K) 个向量，还是进行 (K) 步自回归生成。
> - 输出取自哪些层。
> - Weaver 与 Reasoner 是否共享 KV cache。
> - latent tokens 插入后，原有 hidden states 是否重新计算。
>
> 这些细节决定了方法的实际计算图和可复现性，也是当前方法部分最明显的缺口。

---

## 3. Memory Trigger

**直觉**：频繁注入记忆会增加计算量，也可能干扰原本正确的推理，因此 Trigger 需要学习哪些位置值得暂停并调用 Weaver。

### 3.1 句子边界上的触发

作者没有让 Trigger 在每个 token 后运行，而是定义分隔符集合 (\mathcal D)，例如逗号和句号。只有生成到这些位置时，Trigger 才能调用记忆：

[
p_j=
\begin{cases}
0, & z_j\notin\mathcal D\
T_{\text{trigger}}(H_{t,<j}), & z_j\in\mathcal D
\end{cases}
]

这个设计有两个目的：

- 降低 Trigger 和 Weaver 的调用频率。
- 尽量在语义单元结束后介入，减少 latent memory 对局部句法生成的破坏。

> [!note] 实际调用时间
> 公式使用当前 token (z_j) 是否属于分隔符来决定调用，因此从因果执行顺序看，应当是先生成分隔符，再判断是否为后续 token 注入记忆。
>
> 论文在部分描述中将其称为生成第 (j) 个 token 前的决策，索引存在轻微混乱。更合理的理解是记忆在分隔符之后、下一段推理开始之前插入。

> [!question] 为什么限定在标点位置
> 作者引用了句子边界干预更有效的相关研究，但没有比较其他候选粒度，例如：
>
> - thought step 边界
> - tool call 前后
> - 环境 observation 到达后
> - 熵或置信度突变位置
> - 特定控制 token
>
> 标点符号只是低成本的语义边界代理，在代码、工具调用和结构化输出中未必可靠。

### 3.2 Trigger 的强化学习目标

Trigger 需要同时优化任务奖励和调用成本：

[
\max_{\phi}
\mathbb E_{\tau_i,\tilde d}
\left[
R(\tau_i)
---------

\lambda
\sum_{i,j}
\max(0,\tilde d_{i,j}-\bar p)
\right]
]

其中 (\tilde d_{i,j}\in{0,1}) 是 Trigger 的调用决策。

作者从 batch 中选择奖励不低于中位数的轨迹，并统计这些高奖励轨迹的平均调用率：

[
\bar p
======

\frac{1}{|\mathcal H_{\text{high}}|}
\sum_{i\in\mathcal H_{\text{high}}}
\frac{1}{|\tau_i|}
\sum_j \tilde d_{i,j}
]

[
\mathcal H_{\text{high}}
========================

\left{
i:
R(\tau_i)
\ge
\operatorname{median}_k R(\tau_k)
\right}
]

因为 (\tilde d) 是二元变量，所以单次调用对应的惩罚可以写成：

[
\max(0,1-\bar p)=1-\bar p
]

高奖励轨迹普遍需要频繁调用时，(\bar p) 较大，调用惩罚会减小。高奖励轨迹通常很少调用时，额外调用会承担更高成本。

我认为作者的直觉是合理的：**调用预算应当由成功轨迹中实际需要的记忆频率决定，而不是预先固定一个统一稀疏率。**

> [!warning] 自适应惩罚可能形成移动目标
> (\bar p) 同时由当前策略产生的高奖励轨迹决定。Trigger 改变调用频率后，成功轨迹集合和平均调用率也会变化。
>
> 论文没有分析：
>
> - 这一目标是否容易形成高频调用的自增强状态。
> - batch median 在奖励分布较离散时是否稳定。
> - (\lambda) 对调用频率和性能的敏感性。
> - Trigger 具体使用了哪一种 policy-gradient estimator。
> - 是否加入了熵正则、value baseline 或 KL 约束。

---

## 4. Memory Weaver

**直觉**：Weaver 负责把过去学到的经验重新解释成适合当前推理状态的一小段向量，而不是直接复制一条历史轨迹。

作者将 Weaver 实现为挂载在 Reasoner 上的另一组 LoRA 参数：

[
M_t
===

W_{\theta'}(H_{t,<j})
\in
\mathbb R^{K\times d_{\text{model}}}
]

训练过程中：

- Reasoner 参数 (\theta) 始终冻结。
- Trigger 在 Weaver 训练阶段保持固定。
- 梯度只更新 Weaver 参数 (\theta')。

### 4.1 SFT 训练

**直觉**：让 Weaver 生成的 latent memory 帮助冻结的 Reasoner 更容易复现高质量示范轨迹。

对于专家 token (z^*_{i,t,j})，SFT 损失为：

[
\mathcal L_{\text{SFT}}(\theta')
================================

\mathbb E_{(x_i,\tau_i^*)\sim\mathcal H}
\left[
\sum_t\sum_j
\log
\pi_\theta
\left(
z^**{i,t,j}
\mid
s*{i,t},
z^**{i,t,<j},
M*{i,t,j}
\right)
\right]
]

其中：

[
M_{i,t,j}
=========

W_{\theta'}(H_{i,t,<j})
]

虽然 Reasoner 被冻结，token prediction loss 仍然可以穿过 Reasoner 的计算图，将梯度传回 Weaver。Weaver 因而学习生成能够推动正确 token 概率上升的 latent vectors。

> [!note] SFT 学到的内容
> Weaver 没有显式学习将某条经验压缩成某段 latent memory。它直接接受下游 token loss。
>
> 因此，latent memory 的语义由功能决定：
>
> [
> \text{什么向量能让后续输出更接近示范}
> ]
>
> 这解释了为什么强制解码 latent tokens 后得到的文本通常不可读。

### 4.2 GRPO 训练

**直觉**：当没有唯一专家轨迹时，直接根据最终任务奖励判断 Weaver 生成的记忆是否有效。

对于任务 (x_i)，系统采样一组轨迹：

[
G_i
===

{
\tau_{i,1},
\tau_{i,2},
\ldots,
\tau_{i,K}
}
]

组内平均奖励为：

[
\bar R(G_i)
===========

\frac{1}{K}
\sum_{k=1}^K R(\tau_{i,k})
]

轨迹优势为：

[
A(\tau_{i,k})
=============

R(\tau_{i,k})-\bar R(G_i)
]

随后通过组内相对优势更新 Weaver：

[
J_{\text{GRPO}}(\theta')
========================

\mathbb E
\left[
\frac{1}{K}
\sum_k
A(\tau_{i,k})
\log
\Pi_{\theta}^{W_{\theta'},T}
(\tau_{i,k}\mid x_i)
--------------------

\beta
D_{\mathrm{KL}}
\right]
]

这里的联合策略由三个组件组成：

[
\Pi_{\theta}^{W_{\theta'},T}
============================

\text{Reasoner}
\text{Weaver}
\text{Trigger}
]

实际梯度只更新 Weaver。

> [!warning] GRPO 描述经过明显简化
> 公式中没有展示常见 GRPO 实现中的 probability ratio、clipping 或 token-level advantage 分配。实验超参数还将 (\beta) 设置为 0。
>
> 因此，这部分更接近组相对奖励的 policy-gradient 表述。论文没有提供足够细节判断其与标准 GRPO 实现是否完全一致。
>
> 此外，符号 (K) 同时用于 latent memory 长度和每组 rollout 数量，容易引起混淆。

### 4.3 长期记忆与动态记忆的边界

我注意到，Weaver 同时承担了两个概念上不同的功能：

1. **长期经验存储**

   经验通过 SFT 或 GRPO 写入 Weaver 的 LoRA 参数。

2. **即时记忆构造**

   Weaver 根据当前 hidden states 产生 (K) 个 latent tokens。

因此，可以将 MemGen 写成：

[
\underbrace{\theta'}*{\text{长期经验}}
\xrightarrow[\text{当前状态 }H*{<j}]{W}
\underbrace{M_t}_{\text{即时记忆}}
]

这个分解很重要。论文的动态性主要发生在读出阶段，经验写入仍然依赖离线或阶段性的参数训练。

> [!warning] Self-evolving 的范围需要谨慎理解
> 论文没有展示 Agent 在单次部署过程中自主判断新经验、立即写入 Weaver 并持续更新的完整闭环。
>
> 当前流程更接近：
>
> [
> \text{收集轨迹}
> \rightarrow
> \text{训练 Weaver}
> \rightarrow
> \text{训练 Trigger}
> \rightarrow
> \text{部署推理}
> ]
>
> 它支持基于新数据继续训练，但尚未形成完整的在线 memory write、验证、修正与遗忘机制。

---

## 5. 分阶段训练

**直觉**：先让 Weaver 学会生成有用记忆，再让 Trigger 学习调用时机，可以降低两个模块同时变化造成的训练不稳定。

作者采用两阶段流程。

### 第一阶段：训练 Weaver

此时没有学习好的 Trigger。作者使用两种人工调用策略：

- 在每个标点边界插入 latent memory。
- 随机抽取部分标点边界插入 latent memory。

Weaver 因而能够接触不同推理位置，学习如何根据局部 hidden states 生成记忆。

### 第二阶段：训练 Trigger

- Weaver 固定。
- Reasoner 固定。
- Trigger 根据最终任务奖励学习 `INVOKE` 或 `SKIP`。

主要训练配置包括：

- LoRA rank (r=16)
- LoRA alpha (=32)
- target modules 为 `q_proj` 和 `v_proj`
- learning rate (=10^{-5})
- 训练 2 epochs
- SFT batch size 为 4
- GRPO rollout batch size 和 train batch size 为 8

> [!tip] 分阶段训练的优点
> Trigger 面对的是一个行为稳定的 Weaver。否则 Trigger 需要同时适应不断变化的记忆质量，信用分配会更困难。

> [!warning] 存在调用分布偏移
> Weaver 训练时看到的是全部标点或随机标点，部署时看到的是 Trigger 选择的位置。
>
> 如果 Trigger 偏好某类困难状态，而这些状态在 Weaver 训练阶段出现较少，Weaver 的输出可能并不稳定。
>
> 论文没有比较：
>
> - 联合训练
> - 交替训练
> - Trigger 训练后再次微调 Weaver
> - 基于 Trigger 真实调用分布的第二轮训练

---

## 6. 与 Retrieval-based Memory 结合

**直觉**：外部检索系统负责找到候选事实，Weaver 负责将这些事实与当前推理状态融合并压缩成 latent memory。

当 Trigger 决定调用记忆时，系统将当前已经生成的文本解码为查询：

[
q_{t,j}
=======

\operatorname{Decode}(z_{t,<j})
]

从外部数据库 (\mathcal M_{\text{ext}}) 检索文本片段：

[
C_t
===

\mathcal R(q_{t,j};\mathcal M_{\text{ext}})
]

将检索文本编码为 embedding sequence (E_t)，并与当前 hidden states 拼接：

[
M_t
===

W_{\text{weaver}}
\left(
[H_{t,<j};E_t]
\right)
]

Figure 2 右侧展示了这一过程：Weaver 可以同时读取自身参数中保存的内部经验，以及 ExpeL 等系统检索到的外部文本。

![[Table 8.png]]

在 SmolLM3-3B 上：

- ExpeL 在 ALFWorld 上为 36.18。
- 仅使用 ExpeL 检索内容、关闭 Weaver 内部经验时，MemGen + ExpeL 达到 45.60。
- 同时启用 Weaver 参数记忆后达到 75.90。
- TriviaQA 从 ExpeL 的 46.20 提升到 76.40。
- PopQA 从 28.16 提升到 60.23。

这个结果说明 Weaver 可能具有较强的信息融合能力，但当前实验还无法完全确定增益来自哪里。

> [!warning] 缺少关键对照
> 至少还需要比较：
>
> - 在相同调用位置直接将检索文本加入 Reasoner。
> - 将检索文本压缩成普通文本摘要后加入 Reasoner。
> - 使用相同 Trigger，但不经过 Weaver。
> - 使用 Weaver，但固定在任务开始时调用。
>
> 现有实验同时改变了检索时机、信息编码形式和 Weaver 参数记忆，难以分离每个因素的贡献。

---

# Experiments

## 1. 实验设置

作者在五类任务上进行评估：

- **Web Search**：TriviaQA、PopQA
- **Embodied Action**：ALFWorld
- **Math Reasoning**：AQuA、GSM8K、MATH
- **Scientific Reasoning**：GPQA
- **Code Generation**：KodCode、BigCodeBench

使用三个不同规模的 backbone：

- Qwen2.5-1.5B
- SmolLM3-3B
- Qwen3-8B

基线分为四类：

- Prompt-based：Vanilla、CoT
- Parametric Memory：SFT、GRPO、REINFORCE、REINFORCE++、Agent-FLAN
- Retrieval-based Memory：MemoryBank、ExpeL、AWM
- Latent Computation：SoftCoT、Co-processor

MemGen 包含两个版本：

- MemGen SFT
- MemGen GRPO

> [!warning] Benchmark 数量存在文本不一致
> 摘要中写的是 eight benchmarks，正文实验设置实际列出了九个数据集，Introduction 也使用了 nine benchmarks。
>
> 这是明显的版本编辑问题。

> [!question] Baseline 公平性仍不够透明
> 论文没有在主文中清楚对齐以下资源：
>
> - 每种方法使用的训练样本数量。
> - 每种方法的 rollout 数量。
> - 总训练 FLOPs。
> - 可训练参数量。
> - 外部记忆数据库容量。
> - 检索系统的 token budget。
>
> MemGen 为冻结 Reasoner 增加了 Weaver 和 Trigger，而普通 GRPO 的更新位置和参数规模可能不同。最终结果具有说服力，但严格的同预算比较仍然缺失。

---

## 2. 主实验结果

![[Table 1.png]]

![[Table 3.png]]

整体结果显示，MemGen GRPO 在大多数 benchmark 和 backbone 上取得最佳结果。

几个较有代表性的结果：

- 在 SmolLM3-3B + ALFWorld 上，Vanilla 为 18.96，GRPO 为 55.35，MemGen GRPO 达到 63.60。
- 在 SmolLM3-3B + TriviaQA 上，GRPO 为 65.88，MemGen GRPO 达到 79.30。
- 在 SmolLM3-3B + PopQA 上，GRPO 为 45.16，MemGen GRPO 达到 58.60，绝对提升 13.44。
- 在 Qwen2.5-1.5B + ALFWorld 上，GRPO 为 43.55，MemGen GRPO 达到 54.27。
- 在 Qwen3-8B + ALFWorld 上，GRPO 为 85.60，MemGen GRPO 达到 90.60。
- 在 Qwen3-8B + MATH 上，GRPO 为 83.54，MemGen GRPO 达到 88.24。

这些结果支持两个判断：

1. 将训练信号集中在 Weaver 上，依然可以显著改变冻结 Reasoner 的行为。
2. 动态 latent memory 在较小模型上尤其有效，可能补偿了小模型有限的内部推理和上下文组织能力。

我注意到 MemGen SFT 与 MemGen GRPO 并没有统一的优劣关系：

- Qwen3-8B + GPQA 上，MemGen SFT 为 43.23，MemGen GRPO 为 40.24。
- Qwen2.5-1.5B + GPQA 上，MemGen SFT 为 18.28，MemGen GRPO 为 18.18。
- SmolLM3-3B + GPQA 上，MemGen SFT 为 26.75，MemGen GRPO 为 25.20。

> [!note] 这个反例很重要
> 在知识密集且奖励较稀疏的 GPQA 上，GRPO 没有稳定优于 SFT。
>
> 这提示 Weaver 的强化学习效果可能依赖：
>
> - 奖励密度
> - rollout 多样性
> - backbone 初始能力
> - 任务是否能够通过最终答案准确评价
>
> 论文主要强调最高提升，没有充分讨论这些失效条件。

---

## 3. Cross-domain Generalization

**直觉**：如果 Weaver 学到的是通用推理经验，它在训练域之外也应当产生有帮助的 latent memory。

作者分别在 ALFWorld、TriviaQA、GSM8K 和 KodCode 上训练 MemGen，再测试其他任务。

![[Figure 3.png]]

![[Figure 9.png]]

![[Figure 10.png]]

代表性现象包括：

- 在 KodCode 上训练后，MATH 从 36.6 提升到 54.2。
- 在一个任务上训练的 SFT 和 MemoryBank 经常只提升训练域，部分外部任务明显下降。
- MemGen 在多个训练域外任务上仍然保持正迁移。

作者将这种现象归因于 Trigger：当测试任务与训练经验不匹配时，Trigger 会减少 Weaver 调用，从而避免错误经验干扰 Reasoner。

> [!warning] 通用能力的来源尚未被完全隔离
> 跨域提升可能来自多种因素：
>
> - Weaver 学到通用的推理模式。
> - Weaver 学到统一的答案格式或 Chain-of-Thought 风格。
> - 训练数据中的任务结构存在重叠。
> - Trigger 在不熟悉任务上减少干预，使结果接近原模型。
>
> 当前实验能够证明 transfer 存在，但难以区分 transfer 的具体语义层级。

---

## 4. Trigger 调用频率

作者在 GSM8K 上训练 MemGen，并统计它在 GSM8K、GPQA 和 KodCode 上的调用位置。

![[Figure 4.png]]

在 Qwen3-8B 上：

- GSM8K 平均调用次数为 86.07。
- GPQA 为 46.87。
- KodCode 为 21.02。

这个结果与作者的解释相符：训练域调用最多，较陌生的代码域调用最少。

附录中的其他 backbone 结果提供了更复杂的图景：

![[Figure 7.png]]

![[Figure 8.png]]

- SmolLM3-3B 上，GSM8K 为 74.49，GPQA 为 48.59，KodCode 为 27.96。
- Qwen2.5-1.5B 上，GSM8K 为 75.17，GPQA 为 46.87，KodCode 为 51.70。

> [!warning] 低调用频率与跨域陌生度的关系并不稳定
> Qwen2.5-1.5B 在 KodCode 上的调用频率高于 GPQA，与正文中陌生域调用更少的统一叙述不完全一致。
>
> Trigger 的频率还可能受到以下因素影响：
>
> - 输出长度
> - 标点密度
> - 任务模板
> - Reasoner 的生成风格
> - 每个任务需要的 reasoning steps
>
> 更严谨的指标应当使用每个可触发边界的调用比例，而非每条轨迹的绝对调用次数。

---

## 5. Continual Learning

作者按照以下顺序训练 Qwen2.5-1.5B：

[
\text{AQuA}
\rightarrow
\text{GPQA}
\rightarrow
\text{GSM8K}
\rightarrow
\text{KodCode}
]

每个阶段结束后，在四个任务上重新评价。

![[Table 4.png]]

最终完成 KodCode 训练后：

| 方法         |  AQuA |  GPQA | GSM8K | KodCode |
| ---------- | ----: | ----: | ----: | ------: |
| SFT        | 28.61 |  2.53 | 24.14 |   54.10 |
| ExpeL      | 27.14 |  6.23 | 31.44 |   48.35 |
| MemGen SFT | 40.34 | 20.09 | 53.72 |   52.95 |

这里的结果相当清楚：

- SFT 对最后训练的 KodCode 适应最强，但旧任务严重下降。
- ExpeL 的旧任务保持能力略好，但整体性能有限。
- MemGen 在旧任务和新任务之间取得了更均衡的结果。

> [!tip] 这个实验真正支持的结论
> 将经验更新限制在独立 Weaver 中，确实能够保护冻结 Reasoner 的原始能力。
>
> 同时，Weaver 自身仍然可能遗忘。最终结果说明它在这组任务序列中遗忘较少，但无法说明参数化 Weaver 天然解决了 continual learning。

> [!warning] Continual learning baseline 不充分
> 论文主要比较普通 SFT 和 ExpeL，没有加入专门处理持续学习的方案，例如：
>
> - replay buffer
> - task-specific adapters
> - adapter composition
> - regularization-based continual learning
> - parameter isolation
> - mixture-of-experts memory
>
> Reasoner 冻结本身已经提供了较强的抗遗忘先验，因此需要与同样冻结 backbone 的参数隔离方法比较。

---

## 6. Latent Memory 的几何结构

作者收集推理时生成的 latent memory sequence，并对每段记忆做 mean pooling：

[
\bar m_i
========

\frac{1}{K}
\sum_{l=1}^{K}m_{i,l}
]

随后：

- 使用 t-SNE 将 (\bar m_i) 投影到二维空间。
- 使用 K-means 在原始高维空间聚类。
- 固定聚类数量 (N=4)。

![[Figure 5.png]]

图中可以看到：

- KodCode 与 BigCodeBench 较接近。
- GSM8K 与 MATH 较接近。
- 不同任务形成相对独立的分布。
- 同一数据集内部也出现多个局部簇。

作者还将 latent vectors 强制映射回词表。输出通常不可读，但同一簇会出现类似的词尾模式，例如 `_check`、`_pick` 等。

> [!note] 强制解码的含义
> latent token 是词嵌入空间中的连续向量，并不要求落在某个真实 token 的 embedding 上。
>
> 通过最近邻词表解码得到的字符串只能反映局部几何邻近关系，不能直接视为 latent memory 的真实语义。

> [!warning] t-SNE 只能用于探索
> t-SNE 会主动强化局部聚集效果。不同任务形成簇，可能来自：
>
> - prompt 模板差异
> - 输出格式差异
> - hidden-state 分布差异
> - Reasoner 在不同领域的语言风格
>
> 图形本身无法证明这些簇对应独立的记忆功能。

---

## 7. Human-like Memory Hierarchy

**直觉**：如果不同 latent memory cluster 承担不同认知功能，删除某个 cluster 后，应当出现特定类型的 Agent 错误。

作者在 TriviaQA 上定义八类失败：

- Planning Failure
- Compositional Reasoning
- Tool Response Error
- Tool Parsing Error
- Answer Formatting Failure
- Demand Misunderstanding
- Think-Act Inconsistency
- False Belief

对于新生成的 latent memory (M_t)，作者先计算均值：

[
\bar m_{\text{new}}
===================

\frac{1}{K}
\sum_l m_{t,l}
]

然后将它与四个 cluster centroid 以及词表 embedding 一起比较。若目标 cluster 的 centroid 出现在 top-10 nearest neighbors 中，则丢弃整段 latent memory：

[
\mu_j
\in
S_k(\bar m_{\text{new}})
\Rightarrow
\text{discard }M_t
]

之后观察各类错误数量如何变化。

![[Figure 6.png]]

作者得到的映射为：

- **Cluster 2：Planning Memory**

  删除后 planning failure 和 compositional reasoning error 明显增加。

- **Cluster 3：Procedural Memory**

  删除后 tool response、tool parsing 和 answer formatting error 增加。

- **Cluster 1 与 Cluster 4：Working Memory**

  删除后 demand misunderstanding 和 think-act inconsistency 增加。

> [!tip] 干预比单纯可视化更有价值
> 作者没有仅凭 t-SNE 图给 cluster 命名，而是尝试删除相关记忆并观察行为变化。这至少提供了一定的因果证据，是论文分析部分最有价值的设计。

> [!warning] Human-like hierarchy 的结论明显偏强
> 这套分析仍有若干关键限制：
>
> - K-means 的 cluster 数量预先固定为 4。
> - 记忆序列被 mean pooling，token 顺序和内部结构全部丢失。
> - 错误类型由人工标注，每条失败轨迹只允许一个标签。
> - cluster 名称根据删除后的错误变化事后赋予。
> - 删除的是整段 memory sequence，而非精确删除某个功能 token。
> - 一个 cluster 会同时影响多个错误类型。
> - Cluster 1 和 Cluster 4 被共同解释为 working memory，说明 cluster 与功能并非一一对应。
> - 没有随机方向、随机 cluster 或等量删除作为严格对照。
>
> 因此，更稳妥的结论是：**latent memory 中出现了与若干行为功能相关的可分结构。**
>
> 当前证据还不足以支持完整的 human-like memory hierarchy。

---

## 8. Trigger Ablation

作者比较三类调用策略：

- 随机 token 位置调用
- 所有分隔符位置调用
- 训练得到的 Trigger

![[Table 5.png]]

| 调用策略            |  GPQA | KodCode | TriviaQA |
| --------------- | ----: | ------: | -------: |
| Random (p=0.2)  | 15.66 |   54.55 |    63.55 |
| Random (p=0.5)  | 16.66 |   52.95 |    57.28 |
| Random (p=0.8)  | 12.63 |   53.60 |    62.22 |
| All delimiters  | 17.34 |   56.20 |    64.15 |
| Learned Trigger | 18.28 |   58.16 |    65.02 |

结果支持两个结论：

1. 在语义边界调用优于任意 token 位置调用。
2. 学习调用位置优于在所有语义边界统一调用。

不过 Learned Trigger 相比 All delimiters 的绝对提升为：

- GPQA：0.94
- KodCode：1.96
- TriviaQA：0.87

> [!warning] Trigger 有效，但增益幅度有限
> 大部分收益似乎已经来自句子边界上的 latent insertion。学习得到的 Trigger 提供了进一步提升和潜在计算节省，但其性能贡献小于论文整体叙述给人的印象。
>
> 还需要报告：
>
> - 两种策略的平均调用次数。
> - 相同调用预算下的性能。
> - Trigger 本身的额外计算开销。
> - 固定调用次数、只优化位置的对照。

---

## 9. Memory Length 与 Weaver Capacity

Figure 6 左侧比较了 latent token length：

[
K\in{2,4,6,8,16,32}
]

随着 (K) 增大，KodCode 和 TriviaQA 性能总体上升。

这符合直觉：更多 latent tokens 提供了更大的信息通道。

> [!question] 容量增加带来的收益是否等价于更好的记忆
> 增大 (K) 同时增加：
>
> - 可表达的信息量
> - attention context 长度
> - Weaver 输出维度
> - 推理计算量
>
> 论文没有进行等计算预算或等参数容量比较，因此目前只能得出更长 latent sequence 通常更有效。

Table 6 比较 LoRA Weaver 和 Full SFT Weaver：

| Weaver 参数化 |  GPQA | KodCode | TriviaQA |
| ---------- | ----: | ------: | -------: |
| LoRA       | 18.28 |   58.16 |    65.02 |
| Full SFT   | 21.21 |   60.00 |    67.10 |

完整参数微调稳定优于 LoRA，说明 Weaver 容量确实会限制记忆表达。

---

## 10. Efficiency

![[Table 7.png]]

作者报告 MemGen 相比 Vanilla LLM 的总任务时间更短。例如：

- Qwen2.5-1.5B + KodCode：Vanilla 为 11.96 秒，MemGen 为 2.94 秒。
- SmolLM3-3B + TriviaQA：Vanilla 为 4.26 秒，MemGen 为 3.16 秒。
- Qwen3-8B + ALFWorld：Vanilla 为 55.42 秒，MemGen 为 20.08 秒。

作者解释，经过训练的模型需要生成更少 token 即可得到正确答案，所以总耗时下降。

我认为这里需要区分两种效率：

1. **任务完成效率**

   从问题输入到最终输出的总时间。MemGen 在该指标上通常快于 Vanilla。

2. **机制额外开销**

   在相同输出长度和相同任务轨迹下，Trigger 与 Weaver 增加了多少计算。

Table 7 实际显示 MemGen 始终慢于对应 SFT：

- Qwen2.5 KodCode：2.01 秒到 2.94 秒。
- SmolLM3 TriviaQA：3.05 秒到 3.16 秒。
- Qwen3 ALFWorld：19.76 秒到 20.08 秒。

> [!warning] 低于 Vanilla latency 无法直接证明记忆插入没有开销
> 总时间下降主要来自生成轨迹变短。要评价机制本身，应控制：
>
> - 输出 token 数
> - tool call 数
> - 环境 step 数
> - Trigger 调用次数
> - latent memory 长度
>
> 更合适的指标包括每输出 token latency、每次 Weaver 调用耗时和 KV cache 增量。

---

## 11. 总体实验判断

> [!tip] 论文最扎实的证据
>
> - 三个不同规模 backbone 上均有提升。
> - 覆盖搜索、具身、数学、科学和代码任务。
> - 与普通 SFT、GRPO、外部记忆和 latent computation 方法进行了较全面比较。
> - 冻结 Reasoner 后仍能产生显著性能变化。
> - Trigger、Weaver 长度和 Weaver 参数量均有对应消融。
> - 通过删除 cluster 分析 latent memory 与行为错误的关联。

> [!warning] 论文最需要谨慎对待的部分
>
> - Weaver 的具体 latent token 生成结构描述不完整。
> - 长期记忆本质上仍位于辅助参数中。
> - Self-evolving 尚未覆盖完整在线写入闭环。
> - Human-like memory hierarchy 的解释明显超出实验能够严格支持的范围。
> - Trigger 的独立增益较小。
> - 效率实验受到输出长度变化的混淆。
> - Baseline 的参数量、训练计算量和数据预算没有完全对齐。
> - 跨域与持续学习实验缺少更强的专门 baseline。

---

# Future

## 1. 明确区分 memory write、storage 与 readout

当前 MemGen 将经验训练和 latent generation 都放入 Weaver。后续可以显式拆成：

[
\text{Experience Encoder}
\rightarrow
\text{Persistent Memory State}
\rightarrow
\text{Query-conditioned Decoder}
]

这样可以分别研究：

- 新经验如何写入。
- 冲突经验如何修正。
- 旧经验如何保留。
- 当前查询如何读出。
- 生成记忆如何追溯到原始证据。

## 2. 建立真正的在线自演化闭环

当前 Weaver 依赖离线 SFT 或 GRPO。更完整的 Agent Memory 应当支持：

[
\text{Interaction}
\rightarrow
\text{Outcome Evaluation}
\rightarrow
\text{Credit Assignment}
\rightarrow
\text{Memory Update}
\rightarrow
\text{Immediate Reuse}
]

关键问题包括：

- 哪些轨迹值得写入。
- 失败轨迹应被删除、反转还是保留为负面经验。
- 单条经验能否低成本更新 Weaver。
- 如何阻止错误经验污染后续任务。
- 如何回滚有害更新。

## 3. 联合优化 Trigger 与 Weaver

当前两阶段训练保证稳定性，但可能得到次优解。

值得比较：

- 交替训练
- 双时间尺度更新
- Trigger 和 Weaver 的 actor-critic 联合优化
- 基于调用价值的 advantage decomposition
- Weaver 更新后重新校准 Trigger

可以将一次 memory invocation 的价值定义为反事实差异：

[
\Delta V_j
==========

# R(\tau\mid\text{invoke at }j)

R(\tau\mid\text{skip at }j)
]

这样 Trigger 学习的是调用的边际收益，而非整条轨迹的最终奖励。

## 4. 动态决定记忆长度

当前 (K) 固定。Figure 6 显示更长记忆通常更有效，但不同推理节点需要的信息量明显不同。

可以令 Weaver 同时预测：

[
(K_t,M_t)
=========

W(H_{<j})
]

简单问题输出 1 至 2 个 latent tokens，复杂规划问题输出更长序列，并在奖励中加入 token budget。

## 5. 从分隔符触发转向状态变化触发

更合理的 Trigger 信号可能包括：

- predictive entropy
- hidden-state novelty
- value uncertainty
- repeated reasoning pattern
- tool failure
- observation conflict
- progress stagnation
- plan boundary

这会使调用时机与认知状态直接相关，减少对语言表面标点的依赖。

## 6. 研究可纠正的 latent memory

Weaver 目前只能生成记忆，缺少显式纠正机制。

值得加入：

- stale-memory detection
- contradiction detection
- negative memory
- confidence estimation
- versioned memory state
- evidence-grounded latent generation

一个重要方向是让 Agent 判断：

> [!info] 这段记忆是否仍然适用于当前环境

而非默认历史经验始终有效。

## 7. 更严格地验证功能分工

Human-like memory 的验证可以升级为：

- 多个随机种子重复聚类。
- 自动选择 cluster 数量。
- 使用线性 probe 预测错误类型。
- 使用 causal mediation analysis。
- 与随机方向和等量删除比较。
- 对单个 latent token 进行精细干预。
- 检查同一功能是否能够跨任务、跨 backbone 对齐。
- 验证 cluster 是否具有组合性和可替换性。

只有功能在不同数据集和模型间稳定复现，才可以进一步讨论 memory hierarchy。

## 8. 测试跨模型可迁移性

Weaver 的 latent dimension 和 embedding geometry 绑定到具体 backbone。当前方法中的记忆难以直接从 Qwen2.5 转移到 Qwen3 或其他模型。

后续可以研究：

[
\text{Backbone-independent Memory Space}
\rightarrow
\text{Model-specific Adapter}
]

这将检验方法学到的是可迁移经验，还是特定模型 hidden space 中的控制向量。

## 9. 与外部证据建立可追溯关系

latent memory 不可读会带来审计问题。尤其在搜索和工具 Agent 中，需要知道某次决策依赖了哪些经验或外部证据。

可以让 Weaver 同时输出：

- latent memory
- source attribution
- confidence
- retrieved evidence IDs
- memory provenance graph

这样既保留 latent readout 的表达能力，也能支持错误诊断和安全审计。

## 10. 更严格的计算预算比较

后续实验应统一报告：

- trainable parameters
- training tokens
- rollout 数量
- GPU hours
- memory database size
- retrieval token budget
- average invocation rate
- latent tokens per task
- latency per generated token

这会更准确地回答 MemGen 的提升究竟来自记忆架构、额外参数、更多训练计算，还是更短的推理轨迹。
