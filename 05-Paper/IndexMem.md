# IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference

ICML-2026  

**核心一句话**：作者把 KV Cache 压缩拆成两条信息通路，用可学习索引器保留未来可能被访问的关键 KV，同时把被删除 KV 压缩进固定大小的在线潜在记忆，并通过残差读出补回被删除的注意力贡献。

---

## Key Contribution

- *A learnable indexer predicts the importance of KV tokens and enables adaptive KV eviction.*
- *A fixed-size latent memory compresses evicted tokens and is updated online during inference.*
- *IndexMem improves long-context performance on RULER, NIAH and LongBench across Qwen, Mistral and Llama backbones.*

从技术逻辑上看，这篇论文处理了 KV eviction 中两个相互关联的问题：

- **选择误差**：传统方法依赖局部注意力、Key 范数或预设查询分布，很难准确判断某个 token 是否会在后续生成中重新变得重要。
- **删除误差**：即使选择器整体准确，硬删除依然会产生不可恢复的信息损失，尤其影响多证据问答、摘要和跨段推理。

作者分别用两个模块处理这两个误差：

1. **Indexer** 学习一个低成本的注意力代理，决定哪些 KV 需要精确保留。
2. **Latent Memory** 对被删除 KV 进行连续压缩，保留其近似贡献。

> [!tip] 核心直觉
> IndexMem 实际构造了两级上下文表示：
>
> - 稀疏 KV Cache 保存少量、精确、可直接检索的信息。
> - 潜在记忆保存大量、压缩、近似的信息。
>
> 这种分工很合理。精确检索需要原始 KV，分散在大量低显著性 token 中的背景信息则可以通过连续状态近似保存。

---

## Method

### 问题定义与整体框架

**作者的直觉是：KV Cache 必须保持固定预算，但删除决策不能继续依赖静态启发式规则。**

自回归生成会缓存所有历史 token 的 Key 和 Value。缓存长度随上下文线性增长，长上下文推理中的主要资源瓶颈逐渐从计算转向显存和内存带宽。作者因此选择直接限制 token 数量，即只保留评分最高的 KV。

一般的 KV eviction 可以写成：

$$
S=f(K,V,Q),\qquad
I=\operatorname{TopK}(S,L')
$$

$$
[K',V']=\operatorname{Gather}(K,V,I)
$$

其中：

- $S_t$ 表示第 $t$ 个历史 token 的重要性。
- $L'$ 是允许保留的 KV 数量。
- $I$ 是被保留 token 的位置集合。
- 不同方法的主要区别集中在评分函数 $f$。

SnapKV 根据最近一段查询窗口的注意力分数判断重要性，容易形成近邻偏置；KeyDiff 根据 Key 之间的差异估计信息量；Expected Attention 使用语料级查询统计估计未来注意力。这些代理信号都包含较强假设。

![[98_Assets/IndexMem.png]]

Figure 1 展示了完整数据流：

1. Indexer 为完整上下文中的 token 计算重要性。
2. TopK token 进入主注意力流并保留原始 KV。
3. 其余 KV 被删除，同时写入潜在记忆状态 $(M,b)$。
4. 当前查询从保留 KV 中获得 $o_{\text{attn}}$。
5. 同一查询从潜在记忆中读取 $m(q)$。
6. 两条路径通过门控残差相加。

> [!question] 选择器真的能够预测未来重要性吗
> Indexer 在 prefill 阶段使用当前 prompt 中的全部查询，在 decoding 阶段使用上一次压缩以来已经出现的查询。它仍然没有直接观察未来生成产生的查询。
>
> 因而，这里的预测更接近**根据已有查询模式估计后续访问概率**。当后续任务目标突然改变，或者生成过程引入新的关注对象时，历史查询分布可能无法覆盖未来需求。

---

### Learnable Indexer

#### 设计直觉

**作者希望训练一个尺寸远小于主注意力层的代理模型，让它模仿主模型对历史 token 的关注模式。**

Indexer 接收隐藏状态：

$$
X\in\mathbb R^{L\times d_{\text{model}}}
$$

以及主模型产生的 pre-RoPE Query：

$$
Q\in\mathbb R^{H\times L\times d_{\text{head}}}
$$

输出查询到历史 token 的评分矩阵：

$$
A=\operatorname{Indexer}(X,Q)\in\mathbb R^{L\times L}
$$

其中 $A_{s,t}$ 表示对于位置 $s$ 的查询而言，位置 $t$ 的 Key 有多重要。

![[98_Assets/IndexMem-1.png]]

#### 轻量 Query-Key 表示

Indexer 首先压缩主模型的多头 Query：

$$
q_s=U_q\operatorname{flatten}(Q_s)
\in\mathbb R^{H_{\text{index}}d_{\text{index}}}
$$

并将其重新整理为：

$$
q_s\in\mathbb R^{H_{\text{index}}\times d_{\text{index}}}
$$

历史 token 的 Key 特征直接从隐藏状态投影得到：

$$
k_t=U_kX_t\in\mathbb R^{d_{\text{index}}}
$$

作者采用类似 MQA 的结构，所有 Indexer Query Head 共享同一个 $k_t$。同时满足：

$$
H_{\text{index}}\ll H,\qquad
d_{\text{index}}\ll d_{\text{head}}
$$

这种设计牺牲了一部分 Head 独立性，换取较低的评分成本。Query 和 Key 在相似度计算前经过 RMSNorm，避免向量范数主导评分。

作者在实验中设置：

$$
H_{\text{index}}=\frac H4,\qquad
d_{\text{index}}=\frac{d_{\text{head}}}{8}
$$

因此 Indexer 的相似度计算维度显著小于主注意力。

> [!question] 为什么使用 pre-RoPE Query
> 原文明确指出输入是 pre-RoPE Query，但没有解释位置编码前的表示为什么更适合 token 保留，也没有比较 pre-RoPE、post-RoPE 和隐藏状态 Query。
>
> 我推测作者希望重要性更多反映内容匹配，降低绝对位置旋转对相似度的影响，但论文没有用实验验证这一点。

#### Query-dependent Head Gate

不同 Indexer Head 的重要性由隐藏状态动态决定：

$$
\alpha_s=
\frac{GX_s}
{\sqrt{H_{\text{index}}d_{\text{index}}}}
\in\mathbb R^{H_{\text{index}}}
$$

随后计算各个 Head 的 Query-Key 相似度：

$$
z_{s,t}=\operatorname{act}(q_s,k_t)
$$

并通过门控加权：

$$
A_{s,t}=
\alpha_s^\top z_{s,t}
+
\operatorname{Mask}_{s,t}
$$

这里的 $\alpha_s$ 允许模型根据当前位置的隐藏状态调整各个轻量 Head 的贡献。某些 Head 可以偏向局部依赖，另一些 Head 可以偏向实体、数字或远距离信息。

> [!warning] Indexer 架构缺少分解消融
> 论文没有分别移除以下组件：
>
> - Head Gate $\alpha$
> - QK-Norm
> - MQA 共享 Key
> - 激活函数
> - pre-RoPE Query
>
> 因此，实验只能证明完整 Indexer 有效，无法判断性能主要来自监督训练，还是来自某个具体结构选择。

#### Max 聚合与 Token Importance

作者将查询维度上的评分取最大值：

$$
\operatorname{imp}_t= \max_{s\in\mathcal Q}A_{s,t}
$$

prefill 阶段的 $\mathcal Q$ 包含 prompt 中的全部查询；decoding 阶段则包含当前压缩区间内产生的查询。

这个操作表达了一种偏向召回率的规则：

**只要某个 token 对任意一个查询非常重要，就应该保留它。**

平均值会压低只被少量查询访问的证据，最大值可以保护稀疏依赖，例如姓名、数字、代码变量和跨段实体。

> [!warning] Max 聚合也会放大异常值
> 单个噪声查询或异常高分可能使 token 长时间留在缓存中。论文没有比较 mean、top-$p$ mean、log-sum-exp 或分位数聚合。
>
> 这里的选择符合 NIAH 等稀疏检索任务，但对摘要等分布式信息任务是否最优仍不明确。

#### Attention Distillation

**作者的直觉是：主模型自己的注意力可以充当 token 重要性的教师。**

教师注意力 Logit 为：

$$
T=\frac{QK^\top}{\sqrt d}
$$

教师和 Indexer 都先在 Query 维度上取最大值，再在 Key 维度归一化：

$$
p_T= \operatorname{softmax}
\left(
\max_qT(q,\cdot)
\right)
$$

$$
p_A= \operatorname{softmax}
\left(
\max_qA(q,\cdot)
\right)
$$

Indexer 的训练目标为：

$$
\mathcal L_{\text{index}}= D_{\mathrm{KL}}(p_T\Vert p_A)
$$

训练时冻结主模型，只更新 Indexer。作者还将前几个 Attention Sink 位置从 KL Loss 中排除，防止这些固定高注意力 token 主导梯度。

> [!question] 教师目标是否等价于未来价值
> 教师信号描述 token 在当前序列中实际获得的注意力，KV eviction 需要估计 token 对未来生成的价值。
>
> 两者高度相关，但并不完全等价。一个 token 在 prompt 内没有被强烈关注，仍可能在回答阶段成为关键证据。论文缺少针对这种分布偏移的专门分析。

#### Streaming KL

直接保存完整 $L\times L$ 教师矩阵会产生二次显存开销。作者采用类似 FlashAttention 的分块过程：

1. 按块遍历 Query。
2. 按块遍历 Key。
3. 只维护每个 Key 的 running maximum。
4. 得到长度为 $L$ 的教师和学生重要性向量。
5. 在 Key 维度执行 Softmax 和 KL。

这样，中间状态从 $O(L^2)$ 降为 $O(L)$。

附录伪代码明确展示了 Query Block 和 Key Block 的双层循环。

> [!warning] 显存复杂度下降，评分计算仍然是成对计算
> Streaming 避免了 $L\times L$ 矩阵的物化，但仍需计算 Query-Key Block 之间的相似度。
>
> 因此，该模块的核心收益依赖较小的 $d_{\text{index}}$、较少的 Indexer Head，以及后文的跨层 Index 复用。论文没有完整分解 Indexer FLOPs、Kernel 时间和数据搬运成本。

#### Pre-eviction

作者声称 Indexer 可以先预测应保留的 KV，随后只为选中位置计算和缓存 KV，从而实现 pre-eviction。

> [!question] Pre-eviction 的执行时序不够清楚
> Indexer 本身需要隐藏状态 $X$ 和主模型 Query $Q$。这意味着模型至少已经执行到能够生成这些状态的位置。
>
> 原文没有清楚说明：
>
> - Indexer 是在每层内部提前决定下一层缓存，还是在完整 prefill 后统一决定
> - 哪部分 KV 计算可以真正省去
> - 是否只节省缓存写入和后续读取
>
> Figure 7 主要报告 decoding 时间，没有单独报告 prefill 延迟，这使 pre-eviction 的系统收益难以确认。

---

### Latent Memory for Evicted Tokens

#### 设计直觉

**作者认为硬删除无法避免错误，因此被删除信息需要一个低成本的兜底通道。**

对于 NIAH 一类任务，答案可能集中在少量证据 token 中，保留这些 token 即可完成任务。对于摘要、多跳问答和整体理解任务，有用信息会分散在大量低显著性 token 中。激进 eviction 会持续累积信息损失。

#### 为什么不使用 Memory-as-Tokens

一种直接方案是把被删除 KV 压缩成少量潜在 token，再与正常 KV 一同进入 Softmax Attention。

作者指出这一方案存在三个问题：

- 放在序列前部时，潜在 token 容易形成 Attention Sink。
- 放在序列后部时，较早位置的 Query 无法访问它。
- 将压缩记忆放入 Softmax 后，较小的重构误差可能被归一化过程放大。

潜在 token 还可能退化成对所有输入相似的平均摘要。

#### Residual Compensation

作者让潜在记忆绕开主注意力 Softmax，直接预测被删除信息造成的输出残差：

$$
o
=

o_{\text{attn}}
+
g(q)m(q)
$$

其中：

- $o_{\text{attn}}$ 只由保留 KV 计算。
- $m(q)$ 是潜在记忆针对当前 Query 的读出。
- $g(q)\in[0,1]$ 控制是否使用记忆。
- 当记忆不可靠时，模型可以令 $g(q)\approx0$。

> [!tip] 残差补偿是本文最精妙的设计
> 记忆模块没有尝试重建完整 KV，也没有参与 Softmax 竞争。它只学习主注意力由于 eviction 缺失的那一部分输出。
>
> 这个目标更直接，允许潜在记忆保持近似表达，同时避免压缩状态改变保留 KV 之间的注意力归一化关系。

#### Slow Weights 与 Fast Weights

每个 Transformer Layer 配置一个潜在记忆模块，并在该层的所有 Attention Head 之间共享。

模块包含：

- **Slow Weights**：投影 $\operatorname{Linear}_\theta$ 和 Gate $g$，通过训练更新。
- **Fast Weights**：状态矩阵 $M$ 和归一化向量 $b$，在推理过程中在线更新。

作者指出，仅使用 Slow Weights 容易学习成与数据集相关的平均残差，无法保存当前输入的具体内容。Fast Weights 提供输入相关的在线状态。

令：

$$
\phi(q)=\operatorname{Linear}_\theta(q) \in\mathbb R^{d_{\text{mem}}}
$$

潜在记忆读出为：

$$
m(q) = \frac{\phi(q)^\top M}
{\left(\phi(q)^{\odot2}\right)^\top b+\epsilon}
$$

其中：$M\in\mathbb R^{d_{\text{mem}}\times d_{\text{model}}},b\in\mathbb R^{d_{\text{mem}}},\phi(q)^{\odot 2}=\text{Linear}_{\theta}(q) \odot\text{Linear}_{\theta}(q)$。

分子 $\phi(q)^\top M$ 根据当前 Query 从关联状态中读取 Value 信息；分母使用平方特征和 $b$ 对读出幅度进行归一化，避免频繁写入的特征方向产生过大输出。

#### Online Write

对于本轮被删除的 KV 集合：

$$
E={(k_i,v_i)}_{i\in E}
$$

状态矩阵通过外积更新：

$$
M
\leftarrow
\lambda M
+
\eta
\sum_{i\in E}
\phi(k_i)v_i^\top
$$

归一化状态更新为：

$$
b
\leftarrow
\lambda b
+
\eta
\sum_{i\in E}
\phi(k_i)\odot\phi(k_i)
$$

其中：

- $\lambda$ 控制历史记忆衰减。
- $\eta$ 控制当前写入强度。
- $\phi(k_i)v_i^\top$ 将 Key 特征与 Value 内容绑定。
- 固定大小的 $M$ 不随上下文长度增长。

从公式上看，$M$ 可以理解为一个在线更新的 Key-to-Value 关联表。查询与某些已删除 Key 的特征相似时，对应 Value 会通过矩阵乘法被重新读出。

> [!warning] Head 信息可能在写入前被压缩
> 附录伪代码注明，写入记忆的 evicted Value 会在每个 token 内跨 Head 求和，再进入 $M$。
>
> 这会降低状态规模，同时可能丢失 Head-specific 语义。论文没有比较：
>
> - 每个 Head 独立记忆
> - Grouped Head 记忆
> - 所有 Head 共享记忆
>
> 因此还无法判断共享设计的精度损失。

#### Memory Training

完整注意力输出为 $o$，压缩 KV 后的注意力输出为 $o_{\text{attn}}$。潜在记忆需要拟合两者之间的差：

$$
\mathcal L_{\text{mem}}= \left| o-o_{\text{attn}}-g(q)m(q) \right|_2^2
$$

目标残差为：

$$
r(q)=o-o_{\text{attn}}
$$

这使记忆模块直接学习 eviction 引起的局部输出误差。

> [!question] 总训练目标没有完整给出
> 方法部分为 Indexer 定义了 pooled KL，为 Memory 定义了 MSE；实验部分又提到 LongAlpaca 上的 chunk-wise KL 和两阶段联合训练。
>
> 原文没有给出最终联合 Loss，例如：
>
> $$
> \mathcal L= \mathcal L_{\text{index}} + \beta\mathcal L_{\text{mem}}
> $$
>
> 也没有说明两个目标是否同时优化、权重如何设置，以及第二阶段是否继续使用 Indexer KL。这会直接影响复现。

> [!warning] 参数量与公式维度难以对齐
> 作者设置 $d_{\text{mem}}=d_{\text{model}}/8$，并将 $\operatorname{Linear}_{\theta}$ 写成从 $d_{\text{model}}$ 到 $d_{\text{mem}}$ 的投影。按稠密矩阵理解，单层投影参数量为：
>
> $$
> d_{\text{model}}d_{\text{mem}}=\frac{d_{\text{model}}^2}{8}
> $$
>
> 论文随后报告整个 Memory Module 只增加 0.52M 参数。
>
> 两处描述无法直接对齐。实现可能使用了共享、分组、低秩或无参数映射，但正文与伪代码没有说明。

---

### Training and Inference Schedule

#### 两阶段训练

作者冻结 Backbone，并采用两阶段训练：

1. 先单独训练 Indexer，使其学习稳定的 token 保留决策。
2. 再联合训练 Indexer 和 Memory，让潜在记忆补偿被删除注意力。

训练数据使用 LongAlpaca，采用 DDP。学习率使用 WSD 调度：

- Warmup 100 Steps，升至 $10^{-3}$
- Stable 2000 Steps
- Decay 2000 Steps，降至 $7.5\times10^{-6}$

#### Prefill 与 Decoding

作者主要评估 **long-prefill、short-decode** 场景：

- Prefill 完成后执行一次性压缩。
- 删除比例为 $r$，保留 $(1-r)L$ 个 KV。
- 被删除 KV 同时写入潜在记忆。
- Decoding 每生成 $\tau=128$ 个 token 再压缩一次。
- 每轮根据该区间内出现的 Query 重新评分。
- KV 数量限制在固定预算 $B_{\max}$ 内。

所有方法都强制保留最前面的四个 Sink Token，减少因 Attention Sink 被删除引起的退化。

> [!warning] 评测场景与长 CoT 应用存在距离
> 论文重点针对长 Prefill 和较短 Decoding。长链式推理会持续产生新的 KV，并多次触发压缩和记忆覆盖。
>
> 作者在限制部分也承认训练规模、Token Budget 和模型覆盖范围仍然有限。
>
> 因而，当前结果尚不足以证明 Fast Weight Memory 在数千到数万生成 token 中能够长期稳定工作。

---

### Cross-layer Score Redundancy and Index Reuse

这一部分位于附录，属于降低 Indexer 开销的扩展分析。

#### 设计直觉

**作者观察到相邻 Transformer Layer 往往会选中一部分相同 token，因此没有必要在每一层完全独立地重新评分。**

部分 Layer 的评分接近均匀分布，TopK 容易受微小扰动影响；另一些 Layer 会形成明显的稀疏峰值。作者尝试跨层聚合以降低单层噪声。

#### Running Mean

对于第 $\ell$ 层的 token 分数 $s_\ell$：

$$
\bar s_m^{\text{naive}}=\frac1m
\sum_{\ell=1}^m s_\ell
$$

跨层平均可以降低方差，但也可能把高质量的尖峰信号与低质量的均匀信号混合。对于只依赖少量关键 token 的任务，这种平滑可能压低真正证据的相对优势。

#### Entropy-gated Running Mean

作者使用归一化熵衡量当前 Layer 的评分是否具有区分度：

$$
H(p) = -\frac1{\log T} \sum_{t=1}^T p_t\log(p_t+\epsilon)
$$

- 高熵表示分数接近均匀，当前层缺少明确选择。
- 低熵表示分数集中在少量 token 上。

通过阈值 $\gamma$ 决定是否纳入平均：

$$
\alpha_\ell = \mathbb I[H_\ell\le\gamma]
$$

$$
\bar s^{\text{skip-high}}= \frac{ \sum_{\ell=1}^{N_{\text{layer}}}\alpha_\ell s_\ell }{ \sum_{\ell=1}^{N_{\text{layer}}}\alpha_\ell+\delta }
$$

该策略在多个压缩率下优于直接 Layer Mean。

![[98_Assets/IndexMem-2.png]]
![[98_Assets/IndexMem-3.png]]

#### IndexCache-style Reuse

最终的实用方案没有继续平均所有层，而是每四个相邻 Layer 只计算一次 Indexer Score，并在组内复用评分或保留索引。这样可以摊薄评分开销。

> [!warning] Appendix 结果无法与主表直接比较
> 作者明确说明，IndexMem + IndexCache 使用了更新后的 WSD 训练方案；主表中的 Mistral IndexMem 使用固定学习率，也没有 IndexCache。
>
> 因此，Table 3 的提升同时包含训练方案和跨层复用两项变化，无法单独归因给 IndexCache。

> [!question] 熵是否真的代表评分正确性
> 低熵只表示模型对少量 token 给出了高分，无法保证这些 token 正确。高熵也可能表示任务确实需要分布式信息。
>
> 作者在附录中承认这一点，并将 Running Mean 定位为诊断实验。

---

## Experiments

### Evaluation Setup

作者使用三个 7B 至 8B Backbone：

- Qwen3-8B
- Mistral-7B-v0.3
- Llama-3.1-8B-Instruct

主要 Baseline 包括：

- Expected Attention
- KeyDiff
- TOVA
- SnapKV
- PyramidKV

所有方法在 KVPRESS 框架中实现，主要实验使用单张 NVIDIA H800。

> [!note] Compression Ratio
> $r$ 表示被删除 KV 的比例：
>
> $$
> L_{\text{kept}}=(1-r)L
> $$
>
> $r=0.9$ 意味着只保留原始 KV token 的 10%。

---

### RULER

RULER 使用 String Match 对各子任务评分，最终结果是所有子任务的无权平均。

![[98_Assets/IndexMem-4.png]]

主要趋势很清楚：

- 在 $r\le0.25$ 时，IndexMem 基本接近 Full Cache。
- 在 $r\ge0.5$ 后，启发式方法快速退化，IndexMem 的下降更加平缓。
- 较长的 RULER-16K 更能体现学习式选择器的优势。

部分代表结果：

| Backbone     |                     设置 | IndexMem |                             代表性对比 |
| ------------ | ---------------------: | -------: | --------------------------------: |
| Qwen3-8B     | RULER-16K，90% eviction |     56.0 |          KeyDiff 53.1，SnapKV 26.8 |
| Mistral-7B   | RULER-16K，90% eviction |     52.9 |             TOVA 47.7，SnapKV 21.3 |
| Llama-3.1-8B |  RULER-4K，90% eviction |     55.2 | TOVA 37.5，Expected Attention 30.6 |

> [!warning] IndexMem 并未在每个设置中获胜
> 在 Llama-3.1-8B 的 RULER-16K、90% eviction 下，TOVA 得分为 59.2，IndexMem 为 56.0。
>
> 论文的整体趋势仍然成立，但 **consistently the most robust** 不应理解为所有表格单元都达到第一名。

> [!warning] 无权平均可能掩盖任务结构
> RULER 将不同子任务等权平均。Indexer 更擅长稀疏检索还是多证据聚合，单一总体分数无法回答。
>
> 更有价值的分析应当按任务依赖类型划分，例如单 Needle、多 Needle、变量追踪和聚合推理。

---

### Needle-in-a-Haystack

![[98_Assets/IndexMem-5.png]]

Figure 3 在 Llama-3.1-8B、50% eviction 下改变上下文长度和 Needle 位置。

观察结果：

- SnapKV 和 PyramidKV 出现明显的位置依赖失效。
- Indexer Only 整体较稳定，但仍有少量孤立的灾难性错误。
- 加入 Latent Memory 后，孤立失败显著减少。

作者据此认为潜在记忆主要改善最坏情况，而非简单提升平均检索分数。

> [!tip] 这里的消融证据比较有说服力
> Indexer Only 与完整 IndexMem 的大部分区域都表现良好，差异集中在少量失败点。潜在记忆表现得像一个错误恢复机制，这与其残差补偿设计相符。

> [!warning] 可视化范围仍然有限
> 热力图只展示一个模型和一个压缩率。若不同压缩率、随机 Needle 内容或多 Needle 条件下结论变化，当前图无法反映。

---

### LongBench

![[98_Assets/IndexMem-6.png]]

LongBench 更接近整体长上下文理解。作者在 HotpotQA、MultiFieldQA、TriviaQA、TREC 等任务上绘制 Accuracy-Compression Curve。

IndexMem 在多数任务上随压缩率增加缓慢下降。TriviaQA 中，中等压缩甚至可能提升结果。作者将其解释为信息密度效应，即删除低价值或干扰 token 后，保留上下文更加集中。

附录给出的全任务平均分为：

| Method             |   10% |   25% |       50% |   75% |   90% |
| ------------------ | ----: | ----: | --------: | ----: | ----: |
| IndexMem           | 53.48 | 53.18 | **55.26** | 42.18 | 36.51 |
| Expected Attention | 43.78 | 43.88 |     42.09 | 36.33 | 28.60 |
| PyramidKV          | 43.63 | 41.57 |     42.35 | 39.17 | 29.68 |
| SnapKV             | 43.55 | 43.37 |     41.44 | 39.22 | 29.40 |
| TOVA               | 43.81 | 43.08 |     42.10 | 39.05 | 32.56 |

> [!question] 为什么 50% 压缩优于 10% 和 25%
> IndexMem 的平均分在 50% compression 达到最高值，这可能说明删除干扰信息具有正则化效果。
>
> 我仍然希望看到多次运行的方差和 Full Cache LongBench 分数。当前表格没有 0% 结果，难以判断 55.26 是超越 Full Cache，还是仅在压缩设置中相对更高。

> [!warning] 轻度压缩下的 Baseline 差距异常大
> 在 10% compression 下，IndexMem 已比其他方法高约 10 分。此时只删除很少 token，选择策略通常不应造成如此大的整体差距。
>
> 论文需要进一步解释：
>
> - Baseline 是否采用完全一致的 Local Window 和 Sink Token 设置
> - Memory Residual 是否改变了未删除信息的输出分布
> - 不同方法是否使用相同 Prompt Template 和最大生成长度

---

### Decoding-time Compression

![[98_Assets/IndexMem-7.png]]

作者每生成 128 个 token 执行一次压缩，并在 AIME25 和 Math500 上设置：

$$
B_{\max}\in{512,1024,1536,2048}
$$

随着缓存预算增加，各方法逐渐接近 Full Cache；在相同预算下，IndexMem 的任务得分整体最高。预算达到 1536 以上后，方法差距明显缩小。

这说明 Indexer 的价值主要集中在缓存非常紧张的条件下。当预算足够容纳大部分相关上下文时，复杂选择器带来的边际收益会下降。

---

### Memory Ablation

![[98_Assets/IndexMem-8.png]]

作者进行了两类关键对比：

1. **Indexer Only 与 IndexMem**

   - Memory 在高压缩率下提升最明显。
   - TREC 和 Qasper 中，Memory 能缓解 Indexer Only 的灾难性退化。

2. **SnapKV 与 SnapKV + Memory**

   - 将相同 Memory Module 接入 SnapKV 后，多项任务得到提升。
   - 说明潜在记忆与具体 eviction policy 具有一定正交性。

> [!tip] SnapKV + Memory 是重要的对照
> 这一实验说明性能提升没有完全依赖 Indexer。Memory 可以作为通用的 eviction 补偿组件，这增强了方法的可迁移价值。

> [!warning] Memory 消融仍然不完整
> 缺少以下实验：
>
> - 移除 Gate $g(q)$
> - 移除归一化向量 $b$
> - 固定 $\lambda=1$ 与使用衰减
> - 不同 $d_{\text{mem}}$
> - Slow Weights Only
> - Fast Weights Only
> - 每层独立与跨层共享
>
> 当前实验只能确认完整 Memory 有效，无法确定其内部机制。

---

### Efficiency

![[98_Assets/IndexMem-9.png]]

作者在 32K Prefill 和 1K Decoding 条件下测量效率：

- Indexer Only 始终快于 Full Cache。
- 压缩率越高，Decoding 时间越短。
- 加入 Memory 后会产生在线写入开销，延迟接近 Full Cache。
- Cache Memory 从 7.68 GB 降至 4.08 GB。
- 统计范围包含 KV Cache、Indexer Key Cache 和 Latent Memory State。

从 7.68 GB 到 4.08 GB 对应约 **46.9%** 的实际 Cache Memory 降幅，而此时 token eviction 比例为 90%。

> [!warning] Token 压缩率与真实显存收益差距较大
> 删除 90% KV token 最终只减少约一半 Cache Memory，说明 Indexer Cache、Memory State 或固定缓存占据了明显比例。
>
> 这不影响方法的精度结论，但部署时应关注真实 GB 和 Tokens/s，不能只依据 eviction ratio 判断收益。

> [!warning] Latency 结论偏弱
> 完整 IndexMem 的延迟仅保持在接近 Full Cache 的水平。方法的主要收益目前更偏向显存容量和精度保持，吞吐提升还不充分。
>
> 论文没有给出：
>
> - Prefill Latency
> - End-to-end Tokens/s
> - 不同 Batch Size
> - Kernel 时间分解
> - Memory Write 与 Indexer 的独立耗时

---

### Additional Baselines

附录加入了：

- **Locret**：Learnable Token Retention
- **AdaKV**：Learnable Token Eviction
- **xKV**：Representation-level Compression

在七个 LongBench 任务、75% compression 下：

- IndexMem 平均 56.06
- xKV 平均 42.40
- Locret 平均 31.50

IndexMem 在 WikiQA、HotpotQA、TriviaQA 和 MultiFieldQA 上更强；Locret 在 Passage Retrieval EN 上达到 85.07，显著高于 IndexMem 的 40.00。作者据此承认单一局部答案检索可能更适合另一类保留机制。

![[98_Assets/IndexMem-10.png]]

> [!warning] 最相关的学习式 Baseline 没有进入主实验
> 主表主要比较启发式方法。Locret 和 AdaKV 与 IndexMem 的问题设定更加接近，却只在附录的部分任务或不同训练配置下出现。
>
> 这削弱了论文对 **learnable eviction 优于已有 learnable eviction** 的论证强度。

---

## Critical Summary

### 这篇论文真正有效的部分

> [!tip] 方法的核心价值
> 作者把 eviction error 分成了两类：
>
> - Indexer 降低错误删除概率。
> - Memory 降低错误删除后的损失。
>
> 这种分解比单纯设计更复杂的重要性评分更完整。即使 Indexer 无法完美预测未来，系统仍保留恢复路径。

潜在记忆还具有较好的模块独立性。SnapKV + Memory 的结果说明它可以与其他选择器组合，具备通用组件的潜力。

### 仍然依赖的核心假设

1. 当前和历史 Query 能代表未来 Query 的关注模式。
2. 被删除 token 的作用可以压缩成一个固定大小的线性关联状态。
3. 残差补偿足以替代被删除 KV 在 Softmax 中的精确作用。
4. 跨 Head 汇总不会丢失关键的头部特定信息。
5. 在线记忆在反复写入和衰减后仍能保持稳定。

其中第一个和第二个假设决定方法在超长生成、任务切换和持续流式输入中的上限。

### 公式与记号疑点

> [!warning] Attention Scaling 的维度需要核对
> 正文注意力公式和附录记号都使用 $\sqrt{d_{\text{model}}}$ 作为缩放项。缓存张量则按 Head 表示，Key 和 Query 的内积维度是 $d_{\text{head}}$。
>
> 如果实现执行标准的逐 Head Attention，缩放项通常需要与实际内积维度一致。论文应说明这里采用了特殊定义，还是存在记号错误。

## Future

### 作者明确提出的方向

- 设计更适合极端压缩的重要性监督和 Memory Training Objective。
- 在更大训练规模、更多 Token Budget 和更广模型族上验证。
- 通过 Continual Training 将 Indexer、Eviction 和 Memory 原生集成进 Backbone。
- 从外挂模块逐渐发展为端到端学习的高效注意力架构。

### 我认为更值得继续推进的方向

#### Future-query-aware Eviction

当前监督来自已经出现的 Query。可以让 Indexer 直接预测未来若干生成步骤的注意力需求，或者训练一个带不确定性的保留概率：

$$
P(\text{token }t\text{ will be used in future}\mid X_{1:s})
$$

这会让 Importance 更接近真正的 eviction objective。

#### Uncertainty-aware Dual Storage

Indexer 不只输出分数，还输出置信度：

- 高分、高置信度 token 保留原始 KV。
- 低分、高置信度 token 直接删除。
- 不确定 token 进入容量更高的潜在记忆。
- 极高风险 token 同时进入两条路径。

这种设计可以显式利用 Indexer 的不确定性，降低错误硬决策。

#### Head-aware and Layer-aware Memory

当前 Memory 跨 Head 共享。可以研究：

- Head Group Memory
- 按功能动态路由的 Memory Bank
- 相邻 Layer 共享状态
- 不同 Layer 使用不同衰减速度

这样可以在状态规模与语义分辨率之间建立更细的控制。

#### Adaptive Write and Forgetting

当前 $\lambda$ 和 $\eta$ 是固定超参数。更合理的做法是根据以下因素动态决定写入和遗忘：

- Token Importance
- Indexer Uncertainty
- 当前 Memory Occupancy
- Query-Memory Novelty
- Evicted Residual Magnitude

需要优先保存那些被删除后会造成较大注意力残差的 token，而非等权累积所有 evicted KV。

#### Exact Retrieval from Latent Memory

当前 Memory 只能产生连续残差，无法恢复原始 token 位置和精确 Value。可以增加一个小型回查路径：

1. Latent Memory 判断某段被删除信息可能重新重要。
2. 系统从 CPU、SSD 或压缩存储中恢复少量原始 KV。
3. 恢复结果重新进入主 KV Cache。

这样可以将快速近似记忆和低频精确恢复结合起来。

#### Long CoT Stability

需要专门研究数千步生成中的：

- Fast Weight Saturation
- 旧信息覆盖
- 衰减导致的远程遗忘
- 多轮压缩误差累积
- Gate 随生成长度的变化

当前 Long-prefill、Short-decode 结果无法充分回答这些问题。

#### System Co-design

Indexer 和 Memory 都需要专门 Kernel。未来评价重点应从单一 Accuracy-Compression Curve 扩展为：

$$
\text{Quality}
\quad\text{vs.}\quad
\text{Peak Memory}
\quad\text{vs.}\quad
\text{Prefill Latency}
\quad\text{vs.}\quad
\text{Decode Throughput}
$$

只有当在线 Memory Write 和 Indexer Scoring 能够与 Attention Kernel 融合时，IndexMem 才可能同时获得明显的显存和吞吐收益。
