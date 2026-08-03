# Metis: Memory Foundation Model

arXiv-2026

**核心一句话**：Metis 将历史交互压缩为 Transformer 内部持续演化的参数化状态，并通过前向计算完成写入、更新、遗忘和读取，尝试让记忆从外部工程模块变成模型自身学到的能力。

---

## Key Contribution

- We introduce memory foundation models and native memory with formal definitions, providing further analysis from the perspective of native memory state and native memory procedure.
- We propose the first prototype of memory foundation models, named Metis, which is implemented with novel memory architectures and optimization tasks.
- We conduct extensive experiments to verify the effectiveness of our model, followed by detailed studies from multiple perspectives. We also publicly release our project to benefit the research community and industry.

### 对齐理解

- 作者提出 **Memory Foundation Model** 这一概念，将原生记忆拆成两个必要条件：

  - **Native Memory State**：模型在多次推理之间保持一个持续存在、动态变化的内部状态。
  - **Native Memory Procedure**：模型通过自身前向计算理解记忆指令，并执行记住、更新、遗忘和使用等操作。

- Metis 的关键变化是为每层 Transformer 增加一条记忆分支。历史信息不再作为文本反复拼接到输入中，而是被压缩进固定大小的关联矩阵，并在后续推理时通过 **Memory Attention** 读出。

- 作者将记忆看作一个预测问题。模型接收到信息时并不知道未来会如何使用它，因此写入过程需要预测哪些信息值得保留、以什么表示保留，以及未来查询应如何读出。

- 论文通过中期训练学习记忆行为。在线推理阶段冻结全部训练参数，记忆状态只通过前向计算更新，不进行梯度下降。

- 这篇论文的主要价值在于提出了一套完整链路：

$$
\text{记忆定义}
\rightarrow
\text{内部状态}
\rightarrow
\text{读写计算}
\rightarrow
\text{训练数据}
\rightarrow
\text{优化目标}
\rightarrow
\text{行为评测}
$$

> [!tip] 核心价值
> 论文最值得关注的地方并非某个单独的记忆矩阵，而是作者明确提出：
>
> **记忆能力需要同时训练存储实体和存储过程。**
>
> 只有状态，没有学习到的读写规则，模型只是带有一个递归变量。只有记忆操作，没有跨轮持续状态，也无法形成真正的在线记忆。

---

## Method

## 从 External Memory 到 Native Memory

### Motivation

**作者的直觉是：外部记忆系统和语言模型各自优化自己的局部目标，最终送入上下文的信息未必是语言模型最需要的信息。**

传统 Agent Memory 通常包含若干显式阶段：

$$
\text{原始信息}
\rightarrow
\text{抽取或总结}
\rightarrow
\text{存储}
\rightarrow
\text{检索}
\rightarrow
\text{重排}
\rightarrow
\text{拼接上下文}
\rightarrow
\text{模型推理}
$$

作者认为这一架构存在三个问题。

1. **Architecture Decoupling**

   外部记忆负责生成上下文，语言模型负责条件生成，两部分的表示空间和优化目标彼此分离。检索器认为相关的信息，未必是模型推理真正需要的信息。

2. **Blocked Gradient**

   写入、删除、检索、重排等步骤通常包含离散操作，梯度很难从最终回答完整传播到所有记忆环节。

3. **Sequential Overhead**

   信息抽取、索引、检索和上下文预填充形成额外串行流程，增加查询延迟。

![[98_Assets/Metis.png]]

Figure 1 从 **架构、优化和效率** 三个维度对比外部记忆与原生记忆。

作者希望将流程改写成：

$$
(X_t, M_t)
\xrightarrow{f_\Phi}
(Y_t, M_{t+1})
$$

其中，$M_t$ 是当前记忆状态，$\Phi$ 是训练后保持固定的模型参数。一次前向计算同时产生回答并形成下一时刻的记忆状态。

> [!warning] 概念边界
> 作者将参数化状态直接参与骨干网络计算视为原生记忆，并将独立控制器管理的可微 Memory Slot 归为外部记忆。
>
> 这一区分依赖作者设定的架构边界。Memory-Augmented Neural Network 同样可以端到端训练，因此 **是否属于原生记忆** 更多是一种研究范式划分，还没有形成严格且普遍接受的数学边界。

---

## Memory Foundation Model 的形式化定义

### 多步交互

设连续交互由 $T$ 个离散步骤组成：

$$
{(X_1,Y_1),(X_2,Y_2),\ldots,(X_T,Y_T)}
$$

普通语言模型在第 $t$ 步生成第 $k$ 个 token 时使用固定参数 $\theta$：

$$
y_{t,k}
\sim
P(y\mid X_t,Y_{t,<k};\theta)
$$

外部记忆模型还需要显式构造上下文 $C_t$：

$$
y_{t,k}
\sim
P(y\mid C_t,X_t,Y_{t,<k};\theta)
$$

其中存储和检索发生在模型外部：

$$
\mathcal C_t = \mathcal C_{t-1}\oplus{X_t,Y_t}
$$

$$
C_t = \mathcal C_{t-1}\otimes X_t
$$

$\oplus$ 表示写入，$\otimes$ 表示读取。

Memory Foundation Model 将参数改成随交互变化的 $\theta_t$：

$$
y_{t,k}
\sim
P(y\mid X_t,Y_{t,<k};\theta_t)
$$

可以进一步写成：

$$
\theta_t=\Phi\cup M_t
$$

其中：

- $\Phi$：训练完成后固定的静态参数；
- $M_t$：根据在线交互持续变化的原生记忆状态。

下一时刻状态由模型内部计算产生：

$$
M_{t+1} = F_\Phi(X_t,Y_t,M_t)
$$

> [!note] Memory 与 Knowledge
> 作者只将交互开始后获得的信息称为 Memory。
>
> 预训练阶段获得的信息存储在静态参数中，被归为 Knowledge。在线交互产生的用户信息、轨迹信息和环境变化进入动态状态，被归为 Memory。

> [!question] 定义中的宽泛部分
> 形式化定义只要求一部分参数随交互变化，并由前向计算自主更新。按照这一标准，大量递归状态模型、Fast Weight 模型和某些状态空间模型都可能满足定义。
>
> Metis 证明了一种具体实现可行，但论文尚未给出能够区分 Memory Foundation Model 与一般 Stateful Model 的充分条件。

---

## Native Memory State 与 Native Memory Procedure

### Native Memory State

**作者的直觉是：历史信息需要变成能与当前隐藏状态直接计算的内部对象。**

原生记忆状态需要满足：

- 在多轮推理之间持续存在；
- 根据当前交互动态变化；
- 与模型静态参数处于对齐的语义空间；
- 能直接参与后续层的前向计算。

作者选择参数化表示，原因包括：

- 表示密度高；
- 状态大小可以固定；
- 无需在每次查询时重新预填充历史文本；
- 可与神经网络计算共同优化。

### Native Memory Procedure

作者将记忆过程分为两类：

1. **Storage Procedure**

   根据输入语义执行记住、更新、遗忘等状态变换。

2. **Utilization Procedure**

   根据当前问题判断需要哪些记忆，并将相关信息注入当前推理。

这里的关键判断是：

$$
\text{Memory Storage}
\approx
\text{预测当前信息未来将如何使用}
$$

$$
\text{Memory Utilization}
\approx
\text{预测当前推理需要什么历史信息}
$$

作者因此使用连续函数学习这些过程，避免将复杂语义强制拆成插入、删除、检索、重排等有限规则。

> [!tip] Aha Moment
> 作者把记忆写入和读取都解释为预测问题。
>
> 这使记忆可以使用与语言建模类似的数据驱动方法训练，也解释了为何固定规则难以覆盖隐含更新、局部遗忘、组合调用和语义相关性等情况。

> [!warning] 自主操作的实际范围
> Metis 的写入结构本身仍较固定：
>
> - 每轮选择一组 token；
> - 将其投影为 Key 和 Value；
> - 按预设更新方程写入状态。
>
> 模型学习的是选择、投影和状态变换中的连续参数。它还没有展示动态扩容、创建独立记忆实体、合并记忆、迁移记忆或显式决定长期不写入等更完整的记忆生命周期。

---

# Metis Architecture

## 整体结构

**作者的直觉是：在每层 Transformer 中增加一个固定大小的历史信息通道，让当前 token 同时关注当前上下文和压缩后的历史状态。**

![[98_Assets/Metis-1.png]]

Figure 2 包含三个层次：

- Figure 2(a)：交互从 $t=1$ 持续到 $t=T$，模型参数状态由 $\theta_1$ 演化到 $\theta_T$。
- Figure 2(b)：每个 Transformer 层旁边加入一个 Metis Block。
- Figure 2(c)：Metis Block 包含 Local Memory Block 和 Hyper Memory Block。

每层计算存在三条主要路径：

1. **Original Attention**：处理当前步骤内部的 token 关系。
2. **Memory Attention**：根据当前隐藏状态从历史记忆矩阵中读取信息。
3. **Native Storage**：从当前步骤隐藏状态中挑选信息，更新下一步骤使用的记忆状态。

---

## Local Memory Block

### 目的

**Local Memory Block 负责保存当前时刻已经积累的历史信息。**

第 $l$ 层维护两个动态状态：

$$
M_t^{(l)}
\in
\mathbb R^{d_k\times d_v}
$$

$$
S_t^{(l)}
\in
\mathbb R^{d_k}
$$

其中：

- $M_t^{(l)}$：Dense Memory Network，存储 Key 与 Value 的关联；
- $S_t^{(l)}$：Query-Key Normalization Vector，用于修正不同 Key 累积次数和尺度；
- $d_k$：Memory Key 维度；
- $d_v$：Memory Value 维度。

初始状态设为：

$$
M_1^{(l)}=0
$$

$$
S_1^{(l)}=0
$$

从结构上看，$M_t^{(l)}$ 类似一个压缩后的关联记忆表。多个历史 Key-Value 对通过外积累加到同一矩阵中。

> [!note] 为什么还需要 $S_t$
> 如果只累积 $K^\top V$，某些 Key 方向被频繁写入后会拥有更大的数值尺度。
>
> $S_t$ 记录 Key 方向的累计权重，读取时用它进行归一化，使结果更接近加权平均。

---

## Hyper Memory Block

### 目的

**Hyper Memory Block 负责学习如何把当前交互变成对 Local Memory 的更新。**

Hyper Memory Block 中的参数在中期训练阶段优化，在线交互时保持固定：

$$
\widetilde W_Q^{(l)}
\in
\mathbb R^{d\times d_k}
$$

$$
\widetilde W_K^{(l)}
\in
\mathbb R^{d\times d_k}
$$

$$
\widetilde W_V^{(l)}
\in
\mathbb R^{d\times d_v}
$$

$$
\widetilde w_{\mathrm{agg}}^{(l)}
\in
\mathbb R^d
$$

它们分别负责：

- $\widetilde w_{\mathrm{agg}}^{(l)}$：判断哪些 token 值得写入；
- $\widetilde W_K^{(l)}$：产生记忆地址；
- $\widetilde W_V^{(l)}$：产生需要保存的内容；
- $\widetilde W_Q^{(l)}$：把当前查询映射到专门的记忆检索空间。

作者将 Hyper Memory 参数视为 **Slow Parameters**，将 $M_t$ 和 $S_t$ 视为交互过程中变化的 **Fast Parameters**。

---

## Native Memory Storage

### 1. Adaptive Aggregation

**作者的直觉是：一段输入中只有少量 token 对未来有价值，应当先选出这些 token，再写入固定容量的状态。**

对第 $l$ 层隐藏状态进行归一化：

$$
\widetilde H_t^{(l)} = \operatorname{PreNorm} \left( H_t^{(l-1)} \right)
$$

为每个 token 计算重要性：

$$
p_t^{(l)} = \operatorname{Softmax} \left( \frac{ \widetilde H_t^{(l)} \widetilde w_{\mathrm{agg}}^{(l)} }{ \tau } \right)
$$

其中 $\tau$ 控制分布尖锐程度。

作者按照概率从大到小排序，选择累计概率达到 $\rho$ 的最短前缀：

$$
L_t' = \operatorname{clip} \left( \min \left\{ k: \sum_{r=1}^{k}p_{(r)} \geq \rho \right\}, K_{\min}, L \right)
$$

这一设计与固定 Top-$k$ 不同。信息集中时选择较少 token，信息分散时选择更多 token。

由于离散选择不可导，训练时使用 Straight-Through Estimator，使梯度通过稠密概率分布传回评分向量。

选中的表示为：

$$
\overline H_t^{(l)} = \Pi_t^{(l)} \widetilde H_t^{(l)}
$$

其中 $\Pi_t^{(l)}$ 是选择矩阵。

> [!tip] 最关键的结构
> 消融实验中移除 Adaptive Aggregation 后，整体性能下降 **60.98%**。
>
> 这说明 Metis 的主要能力来源首先是 **从当前输入中提取可写入表示**，随后才是如何更新和读取矩阵。直接使用最后一个 token 无法稳定概括分散在序列不同位置的信息。

> [!question] 重要性与未来效用
> 评分器只根据当前隐藏状态判断 token 重要性，没有访问未来查询。
>
> 它学习到的是训练分布上的平均未来效用。遇到未来用途与训练模式差异较大的信息时，写入器可能提前丢失关键内容。这正是作者所说的记忆预测问题，也是固定容量系统的根本风险。

---

### 2. Memory Key 与 Memory Value

选中隐藏状态后，生成记忆 Key 和 Value：

$$
\widetilde K_t^{(l)} = \overline H_t^{(l)} \widetilde W_K^{(l)}
$$

$$
\widetilde V_t^{(l)} = \overline H_t^{(l)} \widetilde W_V^{(l)}
$$

Key 决定未来查询如何找到信息，Value 决定被找到后向当前推理注入什么内容。

这相当于将当前交互转换成一组隐空间关联：

$$
\widetilde K_{t,i}^{(l)}
\mapsto
\widetilde V_{t,i}^{(l)}
$$

---

### 3. Linear Update

论文先给出线性更新形式：

$$
M_{t+1}^{(l)} = \lambda M_t^{(l)} + \frac{1-\lambda}{L_t'} \frac{ \widetilde K_t^{(l)\top} }{ \sqrt{d_k} } \widetilde V_t^{(l)}
$$

$$
S_{t+1}^{(l)} = \lambda S_t^{(l)} + \frac{1-\lambda}{L_t'} \frac{ \widetilde K_t^{(l)\top}\mathbf 1 }{ \sqrt{d_k} }
$$

各项含义：

- $\widetilde K^\top\widetilde V$：将一组 Key-Value 关联压缩成一个矩阵；
- $\lambda$：保留旧状态的比例；
- $1-\lambda$：当前信息的写入比例；
- $L_t'$：消除当前选中 token 数量带来的尺度变化；
- $S_t$：记录 Key 方向上的归一化统计量。

将更新递归展开后：

$$
M_t^{(l)} = \sum_{j=1}^{t-1} \lambda^{t-(j+1)} \frac{1-\lambda}{L_j'} \frac{ \widetilde K_j^{(l)\top} }{ \sqrt{d_k} } \widetilde V_j^{(l)}
$$

因此，旧信息会受到指数衰减，同时所有历史事实被叠加在同一个固定大小矩阵中。

> [!warning] 固定矩阵中的语义叠加
> 不同事实没有独立 Slot，也没有显式边界。它们通过外积叠加在共享矩阵里。
>
> 当多个 Key 高度相关时，它们对应的 Value 会在读取阶段相互混合。作者后续的理论误差和容量实验都显示，这是 Metis 最核心的限制。

---

### 4. Gated Delta Update

实际 Metis 使用基于 Gated Delta Network 的更新策略，称为 GDU。

作者给出的经验结论是：

- 线性更新在短期、直接记忆操作上具有竞争力；
- GDU 在 LoCoMo 等较长对话任务上更稳定；
- 两者在 4B 模型上的总体差距仅为 **0.58%**；
- 27B 模型中，GDU 在 LoCoMo Gold 上为 **26.74**，线性更新为 **14.16**。

> [!warning] GDU 的证据强度有限
> 主文没有完整展开 GDU 的具体计算过程，主要通过实验说明其长轨迹优势。
>
> 附录也明确指出，不同规模模型使用的训练检查点并不完全匹配，实验为单次运行，当前结果不足以建立稳定的扩展规律或统计优势。

---

## Native Memory Utilization

### 目的

**作者的直觉是：当前查询应直接作用于压缩后的关联矩阵，得到与查询相关的历史 Value。**

Memory Query 为：

$$
\widetilde Q_t^{(l)} = \widetilde H_t^{(l)} \widetilde W_Q^{(l)}
$$

记忆读取结果为：

$$
\widetilde A_t^{(l)} = \operatorname{diag} \left( \widetilde Q_t^{(l)} S_t^{(l)} \right)^{-1} \widetilde Q_t^{(l)} M_t^{(l)}
$$

可以把它拆成两步理解。

首先计算查询和所有历史 Key 的隐式相似度：

$$
\widetilde Q_t
\widetilde K_j^\top
$$

随后用相似度加权历史 Value：

$$
\sum_j
\operatorname{sim}
\left(
\widetilde Q_t,
\widetilde K_j
\right)
\widetilde V_j
$$

$S_t$ 对加权和进行归一化。

这一读取方式不需要显式恢复每条历史记录。所有历史关联已经合并在 $M_t$ 中，因此计算量只与状态维度有关。

---

### 独立 Memory Query

普通 Attention Query 与 Memory Query 使用不同投影：

$$
Q_t^{(l)} = \widetilde H_t^{(l)}W_Q^{(l)}
$$

$$
\widetilde Q_t^{(l)} = \widetilde H_t^{(l)} \widetilde W_Q^{(l)}
$$

普通 Query 学习当前上下文内部的 token-token 关系，Memory Query 学习跨交互步骤的相关性。

消融中移除独立 Memory Query，整体性能下降 **12.23%**，在 Memory QA 上下降 **16.33%**。

> [!tip] 设计上的精妙点
> 当前上下文检索和跨轮记忆检索面对不同噪声结构。
>
> 使用单独的 Query 投影允许模型重新塑造跨步骤相似度，使相关历史方向被放大，无关历史方向被压低。这一设计比直接复用原始 Attention Query 更合理。

---

### Query-Key Normalization

移除归一化后，整体性能下降 **28.44%**，Memory QA 下降 **36.55%**。

这说明关联矩阵读取的主要风险来自不同历史 Key 的累积尺度和干扰。归一化向量并非辅助细节，它是稳定读取的必要组成。

---

### 与 Original Attention 融合

最终输出将当前上下文 Attention 和 Memory Attention 相加：

$$
A_t^{(l)} = \gamma \operatorname{Softmax} \left( \frac{ Q_t^{(l)}K_t^{(l)\top} }{ \sqrt{d_k} } + \operatorname{Mask}(L) \right) V_t^{(l)} + (1-\gamma) \operatorname{Norm} \left( \widetilde A_t^{(l)} \right)
$$

其中：

- 第一项处理当前步骤的局部上下文；
- 第二项提供跨步骤历史信息；
- $\gamma$ 控制两条分支的权重；
- $\operatorname{Norm}$ 对齐两条分支的数值尺度。

这种设计保留原始 Transformer 的完整当前上下文能力，Memory Attention 作为残差信息加入。

> [!warning] Memory 始终进入前向计算
> 只要记忆状态非空，Memory Attention 就可能影响当前任务。
>
> 论文虽通过 Memory Pollution 数据训练模型抑制无关记忆，但 General Capability 实验显示，激活记忆后所有通用任务均出现下降，IFEval 严格指标下降 **22.18**。这说明模型还没有学到可靠的全局读取门控。

---

## 理论解释

### Virtual Memory Prefix

**作者的直觉是：Metis 的 Memory Attention 可以近似理解为将历史表示压缩成一组虚拟前缀 token，再让当前 token 关注这些前缀。**

假设在当前层输入前添加虚拟前缀：

$$
\widehat H_t^{(l)} = \begin{bmatrix} P_t^{(l)}\ \widetilde H_t^{(l)} \end{bmatrix}
$$

标准 Attention 可以拆成：

$$
A_t^{(l)} = A_{\mathrm{original}}^{(l)} + A_{\mathrm{memory}}^{(l)}
$$

其中 Memory 部分为：

$$
A_{\mathrm{memory}} = \operatorname{Softmax} \left( \frac{ Q_t (P_tW_K)^\top }{ \sqrt{d_k} } \right) P_tW_V
$$

作者随后将指数相似度近似成线性内积，把 Memory Prefix 的计算重新排列为：

$$
\operatorname{diag}(Q S)^{-1}QM
$$

这与 Metis 的读取形式一致。

> [!warning] 理论等价中的近似
> 这一推导包含两个重要近似：
>
> - 用全局标量 $\gamma$ 近似每个 token 各自的 Original 与 Memory 权重；
> - 将 Softmax 中的指数相似度替换为线性点积。
>
> 因此，该推导提供的是结构直觉，不能视为严格证明 Metis 与 Prefix Attention 完全等价。

---

## 理论误差分析

作者假设当前查询真正需要第 $c$ 个历史步骤的信息，并且：

$$
\widetilde Q_t
\widetilde K_c^\top
\gg
\widetilde Q_t
\widetilde K_j^\top,
\quad j\neq c
$$

读取结果可以表示为目标项和三个误差项：

$$
\widetilde A_t \approx \check A_t + \epsilon_1 - \epsilon_2 - \epsilon_3
$$

其中：

- $\epsilon_1$：无关 Value 直接进入读出造成的 Attention Error；
- $\epsilon_2$：无关 Key 改变归一化分母造成的 Structural Error；
- $\epsilon_3$：无关 Key 和无关 Value 共同作用的高阶误差。

三个误差项都包含：

$$
\widetilde Q_t
\widetilde K_j^\top,
\quad j\neq c
$$

因此，若独立 Memory Query 能让无关步骤的相似度足够小，误差也会变小。

> [!question] 理论分析依赖的关键假设
> 该分析首先假定目标步骤的相似度显著高于其他步骤。
>
> 现实中的困难恰好在于大量事实可能共享主体、关系或语义方向。此时目标 Key 与干扰 Key 很难自然分离。理论分析解释了查询投影为何有帮助，但没有证明训练后一定能获得足够大的相似度间隔，也没有给出随事实数量增长的误差上界。

---

# Data Construction

## Primary Data

### 目的

**作者的直觉是：要让模型学会语义级记忆操作，训练样本必须明确展示信息如何改变未来状态。**

Primary Data 包含四类操作：

| Operation | 交互结构                                                                                                  | 目标       |
| --------- | ----------------------------------------------------------------------------------------------------- | -------- |
| Remember  | $\operatorname{Info}(A_1)\rightarrow\operatorname{Query}(A)$                                          | 保存并恢复信息  |
| Update    | $\operatorname{Info}(A_1)\rightarrow\operatorname{Info}(A_2)\rightarrow\operatorname{Query}(A)$       | 用新值覆盖旧值  |
| Forget    | $\operatorname{Info}(A_1)\rightarrow\operatorname{Info}(\bar A_1)\rightarrow\operatorname{Query}(A)$  | 抑制已撤销信息  |
| Reflect   | $\operatorname{Info}(A_1)\rightarrow\operatorname{Info}(B_1)\rightarrow\operatorname{Query}(A\cap B)$ | 组合多条记忆推理 |

数据来自 27 个公开 Benchmark，包括 LoCoMo、LongMemEval、RULER、LongBench、ZsRE、TOFU、MuSiQue 和 StrategyQA 等。

作者沿三个维度扩展样本：

1. **Memory Operation**

   Remember、Update、Forget、Reflect。

2. **Instruction Salience**

   - Explicit：明确要求记住或忘记；
   - Implicit：以自然叙述隐含信息变化。

3. **Noise Level**

   在信息与查询之间插入无关对话，测试长距离保持能力。

数据构造流程：

$$
\text{Seed Extraction}
\rightarrow
\text{Static Synthesis}
\rightarrow
\text{Quality Verification}
$$

Primary Data 最终包含：

- 357,137 个样本；
- 约 406.1M tokens；
- 其中 Remember 样本的长干扰上下文占据大部分 token。

> [!tip] 数据设计的合理性
> Explicit 与 Implicit 的配对能够减少模型对固定操作关键词的依赖。
>
> Distractor 样本则迫使模型在多个无关交互后保持信息。三种形式共同训练的是语义状态变化，而非单纯模板匹配。

> [!warning] 合成分布的局限
> 数据由统一结构生成，最终查询通常明确对应前面注入的事实。
>
> 真实 Agent 中，信息价值经常在很久以后才显现，查询也可能需要跨任务、跨模态或跨环境状态组合。当前合成数据仍然具有明显的事实记忆和问答导向。

---

## Auxiliary Data

### 目的

**Auxiliary Data 用于解决两个具体失败模式：事实间干扰和记忆污染。**

包含四种子类型：

| Subtype                    | 交互结构                                                                                                                          | 训练目标       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Multi-Entity Binding       | $\operatorname{Info}(A)\rightarrow\operatorname{Info}(B)\rightarrow\operatorname{Query}(A)\rightarrow\operatorname{Query}(B)$ | 区分相似事实     |
| Selective Forgetting       | 写入两条事实后撤销一条                                                                                                                   | 局部遗忘，保留另一条 |
| Post-Memory Dialogue       | 记忆问答后继续普通对话                                                                                                                   | 防止历史值泄漏    |
| Memory-Irrelevant Dialogue | 保持记忆状态并回答无关问题                                                                                                                 | 学习何时不使用记忆  |

Auxiliary Data 总计 609,443 个样本：

- Multi-Entity Binding：76,153；
- Selective Forgetting：76,153；
- Post-Memory Dialogue：357,137；
- Memory-Irrelevant Dialogue：100,000。

> [!tip] 值得借鉴的数据构造
> 作者没有只增加更多事实，而是针对结构性错误构造训练信号：
>
> - 相似事实绑定错误；
> - 遗忘造成连带删除；
> - 记忆值泄漏进普通回答；
> - 面对无关问题仍强行使用记忆。
>
> 消融显示，移除全部 Auxiliary Data 后整体性能下降 **19.31%**，说明记忆鲁棒性很大程度来自专门构造的反例。

---

# Model Optimization

## 总体训练方式

每个样本是一条多步轨迹：

$$
s
=

{(X_t,Y_t)}_{t=1}^{T_s}
$$

所有步骤按时间顺序前向传播。第 $t$ 步使用的 $\theta_t$ 已经包含之前所有步骤形成的记忆状态。

只对查询步骤集合 $Q_s$ 计算监督损失：

$$
\ell(s,t) = - \frac{1}{|Y_t|} \sum_{k=1}^{|Y_t|} \log P \left( y_{t,k} \mid X_t,Y_{t,<k};\theta_t \right)
$$

训练阶段：

- 冻结原始 backbone；
- 只优化 Hyper Memory 相关参数；
- 记忆状态在轨迹内部按前向更新；
- 梯度通过多步计算训练写入和读取过程。

> [!note] End-to-End 的准确含义
> 论文所说的端到端优化，指最终回答损失能够训练 Memory Selection、Memory Projection 和 Memory Readout。
>
> Metis 原型并未联合优化全部 backbone 参数。作者冻结了原始模型，因此当前实现属于 **Memory Module 与 Backbone Forward 的端到端连接**，还没有完成整个基础模型参数层面的联合训练。

---

## Memory Reconstruction Objective

### 目的

**先让模型具备将信息压入状态并恢复出来的基本能力。**

训练样本先提供一段参考内容，后续要求模型完整重建：

$$
\mathcal L_{\mathrm{rec}} = \pi_{\mathrm{rec}}(e) \mathbb E_{s\sim D_{\mathrm{rec}}} \left[ \sum_{t\in Q_s} \ell(s,t) \right]
$$

该目标逼迫记忆状态尽可能无损地保留来源内容。

作者明确指出其内在冲突：

- Reconstruction 要求保留所有细节；
- Instruction Following 需要抽象和泛化；
- 实际记忆通常应根据未来用途进行有损压缩；
- Reconstruction 会鼓励无差别存储。

> [!tip] 作者对目标冲突有清楚认识
> Reconstruction 被定位为 Warm-up 和容量上界训练，而非最终记忆行为本身。
>
> 这比长期使用纯重建损失更合理，因为完整重建容易把记忆系统退化成长上下文压缩器。

---

## Memory Operation Objective

### 目的

**让模型根据语义指令改变状态，使后续答案反映正确的净状态。**

$$
\mathcal L_{\mathrm{op}} = \pi_{e/i}(e) \mathbb E_{s\sim D_{\mathrm{op}}^{e/i}} \left[ \sum_{t\in Q_s}\ell(s,t) \right] + \pi_d(e) \mathbb E_{s\sim D_{\mathrm{op}}^d} \left[ \sum_{t\in Q_s}\ell(s,t) \right]
$$

其中：

- $D_{\mathrm{op}}^{e/i}$：Explicit 与 Implicit 操作；
- $D_{\mathrm{op}}^d$：包含长距离干扰的操作。

监督发生在最终查询上。例如 Update 样本只奖励新值，Forget 样本不允许重新输出旧值。

> [!question] 状态变化的可辨识性
> 最终回答正确不意味着内部执行了预期操作。
>
> 多种状态更新方式可能产生相同答案。模型可能通过覆盖、抑制、查询侧过滤或短期模式匹配完成任务。论文通过多种辅助场景限制退化解，但仍缺少对内部状态操作语义的直接验证。

---

## Regularization Objective

### 目的

**约束模型不要混淆相似事实，也不要在无关回答中泄漏记忆。**

$$
\mathcal L_{\mathrm{reg}} = \pi_{\mathrm{mf}}(e) \mathbb E_{s\sim D_{\mathrm{mf}}} \left[ \sum_{t\in Q_s}\ell(s,t) \right] + \pi_{\mathrm{mp}}(e) \mathbb E_{s\sim D_{\mathrm{mp}}} \left[ \sum_{t\in Q_s}\ell(s,t) \right]
$$

- $D_{\mathrm{mf}}$：Multi-Fact 数据；
- $D_{\mathrm{mp}}$：Memory Pollution 数据。

作者没有直接给三类损失设置固定系数，而是调整各数据子集的采样概率：

$$
\pi_\tau(e) = \frac{ w_\tau(e) }{ \sum_{\tau'\in\mathcal T}w_{\tau'}(e) }
$$

采样权重随 epoch 线性变化：

$$
w_\tau(e) = w_\tau^s + (w_\tau^e-w_\tau^s) \min \left( \frac{e}{E-1}, 1 \right)
$$

训练前期偏向 Reconstruction，后期逐渐增加长距离操作和正则数据。

---

# Experiments

## Experimental Settings

### 任务

1. **Memory Operation**

   - MemOps Gold；
   - MemOps Full；
   - Metis Test Set。

2. **Memory-based QA**

   - LoCoMo Gold；
   - NextMem Contextual Generation；
   - 包含 SQuAD、HotpotQA、LongMemEval 和 LoCoMo 子集。

3. **OOD Memory**

   - ATM-Bench；
   - MemDaily。

4. **General Capability**

   - MMLU-Pro；
   - IFEval；
   - GSM8K；
   - MMMLU。

### 对比方法

- **Full Context**：将完整历史提供给 Qwen3.5；
- **Partial Context**：Dense RAG 检索 Top-5 片段；
- **No Context**：只输入问题；
- **Temp-LoRA**：测试时训练临时 LoRA；
- **$\delta$-Mem**：使用低秩在线参数修正；
- **Metis**：查询阶段只输入当前问题和内部记忆状态。

### 训练配置

- Backbone：Qwen3.5-4B、9B、27B；
- 硬件：8 张 H100；
- 冻结 Backbone；
- AdamW；
- 学习率 $2\times10^{-4}$；
- BF16；
- Metis-4B 和 27B 训练 14,000 steps；
- Metis-9B 根据验证集在 8,000 steps 提前停止。

评测主要使用 GPT-4.1-mini 作为 Judge，重复三次并报告中位数。

> [!warning] 评测协议
> 主实验采用静态两阶段范式：
>
> 1. 连续输入信息步骤；
> 2. 再集中提出查询。
>
> LoCoMo 采用 Gold Evidence Session。该设置移除了真实系统中的信息发现、检索和跨任务选择困难，主要测量 **已知相关信息能否写入并恢复**。
>
> 它适合验证内部状态的存在，但距离开放环境中的自主记忆仍有明显差距。

---

## Overall Performance

### Memory Operation

在 No Context 设置下：

| Model         | MemOps Gold Avg. | Metis Test Avg. |
| ------------- | ---------------: | --------------: |
| Temp-LoRA-27B |             9.70 |           23.86 |
| $\delta$-Mem  |             4.38 |           15.03 |
| Metis-4B      |            17.84 |           56.72 |
| Metis-9B      |            19.63 |           57.92 |
| Metis-27B     |        **24.76** |       **73.77** |

Metis 在相同 No Context 条件下明显领先其他参数化记忆基线，说明历史信息确实进入了动态状态。

然而 Full Context 仍然更强：

| Model                    | MemOps Gold Avg. | Metis Test Avg. |
| ------------------------ | ---------------: | --------------: |
| Qwen3.5-27B Full Context |            87.90 |           78.87 |
| Metis-27B No Context     |            24.76 |           73.77 |

> [!warning] 应如何理解结果
> Metis Test Set 上，Metis-27B 已接近 Full Context。
>
> 在外部 MemOps Gold 上，Metis-27B 与 Full Context 仍相差超过 60 分。训练分布内部和外部基准之间存在显著差异，因此当前结果更支持 **Metis 学会了可迁移的短期记忆行为**，还不足以证明它可以替代完整历史。

### Forgetting

Forgetting 是最不稳定的操作：

- Metis-27B 在 Metis Test 上 Forget 为 **77.50**；
- 在 MemOps Gold 上仅为 **10.91**；
- MemOps Full 中 Metis-27B Forget 为 **3.86**，低于 Metis-9B 的 **7.27**。

这说明删除或抑制共享矩阵中的某条信息，比继续写入新关联更困难，并且扩大 Backbone 规模没有稳定改善遗忘。

---

## Memory-based QA

No Context 设置：

| Model         | LoCoMo Gold Avg. | NextMem Avg. |
| ------------- | ---------------: | -----------: |
| Temp-LoRA-27B |             4.24 |        30.97 |
| $\delta$-Mem  |            10.79 |        20.42 |
| Metis-4B      |            16.31 |        41.69 |
| Metis-9B      |            16.81 |        43.39 |
| Metis-27B     |        **26.74** |    **50.82** |

Full Context 上界：

| Model                    | LoCoMo Gold Avg. | NextMem Avg. |
| ------------------------ | ---------------: | -----------: |
| Qwen3.5-27B Full Context |            65.03 |        78.80 |

Metis 在 HotpotQA、LongMemEval、LoCoMo 和多跳任务上相对基线提升明显，说明压缩状态能够支持一定程度的组合读取。

> [!note] Scaling 现象
> 4B 到 9B 的提升较小，27B 出现明显跃升。
>
> 这可能表示原生记忆需要 Backbone 具备足够强的表示和推理能力，才能有效解释压缩状态。但论文只有三个规模和单次训练结果，尚不能据此确定稳定的 Scaling Law。

---

## Ablation Study

![[98_Assets/Metis-2.png]]

关键结果：

| Variant                     | Overall |        相对下降 |
| --------------------------- | ------: | ----------: |
| Full Metis                  |   33.14 |           0 |
| w/o Multi-Fact Scenario     |   29.68 |     -10.43% |
| w/o All Auxiliary Data      |   26.74 |     -19.31% |
| w/o GDU                     |   32.95 |      -0.58% |
| w/o Adaptive Aggregation    |   12.93 | **-60.98%** |
| w/o Optimizable Query       |   29.09 |     -12.23% |
| w/o Query-Key Normalization |   23.72 | **-28.44%** |

### 结论

1. **Adaptive Aggregation 是决定性模块**

   写入前的信息选择远比复杂更新规则更重要。

2. **Query-Key Normalization 是第二重要模块**

   固定矩阵中的叠加噪声需要强归一化才能稳定读取。

3. **独立 Memory Query 确实改善跨步骤路由**

   结果与理论分析一致。

4. **GDU 的总体贡献较小且任务相关**

   它主要改善长对话任务，在线性记忆操作测试中未稳定领先。

5. **Auxiliary Data 是能力组成部分**

   相似事实、选择性遗忘和记忆污染不能只依赖基础 Remember/Update 数据自然学会。

> [!warning] 方法贡献的真实排序
> 从消融结果看，Metis 的性能来源可大致排序为：
>
> $$\text{Token Selection}
>
> >
>
> \text{Readout Normalization}
>
> >
>
> \text{Auxiliary Data}
>
> >
>
> \text{Dedicated Memory Query}
>
> >
>
> \text{GDU Update}$$
>
> 因此，论文叙述中较受强调的 GDU 并非主要增益来源。

---

## Out-of-Distribution Tasks

### ATM-Bench

平均分：

- $\delta$-Mem：2.27；
- Temp-LoRA-27B：2.57；
- Metis-4B：10.22；
- Metis-9B：16.49；
- Metis-27B：**18.56**。

Metis 在 ATM-Bench 上表现出较强的分布外迁移，尤其是 Number 和 Open 类型。

### MemDaily

平均分：

- Temp-LoRA-27B：**59.45**；
- Metis-27B：59.04；
- Metis-4B：52.54；
- Metis-9B：47.29。

Metis 在 MemDaily 上没有稳定领先，9B 甚至弱于 4B。

> [!warning] OOD 结论应保持克制
> 两个 OOD 数据集给出不同结果：
>
> - ATM-Bench 明显支持 Metis；
> - MemDaily 只显示其具有竞争力。
>
> 这说明记忆操作的迁移依赖任务结构。当前证据支持部分泛化，尚未支持统一的分布外记忆能力。

---

## Memory Capacity

### Step-level Capacity

**测试一次更新能够压入多少信息。**

作者将前 $t$ 条人物事实拼接成一次输入，随后查询最早、中间和最后一条事实。

![[98_Assets/Metis-3.png]]

Figure 3 左侧显示：

- 输入较短时 Metis 表现较好；
- Word Count 增加后准确率快速下降；
- 最早事实下降最明显；
- 超过数百词后，三个位置的性能都很低；
- Full Context 对最早事实基本保持稳定。

### Trajectory-level Capacity

**测试重复更新后状态能够维持多少步。**

40 条事实每 5 条组成一组，依次更新状态。每次更新后查询最早、中间和最新事实。

Figure 3 右侧显示：

- 最早事实随更新次数持续衰减；
- 中间事实和最新事实也存在明显波动；
- 新写入不只覆盖最旧信息，而会干扰整个共享状态；
- 重复状态变换会积累压缩误差。

> [!warning] 论文最重要的负面结果
> Metis 在固定大小状态中实现了短期记忆，但长程容量迅速退化。
>
> 这表明其当前状态更接近 **有限容量的递归工作记忆**，尚未形成可扩展的长期记忆。固定矩阵避免了上下文长度增长，也使信息容量存在硬上限。

---

## General Capability

作者比较了两个阶段：

- **Initial Stage**：记忆状态为空；
- **Active Stage**：先写入一组与任务无关的信息，再完成通用任务。

| Benchmark | Initial Gap | Active Gap |
| --------- | ----------: | ---------: |
| MMLU-Pro  |       -0.80 |      -5.10 |
| IFEval    |       +0.55 | **-22.18** |
| GSM8K     |       -1.06 |      -5.61 |
| MMMLU     |       -0.80 |      -3.30 |

空状态下，Metis 基本保持 Backbone 能力。

状态被激活后，所有任务都出现下降，IFEval 最严重。

> [!warning] Memory Pollution 仍未解决
> 辅助训练数据专门包含 Memory-Irrelevant Dialogue，但无关记忆仍会影响正常推理。
>
> 这说明当前 Memory Attention 缺少足够可靠的读取抑制机制。独立 Query 可以降低噪声，却无法保证在完全无关任务中将 Memory Branch 关闭。

---

## Low-rank Decomposition

作者对每层记忆矩阵执行 SVD：

$$
M_t^{(l)} = U_t^{(l)} \Sigma_t^{(l)} V_t^{(l)\top}
$$

只保留前 $k$ 个奇异方向：

$$
\widehat M_t^{(l)} = \widehat U_t^{(l)} \widehat\Sigma_t^{(l)} \widehat V_t^{(l)\top}
$$

![[98_Assets/Metis-4.png]]

总体结果：

| Rank | Overall | Full-state Recovery |
| ---: | ------: | ------------------: |
|    1 |   14.43 |               43.5% |
|    4 |   22.84 |               68.9% |
|   16 |   31.22 |               94.2% |
|   64 |   33.10 |           **99.9%** |
|  128 |   33.26 |              100.4% |
|  256 |   33.17 |              100.1% |
| 1024 |   33.14 |                100% |

大部分有效信息集中在约 64 维子空间中。

> [!tip] 重要观察
> Full Memory 的维度为 1024，但 Rank 64 已恢复 99.9% 性能。
>
> 这说明当前训练形成的记忆状态高度冗余，也说明未来可以直接约束写入状态为低秩结构，避免先生成完整矩阵再进行 SVD。

> [!question] 压缩实验的计算成本
> 论文主要强调持久化存储节省，并在使用前重建矩阵。
>
> 对每层状态执行 SVD 和重建本身具有额外计算成本。论文没有将在线分解成本与普通 Metis 进行完整端到端比较，因此 Rank 64 当前更像一种存储优化证据。

---

## Case Studies

![[98_Assets/Metis-5.png]]

案例展示四类行为：

1. **Remember**

   保存 Alice 喜欢牛肉汉堡，并在后续问题中正确回答。

2. **Multi-Entity**

   连续保存年龄、食物和居住地，能够根据属性选择正确值。

3. **Distract**

   插入无关对话后仍能恢复偏好。

4. **Forget**

   收到遗忘指令后，后续查询不再输出旧偏好。

遗忘案例中存在一个细节：模型在接收遗忘指令的当前回复里仍重复了旧事实，随后查询阶段才表现为已遗忘。

> [!warning] 输出行为与状态行为不一致
> 最终状态满足遗忘目标，但操作发生当下的语言回复仍受到旧记忆影响。
>
> 这说明回答生成和状态更新虽然共享计算，却没有完全对齐。一个安全的记忆系统需要同时保证：
>
> - 操作响应正确；
> - 更新后的状态正确；
> - 未来读取行为正确。

---

## Efficiency

### 理论并行性

Original Attention、Memory Attention 和 Native Storage 都使用同一层隐藏状态，彼此不存在必要的串行依赖。

理论层延迟为：

$$
T_{\mathrm{parallel}}^{(l)} = \max \left( T_{\mathrm{orig}}^{(l)}, T_{\mathrm{util}}^{(l)}, T_{\mathrm{store}}^{(l)} \right) + T_{\mathrm{fuse}}^{(l)}
$$

不过最终实现没有启用 CUDA Stream 异步重叠，因为细粒度 Kernel Launch 和同步开销可能抵消收益。

### LoCoMo Gold 延迟

平均端到端延迟：

- Partial Context：0.268 秒；
- Metis：0.562 秒；
- Full Context：0.607 秒；
- $\delta$-Mem：约 0.884 秒；
- Temp-LoRA：约 1.565 秒。

P95：

- Metis：0.926 秒；
- Full Context：3.012 秒。

Metis 相对 Full Context 将 P95 降低 **69.3%**。

### 长上下文扩展

![[98_Assets/Metis-6.png]]

- 32K 及以下，Metis 端到端延迟略高于 Full Context；
- 64K 首次超过 Full Context；
- 128K、生成 32 tokens 时加速 **1.497×**；
- 128K、生成 128 tokens 时加速 **1.595×**。

> [!warning] 效率结论的适用范围
> Metis 的优势主要出现在长上下文区域。
>
> 在常见的中短上下文和冷启动测试中，Partial Context RAG 仍然最快。论文使用的 Qwen3.5-4B 有 24 层 Linear Attention 和 8 层 Full Attention，这也会显著影响与 Full Context 的交叉点。

### 持久化存储

![[98_Assets/Metis-7.png]]

32K 历史下：

- Full Context KV Cache：1,118.86 MB；
- Metis Full State：16.79 MB；
- Metis Rank 64：2.11 MB；
- Temp-LoRA：190.3 MB；
- $\delta$-Mem：4.6 KB；
- RAG Store：6.051 MB。

Metis Full State 相比 Full Context 小约 **67×**，Rank 64 小约 **529×**。

---

## Baseline 与实验设计问题

> [!warning] Full Context 依然是明显上界
> Metis 在多个外部任务上与 Full Context 存在巨大差距。
>
> 当前结果证明内部状态可以承载一部分信息，并显示出效率潜力。它尚未达到完全移除外部记忆和原始证据的目标。

> [!warning] 训练数据与 Metis Test 的耦合
> Metis Test Set 来自作者的数据构造分布，Metis 在该数据集上的分数显著高于 MemOps。
>
> OOD 实验缓解了这一疑问，但两个 OOD 数据集表现并不一致。

> [!warning] LLM-as-a-Judge 波动
> 主实验大量依赖 LLM Judge。作者重复三次并取中位数，能够降低随机波动，但低秩实验中超过 100% 的恢复率也说明评测噪声仍然存在。
>
> ATM-Bench Number 和 MemDaily 的确定性指标提供了补充证据。

> [!warning] 比较对象并不完全同构
> $\delta$-Mem 使用 Qwen3-4B-Instruct-2507，Metis 使用 Qwen3.5 Backbone。
>
> Temp-LoRA、RAG 和 Metis 的状态容量、训练数据、输入处理方式也有明显差异。因此，结果适合比较完整系统表现，较难将差距完全归因于记忆架构。

> [!warning] 缺少联合系统
> 论文将目标设为完全取消外部文本记忆，但没有系统评估 Native Memory 与 RAG、原始日志或分层存储联合使用的效果。
>
> 考虑到 Metis 的容量下降和不可解释性，混合架构可能比单独依赖固定状态更接近实际部署需求。

---

## 对论文核心论证的判断

### 成立的部分

- 动态参数状态能够跨交互保存信息；
- 前向计算可以学习记住、更新、遗忘和读取等行为；
- 独立 Memory Query 和 Query-Key Normalization 能有效降低干扰；
- 固定大小状态具有显著存储优势；
- 原生状态在短期 Memory Operation 和 QA 上明显优于无上下文 Backbone；
- 记忆能力可以通过专门构造的数据和监督目标在中期训练阶段获得。

### 证据仍不足的部分

- Metis 是否足以被称为 Foundation Model 层面的通用记忆；
- 固定状态能否支持长期、开放世界和持续数月的交互；
- 记忆操作是否在隐空间中形成了稳定、可组合的语义；
- 模型是否能够可靠决定何时完全不读取记忆；
- 忘记是否意味着信息被真正删除，而非查询阶段受到抑制；
- Native Memory 是否能在安全性、审计和隐私要求下替代显式存储；
- 端到端训练优势是否会在联合训练 Backbone 后继续成立。

> [!note] 总体评价
> Metis 是一个有说服力的 **Memory Foundation Model Prototype**。
>
> 它完成了从概念定义、架构、数据、目标到实验的闭环。论文同时公开展示容量衰减、记忆污染和通用能力下降，没有掩盖固定状态的主要问题。
>
> 当前 Metis 更接近具备语义操作能力的快速权重工作记忆。它为原生长期记忆提供了起点，还没有解决长期记忆所要求的容量扩展、结构化组织、精确删除和可验证读取。

---

## Future

### 1. 从单一共享矩阵转向可扩展状态结构

当前 $M_t$ 将全部事实叠加在同一矩阵中。可以研究：

- 动态数量的 Memory Block；
- 分层或分区状态；
- 稀疏激活的参数化 Slot；
- 根据语义路由到不同子空间；
- 可合并和可拆分的状态单元。

目标是让容量随记忆复杂度扩展，同时保留前向计算中的原生读取。

### 2. Query-conditioned Consolidation

当前信息在未来查询出现前已经被固定压缩。可以让模型保留较低成本的中间痕迹，并在查询到来后重新组织：

$$
M_t'
====

\operatorname{Consolidate}(M_t,q_t)
$$

这样能够缓解写入时不知道未来用途的问题。

### 3. 显式 Memory Read Gate

为 Memory Attention 增加独立门控：

$$
g_t
===

\sigma
\left(
f_{\mathrm{gate}}(H_t,M_t)
\right)
$$

$$
A_t
===

A_{\mathrm{orig}}
+
g_t\cdot A_{\mathrm{mem}}
$$

并使用无关任务、对抗记忆和通用能力保持目标训练 $g_t$。这一机制直接针对 Active Stage 的能力下降。

### 4. 可验证的 Forgetting

需要区分三种行为：

- 读取时暂时抑制；
- 从状态中删除关联；
- 无法通过任何攻击查询恢复。

可以训练状态反演器或隐私攻击器，验证被遗忘内容是否仍可从 $M_t$ 中解码。

### 5. Native Memory 与 External Memory 的分层协作

固定状态适合：

- 高频使用信息；
- 当前任务状态；
- 用户短期偏好；
- 紧凑技能和经验表示。

外部存储适合：

- 长期原始证据；
- 可审计记录；
- 低频历史；
- 需要精确引用的内容。

更现实的架构可能是：

$$
\text{Native Working Memory}
+
\text{External Episodic Store}
+
\text{Consolidation Procedure}
$$

### 6. 从事实记忆扩展到经验记忆

当前数据主要训练事实保存和问答。后续应构造包含以下内容的轨迹：

- 动作、结果与奖励；
- 失败原因；
- 策略修正；
- 环境动力学；
- 多次尝试之间的因果关系。

评测重点从事实恢复扩展到：

$$
\text{Experience}
\rightarrow
\text{Policy Improvement}
$$

### 7. 长时程训练与稳定性

需要研究：

- 数千次乃至数万次状态更新；
- 信息随时间过期；
- 冲突证据持续出现；
- 用户偏好渐变；
- 多任务共享状态；
- 状态重置和迁移；
- 错误写入后的恢复。

### 8. 直接约束低秩或稀疏状态

Rank 64 已保留约 99.9% 性能，可以在训练中直接参数化：

$$
M_t
===

U_tV_t^\top
$$

使更新只作用于低秩因子，降低状态存储和读写成本，也可能减少无关方向带来的干扰。

### 9. 内部状态的解释与审计

参数化记忆缺少文本记忆的可读性。需要提供：

- 状态到事实的可控解码；
- 记忆来源追踪；
- 更新前后差异解释；
- 哪条记忆影响了当前 token；
- 状态版本管理和回滚。

没有这些机制，原生记忆很难用于要求可追溯性的真实 Agent 系统。
