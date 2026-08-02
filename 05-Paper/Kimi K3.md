# Kimi K3: Open Frontier Intelligence

arXiv Technical Report-2026

**核心一句话**：Kimi K3 同时扩大预训练模型规模与测试时计算规模，并围绕百万 token 长上下文、超稀疏 MoE 和长轨迹 Agent 强化学习重新设计模型结构、训练环境与基础设施，最终构成一个面向长程智能任务的完整技术栈。

---

## Key Contribution

- **Pre-training at the open frontier**
- **Reinforcement learning for multi-effort test-time scaling**
- **Infrastructure for multi-trillion-parameter, million-token intelligence**
- **An open frontier model**

### 我对贡献的理解

- Kimi K3 将开放权重模型的预训练规模提升到 **2.78T 总参数、104.2B 激活参数**，并把训练上下文扩展到 **1M token**。
- 模型结构围绕三种信息流动展开：

  - **序列维度**：KDA 与 MLA 负责 token 之间的信息交换。
  - **深度维度**：Attention Residuals 负责不同网络层之间的信息交换。
  - **宽度维度**：Stable LatentMoE 负责不同专家通道之间的信息交换。
- 后训练同时覆盖 **general、general agent、coding agent** 三类能力，以及 **low、high、max** 三种推理强度，共训练九个专家策略，再蒸馏回统一模型。
- 为了真正训练百万 token 的 Agent 轨迹，作者还构建了可暂停、恢复和分叉的微虚拟机环境，并保留模型 KV Cache 与外部环境状态。
- 完整模型权重开放，但预训练数据、内部 RL 环境、奖励模型和多数基础设施并未完整开放。因此这里的 **open frontier** 主要指模型权重开放，完整复现仍然困难。

> [!note] 先区分技术继承与 K3 新增内容
> KDA、Attention Residuals、LatentMoE、MLA、Muon 等基本思想已经在前序工作中提出。Kimi K3 的核心价值主要来自三个方面：
>
> 1. 将这些结构扩展到 2.8T 参数规模；
> 2. 为极端规模重新设计稳定性机制与系统实现；
> 3. 将模型结构、Agent RL、量化和在线部署共同优化。
>
> 因此，K3 更接近一次**系统级架构收敛**，并非依赖某个单独模块取得全部提升。

---

## Method

## 总体思路：同时扩展两条 Scaling 轴

**直觉**：更强的 Agent 既需要更强的基础模型，也需要在实际任务中投入更多推理、工具调用和环境交互。

作者将模型能力的扩展分成两条轴：

1. **Pre-training scaling**

   - 更大的参数规模；
   - 更长的训练上下文；
   - 原生视觉预训练；
   - 更高容量的专家系统。

2. **Test-time scaling**

   - 更多思考 token；
   - 更多工具调用；
   - 更长的任务轨迹；
   - 多轮观察、验证和修正；
   - 最高达到数千次工具调用和数百万累计上下文 token。

我认为这是理解 Kimi K3 最关键的入口。作者认为，单纯在相近规模的基础模型上继续堆强化学习，最终会出现收益收敛。因此 K3 同时扩大底座规模与测试时计算。

---

## 模型架构：沿序列、深度与宽度扩展信息流

**直觉**：Transformer 的能力受限于信息能否在长序列、深层网络和大量专家之间有效传递。

![[figure2.png]]

图 2 将整个模型拆成三条信息流：

| 信息维度           | 对应结构                | 目标            |
| -------------- | ------------------- | ------------- |
| Token mixing   | Hybrid KDA-MLA      | 处理百万 token 序列 |
| Layer mixing   | Attention Residuals | 避免深层信息被残差累积压缩 |
| Channel mixing | Stable LatentMoE    | 扩展专家容量并控制计算成本 |

模型共有 93 层，其中包括 **69 层 KDA 和 24 层 MLA**。每个基本 block 使用三层 KDA 加一层 Gated MLA，形成 3:1 的混合比例。每个注意力层后面都连接 Stable LatentMoE。

> [!tip] 一个很好的架构表达方式
> 作者没有把 KDA、AttnRes 和 MoE 描述成互不相关的模块，而是统一解释为：
>
> - KDA 和 MLA 解决长度维度的信息流动；
> - AttnRes 解决深度维度的信息流动；
> - MoE 解决宽度维度的信息流动。
>
> 这种表达使复杂架构具备明确的设计主线。

> [!warning] 缺少组件级归因
> 报告给出了整体约 2.5 倍 scaling efficiency 提升，但没有完整展示：
>
> - 单独加入 KDA 的收益；
> - 单独加入 AttnRes 的收益；
> - Stable LatentMoE 各组件分别贡献多少；
> - 三者之间是否存在互补或冗余。
>
> 因此，报告证明了整个配方有效，却没有充分回答具体哪项创新最重要。

---

## Hybrid Attention：KDA 负责连续记忆，MLA 负责全局检索

### 核心直觉

**长序列中的大部分信息可以压缩进固定大小的递归状态，但模型仍需要周期性的全局注意力纠正压缩损失。**

Kimi K3 每三层 KDA 插入一层 Gated MLA，并在骨干网络最后额外放置一层 MLA，保证最终表示经过一次全局信息交互。

可以将两种机制理解为：

- **KDA**：将历史压缩成持续更新的状态，计算和缓存成本不随序列长度线性增长。
- **MLA**：保留显式 token 级全局访问能力，弥补递归压缩带来的信息损失。

这种组合将长期记忆与精确回看分开处理。

---

### Kimi Delta Attention

**直觉**：维护一个可写入、可纠正、可遗忘的矩阵状态，将历史 token 压缩到固定大小的状态中。

对于单个注意力头，KDA 的状态更新为：

$$
S_t =
\left(I-\beta_t k_tk_t^\top\right)
\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
$$

$$
\tilde{o}_t=S_t^\top q_t
$$

其中：

- $S_t$ 是当前时刻保存的递归状态；
- $\alpha_t$ 是逐通道保留系数；
- $\beta_t$ 控制当前 token 的写入强度；
- $k_tv_t^\top$ 将当前信息写入状态；
- $I-\beta_tk_tk_t^\top$ 会沿当前 key 方向修正旧内容；
- $q_t$ 从状态中读取与当前查询相关的信息。

我更愿意把这个更新理解成三步：

1. 通过 $\alpha_t$ 对旧状态进行通道级遗忘；
2. 沿 $k_t$ 对应方向删除或修正旧关联；
3. 写入新的 $k_t \rightarrow v_t$ 关联。

这比简单的累加式线性注意力多了一种**覆盖旧记忆**的能力。

> [!note] KDA 与普通 KV Cache 的区别
> 标准注意力保存每个历史 token 的 key 和 value。
>
> KDA 将历史压缩到固定大小的矩阵 $S_t$。序列变长时，状态尺寸保持不变，但多个历史信息可能在同一个状态中发生叠加和干扰。

---

### Chunkwise Parallelism

**直觉**：递归状态适合逐 token 推理，却不适合 GPU 训练，因此作者将序列分块，在块内并行，在块间递归。**

对于一个 chunk，输出被拆成：

$$
O^{[t]}
=======

\underbrace{
\left(\Gamma^{[t]}\odot Q^{[t]}\right)S^{[t]}
}*{\text{来自前序 chunk}}
+
\underbrace{
A^{[t]}\widetilde{V}^{[t]}
}*{\text{当前 chunk 内部交互}}
$$

第一项读取前面 chunk 压缩出的递归状态，第二项计算当前 chunk 内部的因果交互。

这是线性注意力常见的工程化路径。真正影响 K3 系统效率的改动出现在 decay 参数化。

---

### Lower-bounded Decay

**直觉**：限制单步遗忘强度，让所有 chunk 内计算都能转成 Tensor Core 擅长的稠密矩阵乘法。**

Kimi Linear 使用无下界的负 Softplus：

$$
g_t=-e^A\operatorname{Softplus}(z_t)
$$

Kimi K3 改为：

$$
g_t=g_{\min}\operatorname{Sigmoid}(e^Az_t)
$$

$$
\alpha_t=\exp(g_t)
$$

其中 $g_{\min}=-5$，因此：

$$
\alpha_t \in \left(e^{-5},1\right)
$$

![[figure3.png]]

原来的累计 decay 可能极小，计算其倒数时容易溢出。K3 将 log-decay 限制在有限区间，使 16-token tile 内的缩放范围始终落在 BF16 可表示范围中。

结果是：

- 对角 tile 不再需要逐位置计算；
- 所有因果 tile 都能使用稠密 Tensor Core GEMM；
- 算法约束直接转化为硬件吞吐提升。

> [!tip] 算法与系统共同设计的典型案例
> 作者通过修改可学习门的取值范围，换取统一的矩阵乘法执行路径。
>
> 这里没有引入复杂近似，计算语义仍然保持精确。它体现了 K3 报告最鲜明的思路：**模型公式需要主动适配硬件执行结构**。

> [!question] 遗忘能力是否受到限制
> 单步 retention 被限制为 $\alpha_t>e^{-5}$。虽然连续多步仍可形成很强的累计遗忘，但该下界可能限制模型对单个异常 token 进行瞬时清空的能力。
>
> 报告主要展示了数值稳定性和计算收益，没有单独分析 lower bound 对模型表达能力的影响。

---

### Full-rank Output Gate

KDA 输出为：

$$
y_t =
W_o
\left[
\operatorname{Sigmoid}(W_gx_t)
\odot
\operatorname{RMSNorm}(\tilde{o}_t)
\right]
$$

K3 将之前的低秩输出门改成 full-rank projection。

**目的**：允许当前 token 独立控制每个输出通道是否读取递归状态，减少无关历史信息进入后续网络。

这个门可以看作 KDA 的读取控制器：

- $\alpha_t$ 控制旧信息保留；
- $\beta_t$ 控制当前信息写入；
- $W_gx_t$ 控制当前信息读取。

因此 KDA 已经具备一个较完整的读写遗忘机制。

---

## Gated MLA：周期性的精确全局访问

**直觉**：递归状态擅长低成本积累历史，显式注意力擅长精确访问某个历史位置，两者适合交替使用。**

MLA 将每个 token 的 key 和 value 压缩到低维 latent：

$$
c_t=W_cx_t
$$

推理时只缓存 $c_t$，再通过投影恢复各注意力头需要的 key 和 value，从而减少 KV Cache。

K3 对 MLA 做了两个主要调整：

1. 所有 MLA 层采用 **NoPE**；
2. 加入与 KDA 相同的 full-rank output gate。

MLA 输出为：

$$
y_t =
W_o
\left[
\operatorname{Sigmoid}(W_gx_t)\odot\tilde{o}_t
\right]
$$

### NoPE 为什么可行

作者将位置建模交给 KDA：

- KDA 的递归顺序、卷积和 decay 自然包含位置与近期性；
- MLA 只负责全局内容匹配；
- 上下文扩展时无需修改 RoPE base 或使用位置插值。

我注意到这里形成了一种职责分离：

- **KDA 提供顺序感**
- **MLA 提供全局内容访问**

> [!question] NoPE MLA 的顺序信息从哪里来
> MLA 自身无法直接区分相同内容的不同排列。它依赖输入 hidden state 已经被相邻 KDA 层编码过顺序信息。
>
> 这要求 KDA 产生的隐状态能长期保留足够的位置结构。报告没有提供序列置换或位置敏感性消融，因此这种职责分离的边界仍不清楚。

> [!warning] 3:1 比例缺少详细解释
> 报告直接采用三层 KDA 加一层 MLA，没有展示 1:1、7:1 或仅在高层使用 MLA 的比较。当前比例更像 scaling study 中选出的经验结果，缺乏公开的选择依据。

---

## Attention Residuals：让网络层之间也进行注意力检索

### 核心直觉

**标准残差连接把所有历史层压进一个不断累加的状态，深层网络无法主动选择需要回看的浅层表示。**

传统残差更新中，第 $l$ 层只能接收上一层累积后的 $h_l$。早期信息必须经过连续变换才能传递到深层，可能发生覆盖和干扰。

AttnRes 将网络深度看成另一条序列，让当前层从所有前序层中选择信息：

$$
\alpha_{i\rightarrow l} = \frac{ \exp\left( q_l^\top\operatorname{RMSNorm}(k_i) \right) }{ \sum_{j=0}^{l-1} \exp\left( q_l^\top\operatorname{RMSNorm}(k_j) \right) }
$$

$$
h_l=\sum_{i=0}^{l-1}\alpha_{i\rightarrow l}v_i
$$

其中：

- $q_l=w_l$ 是第 $l$ 层学习到的 pseudo-query；
- $k_i=v_i$ 是前面第 $i$ 层的输出；
- $\alpha_{i\rightarrow l}$ 表示当前层从第 $i$ 层读取多少信息。

我对这个机制的理解是：每一层都在学习自己需要哪种抽象层次。

例如：

- 浅层更接近词法和局部模式；
- 中层可能包含实体、结构和关系；
- 深层更接近任务决策；
- 当前层可以跳过中间累积状态，直接读取适合自己的表示。

---

### Block Attention Residuals

完整 AttnRes 需要长期保存每一层输出，内存和 pipeline 通信成本较高。K3 将 93 层分成约 8 个完整 block，每个 block 包含 12 层。

block 内部先累加：

$$
b_n=\sum_{j\in B_n}f_j(h_j)
$$

block 之间再执行注意力。

这样，保存状态数量从：

$$
O(Ld)
$$

下降到：

$$
O(Nd)
$$

其中 $L$ 是总层数，$N$ 是 block 数量。

> [!tip] 将深度变成可检索记忆
> AttnRes 可以被理解成一种模型内部的纵向 memory。
>
> 普通残差连接只维护单一累积状态；AttnRes 保存多个深度检查点，再由后续层进行检索。这个思想与长上下文 memory 的问题高度相似。

> [!warning] Block 压缩重新引入信息瓶颈
> 一个 12 层 block 内的表示被求和压缩为单个 $b_n$。这降低了开销，也丢失了逐层访问能力。
>
> 作者引用前序研究说明大约 8 个 block 能恢复大部分收益，但 K3 报告缺少该超参数在 93 层、混合 KDA-MLA 架构上的独立实验。

> [!question] pseudo-query 的适应性是否足够
> $q_l$ 是每层固定的可学习向量，实际权重会随 token 的 layer representation 变化，但当前任务和当前 token 无法直接生成新的 query。
>
> 更强的形式可以令 query 同时依赖当前 token 状态，使不同输入主动选择不同的深度路径。

---

## Stable LatentMoE：在极端稀疏条件下扩展模型宽度

### 核心直觉

**大量专家能够提供更丰富的专业化空间，但直接让每个专家处理完整 hidden state，会带来过高通信量与不稳定激活。**

K3 使用：

- 896 个 routed experts；
- 每个 token 激活 16 个；
- 2 个 full-width shared experts；
- routed latent width 为 3584，是主 hidden dimension 7168 的一半。

其计算为：

$$
u=
\sum_{i\in T_k(x)}
p_iE_i^{\text{routed}}(W_\downarrow x)
$$

$$
y=
\sum_{j=1}^{N_s}E_j^{\text{shared}}(x)
+
W_\uparrow\operatorname{RMSNorm}(u)
$$

shared experts 处理完整表示，负责通用变换。routed experts 在低维 latent space 中工作，负责专业化变换。

这使得模型可以同时增加：

- 专家总数；
- 每 token 激活专家数；
- 专家专业化组合数量。

而通信张量仍然只有 latent width。

![[figure2.png]]

---

### Normalized LatentMoE

在 routed experts 聚合后、升维之前加入 RMSNorm：

$$
W_\uparrow\operatorname{RMSNorm}(u)
$$

**目的**：消除不同专家组合与 router 权重造成的尺度差异，避免升维矩阵将异常尺度进一步放大。

这个修改很简单，却直接对应 2.8T 模型中的实际训练故障。报告称它同时改善训练稳定性、验证损失和下游能力。

---

### SiTU-GLU

### 核心直觉

**保留 SwiGLU 在常见输入范围内的形状，同时限制极端激活值。**

SwiGLU 的两个乘法分支都可能无界增长。K3 提出：

$$
\operatorname{SiTU\text{-}GLU}(x) = \left[ \beta_1\tanh\left(\frac{W_gx}{\beta_1}\right) \odot\operatorname{Sigmoid}(W_gx) \right] \odot \left[ \beta_2\tanh\left(\frac{W_ux}{\beta_2}\right) \right]
$$

其中：

$$
\beta_1=4,\qquad \beta_2=25
$$

由于：

$$
\left|
\beta\tanh\left(\frac{x}{\beta}\right)
\right|
\leq \beta
$$

最终输出存在上界：

$$
|\operatorname{SiTU\text{-}GLU}(x)|
\leq \beta_1\beta_2=100
$$

![[figure4.png]]

图 4 显示：

- 在原点附近，SiTU-GLU 接近 SwiGLU；
- 输入很大时，SwiGLU 持续增长；
- SiTU-GLU 平滑趋近固定上界。

> [!tip] 平滑限制优于训练后补救
> 这里直接在激活函数层面限制异常值来源，减少依赖 gradient clipping 或异常检测等外部补丁。
>
> 它特别适合 FP8、FP4 等低精度训练和推理，因为低精度格式对离群值非常敏感。

> [!question] 是否牺牲大幅值通道的信息表达
> 有界激活必然压缩极端正激活之间的差异。作者证明了稳定性与数学上界，但没有展示 saturation 比例、激活分布或不同 $\beta_1,\beta_2$ 的下游消融。

---

### Quantile Balancing

### 核心直觉

**不要通过固定步长慢慢纠正专家负载，而是直接计算让每个专家接近目标负载所需的 router bias。**

router 首先计算：

$$
s_i=\operatorname{Sigmoid}(W_rx_i)
$$

选择专家时加入 bias：

$$
T_i=\operatorname{argtopk}(s_i+b)
$$

实际 mixture 权重仍由原始 $s_i$ 计算，因此 $b$ 只影响分配，不直接改变专家输出权重。

传统方法按照专家过载或欠载，用固定步长更新 $b_j$。896 个专家下，这种方式容易出现：

- 更新过慢；
- 负载振荡；
- 部分专家长期得不到训练；
- 不同设备计算时间严重不平衡。

K3 根据当前 batch 中的 router margin 分位数计算下一步 bias：

$$
\hat{b}_j^{(t+1)} = -\operatorname{quantile}_{1-k/n} \left( s_{:,j}-\alpha^{(t)} \right)
$$

再移除公共偏移：

$$
b^{(t+1)} =\hat{b}^{(t+1)} -\operatorname{mean} \left( \hat{b}^{(t+1)} \right)\mathbf{1}
$$

![[figure5.png]]

图 5 中，原本四个专家负载为：

$$
(4,3,1,0)
$$

QB 调整后变为：

$$
(2,2,2,2)
$$

实际训练中无法聚合数百万 margin 求精确分位数，因此作者使用分布式直方图估计。各 rank 只需要 all-reduce 每个 bin 的计数。

> [!tip] QB 同时服务模型训练与分布式系统
> 路由均衡通常被视为模型优化问题。K3 中，均衡还决定各 EP rank 的执行时间、通信缓冲区大小与内存碎片。
>
> QB 先让专家层面的统计更稳定，MoonEP 再让设备层面的 token 数量完全一致。

> [!warning] 路由均衡与专家专业化可能存在张力
> 强制每个专家接近相同负载，有利于系统吞吐，却可能迫使部分 token 进入语义上次优的专家。
>
> 作者通过只修改 Top-k 选择、不修改 mixture weight 来减轻影响，但报告没有展示 QB 对专家专业化程度或路由质量的影响。

---

## Native Vision：从第一天开始共同训练视觉和语言

### 核心直觉

**视觉表示应当由语言建模目标直接塑造，避免预训练视觉编码器与语言模型联合训练时出现目标和尺度不匹配。**

Kimi K3 的视觉路径为：

$$
\text{Image or Video}
\rightarrow
\text{MoonViT-V2}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{Shared LLM Backbone}
$$

MoonViT-V2：

- 27 层；
- 约 401M 参数；
- 使用 RMSNorm；
- 去除线性层和注意力投影中的 bias；
- 图像与视频共享参数；
- 空间注意力与时间注意力分解；
- 通过 temporal pooling 压缩视频 token；
- 通过 $2\times2$ pixel shuffle 将视觉 token 数量减少四倍；
- 支持最高约 $3584\times3584$ 的图像输入。

### 从随机初始化训练视觉编码器

K2.5 使用 SigLIP 初始化视觉编码器。K3 的 MoonViT-V2 完全从随机初始化开始，与语言模型一起执行 next-token prediction。

![[figure6.png]]

图 6 展示：

- SigLIP 初始化的 MoonViT-3D 梯度范数更高；
- 训练过程中出现更多梯度尖峰；
- 从零训练的 MoonViT-V2 更稳定；
- 最终视觉评测与 SigLIP 初始化基线相当。

作者给出的解释是，contrastive pre-training 更偏向全局语义，而 next-token prediction 会推动编码器保留 OCR、结构、局部文字和细粒度视觉线索。

> [!tip] 原生多模态的实际含义
> 这里的原生多模态包含两层含义：
>
> - 视觉编码器从训练开始就参与语言建模；
> - 文本、截图、代码、视频帧和工具反馈进入同一上下文流。
>
> 这为 Agent 执行代码、观察截图、修改代码、再次观察提供了统一接口。

> [!warning] 从零训练的优势仍缺少完整控制实验
> 图 6 主要展示梯度稳定性。报告称两种初始化在视觉任务上表现相当，但没有给出完整结果表、训练数据控制或收敛速度对比。
>
> 因此可以确认从零训练可行且更稳定，暂时无法确认它在最终能力上显著优于对比学习初始化。

---

## Per-Head Muon：按注意力头分别归一化优化方向

### 核心直觉

**不同注意力头的梯度尺度差异较大，将整个 Q、K、V 矩阵共同正交化，容易让大梯度 head 主导更新。**

Muon 对矩阵参数的 momentum 执行 Newton-Schulz orthogonalization。K3 将 Q、K、V 按 head 划分，对每个 head 的 momentum block 分别正交化。

这样可以：

- 平衡不同 head 的更新尺度；
- 避免小梯度 head 被大梯度 head 淹没；
- 提升大规模训练稳定性；
- 减少对完整高矩阵执行 Newton-Schulz 的开销。

> [!question] 缺少独立量化结果
> 报告称 Per-Head Muon 改善稳定性，但没有公开 loss 曲线、head 梯度分布或相对普通 Muon 的下游提升。
>
> 这是一个合理的工程改进，目前证据主要来自作者陈述。

---

## Pre-Training

## Scaling Law 与 2.5 倍效率

K3 相比 K2 的主要规模变化包括：

| 项目             | Kimi K2 |        Kimi K3 |
| -------------- | ------: | -------------: |
| 总参数            |   1.04T |          2.78T |
| 激活参数           |   32.6B |         104.2B |
| 层数             |      61 |             93 |
| Routed Experts |     384 |            896 |
| 每 token 激活专家   |       8 |             16 |
| 训练上下文          |    128K |             1M |
| 注意力            |     MLA | KDA-MLA Hybrid |
| 激活函数           |  SwiGLU |       SiTU-GLU |

作者重新搜索了 batch size、learning rate、tokens-per-parameter 和模型形状等超参数。

![[figure7.png]]

图 7 中 K3 的 scaling curve 相对 K2 向左下移动。按照拟合曲线，在达到相同验证损失时，K3 所需计算量约为 K2 的：

$$
\frac{1}{2.5}
$$

作者将其描述为约 2.5 倍 overall scaling efficiency。

> [!warning] 2.5 倍是配方整体收益
> 这个数字同时包含：
>
> - 架构变化；
> - 数据变化；
> - 训练超参数变化；
> - 模型形状变化；
> - 优化器变化。
>
> 它不能被直接理解为 KDA 相比标准注意力快 2.5 倍，也不能归因到某一个模块。

作者还分别为 cosine decay 与 WSD 搜索最佳超参数，并发现 cosine decay 最终 loss 更低。这个实验强调了一个常被忽视的问题：学习率调度器必须在各自最佳超参数下比较。

---

## 百万 Token 上下文扩展

### 核心直觉

**上下文窗口长度需要通过包含真正长程依赖的任务训练，单纯输入很长的文档无法形成长程能力。**

K3 使用四阶段 curriculum：

$$
8K\rightarrow64K\rightarrow256K\rightarrow1M
$$

其中：

- 8K 到 64K 在主要预训练阶段完成；
- 256K 到 1M 集中在 cooldown 阶段；
- 昂贵的百万 token 训练只占总训练的一小部分。

长上下文数据包括：

- 自然长文档；
- 长视频；
- 多模态文档；
- 人工排列和拼接的多个子任务；
- 必须整合上下文不同位置才能解决的合成任务。

> [!tip] 长度与长程依赖被明确区分
> 作者明确指出，输入长度本身不能产生 long-range capability。
>
> 合成任务会把必要证据分散到整个 1M 上下文中，迫使模型真正使用远距离信息。这个训练原则比单纯扩展 context window 更重要。

> [!warning] 缺少位置分辨率评估
> 报告没有给出完整的：
>
> - 不同上下文长度下的性能曲线；
> - 不同证据位置的召回率；
> - lost-in-the-middle 分析；
> - KDA 状态随长度增长的信息覆盖情况。
>
> 因此，1M token 的可运行性得到充分说明，1M 范围内的细粒度有效记忆能力仍缺乏系统刻画。

---

## Post-Training：先训练专家，再合并成一个模型

### 总体流程

**直觉**：不同任务领域和不同思考预算需要不同策略，先分别优化这些策略，再将它们压回一个统一模型。**

后训练分为三阶段：

$$
\text{SFT}
\rightarrow
\text{Domain and Effort RL}
\rightarrow
\text{MOPD}
$$

1. SFT 建立初始 Agent 能力；
2. RL 训练领域和推理强度专家；
3. MOPD 将九个专家整合到统一模型。

---

## SFT：建立可执行的 Agent 冷启动策略

SFT 数据由前代 Kimi 领域专家生成，再经过：

- 多阶段验证；
- human-in-the-loop 标注；
- XTML 统一序列化；
- 长轨迹工具调用数据扩展。

SFT 的目标主要包括：

- 自适应推理；
- 精确工具调用；
- 长任务执行；
- 结构化工具参数生成；
- 不同 Agent harness 下的交互格式。

K3 从 SFT 阶段开始就执行量化感知训练，为后续部署做准备。

---

## Multi-Domain × Multi-Effort RL

### 核心直觉

**任务领域决定模型应该学习什么，推理 effort 决定模型愿意为问题投入多少计算。**

作者划分三个 RL 领域：

1. **General**

   - 知识；
   - 推理；
   - 视觉；
   - 忠实性；
   - 搜索；
   - 专业知识工作。

2. **General Agents**

   - 长程助手任务；
   - 深度研究；
   - 写作；
   - 多工具工作流。

3. **Coding Agents**

   - 软件工程；
   - GPU kernel；
   - Web 开发；
   - 编程经验。

每个领域训练：

$$
e\in{\text{low},\text{high},\text{max}}
$$

总共得到：

$$
3\times3=9
$$

个专家模型。

![[figure8.png]]

图 8 显示，随着 RL FLOPs 增加：

- 任务分数总体提高；
- 平均工具调用步数也增加；
- 模型逐渐学会执行更长的轨迹。

> [!note] 步数增加本身不是目标
> 更多工具调用可能代表更强的探索，也可能代表效率下降。作者进一步引入推理预算控制，避免模型通过无限延长轨迹取得奖励。

---

## Reasoning Effort RL

对于问题 $x$，先用 cold-start 模型估计基础预算：

$$
b_0(x)
$$

若轨迹 $y$ 的 token 消耗满足：

$$
T(y)>\tau b_0(x)
$$

则直接将奖励覆盖为：

$$
r=-1
$$

其中：

- general task 中，$T(y)$ 主要统计思考 token；
- agentic task 中，$T(y)$ 包含推理文本和工具调用参数；
- $\tau$ 是预算倍数。

训练顺序为：

1. 先使用较大的 $\tau$ 训练 max expert；
2. 逐步减小 $\tau$；
3. 获得 high 和 low expert。

**本质上，作者将推理强度转化成一个受约束的强化学习问题。**

> [!tip] 以问题自身难度为基准
> 不同问题使用不同的 $b_0(x)$，避免对简单题和复杂题设置相同的绝对 token 限制。
>
> 这比固定最大生成长度更合理，因为预算相对于问题难度进行归一化。

> [!warning] 人工调节仍然较多
> $\tau$ 按领域配置，并使用 human-in-the-loop 指导。报告没有说明预算估计误差、不同领域的具体 $\tau$，也没有解释统一模型是否能在部署时平滑插值 effort。

---

## Agentic Generative Reward Model

### 核心直觉

**不可验证任务无法通过单一规则评分，因此让奖励模型先构造 rubric，再基于 rubric 比较候选答案。**

GRM 必须按以下流程执行：

1. 阅读最终产物；
2. 生成评价 rubric；
3. 按 rubric 对每个候选评分；
4. 将分数记录到 scorepad；
5. 进行 tournament-style 二元比较。

为防止模型通过冗长输出赢得奖励，作者设置 verbosity budget。若候选长度超过：

$$
\sigma\ell_0
$$

则该候选自动输掉比较，其中 $\ell_0$ 是 cold-start 模型的基础输出长度。

> [!tip] Rubric generation 提高了奖励的任务适应性
> 专业工作流、网页、报告和研究任务的评价标准差异很大。先生成 rubric，可以让同一个奖励模型适配不同产物类型。

> [!warning] 奖励模型仍可能自洽地犯错
> 强制 rubric 和 scorepad 能改善过程纪律，却无法保证 rubric 本身正确。特别是专业知识任务中，奖励模型可能构造出看似完整但遗漏关键要求的评价标准。

---

## Partial Rollout：长轨迹不再阻塞整个 RL Batch

### 核心直觉

**Agent 轨迹长度差异极大，等待最慢轨迹结束会浪费大量 GPU 时间。**

每轮对 $N$ 个 prompt 采样 $K$ 条轨迹，总共维护：

$$
N\times K
$$

条活动轨迹。

当完成比例达到：

$$
\lambda NK
$$

时，当前 rollout 阶段暂停，其余轨迹保存并放入队列，在下一次迭代中优先恢复。

这允许：

- 已完成样本立即进入训练；
- 超长轨迹跨越多个 iteration；
- 减少 straggler 导致的空闲。

代价是产生严重 data staleness。作者使用 per-token regularization，限制策略更新落在局部邻域，使旧策略采样的轨迹仍可用于训练。

> [!warning] RL 算法细节透明度有限
> 报告没有完整给出 per-token regularization 的公式，只说明沿用 Kimi K2.5 算法。
>
> 因此，很难独立判断其在极端 off-policy 条件下稳定的具体原因。

---

## Multi-Teacher On-Policy Distillation

### 核心直觉

**让统一 student 在自己的分布上生成轨迹，同时用对应领域与 effort 的 teacher 提供逐 token 密集奖励。**

对领域 $d$ 和 effort $e$，选择对应 teacher：

$$
\pi_{\text{teacher}}^{(d,e)}
$$

逐 token 蒸馏奖励为：

$$
r_{\text{opd}}^d \left( y_t\mid e,x,y_{<t} \right) = \operatorname{clip} \left[ \operatorname{sg} \left( \log \frac{ \pi_{\text{teacher}}^{(d,e)} (y_t\mid x,y_{<t}) }{ \pi_\theta (y_t\mid e,x,y_{<t}) } \right), -R_{\max},R_{\max} \right]
$$

其中：

- student 负责采样当前轨迹；
- teacher 评价 student 已经采样出的 token；
- log-ratio 表示 teacher 相对 student 更偏好该 token 的程度；
- stop-gradient 防止奖励信号反向进入 teacher；
- clipping 防止极端比值破坏训练。

与离线蒸馏相比，on-policy distillation 在 student 自己会访问的状态上提供监督，更适合长轨迹任务中的分布偏移。

> [!tip] 把能力组合问题转成策略蒸馏
> 作者没有直接在一个模型里同时训练所有领域和 effort，而是先形成九个稳定策略，再将它们作为行为教师。
>
> 这降低了不同奖励目标在早期训练阶段互相干扰的风险。

> [!question] 为何需要九个完整教师
> 这种设计训练成本很高。报告没有比较：
>
> - 单模型直接条件化 effort；
> - 共享骨干加领域 adapter；
> - 一个 teacher 通过 prompt 控制领域和预算；
> - 九教师 MOPD。
>
> 九教师方案有效，但其必要性尚未得到充分证明。

---

## Deployment-Aware Post-Training

### MXFP4 Quantization-Aware Training

MoE expert 权重占据绝大多数参数内存。K3 将：

- routed expert 权重量化为 MXFP4；
- expert 输入 activation 使用 MXFP8；
- attention、router、shared expert 和 latent projection 保持更高精度。

量化感知训练贯穿：

$$
\text{SFT}+\text{RL}
$$

rollout 和训练使用相同量化格式，避免训练策略与部署策略之间出现数值差异。

> [!tip] 部署约束进入 RL 阶段
> 很多模型在后训练结束后才做 PTQ，容易破坏工具调用和长轨迹稳定性。K3 从 SFT 开始适应量化误差，使策略学习发生在实际部署精度下。

---

### EAGLE-3 Draft Model

K3 在预训练阶段包含一个 MTP layer。作者将其进一步微调为 EAGLE-3 风格的 speculative decoding draft model。

draft 输入融合 AttnRes 第 1、第 4 和最后一个 block 的表示：

$$
h_{\text{fuse}} = W_{E3} [h_{\text{low}};h_{\text{mid}};h_{\text{high}}]
$$

初始化：

$$
W_{E3}=[0\quad0\quad I]
$$

因此初始状态只使用已经熟悉的 high-level feature，随后逐渐学习使用低层和中层表示。

作者没有使用普通 KL loss，而是直接优化 speculative decoding 的接受率：

$$
L_{\text{LK}} = -\log \sum_{x\in V} \min \left( p(x),q(x) \right)
$$

其中 $p$ 是 target model，$q$ 是 draft model。

> [!tip] 目标函数直接对齐部署指标
> speculative decoding 的实际收益由 token 接受率决定。KL 较低并不保证接受率最高，因此作者直接优化分布重叠面积。

---

## RL Task Synthesis and Agentic Environments

## Unified White-Box RL Environment

### 核心直觉

**只在一种 Agent harness 上训练，会让模型记住工具格式和上下文管理习惯，难以迁移到其他 Agent 系统。**

作者将 Agent harness 拆成可组合模块：

- tool interface；
- system prompt；
- context management；
- skill；
- memory；
- subagent；
- interaction protocol。

这些模块可以组合成：

- Kimi Code；
- Claude Code；
- Codex；
- OpenClaw；
- Hermes；
- 新的自定义 harness。

训练时，不同任务使用不同组合，迫使模型学习更通用的工具调用和任务执行策略。

> [!tip] Agent 泛化的训练单位从任务升级为 harness 配置
> 传统 RL 主要增加任务多样性。K3 同时增加任务分布与 Agent runtime 分布，试图减少模型对某套 scaffolding 的依赖。

> [!question] 是否真正实现 cross-harness generalization
> 报告展示了不同 benchmark 使用不同 harness，但没有给出：
>
> - 训练时未见 harness 上的零样本迁移；
> - 固定任务、切换 harness 的控制实验；
> - memory 和 context management 模块变化造成的性能曲线。
>
> 该方向非常重要，公开证据仍然有限。

---

## Knowledge-Graph-Guided Task Synthesis

### 核心直觉

**依靠随机网页采样容易重复热门知识，层级知识图谱可以主动控制任务覆盖范围和知识粒度。**

![[figure9.png]]

构建流程为：

1. 从粗粒度 seed node 开始；
2. 为每个节点分配 Agent；
3. Agent 搜索网页理解概念；
4. 检查图中是否已有等价或相关节点；
5. 添加更细粒度子节点；
6. 递归扩展，直到概念足够原子化。

任务生成时：

1. 从不同层级采样一个或多个相关节点；
2. 结合祖先节点生成搜索关键词；
3. 检索论文、博客、代码仓库等真实材料；
4. 选择 coding、knowledge、vision 等任务类型；
5. 基于材料合成训练任务。

我认为这里的关键并非知识图谱本身，而是作者用它显式控制：

- 领域覆盖；
- 概念粒度；
- 长尾知识；
- 相关概念组合；
- 任务类型分布。

> [!warning] 自演化知识图谱可能累积语义错误
> 节点是否足够原子、两个概念是否等价，主要由 Agent 判断。错误合并或错误层级关系可能持续传播到后续任务合成。
>
> 报告没有说明图谱质量评估、人工抽检比例或错误节点修复机制。

---

## Verifiable Agentic Tasks

作者强调尽可能将 Agent RL 转化成可验证任务，主要包括：

### 多步搜索

模型需要：

- 分解研究问题；
- 逐步搜索；
- 收集证据；
- 交叉验证；
- 输出可核查答案。

### 专业知识工作

覆盖：

- 投资银行；
- 数据分析；
- 法律；
- 财务；
- Office deliverables。

任务通常包含几十到几百步工具操作，并要求交付最终文件或分析结果。

### 视觉工具使用

模型在沙箱中调用 Python：

- crop；
- zoom；
- 图像变换；
- 数值计算；
- 中间结果验证。

新的截图和生成图像重新进入上下文，形成视觉闭环。

> [!tip] 视觉能力从单次识别转向主动观测
> 模型可以决定需要放大哪里、执行什么变换、验证哪个中间量。视觉推理逐渐接近主动感知问题。

---

## Kernel Optimization Tasks

奖励同时考虑正确性与性能：

- 超过数值误差阈值，奖励为 0；
- 达到专家实现，奖励约为 0.5；
- 接近硬件 roofline，奖励趋近 1。

任务覆盖：

- CUDA；
- Triton；
- CuTe DSL；
- ThunderKittens；
- TileLang；
- BF16、FP8 和 FP4。

作者还构建反作弊系统，检测：

- CUDA graph replay；
- 输入缓存；
- 非法降低精度；
- 针对 benchmark 的投机实现。

这里反映了一个重要原则：**可验证奖励不仅要判断任务完成，还要判断模型是否通过预期方式完成。**

---

## Persistent Personal Assistant Environment

### 核心直觉

**真正的助手任务会跨越多个应用和多个时间点，环境状态需要持续演化。**

作者实现 Gmail、Notion、Slack、Canvas 等应用的 mock environment。

单条轨迹可能包含：

- 多个模拟工作日；
- 数十个相互依赖事件；
- 数千次工具调用；
- 数百万累计上下文 token；
- 持续变化的 workspace 状态。

任务的初始 workspace 也由 Agent 根据网页材料自动构造。

我注意到这里训练的能力已经超出单轮 tool use，开始涉及：

- 延迟任务；
- 跨应用状态同步；
- 事件流处理；
- 长期目标保持；
- 中断后恢复；
- 对过去操作结果的持续追踪。

---

## Autonomous Execution Tasks

### 核心直觉

**不给模型参考轨迹，只提供目标、工具、限制和 verifier，让模型自己发现完成任务的方法。**

每个 AET 包含：

- initial state；
- constrained goal；
- tool action space；
- execution budget；
- independent verifier。

模型需要自行完成：

- 任务分解；
- 工具选择；
- 规划；
- 错误恢复；
- 终止判断。

奖励来自最终环境状态，而非模型声称任务已经完成。

![[figure10.png]]

作者使用：

- public verifier 提供诊断反馈；
- hidden verifier 检查 held-out 场景；
- 有限提交预算；
- verifier 与 Agent 隔离。

这些设计减少模型直接攻击或过拟合 verifier。

> [!tip] Verify-in-the-loop
> 模型在任务中不断经历：
>
> $$
> \text{Hypothesis}
> \rightarrow
> \text{Action}
> \rightarrow
> \text{Verification}
> \rightarrow
> \text{Adaptation}
> $$
>
> 这是 K3 Agent 训练中最核心的行为闭环。

---

## Infrastructure

## KDA Kernel 与 Context Parallelism

### 核心直觉

**KDA 的状态固定大小，适合跨设备传输，但状态更新具有递归依赖，需要重新设计并行方式。**

作者分别为不同阶段设计内核：

- 训练与 prefill：FlashKDA chunkwise kernel；
- 单卡超长 prefill：SM 级 context parallelism；
- 跨卡长序列：KDA Context Parallelism；
- decoding：融合 recurrence、卷积、gate 和 normalization 的专用内核。

KDA Context Parallelism 将每个序列片段表示成：

1. 对输入状态的线性 transition；
2. 从零状态开始产生的 local state。

若片段 $i$ 的变换写成：

$$
S_{\text{out}} = M_iS_{\text{in}} + \widetilde{S}_i
$$

则两个片段可以组合为：

$$
S_2 = M_2M_1S_0 + M_2\widetilde{S}_1 + \widetilde{S}_2
$$

这种变换具有结合性，因此可以使用 prefix scan 并行恢复每个设备的初始状态。

与 softmax attention 不同，设备之间传输的是固定大小状态，而非随序列增长的 KV block。

> [!tip] 固定状态是百万 token 并行的关键
> KDA 的主要价值不仅是理论复杂度更低，还在于跨设备通信量与上下文长度解耦。

---

## MoonEP：完全均衡的 Expert Parallelism

### 核心直觉

**MoE 的实际吞吐由最繁忙的设备决定，因此平均负载均衡仍然不够，需要让每个 EP rank 接收完全相同数量的 token。**

传统 EP 中，router 会造成：

- 各 rank token 数量不同；
- 动态 tensor shape；
- 内存碎片；
- host-device 同步；
- 慢 rank 阻塞整个训练 step。

MoonEP 动态复制部分专家，并迁移 token，使每个 rank 恰好接收：

$$
S\times K
$$

个 token。

作者证明，每个 rank 最多保留：

$$
\frac{E}{R}
$$

个 redundant expert slot，就一定存在可行的完全均衡方案。

MoonEP 的系统收益包括：

- 固定计算 shape；
- 消除逐层 host 同步；
- zero-copy token permutation；
- 更小的通信 buffer；
- workload-aware expert GEMM scheduling；
- shared expert 与 routed expert 重叠执行。

> [!tip] QB 与 MoonEP 解决不同层次的均衡
>
> - QB 调整 token 到 expert 的统计负载；
> - MoonEP 调整 expert 到 device 的物理放置。
>
> 前者服务模型学习，后者保证硬件执行。

---

## Memory-Efficient Training

作者构建统一 activation manager，将以下机制抽象成 tensor 级 storage policy：

- recomputation；
- FP8 activation quantization；
- CPU offload；
- remote offload；
- prefetch；
- cross-layer recomputation。

这些策略通过 annotation 声明，与模型代码解耦。

同时使用：

- ZeRO-1；
- Pipeline ZeRO-2；
- Pipeline Parallelism；
- Virtual Pipeline；
- Expert Parallelism；
- Context Parallelism。

图 11 展示计算、通信、激活卸载、ViT 前后向和专家通信在 pipeline 中的重叠。

![[figure11.png]]

这种基础设施没有直接改变模型表达能力，却决定 2.8T 模型能否在有限显存下保持有效吞吐。

---

## Million-Token Agentic RL Infrastructure

### External KV Cache Pool

长轨迹跨越多个 RL iteration 时，未完成请求需要保留 prefix cache。若重新 prefill 数十万 token，成本极高。

作者采用 write-back 设计：

- 活跃 decoding block 留在 GPU；
- 即将被 GPU 淘汰的可复用 prefix 写入 CPU DRAM；
- 再次使用前预取回 GPU；
- KDA state 与 MLA KV block 共同迁移；
- 训练阶段将 model weight 和 optimizer state 临时卸载到 NVMe，为 CPU cache pool 腾出空间。

### Rollout Auto-Throttling

随着 Agent 轨迹增长，KV Cache 压力逐渐增大。固定并发量会出现：

- 初期 GPU 利用不足；
- 后期 cache 爆满和请求抢占。

调度器根据：

- 活跃请求数；
- 等待请求数；
- KV Cache 利用率；

动态调整发送到 inference engine 的请求数量。

---

## AgentENV：可恢复的微虚拟机沙箱

### 核心直觉

**长程 Agent RL 不只需要保存模型上下文，还必须保存文件系统、进程、应用和操作系统状态。**

AgentENV 基于 Firecracker microVM，提供：

- **Pause and Resume**：等待模型生成时暂停 VM；
- **Fork**：从相同状态创建副本，用于无副作用 reward judging；
- **Snapshot**：定期保存状态，用于错误恢复；
- 增量 checkpoint：只保存变更的内存页；
- checkpoint 延迟最低约 133 ms；
- resume 延迟最低约 49 ms；
- copy-on-write 与 page cache 优化；
- 实际工作负载中最高约 6.5 倍内存超分。

训练和评测过程中共创建：

- 51,219,741 个 sandbox；
- 1,505,678 个 image。

> [!tip] Agent 状态被分成两部分
>
> - 模型侧状态：context、MLA KV Cache、KDA recurrent state；
> - 环境侧状态：文件、进程、应用数据和 VM 内存。
>
> K3 的长期 Agent 训练能够跨 iteration 延续，依赖两类状态同时持久化。

> [!warning] 这是能力优势，也是复现壁垒
> 没有相同规模的 sandbox 调度、cache pool 和分布式训练系统，外部研究者很难复现其长轨迹 RL 配方。

---

## KDA-Aware Prefix Cache 与在线服务

混合 KDA-MLA 模型同时维护：

- MLA 的 token 级 KV Cache；
- KDA 的固定 recurrent state。

作者将二者共同放入 paged cache，并在细粒度 hash block 边界保存可恢复状态。

![[figure12.png]]

核心设计包括：

- 物理 cache block 与 prefix hash block 解耦；
- 每 512 token 可以建立 prefix checkpoint；
- cache hit 时同时恢复 MLA block 与 KDA state；
- 多 cache group 的 checkpoint 必须原子失效；
- 命中的 block 在复制前需要 pin，避免并发 eviction。

Fleet 层再使用：

- cache-aware affinity scheduling；
- consistent hashing 主备集群；
- 按请求成本划分资源预算；
- 长上下文请求与短请求分开 admission control。

这里解决的主要问题是，1M token 请求的成本可能是普通请求的上千倍，不能继续按照平均请求数进行容量规划。

---

## Experiments

## 总体结果

![[figure1.png]]

Kimi K3 的整体定位比较清楚：

- 接近 Claude Fable 5 与 GPT-5.6 Sol；
- 多数项目领先 Claude Opus 4.8、GPT-5.5 和 GLM-5.2；
- Agent、长程编程和工具增强视觉是最突出的能力；
- 研究级纯推理和部分复杂 computer-use 任务仍有明显差距。

---

### Reasoning and Knowledge

代表结果：

- GPQA Diamond：93.5
- CritPt：23.4
- HLE-Full：43.5
- HLE-Full with tools：56.0

> [!note] 能力边界
> K3 在研究生级知识推理上已经接近最强模型，但 CritPt 和 HLE 表明其研究级开放问题能力仍落后于 Claude Fable 5 和 GPT-5.6 Sol。
>
> 这说明 Agent 执行能力的领先没有完全转化成纯推理上限。

---

### Coding

代表结果：

- ProgramBench：77.8，表中最高；
- Terminal-Bench 2.1：88.3，接近 GPT-5.6 Sol 的 88.8；
- FrontierSWE：81.2；
- SWE-Marathon：42.0，表中最高；
- DeepSWE：67.5。

K3 在 GPU kernel、终端操作和长程软件工程上尤其强。这与 RL 中大量 kernel optimization、web development 和 sandbox execution 数据高度一致。

> [!warning] Harness 影响不能忽略
> 不同模型在 Kimi Code、Claude Code 或 Codex 中运行。Agent benchmark 的最终得分同时受到：
>
> - 基础模型；
> - system prompt；
> - 工具接口；
> - context management；
> - harness retry 策略；
>
> 等因素影响。因此表格并非完全纯粹的模型能力比较。

---

### Agentic

代表结果：

- BrowseComp：91.2；
- DeepSearchQA：95.0；
- ResearchRubrics：76.2；
- MCPMark-Verified：94.5；
- AutomationBench：30.8；
- SpreadsheetBench 2：34.8；
- Harvey Lab-AA：94.6。

这些结果与训练配方高度对齐：

- 多步搜索对应 BrowseComp 和 DeepSearchQA；
- 白盒工具环境对应 MCP 系列；
- 专业工作流对应 GDPval、Finance 和 Legal；
- 长程执行对应 AutomationBench 与 OSWorld。

我认为 Agent 结果是整篇报告最有说服力的部分。训练任务、环境结构和 benchmark 能力之间存在清晰的因果对应关系。

---

### Vision

代表结果：

- OmniDocBench：91.1；
- Video-MME：90.0；
- Math-Vision：94.3，使用 Python 后 97.8；
- ZeroBench：23.0，使用 Python 后 41.0；
- CharXiv：84.8，使用工具后 91.3。

工具带来的提升非常明显。K3 的视觉能力更适合被理解为：

$$
\text{Visual Perception}
+
\text{Code Execution}
+
\text{Iterative Observation}
$$

而非单次视觉编码器前向。

> [!tip] Tool-augmented vision 是 K3 的重要能力形态
> 模型能够对图像执行裁剪、缩放、计算和验证。视觉编码器提供初始观察，Agent loop 决定后续观察过程。

---

## Cost Efficiency

![[figure13.png]]

报告比较了 Kimi Code Bench、BrowseComp、GDPval-AA 和 AA-Briefcase 上的分数与单任务成本。

K3 多数情况下位于 cost-performance frontier 附近：

- BrowseComp 达到 91.2，单任务成本约 2.03 美元；
- 成本约为 GPT-5.6 Sol 的一半；
- 相比 Claude 最大 effort 版本便宜一个数量级左右；
- GDPval-AA 上接近 GPT-5.6 Sol，成本低约 13%。

> [!warning] 成本比较受定价和 harness 影响
> 部分成本来自内部测量，部分来自公开 API 定价。不同模型的：
>
> - 缓存折扣；
> - tool token 计费；
> - fallback；
> - reasoning token 可见性；
> - harness overhead；
>
> 可能不同。因此图 13 更适合判断大致成本位置，不宜视为严格的硬件效率比较。

---

## Critical Assessment

### 真正有价值的创新

我认为 Kimi K3 最值得关注的创新可以分为四层。

#### 1. 模型结构层

- KDA 与 MLA 的职责分离；
- lower-bounded decay 对齐 Tensor Core；
- AttnRes 将深度变成可检索维度；
- Stable LatentMoE 处理极端专家稀疏；
- SiTU-GLU 限制低精度激活异常；
- QB 直接求解目标专家负载；
- Per-Head Muon 平衡注意力头更新。

#### 2. 后训练层

- domain × effort 的九专家训练；
- 基于问题难度的推理预算控制；
- MOPD 统一多种专家策略；
- Agentic GRM 的 rubric-scorepad 评价流程；
- QAT 贯穿 SFT 和 RL；
- draft model 直接优化 speculative acceptance rate。

#### 3. 环境层

- 可组合的 white-box Agent harness；
- 基于知识图谱控制任务覆盖；
- persistent assistant environment；
- verify-in-the-loop AET；
- 视觉与代码工具共同参与的多轮观察。

#### 4. 系统层

- KDA Context Parallelism；
- MoonEP 完全均衡专家并行；
- 外部 KV Cache pool；
- AgentENV 可恢复 microVM；
- 混合 KDA-MLA prefix cache；
- 面向百万 token 请求的 fleet scheduling。

---

### 报告最明显的证据缺口

> [!warning] 缺少完整架构消融
> 2.5 倍效率是整体配方结果，无法判断各模块的边际贡献。

> [!warning] 训练数据透明度不足
> 报告没有公开总训练 token、详细数据比例、过滤阈值、合成数据占比和具体 cooldown 预算。

> [!warning] RL 算法细节依赖前序报告
> 极端 stale rollout 下最关键的 per-token regularization 没有在本报告中完整展开。

> [!warning] 内部环境难以复现
> Persistent assistant、AET、reward model、kernel task 和专业工作流多数依赖内部数据与基础设施。

> [!warning] 评测协议存在异质性
> 不同 harness、fallback、cyberguard、内部 benchmark 和最大 effort 设置降低了部分横向比较的严格性。

> [!warning] 百万上下文缺少功能性剖析
> 系统已经能运行 1M token，但有效记忆距离、位置鲁棒性和信息干扰仍缺少充分评估。

---

## Future

### 作者明确暴露出的改进方向

- 提升 CritPt、HLE 等研究级推理能力；
- 改善复杂 computer-use 和专业知识工作；
- 减少长轨迹中的无效 debugging loop；
- 提升最终产物提交前的验证能力；
- 提升复杂执行链最后阶段的完成率。

### 我认为更值得继续研究的方向

#### 1. Token、Layer 和 Expert Routing 的统一理论

K3 分别在三个维度使用：

- KDA 和 MLA；
- AttnRes；
- MoE Router。

三者本质上都在解决信息选择问题。可以研究统一的 routing framework，让 token、layer 和 expert 的选择共享查询表示或稀疏预算。

#### 2. Query-Dependent Attention Residuals

当前 AttnRes 使用每层固定 pseudo-query。可以令 query 依赖当前 token、当前任务或当前推理阶段，使模型动态选择深度路径。

#### 3. KDA 状态的可解释性与容量边界

需要研究：

- 哪些信息进入递归状态；
- 新写入如何覆盖旧关联；
- 状态容量何时饱和；
- 百万 token 下发生何种干扰；
- periodic MLA 如何修复 KDA 的压缩误差。

#### 4. 直接训练统一 Multi-Effort Policy

九专家加 MOPD 成本很高。可以研究单一条件策略：

$$
\pi(y\mid x,e,d)
$$

直接根据领域 $d$ 和 effort $e$ 调整策略，同时避免多个 teacher 的训练与存储成本。

#### 5. Agent Harness 的真正分布外泛化

应构造固定任务集，只改变：

- tool schema；
- system prompt；
- memory system；
- context compaction；
- subagent protocol。

从而测量模型是否学会了通用 Agent 策略。

#### 6. Environment-State Memory

K3 通过保存完整 VM 状态解决长期任务连续性。未来可以研究如何从环境状态中提取结构化 memory，使模型无需长期保存完整上下文和完整 sandbox snapshot。

#### 7. 可验证奖励的自动构造

AET 已经将 verifier 放入训练闭环。下一步可以让模型自动发现：

- 可验证子目标；
- 中间 invariant；
- 失败诊断；
- verifier blind spot；
- 防 reward hacking 约束。

#### 8. 开放训练基础设施

完整开放以下组件会显著提高研究价值：

- MoonEP；
- KCP；
- AgentENV 接口；
- white-box harness specification；
- RL environment generator；
- long-context evaluation suite；
- 组件级 scaling ablation。

---

# 可拆分的原子笔记列表

以下条目不属于上面的笔记正文：

- [[Kimi K3 的双轴 Scaling：预训练规模与测试时计算]]
- [[Kimi K3 的三维信息流：Token、Layer 与 Channel Mixing]]
- [[KDA 的 Delta Rule 状态更新]]
- [[KDA 中的读写遗忘机制]]
- [[Lower-Bounded Decay 如何将 KDA 映射到 Tensor Core]]
- [[KDA 与 MLA 的混合注意力分工]]
- [[NoPE MLA 的位置建模来源]]
- [[Attention Residuals：将网络深度变成可检索记忆]]
- [[Block Attention Residuals 的效率与信息瓶颈]]
- [[LatentMoE：在低维空间中执行专家路由]]
- [[Stable LatentMoE 的三项稳定化设计]]
- [[SiTU-GLU：面向低精度训练的有界激活函数]]
- [[Quantile Balancing：基于分位数的 MoE 负载均衡]]
- [[QB 与 MoonEP 的两级负载均衡]]
- [[MoonViT-V2：从零开始的原生视觉预训练]]
- [[Per-Head Muon：按注意力头正交化优化更新]]
- [[Kimi K3 的百万 Token 渐进式训练]]
- [[长上下文训练需要长程依赖而非单纯长文本]]
- [[Reasoning Effort RL：基于问题难度的推理预算]]
- [[Agentic Generative Reward Model 的 Rubric-Scorepad 流程]]
- [[Partial Rollout：跨 RL Iteration 的长轨迹训练]]
- [[Multi-Teacher On-Policy Distillation]]
- [[从 MTP Layer 到 EAGLE-3 Draft Model]]
- [[直接优化 Speculative Decoding Acceptance Rate]]
- [[Unified White-Box RL Environment]]
- [[基于自演化知识图谱的 Agent 任务合成]]
- [[Verify-in-the-Loop Autonomous Execution Tasks]]
- [[Persistent Personal Assistant Environment]]
- [[Tool-Augmented Visual Reasoning]]
- [[KDA Context Parallelism]]
- [[MoonEP：完全均衡的专家并行]]
- [[AgentENV：可恢复的 MicroVM Agent 沙箱]]
- [[模型状态与环境状态的双重持久化]]
- [[KDA-MLA 混合模型的 Prefix Cache]]
- [[百万 Token 在线请求的预算式调度]]
- [[Kimi K3 的 2.5 倍 Scaling Efficiency 应如何理解]]
- [[Kimi K3 技术报告的证据缺口与复现壁垒]]
