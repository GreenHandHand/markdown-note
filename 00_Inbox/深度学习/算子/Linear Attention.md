---
tags:
  - 深度学习
---

# Linear Attention

Transformer 中的自注意力通过显式计算任意两个 token 之间的相关性获得了很强的全局建模能力，但这种两两交互需要构造规模为 $N\times N$ 的注意力矩阵，使计算与显存开销随序列长度平方增长。Linear Attention 最初就是从这一计算瓶颈出发，希望通过重新组织注意力的计算形式，在保留全局信息交互能力的同时，使复杂度关于序列长度近似线性。

随着研究不断发展，Linear Attention 又逐渐与核方法、递归神经网络、Fast Weight、状态空间模型等方向产生联系，并从早期的高效 Attention 近似，发展为一类具有独立状态更新机制的序列建模方法。

## Softmax Attention

一切要从标准的 Softmax Attention 开始说起。标准的单头 Attention 可以写作
$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(  \dfrac{QK^{\top}}{\sqrt{ d }} \right)V
$$
其中
$$
\begin{align}
&q_{i},k_{i},v_{i},o_{i}\in \mathbb{R}^{d\times 1} \\
Q&=[q_{1},q_{2},\cdots,q_{n}]^{\top}\in \mathbb{R}^{n \times d} \\
K&=[k_{1},k_{2},\cdots,k_{n}]^{\top}\in \mathbb{R}^{n \times d} \\
V&=[v_{1},v_{2},\cdots,v_{n}]^{\top}\in \mathbb{R}^{n \times d_{v}} \\
O&=\operatorname{Attention}(Q,K,V)=[o_{1},o_{2},\cdots,o_{n}]\in \mathbb{R}^{n\times d_{v}}
\end{align}
$$

> [!tip] 这里全部采用列向量

> [!note]
> 一般只考虑 Causal 场景，意味着 $o_{t}$ 至多和 $Q_{[:t]},K_{[:t]},V_{[:t]}$ 相关。在这种情况下，Attention 公式可以写作 $O=\operatorname{softmax}(QK^{\top}/\sqrt{ d }+\log M)V$。其中注意力掩码矩阵 $\log M\in \mathbb{R}^{n\times n}$ 是一个严格上三角矩阵，对角线之上的元素使用 $-\infty$，从而屏蔽未来 token。

> [!info]- Softmax
> 为了方便推导，这里给出 Softmax 的公式。Softmax 即指数归一化，对于某一个维度中的一个向量的一个分量来说，就是
> $$
> \operatorname{softmax}(x_{t})=\dfrac{\exp(x_{t})}{\sum\limits_{i=0}^{n}\exp(x_{i})}
> $$
> 向量形式为
> $$
> \operatorname{softmax}(x)=\dfrac{\exp(x)}{\operatorname{sum}(\exp(x))}
> $$

> [!info] Softmax Attention 的定义
> 标准的 Softmax Attention 来自 《Attention is All You Need》，下面按照 [科学空间](https://kexue.fm/archives/11033/comment-page-1) 中给出的形式定义，也就是
> $$
> O=\operatorname{softmax}(QK^{\top}+\log M)V
> $$
> 这里省略了缩放因子 $1/\sqrt{ d }$，因为它总可以被吸收到 $Q,K$ 里面，$\operatorname{softmax}$ 是对第二个维度进行指数归一化，其中的 $M\in \mathbb{R}^{n\times n}$ 是一个掩码矩阵。
>
> Softmax Attention 用分量的形式^[这里实际包含了注意力掩码，也就是求和只到当前时间 $t$] 写出来则是
> $$
> o_{t}=\dfrac{\sum\limits_{j=1}^{t}\exp(q_{t}^{\top}k_{j})v_{j}}{\sum\limits_{j=1}^{t}\exp(q_{t}^{\top}k_{j})}
> $$
>
> 其中分母的作用主要是权重归一化，保持数值稳定性。另外就是如果我们给 $O$ 加上 RMSNorm^[关于这里自动消去的原因，RMSNorm 具有尺度不变性，而 softmax 计算得到的分母是对于分量来说是常数，因此在考虑 Softmax 后立即跟 RMSNorm 的情况，当忽略 $\epsilon$ 时 Softmax 的分母在数学上是可以消去的。]，那么分母也会自动消去，所以 Softmax Attention 的核心是分子部分（即 Attention 中真正负责信息混合的核心算子），即
> $$
> O=\exp(QK^{\top}+\log M)V=(\exp(QK^{\top}) \odot M)V
> $$
> 其中 $\odot$ 是 Hadamard 积，$\exp$ 是逐分量取指数。

从计算复杂度的角度来看 Attention，计算量主要来自中间的两个矩阵计算
$$
\begin{align}
QK^{\top}\in \mathbb{R}^{n\times n}:\quad & \mathcal{O}(n^{2}d) \\
(\exp(QK^{\top}) \odot M)V\in \mathbb{R}^{n\times d_{v}}: \quad & \mathcal{O}(n^{2}d_{v})
\end{align}
$$
这里的 $QK^{\top}$ 有明确的含义，它保存了所有 Query 到 Key 的相关性。标准 Attention 获得强大的检索能力正是因为它保留了这种细粒度的关系。

由于标准实现的 Softmax Attention 需要将这两个矩阵都算出来，计算两个矩阵乘法，因此时间复杂度和空间复杂度均为 $\mathcal{O}(n^{2})$。[[Flash Attention]] 的提出降低了空间的需求，但是平方的时间复杂度依然无法避免。

> [!note] 关于 GPU 计算特性
> 然而，上面得到的 $O(n^{2})$ 表示的是总的计算量，而并非并行硬件上的关键路径长度。对于 $QK^{\top}$，不同的 $(q_{i},k_{i})$ 内积之间彼此独立，因此 Attention 具有极高的计算并行度。
>
> 在具有固定计算资源的 GPU 上，随着 $n$ 的增长，Softmax Attention 实际运行时间最终仍然服从平方增长。但是大量的独立计算可以被组织为规则的矩阵乘法，并充分利用 Tensor Core，因此其硬件执行小路非常高。
>
> 这也解释了 Linear Attention 不一定会比 Softmax Attention 更快。

从前面的分析可以看出来，Softmax Attention 关于序列长度 $n$ 是平方时间复杂度的。Linear Attention 想要降低这个计算复杂度，将整体时间复杂度降低到序列长度 $n$ 的量级。

## Softmax Attention 的线性复杂度形式

最初的 Linear Attention 的思路只是简单的改变 Attention 的计算顺序。为了描述清晰，这里重复一下每个矩阵的形状：
$$
Q,K\in \mathbb{R}^{n\times d}\quad V\in \mathbb{R}^{n\times d_{v}}
$$
从 Attention 的计算过程中，我们可以看到计算主要是针对 $Q,K,V$ 三个矩阵来进行的。暂时去掉 Softmax：
$$
O=(QK^{\top})V
$$
由于矩阵乘法满足结合律，因此可以修改计算的顺序，这会引发显著的区别：
$$
\begin{align}
O&=(QK^{\top})V \implies QK^{\top}\in \mathbb{R}^{n\times n}\\
&=Q(K^{\top}V) \implies K^{\top}V\in \mathbb{R}^{d\times d_{v}}
\end{align}
$$
于是第二种算法的时间复杂度变化为 $\mathcal{O}(dd_{v}n)$，由于 $d,d_{v}$ 都是相对于 $n$ 而言很小的常数，因此该算法为线性复杂度的。

回到 Softmax Attention 上，标准的 Attention 中实际计算的是
$$
O=\exp(QK^{\top})V \tag{1}
$$
矩阵结合律只能调整矩阵乘法的顺序，无法穿过 Softmax。于是后续的研究方向就变成了找到一个方式将
$\exp(QK^{\top})$ 转换为两个矩阵的乘积。这个问题在机器学习中常利用核方法解决。关于机器学习中使用的核方法，见 [[00_Inbox/机器学习/支持向量机#核函数|核方法]]。

> [!note] Causal 形式
> 我们之前描述的 Softmax Attention 中的核心是
> $$
> O=(\exp(QK^{\top}) \odot M) V \tag{2}
> $$
> 上面的分析忽略了注意力掩码 $M$。按照分量形式展开，如下：
> $$
> o_{t}=\sum\limits_{j=1}^{t}(q_{t}^{\top}k_{j})v_{j}=\sum\limits_{j=1}^{t}v_{j}(k_{j}^{\top}q_{t})=\sum\limits_{j=1}^{t}(v_{j}k_{j}^{\top})q_{t}=\left( \sum\limits_{j=1}^{t}v_{j}k_{j}^{\top} \right) q_{t}
> $$
> 如果我们记 $S_{t}=\sum\limits_{j=1}^{t}v_{j}k_{j}^{\top}$，则可以得到递推形式
> $$
> o_{t}=S_{t}q_{t},\quad S_{t}=S_{t-1}+v_{t}k_{t}^{\top} \tag{3}
> $$
> 于是 casual 形式的 Attention 可以写为一个以 $S_{t}$ 为 State 的线性 RNN，递推每一步的复杂度都为 $\mathcal{O}(dd_{v})$，总的复杂度为 $\mathcal{O}(ndd_{v})$，这和我们之前的分析一致。
>
> *之后本文的讨论都是基于 Causal 的*。

### Kernelized Attention

将 Casual Attention 写为更加一般的归一化核平滑形式：
$$
o_{t}=\dfrac{\sum\limits_{j=1}^{t}\kappa(q_{t},k_{j})v_{j}}{\sum\limits_{j=1}^{t}\kappa(q_{t},k_{j})}
$$
其中 $\kappa(q,k)$ 是 Query 和 Key 之间的相似度函数。标准的 Softmax Attention 对应的相似度函数为
$$
\kappa_{\text{softmax}}(q,k)=\exp \left( \dfrac{q^{T}k}{\sqrt{ d }} \right) 
$$
该函数虽然在数学一样上是一个正定核，但是其对应的映射是无限维度的，通常不可直接计算。因此 Linear Attention 要么更换有限维度 Kernel，要么使用随机特征近似指数核。

#### 选择有限维特征映射

现在假设存在有限维特征映射
$$
\phi \in \mathbb{R}^{r\times d}:\mathbb{R}^{d}\to \mathbb{R}^{r}
$$
使得
$$
\kappa(q,k)=\braket{ \phi(q), \phi(k) } =\phi(q)^{\top}\phi(k)=\phi(k)^{\top}\phi(q)
$$
于是可以对 Causal Attention 进行一些变换
$$
o_{t}=\dfrac{\sum\limits_{j=0}^{t}\phi(k_{j})^{\top}\phi(q_{t})v_{j}}{\sum\limits_{j=0}^{t}\phi(k_{j})^{\top}\phi(q_{t})}=\dfrac{\left( \sum\limits_{j=0}^{t}v_{j}\phi(k_{j})^{\top} \right)\phi(q_{t}) }{\left( \sum\limits_{j=0}^{t}\phi(k_{j})^{\top} \right)\phi(q_{t}) }
$$
记
$$
\begin{align}
S_{t}&=\sum\limits_{j=0}^{t}v_{j}\phi(k_{j})^{\top}&\in \mathbb{R}^{d\times r} \\
z_{t}&=\sum\limits_{j=0}^{t}\phi(k_{j})^{\top}&\in \mathbb{R}^{1\times r}
\end{align}
$$
于是输出变成了
$$
o_{t}=\dfrac{S_{t}\phi(q_{t})}{z_{t}\phi(q_{t})}
$$
这里分母依旧是归一化系数，用于保持数值稳定。

这里的计算复杂度变为了 $\mathcal{O}(nrd)$，其中 $r,d$ 都是事先确定的常数，Attention 变成了线性复杂度，相应的递推表达式为
$$
\begin{align}
S_{t}&=S_{t-1}+v_{t}\phi(k_{t})^{\top} \\
z_{t}&=z_{t-1}+\phi(k_{t})^{\top} \\
o_{t}&=\dfrac{S_{t}\phi(q_{t})}{z_{t}\phi(q_{t})}
\end{align} \tag{6}
$$

> [!info]
> *2020, Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* 系统提出了这种 Kernelized attention 形式。
>
> 这样对于新 Kernel 的计算是精确的，但是它不再是标准的 Softmax Attention 了。在 *Transformers are RNNs* 原始论文中，采用如下特征映射
> $$
> \phi(x)=\operatorname{ELU}(x)+1
> $$
>
> 从设计思路上看，为了模仿 Softmax Attention 的特点，需要加入分母来归一化，而为了归一化，采用了非负的核函数与映射。为了避免负输入区域直接产生零梯度，采用了平滑有梯度的 ELU^[$\operatorname{ELU}(x)=\begin{cases} x, & x>0 \\ e^{x}-1, & x \leqslant 0 \end{cases}$] 激活函数。

#### 近似 softmax

一部分研究坚持近似标准的 softmax，例如 *Performer, RFA* 等工作就属于这条路线。这条路线希望
$$
\phi(q)^{\top}\phi(k)\approx \exp(q^{\top}k)
$$
这里简单介绍一下 *Performer* 的思路，由于 softmax 所使用的指数函数被理解为无限维的，因此无法构造出完整的特征映射，因此 Performer 构造了一个有限维随机特征映射 $\phi(x)\in \mathbb{R}^{r\times 1}$ 使得 $\phi(q)^{\top}\phi(k)$ 成为 $\exp(q^{\top}k)$ 的随机近似。

具体的方法类似蒙特卡洛法，只随机采样一组有限特征。随着采样次数 $r$ 的增大，近似会更加准确，但是复杂度也相应上升。

> [!note] 为什么 softmax 被看作无限维度的 Linear Attention
> Softmax Attention 计算的指数注意力分数 $\exp(q^{\top}k)$ 根据泰勒级数展开，可以得到
> $$
> \exp(q^{\top}k)=\sum\limits_{n=0}^{\infty}\dfrac{(q^{\top}k)^{n}}{n!}=\sum\limits_{n=0}^{\infty}\dfrac{\braket{ q^{\otimes n}, k^{\otimes n}} }{n!}
> $$
> 于是可以构造一个无限维特征映射
> $$
> \Phi(q)=\left[ 1,q,\dfrac{q^{\otimes 2}}{\sqrt{ 2! }},\dfrac{q^{\otimes 3}}{\sqrt{ 3! }},\cdots \right] 
> $$
> 使得
> $$
> \exp(q^{\top}k)=\Phi(q)^{\top}\Phi(k)
> $$

## 重新理解 Linear Attention

从式 $(3)$ 和 $(6)$ 可以看出来，Linear Attention 的内涵似乎不止线性复杂度、改变计算顺序这么简单。到了因果场景，它变成了一个状态更新公式。这意味着历史信息不再以 token 列表而存在，而是不断被压缩进一个矩阵状态中。

> [!note] 为什么早期 Linear Attention 没有替代 Softmax?
> 1. **表达能力不足**：Softmax 具有很强的选择性，会在同一行内进行归一化斗争，一个位置的权重上升，其他位置的相对权重都会下降。指数归一化的性质又导致它非常容易产生尖锐、低熵的注意力分布。而简单的正值内积 Kernel 则无法产生尖锐的分布，输出容易变成许多 value 的平滑混合。
> 	- 之后的研究，例如 *cosFormer*、*Hedgehog* 等都尝试保留了 Softmax 的非负性、单调性等行为。
> 2. **状态管理缺失**：Linear Attention 将无限维度的 softmax 压缩到固定的 $r$ 状态维度，随着历史的增长，不同的信息会共享有限的状态维度。模型虽然仍然可以学习有效的压缩策略，但是很难保证所有细节都独立可寻址。
> 3. **并行度**：Linear Attention 的计算存在严格的时间依赖，GPU 更加擅长大规模规则矩阵乘法，而不擅长大量串行小矩阵更新。早期的 Linear Attention 实现缺乏并行度，理论上更低的复杂度并没有带来更高的实际速度。

基于这个认知，后续研究 Fast Weight、Delta Rule、gating 和 state-model 等方法，针对 Linear Attention 的表达能力、状态管理、硬件并行方面进行了研究。

为什么说**重新理解 Linear Attention**？现代的 Linear Attention 研究基本围绕一个统一的状态更新：
$$
\begin{align}
S_{t}&=\mathcal{T}_{t}(S_{t-1})+B_{t}  \\
o_{t}&=S_{t}q_{t}
\end{align}
$$
其中 $S_{t}$ 是当前矩阵状态，$\mathcal{T}_{t}$ 决定旧状态如何被保留、衰减或修改，$B_{t}$ 表示当前 token 写入的新内容。最后的输出由 Query 读取。从形态上看，Linear Attention 几乎已经偏离了原始 Attention 的形态。

对于很多的现代方法，状态转移可以进一步写成
$$
S_{t}=S_{t-1}A_{t}+B_{t}\tag{7}
$$

其中 $A_{t}\in \mathbb{R}^{d\times d}$，也就是利用一个线性映射来处理状态转移。

> [!note] 一个理解 $(7)$ 的角度
> $A_{t}$ 决定了旧信息到达未来时还剩多少，位于什么方向，是否被覆盖。我们将式 $(7)$ 展开，可以得到
> $$
> \begin{align}
> A_{[j:t]}&=A_{i}A_{j+1}\cdots A_{t} \\
> S_{t}&=S_{0}A_{[1:t]} + \sum\limits_{j=1}^{t}B_{i}A_{[j+1:t]}
> \end{align}
> $$
> 对于 Linear Attention，$B_{t}$ 一般取 $\eta_{t}v_{t}k_{t}^{\top}$，我们假设初始状态 $S_{0}=0$，那么
> $$
> o_{t}=\sum\limits_{j=1}^{t}\eta_{j}v_{j}\left( k_{j}^{\top}A_{[j+1:t]}q_{t} \right) 
> $$
> 根据 Softmax Attention 中对于注意力分数的定义，可以发现 $k_{j}^{\top}A_{[j+1:t]}q_{t}$ 就表示了 token $j$ 在时刻 $t$ 的有效注意力分数，对于 Softmax Attention 而言，$A_{t}=I$，所以注意力对于历史 key 而言是永远不变的。如果考虑 $A_{t}$ 的影响，那么可以进行如下变换：
> $$
> k_{j}^{\top}A_{[j+1:t]}q_{t}=\left( A_{[j+1:t]}^{\top}k_{j} \right)^{\top}q_{t} 
> $$
> 也就是第 $j$ 个 token 的 key 随时间逐渐变成了
> $$
> \tilde{k}_{j\to t}=A_{[j+1:t]}^{\top}k_{j}
> $$
> 所以 $A_{t}$ 实际上是在改变**历史信息如何被寻址**的问题。

### 遗忘门

由于 Softmax Attention 

### Softmax Attention

 让我们在回到 Softmax Attention 的视角来看这个式子。Softmax Attention 是该式子的一个特例，根据前面的推导，我们知道 $S_{t} \in \mathbb{R}^{r\times d}$，其中 $r$ 是特征映射的目标维度。对于 softmax 而言，使用了一个无限维的特征映射 $\exp$，即
 $$
 \exp(x^{\top}y)=\Phi(x)\Phi(y)
 $$
 于是可以得到 $S_{t}=\sum\limits_{j=0}^{t}v_{j}\phi(k_{j})^{\top}\in \mathbb{R}^{d\times \infty}$。

 一个非常有意思的视角是，这个无限维状态还有另一种表示方式，虽然
 $$
 S_{t}=\sum\limits_{j=1}^{t}v_{j}\Phi(k_{j})^{\top}
 $$
 处于无限维空间中，但是读取它的时候有
 $$
 S_{t}\Phi(q)=\sum\limits_{j=1}^{t}v_{j}
 \underbrace{\Phi(k_{j})^{\top}\Phi(q)}_{e^{k_{j}^{\top}q}}
 $$
 这实际上是非常漂亮的一个对偶表示
 $$
 \begin{array}{ccc}
 \text{Feature-space memory} & \iff & \text{Sample-space memory} \\
 S_{t}=\sum_{j}v_{j}\Phi(k_{j})^{T} && \left\{ (k_{j},v_{j}) \right\} _{j=1}^{t}
 \end{array}
 $$
 当 feature dimension 非常小的时候，使用前者非常划算。当 feature dimension 是无限的时候，反而只有后者是可行的，可以通过存储所有可用 token 来避免进入无限维空间。

 关于 Softmax Attention 的表达能力有一点非常有意思，对于 $\exp$ 核而言，它从一开始就是无限维度的，并没有随着输入 token 的增长而变多。真正随输入增长的，是输入表示形成的子空间维数，也就是
 $$
 \operatorname{span}\left\{ \Phi(k_{1}),\Phi(k_{2}),\cdots,\Phi(k_{t}) \right\} 
 $$
 从这个角度看，如果一个 token 带来了一个新的独立方向，那么就会导致输入的子空间维数增长，信息容量也就增加。这就是为什么 Softmax Attention 的模型参数固定，但是却能处理无限长度的输入。

> [!todo]
> 从这个视角看，我们真的需要无限维度的表示能力吗？对于一个长度为 $t$ 的 token 序列，真正被使用的子空间维度最多只有 $t$ 维。这给出了一个新的研究方向，即无需固定 $r$ 也无需永久保留全部的 $t$ 个 token，而是维护一个自适应增长的基底：
> $$
> U_{t}=\left\{ u_{1},u_{2},\cdots,u_{m_{t}} \right\} 
> $$
> 其中基底的维数由数据决定。当新的 token 到来时，尝试在当前表示空间中进行描述。如果已经基本冗余，则增加基底维数。
