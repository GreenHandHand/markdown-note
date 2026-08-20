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
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}(\dfrac{QK^{\top}}{\sqrt{ d }})V
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
> 一般只考虑 Causal 场景，意味着 $o_{t}$ 至多和 $Q_{[:t]},K_{[:t]},V_{[:t]}$ 相关。在这种情况下，Attention 公式可以写作 $O=\operatorname{softmax}(QK^{\top}/\sqrt{ d }+\log M)V$。其中注意力掩码矩阵 $\log M\in \mathbb{R}^{n\times n}$ 是一个下三角矩阵，对角线之下的元素使用 $-\infty$ 来屏蔽。

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
> Softmax Attention 用分量的形式写出来则是
> $$
> o_{t}=\dfrac{\sum\limits_{j=1}^{t}\exp(q_{t}^{\top}k_{j})v_{j}}{\sum\limits_{j=1}^{t}\exp(q_{t}^{\top}k_{j})}
> $$
>
> 其中分母的作用主要是保持数值稳定性，另外就是如果我们给 $O$ 加上 RMSNorm，那么分母也会自动消去，所以 Softmax Attention 的核心是分子部分，即
> $$
> O=\exp(QK^{\top}+\log M)V=(\exp(QK^{\top}) \odot M)V
> $$
> 其中 $\odot$ 是 Hadamard 积，$\exp$ 是逐分量取指数。

从计算复杂度的角度来看 Attention 部分，计算量主要来自中间的矩阵计算
$$
\begin{align}
QK^{\top}\in \mathbb{R}^{n\times n}:\quad & \mathcal{O}(n^{2}d) \\
\operatorname{softmax}(QK^{\top})V\in \mathbb{R}^{n\times d_{v}}: \quad & \mathcal{O}(n^{2}d_{v})
\end{align}
$$
这里的 $QK^{\top}$ 有明确的含义，即注意力分数。它保存了所有 Query 到 Key 的相关性。标准 Attention 获得强大的检索能力正是因为它保留了这种细粒度的关系。
