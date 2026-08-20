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
V&=[v_{1},v_{2},\cdots,v_{n}]^{\top}\in \mathbb{R}^{n \times d} \\
O&=\operatorname{Attention}(Q,K,V)=[o_{1},o_{2},\cdots,o_{n}]\in \mathbb{R}^{n\times d}
\end{align}
$$

> [!tip] 这里全部采用列向量

这里只考虑 Causal 场景，意味着 $o_{t}$ 至多和 $Q_{[:t]},K_{[:t]},V_{[:t]}$ 相关。
