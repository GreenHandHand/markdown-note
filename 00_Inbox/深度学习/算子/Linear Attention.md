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
> Softmax Attention 用分量的形式^[这里实际包含了注意力掩码，也就是求和只到当前时间 $t$] 写出来则是
> $$
> o_{t}=\dfrac{\sum\limits_{j=1}^{t}\exp(q_{t}^{\top}k_{j})v_{j}}{\sum\limits_{j=1}^{t}\exp(q_{t}^{\top}k_{j})}
> $$
>
> 其中分母的作用主要是保持数值稳定性，另外就是如果我们给 $O$ 加上 RMSNorm^[关于这里自动消去的原因，RMSNorm 具有尺度不变性，而 softmax 计算得到的分母是对于分量来说是常数，因此在考虑 Softmax 后立即跟 RMSNorm 的情况，Softmax 的分母在数学上是可以消去的]，那么分母也会自动消去，所以 Softmax Attention 的核心是分子部分（即 Attention 中真正负责信息混合的核心算子），即
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

## Linear Attention 的形式化

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
O=\exp(QK^{\top})V
$$
矩阵结合律只能调整矩阵乘法的顺序，无法穿过 Softmax。于是后续的研究方向就变成了找到一个方式将
$\exp(QK^{\top})$ 转换为两个矩阵的乘积。这个问题在机器学习中常利用核方法解决。关于机器学习中使用的核方法，见 [[00_Inbox/机器学习/支持向量机#核函数|核方法]]。

> [!note] Causal 形式
> 我们之前描述的 Softmax Attention 中的核心是
> $$
> O=(\exp(QK^{\top}) \odot M) V
> $$
> 上面的分析忽略了注意力掩码 $M$。按照分量形式展开，如下：
> $$
> o_{t}=\sum\limits_{j=1}^{t}(q_{t}^{\top}k_{j})v_{j}=\sum\limits_{j=1}^{t}v_{j}(k_{j}^{\top}q_{t})=\sum\limits_{j=1}^{t}(v_{j}k_{j}^{\top})q_{t}=q_{t}\sum\limits_{j=1}^{t}v_{j}k_{j}^{\top}
> $$
> 如果我们记 $S_{t}=\sum\limits_{j=1}^{t}v_{j}k_{j}^{\top}$，则可以得到递推形式
> $$
> o_{t}=S_{t}q_{t},\quad S_{t}=S_{t-1}+v_{t}k_{t}^{\top}
> $$
> 于是 casual 形式的 Attention 可以写为一个以 $S_{t}$ 为 State 的线性 RNN，递推每一步的复杂度都为 $\mathcal{O}(dd_{v})$，总的复杂度为 $\mathcal{O}(ndd_{v})$，这和我们之前的分析一致。

### Kernelized Attention

将 Casual Attention 写为更加一般的归一化核平滑形式：
$$
o_{t}=\dfrac{\sum\limits_{j=1}^{t}\kappa(q_{t},k_{j})v_{j}}{\sum\limits_{j=1}^{t}\kappa(q_{t},k_{j})}
$$
其中 $\kappa(q,k)$ 是 Query 和 Key 之间的相似度函数。标准的 Softmax Attention 对应的相似度函数为
$$
\kappa_{\text{softmax}}(q,k)=\exp \left( \dfrac{q^{T}k}{\sqrt{ d }} \right) 
$$
该函数不是数学定义上的核函数，它不能转换为 $\braket{ \phi(q), \phi(k) }$ 的形式，因此需要找到一个替代品。

现在假设存在有限维特征映射
$$
\phi:\mathbb{R}^{d}\to \mathbb{R}^{r}
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
S&=\sum\limits_{j=0}^{t}v_{j}\phi(k_{j})^{\top}&\in \mathbb{R}^{d\times r} \\
z&=\sum\limits_{j=0}^{t}\phi(k_{j})&\in \mathbb{R}^{d\times 1}
\end{align}
$$
于是输出变成了
$$
o_{t}=\dfrac{S\phi(q_{t})}{z\phi(q_{t})}
$$
