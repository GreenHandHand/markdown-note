---
tags:
  - 并行编程
  - CUDA
---

# CUDA 共享显存

在向量加法的例子中，结果中的每个值都只取决于输入的相应位置的值。但是现实的例子并不总是如此简单。

本节通过一维模板 (1D stencil) 的例子来引入 CUDA 中的共享显存的概念。

> [!info] 一维模板
> 模板计算时高性能计算中重要的一个内容。其含义为：整体的计算问题可以划分为局部模板的计算。在模板计算中，一个值的计算与其相邻位置的其他元素有关。模板计算应用广泛，常见的例子包括热传导方程、图卷积运算、图像滤波等。
>
> 本节中引入的模板计算定义如下：
> $$
> \begin{align}
>	\mathbf{stencil}(x_{i}) &= \begin{cases}
>	\displaystyle\sum\limits_{k=-r}^{r}x_{i+k} & r \leqslant i \leqslant n - r \\[0.3em]
>	\displaystyle\sum\limits_{k=-i}^{k=r}x_{i+k} & i < r \\[0.3em]
>	\displaystyle\sum\limits_{k=-r}^{n-i}x_{i+k} & i > n - r
>	\end{cases} \\[0.5em]
> \mathbf{stencil}(X) &= [\mathbf{stencil}(x_{1}),\mathbf{stencil}(x_{2}),\cdots,\mathbf{stencil}(x_{n})]
> \end{align}
> $$
> 即使用一个滑动窗口，计算窗口内的值并求和。该操作等价与 `padding` 为 0 的一维卷积操作，其中 `r` 为超参数，表示大小为 `2*r+1` 的窗口。

在这个例子中，每个线程中的输出值不仅与自己编号相同的输入元素有关，还和编号距离半径为`r`的输出元素有关。因此，
