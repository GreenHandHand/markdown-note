---
aliases:
  - 增强
tags:
  - 机器学习
  - 监督学习
  - 聚合
---

# Boosting

Boosting 是一种常用的统计学习方法，应用广泛且有效。在分类问题中，它通过改变样本权重，学习多个分类器，并将这些分类器进行线性组合，提高分类的性能。

Boosting 基于这样一种思路：对于一个复杂任务而言，将多个专家的判断进行适当的综合所得出的判断，要比其中任何一个专家单独的判断好。

历史上, Kearns 和 Valiant 首先提出了*强可学习* (strongly learnable) 和*弱可学习* (weakly learnable) 的概念。指出，在概率近似正确 (probably approximately correct, PAC) 学习的框架中：
- 一个概念，如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么称这个概念是强可学习的。
- 一个概念，如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么这个概念是弱可学习的。

后来，Schapire 证明强可学习的算法与弱可学习的算法是等价的。也就是说，在 PAC 学习的框架下，一个概念强可学习的充分必要条件是这个概念弱可学习。

于是，人们提出了这样的一个问题：在学习中，如果发现了弱可学习算法，那么是否能将它提升 (boost) 为强可学习算法？

弱可学习算法通常比强可学习算法要容易发现。于是如果进行提升，就成为了开发 Boosting 时要解决的问题。关于 Boosting 的研究很很多，其中最具代表性的是 AdaBoost 算法。

## AdaBoost 算法

假定给定一个二分类的训练数据集：
$$
T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$
其中，每个样本点由实例与标记组成。实例 $x_i\in\mathcal X\subseteq \mathbf R^n$，标记 $y_i\in \mathcal Y=\{-1,+1\}$，$\mathcal X$ 是实例空间，$\mathcal Y$ 是标记空间。AdaBoost 利用以下算法，从训练数据中学习一系列弱分类器或者基本分类器，并将这些弱分类器线性组合称为一个强分类器。

> AdaBoost 算法
- 输入：训练数据集 $T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$, 其中 $x_i\in \mathcal X\subseteq R^n$, $y_i\in \mathcal Y=\{-1,+1\}$}。弱学习算法。
- 输出：最终分类器 $G(x)$
1. 初始化训练数据的权重分布
$$
	D_1=(w_{11}, \cdots, w_{1i}, \cdots, w_{1N}),\quad w_{1i}=\frac{1}{N},\quad i=1,2,\cdots,N
	$$
2. 对 $m=1,2,\cdots, M$
	1. 使用具有权重分布 $D_m$ 的训练数据集学习，得到基本分类器
	$$
	G_m(x):\mathcal X\to \{-1,+1\}
	$$
	2. 计算 $G_m(x)$ 在训练数据集上的分类误差率
	$$
	e_m=\sum_{i=1}^{N} P(G_m(x_i)\ne y_i) = \sum_{i=1}^N w_{mi} I(G_m(x_i) \ne y_i)\tag 1
	$$
	3. 计算 $G_m(x)$ 的系数
	$$
	\alpha_m=\frac{1}{2}\ln \frac{1-e_m}{e_m}\tag 2
	$$
	4. 更新训练数据集的权值分布
	$$
	\begin{aligned}
	D_{m+1} &= (w_{m+1, 1},\cdots, w_{m+1, i}, \cdots, w_{m+1, N})\\
	w_{m+1, i}&=\frac{w_{mi}}{Z_m}\exp(-\alpha_m y_i G_m(x_i)),\quad i=1,2,\cdots, N
	\end{aligned}\tag 3
	$$
	这里 $Z_m$ 是规范化因子
	$$
	Z_m=\sum_{i=1}^N w_{mi}\exp(-\alpha_m y_i G_m(x_i))\tag 4
	$$
	它使 $D_{m+1}$ 成为一个概率分布。
3. 构建基本分类器的线性组合
$$
f(x)=\sum_{m=1}^M\alpha_m G_m(x)\tag 5
$$
得到最终分类器
$$
\begin{aligned}
G(x)&=\textbf{sign}(f(x))\\
&=\textbf{sign} \left(\sum_{m=1}^M \alpha_m G_m(x)\right)
\end{aligned}\tag 6
$$

对于 AdaBoost 算法作如下说明：
1. 步骤一：假设训练数据集具有均匀的权值分布，即每个训练样本在基本分类器的学习中作用相同。这一假设保证了第一步能够在原始数据上学习基本分类器 $G_1(x)$
2. 步骤二：AdaBoost 反复学习基本分类器，在每个轮次 $m=1,2,\cdots, M$ 中顺序执行下列操作：
	1. 使用当前分布 $D_m$ 加权的训练数据集，学习基本分类器 $G_m(x)$
	2. 计算基本分类器 $G_m(x)$ 在加权训练数据集上的分类误差率：
	$$
	\begin{aligned}
	e_m&=\sum_{i=1}^N P(G_m(x_i)\ne y_i)\\
	&=\sum_{G_m(x_i)\ne y_i}w_{mi}
	\end{aligned}
	$$
	这里，$w_{mi}$ 代表第 $m$ 轮中第 $i$ 个实例的权值，$\sum_{i=1}^Nw_{mi}=1$，这表明 $G_m(x)$ 在加权训练数据集上的分类误差率是被 $G_m(x)$ 误分类样本的权值之和。由此，可以看出数据权值分布 $D_m$ 与基本分类器 $G_m(x)$ 的分类误差率的关系。
	3. 计算基本分类器 $G_m(x)$ 的系数 $\alpha_m$。$\alpha_m$ 表示 $G_m(x)$ 在最终分类器中的重要性。当 $e_m\leqslant \frac{1}{2}$ 时，$\alpha_m\geqslant 0$，并且 $\alpha_m$ 随着 $e_m$ 的减小而增大，所以分类误差率越小的基本分类器在最终分类器中的作用越大。
	4. 更新训练数据的权重分布，为下一轮作准备。式 $(3)$ 可以写作
	$$
	w_{m+1,i}=\begin{cases}\dfrac{w_{mi}}{Z_m}e^{-\alpha_m},& G_m(x_i)=y_i\\
	\dfrac{w_{mi}}{Z_m}e^{\alpha_m},&G_m(x_i)\ne y_i\end{cases}
	$$
	由此可以知道，被基本分类器 $G_m(x)$ 误分类的样本的权值得以扩大，而被正确分类样本的权值得以缩小。两式比较，由式 $(2)$ 可以得到误分类样本被放大 $e^{2\alpha_m}=\dfrac{1-e_m}{e_m}$ 倍。因此，误分类样本在下一轮学习中起到更大的作用。在不改变所给的训练数据，而不断改变训练数据权值分布，使得训练数据在基本分类器的学习中起不同的作用。
3. 步骤三：线性组合 $f(x)$ 实现 $M$ 个基本分类器的加权表决。系数 $\alpha_m$ 表示了基本分类器 $G_m(x)$ 的重要性。这里，所有 $\alpha_m$ 之和并不为 1，$f(x)$ 的符号决定实例 $x$ 的类，$f(x)$ 的决定值表示分类的确信度。利用基本分类器的线性组合构建最终分类器。
