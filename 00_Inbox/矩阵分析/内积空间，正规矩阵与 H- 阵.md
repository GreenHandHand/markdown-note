---
tags:
  - 矩阵分析
---

# 内积空间，正规矩阵与 H- 阵

## 内积空间

> [!definition|Definition] 实数域上的内积
> 设 $V$ 是**实数域** $R$ 上的 $n$ 维线性空间，对于 $V$ 中的任意两个向量 $\alpha, \beta$，按照某一确定法则对应着一个实数，这个实数称为 $\alpha$ 与 $\beta$ 的内积，记为 $(\alpha, \beta)$，并且要求内积满足下列运算条件：
> 1. $(\alpha, \beta)=(\beta, \alpha)$
> 2. $(k\alpha,\beta)=k(\alpha,\beta)$
> 3. $(\alpha+\beta, \gamma)=(\alpha, \gamma)+(\beta,\gamma)$
> 4. $(\alpha,\alpha)\geqslant 0$
> 
> 这里 $\alpha,\beta,\gamma$ 是 $V$ 中任意向量，$k$ 为任意实数。

> [!definition|欧式空间] 只有当 $\alpha=0$ 时 $(\alpha,\alpha)=0$，我们称带有这样内积的 $n$ 维线性空间 $V$ 为 **欧式空间**。

> [!definition|Definition] 复数域上的内积
> 设 $V$ 是**复数域** $C$ 上的 $n$ 维线性空间，对于 $V$ 中的任意两个向量 $\alpha, \beta$，按照某一确定法则对应着一个复数，这个复数称为 $\alpha$ 与 $\beta$ 的内积，记为 $(\alpha, \beta)$，并且要求内积满足下列运算条件：
> 1. $(\alpha, \beta)=\overline{(\beta, \alpha)}$
> 2. $(k\alpha,\beta)=k(\alpha,\beta)$
> 3. $(\alpha+\beta, \gamma)=(\alpha, \gamma)+(\beta,\gamma)$
> 4. $(\alpha,\alpha)\geqslant 0$
> 
> 这里 $\alpha,\beta,\gamma$ 是 $V$ 中任意向量，$k$ 为任意复数。

> [!definition|酉空间] 只有当 $\alpha=0$ 时 $(\alpha,\alpha)=0$，我们称带有这样内积的 $n$ 维线性空间 $V$ 为**酉空间**。

> [!definition|线性空间] 欧式空间与酉空间通称为**内积空间**。

> [!note] 基本性质
> 1. $(\alpha,k\beta)=\overline{k}(\alpha,\beta)$
> 2. $(\alpha,\beta+\gamma)=(\alpha,\beta)+(\alpha,\gamma)$
> 3. $\left( \sum\limits_{i=}^{t}k_{i}\alpha_{i},\beta \right)=\sum\limits_{i=1}^{t}k_{i}(\alpha_{i},\beta)$
> 4. $\left( \alpha,\sum\limits_{i=1}^{t}k_{i}\beta_{i} \right)=\sum\limits_{i=1}^{t}\overline{k}_{i}(\alpha_{i},\beta_{i})$

> [!definition|Definition] 度量矩阵
> 设 $V$ 是 $n$ 维酉空间，$\left\{ \alpha_{i} \right\}$ 为其一组基底，令 $g_{ij}=(\alpha_{i}, \alpha_{j})$，则
> $$
> G=\begin{bmatrix}
> g_{11} & g_{12} & \cdots & g_{1n} \\
> g_{21} & g_{22} & \cdots & g_{2n} \\
> \vdots & \vdots & \ddots & \vdots \\
> g_{n1} & g_{n2} & \cdots & g_{nn}
> \end{bmatrix}
> $$
> 称为基底 $\left\{ \alpha_{i} \right\}$ 的**度量矩阵**，且满足
> $$
> g_{ji}=\overline{g_{ij}},\quad (\overline{G})^{T}=G,\quad (\alpha,\beta)=X^{T}G\overline{Y}
> $$

> [!definition|Definition] 复共轭转置矩阵
> 设 $A\in C^{n\times n}$，用 $\overline{A}$ 表示以 $A$ 的元素的共轭复数为元素组成的矩阵，记为 $A^{H}=(\overline{A})^{T}$

> [!note] 复共轭转置矩阵的性质
> 1. $(A+B)^{H}=A^{H}+B^{H}$
> 2. $(kA)^{H}=\overline{k}A^{H}$
> 3. $(AB)^{H}=B^{H}A^{H}$
> 4. $(A^{k})^{H}=(A^{H})^{k}$
> 5. $(A^{H})^{H}=A$
> 6. $\left| \overline{A} \right|=\overline{\left| A \right|}$
> 7. $(A^{H})^{-1}=(A^{-1})^{H}$

> [!definition|Definition] Hermite 矩阵与反 Hermite 矩阵
> 设 $A \in C^{n\times n}$，则：
> - $A^{H}=A\iff A$ 为 Hermite 矩阵
> - $A^{H}=-A\iff A$ 为反 Hermite 矩阵

> [!tip] Hermite 矩阵即复数域概念上的对称矩阵，反 Hermite 矩阵即复数域概念上的反对称矩阵

> [!definition|Definition] 内积空间的度量
> 设 $V$ 为酉 (欧式) 空间，向量 $\alpha \in V$ 的长度定义为非负实数
> $$
> \lVert \alpha \rVert =\sqrt{ (\alpha,\alpha) }
> $$
> 一般的，对于 $C^{n\times n}$ 中的向量，其长度可以计算为
> $$
> \lVert \alpha \rVert =\sqrt{ \sum\limits_{i=1}^{n}\lvert a_{i} \rvert ^{2} }
> $$
> 其中 $\lvert a_{i} \rvert$ 表示复数 $a_{i}$ 的模长。

> [!note] 向量长度的性质
> 1. $\lVert \alpha \rVert \geqslant 0$ 当且仅当 $\alpha=0$ 时 $\lVert \alpha \rVert=0$
> 2. $\lVert k\alpha \rVert=\lvert k \rvert\lVert \alpha \rVert,k\in C$
> 3. $\lVert \alpha+\beta \rVert\leqslant\lVert \alpha \rVert+\lVert \beta \rVert$，即三角不等式
> 4. $\lvert (\alpha,\beta) \rvert\leqslant\lVert \alpha \rVert\lVert \beta \rVert$

> [!definition|Definition] 夹角
> 设 $V$ 为欧式空间，两个非零向量 $\alpha,\beta$ 的夹角定义为
> $$
> \langle \alpha,\beta \rangle =\arccos\dfrac{\lvert (\alpha,\beta) \rvert }{\lVert \alpha \rVert \lVert \beta \rVert }
> $$

> [!note] 一些定义
> 1. **正交**：在酉空间 $V$ 中，如果 $(\alpha,\beta)=0$，则称 $\alpha$ 与 $\beta$ 正交。
> 2. **单位化**：长度为 1 的向量称为单位向量，对于任何一个非零的向量 $\alpha$，向量 $\alpha/\lVert \alpha \rVert$ 总是单位向量，称此过程为单位化。

> [!definition|Definition] 正交向量组
> 设 $\left\{ \alpha_{i} \right\}$ 为一组不含有零向量的向量组，如果 $\left\{ \alpha_{i} \right\}$ 内的任意两个向量彼此正交，则称其为**正交的向量组**。如果一个正交向量组中任何一个向量都是单位向量，则称次向量组为**标准的正交向量组**。

> [!note] 正交基底与标准正交基底
> 在 $n$ 为内积空间中，由 $n$ 个正交向量组成的基底称为**正交基底**，由 $n$ 个标准的正交向量组成的基底称为**标准正交基底**。
>
> *显然，标准正交基底不唯一*

> [!note]
> 1. 如果向量组中的每一对向量都满足 $(\alpha,\beta)=0$，则该向量组为正交向量组。
> 2. 如果向量组中的每一对向量都满足 $(\alpha_{i},\alpha_{j})=\delta_{ij}=\begin{cases}1& i=j \\ 0 & i\neq j\end{cases}$，则该向量组为标准正交向量组。

> [!definition|定理] 正交的向量组是一个线性无关的向量组。反之，由一个线性无关的向量组出发可以构造一个正交向量组，甚至一个标准正交向量组。

![[02_Areas/线性代数/特征值与特征向量#^b2e9da|特征值与特征向量]]

## 酉矩阵与正交矩阵

> [!definition|Definition] 酉矩阵
> 设 $A$ 是一个 $n$ 阶复矩阵，如果其满足
> $$
> A^{H}A=AA^{H}=I
> $$
> 则称 $A$ 是**酉矩阵**，一般记为 $A\in U^{n\times n}$。

> [!note] 正交矩阵
> 正交矩阵的定义为 $A^{T}A=AA^{T}=I$，记为 $A\in E^{n\times n}$。更多正交矩阵的性质参照 [[02_Areas/线性代数/向量组#正交矩阵|向量组]]。

> [!definition|Householder 矩阵] 设 $\alpha \in C^{n\times 1}$ 且 $\alpha^{H}\alpha=1$，如果 $A=I-2\alpha\alpha^{H}$，则 $A$ 是一个**酉矩阵**，通常称为 **Householder 矩阵**。

> [!note] 酉矩阵与正交矩阵的性质
> 设 $A,B\in U^{n\times n}$，那么
> 1. $A^{-1}=A^{H}\in U^{n\times n}$
> 2. $\lvert \det(A) \rvert=1$
> 3. $AB,BA\in U^{n\times n}$
> 
> 对正交矩阵同理，设 $A,B\in E^{n\times n}$，那么
> 1. $A^{-1}=A^{T}\in E^{n\times n}$
> 2. $\det(A) =\pm1$
> 3. $AB,BA\in E^{n\times n}$

> [!definition|定理] 设 $A\in C^{n\times n}$，$A$ 是酉矩阵的充分必要条件是 $A$ 的 $n$ 个列（或行）向量组是标准正交向量组。

> [!definition|Definition] 酉变换
> 设 $V$ 是一个 $n$ 为酉空间，$\sigma$ 是 $V$ 的一个线性变换，如果对任意的 $\alpha,\beta \in V$ 都有
> $$
> (\sigma(\alpha),\sigma(\beta))=(\alpha,\beta)
> $$
> 则称 $\sigma$ 是 $V$ 的一个**酉变换**。

> [!note]
> 设 $V$ 是一个 $n$ 维酉空间，$\sigma$ 是 $V$ 的一个线性变换，那么下列陈述等价：
> 1. $\sigma$ 是酉变换
> 2. $\lVert \sigma(\alpha) \rVert=\lVert \alpha \rVert,\forall \alpha \in V$
> 3. 将 $V$ 的标准正交基底变成标准正交基底
> 4. 酉变换在标准正交基下的矩阵表示为酉矩阵

> [!tip] 正交矩阵具有同样的性质。

## 幂等矩阵

> [!definition|Definition] 幂等矩阵
> 设 $A\in C^{n\times n}$，如果 $A$ 满足
> $$
> A^{2}=A
> $$
> 则称 $A$ 是一个**幂等矩阵**。

> [!note] 幂等矩阵的性质
> 设 $A$ 是幂等矩阵，那么有
> 1. $A^{T},A^{H},I-A,I-A^{T},I-A^{H}$ 都是幂等矩阵。
> 2. $A(I-A)=(I-A)A=0$
> 3. $N(A)=R(I-A)$
> 4. $Ax=x$ 的充分必要条件是 $x \in R(A)$
> 5. $C^{n\times 1}=R(A)\oplus N(A)$

> [!note] 充分必要条件
> 设 $A$ 是一个秩为 $r$ 的 $n$ 阶矩阵，那么 $A$ 是幂等矩阵的充分必要条件是存在 $P\in C_{n}^{n\times n}$ 使得
> $$
> P^{-1}AP=\begin{bmatrix}
> I_{r} & O \\ O & O
> \end{bmatrix}
> $$
>
> **推论**(必要条件)：设 $A$ 是一个幂等矩阵，则有 $Tr(A)=Rank(A)$。

> [!definition|Definition] 次酉矩阵
> 设 $\left\{ \alpha_{1},\alpha_{2}\cdots,\alpha_{r} \right\}$ 为一个 $n$ 维标准正交列向量组，那么称 $n\times r$ 型矩阵
> $$
> U_{1}=\left[ \alpha_{1},\alpha_{2},\cdots,\alpha_{r} \right] 
> $$
> 为一个次酉矩阵。一般记为 $U_{1} \in U_{r}^{n\times r}$

> [!tip] 酉矩阵是复数域下的正交矩阵，即酉矩阵是方阵。而次酉矩阵则是由复数域下的正交向量组构成的矩阵。

> [!note] 次酉矩阵与幂等矩阵的关系
> 设 $A$ 为一个 $n$ 阶矩阵，则 $A=A^{H}=A^{2}$ 的充分必要添加是存在一个 $n\times r$ 的次酉矩阵 $U\in U_{r}^{n\times r}$ 使得
> $$
> A=UU^{H}
> $$
> 其中 $r=Rank(A)$

## 对称变换与反对称变换

对称变换与反对称变换是欧式空间上的特殊线性变换。
- 对称变换在任意一个标准正交基下的矩阵表示为对称表示，满足 $(f(\alpha),\beta)=(\alpha,f(\beta))$。
- 反对称变换在任意一个标准正交基下的矩阵表示为反对称矩阵，满足 $(f(\alpha),\beta)=-(\alpha,f(\beta))$。

> [!definition|Definition] 酉相似
> 设 $A,B\in C^{n\times n}$，若存在 $U\in U^{n\times n}$ 使得
> $$
> U^{H}AU^{-1}=U^{-1}AU=B
> $$
> 则称 $A$ 酉相似于 $B$。

> [!definition|Definition] 正交相似
> 设 $A,B\in E^{n\times n}$，若存在 $U\in E^{n\times n}$ 使得
> $$
> U^{T}AU^{-1}=U^{-1}AU=B
> $$
> 则称 $A$ 正交相似于 $B$。

> [!tip] 对于实数域上的相似矩阵的定义，参见 [[02_Areas/线性代数/特征值与特征向量#矩阵相似|相似矩阵]]。

> [!definition|Schur 引理] 任何一个 $n$ 阶复矩阵 $A$ 酉相似于一个上 (下) 三角矩阵。

> [!example] 求解酉相似矩阵的方法

## 正规矩阵

> [!definition|Definition] 正规矩阵
> 设 $A\in C^{n\times n}$，如果 $A$ 满足
> $$
> AA^{H}=A^{H}A
> $$
> 那么称矩阵 $A$ 为一个**正规矩阵**。

> [!definition|Definition] 实正规矩阵
> 设 $A\in R^{n\times n}$，如果 $A$ 满足
> $$
> AA^{T}=A^{T}A
> $$
> 那么称矩阵 $A$ 为一个**实正规矩阵**。

> [!tip] Hermite 矩阵、反 H- 矩阵，正交矩阵，酉矩阵，对角矩阵都是正规矩阵。

> [!note] 定理
> 引理：
> 1. 设 $A$ 是一个正规矩阵，则与 $A$ 酉相似的矩阵一定是正规矩阵。
> 2. 设 $A$ 是一个正规矩阵，且又是三角矩阵，则 $A$ 必为对角矩阵。
> 
> 设 $A\in C^{n\times n}$，则 $A$ 是正规矩阵的充分必要条件是存在一个酉矩阵 $U$ 使得
> $$
> U^{H}AU=\begin{bmatrix}
> \lambda_{1} \\ & \lambda_{2} \\ && \ddots \\ &&& \lambda_{n}
> \end{bmatrix}
> $$
> 其中 $\lambda_{1},\lambda_{2},\cdots,\lambda_{n}$ 是矩阵 $A$ 的特征值。

> [!note] 推论
> 1. $n$ 阶正规矩阵有 $n$ 个线性无关的特征向量。
> 2. 正规矩阵属于不同特征值的特征向量彼此正交。

> [!tip] 可以注意到这里与 [[02_Areas/线性代数/特征值与特征向量#实对称矩阵的相似对角化|相似对角化]] 的共性。常见的求解酉矩阵的过程同相似对角化。

## Hermite 二次型

> [!note] Hermite 矩阵的基本性质
> 设 $A\in C^{n\times n}$，则
> 1. $A+A^{H},AA^{H},A^{H}A$ 都是 H- 阵。
> 2. $A-A^{H}$ 是反 H- 阵。
> 3. 如果 $A$ 是 H- 阵，那么 $A^{k}$ 也是 $H-$ 阵。$k$ 为任意正整数。
> 4. 如果 $A$ 是可逆的 H- 阵，那么 $A^{-1}$ 也是可逆的 H- 阵。
> 5. 如果 $A$ 是 H- 阵 (反 H- 阵)，那么 $iA$ 是反 H- 阵 (H- 阵)，这里 $i$ 是虚数单位。
> 6. 如果 $A,B$ 都是 $H-$ 阵，那么 $kA+lB$ 也是 $H-$ 阵，这里 $k,l$ 均为实数。
> 7. 如果 $A,B$ 都是 $H-$ 阵，那么 $AB$ 也是 $H-$ 阵的充分必要条件是 $AB=BA$。

> [!definition|Definition] H- 阵的结构定理
> 设 $A\in C^{n\times n}$，则 $A$ 是 H- 阵的充分必要条件是存在一个酉矩阵 $U\in U^{n\times n}$ 使得
> $$
> U^{H}AU=\begin{bmatrix}
> \lambda_{1} \\ &\lambda_{2} \\ && \ddots \\ &&&\lambda_{n}
> \end{bmatrix}
> $$
> 其中 $\lambda_{1},\lambda_{2},\cdots,\lambda_{n}\in R$。也就是，**H- 阵酉相似于实对角矩阵**。

> [!tip] 推论：实对称矩阵正交相似于实对角矩阵。也就是 [[02_Areas/线性代数/特征值与特征向量#实对称矩阵的相似对角化|线性代数中对于实对称矩阵相似对角化的描述]]。

> [!definition|Definition] Hermite 二次型
> 由 $n$ 个复变量 $x_{1},x_{2},\cdots,x_{n}$，系数为复数的二次齐次多项式
> $$
> f(x_{1},x_{2},\cdots,x_{n})=\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{n}a_{ij}\overline{x}_{i}x_{j}
> $$
> 称为**Hermite 二次型**，这里
> $$
> a_{ji}=\overline{a}_{ij}
> $$
> 如果记
> $$
> X=\left[ x_{1},x_{2},\cdots,x_{n} \right] ^{T}\in C^{n}\quad A=\begin{bmatrix}
> a_{11} & a_{12} & \cdots & a_{1n} \\
> a_{21} & a_{22} & \cdots & a_{2n} \\
> \vdots & \vdots & \ddots & \vdots \\
> a_{n1} & a_{n2} & \cdots & a_{nn}
> \end{bmatrix}
> $$
> 则上面的 Hermite 二次型可以记为
> $$
> f(x_{1},x_{2},\cdots,x_{n})=X^{H}AX
> $$
> 称为 Hermite 二次型对应的矩阵，并称 $A$ 的秩为 Hermite 二次型的秩。

> [!tip] 对于线性代数中的定义，见 [[02_Areas/线性代数/二次型|二次型]]。

对 Hermite 二次型作可逆的线性替换
$$
X=CY
$$
可以得到
$$
f(x_{1},x_{2},\cdots,x_{n})=X^{H}AX=Y^{H}(C^{H}AC)Y=Y^{H}BY
$$

Hermite 二次型中最简单的一种是只含有纯平方项而无交叉项的二次型，称为**标准形的 Hermite 二次型**。其形式为
$$
f(y_{1},y_{2},\cdots,y_{n})=\lambda_{1}\overline{y}_{1}y_{1}+\lambda_{2}\overline{y}_{2}y_{2}+\cdots+\lambda_{n}\overline{y}_{n}y_{n}
$$

> [!definition|定理] 对于任何 Hermite 二次型，必然存在一个酉线性变换可以将 Hermite 二次型化为标准形。

> [!tip] 关于二次型的标准形、规范形及其计算，见 [[02_Areas/线性代数/二次型|二次型]]。

### 正定 Hermite 二次型

> [!definition|Definition] 正定二次型
> 对于给定的 Hermite 二次型
> $$
> f(X)=X^{H}AX
> $$
> 如果对于任意的一组不全为零复数都有
> $$
> f(x_{1},x_{2},\cdots,x_{n})> 0(\geqslant 0)
> $$
> 则称该 Hermite 二次型是正定的（半正定的），并且相应的 $H-$ 矩阵 $A$ 是正定的（半正定的）。

> [!note]
> 对于给定的 Hermite 二次型 $f(X)=X^{H}AX$，下列描述是等价的
> 1. $f(X)$ 是正定的
> 2. 对于任何 $n$ 阶可逆矩阵 $P$ 都有 $P^{H}AP$ 为正定矩阵
> 3. $A$ 的 $n$ 个特征值都大于 0
> 4. 存在 $n$ 阶可逆矩阵 $P$ 使得 $P^{H}AP=I$
> 5. 存在 $n$ 阶可逆矩阵 $Q$ 使得 $A=Q^{H}Q$
> 6. 存在正线上三角矩阵 $R$ 使得 $A=R^{H}R$，且此分解是唯一的

> [!note] 判定方式
> 1. $n$ 阶 Hermite 矩阵 $A$ 为正定的充要条件是 $A$ 的 $n$ 个顺序主子式都大于 0。
> 2. $n$ 阶 Hermite 矩阵 $A$ 为负定的充要条件是 $A$ 的 $n$ 个顺序主子式正负相间。
