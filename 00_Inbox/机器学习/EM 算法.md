---
aliases:
  - 期望最大化算法
  - EM
  - Expectation Maximum
  - EM 算法
  - 高斯混合模型
tags:
  - 机器学习
  - 基本方法
  - 无监督学习
---

# 期望最大化 (EM, Expectation-Maximum) 算法及其应用

对参数的一种求解的方法，是一类通过迭代进行极大似然估计的优化方法，通过用于对包含隐变量或缺失数据的概率模型进行参数估计。也称为 Dempster-Laird-Rubin 算法。

## 理论的准备

### 极大似然估计

思想：通过求解 使得样本服从分布的概率最大时 对应的参数的值来估计参数。因此，当给定的数据集 X 的分布已知时，估计分布的参数可以使用极大似然估计。而 EM 算法用于分布未知时，估计参数的取值。

极大似然估计可以使用数学语言描述如下：
$$
\hat{\theta}=\arg_{\theta}\max L(\theta)
$$
其中 $L(\theta)$ 为似然函数，用于表征在参数 $\theta$ 下得到样本的概率，常记为 $f(x,\theta)$ 或 $L(x_1,x_2,\dots,x_n;\theta)$，对于一系列的样本和一个样本服从的概率分布，我们可以使用 $\displaystyle\arg\max\prod_ip(x_i)$ 进行计算，在实际使用中，我们常常将该式取对数以简化计算。

### Jensen 不等式

定理：假设 $f$ 是一个凸函数，$X$ 是一个 [[02_Areas/概率论与数理统计/一维随机变量及其分布|随机变量]]，有：
$$
E[f(X)]\ge f(EX)
$$
进一步的，如果 f 是严格凸函数，则 $E[f(x)]=f(EX)$ 成立的条件为 $P(X=EX)=1$，即 $X$ 是一个常数。

## EM 算法的推导

首先，有极大似然估计我们可以得到参数 $\theta$ 的估计为：
$$
\begin{align}\hat{\theta} = \arg_{\theta}\max\sum_i\log p(x_i|\theta)\end{align}$$ 
而在不知道 $x_i$ 的分布下，或者说 $x_i$ 的分布与其他隐藏变量有关时，我们无法直接求出 $p(x_i)$ 的值。因此我们设所有隐变量 $z=(z_1,z_2,\dots)$，于是含有隐变量的极大似然估计表示为：
$$
\begin{align}\hat\theta=\arg_\theta\max\sum_i\log\sum_{z(i)}p(x_i,z_i|\theta)\end{align}$$
其中，根据概率的性质有：
$$
\begin{align}\sum_{z(i)}p(x_i,z_i|\theta)=p(x_i|\theta)\end{align}$$
上面的算式难以进行计算，所以引入 $z$ 的分布 $Q_i(z_i)$：
$$
\begin{align}\hat\theta=\arg_\theta\max\sum_i\log\sum_{z(i)}Q_i(z_i)\frac{p(x_i,z_i|\theta)}{Q_i(z_i)}\end{align}
$$
由于 $\log$ 函数为严格凹函数，所以由 Jensen 不等式可以得到：
$$
\begin{align}\log[EX]\ge E[\log(X)]\end{align}
$$
设 $\dfrac{p(x_i,z_i|\theta)}{Q_i(z_i)}$ 为 $g_i(z_i)$，则由[[02_Areas/概率论与数理统计/随机变量的数字特征|数学期望]]的性质可以得到：
$$
\left\{\begin{aligned}\log E[g_i(z)]&=\log\sum_{z_i}g(z_i)\cdot Q_i(z_i)\\[3mm]E[\log g_i(z)]&=\displaystyle\sum_{z_i} Q_i(z_i)\log g_i(z_i)\end{aligned}\right.
$$
结合 Jensen 不等式有：
$$
\begin{align}\log\sum_{z_i}g_i(z_i)\cdot Q_i(z_i) \ge Q_i(z_i)\log\sum_{z_i}g_i(z_i)\end{align}
$$
所以 $\theta$ 的估计可以转化为：
$$
\begin{align}\hat\theta \ge \arg_\theta\max\sum_i\sum_{z_i}Q_i(z_i)\log \dfrac{p(x_i,z_i|\theta)}{Q_i(z_i)}\end{align}
$$
现在我们考虑该式取等号的条件，由 Jensen 不等式去等的条件我们可以得到：
$$\begin{align}g_i(z_i)=\dfrac{p(x_i,z_i|\theta)}{Q_i(z_i)}=c_i\end{align}
$$
因为 $Q_i(z_i)$ 是 $z$ 的分布，所以有：
$$
\begin{align}\sum_{z_i}Q_i(z_i)=1\end{align}
$$
将取等条件朝着上式转换可以得到：
$$
\begin{align}\sum_{z_i}p(x_i,z_i|\theta)=c_i\sum_{z_i}Q_i(z_i)=c_i\end{align}
$$
于是可以得到 $Q_i(z_i)$ 的表达式为：
$$
\begin{align}Q_i(z_i)=\frac{p(x_i,z_i|\theta)}{c_i}=\dfrac{p(x_i,z_i|\theta)}{p(x_i|\theta)}=p(z_i|x_i,\theta)\end{align}
$$
通过该式即可通过已知的 $\theta$ 与样本 $x_i$、假设的隐变量 $z_i$ 来计算真实的隐变量 $z$ 的分布。该式便是 EM 算法中的 E 步。

通过将上式计算得到的 $Q_i(z_i)$ 带入 $\theta$ 的估计可以求得新的 $\theta$ 估计值：$$\begin{align}\hat\theta = \arg\max_\theta\sum_i\sum_{z(i)}Q_i(z_i)\log \dfrac{p(x_i,z_i|\theta)}{Q_i(z_i)}\end{align}$$ 该式便是 EM 算法中的 M 步。将目标函数记为 $Q$ 函数，省去一些对极大化没有帮助的常数，再向后进行化简可以得到常用的 $Q$ 函数形式：
$$
\begin{aligned}
Q(\theta,\hat\theta)&=\sum_i\sum_{z(i)}Q_i(z_i)\log \dfrac{p(x_i,z_i|\theta)}{Q_i(z_i)}\\
&=\sum_i\sum_{z_i}p(z_i|x_i,\theta)\log \frac{p(x_i,z_i|\theta)}{p(z_i|x_i,\hat \theta)}\\
&=\sum_i\sum_{z_i}p(z_i|x_i,\hat \theta)\log p(x_i,z_i|\theta)
\end{aligned}
$$
所以 $Q$ 函数表示为如下形式
$$
\begin{aligned}
Q(\theta,\hat\theta)&=E_Z[\log P(X,Z|\theta)|X,\theta^{(i)}]\\
&=\sum_Z P(Z|X,\theta^{(i)})\log P(X,Z|\theta)
\end{aligned}
$$
M 步就可以描述为
$$
\theta^{(n+1)}=\arg\max_\theta Q(\theta,\theta^{(n)})
$$

通过不断的迭代上面的两个式子，最终 $\theta$ 将会收敛，收敛的值便是我们所求的参数。

总结：
> EM 算法总共分为 E 步与 M 步：
> 1. 给定 $\theta^{(0)}$ 用于第一次迭代
> 2. E 步：使用 $Q_i^n(z_i)=p(z_i|x_i;\theta^{n-1})$ 计算得到第 n 次 $z$ 的分布
> 3. M 步：将 $Q_i^n(z_i)$ 带入 $\hat\theta^n = \displaystyle\arg_\theta\max\sum_i\sum_{z(i)}Q_i^n(z_i)\log \dfrac{p(x_i,z_i;\theta)}{Q_i^n(z_i)}$ 中求得 $\theta^n$ 的值
> 4. 重复 2 与 3 直至收敛。

或者使用

>EM 算法：
>1. 给定 $\theta^{(0)}$ 作为初值，开始迭代
>2. E 步：计算第 $i+1$ 步的 $Q$ 函数:
>$$\begin{aligned}
Q(\theta, \theta^{(i)})&=E_Z[\log P(X,Z|\theta)|X,\theta^{(i)}]\\
&=\sum_Z P(Z|X,\theta^{(i)})\log P(X,Z|\theta)
\end{aligned}$$
>3. M 步：求使 $Q$ 函数极大化的参数 $\theta$ 作为 $\theta^{(i+1)}$ 的值

### EM 算法的收敛性

## 高斯混合模型

EM 算法的一个重要的应用是高斯混合模型 (Gaussian mixture model) 的参数估计。高斯混合模型是指具有如下形式的概率分布模型：
$$
P(y|\theta)=\sum_{k=1}^K\alpha_k\phi(y|\theta_k)
$$
其中 $\alpha$ 是系数，$\alpha_k\geqslant 0，\sum_{k=1}^K\alpha_k=1$，$\phi(y|\theta_k)$ 是高斯分布密度，$\theta_k=(\mu_k,\sigma_k^2)$。
$$
\phi(y|\theta_k)=\frac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\frac{(y-\mu_k)^2}{2\sigma_k^2}\right)
$$
称为第 k 个模型，一般的混合模型可以使用任意的概率密度函数替代高斯分布密度。高斯混合模型是一种无监督 [[00_Inbox/机器学习/聚类|聚类]] 算法，是一种概率聚类算法。高斯混合模型假设每一类都符合一个高斯分布，以该假设为原则对无标签的数据进行聚类。

使用 EM 算法求解问题需要我们明确模型中的隐变量是什么，从而在 E 步中使用 $x$ 与 $\theta^{(n)}$ 对隐变量的分布进行估计。其次需要计算完全数据的似然函数，在 M 步中求解最大似然来重新估计参数 $\theta^{(n+1)}$。

体现在高斯混合模型中，可以设想数据由下面的方式得到：先依概率 $\alpha_k$ 选择第 k 个高斯分布，然后根据第 k 个概率密度分布生成观测数据 $x_j$。这时观测数据 $x$ 是已知的，反映观测数据来自第 k 类的数据是未知的。我们可以假设隐变量 $\gamma_{kj}$ 表示观测 $x_j$ 在来自第 k 个分布中：
$$
\gamma_{kj}=\begin{cases}1, &x_j\sim\phi_k\\0,&\text{other.}\end{cases}
$$
可以得到：
$$
p(x_i,\gamma_{1i},\gamma_{2i},\cdots|\theta)=\prod_{k=1}^K[\alpha_{k}\phi(x_i|\theta)]^{\gamma_{ki}}
$$
根据先验概率：
$$
p(x_i|\gamma_{ki}=1)=\phi(x_i)
$$
于是可以得到在数据 $x_i$ 与模型参数 $\theta$ 确定的情况下 $\gamma$ 的分布的估计：
$$
\begin{aligned}
Q_k(\gamma_{ki})&=p(\gamma_{ki}=1|x_i)\\&=\frac{p(x_i|\gamma_{ki}=1)p(\gamma_{ki}=1)}{\sum_{k=1}^Kp(\gamma_{ki}=1)p(x_i|\gamma_{ki}=1)}\\&=\frac{\alpha_k\phi(x_i)}{\sum_{k=1}^K\alpha_k\phi(x_i)}\\
\end{aligned}
$$
因此 E 步使用:
$$
Q_k(\gamma_{ki})=\frac{\alpha_k\phi(x_i)}{\sum_{k=1}^K\alpha_k\phi(x_i)}
$$
来估计隐变量的分布。

在 M 步中，我们需要计算:
$$
\begin{align}\hat\theta &= \arg_\theta\max\sum_{i=1}^N\sum_{k=1}^KQ_k(\gamma_{ki})\log \dfrac{p(x_i,\gamma_{ki}|\theta)}{Q_k(\gamma_{ki})}
\\&=\arg_\theta\max\sum_{i=1}^N\sum_{k=1}^KQ_k(\gamma_{ki})\log\frac{p(x_i|\gamma_{ki},\theta)p(\gamma_{ki}|\theta)}{Q_k(\gamma_{ki})}\\
&=\arg_\theta\max\sum_{i=1}^N\sum_{k=1}^KQ_k(\gamma_{ki})\log\frac{\alpha_{k}\phi(x_i|\theta)}{Q_k(\gamma_{ki})}\end{align}
$$
通过将偏导置零可以解得
$$
\begin{aligned}
\hat \mu_k&=\frac{\sum_{i=1}^NQ_k(\gamma_{ki})x_i}{N_k}\\
\hat \sigma_k^2&=\frac{\sum_{i=1}^NQ_k(\gamma_{ki})(x_i-\mu_k)^2}{N_k}\\
\hat \alpha_k&=\frac{\sum_{i=1}^N Q_{k}(\gamma_{ki})}{N}\\
N_k&=\sum_{i=1}^NQ_k(\gamma_{ki})
\end{aligned}
$$
使用训练后的高斯混合模型对数据进行分类即计算后验概率：
$$
p(\gamma_{k}|x_i,\theta)
$$
发现这其实就是训练中的 E 步，我们只需执行 E 步后，选择后验概率中最大的即可。
