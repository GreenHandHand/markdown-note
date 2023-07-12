# Brown 运动

## Brown 运动

从对称随机徘徊开始，在每一个时间单位它等可能的向左或者向右走一步。现在假设在越来越小的时间区间走越来越小的一步以加速这个过程，若我们以正确的形式趋向极限，那么我们将会得到 Brown 运动。

下面推导 Brown 运动的定义。假设每个 $\Delta t$ 单位时间我们以相等的概率向左或者向右走大小为 $\Delta x$ 的一步，若我们使用 $X(t)$ 记为在时刻 $t$ 的位置，那么
$$
X(t)=\Delta x(X_1+X_2+\cdots+X_{[t/\Delta t]})
$$
其中
$$
X_i=\begin{cases}+1&向左\\-1&向右\end{cases}
$$
且 $X_i$ 假定是独立的，具有
$$
P\{X_i=1\}=P\{X_i=-1\}=\frac{1}{2}
$$
根据 $E[X_i]=0,Var(X_i)=1$ 于是可以得到期望与方差为
$$
E[X(t)]=0,\quad Var(X{\small(t)})=(\Delta x)^2[\frac{t}{\Delta t}].
$$

现在将 $\Delta t$ 与 $\Delta x$ 趋于零，同时为了排除平凡的情况，我们令 $\Delta x=c\sqrt{\Delta t}$，其中 $c$ 不为 0，于是在 $\Delta t\to0$ 时，有
$$
E[X(t)]=0\quad Var(X{\small (t)})\to c^2t
$$

根据上面的推导可以得到 Brown 的性质：
1. $X(t)$ 是均值为 0 和方差为 $c^2t$ 的正态随机变量
2. $\{X(t),t\geqslant 0\}$ 有独立增量，因为随机徘徊在不相交的时间区间上的值的变化是独立的
3. $\{X(t),t\geqslant 0\}$ 有平稳增量，因为随机徘徊在任意时间区间中的位置变化的分布只依赖区间的长度

综上，我们可以得到：
>随机过程 $\{X(t),t\geqslant0\}$ 若满足
> 1. $X(0)=0$
> 2. $\{X(t),t\geqslant0\}$ 有平稳的独立增量
> 3. 对任意 $t>0$，$X(t)$ 服从均值为 $0$ 和方差为 $c^2t$ 的正态分布
> 
> 则称为 **Brown 运动过程**

Brwon 运动过程有时也称为 Wiener 过程，它是应用概率论中最有用的随机过程之一，源自于物理学中作为 Brown 运动现象的一种描述。

当 $c=1$ 时，这个过程称为标准 Brown 运动，任何的 Brown 运动都可以通过 $X(t)/c$ 转换为标准 Brown 运动，因此后面讨论的均为标准 Brwon 运动。

## 对 Brown 运动的其他理解

### Brown 运动与随机徘徊

Brown 运动可以理解为随机徘徊的极限，根据
$$
X(t)=\Delta x(X_1+X_2+\cdots+X_{[t/\Delta t]})
$$
我们对上式取了极限，因此 Brown 运动中 $X(t)$ 应当为 $t$ 的连续函数，实际上可以证明 $X(t)$ 以 1 的概率是 $t$ 的连续函数。同时，与随机徘徊相同，$X(t)$ 总是存在尖角，故而绝对不光滑。同样可以证明，Brown 运动以 1 的概率 $X(t)$ 处处不可微。

### Brown 运动与 Markov 过程

独立增量假设蕴含了 $X(s+t)-X(s)$ 与过去在时刻 $t$ 以前的一切值独立，因此有
$$
\begin{aligned}
&\quad P\{X(t+s)\leqslant a\mid X(s)=x,X(u),0\leqslant u\leqslant s\}\\
&=P\{X(t+s)-X(s)\leqslant a-x\mid X(s)=x,X(u),0\leqslant u\leqslant s\}\\
&=P\{X(t+s)-X(s)\leqslant a-x\}\\
&=P\{X(t+s)\leqslant a\mid X(s)=x\}
\end{aligned}
$$
它说明了在给定当前状态 $X(s)$ 和过去状态 $X(u),0<u<s$ 时，将来的状态 $X(s+t)$ 的条件分布只依赖当前的状态。满足这个条件的性质称为 Markov 过程。

### Brown 运动与 Gauss 过程

由于 $X(t)$ 是均值为 0 和方差为 $t$ 的正态随机变量 (这里考虑的是标准 Brown 运动)，它的密度函数为
$$
f_t(x)=\frac{1}{\sqrt{2\pi t}}e^{-x^2/2t}
$$
由平稳独立增量假设可以推出，$(X(t_1),\cdots,X(t_n))$ 的联合密度为
$$
f(x_1,x_2,\cdots,x_n)=f_{t_1}(x_1)f_{t_2-t_1}(x_2-x_1)\cdots f_{t_n-t_{n-1}}(x_n-x_{n-1})
$$
利用上式，我们原则上可以计算任意需求的概率。同时，也可以推出 $(X(t_1),\cdots,X(t_n))$ 的联合分布是多元正态分布，于是 Brown 运动是一个 Gauss 过程。

> Gauss 过程定义：若对一切的 $t_1,\cdots,t_n$，$(X(t_1),\cdots,X(t_n))$ 具有多元正态分布，则称随机过程 $\{X(t),t\geqslant 0\}$ 为 Gauss 过程。

由于多元正态分布由边缘分布的均值和协方差的值完全确定，由此推出 Brown 运动过程也可以定义为具有 $E[X(t)]=0$ 和对 $s\leqslant t$ 有
$$
\begin{aligned}
\mathrm{Cov}(X(s),X(t))&=\mathrm{Cov}(X(s),X(s)+X(t)-X(s))\\
&=\mathrm{Cov}(X(s),X(s)) + \mathrm{Cov}(X(s),X(t)-X(s))=s
\end{aligned}
$$
最后一步得自独立增量与 $Var(X(s))=s$。

### Brown 桥

令 $\{X(t),t\geqslant 0\}$ 是一个 Brown