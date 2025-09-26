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
1. $X(t)$ 是均值为 0 和方差为 $c^2t$ 的正态[[00-笔记/概率论与数理统计/一维随机变量及其分布|随机变量]]
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

## 击中时刻，最大随机变量，反正弦律

我们以 $T_a$ 记 Brown 运动过程首次击中 $a$ 的时刻。当 $a>0$ 时我们要通过考虑 $P\{X(t)\geqslant a\}$ 并取条件于是否有 $T_a\leqslant t$，来计算 $P\{T_a\leqslant t\}$。这就给出了
$$
\begin{aligned}
P\{X(t)\geqslant a\}=&P\{X(t)\geqslant a\mid T_a\leqslant t\}P\{T_a\leqslant t\}\\
&+P\{X(t)\geqslant a\mid T_a>t\}P\{T_a>t\}
\end{aligned}\tag{1}
$$
现在若 $T_a\leqslant t$，则过程在 $[0,t]$ 的某个点击中 $a$，由对称性，在时刻 $t$ 它等可能地在 $a$ 上方或者 $a$ 下方，这就是
$$
P\{X(t)\geqslant a\mid T_a\leqslant t\}=\frac{1}{2}
$$
由于式 $(1)$ 右方第二项显然等于 0（因为由连续性在击中 $a$ 之前过程的值不可能大于 $a$），我们可见
$$
\small P\{T_a\leqslant t\}=2P\{X(t)\geqslant a\}=\frac{2}{\sqrt{2\pi t}}\int_a^\infty e^{-x^2/2t}\mathrm d x=\frac{2}{\sqrt{2\pi}}\int_{a/\sqrt{t}}^\infty e^{-y^2/2}\mathrm dy\quad a>0
$$
因此我们可见
$$
P\{T_a<\infty\}=\lim_{t\to\infty}P\{T_a\leqslant t\}=\frac{2}{\sqrt{2\pi}}\int_0^\infty e^{-y^2/2}\mathrm dy=1
$$
此外，还可以推导
$$
\begin{aligned}
E[T_a]&=\int_0^\infty P\{T_a>t\}\mathrm dt=\int_0^\infty\left(1-\frac{2}{\sqrt{2\pi}}\int_{a/\sqrt{t}}^\infty e^{-y^2/2}\mathrm dy\right)\mathrm dt\\
&=\frac{2}{\sqrt{2\pi}}\int_0^\infty\int_0^{a/\sqrt{t}}e^{-y^2/2}\mathrm dy\mathrm dt=\frac{2}{\sqrt{2\pi}}\int_0^{\infty}\int_0^{a^2/y^2}\mathrm dte^{-y^2/2}\mathrm dy\\
&=\frac{2a^2}{\sqrt{2\pi}}\int_0^\infty \frac{1}{y^2}e^{-y^2/2}\mathrm dy\geqslant \frac{2a^2e^{-1/2}}{\sqrt{2\pi}}\int_0^1\frac{1}{y^2}\mathrm dy=\infty
\end{aligned}
$$

于是可以推出 $T_a$（虽然以概率 1 的有限）由无穷的期望。即以概率为 1 地 Brown 运动过程迟早会击中 $a$，但是平均时间是无穷的。

对 $a<0$，由对称性，$T_a$ 的分布与 $T_{-a}$ 的分布相同，因此，我们可以得到
$$
P\{T_a\leqslant t\}=\frac{2}{\sqrt{2\pi}}\int_{|a|/\sqrt{t}}^\infty e^{-y^2/2}\mathrm dy
$$
另一个有趣的随机变量过程在 $[0,t]$ 中达到最大值，它的分布可以如下得到，对 $a>0$，
$$
\begin{aligned}
P\{\max_{0\leqslant s\leqslant t}X(s)\geqslant a\}&=P\{T_a\leqslant t\}\\
&=2P\{X(t)\geqslant a\}=\frac{2}{\sqrt{2\pi}}\int_{a/\sqrt{t}}^\infty e^{-y^2/2}\mathrm dy\end{aligned}
$$
以 $0(t_1,t_2)$ 记 Brown 运动过程在区间 $(t_1,t_2)$ 中至少有一次取 0 这一事件，为了计算 $P\{0(t_1,t_2)\}$，我们取条件于 $X(t_1)$ 如下：
$$
P\{0(t_1,t_2)\}=\frac{2}{\sqrt{2\pi t_1}}\int_{-\infty}^\infty P\{0(t_1,t_2)\mid X(t_1)=x\}e^{-x^2/2t_1}\mathrm dx
$$
利用 Brown 运动关于原点的对称性和路径的连续性给出
$$
P\{0(t_1,t_2)\mid X(t_1)=x\}=P\{T_{|x|}\leqslant t_2-t_1\}
$$
因此得到
$$
P\{0(t_1,t_2)\}=\frac{1}{\pi\sqrt{t_1(t_2-t_1)}}\int_0^\infty\int_x^\infty e^{-y^2/2(t_2-t_1)}\mathrm dye^{-x^2/2t_1}\mathrm dx
$$
上述积分可由显示算出，它导致
$$
P\{0(t_1,t_2)\}=1-\frac{2}{\pi}\arcsin\sqrt{t_1/t_2}
$$
因此我们得到了下面的命题：对 $0<x<1$，有
$$
P\{\text{Brown运动在($xt,t$)无零点}\}=\frac{2}{\pi}\arcsin\sqrt{x}
$$

> 对于对称随机徘徊，有
> $$P\{\ 在(nx,n)无零点\}\approx \frac{2}{\pi}\arcsin{\sqrt{x}}$$
> 当 $n\to\infty$，近似就变为了精确的相等，因此这是符合逻辑的。


