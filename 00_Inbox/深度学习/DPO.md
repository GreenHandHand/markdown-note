---
aliases:
  - 偏好学习
tags:
  - 深度学习
---

# DPO

[[RLHF]] (Reinforcement Learning from Human Feedback) 通过人类对模型回答的偏好，引导语言模型生成更符合人类期望的内容。经典的 RLHF 流程首先利用人类偏好数据训练一个**奖励模型** (Reward Model)，然后将该奖励模型作为优化目标，使用 PPO 等强化学习算法调整语言模型的生成策略。

具体而言，给定偏好数据集，
$$
\mathcal{D}=\left\{ (x, y_{w},y_{l}) \right\} 
$$
其中 $y_{w}\succ y_{l}$ 表示人类偏好 $y_{w}$，RLHF 首先学习一个能够反映人类偏好的奖励函数 $r_{\phi}(x,y)$，然后通过强化学习优化语言模型的生成策略:
$$
\max_{\pi}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi(\cdot\mid x)}\left[ r_{\phi}(x,y) \right] -\beta \mathbb{E}_{x\sim \mathcal{D}}\left[ D_{\text{KL}}(\pi(\cdot\mid x)\Vert \pi_{\text{ref}}(\cdot\mid x)) \right] \tag{1}
$$
其中，$\pi_{\text{ref}}$ 为固定的参考测量，通常由 SFT 模型提供；KL 正则化用于控制优化后的测量相对于参考测量的偏移。该优化目标明确描述了 RLHF 希望得到的结果：**在控制策略偏移的条件下，使模型生成的回答获得更高的人类偏好奖励**。

> [!note]- 优化目标的解释
> 其中
> - $x$ 表述输入 prompt
> - $y\in \mathcal{Y}$ 表示一个完整的候选回答，其中 $\mathcal{Y}$ 为回答空间
> - $r(x,y)$ 为固定的奖励函数，在传统 RLHF 中对应训练完成的 $r_{\phi}$
> - $\pi_{\text{ref}}(y\mid x)$ 为固定的参考测量
> - $\pi(y\mid x)$ 为需要优化的概率分布
> - $\beta > 0$ 为 KL 正则化系数

传统的 RLHF 需要依次完成两个学习过程：
1. **奖励学习**：从人类对回答的相对偏好中，学习一个能对任意回答进行评分的奖励函数。
2. **策略优化**：利用学习到的奖励函数，通过强化学习寻找更高奖励的生成策略。

这个两阶段流程带来了较高的计算开销和训练复杂度。奖励模型需要从有限的偏好数据中学习评价标准，PPO 则需要在训练过程中反复采样、计算奖励并更新策略，通常还涉及额外的价值模型。

RLHF 的最终目标是得到符合人类偏好的语言模型策略。那么，能否直接从偏好数据中学习这个策略，省去显式奖励建模和强化学习优化过程？**DPO (Direct Preference Optimization, 直接偏好优化)** 就是从这个问题出发的。

## 从偏好数据到奖励函数

在进一步讨论 DPO 之前，需要明确 RLHF 如何从人类偏好数据中学习奖励函数。经典 RLHF 使用 PPO 等强化学习算法优化语言模型策略。在训练过程中，策略会不断生成新的回答，而强化学习算法需要根据这些回答的质量获得奖励信号。然而，人类偏好数据通常只包含同一个 prompt 下两个回答的比较结果，这类数据能够反映回答之间的相对质量，却没有直接提供具体的奖励数值。因此，RLHF 希望从有限的偏好数据中学习一个奖励函数 $r_{\phi}(x,y)$，使其能够对任意给定的 prompt 和候选回答输出标量评分，为候选的策略优化提供奖励信号。

为了从相对偏好中学习奖励函数，我们首先需要建立奖励值与人类偏好之间的概率关系。Bradly-Terry 模型是一种经典的成对比较概率模型。它假设每个候选回答有一个正的潜在强度 $s(x,y)>0$，并将某个回答被偏好的概率表示为其强度在两个候选回答总强度中的占比：
$$
p(y_{1} \succ y_{2}\mid x)=\dfrac{s(x,y_{1})}{s(x,y_{1})+s(x,y_{2})}
$$
为了使用实数值奖励函数描述回答的潜在质量，我们通常使用指数函数 $s(x,y)=\exp(r(x,y))$，从而得到
$$
\begin{align}
p(y_{1} \succ y_{2}\mid x) &= \dfrac{\exp(r(x,y_{1}))}{\exp(r(x,y_{1}))+\exp(r(x,y_{2}))} \\
&=\dfrac{1}{1+\exp(-(r(x,y_{1})-r(x,y_{2})))}\\
&=\sigma \left( r(x,y_{1})-r(x,y_{2}) \right) 
\end{align}
$$
其中 $\sigma(z)=1/(1+e^{ -z })$ 是 Sigmoid 函数。

在 Bradley-Terry 模型中，两个回答的奖励差决定了人类偏好其中一个回答的概率。奖励差越大，模型预测高奖励回答被选择的概率越高。于是，我们可以通过这个关系从人类偏好数据中学习奖励函数。具体而言，我们构造一个参数化奖励模型 $r_{\phi}(x,y)$。对于偏好数据中的样本 $(x,y_{w},y_{l})$，分别计算两个回答的奖励：
$$
r_{w}=r_{\phi}(x,y_{w}),\qquad r_{l}=r_{\phi}(x,y_{l})
$$
基于 Bradley-Terry 模型，奖励模型预测人类选择 $y_{w}$ 的概率为
$$
p_{\phi}(y_{w} \succ y_{l}\mid x)=\sigma(r_{w}-r_{l})
$$
而偏好数据则告诉我们人类实际上选择了 $y_{w}$。因此，可以将这个观测结果视为一个二分类任务中的正标签，通过最大化其概率来学习奖励模型。使用极大似然估计，相当于优化：
$$
\mathcal{L}_{\text{RM}}(\phi)=-\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}}\left[ \log\sigma(r_{\phi}(x,y_{w})-r_{\phi}(x,y_{l})) \right] 
$$
该损失鼓励奖励模型给人类偏好的回答分配更高的奖励。在奖励模型训练完成后，经典的 RLHF 再通过强化学习的方式优化策略。

> [!note] 一个小的理解
> Bradley-Terry 模型可以看成是一种特殊的二分类，普通的二分类通常学习
> $$
> p(c=1\mid z) = \sigma(f_{\phi}(z))
> $$
> 其中 $f_{\phi}(z)$ 是分类的 logit，Sigmoid 将其转换为类别概率。而 Bradley-Terry 模型采用了类似的形式
> $$
> p(y_{1}\succ y_{2}\mid x)=\sigma(r_{\phi}(x,y_{1})-r_{\phi}(x,y_{2}))
> $$
> 因此，从二分类角度看，我们 Bradley-Terry 本质上是在判断**模型在 prompt $x$ 下相较于 $y_{2}$ 更加偏向 $y_{1}$。** 其输出就是选择 $y_{1}$ 的概率。因此，BT 模型可以理解为一个具有特殊结构的二分类模型：**先分别计算两个样本的分数，再通过分数差进行二分类**。
>
> 虽然训练奖励模型时，我们需要配对数据。但是经过训练之后，我们得到的是共享的评分函数 $r_{\phi}(x,y)\in \mathbb{R}$。因此 PPO 对新生成的回答 $y$ 可以直接计算$r_{\phi}(x,y)$，无需再找一个配对回答。

## 从奖励函数到最优策略

前面介绍了经典 RLHF 如何利用 Bradley-Terry 模型，从人类偏好数据中学习奖励函数 $r_{\phi}(x,y)$。得到奖励模型后，RLHF 还需要通过 PPO 等强化学习算法，优化语言模型的生成策略。

RLHF 的最终目的是获得一个最大化人类偏好奖励的策略。DPO (Direct Preference Optimization, 直接偏好优化) 重新审视了奖励函数和最优策略之间的关系，并尝试直接从偏好数据中学习最优策略。对于式 $\ref{eq:1}$ 而言，最优策略实际上具有一个解析形式的解：
$$
\pi_{r}^{*}(y\mid x) = \dfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp \left( \dfrac{r(x,y)}{\beta} \right) \tag{2}
$$
其中 $Z(x)$ 是归一化常数，具体形式见推导。式 $\ref{eq:2}$ 表明，**给定奖励函数和参考策略，最优策略在数学上就已经被唯一确定**。这一结果建立了奖励函数到最优策略的对应关系。DPO 进一步考虑，既然 RLHF 的目标是学习最优策略，那么能否利用这一对应关系，将奖励函数表示为策略的函数，从而直接优化策略？

> [!note]- 关于式 $\ref{eq:2}$ 的推导
> 考虑 RLHF 的优化目标
>
> $$
> \max_{\pi}\mathbb{E}_{x\sim \mathcal{D},y\sim \pi(\cdot\mid x)}\left[ r_{\phi}(x,y) \right] -\beta \mathbb{E}_{x\sim \mathcal{D}}\left[ D_{\text{KL}}(\pi(\cdot\mid x)\Vert \pi_{\text{ref}}(\cdot\mid x)) \right]
> $$
>
> 其中
> - $x$ 表述输入 prompt
> - $y\in \mathcal{Y}$ 表示一个完整的候选回答，其中 $\mathcal{Y}$ 为回答空间
> - $r(x,y)$ 为固定的奖励函数，在传统 RLHF 中对应训练完成的 $r_{\phi}$
> - $\pi_{\text{ref}}(y\mid x)$ 为固定的参考测量
> - $\pi(y\mid x)$ 为需要优化的概率分布
> - $\beta > 0$ 为 KL 正则化系数
> 
> 将 KL 散度展开
> $$
> D_{\text{KL}}(\pi \Vert \pi_{\text{ref}}) = \sum\limits_{y'\in \mathcal{Y}}\pi(y\mid x)\log \dfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
> $$
> 代入优化目标中得到
> $$
> \mathcal{J}_{x}(\pi)=\sum\limits_{y'\in \mathcal{Y}}\pi(y\mid x)r(x, y) - \beta \sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)\log \dfrac{\pi(y'\mid x)}{\pi_{\text{ref}}(y'\mid x)}
> $$
> 这是一个关于概率分布 $\pi(\cdot\mid x)$ 的优化问题。我们可以使用拉格朗日乘数法求解，使用等式约束 $\sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)=1$，于是可以得到：
> $$
> \mathcal{F}(\pi,\lambda)=\sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)r(x,y')-\beta \sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)\log \dfrac{\pi(y'\mid x)}{\pi_{\text{ref}}(y'\mid x)}+\lambda(x) \left( \sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x) - 1 \right) 
> $$
> 对于 $y' \in \mathcal{Y}$, 对 $\pi(y'\mid x)$ 求导并令导数为零得到
> $$
> r(x,y')-\beta \left( \log\dfrac{\pi(y'\mid x)}{\pi_{\text{ref}}(y'\mid x)} + 1 \right) + \lambda(x)=0
> $$
> 整理得到
> $$
> \pi(y'\mid x) = \pi_{\text{ref}}(y'\mid x)\exp \left( \dfrac{r(x,y')}{\beta} \right) \exp \left( \dfrac{\lambda(x)}{\beta}-1 \right) 
> $$
> 代入 $\sum\limits_{y'}\pi(y'\mid x)=1$ 得到
> $$
> \dfrac{1}{Z(x)} = \exp \left( \dfrac{\lambda(x)}{\beta}-1 \right) =1 / \sum\limits_{y'\in \mathcal{Y}}\pi_{\text{ref}}(y'\mid x)\exp \left( \dfrac{r(x,y')}{\beta} \right) 
> $$
> 于是带回原式得到
> $$
> \pi_{r}^{*}(y\mid x)=\dfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp(\dfrac{r(x,y)}{\beta})
> $$
> 即于是得到式 $\ref{eq:2}$。$\mathcal{J}_{x}(\pi)$ 是关于 $\pi$ 的一个严格凸函数，在全正的 $\pi$ 支持下，求得的驻点是唯一最大值。

对式 $\ref{eq:2}$ 进行变换，可以得到
$$
r(x,y)=\beta \log\dfrac{\pi_{r}^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
$$
这意味着，给定一个奖励函数及其对应的最优策略，我们可以通过**最优策略**与**参考策略**的对数概率比，重新表示该奖励函数。

然而，这一表达式仍然存在问题，归一化常数 $Z(x)$ 需要对整个回答空间求和，实际计算起来非常困难。虽然 $Z(x)$ 难以计算，但这个有一个值得利用的性质：**对于同一个 prompt 而言，所有回答的 $Z(x)$ 是相同的**。因此，对于两个回答 $y_{1},y_{2}$ 而言，计算奖励差可以消去这一项：
$$
r(x,y_{1})-r(x,y_{2})=\beta \log\dfrac{\pi_{r}^{*}(y_{1}\mid x)}{\pi_{\text{ref}}(y_{1}\mid x)}-\beta \log\dfrac{\pi_{r}^{*}(y_{2}\mid x)}{\pi_{\text{ref}}(y_{2}\mid x)}\tag{3}
$$
回顾前面的 Bradley-Terry 模型，可以发现式 $\ref{eq:3}$ 恰好可以在不计算 $Z(x)$ 的情况下表示奖励差。因此，我们可以在不计算归一化常数的情况下，直接通过最优策略和参考策略表示 Bradley-Terry 模型所需要的奖励差。这为直接使用偏好数据优化策略提供了可能。

## 从最优策略到 DPO 损失

将式 $\ref{eq:3}$ 代入 Bradley-Terry 偏好模型中，可以得到
$$
p_{r}(y_{w}\succ y_{l}\mid x)=\sigma \left( \beta \log\dfrac{\pi_{r}^{*}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)} -\beta \log\dfrac{\pi_{r}^{*}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right) \tag{4}
$$
式 $\ref{eq:4}$ 建立了**奖励函数、最优策略与人类偏好概率之间的关系**。对于给定的奖励函数 $r$，其对应的最优策略 $\pi_{r}^{*}$ 可以用于计算 Bradley-Terry 模型中的偏好概率。换句话说，我们可以通过最优策略与参考策略的概率比，来表示原本由奖励函数决定的偏好概率。

在传统的 RLHF 中，我们首先通过偏好数据学习奖励函数 $r_{\phi}$，在通过强化学习求解对应的最优策略 $\pi_{r}^{*}$。现在，式 $\ref{eq:4}$ 为我们提供了另外一种思路：**既然最优策略本身就能够表示偏好概率，那么能否直接通过偏好数据学习最优策略？**

### DPO 损失

式 $\ref{eq:4}$ 中的 $\pi_{r}^{*}$ 是未知奖励函数 $r$ 对应的最优策略，我们无法直接计算它。DPO 引入了一个可训练的参数化策略模型 $\pi_{\theta}$，用它来估计未知的 $\pi_{r}^{*}$。具体而言，DPO 将式 $\ref{eq:4}$ 中的最优策略替换为当前策略模型，构造参数化的偏好概率：
$$
p_{\theta}(y_{w}\succ y_{l}\mid x) = \sigma \left( \beta \log\dfrac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)} -\beta \log\dfrac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right)\tag{5} 
$$
其中 $\pi_{\theta}$ 是需要训练的策略模型，$\pi_{\text{ref}}$ 是固定的参考策略，$p_{\theta}$ 是由当前策略预测的人类偏好概率。这样，我们就将策略模型 $\pi_{\theta}$ 转换为一个可以预测人类偏好的二分类模型，然后沿用之前奖励模型训练的方法，利用偏好数据进行极大似然估计。

对于偏好数据 $\mathcal{D}=\left\{ (x_{i},y_{w,i},y_{l,i}) \right\}_{i=1}^{N}$，每个样本都表示了人类选择 $y_{w}$ (Accept) 而拒绝 $y_{l}$ (Reject)。因此，我们希望最大化模型对这些已观测偏好的预测概率：
$$
\max_{\theta}\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}} \left[ \log p_{\theta}(y_{w}\succ y_{l}\mid x) \right]
$$
也就是说，DPO 的损失函数为
$$
\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}}\left[ 
\log\sigma \left( \beta \log\dfrac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)} \right) - \beta \log\dfrac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)} 
 \right] 
$$
通过最小化这一损失，模型会倾向于提高被选择回答相对于参考策略的概率优势，同时降低未被选择回答的相对优势。因此，DPO 可以直接利用偏好数据，通过常规的梯度下降优化生成策略，无需显示训练奖励模型，也无须使用 PPO 等强化学习算法。

> [!note] 为什么要最大化 log?
> 实际上就是交叉熵损失（或者 BCELoss），由于是二分类问题，且 $p_{\theta}(y_{w}\succ y_{l}\mid x)$ 的标签永远是 1，因此写成交叉熵损失就是
> $$
> \begin{align}
> &\min_{\theta} \mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}}\left[ -1 \cdot \log p_{\theta}(y_{w}\succ y_{l}\mid x) - 0 \cdot \log p_{\theta}(y_{l}\succ y_{w}\mid x)  \right] \\
> \iff &\max_{\theta}\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}} \left[ \log p_{\theta}(y_{w}\succ y_{l}\mid x) \right]
> \end{align}
> $$

### 收敛性证明

到这里，我们已经得到了完整的 DPO 算法，但是还有一个理论问题：**将式 $\ref{eq:4}$ 中的 $\pi_{r}^{*}$ 替换为 $\pi_{\theta}$ 为什么能够帮助我们学习原本的最优策略？**

根据二分类的最大似然估计的性质，在模型表达能力充分的条件下，最小化期望交叉熵可以使得模型预测的偏好概率趋近于真实偏好概率，也就是
$$
p_{\theta}(y_{1}\succ y_{2}\mid x) \to p_{r}(y_{1}\succ y_{2}\mid x)
$$
由于 Sigmoid 函数严格单调，因此当二者相等时，有
$$
\dfrac{\pi_{\theta}(y_{1}\mid x)}{\pi_{\theta}(y_{2}\mid x)}=\dfrac{\pi_{r}^{*}(y_{1}\mid x)}{\pi_{r}^{*}(y_{2}\mid x)}
$$
