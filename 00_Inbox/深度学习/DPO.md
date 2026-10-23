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

但是传统的 RLHF 需要依次完成两个学习过程：
1. **奖励学习**：从人类对回答的相对偏好中，学习一个能对任意回答进行评分的奖励函数。
2. **策略优化**：利用学习到的奖励函数，通过强化学习寻找更高奖励的生成策略。

这个两阶段方式增加了训练的复杂度。奖励模型只能通过有限的偏好数据学习人类的评价标准。PPO 等强化学习算法也需要在训练过程中反复采样，计算奖励并更新策略。这导致训练过程比较复杂，至少需要奖励模型和策略模型之间进行复杂的交互，甚至 PPO 算法还需要额外引入价值模型，带来了较高的计算开销和优化复杂度。

RLHF 的最终目标是需要一个符合人类偏好的最优策略，而 DPO 的出现则使得我们可以直接从偏好数据中学习这个策略，省去了奖励建模和强化学习优化这两个中间步骤。

## DPO 算法

DPO (Direct Preference Optimization, 直接偏好优化) 重新审视了 RLHF 的优化目标。传统 RLHF 给定奖励 $r$，然后通过强化学习求解最优策略 $\pi^{*}$。对于式 $\ref{eq:1}$ 而言，最优策略实际上具有一个解析形式的解：
$$
\pi_{r}^{*}(y\mid x) = \dfrac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp \left( \dfrac{r(x,y)}{\beta} \right) \tag{2}
$$
其中 $Z(x)$ 是归一化常数。这意味着，**给定参考策略和奖励函数，理论上的最优策略已经由式 $\ref{eq:2}$ 确定了**。于是 DPO 反过来思考：如果奖励函数与最优策略之间存在明确的数学对应关系，那么能否将奖励函数直接表示为策略的函数？

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
> D_{\text{KL}}(\pi \Vert \pi_{\text{ref}}) = \sum\limits_{y}\pi(y\mid x)\log \dfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
> $$
> 代入优化目标中得到
> $$
> \mathcal{J}_{x}(\pi)=\sum\limits_{y}\pi(y\mid x)r(x, y) - \beta \sum\limits_{y}\pi(y\mid x)\log \dfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
> $$
> 这是一个关于概率分布 $\pi(\cdot\mid x)$ 的优化问题。我们可以使用拉格朗日乘数法求解，该最优化式只有一个等式条件 $\sum\limits_{y}\pi(y\mid x)=1$，于是可以得到：
> $$
> \mathcal{F}(\pi,\lambda)=\sum\limits_{y}\pi(y\mid x)r(x,y)-\beta \sum\limits_{y}\pi(y\mid x)\log \dfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\lambda(x) \left( \sum\limits_{y}\pi(y\mid x) - 1 \right) 
> $$
> 对 $y$ 求导并令导数为零得到
> $$
> r(x,y)-\beta \left( \log\dfrac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + 1 \right) + \lambda(x)=0
> $$
> 整理得到
> $$
> \pi(y\mid x) = \pi_{\text{ref}}(y\mid x)\exp \left( \dfrac{r(x,y)}{\beta} \right) \exp \left( \dfrac{\lambda(x)}{\beta}-1 \right) 
> $$
> 代入 $\sum\limits_{y}\pi(y\mid x)=1$ 得到
> $$
> \pi_{r}^{*}(y\mid x)=\dfrac{\pi_{\text{ref}(y\mid x)}\exp(\dfrac{r(x,y)}{\beta})}{\sum\limits_{y'\in \mathcal{Y}}\pi_{\text{ref}(y'\mid x)}\exp \left( \dfrac{r(x,y')}{\beta} \right) }
> $$
> 记
> $$
> Z(x)=\sum\limits_{y'\in \mathcal{Y}}\pi_{\text{ref}(y'\mid x)}\exp \left( \dfrac{r(x,y')}{\beta} \right) 
> $$
> 于是得到式 $\ref{eq:2}$。

对式 $\ref{eq:2}$ 进行变换，可以得到
$$
r(x,y)=\beta \log\dfrac{\pi_{r}^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)
$$
于是，奖励函数 $r(x,y)$ 就可以被最优策略与参考测量表示。
