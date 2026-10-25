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
其中，$\pi_{\text{ref}}$ 为固定的参考策略，通常由 SFT 模型提供；KL 正则化用于控制优化后的策略相对于参考策略的偏移。该优化目标明确描述了 RLHF 希望得到的结果：**在控制策略偏移的条件下，使模型生成的回答获得更高的人类偏好奖励**。

> [!note]- 优化目标的解释
> 其中
> - $x$ 表示输入 prompt
> - $y\in \mathcal{Y}$ 表示一个完整的候选回答，其中 $\mathcal{Y}$ 为回答空间
> - $r(x,y)$ 为固定的奖励函数，在传统 RLHF 中对应训练完成的 $r_{\phi}$
> - $\pi_{\text{ref}}(y\mid x)$ 为固定的参考策略
> - $\pi(y\mid x)$ 为需要优化的概率分布
> - $\beta > 0$ 为 KL 正则化系数

传统的 RLHF 需要依次完成两个学习过程：
1. **奖励学习**：从人类对回答的相对偏好中，学习一个能对任意回答进行评分的奖励函数。
2. **策略优化**：利用学习到的奖励函数，通过强化学习寻找更高奖励的生成策略。

这个两阶段流程带来了较高的计算开销和训练复杂度。奖励模型需要从有限的偏好数据中学习评价标准，PPO 则需要在训练过程中反复采样、计算奖励并更新策略，通常还涉及额外的价值模型。

RLHF 的最终目标是得到符合人类偏好的语言模型策略。那么，能否直接从偏好数据中学习这个策略，省去显式奖励建模和强化学习优化过程？**DPO (Direct Preference Optimization, 直接偏好优化)** 就是从这个问题出发的。

## 从偏好数据到奖励函数

在进一步讨论 DPO 之前，需要明确 RLHF 如何从人类偏好数据中学习奖励函数。经典 RLHF 使用 PPO 等强化学习算法优化语言模型策略。在训练过程中，策略会不断生成新的回答，而强化学习算法需要根据这些回答的质量获得奖励信号。然而，人类偏好数据通常只包含同一个 prompt 下两个回答的比较结果，这类数据能够反映回答之间的相对质量，却没有直接提供具体的奖励数值。因此，RLHF 希望从有限的偏好数据中学习一个奖励函数 $r_{\phi}(x,y)$，使其能够对任意给定的 prompt 和候选回答输出标量评分，为候选的策略优化提供奖励信号。

为了从相对偏好中学习奖励函数，我们首先需要建立奖励值与人类偏好之间的概率关系。Bradley-Terry 模型是一种经典的成对比较概率模型。它假设每个候选回答有一个正的潜在强度 $s(x,y)>0$，并将某个回答被偏好的概率表示为其强度在两个候选回答总强度中的占比：
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
> - $\pi_{\text{ref}}(y\mid x)$ 为固定的参考策略
> - $\pi(y\mid x)$ 为需要优化的概率分布
> - $\beta > 0$ 为 KL 正则化系数
> 
> 将 KL 散度展开
> $$
> D_{\text{KL}}(\pi \Vert \pi_{\text{ref}}) = \sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)\log \dfrac{\pi(y'\mid x)}{\pi_{\text{ref}}(y'\mid x)}
> $$
> 代入优化目标中得到
> $$
> \mathcal{J}_{x}(\pi)=\sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)r(x, y) - \beta \sum\limits_{y'\in \mathcal{Y}}\pi(y'\mid x)\log \dfrac{\pi(y'\mid x)}{\pi_{\text{ref}}(y'\mid x)}
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
> 于是得到式 $\ref{eq:2}$。$\mathcal{J}_{x}(\pi)$ 是关于 $\pi$ 的一个严格凹函数，在全正的 $\pi$ 支持下，求得的驻点是唯一最大值。

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

在传统的 RLHF 中，我们首先通过偏好数据学习奖励函数 $r_{\phi}$，再通过强化学习求解对应的最优策略 $\pi_{r}^{*}$。现在，式 $\ref{eq:4}$ 为我们提供了另外一种思路：**既然最优策略本身就能够表示偏好概率，那么能否直接通过偏好数据学习最优策略？**

### DPO 损失

式 $\ref{eq:4}$ 中的 $\pi_{r}^{*}$ 是未知奖励函数 $r$ 对应的最优策略，我们无法直接计算它。DPO 引入了一个可训练的参数化策略模型 $\pi_{\theta}$，用它来估计未知的 $\pi_{r}^{*}$。具体而言，DPO 将式 $\ref{eq:4}$ 中的最优策略替换为当前策略模型，构造参数化的偏好概率：
$$
p_{\theta}(y_{w}\succ y_{l}\mid x) = \sigma \left( \beta \log\dfrac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)} -\beta \log\dfrac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right)\tag{5} 
$$
其中 $\pi_{\theta}$ 是需要训练的策略模型，$\pi_{\text{ref}}$ 是固定的参考策略，$p_{\theta}$ 是由当前策略预测的人类偏好概率。这样，我们就将策略模型 $\pi_{\theta}$ 转换为一个可以预测人类偏好的二分类模型，然后沿用之前奖励模型训练的方法，利用偏好数据进行极大似然估计。

对于偏好数据 $\mathcal{D}=\left\{ (x_{i},y_{w,i},y_{l,i}) \right\}_{i=1}^{N}$，每个样本都表示了人类选择 $y_{w}$ (chosen) 而拒绝 $y_{l}$ (rejected)。因此，我们希望最大化模型对这些已观测偏好的预测概率：
$$
\max_{\theta}\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}} \left[ \log p_{\theta}(y_{w}\succ y_{l}\mid x) \right]
$$
也就是说，DPO 的损失函数为
$$
\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}}\left[ 
\log\sigma \left( \beta \log\dfrac{\pi_{\theta}(y_{w}\mid x)}{\pi_{\text{ref}}(y_{w}\mid x)} - \beta \log\dfrac{\pi_{\theta}(y_{l}\mid x)}{\pi_{\text{ref}}(y_{l}\mid x)}\right)  \tag{6}
 \right] 
$$
通过最小化这一损失，模型会倾向于提高被选择回答相对于参考策略的概率优势，同时降低未被选择回答的相对优势。因此，DPO 可以直接利用偏好数据，通过常规的梯度下降优化生成策略，无需显式训练奖励模型，也无须使用 PPO 等强化学习算法。

> [!note] 为什么要最大化 log?
> 实际上就是交叉熵损失（或者 BCELoss），由于是二分类问题，且 $p_{\theta}(y_{w}\succ y_{l}\mid x)$ 的标签永远是 1，因此写成交叉熵损失就是
> $$
> \begin{align}
> &\min_{\theta} \mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}}\left[ -1 \cdot \log p_{\theta}(y_{w}\succ y_{l}\mid x) - 0 \cdot \log p_{\theta}(y_{l}\succ y_{w}\mid x)  \right] \\
> \iff &\max_{\theta}\mathbb{E}_{(x,y_{w},y_{l})\sim \mathcal{D}} \left[ \log p_{\theta}(y_{w}\succ y_{l}\mid x) \right]
> \end{align}
> $$

> [!tip]
> 准确来说，DPO 希望当前 chosen 与 rejected 的相对偏好，强于参考策略对这两个回答的相对偏好。即使 chosen 的绝对生成概率下降，只要 rejected 的概率下降得更多，DPO 损失仍然可能减小。

### 学习最优策略的证明

到这里，我们已经得到了完整的 DPO 算法，但是还有一个理论问题：**将式 $\ref{eq:4}$ 中的 $\pi_{r}^{*}$ 替换为 $\pi_{\theta}$ 为什么能够帮助我们学习原本的最优策略？**

根据二分类的最大似然估计的性质，在模型表达能力充分的条件下，最小化期望交叉熵可以使得模型预测的偏好概率趋近于真实偏好概率，也就是
$$
p_{\theta}(y_{1}\succ y_{2}\mid x) \to p_{r}(y_{1}\succ y_{2}\mid x)
$$
由于 Sigmoid 函数严格单调，因此当二者相等时，有
$$
\dfrac{\pi_{\theta}(y_{1}\mid x)}{\pi_{\theta}(y_{2}\mid x)}=\dfrac{\pi_{r}^{*}(y_{1}\mid x)}{\pi_{r}^{*}(y_{2}\mid x)}
$$
因此，相同偏好概率意味着两个策略对同一组回答具有相同的生成概率比。基于概率分布的归一化性质，只要能够恢复足够多回答对的真实偏好概率，使这些比较关系连接整个回答空间，就可以唯一确定相应的最优策略。

> [!note]- 相等的证明
> 固定一个回答 $y_{0}$，那么对于其他任意回答 $y$ 都有
> $$
> \dfrac{\pi_{\theta}(y\mid x)}{\pi_{\theta}(y_{0}\mid x)}=\dfrac{\pi_{r}^{*}(y\mid x)}{\pi_{r}^{*}(y_{0}\mid x)}
> $$
> 整理得到
> $$
> \pi_{\theta}(y\mid x)=\dfrac{\pi_{\theta}(y_{0}\mid x)}{\pi_{r}^{*}(y_{0}\mid x)}\pi_{r}^{*}(y\mid x)
> $$
> 记 $c(x)=\pi_{\theta}(y_{0}\mid x)/\pi_{r}^{*}(y_{0}\mid x)$，于是得到
> $$
> \pi_{\theta}(y\mid x)=c(x)\pi_{r}^{*}(y\mid x)
> $$
> 由于它们都是概率分布，有
> $$
> \sum\limits_{y}\pi_{\theta}(y\mid x)=\sum\limits_{y}\pi_{r}^{*}(y\mid x)=1
> $$
> 因此，两边求和可以得到
> $$
> 1=c(x)\sum\limits_{y}\pi_{r}^{*}(y\mid x)=c(x)
> $$
> 即 $c(x)=1$，于是得到
> $$
> \pi_{\theta}(y\mid x)=\pi_{r}^{*}(y\mid x)
> $$

^282459

## DPO 的实际 pipeline

前面从 RLHF 的优化目标出发，推导出了 DPO 的损失函数。在实际应用中，我们还需要确定三个对象的具体来源：待训练的策略模型 $\pi_{\theta}$、固定的参考模型 $\pi_{\text{ref}}$，以及偏好数据集 $\mathcal{D}$。

对于语言模型，一个典型的 DPO 训练流程可以分为四个阶段：
1. **初始化策略与参考模型**：使用指令数据微调预训练模型，得到能够正常回答用户问题的模型 $\pi_{\text{SFT}}$。
2. **构造偏好数据**：为同一个 prompt 准备多个候选回答，通过人工、模型评审或客观规则判断哪个更好。
3. **DPO 训练**：对已有偏好数据计算损失，通过梯度下降更新 $\pi_{\theta}$，参考模型保持不变。
4. **评估和调整模型**：

> [!tip] DPO 训练阶段通常不需要再生成回答。它使用事先收集好的偏好数据，属于离线偏好优化。

### 初始化策略模型和参考模型

DPO 通常从已经完成监督微调 (SFT) 的语言模型开始。预训练模型已经具备基础的语言生成能力，而 SFT 进一步使模型能够遵循指令、处理对话并按照期望的格式回答问题。在此基础上，DPO 利用偏好数据调整模型的生成行为。假设完成 SFT 后得到策略 $\pi_{\text{SFT}}$，通常将其同时用于初始化待训练策略和参考策略：
$$
\pi_{\theta}^{(0)}=\pi_{\text{ref}}=\pi_{\text{SFT}}
$$
两者虽然有相同的权重，但在 DPO 训练中承担不同的作用。
- **策略模型** $\pi_{\theta}$：实际上被优化的语言模型，其参数通过 DPO 损失不断更新。
- **参考模型** $\pi_{\text{ref}}$：冻结的初始策略，在整个训练中不会被改动。

> [!note]
> 这种设置与之前推导的 KL 正则目标相对应。DPO 的训练目标本身就和 RLHF 相同，即**希望模型学会符合偏好的生成行为，同时通过参考策略引入对原有行为的约束**。

> [!tip] 关于参考模型的选择问题
> 参考模型通常是 DPO 开始训练时的策略模型。如果使用已经完成指令微调的模型，可以直接将该模型作为起点，不一定需要重新 SFT。它的作用和 RLHF 中 KL 正则的参考模型相同，用于避免策略模型偏离原有的模型太远，减少 *reward hacking* 现象。虽然 DPO 中不存在显式的 reward，但是其参考模型的思想和 RLHF 同源。

### 构造偏好数据

DPO 的一条训练数据可以表示为 $(x,y_{w},y_{l})$，其中
- $x$：用户的问题或者完整的对话上下文。
- $y_{w}$：更受偏好的回答，称为优选回答 (**chosen**)。
- $y_{l}$：相对较差的回答，称为劣选回答 (**rejected**)。

通常来讲，偏好数据通过下面的三个步骤来构造。
1. **收集 prompt**：prompt 可以来自公开指令数据集、真实用户请求，也可以使用语言模型生成。在构造数据时，应尽量覆盖模型实际处理的任务^[从 [[#^282459|证明]] 可以看到，只有回答基本覆盖了整个响应空间，此时 DPO 的优化得到的结果才是我们期望的最优策略]。
2. **生成候选回答**：对于同一个 Prompt $x$，使用语言模型生成多个候选回答 $\left\{ y_{1},y_{2},\cdots,y_{K} \right\}$，候选回答可以来自同一个模型的多次采样，也可以来自不同的语言模型。
3. **标注偏好关系**：获取候选回答后，需要判断哪些回答更符合训练目标。常见的偏好标注方式包括：
	- 人类偏好（RLHF）：由人类标注人员比较回答，根据正确性、有用性、清晰度标准选择更好的回答。人类偏好数据一般质量比较好，但也是比较难大规模获取的数据。
	- AI 偏好（RLAIF）：使用能力较强的语言模型作为评审者，对候选回答进行比较或评分。
	- 可验证反馈：对于代码、数学等任务，可以使用单元测试、标准答案或者规则检查判断回答的质量。

无论采用哪种方法，最终都可以将评审的结果转换为 $(x,y_{w},y_{l})$ 三元组。DPO 只使用其中的偏序关系，不会用到评审时产生的绝对奖励分数。

> [!note]
> 构造偏好数据时，候选回答不要求由参考模型产生。例如，可以同时使用当前的 SFT 模型和能力更强的模型生成回答，这样既能收集当前模型容易犯的错误，也能获得具有高质量的候选答案。

> [!example] 偏好数据示例
>
> ```json
> {
>   "prompt": "使用 Python 对列表去重，并保留元素第一次出现的顺序。",
>   "chosen": "return list(dict.fromkeys(xs))",
>   "rejected": "return list(set(xs))"
> }
> ```

> [!note] 偏好数据的最佳实践
> 偏好数据的来源和质量会显著影响最终效果。我们在构造数据时需要关注以下问题：
> - **数据覆盖**：包含目标应用中的主要任务，避免训练数据集中缺失少数类型的 prompt。
> - **偏好可靠性**：过滤标签错误、回答完全相同或偏好难以判断的样本。
> - **候选质量**：兼顾高质量回答与当前模型实际可能生成的回答。
> - **数据划分**：按 prompt 划分训练集和验证集。
> 
> 提升偏好数据集质量带来的性能收益甚至可以超过改变偏好优化算法的收益。数据构造是 DPO 实践中非常重要的环节。

### DPO 训练

不同于 PPO 之类的在线强化学习算法，DPO 的训练比较直接，和预训练、SFT 的流程大致相同，差别只有数据和损失函数。对于一条样本 $(x,y_{w},y_{l})$，一次训练包含一下计算：
1. 计算参考模型 $\pi_{\text{ref}}$、策略模型$\pi_{\theta}$ 对两个候选回答的概率，训练时一般采用 teach forcing。这一步对每个 token 都会计算四个条件概率：
$$
\log \pi_{\theta}(y_{w}\mid x),\quad \log \pi_{\theta}(y_{l}\mid x),\quad \log \pi_{\text{ref}}(y_{w}\mid x),\quad \log \pi_{\text{ref}}(y_{l}\mid x)
$$
而完整回答的概率则为
$$
\log \pi_{\theta}(y\mid x)=\sum\limits_{t=1}^{\left| y \right| }\log \pi_{\theta}(y_{t}\mid x,y_{<t})
$$
2. 基于 $\ref{eq:6}$ 计算损失函数，随后通过反向传播更新 $\theta$。

> [!note]
> 由于 DPO 直接优化的是两个回答之间的**相对偏好差异**，因此损失函数下降不一定意味着 $\pi_{\theta}(y_{w}\mid x)$ 的生成概率持续增加，也可能优选回答与劣选回答的概率同时下降，但是 $\pi_{\theta}(y_{l}\mid x)$ 的生成概率下降得更多。因此在实际训练中还需要评估模型的生成质量。

### 评估与调整模型

完成 DPO 训练后，我们需要通过评估检查模型是否真正学会了目标偏好，以及原有能力是否得到了保留。评估可以从两个方面进行：
1. 偏好学习的训练指标：
	- `loss`：DPO 损失，反映模型对训练数据中偏好关系的拟合情况。
	- `reward/accuracies`：当前策略给优选回答的隐式奖励高于劣选回答的样本比例。
	- `rewards/margins`：也就是 $\Delta$，优选回答和劣选回答之间的平均隐式奖励差距。
2. 评估模型实际生成的回答：在独立的验证 prompt 上，分别让 SFT 模型和 DPO 模型生成回答，并根据任务目标进行比较，例如
	- 偏好胜率：有人工或评审模型比较两个模型的回答。
	- 任务正确率：对于数学、代码等任务，使用标准答案或测试用例评价。
	- 行为变化：检查回答长度、指令遵循、拒答行为及其他重要能力是否发生预期之外的变化。
