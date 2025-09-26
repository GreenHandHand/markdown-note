---
tags:
  - 强化学习
---
# REINFORCE 算法

在强化学习中，除了 [[00-笔记/强化学习/时序差分算法#Q-learning 算法|Q-learning]]、[[00-笔记/强化学习/DQN算法|DQN]] 等学习价值函数的方法，还有一种直接学习策略的方法。Q-learning 系列方法被称为**基于价值** (value-based) 的方法，而这种直接学习策略的方法称为**基于策略** (policy-based) 的方法。策略梯度是基于策略的方法的基础。

## 策略梯度

基于策略的方法需要首先将策略参数化。假设 $\pi_\theta$ 是一个随机性的策略，并且处处可微，其中 $\theta$ 是对应的参数，我们可以使用一个[[00-笔记/机器学习/线性模型|线性模型]]或者神经网络模型来描述这样一个策略函数：_输入一个状态，然后输出动作的概率分布_。我们的目标是找到一个最优的策略，使得这个策略在环境中的期望回报最大，因此我们可以将目标函数定义为：
$$
J(\theta)=E_{s_0}[V^{\pi_{\theta}}(s_0)]
$$
其中，$s_0$ 表示初始状态，上面的式子表达了在策略 $\pi_\theta$ 下从状态 $s_0$ 出发所获得的回报的期望。为了最大化期望回报，我们对上面的式子进行求导
$$
\nabla_\theta J(\theta)=\frac{\partial J}{\partial \theta}=E_{s_0}[\nabla_\theta V^{\pi_\theta}(s_0)]
$$
于是我们先计算状态价值函数的梯度:
$$
\begin{align}
\nabla_\theta V^{\pi_\theta}(s_0)&=\nabla_\theta\sum_{a\in \mathcal{A}}Q^{\pi_\theta}(s,a)\pi_\theta(a|s)
\\&=\sum_{a\in\mathcal A}\nabla_\theta Q^{\pi_\theta}(s,a)\pi_\theta(a|s)+Q^{\pi_\theta}(s,a)\nabla_\theta\pi_\theta(a|s)
\end{align}
$$
设 $\phi(s)=\sum_{a\in\mathcal A}\nabla_\theta\pi_\theta(s|a) Q^{\pi_\theta}(s,a)$，将动作价值函数展开有：
$$
\begin{align}
\nabla_\theta V^{\pi_\theta}(s_0)&=\phi(s)+\sum_{a\in\mathcal A}\pi_\theta(a|s)\nabla_\theta\left(r(s,a)+\gamma\sum_{s'\in S}P(s'|s,a)V^{\pi_\theta}(s')\right)
\\&=\phi(s)+\sum_{a\in\mathcal A}\pi_\theta(a|s)\gamma\sum_{s'\in\mathcal S}P(s'|s,a)\nabla_{\theta}V^{\pi_\theta}(s')
\\&=\phi(s)+\gamma\sum_{a\in\mathcal A}\sum_{s'\in S}\pi_\theta(a|s)P(s'|s,a)\nabla_\theta V^{\pi_\theta}(s')
\\&=\phi(s)+\gamma\sum_{s'\in S}\nabla_\theta V^{\pi_\theta}(s')\sum_{a\in\mathcal A}\pi_\theta(a|s)P(s'|s,a)
\end{align}
$$
由于 $\sum_{a\in\mathcal A}\pi_\theta(a|s)P(x|s,a)$ 表示从状态 $s$ 在策略 $\pi_\theta$ 下下一步达到状态 $x$ 的概率，我们将其表示为 $d^{\pi_\theta}(s\to x,k)$ ，其中 $k$ 表示到达 $x$ 的步数，于是上式有：
$$
\begin{align}
\nabla_\theta V^{\pi_\theta}(s_0)&=\phi(s)+\gamma\sum_{s'\in S}\nabla_{\theta}V^{\pi_\theta}(s')d^{\pi_\theta}(s\to s',1)
\end{align}
$$
上式是一个迭代式，我们将其展开：
$$
\begin{align}
\nabla_\theta V^{\pi_\theta}(s_0)&=\phi(s)+\gamma\sum_{s'\in S}d^{\pi_\theta}(s\to s',1)\cdot\\
&\quad\left(\phi(s')+\gamma\sum_{s''\in\mathcal S}d^{\pi_\theta}(s'\to s'',1)\nabla_{\theta}V^{\pi_\theta}(s'')\right)
\\&\begin{alignedat}{2}
=\phi(s)&+\gamma\sum_{s'\in S}d^{\pi_\theta}(s\to s',1)\phi(s')
\\&+\gamma^2\sum_{s''\in\mathcal S}\sum_{s'\in\mathcal S}d^{\pi_\theta}(s\to s',1)d^{\pi_\theta}(s'\to s'',1)\nabla_{\theta}V^{\pi_\theta}(s'')
\end{alignedat}
\\&\begin{alignedat}{2}
=\phi(s)&+\gamma\sum_{s'\in S,1}d^{\pi_\theta}(s\to s',1)\phi(s')
\\&+\gamma^2\sum_{s''\in\mathcal S}d^{\pi_\theta}(s\to s'',2)\nabla_{\theta}V^{\pi_\theta}(s'')
\end{alignedat}
\\&=\sum_{x\in\mathcal S}\sum_{k=0}^\infty\gamma^kd^{\pi_\theta}(s\to x,k)\phi(x)
\end{align}
$$
我们令 $\eta(x)=E_{s_0}[\sum_{k=0}^\infty\gamma^kd^{\pi_\theta}(s_0\to x,k)]$，于是有
$$
\begin{align}
\nabla_\theta J(\theta)&=E_{s_0}[\nabla_\theta V^{\pi_\theta}(s_0)]
\\&=E_{s_0}[\sum_{x\in\mathcal S}\sum_{k=0}^\infty\gamma^kd^{\pi_\theta}(s_0\to x,k)\phi(x)]
\\&=\sum_{x\in\mathcal S}\phi(x)E_{s_0}[\sum_{k=0}^\infty\gamma^kd^{\pi_\theta}(s_0\to x,k)]
\\&=\sum_{x\in\mathcal S}\phi(x)\eta(x)
\\&=\sum_x\frac{\sum_{x}\eta(x)}{\sum_x\eta(x)}\eta(x)\phi(s)
\\&=\left(\sum_x\eta(x)\right)\sum_x\frac{\eta(x)}{\sum_x\eta(x)}\phi(x)
\\&\propto \sum_x\frac{\eta(x)}{\sum_x\eta(x)}\phi(x),\quad (x\to s)
\\&=\sum_{s}E_{s_0}[\sum_{k=0}^\infty\gamma^kd^{\pi_\theta}(s_0\to s,k)]\sum_{a\in\mathcal A}\nabla_\theta\pi_\theta(s|a) Q^{\pi_\theta}(s,a)
\\&=\sum_s v^{\pi_\theta}(s)\sum_{a\in\mathcal A}\nabla_\theta\pi_\theta(s|a) Q^{\pi_\theta}(s,a)
\\&=E_{\pi_\theta}[\sum_{a\in\mathcal A}\nabla_\theta\pi_\theta(a|s)Q^{\pi_\theta}(s,a)]
\end{align}
$$
由于 $v^{\pi_\theta}(s)$ 为策略 $\pi_\theta$ 下的状态访问分布，即到达所有状态的概率，可以视为加权求和。上式便是策略梯度的损失函数，使用上式进行更新需要使用所有的动作状态函数的加权求和运算，过于繁琐，因此将上式进行修改，我们就可以得到：
$$
\begin{align}
\nabla_\theta J(\theta)&\propto E_{\pi_\theta}\left[\sum_{a\in\mathcal A}\nabla_\theta\pi_\theta(a|s) Q^{\pi_\theta}(s,a)\right]
\\&=E_{\pi_\theta}\left[\sum_{a\in\mathcal A}\pi_\theta(a|s)Q^{\pi_\theta}(s,a)\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}\right]
\\&=E_{\pi_\theta}\Big[E\left[Q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)\big|S=s\right]\Big]
\\&=E_{\pi_\theta}[Q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)]
\end{align}
$$
使用该式常作为策略梯度算法的损失函数。需要注意的是，由于上式中的期望的变量是 $\pi_\theta$，因此策略梯度算法为在线算法，必须使用当前的策略进行采样。直观的理解策略梯度的更新公式，我们发现在每个状态下，梯度的修改都是让策略更多的采样到较高 $Q$ 值的动作，更少的采样到较低 $Q$ 值的动作。

>简单说明一下 $\nabla_\theta\log\pi_\theta(a|s)$ 的含义，我在推导过程中对这个式子的含义一直不太明确。该式表示策略的输出的对数关于策略函数的参数 $\theta$ 的梯度。在策略梯度中，策略函数应当是输入一个状态，输出该状态对应的动作概率分布，例如，我们使用神经网络作为策略函数，那么 $\theta$ 指代的就是神经网络的参数。实际上我们在计算时使用的是下面的式子： 
$$
\nabla_\theta J(\theta)=\nabla_\theta E[Q^{\pi_\theta}(s,a)\log\pi_\theta(a|s)]
$$
>由于对于参数 $\theta$ 而言 $Q$ 值为一个常数，因此在求梯度时可以直接将其放到到外面。因此在实际使用中，我们只需要计算 $J(\theta)$，然后使用优化器进行梯度下降即可。

## REINFORCE 算法

在策略梯度的更新公式中，我们需要用到 $Q^{\pi_\theta}(s,a)$，REINFORCE 算法就是使用蒙特卡洛方法对其进行估计，因此，对于有限步数的环境来说，REINFORCE 算法的策略梯度估计为：
$$
\nabla_\theta J(\theta)=E_{\pi_\theta}\left[\sum_{i=1}^T\left(\sum_{t'=t}^T\gamma^{t'-t}r_{t'}\right)\nabla_\theta\log\pi_\theta(a_t|s_t)\right]
$$

于是算法的具体流程为：
```python
# 初始化策略参数
pi.init_theta()
optim.init(pi.parameters(), learning_rate)
for episode in range(num_episodes):
	# 使用当前策略采样获得序列
	state_list, action_list, reward_list = [], [], []
	state = env.reset()
	done = False
	while not done:
		action = agent.choice_action(state)
		next_state, reward, done = env.step(action)
		state_list.append(state)
		action_list.append(action)
		reward_list.append(reward)

		state = next_state

	# 计算当前序列每个时刻 t 往后的回报
	G = 0
	optim.zero_gard()
	for i in range(reversed(0, len(state_list))):
		G = G * gamma + reward[i]

		# 对 theta 进行更新
		log_prob = log(pi[state[i]][action[i]])
		loss = - G * log_prob
		loss.backward()
	optim.step()
```

本算法相比于 DQN 算法的区别是：
1. 在 choice_action 中，我们对获得的动作概率分布进行采样来获得当前状态下进行的动作。
2. 在更新过程中，我们将损失函数设置为期望回报 $J(\theta)$ 的相反数 $-J(\theta)$，然后使用梯度下降法来对策略函数进行跟更新。
3. 在定义策略网络 PolicyNet 时，其输入为某个状态，输出则是该状态下的动作概率分布，所以在输出层上我们使用 softmax 函数来将其输出转化为每个动作的概率。

## 算法评估

REINFORCE 算法是策略学习乃至强化学习的代表算法，该算法在理论上可以保证局部最优解，它使用蒙特卡洛方法估计动作价值，虽然可以得到无偏的梯度，但是梯度估计的方差却非常大，造成了一定程度上的不稳定性。这也是 [[00-笔记/强化学习/Actor-Critic算法]] 要解决的问题。