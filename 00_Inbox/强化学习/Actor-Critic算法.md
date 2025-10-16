---
tags:
  - 强化学习
---
# Actor-Critic 算法

在强化学习中，存在基于值函数的方法 ([[00_Inbox/强化学习/DQN算法]])与基于策略梯度的方法（[[00_Inbox/强化学习/REINFROCE算法]]），基于值函数的方法只学习一个价值函数，而基于策略的方法只学习一个策略函数，那么很自然的，应当可以提出一个方法既学习价值函数，又学习策略函数。这个方法就是 Actor-Critic。Actor-Critic 是囊括了一系列算法的整体架构，目前很多高效的前沿算法都属于 Actor-Critic 算法。本节介绍一种最简单的 Actor-Critic 算法。

## Actor-Cirtic

在策略梯度算法中，我们使用策略函数来计算表达当前策略，并使用 [[00_Inbox/强化学习/REINFROCE算法#策略梯度|策略梯度]] 来对策略进行更新。在 [[00_Inbox/强化学习/REINFROCE算法]] 算法中，我们使用蒙特卡洛方法来估计 $Q(s,a)$，很自然的，我们考虑能否拟合一个值函数来指导策略进行学习。我们可以将策略梯度表达为更加一般的形式：
$$
g=E[\sum_{t=0}^T\psi_t\nabla_\theta\log\pi_\theta(a_t|s_t)]
$$
其中，$\psi_t$ 可以是多种形式：

| $\psi_t$                                                 | 意义                  |
| :--------------------------------------------------------: | --------------------- |
| $\sum_{t'=0}^T\gamma^{t'}r_{t'}$                         | 轨迹的总回报          |
| $\sum_{t=t'}^T\gamma^{t'-t}r_{t'}$                       | 动作 $a_t$ 之后的回报 |
| $\sum_{t'=t}^T\gamma^{t'-t}r_{t'}-b(s_t)$                | 基准线版本的改进      |
| $Q^{\pi_\theta}(s_t,a_t)$                                | 动作价值函数          |
| $A^{\pi_\theta}(s_t,a_t)$                                | 优势函数              |
| $r_t+\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}(s_t)$ | 时序差分残差          |
|                                                         |                       |

下面对上式进行一些解释。在 REINFORCE 算法中，我们使用 (2) 估计 $Q$ 的值，虽然估计是无偏的，但是得到的方差非常大，我们可以使用形式 (3) 引入基线函数 (baseline function) $b(s_t)$ 来减小方法。此外，我们还可以使用 Actor-Critic 算法估计一个动作价值 $Q$，代替蒙特卡洛方法采样得到的回报，即 (4)，在这个基础上，使用状态价值作为基线函数，于是得到 (5)，更进一步，可以使用 $Q=r+\gamma V$ 得到形式 (6)。

接下来对 (6) 展开，即通过时序差分残差来指导策略梯度进行学习。实际上，直接使用 $Q$ 值或者 $V$ 值本质上也是用奖励来进行指导，但是使用神经网络进行估计的方法可以减小方差，提高鲁棒性。除此之外，REINFORCE 方法是基于蒙特卡洛采样，因此需要完整采样完成一个序列之后才可以更新参数，相比之下使用时序差分残差可以在每一步之后都进行更新，且可以运用于步数无限或者非常大的任务。

我们将 Actor-Critic 算法分为两个部分：Actor (策略网络) 和 Critic (价值网络)。其中策略网络主要和环境进行交互，并在价值网络的指导下学习一个更好的策略，而价值网络是通过策略网络与函数交互收集到的数据来学习一个价值函数，通过这个价值函数来判断动作的价值，进而指导策略网络的学习。

- 策略网络学习一个策略 $\pi_\theta$ 的参数，输入状态，输出动作的概率分布
- 价值网络学习一个状态价值函数 $V_\omega$ 的参数，输入状态，输出状态价值

在策略网络中，我们使用策略梯度来对网络进行更新，使用将 $r+\gamma V_\omega(s_{t+1})$ 作为时序差分目标来对价值网络进行更新，即：
$$
\left\{\begin{align}
L(\omega)&=\frac{1}{2}(r+\gamma V_\omega(s_{t+1})-V_\omega(s_t))^2
\\[3mm]J(\theta)&=E_{s_0}[V^{\pi_\theta}(s_0)]
\end{align}\right.
$$
即：
$$
\left\{
\begin{align}
\nabla_\omega L(\omega)&=-(r+\gamma V_\omega(s_{t+1})-V_\omega(s_t))\nabla_\omega V_\omega(s_t)
\\[3mm]\nabla_\theta J(\theta)&=-(r+\gamma V_{\omega}(s_{t+1})-V_\omega(s_{t}))\nabla_\theta\log\pi_\theta(a_t|s_t)
\end{align}
\right.
$$

于是 Actor-Critic 的伪代码为：

```python
ActorNet.init_param()
CriticNet.init_param()
optim_actor.init(ActorNet.parameters(), learning_rate_actor)
optim_critic.init(CriticNet.parameters(), learning_rate_critic)
for episode in range(num_episodes):
	experience_dict = []
	state = env.reset()
	done = False
	while not done:
		agent.choice_action(state)
		next_state, reward, done = env.step(action)
		
		experience_dict['state'].append(state)
		experience_dict['action'].append(action)
		experience_dict['reward'].append(reward)
		experience_dict['next_state'].append(next_state)
		experienct_dict['done'].append(done)
		
		state = next_state

	td_target = reward + gamma * CriticNet(next_state) * (1 - done)
	td_error = td_target - CriticNet(state)
	log_prob = torch.log(ActorNet(next_state).gather(1, action))
	loss_critic = mse_loss(CriticNet(state), td_target)
	loss_actor = -td_error * log_prob

	optim_actor.zero_gard()
	optim_critic.zero_gard()
	loss_critic.backward()
	loss_actor.backward()
	optim_actor.step()
	optim_critic.step()
```

## 算法评价

Actor-Critic 算法是基于值函数的方法与基于策略的方法的叠加。价值模块 Critic 在策略模块 Actor 采样的数据中学习分辨什么是好的动作，什么是不好的动作，进而指导 Actor 模块进行策略更新，其与环境交互所产生的数据分布也发生改变，这需要 Critic 尽快适应新的数据分布并给出好的判别。