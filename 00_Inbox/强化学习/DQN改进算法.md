---
tags:
  - 强化学习
---
# DQN 改进算法

DQN 算法敲开了深度强化学习的大门，但是作为先驱性的工作，本身存在着一些问题以及一些可以改进的地方。本节介绍两个著名的算法：Double DQN 和 Dueling DQN。

## Double DQN

普通的 DQN 算法通过会导致对 $Q$ 值的过高估计 (overstimation)。由于 DQN 的学习目标为：
$$
r+\gamma\max_{a'}Q_{\omega^-}(s',a')
$$
其中 $\max_{a'}Q_{\omega^-}(s',a')$ 由神经网络得出，我们还可以将其写为下面的形式：
$$
\max_{a'}Q(\omega^-)(s',a')=Q_{\omega^-}(s', \arg\max_{a'}Q_{\omega^-}(s',a'))
$$
考虑到在每次估计时，神经网络的估计都会存在偏高或者偏低的估计，而使用 $\max$ 函数导致我们可能选取了由于偏差导致的最高动作价值，使得每一步中 $Q(s,a)$ 的估计实际上偏高了。对于动作空间较小的环境来说这确定并不明显，但是对于动作空间较高的环境，DQN 中的过高估计问题会非常的严重，导致无法有效的工作。

Double DQN 算法解决这一问题的方法是使用两套独立训练的神经网络，一套用于估计目标函数，一套用于选取价值最大的动作，即将下面的式子作为优化目标：
$$
r+\gamma Q_{\omega^-}(s', \arg\max_{a'}Q_{\omega}(s',a'))
$$
其中 $w$ 与 $w^-$ 是独立训练的两个神经网络的参数。Double DQN 算法在 DQN 算法的基础上只进行了很小的改进，对 DQN 算法的代码进行简单修改就可以实现，但是可以在极大的缓解 DQN 算法中 $Q$ 值的过高估计的问题。

于是得到 Doubld DQN 算法的描述如下：
```python
# 初始化训练神经网络
net = Network()
# 初始化目标神经网络
target_net = Network()
# 经验缓冲池
replay_buffer = ReplayBuffer()
for x in X:
	s = start_state
	done = False
	while not done:
		a = choice_action(s)
		s_next, reward, done = env.step(s, a)
		replay_buffer.add(s, a, reward, s_next, done)
		s = s_next
		if replay_buffer.size > minizial_size:
			s_array, a_array, r_array, s_next_array, done_array = replay_buffer.sample(batch_size)

			q_value = net(s).gather(1, a_array)
			# 这里与DQN不同
			max_action = argmax(net(s_next), 1)
			max_q = target_net(s_next).gather(1, max_action)
			q_target = reward + gamma * max_q * (1 - done)
			loss = mse_loss(q_value, q_target)
			
			optim.zero_gard()
			loss.backward()
			optim.step()
			count++
			
			if count % N == 0:
				target_net.w = net.w
```

## Dueling DQN

Dueling DQN 是 DQN 算法的另一中改进算法，它在 DQN 算法上进行了微小的改动，但是能够大幅提升 DQN 算法的效果。

Dueling 算法中，首先定义了优势函数 $A$
$$
A(s,a)=Q(s,a)-V(s)
$$
通过状态价值函数与动作价值函数的定义可以知道，所有动作的优势函数和为 0。在 Dueling DQN 算法中，Q 网络被修改为
$$
Q_{\eta,\alpha,\beta}(s,a)=V_{\eta,\alpha}(s)+A_{\eta,\beta}(s,a)
$$
其中 $\eta$ 为动作价值函数网络与优势函数网络共同的参数，即神经网络的前几层，一般用于提取特征，而 $\alpha,\beta$ 为动作价值函数与优势函数分别的参数，即神经网络的后几层。在这样的模型下，神经网络不再直接输出 $Q$ 值，而是输出该状态下的状态价值函数与执行动作 $a$ 的优势函数，最后再求和得到动作价值函数。

将状态价值函数与优势函数分布建模的好处在于在某些环境下，智能体只关注状态价值而不关系不同动作导致的差异，此时将二者分开建模能够使智能体更好的处理动作关联较小的状态。

在 Dueling DQN 算法中，Q 值计算存在不唯一性，即如果 $V$ 减去一个常数 $C$，$A$ 加上这个常数 $C$，所得到的结果不会改变，这导致了模型训练的不稳定性。为了解决这个问题，Dueling DQN 算法强制最优动作的优势函数输出为 0，即
$$
Q_{\eta,\alpha,\beta}(s,a)=V_{\eta,\alpha}(s)+A_{\eta,\beta}(s,a)-\max_{a'}A_{\eta,\beta}(s,a')
$$
即最终收敛时有 $A=0$，此时 $V(s)=\max_aQ(s,a)$ 这样做可以保证 $V$ 的唯一性，

在实际使用中，我们会将 $\max$ 替换为均值，虽然这样不满足贝尔曼最优方程，但是实际效果根据稳定。综上，神经网络定义为了：
$$
Q_{\eta,\alpha,\beta}(s,a)=V_{\eta,\alpha}(s)+A_{\eta,\beta}(s,a)-\frac{1}{|A|}\sum_{a'} A_{\eta,\beta}(s,a')
$$

Dueling DQN 算法与 DQN 算法的差异只在结构上，因此这里不给出伪代码。