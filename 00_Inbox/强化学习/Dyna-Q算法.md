---
tags:
  - 强化学习
---
# Dyna-Q 算法

在强化学习中，模型通常指与智能体交互的环境模型，即对环境的状态转移概率和奖励函数进行建模。根据是否有模型，强化学习算法分为：**基于模型的强化学习算法** (model-based reinforcement learning)和**无模型的强化学习** (model-free reinforcement learning)。[[00_Inbox/强化学习/时序差分算法#Sarsa 算法|Sarsa算法]] 与 [[00_Inbox/强化学习/时序差分算法#Q-learning 算法|Q-learning]] 算法就是无模型的强化学习算法。Dyna-Q 是基于模型的强化学习算法。

## Dyna-Q

Dyna-Q 使用一种叫做 Q-planning 的方法基于模型来生成一些模拟数据，然后用模拟数据和真实数据一起改进策略。Q-planning 每次选取一个曾经访问过的状态 $s$，采取一个在当前状态下执行过的动作 $a$，通过模型获得转移后的状态 $s'$ 和奖励 $r$，并根据这个模拟数据 $(s,a,r,s')$，使用 Q-learning 的方法更新动作价值函数。

算法的具体流程如下：
```python
# initialize the Q[s][a] and M[s][a]
for x in X:
	s = 0
	done = False
	while not done:
		a = choice_action(s)
		r, s_next = step(s, a)
		Q[s][a] += alpha * (r + gamma * max([Q[s_next][a_next] for a_next in A]) - Q[s][a])
		M[s][a] = r, s_next
		for i in range(q_planning_count):
			# 获得一个访问过的sm，采取一个在sm下进行过的动作am
			rm, sm_next = M[s][a]
			Q[sm][am] += alpha * (r + gamma * max([Q[sm_next][a_next] for a_next in A]) - Q[sm][am])
		s = s_next
```
上式中的 q_planning_count 是一个可以设置的超参数，当 q_planning_count = 0 时，Dyna-Q 就相当于 Q-learning 算法。Dyna-Q 算法是执行在离散并且确定的环境中的。

在状态转移确定性高的环境中，构建的模型的精度高，可以通过增加 Q-planning 的次数来降低算法的样本复杂度。但不是在所有的环境中 Q-planning 越大算法收敛越快。

## 基于模型的强化学习算法优劣

强化学习算法有两个重要的指标，一个是算法收敛后的策略在初始状态下的期望回报，另一个是样本复杂度，即算法达到收敛结果需要在真实环境中采样的样本数量。

基于模型的强化学习算法由于具有一个环境模型，智能体可以通过和环境模型进行交互，因此对样本的需求量往往会减少，因此基于模型的强化学习算法具有更低的样本复杂度。

但是环境模型可能并不准确，不能代替真实环境，因此基于模型的强化学习算法收敛后策略的期望回报可能不如无模型的强化学习算法。